"""A long-running ``whisper-server`` used for gap filling.

Gap filling transcribes short speech pieces after the main pass of every segment. Starting whisper-cli for that
costs a full model load each time (about two seconds for a large model on a GPU); the server loads the model once
per run and receives the pieces over HTTP, straight from memory.

The server never uses VAD. whisper-server keeps the VAD time mapping of a request in its state, so after a VAD
request the timestamps of later non-VAD requests come back shifted. The main pass, which needs VAD (and its speech
segments on stderr), therefore stays on whisper-cli.
"""

import logging
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from pathlib import Path

from .config import Settings
from .errors import Cancelled, ToolError
from .media import resolve_executable
from .proc import ProcessGroup

log = logging.getLogger(__name__)

STARTUP_TIMEOUT_S = 300  # loading a large model from a slow disk can take a while
REQUEST_TIMEOUT_S = 600
# whisper-cli's decoding defaults; whisper-server's own defaults are greedy, which is faster but less accurate.
_DECODING = ["-bs", "5", "-bo", "5"]


def resolve_whisper_server(whisper_cli: str, whisper_server: str | None) -> str:
    """``V2SRT_WHISPER_SERVER`` if set, otherwise whisper-server next to whisper-cli, otherwise PATH."""
    if whisper_server:
        return resolve_executable(whisper_server, "whisper-server", "V2SRT_WHISPER_SERVER")
    cli = Path(whisper_cli)
    sibling = cli.with_name(cli.name.replace("whisper-cli", "whisper-server"))
    if sibling != cli and sibling.is_file():
        return str(sibling)
    return resolve_executable("whisper-server", "whisper-server", "V2SRT_WHISPER_SERVER")


class WhisperServer:
    """Starts whisper-server on first use and keeps it running until :meth:`close`.

    Requests may come from several threads; the server processes them one at a time.
    """

    def __init__(
        self, settings: Settings, executable: str, procs: ProcessGroup, command_prefix: list[str] | None = None
    ):
        self.settings = settings
        self.procs = procs
        self.prefix = command_prefix or [executable]
        self._lock = threading.Lock()
        self._proc = None
        self._url = ""
        self._output: deque[str] = deque(maxlen=30)
        # Never send requests to 127.0.0.1 through HTTP_PROXY and friends.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def command(self, port: int) -> list[str]:
        s = self.settings
        cmd = [*self.prefix, "-m", s.whisper_model, "-l", s.language, "--host", "127.0.0.1", "--port", str(port)]
        cmd += _DECODING
        if s.whisper_threads:
            cmd += ["-t", str(s.whisper_threads)]
        if s.whisper_prompt:
            cmd += ["--prompt", s.whisper_prompt]
        return cmd + s.whisper_args

    def transcribe(self, audio: bytes, *, language: str) -> str:
        """Transcribe a WAV file (16 kHz mono PCM) and return the result as SRT text."""
        url = self._ensure_running()
        body, content_type = _multipart({"language": language, "response_format": "srt"}, "file", "piece.wav", audio)
        request = urllib.request.Request(url + "/inference", data=body, headers={"Content-Type": content_type})
        try:
            with self._opener.open(request, timeout=REQUEST_TIMEOUT_S) as response:
                text = response.read().decode("utf-8", errors="replace")
                kind = response.headers.get_content_type()
        except urllib.error.HTTPError as e:
            answer = e.read().decode("utf-8", errors="replace").strip()
            raise ToolError(f"whisper-server failed with HTTP {e.code}: {answer}{self._details()}") from None
        except OSError as e:
            if self.procs.cancelled.is_set():
                raise Cancelled("cancelled") from None
            raise ToolError(f"whisper-server request failed: {e}{self._details()}") from None
        if kind == "application/json":  # errors come back as {"error": "..."}
            raise ToolError(f"whisper-server failed: {text.strip()}{self._details()}")
        return text

    def close(self) -> None:
        with self._lock:
            if self._proc is not None:
                self.procs.stop(self._proc)
                self._proc = None

    # ---------------------------------------------------------------- internals

    def _ensure_running(self) -> str:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return self._url
            if self._proc is not None:
                log.warning("whisper-server exited unexpectedly (code %s), restarting it", self._proc.returncode)
                self.procs.stop(self._proc)
            self._start()
            return self._url

    def _start(self) -> None:
        port = _free_port()
        self._output.clear()

        def on_line(line: str) -> None:
            if line.strip():
                self._output.append(line)
                log.debug("whisper-server: %s", line)

        self._proc = self.procs.spawn(self.command(port), on_line=on_line, name="whisper-server")
        self._url = f"http://127.0.0.1:{port}"
        started = time.monotonic()
        while True:
            if self.procs.cancelled.is_set():
                raise Cancelled("cancelled")
            code = self._proc.poll()
            if code is not None:
                self._proc = None
                raise ToolError(f"whisper-server exited with code {code} while starting{self._details()}")
            if self._responds():
                log.debug("whisper-server ready at %s after %.1fs", self._url, time.monotonic() - started)
                return
            if time.monotonic() - started > STARTUP_TIMEOUT_S:
                self.procs.stop(self._proc)
                self._proc = None
                raise ToolError(f"whisper-server did not start within {STARTUP_TIMEOUT_S}s{self._details()}")
            time.sleep(0.2)

    def _responds(self) -> bool:
        try:
            with self._opener.open(self._url + "/", timeout=2):
                return True
        except urllib.error.HTTPError:
            return True  # any HTTP answer means the model is loaded and the server is listening
        except OSError:
            return False

    def _details(self) -> str:
        lines = [line for line in self._output if line.strip()]
        return (":\n" + "\n".join(lines[-10:])) if lines else ""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _multipart(fields: dict[str, str], file_field: str, filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode()
    )
    parts.append(content)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
