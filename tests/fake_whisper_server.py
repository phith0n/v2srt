"""Stand-in for whisper-server: answers POST /inference with one SRT cue per 2 seconds of the uploaded WAV.

Environment knobs:
- FAKE_SERVER_FAIL: exit with an error instead of starting
- FAKE_WHISPER_BACKCHANNEL: also return a short backchannel cue per request
- FAKE_SERVER_LOG: append one line per request to this file
"""

import io
import itertools
import os
import sys
import wave
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REQUESTS = itertools.count(1)


def stamp(ms: int) -> str:
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._send(200, "text/html", "<html></html>")

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        head = f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode()
        message = BytesParser(policy=policy.default).parsebytes(head + body)
        form = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
                for part in message.iter_parts()}  # fmt: skip
        if "file" not in form:
            self._send(400, "application/json", '{"error":"no \'file\' field in the request"}')
            return
        with wave.open(io.BytesIO(form["file"])) as w:
            duration = w.getnframes() * 1000 // w.getframerate()
        if log := os.environ.get("FAKE_SERVER_LOG"):
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"{form['language'].decode()} {form['response_format'].decode()} {duration}\n")
        number = next(REQUESTS)  # makes every text unique, so the cleaner keeps them
        cues = [(start, min(start + 2000, duration), f"Recovered {number}.{n}")
                for n, start in enumerate(range(0, max(1, duration - 200), 2000), 1)]  # fmt: skip
        if os.environ.get("FAKE_WHISPER_BACKCHANNEL"):
            cues.append((0, min(500, duration), "うん、うん"))
        srt = "".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{text}\n\n" for i, (a, b, text) in enumerate(cues, 1))
        self._send(200, "application/x-subrip", srt)

    def _send(self, status: int, kind: str, text: str) -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        print("request:", format % args, file=sys.stderr, flush=True)


def main() -> int:
    args = sys.argv[1:]
    assert "-m" in args and os.path.isfile(args[args.index("-m") + 1])
    if os.environ.get("FAKE_SERVER_FAIL"):
        print("error: failed to load model", file=sys.stderr, flush=True)
        return 3
    port = int(args[args.index("--port") + 1])
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"whisper server listening at http://127.0.0.1:{port}", file=sys.stderr, flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
