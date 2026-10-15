"""Subprocess and thread helpers that stay responsive to cancellation."""

import logging
import queue
import subprocess
import threading
from collections import deque
from concurrent.futures import Future
from typing import Callable, Iterator, Sequence

from .errors import Cancelled, ToolError

log = logging.getLogger(__name__)

# Exit codes of a process that was stopped by Ctrl+C (POSIX SIGINT, Windows STATUS_CONTROL_C_EXIT).
_INTERRUPTED_CODES = {-2, 130, 0xC000013A, -1073741510}


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace").rstrip("\r\n")


class ProcessGroup:
    """Keeps track of running subprocesses so that all of them can be stopped at once."""

    def __init__(self) -> None:
        self.cancelled = threading.Event()
        self._procs: set[subprocess.Popen] = set()
        self._lock = threading.Lock()

    def _start(self, cmd: Sequence[str]) -> subprocess.Popen:
        with self._lock:
            if self.cancelled.is_set():
                raise Cancelled("cancelled")
            log.debug("run: %s", subprocess.list2cmdline(list(cmd)))
            try:
                proc = subprocess.Popen(
                    list(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            except OSError as e:
                raise ToolError(f"failed to start {cmd[0]}: {e}") from None
            self._procs.add(proc)
            return proc

    def _release(self, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        with self._lock:
            self._procs.discard(proc)

    def cancel(self) -> None:
        self.cancelled.set()
        with self._lock:
            procs = list(self._procs)
        for proc in procs:
            try:
                proc.kill()
            except OSError:
                pass

    def stream(
        self,
        cmd: Sequence[str],
        *,
        on_stderr: Callable[[str], None] | None = None,
        name: str | None = None,
    ) -> Iterator[str]:
        """Run ``cmd`` and yield its stdout line by line while it is running.

        stderr is drained on a background thread (so the pipe never fills up) and passed to ``on_stderr``.
        Raises :class:`ToolError` with the tail of stderr if the process fails.
        """
        name = name or str(cmd[0])
        proc = self._start(cmd)
        tail: deque[str] = deque(maxlen=30)

        def drain() -> None:
            assert proc.stderr is not None
            for raw in proc.stderr:
                line = _decode(raw)
                tail.append(line)
                if on_stderr is not None:
                    try:
                        on_stderr(line)
                    except Exception:  # never let a callback kill the drain thread
                        log.exception("stderr handler failed")

        reader = threading.Thread(target=drain, name=f"{name}-stderr", daemon=True)
        reader.start()
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                yield _decode(raw)
            code = proc.wait()
            reader.join(timeout=5)
        finally:
            self._release(proc)

        if self.cancelled.is_set() or code in _INTERRUPTED_CODES:
            raise Cancelled("cancelled")
        if code != 0:
            details = "\n".join(line for line in tail if line.strip())
            raise ToolError(f"{name} failed with exit code {code}" + (f":\n{details}" if details else ""))

    def run(self, cmd: Sequence[str], **kwargs) -> list[str]:
        return list(self.stream(cmd, **kwargs))


class DaemonExecutor:
    """A minimal thread pool built on daemon threads.

    Unlike ``ThreadPoolExecutor`` it never blocks interpreter shutdown, so Ctrl+C exits immediately even while
    a slow HTTP request is still in flight.
    """

    def __init__(self, workers: int, name: str):
        self._queue: queue.Queue = queue.Queue()
        self._threads = [
            threading.Thread(target=self._work, name=f"{name}-{i}", daemon=True) for i in range(max(1, workers))
        ]
        for t in self._threads:
            t.start()

    def submit(self, fn: Callable, *args) -> Future:
        future: Future = Future()
        self._queue.put((future, fn, args))
        return future

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            future, fn, args = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args))
            except BaseException as e:  # noqa: BLE001 - delivered through the future
                future.set_exception(e)

    def shutdown(self, cancel_pending: bool = False) -> None:
        if cancel_pending:
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    item[0].cancel()
        for _ in self._threads:
            self._queue.put(None)
