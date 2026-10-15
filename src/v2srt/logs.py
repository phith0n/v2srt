"""Per-run log files.

Every run writes a debug level log (commands, tool output, retries, statistics) to the platform's log directory:

- Windows: ``%LOCALAPPDATA%\\v2srt\\Logs``
- macOS:   ``~/Library/Logs/v2srt``
- Linux:   ``~/.local/state/v2srt/log``

Optionally every Gemini request is also recorded as one JSON line in ``<log name>.gemini.jsonl``.
"""

import json
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import platformdirs

from .config import APP_NAME, LOG_OFF

KEEP_RUNS = 50
LOG_FILE_RE = re.compile(r"^\d{8}-\d{6}(?:-\d+)?_[^\\/]*\.(?:log|gemini\.jsonl)$")
_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"

log = logging.getLogger(__name__)


def default_log_dir() -> Path:
    return Path(platformdirs.user_log_dir(APP_NAME, appauthor=False))


def resolve_log_dir(setting: str | None) -> Path | None:
    """The configured log directory; None when logging is disabled."""
    if setting == LOG_OFF:
        return None
    return Path(setting) if setting else default_log_dir()


def log_files(directory: Path) -> list[Path]:
    """Files in ``directory`` that were written by v2srt (never anything else)."""
    if not directory.is_dir():
        return []
    return [p for p in directory.iterdir() if p.is_file() and LOG_FILE_RE.match(p.name)]


def _run_key(path: Path) -> str:
    return path.name.split(".", 1)[0]


def prune(directory: Path, keep: int = KEEP_RUNS) -> None:
    """Delete the files of all but the ``keep`` most recent runs."""
    runs: dict[str, list[Path]] = {}
    for path in log_files(directory):
        runs.setdefault(_run_key(path), []).append(path)
    ordered = sorted(runs.values(), key=lambda files: max(f.stat().st_mtime for f in files), reverse=True)
    for files in ordered[keep:]:
        for f in files:
            try:
                f.unlink()
            except OSError:
                pass


def _slug(label: str) -> str:
    # No dots: the part before the first dot identifies the run (.log and .gemini.jsonl belong together).
    slug = re.sub(r"[^\w-]+", "-", label).strip("-")
    return slug[:40] or "run"


class GeminiRecorder:
    """Appends one JSON object per Gemini request to a ``.gemini.jsonl`` file."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def record(self, entry: dict[str, Any]) -> None:
        line = json.dumps({"time": datetime.now().isoformat(timespec="milliseconds"), **entry}, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class RunLog:
    def __init__(self, path: Path, handler: logging.Handler, gemini: GeminiRecorder | None):
        self.path = path
        self.gemini = gemini
        self._handler = handler

    @classmethod
    def start(cls, directory: Path, label: str, *, record_gemini: bool = False) -> "RunLog | None":
        """Start logging to a new file. Returns None (after a warning) if the directory is not writable."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            prune(directory, KEEP_RUNS - 1)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            name = f"{stamp}_{_slug(label)}"
            path = directory / f"{name}.log"
            n = 1
            while path.exists():
                n += 1
                name = f"{stamp}-{n}_{_slug(label)}"
                path = directory / f"{name}.log"
            handler = logging.FileHandler(path, encoding="utf-8")
        except OSError as e:
            log.warning("cannot write log files to %s (%s); set V2SRT_LOG_DIR or V2SRT_LOG_DIR=off", directory, e)
            return None
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(handler)
        gemini = GeminiRecorder(directory / f"{name}.gemini.jsonl") if record_gemini else None
        return cls(path, handler, gemini)

    def close(self) -> None:
        logging.getLogger().removeHandler(self._handler)
        self._handler.close()
