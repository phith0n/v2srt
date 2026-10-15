"""Streaming speech recognition with whisper.cpp's ``whisper-cli``.

whisper-cli prints every segment to stdout as soon as it is decoded::

    [00:00:01.240 --> 00:00:04.020]   Hello there.

so instead of waiting for the JSON file written at the very end, the output is parsed line by line and cues are
handed to the translator while recognition is still running.
"""

import logging
import re
from pathlib import Path
from typing import Callable, Iterator

from .config import Settings
from .media import Chunk
from .proc import ProcessGroup
from .subtitle import Cue

log = logging.getLogger(__name__)

_SEGMENT_RE = re.compile(
    r"^\[(\d+):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[.,](\d{3})\]\s?(.*)$"
)
_LANGUAGE_RE = re.compile(r"auto-detected language:\s*([A-Za-z-]+)")
_PROGRESS_RE = re.compile(r"progress\s*=\s*(\d+)\s*%")


def _ms(h: str, m: str, s: str, ms: str) -> int:
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def parse_segment_line(line: str) -> Cue | None:
    m = _SEGMENT_RE.match(line.strip())
    if not m:
        return None
    g = m.groups()
    return Cue(start=_ms(*g[0:4]), end=_ms(*g[4:8]), text=g[8].strip())


class Whisper:
    def __init__(self, settings: Settings, executable: str, procs: ProcessGroup, command_prefix: list[str] | None = None):
        if not settings.whisper_model:
            raise ValueError("whisper model is required")
        self.settings = settings
        self.procs = procs
        self.prefix = command_prefix or [executable]

    def command(self, wav: Path) -> list[str]:
        s = self.settings
        cmd = [*self.prefix, "-m", s.whisper_model, "-l", s.language, "-pp"]
        if s.whisper_threads:
            cmd += ["-t", str(s.whisper_threads)]
        if s.use_vad and s.vad_model:
            cmd += ["--vad", "--vad-model", s.vad_model, "--vad-threshold", str(s.vad_threshold)]
        if s.whisper_prompt:
            cmd += ["--prompt", s.whisper_prompt]
        cmd += [*s.whisper_args, "-f", str(wav)]
        return cmd

    def transcribe(
        self,
        chunk: Chunk,
        *,
        on_language: Callable[[str], None] | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> Iterator[Cue]:
        """Yield cues (with absolute timestamps) while whisper-cli is still working on the chunk."""

        def on_stderr(line: str) -> None:
            if m := _PROGRESS_RE.search(line):
                if on_progress:
                    on_progress(chunk.duration * int(m.group(1)) // 100)
                return
            if (m := _LANGUAGE_RE.search(line)) and on_language:
                on_language(m.group(1).lower())
            log.debug("whisper[%d]: %s", chunk.index, line)

        for line in self.procs.stream(self.command(chunk.path), on_stderr=on_stderr, name="whisper-cli"):
            cue = parse_segment_line(line)
            if cue is None:
                if line.strip():
                    log.debug("whisper[%d] stdout: %s", chunk.index, line)
                continue
            cue.start = min(cue.start, chunk.duration) + chunk.start
            cue.end = min(cue.end, chunk.duration) + chunk.start
            yield cue
