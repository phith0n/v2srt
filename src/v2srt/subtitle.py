"""Subtitle cues, SRT reading/writing and clean-up of raw speech recognition output."""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, TextIO

from .timeutil import format_srt_time, parse_duration

MIN_CUE_MS = 300


@dataclass(slots=True)
class Cue:
    start: int  # milliseconds
    end: int
    text: str
    translation: str | None = None


class OutputMode(StrEnum):
    SOURCE = "source"  # recognized text only
    TRANSLATED = "translated"  # translation, falls back to the source text
    BILINGUAL = "bilingual"  # translation on the first line, source text below


def cue_body(cue: Cue, mode: OutputMode) -> str:
    if mode is OutputMode.SOURCE or not cue.translation:
        return cue.text
    if mode is OutputMode.BILINGUAL and cue.translation.strip() != cue.text.strip():
        return f"{cue.translation}\n{cue.text}"
    return cue.translation


def _clean_body(text: str) -> str:
    # Blank lines would terminate the SRT block early.
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


class SrtWriter:
    """Appends numbered cues to an SRT file, flushing after every cue so progress is never lost."""

    def __init__(self, path: Path, mode: OutputMode):
        self.path = path
        self.mode = mode
        self.count = 0
        self._fh: TextIO = open(path, "w", encoding="utf-8", newline="\n")

    def write(self, cue: Cue) -> None:
        body = _clean_body(cue_body(cue, self.mode))
        if not body:
            return
        self.count += 1
        self._fh.write(f"{self.count}\n{format_srt_time(cue.start)} --> {format_srt_time(cue.end)}\n{body}\n\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


_SRT_TIME_LINE = re.compile(r"^\s*(\S+)\s*-->\s*(\S+)")


def parse_srt(text: str) -> list[Cue]:
    """Parse SRT content leniently (BOM, CRLF, missing or wrong indices, ``.`` as millisecond separator)."""
    cues: list[Cue] = []
    lines = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        m = _SRT_TIME_LINE.match(lines[i])
        if not m:
            i += 1
            continue
        try:
            start, end = parse_duration(m.group(1)), parse_duration(m.group(2))
        except ValueError:
            i += 1
            continue
        i += 1
        body: list[str] = []
        while i < len(lines) and lines[i].strip():
            # A following cue without a blank separator line: index line + timing line.
            if _SRT_TIME_LINE.match(lines[i]) or (
                lines[i].strip().isdigit() and i + 1 < len(lines) and _SRT_TIME_LINE.match(lines[i + 1])
            ):
                break
            body.append(lines[i].strip())
            i += 1
        if body:
            cues.append(Cue(start=start, end=max(end, start + MIN_CUE_MS), text="\n".join(body)))
    return cues


_NON_SPEECH = re.compile(
    r"^[\[(（【*♪\s]*(blank[_ ]?audio|silence|no speech|music|music playing|background music|"
    r"inaudible|noise|音楽|音乐|无声|字幕由.*提供.*|(?:subtitles?|captions?) by .*)[\])）】*♪\s.]*$",
    re.IGNORECASE,
)
_HAS_CONTENT = re.compile(r"[\w♪]")


class CueCleaner:
    """Filters typical whisper artefacts from a stream of cues.

    - drops non-speech tags such as ``[BLANK_AUDIO]`` or ``(music)`` and punctuation-only cues
    - drops hallucination loops (the same text repeated over and over)
    - repairs broken timings
    """

    def __init__(self, max_repeats: int = 2):
        self.max_repeats = max_repeats
        self._last_key: str | None = None
        self._repeats = 0
        self._last_start = 0

    def __call__(self, cue: Cue) -> Cue | None:
        text = " ".join(cue.text.replace("�", "").split())
        if not text or not _HAS_CONTENT.search(text) or _NON_SPEECH.match(text):
            return None

        key = re.sub(r"\W+", "", text.lower())
        if key == self._last_key:
            self._repeats += 1
            if self._repeats >= self.max_repeats:
                return None
        else:
            self._last_key, self._repeats = key, 0

        start = max(cue.start, self._last_start)
        end = cue.end if cue.end > start else start + MIN_CUE_MS
        self._last_start = start
        return Cue(start=start, end=end, text=text, translation=cue.translation)


class CueSequencer:
    """Holds back one cue so that its end time can be trimmed to not overlap the next cue."""

    def __init__(self, emit: Callable[[Cue], None]):
        self._emit = emit
        self._pending: Cue | None = None

    def push(self, cue: Cue) -> None:
        prev = self._pending
        if prev is not None:
            if prev.end > cue.start > prev.start:
                prev.end = cue.start
            if prev.end - prev.start < MIN_CUE_MS:
                prev.end = prev.start + MIN_CUE_MS
            self._emit(prev)
        self._pending = cue

    def flush(self) -> None:
        if self._pending is not None:
            self._emit(self._pending)
            self._pending = None
