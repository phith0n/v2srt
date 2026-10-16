"""Streaming speech recognition with whisper.cpp's ``whisper-cli``.

whisper-cli prints every segment to stdout as soon as it is decoded::

    [00:00:01.240 --> 00:00:04.020]   Hello there.

so instead of waiting for the JSON file written at the very end, the output is parsed line by line and cues are
handed to the translator while recognition is still running.

With ``--vad``, the speech segments Silero VAD found are reported on stderr (in seconds, relative to the input file)::

    whisper_vad_segments_from_probs: VAD segment 0: start = 3.20, end = 4.85 (duration: 1.65)

They are collected in :class:`ChunkInfo` so the caller can compare them with what whisper actually transcribed.

Gap filling: with VAD, whisper.cpp concatenates all speech of a chunk and decodes it as one stream, which is unstable
on sparse speech — a chunk can come back nearly empty although VAD found plenty of speech. After the main
pass, :meth:`Whisper.recognize` re-transcribes every speech region that no subtitle covers, each region on its own
and without VAD, through a :class:`~v2srt.whisper_server.WhisperServer` that keeps the model loaded for the whole
run, and merges the results.
"""

import io
import logging
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from .config import Settings
from .media import Chunk
from .proc import ProcessGroup
from .subtitle import Cue, CueCleaner, parse_srt
from .timeutil import format_clock
from .whisper_server import WhisperServer

log = logging.getLogger(__name__)

COVER_PAD_MS = 500  # a cue covers speech within this distance of its span
GAP_MERGE_MS = 1000  # uncovered regions closer than this are transcribed as one piece ...
GAP_PIECE_MAX_MS = 30_000  # ... of at most this length (whisper's window)
GAP_PAD_MS = 300  # audio added around each piece
GAP_MIN_SPEECH_MS = 300  # pieces with less speech are breaths and clicks, not worth transcribing

_SEGMENT_RE = re.compile(
    r"^\[(\d+):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[.,](\d{3})\]\s?(.*)$"
)
_LANGUAGE_RE = re.compile(r"auto-detected language:\s*([A-Za-z-]+)")
_PROGRESS_RE = re.compile(r"progress\s*=\s*(\d+)\s*%")
# The raw segments; the later "whisper_vad: Including segment ..." lines add whisper's own padding.
_VAD_SEGMENT_RE = re.compile(r"VAD segment \d+: start = (\d+(?:\.\d+)?), end = (\d+(?:\.\d+)?)")


def _ms(h: str, m: str, s: str, ms: str) -> int:
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def parse_segment_line(line: str) -> Cue | None:
    m = _SEGMENT_RE.match(line.strip())
    if not m:
        return None
    g = m.groups()
    return Cue(start=_ms(*g[0:4]), end=_ms(*g[4:8]), text=g[8].strip())


def parse_vad_segment(line: str) -> tuple[int, int] | None:
    """``(start_ms, end_ms)`` of a speech segment reported by whisper.cpp's VAD, relative to the input file."""
    m = _VAD_SEGMENT_RE.search(line)
    if not m:
        return None
    return round(float(m.group(1)) * 1000), round(float(m.group(2)) * 1000)


@dataclass
class ChunkInfo:
    """What whisper-cli reported about a chunk on stderr."""

    speech: list[tuple[int, int]] = field(default_factory=list)  # VAD speech segments, ms relative to the chunk
    language: str | None = None

    @property
    def speech_ms(self) -> int:
        return sum(end - start for start, end in self.speech)


def uncovered_speech(speech: list[tuple[int, int]], cues: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Speech segments that do not touch any cue span (widened by ``COVER_PAD_MS``)."""
    spans = sorted((start - COVER_PAD_MS, end + COVER_PAD_MS) for start, end in cues)
    result = []
    for start, end in speech:
        if not any(a < end and b > start for a, b in spans):
            result.append((start, end))
    return result


def plan_pieces(segments: list[tuple[int, int]], duration: int) -> list[tuple[int, int]]:
    """Group speech segments into padded pieces of at most ``GAP_PIECE_MAX_MS``, dropping near-silent ones."""
    groups: list[list[tuple[int, int]]] = []
    for start, end in sorted(segments):
        last = groups[-1] if groups else None
        if last and start - last[-1][1] < GAP_MERGE_MS and end - last[0][0] <= GAP_PIECE_MAX_MS:
            last.append((start, end))
        else:
            groups.append([(start, end)])
    pieces = []
    for group in groups:
        if sum(end - start for start, end in group) < GAP_MIN_SPEECH_MS:
            continue
        pieces.append((max(0, group[0][0] - GAP_PAD_MS), min(duration, group[-1][1] + GAP_PAD_MS)))
    return pieces


def _wav_bytes(params: tuple, frames: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setparams(params)
        w.writeframes(frames)
    return buf.getvalue()


class Whisper:
    def __init__(
        self,
        settings: Settings,
        executable: str,
        procs: ProcessGroup,
        command_prefix: list[str] | None = None,
        server: WhisperServer | None = None,
    ):
        if not settings.whisper_model:
            raise ValueError("whisper model is required")
        self.settings = settings
        self.procs = procs
        self.prefix = command_prefix or [executable]
        self.server = server  # for gap filling; without it gaps are left as they are

    @property
    def uses_vad(self) -> bool:
        return bool(self.settings.use_vad and self.settings.vad_model)

    @property
    def fills_gaps(self) -> bool:
        return bool(self.settings.gap_fill and self.uses_vad and self.server)

    def close(self) -> None:
        if self.server is not None:
            self.server.close()

    def command(self, wav: Path) -> list[str]:
        s = self.settings
        cmd = [*self.prefix, "-m", s.whisper_model, "-l", s.language, "-pp"]
        if s.whisper_threads:
            cmd += ["-t", str(s.whisper_threads)]
        if self.uses_vad:
            cmd += ["--vad", "--vad-model", s.vad_model, "--vad-threshold", str(s.vad_threshold)]
        if s.whisper_prompt:
            cmd += ["--prompt", s.whisper_prompt]
        cmd += [*s.whisper_args, "-f", str(wav)]
        return cmd

    def recognize(
        self,
        chunk: Chunk,
        *,
        on_language: Callable[[str], None] | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> list[Cue]:
        """Transcribe a chunk, clean the cues and fill in speech the main pass missed. Cues are absolute and sorted."""
        info = ChunkInfo()

        def language_found(language: str) -> None:
            info.language = language
            if on_language:
                on_language(language)

        cleaner = CueCleaner(label=f"whisper[{chunk.index}]")
        stream = self.transcribe(chunk, on_language=language_found, on_progress=on_progress, info=info)
        cues = [cue for raw in stream if (cue := cleaner(raw)) is not None]
        if not (self.fills_gaps and info.speech):
            return cues

        relative = [(c.start - chunk.start, c.end - chunk.start) for c in cues]
        missed = uncovered_speech(info.speech, relative)
        pieces = plan_pieces(missed, chunk.duration)
        if not pieces:
            return cues
        missed_ms = sum(end - start for start, end in missed)
        filled = self._fill(chunk, pieces, info.language or self.settings.language)
        log.info(
            "segment %d (%s): %d of %d speech regions had no subtitles (%s); transcribed them separately, %d lines added",
            chunk.index + 1, format_clock(chunk.start), len(missed), len(info.speech), format_clock(missed_ms), len(filled),
        )  # fmt: skip
        return sorted(cues + filled, key=lambda c: c.start)

    def _fill(self, chunk: Chunk, pieces: list[tuple[int, int]], language: str) -> list[Cue]:
        assert self.server is not None
        cleaner = CueCleaner(label=f"whisper[{chunk.index}] gap", strict=True)
        filled = []
        with wave.open(str(chunk.path), "rb") as src:
            rate = src.getframerate()
            for start, end in pieces:
                src.setpos(start * rate // 1000)
                audio = _wav_bytes(src.getparams(), src.readframes((end - start) * rate // 1000))
                for cue in parse_srt(self.server.transcribe(audio, language=language)):
                    cue.start = chunk.start + start + min(cue.start, end - start)
                    cue.end = chunk.start + start + min(cue.end, end - start)
                    if (kept := cleaner(cue)) is not None:
                        filled.append(kept)
        return filled

    def transcribe(
        self,
        chunk: Chunk,
        *,
        on_language: Callable[[str], None] | None = None,
        on_progress: Callable[[int], None] | None = None,
        info: ChunkInfo | None = None,
    ) -> Iterator[Cue]:
        """Yield cues (with absolute timestamps) while whisper-cli is still working on the chunk.

        ``info`` is filled from stderr; it is complete once the generator is exhausted.
        """

        def on_stderr(line: str) -> None:
            if m := _PROGRESS_RE.search(line):
                if on_progress:
                    on_progress(chunk.duration * int(m.group(1)) // 100)
                return
            if info is not None and (segment := parse_vad_segment(line)):
                info.speech.append(segment)
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
