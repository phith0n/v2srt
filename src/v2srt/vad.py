"""Speech detection for choosing split points, with whisper.cpp's ``whisper-vad-speech-segments``.

The tool ships with whisper.cpp next to ``whisper-cli`` and runs the same Silero VAD model that whisper-cli uses with
``--vad``. Unlike ffmpeg's ``silencedetect``, which needs a fixed loudness threshold, it still finds the pauses
between sentences under background music or noise.

It prints, on stdout, after processing the whole file (there is no progress output)::

    <empty line>
    Detected 2 speech segments:
    Speech segment 0: start = 29.00, end = 221.00
    Speech segment 1: start = 330.00, end = 377.00

The times are **centiseconds** (``printf("%.2f")`` of whole centiseconds), see
``examples/vad-speech-segments/speech.cpp`` and ``whisper_vad_segments_from_probs`` in whisper.cpp. The example in
that directory's README shows seconds, which is outdated. Errors in the command line arguments print the usage and
exit with code 0, so the header line is required for a run to count as successful.
"""

import logging
import re
import shutil
import sys
from pathlib import Path

from .config import Settings
from .errors import ToolError
from .proc import ProcessGroup

log = logging.getLogger(__name__)

VAD_TOOL = "whisper-vad-speech-segments"
MIN_PAUSE_MS = 400  # shorter gaps between speech segments are not used as split points (silencedetect uses 0.4 s)

_HEADER_RE = re.compile(r"^Detected (\d+) speech segments:")
_SEGMENT_RE = re.compile(r"^Speech segment \d+: start = (\d+(?:\.\d+)?), end = (\d+(?:\.\d+)?)")


def find_vad_tool(whisper_cli: str) -> str | None:
    """``whisper-vad-speech-segments`` next to the resolved whisper-cli executable, else on PATH."""
    cli = Path(whisper_cli)
    names = (VAD_TOOL + ".exe", VAD_TOOL) if sys.platform == "win32" else (VAD_TOOL,)
    for name in names:
        candidate = cli.with_name(name)
        if candidate.is_file():
            return str(candidate)
    return shutil.which(VAD_TOOL)


def parse_speech_segment(line: str) -> tuple[int, int] | None:
    """``(start_ms, end_ms)`` from a ``Speech segment N: start = <cs>, end = <cs>`` line."""
    m = _SEGMENT_RE.match(line.strip())
    if not m:
        return None
    return round(float(m.group(1)) * 10), round(float(m.group(2)) * 10)


def speech_to_pauses(
    speech: list[tuple[int, int]], duration_ms: int, *, min_pause_ms: int = MIN_PAUSE_MS
) -> list[tuple[int, int]]:
    """The stretches without speech, ``(start_ms, end_ms)``, at least ``min_pause_ms`` long.

    Includes the stretch before the first and after the last speech segment. Overlapping segments are tolerated.
    """
    pauses = []
    position = 0
    for start, end in sorted(speech):
        start = min(start, duration_ms)
        if start - position >= min_pause_ms:
            pauses.append((position, start))
        position = max(position, min(end, duration_ms))
    if duration_ms - position >= min_pause_ms:
        pauses.append((position, duration_ms))
    return pauses


class SpeechDetector:
    """Runs ``whisper-vad-speech-segments`` on a whole audio file."""

    def __init__(self, settings: Settings, executable: str, procs: ProcessGroup, command_prefix: list[str] | None = None):
        if not settings.vad_model:
            raise ValueError("VAD model is required")
        self.settings = settings
        self.procs = procs
        self.executable = executable
        self.prefix = command_prefix or [executable]

    def command(self, wav: Path) -> list[str]:
        s = self.settings
        # VAD runs on the CPU with the tool's defaults (4 threads), like whisper-cli's own --vad.
        return [*self.prefix, "--vad-model", str(s.vad_model), "--vad-threshold", str(s.vad_threshold), "-f", str(wav)]

    def detect(self, wav: Path, duration_ms: int) -> list[tuple[int, int]]:
        """Speech segments ``(start_ms, end_ms)`` of ``wav``. Raises :class:`ToolError` on failure or odd output."""
        expected: int | None = None
        segments: list[tuple[int, int]] = []

        def on_stderr(line: str) -> None:
            if line.strip():
                log.debug("vad: %s", line)

        for line in self.procs.stream(self.command(wav), on_stderr=on_stderr, name=VAD_TOOL):
            if (segment := parse_speech_segment(line)) is not None:
                segments.append(segment)
            elif m := _HEADER_RE.match(line.strip()):
                expected = int(m.group(1))
            elif line.strip():
                log.debug("vad stdout: %s", line)

        if expected is None:
            raise ToolError(f"{VAD_TOOL} printed no speech segments (wrong arguments or an incompatible version?)")
        if len(segments) != expected:
            raise ToolError(f"{VAD_TOOL} announced {expected} speech segments but printed {len(segments)}")
        if any(start > end or end > duration_ms + 1000 for start, end in segments):
            raise ToolError(f"{VAD_TOOL} reported speech segments outside the audio (unexpected time unit?)")
        return segments
