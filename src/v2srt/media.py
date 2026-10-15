"""Locating tools and preparing audio with ffmpeg."""

import json
import re
import shutil
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .errors import ToolError
from .proc import ProcessGroup

SAMPLE_RATE = 16_000


def resolve_executable(value: str, name: str, env_var: str) -> str:
    """Find an executable given a path, a directory containing it, or a command name on PATH."""
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        for exe in (name + ".exe", name) if sys.platform == "win32" else (name,):
            if (candidate / exe).is_file():
                return str(candidate / exe)
    elif candidate.is_file():
        return str(candidate)
    found = shutil.which(value)
    if found:
        return found
    raise ToolError(f"cannot find {name} ({value!r}). Install it, add it to PATH, or set {env_var}.")


def resolve_ffprobe(ffmpeg: str, ffprobe: str | None) -> str:
    if ffprobe:
        return resolve_executable(ffprobe, "ffprobe", "V2SRT_FFPROBE")
    ffmpeg_path = Path(ffmpeg)
    sibling = ffmpeg_path.with_name(ffmpeg_path.name.replace("ffmpeg", "ffprobe"))
    if sibling != ffmpeg_path and sibling.is_file():
        return str(sibling)
    return resolve_executable("ffprobe", "ffprobe", "V2SRT_FFPROBE")


@dataclass(frozen=True)
class MediaInfo:
    duration_ms: int
    audio_streams: int


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int  # ms, position in the full audio
    end: int
    path: Path

    @property
    def duration(self) -> int:
        return self.end - self.start


class Media:
    def __init__(self, ffmpeg: str, ffprobe: str, procs: ProcessGroup):
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.procs = procs

    def probe(self, path: Path) -> MediaInfo:
        out = self.procs.run(
            [self.ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(path)],
            name="ffprobe",
        )
        try:
            data = json.loads("\n".join(out))
        except json.JSONDecodeError:
            raise ToolError(f"ffprobe returned unexpected output for {path}") from None
        audio = sum(1 for s in data.get("streams", []) if s.get("codec_type") == "audio")
        if audio == 0:
            raise ToolError(f"{path.name} has no audio track")
        try:
            duration = round(float(data["format"]["duration"]) * 1000)
        except (KeyError, TypeError, ValueError):
            duration = 0  # unknown (some streams do not report it); progress will be indeterminate
        return MediaInfo(duration_ms=duration, audio_streams=audio)

    def extract_audio(
        self,
        src: Path,
        dst: Path,
        *,
        audio_track: int | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> None:
        """Convert the (selected) audio track to 16 kHz mono PCM, the format whisper.cpp expects."""
        cmd = [self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
        if audio_track is not None:
            cmd += ["-map", f"0:a:{audio_track}"]
        cmd += [
            "-vn", "-sn", "-dn",
            "-af", "aresample=async=1",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
            "-progress", "pipe:1", "-nostats",
            str(dst),
        ]  # fmt: skip
        for line in self.procs.stream(cmd, name="ffmpeg"):
            key, _, value = line.partition("=")
            if key == "out_time_us" and on_progress and value.strip().isdigit():
                on_progress(int(value) // 1000)

    def detect_silences(
        self, wav: Path, *, noise_db: int = -35, min_silence: float = 0.4, on_progress: Callable[[int], None] | None = None
    ) -> list[tuple[int, int]]:
        """Return ``(start_ms, end_ms)`` of silent stretches."""
        silences: list[tuple[int, int]] = []
        start: list[float] = []
        start_re = re.compile(r"silence_start:\s*(-?[\d.]+)")
        end_re = re.compile(r"silence_end:\s*([\d.]+)")

        def on_stderr(line: str) -> None:
            if m := start_re.search(line):
                start[:] = [max(0.0, float(m.group(1)))]
            elif (m := end_re.search(line)) and start:
                silences.append((round(start[0] * 1000), round(float(m.group(1)) * 1000)))
                start.clear()

        cmd = [
            self.ffmpeg, "-nostdin", "-hide_banner", "-i", str(wav),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
            "-f", "null", "-progress", "pipe:1", "-nostats", "-",
        ]  # fmt: skip
        for line in self.procs.stream(cmd, on_stderr=on_stderr, name="ffmpeg"):
            key, _, value = line.partition("=")
            if key == "out_time_us" and on_progress and value.strip().isdigit():
                on_progress(int(value) // 1000)
        return silences


def plan_boundaries(
    duration_ms: int,
    *,
    segment_ms: int,
    cut_ms: list[int],
    silences: list[tuple[int, int]] | None = None,
) -> list[int]:
    """Choose split points (ms) for the audio.

    Explicit cut points are used as-is. Otherwise the audio is divided into roughly equal segments of about
    ``segment_ms`` and every split point is moved to the most suitable nearby silence, so no sentence is cut
    in half.
    """
    if cut_ms:
        return sorted({c for c in cut_ms if 0 < c < duration_ms})
    if segment_ms <= 0 or duration_ms <= segment_ms * 1.25:
        return []

    count = max(1, round(duration_ms / segment_ms))
    window = min(90_000, segment_ms // 4)
    boundaries: list[int] = []
    for k in range(1, count):
        target = duration_ms * k // count
        best = target
        candidates = [
            (s, e) for s, e in (silences or []) if abs((s + e) // 2 - target) <= window and (s + e) // 2 > 0
        ]
        if candidates:
            # Prefer long silences (capped, a 5s pause is not better than a 2s one), then closeness.
            s, e = min(candidates, key=lambda c: (-min(c[1] - c[0], 2000), abs((c[0] + c[1]) // 2 - target)))
            best = (s + e) // 2
        if not boundaries or best > boundaries[-1] + 1000:
            boundaries.append(best)
    return boundaries


def split_wav(wav: Path, boundaries: list[int], out_dir: Path) -> list[Chunk]:
    """Split a PCM wav file at ``boundaries`` (ms). Sample-exact and without re-encoding."""
    with wave.open(str(wav), "rb") as src:
        rate = src.getframerate()
        total_frames = src.getnframes()
        duration = total_frames * 1000 // rate
        if not boundaries:
            return [Chunk(0, 0, duration, wav)]

        edges = [0, *boundaries, duration]
        chunks: list[Chunk] = []
        for i, (start, end) in enumerate(zip(edges, edges[1:])):
            path = out_dir / f"chunk-{i:03}.wav"
            first, last = start * rate // 1000, min(total_frames, end * rate // 1000)
            src.setpos(first)
            with wave.open(str(path), "wb") as dst:
                dst.setparams(src.getparams())
                remaining = last - first
                while remaining > 0:
                    data = src.readframes(min(remaining, rate * 60))
                    if not data:
                        break
                    dst.writeframes(data)
                    remaining -= len(data) // (src.getsampwidth() * src.getnchannels())
            chunks.append(Chunk(i, start, end, path))
        return chunks
