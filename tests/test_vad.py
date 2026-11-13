"""Split points from VAD speech segments (whisper-vad-speech-segments), with a fake tool."""

import io
import math
import sys
import wave
from array import array
from pathlib import Path

import pytest
from rich.console import Console

from v2srt.config import Settings
from v2srt.errors import Cancelled, ToolError
from v2srt.media import plan_boundaries
from v2srt.pipeline import Pipeline
from v2srt.proc import ProcessGroup
from v2srt.ui import Reporter
from v2srt.vad import VAD_TOOL, SpeechDetector, find_vad_tool, parse_speech_segment, speech_to_pauses

FAKE_VAD = Path(__file__).with_name("fake_vad.py")
RATE = 16_000


def write_wav(path: Path, seconds: float, pauses: list[tuple[float, float]], *, amplitude: float = 0.5) -> Path:
    """A 440 Hz tone, silent during ``pauses`` (in seconds)."""
    samples = array("h")
    for n in range(int(seconds * RATE)):
        t = n / RATE
        loud = not any(a <= t < b for a, b in pauses)
        samples.append(int(amplitude * 32767 * math.sin(2 * math.pi * 440 * t)) if loud else 0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.tobytes())
    return path


def make_detector(tmp_path: Path, procs: ProcessGroup | None = None, **settings) -> SpeechDetector:
    model = tmp_path / "ggml-silero-fake.bin"
    model.write_bytes(b"0")
    options = dict(whisper_model=str(model), vad_model=str(model)) | settings
    return SpeechDetector(
        Settings(**options), "unused", procs or ProcessGroup(), command_prefix=[sys.executable, str(FAKE_VAD)]
    )


def test_parse_speech_segment():
    # stdout of examples/vad-speech-segments/speech.cpp; the values are centiseconds
    assert parse_speech_segment("Speech segment 0: start = 29.00, end = 221.00") == (290, 2210)
    assert parse_speech_segment("Speech segment 12: start = 360012.00, end = 360530.00\n") == (3_600_120, 3_605_300)
    assert parse_speech_segment("Detected 5 speech segments:") is None
    assert parse_speech_segment("") is None
    # whisper.cpp's own log line on stderr (seconds) is a different format
    assert parse_speech_segment("whisper_vad_segments_from_probs: VAD segment 0: start = 0.29, end = 2.21") is None


def test_speech_to_pauses():
    speech = [(1_000, 5_000), (5_200, 9_000), (8_000, 8_500), (12_000, 20_000)]
    # leading pause, 200 ms gap is too short, overlapping segments, trailing pause
    assert speech_to_pauses(speech, 30_000) == [(0, 1_000), (9_000, 12_000), (20_000, 30_000)]
    assert speech_to_pauses(speech, 30_000, min_pause_ms=100) == [(0, 1_000), (5_000, 5_200), (9_000, 12_000),
                                                                   (20_000, 30_000)]  # fmt: skip
    assert speech_to_pauses([(0, 30_050)], 30_000) == []  # segments are clamped to the audio
    assert speech_to_pauses([], 30_000) == [(0, 30_000)]


def test_find_vad_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    suffix = ".exe" if sys.platform == "win32" else ""
    cli = tmp_path / f"whisper-cli{suffix}"
    cli.write_bytes(b"")
    assert find_vad_tool(str(cli)) is None
    tool = tmp_path / f"{VAD_TOOL}{suffix}"
    tool.write_bytes(b"")
    assert find_vad_tool(str(cli)) == str(tool)


def test_detect_speech(tmp_path):
    wav = write_wav(tmp_path / "audio.wav", 10, [(2, 3), (6, 8.5)])
    detector = make_detector(tmp_path, vad_threshold=0.3)
    cmd = detector.command(wav)
    assert cmd[2:] == ["--vad-model", detector.settings.vad_model, "--vad-threshold", "0.3", "-f", str(wav)]
    assert detector.detect(wav, 10_000) == [(0, 2_000), (3_000, 6_000), (8_500, 10_000)]


def test_detect_speech_real_output(tmp_path, monkeypatch):
    # stdout of whisper-vad-speech-segments 1.9.5 (Homebrew) with ggml-silero-v5.1.2.bin and --vad-threshold 0.3 on
    # 4:32 of synthetic speech under music; its stderr reported segment 0 as 0.00-3.68 s and 63 as 267.97-271.77 s
    monkeypatch.setenv("FAKE_VAD_REPLAY", str(Path(__file__).parent / "data" / "vad-speech-segments.out"))
    wav = write_wav(tmp_path / "audio.wav", 1, [])
    speech = make_detector(tmp_path).detect(wav, 272_124)
    assert len(speech) == 64
    assert speech[0] == (0, 3_680) and speech[1] == (4_290, 8_160) and speech[-1] == (267_970, 271_770)
    pauses = speech_to_pauses(speech, 272_124)
    assert pauses[0] == (3_680, 4_290) and pauses[-1] == (267_130, 267_970)  # the last 354 ms are too short
    assert all(end - start >= 400 for start, end in pauses)


@pytest.mark.parametrize(
    "env, duration, message",
    [
        ("FAKE_VAD_FAIL", 10_000, "exit code 2"),
        ("FAKE_VAD_USAGE", 10_000, "printed no speech segments"),  # exit code 0 with only the usage text
        (None, 5_000, "outside the audio"),  # e.g. a future version printing another time unit
    ],
)
def test_detect_speech_errors(tmp_path, monkeypatch, env, duration, message):
    if env:
        monkeypatch.setenv(env, "1")
    wav = write_wav(tmp_path / "audio.wav", 10, [(2, 3)])
    with pytest.raises(ToolError, match=message):
        make_detector(tmp_path).detect(wav, duration)


def test_plan_boundaries_uses_vad_pauses(tmp_path):
    # 60 s of "speech" with pauses; with 20 s segments the targets are 20 s and 40 s
    pauses = [(12, 12.3), (18.5, 19.5), (25, 26), (37, 37.2), (43, 44.5)]
    wav = write_wav(tmp_path / "audio.wav", 60, pauses)
    speech = make_detector(tmp_path).detect(wav, 60_000)
    boundaries = plan_boundaries(60_000, segment_ms=20_000, cut_ms=[], silences=speech_to_pauses(speech, 60_000))
    # the 300 ms and 200 ms pauses are ignored; the 1 s pause near 20 s and the 1.5 s pause near 40 s are used
    assert boundaries == [19_000, 43_750]


def test_plan_boundaries_long_pause():
    # music without speech from 9:00 to 15:00: the split point is the target itself, not the middle of the pause
    pauses = [(540_000, 900_000)]
    assert plan_boundaries(3_600_000, segment_ms=600_000, cut_ms=[], silences=pauses)[0] == 600_000
    # a target just before a long pause moves into it, 1 s away from the speech
    pauses = [(605_000, 900_000)]
    assert plan_boundaries(3_600_000, segment_ms=600_000, cut_ms=[], silences=pauses)[0] == 606_000


class StubMedia:
    def __init__(self) -> None:
        self.calls = 0

    def detect_silences(self, wav, *, on_progress=None):
        self.calls += 1
        return [(1, 2)]


def make_pipeline(tmp_path: Path, **settings) -> tuple[Pipeline, StubMedia]:
    procs = ProcessGroup()
    detector = make_detector(tmp_path, procs, **settings)
    media = StubMedia()
    reporter = Reporter(Console(file=io.StringIO(), width=120))
    pipeline = Pipeline(detector.settings, reporter, procs=procs, media=media, speech=detector)  # type: ignore[arg-type]
    return pipeline, media


def test_pipeline_prefers_vad(tmp_path):
    wav = write_wav(tmp_path / "audio.wav", 10, [(2, 3)])
    pipeline, media = make_pipeline(tmp_path)
    with pipeline.reporter.job():
        assert pipeline._pauses(wav, 10_000) == [(2_000, 3_000)]
    assert media.calls == 0


def test_pipeline_falls_back_to_silences(tmp_path, monkeypatch):
    pipeline, media = make_pipeline(tmp_path)
    # no speech at all (e.g. a threshold that is too high): ffmpeg's silences are used
    silent = write_wav(tmp_path / "silent.wav", 3, [(0, 3)])
    with pipeline.reporter.job():
        assert pipeline._pauses(silent, 3_000) == [(1, 2)]
    # the tool fails: the run goes on with ffmpeg's silences
    monkeypatch.setenv("FAKE_VAD_FAIL", "1")
    wav = write_wav(tmp_path / "audio.wav", 3, [(1, 2)])
    with pipeline.reporter.job():
        assert pipeline._pauses(wav, 3_000) == [(1, 2)]
    # no VAD tool
    pipeline.speech = None
    with pipeline.reporter.job():
        assert pipeline._pauses(wav, 3_000) == [(1, 2)]
    assert media.calls == 3


def test_pipeline_vad_cancelled(tmp_path):
    wav = write_wav(tmp_path / "audio.wav", 3, [(1, 2)])
    pipeline, media = make_pipeline(tmp_path)
    pipeline.procs.cancel()
    with pytest.raises(Cancelled), pipeline.reporter.job():
        pipeline._pauses(wav, 3_000)
    assert media.calls == 0
