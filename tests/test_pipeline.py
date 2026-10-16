import io
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fakes import FakeClient, srt_texts
from rich.console import Console

from v2srt.config import Settings
from v2srt.errors import ToolError
from v2srt.media import Media, resolve_ffprobe
from v2srt.pipeline import Pipeline
from v2srt.proc import ProcessGroup
from v2srt.subtitle import parse_srt
from v2srt.translator import GeminiTranslator
from v2srt.ui import Reporter
from v2srt.whisper import Whisper
from v2srt.whisper_server import WhisperServer

FFMPEG = shutil.which("ffmpeg")
FAKE_WHISPER = Path(__file__).with_name("fake_whisper.py")
FAKE_SERVER = Path(__file__).with_name("fake_whisper_server.py")
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is required")


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    """70 s of tone with a 1 s pause every 10 s, in a video container."""
    path = tmp_path_factory.mktemp("media") / "clip.mkv"
    subprocess.run(
        [
            FFMPEG, "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=70",
            "-f", "lavfi", "-i", "sine=f=440:d=70",
            "-af", "volume='if(between(mod(t,10),4,5),0,1)':eval=frame",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
    )  # fmt: skip
    return path


def make_pipeline(tmp_path, *, translate=True, behaviour=None, vad=False, **overrides):
    model = tmp_path / "ggml-fake.bin"
    model.write_bytes(b"0")
    options = dict(segment_ms=30_000, whisper_jobs=2, batch_size=4, translate_workers=3)
    if vad:
        options["vad_model"] = str(model)
    options.update(overrides)
    settings = Settings(whisper_model=str(model), gemini_api_key="k" if translate else None, **options)
    procs = ProcessGroup()
    media = Media(FFMPEG, resolve_ffprobe(FFMPEG, None), procs)
    server = WhisperServer(settings, "unused", procs, command_prefix=[sys.executable, str(FAKE_SERVER)])
    whisper = Whisper(settings, "unused", procs, command_prefix=[sys.executable, str(FAKE_WHISPER)], server=server)
    translator = GeminiTranslator(settings, client=FakeClient(behaviour)) if translate else None
    console = Console(file=io.StringIO(), width=120)
    return Pipeline(settings, Reporter(console), procs=procs, media=media, whisper=whisper, translator=translator)


def test_transcribe_and_translate(tmp_path, video):
    pipeline = make_pipeline(tmp_path, keep_source=True)
    out = tmp_path / "clip.srt"
    result = pipeline.run_media(video, out)

    cues = parse_srt(out.read_text(encoding="utf-8"))
    # 3 segments (split at the pauses near 23 s and 47 s), whisper emits a cue every 2 s, minus [BLANK_AUDIO]
    assert result.cues == len(cues) > 20
    assert all(c.text.startswith("T:Line ") for c in cues)
    starts = [c.start for c in cues]
    assert starts == sorted(starts)
    assert all(a.end <= b.start for a, b in zip(cues, cues[1:]))
    assert cues[-1].end <= 70_100  # AAC adds a few ms of priming samples
    assert result.language == "en" and result.untranslated == 0
    assert result.source_output == tmp_path / "clip.en.srt"
    assert all(t.startswith("Line ") for t in srt_texts(result.source_output))
    assert not list(tmp_path.glob("*.part"))

    # translation requests were made while whisper was still running: several batches, all with context
    calls = pipeline.translator.client.models.calls
    assert len(calls) >= 6
    assert all(len(c["cues"]) <= 4 for c in calls)


def test_split_points_are_in_silences(tmp_path, video):
    from v2srt.media import plan_boundaries

    pipeline = make_pipeline(tmp_path)
    wav = tmp_path / "a.wav"
    pipeline.media.extract_audio(video, wav)
    silences = pipeline.media.detect_silences(wav)
    assert len(silences) >= 5
    for b in plan_boundaries(70_000, segment_ms=30_000, cut_ms=[], silences=silences):
        assert b % 10_000 in range(4_000, 5_000), b


def test_transcribe_only(tmp_path, video):
    pipeline = make_pipeline(tmp_path, translate=False, segment_ms=0)
    out = tmp_path / "clip.srt"
    result = pipeline.run_media(video, out)
    texts = srt_texts(out)
    assert result.cues == len(texts) and texts[0] == "Line 1 at 0"


def test_whisper_failure_keeps_partial_file(tmp_path, video, monkeypatch):
    monkeypatch.setenv("FAKE_WHISPER_FAIL", "1")
    pipeline = make_pipeline(tmp_path)
    out = tmp_path / "clip.srt"
    with pytest.raises(ToolError, match="failed to load model"):
        pipeline.run_media(video, out)
    assert not out.exists()


def test_translate_srt_file(tmp_path):
    src = tmp_path / "talk.en.srt"
    src.write_text(
        "".join(f"{i}\n00:00:{i:02},000 --> 00:00:{i:02},900\nSentence {i}\n\n" for i in range(1, 11)),
        encoding="utf-8",
    )
    pipeline = make_pipeline(tmp_path, bilingual=True)
    out = tmp_path / "talk.zh-CN.srt"
    result = pipeline.run_srt(src, out)
    texts = srt_texts(out)
    assert result.cues == 10
    assert texts[0] == "T:Sentence 1\nSentence 1"


def test_interrupt_stops_quickly_and_keeps_partial(tmp_path, video, monkeypatch):
    import time

    import v2srt.pipeline as pipeline_module

    monkeypatch.setenv("FAKE_WHISPER_DELAY", "0.05")  # ~0.25 s per 10 s chunk, many chunks still pending at 1.5 s
    pipeline = make_pipeline(tmp_path, translate=False, segment_ms=10_000, whisper_jobs=1, batch_size=1)
    real_wait = pipeline_module._wait_all
    started = []

    def interrupted_wait(futures):
        if not started:
            started.append(time.monotonic())
            time.sleep(1.5)  # let the first chunks complete; cues are released chunk by chunk
            raise KeyboardInterrupt
        return real_wait(futures)

    monkeypatch.setattr(pipeline_module, "_wait_all", interrupted_wait)
    temp_dirs_before = set(Path(pipeline_module.tempfile.gettempdir()).glob("v2srt-*"))
    out = tmp_path / "clip.srt"
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_media(video, out)

    assert time.monotonic() - started[0] < 6
    assert not out.exists()
    partial = srt_texts(tmp_path / "clip.srt.part")
    assert partial and partial[0] == "Line 1 at 0"
    assert set(Path(pipeline_module.tempfile.gettempdir()).glob("v2srt-*")) == temp_dirs_before


def test_gap_filling_in_pipeline(tmp_path, video, monkeypatch):
    monkeypatch.setenv("FAKE_WHISPER_DROP", "3")
    monkeypatch.setenv("FAKE_WHISPER_BACKCHANNEL", "1")
    pipeline = make_pipeline(tmp_path, vad=True)
    out = tmp_path / "clip.srt"
    try:
        result = pipeline.run_media(video, out)
    finally:
        pipeline.whisper.close()

    cues = parse_srt(out.read_text(encoding="utf-8"))
    texts = [c.text for c in cues]
    assert result.cues == len(cues) and result.untranslated == 0
    assert any(t.startswith("T:Recovered ") for t in texts)  # recovered speech is translated like the rest
    assert all(t.startswith(("T:Line ", "T:Recovered ")) for t in texts)  # no backchannel, nothing untranslated
    assert len(texts) == len(set((c.start, c.text) for c in cues))  # no duplicates
    starts = [c.start for c in cues]
    assert starts == sorted(starts)
    assert all(a.end <= b.start for a, b in zip(cues, cues[1:]))
