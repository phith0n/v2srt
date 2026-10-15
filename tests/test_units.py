import sys
import wave
from pathlib import Path

import pytest

from v2srt.config import Settings, language_base, language_name, settings_from_env, split_args
from v2srt.errors import ConfigError
from v2srt.media import Chunk, plan_boundaries
from v2srt.pipeline import OrderedSink
from v2srt.proc import ProcessGroup
from v2srt.subtitle import Cue, CueCleaner, CueSequencer, OutputMode, cue_body, max_display_ms, parse_srt
from v2srt.timeutil import format_clock, format_srt_time, parse_duration
from v2srt.whisper import ChunkInfo, Whisper, parse_segment_line, parse_vad_segment, plan_pieces, uncovered_speech


@pytest.mark.parametrize(
    "text, ms",
    [
        ("90", 90_000),
        ("1.5", 1_500),
        ("20m", 1_200_000),
        ("1h30m", 5_400_000),
        ("1m30s", 90_000),
        ("500ms", 500),
        ("01:30:00", 5_400_000),
        ("1:30", 90_000),
        ("00:01:02,500", 62_500),
        ("00:01:02.5", 62_500),
    ],
)
def test_parse_duration(text, ms):
    assert parse_duration(text) == ms


@pytest.mark.parametrize("text", ["", "abc", "1x", "-5"])
def test_parse_duration_invalid(text):
    with pytest.raises(ValueError):
        parse_duration(text)


def test_time_formatting():
    assert format_srt_time(3_723_004) == "01:02:03,004"
    assert format_clock(3_723_004) == "1:02:03"
    assert format_clock(62_000) == "01:02"


def test_parse_segment_line():
    cue = parse_segment_line("[00:01:02.340 --> 00:01:04.000]   Hello there.")
    assert cue == Cue(62_340, 64_000, "Hello there.")
    assert parse_segment_line("whisper_init: loading model") is None


def test_parse_vad_segment():
    # whisper.cpp reports seconds relative to the input file
    line = "whisper_vad_segments_from_probs: VAD segment 17: start = 512.35, end = 512.83 (duration: 0.48)"
    assert parse_vad_segment(line) == (512_350, 512_830)
    assert parse_vad_segment("whisper_vad_segments_from_probs: Final speech segments after filtering: 18") is None
    # padded copies of the same segments must not be counted twice
    assert parse_vad_segment("whisper_vad: Including segment 0: 3.20 - 4.95 (duration: 1.75)") is None
    assert parse_vad_segment("whisper_vad: vad_segment_info: orig_start: 3.20, orig_end: 4.85, vad_start: 0.00") is None


def test_transcribe_collects_vad_segments(tmp_path):
    wav = tmp_path / "chunk-000.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(bytes(2 * 16_000 * 7))
    model = tmp_path / "ggml-fake.bin"
    model.write_bytes(b"0")
    fake = Path(__file__).with_name("fake_whisper.py")
    settings = Settings(whisper_model=str(model), vad_model=str(model), use_vad=True)
    whisper = Whisper(settings, "unused", ProcessGroup(), command_prefix=[sys.executable, str(fake)])

    info = ChunkInfo()
    cues = list(whisper.transcribe(Chunk(0, 60_000, 67_000, wav), info=info))
    assert len(cues) == 4 and cues[0].start == 60_000  # cues are absolute, VAD segments chunk-relative
    assert info.speech == [(700, 1300), (2700, 3300), (4700, 5300), (6700, 7000)]
    assert info.speech_ms == 600 * 3 + 300


def test_uncovered_speech_and_pieces():
    speech = [(1_000, 2_000), (10_000, 11_000), (11_500, 12_000), (30_000, 30_100), (50_000, 70_000), (70_500, 85_000)]
    cues = [(400, 600), (9_000, 10_200)]
    missed = uncovered_speech(speech, cues)
    # (1000, 2000) is 400 ms after a cue end -> covered; (10000, 11000) overlaps a cue
    assert missed == [(11_500, 12_000), (30_000, 30_100), (50_000, 70_000), (70_500, 85_000)]
    pieces = plan_pieces(missed, duration=84_000)
    # 30.0-30.1 s has too little speech; 50-70 s and 70.5-85 s would exceed 30 s together; end clamped to duration
    assert pieces == [(11_200, 12_300), (49_700, 70_300), (70_200, 84_000)]


def test_strict_cleaner_drops_backchannel_loops_and_stock_phrases():
    strict, normal = CueCleaner(strict=True), CueCleaner()
    for text in ("うん、うん", "はい。", "あっ", "Yeah.", "嗯嗯", "あああああああああ", "おやすみなさい", "お疲れ様でした"):
        assert strict(Cue(0, 1000, text)) is None, text
    assert normal(Cue(0, 1000, "うん、うん")) is not None  # main-pass cues are not affected
    for text in ("ちょっと待って", "そうですね", "お疲れ様でした、また明日", "本当にありがとう"):
        assert strict(Cue(0, 1000, text)) is not None, text


def _fake_whisper(tmp_path, seconds: int, **settings) -> tuple[Whisper, Path]:
    wav = tmp_path / "chunk-000.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(bytes(2 * 16_000 * seconds))
    model = tmp_path / "ggml-fake.bin"
    model.write_bytes(b"0")
    fake = Path(__file__).with_name("fake_whisper.py")
    options = dict(whisper_model=str(model), vad_model=str(model), use_vad=True) | settings
    return Whisper(Settings(**options), "unused", ProcessGroup(), command_prefix=[sys.executable, str(fake)]), wav


def test_recognize_fills_missed_speech(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_WHISPER_DROP", "3")  # every 3rd line is missing although VAD reports speech
    monkeypatch.setenv("FAKE_WHISPER_BACKCHANNEL", "1")
    whisper, wav = _fake_whisper(tmp_path, 20)
    cues = whisper.recognize(Chunk(0, 60_000, 80_000, wav))

    texts = [c.text for c in cues]
    main = [t for t in texts if t.startswith("Line ")]
    filled = [t for t in texts if t.startswith("Recovered ")]
    # lines 3, 6, 9 were dropped by the main pass and line 2 is [BLANK_AUDIO]; their regions get recovered
    assert main == ["Line 1 at 0", "Line 4 at 6000", "Line 5 at 8000", "Line 7 at 12000", "Line 8 at 14000",
                    "Line 10 at 18000"]  # fmt: skip
    assert len(filled) == 4
    assert "うん、うん" not in texts  # backchannel from the gap pass is discarded
    starts = [c.start for c in cues]
    assert starts == sorted(starts) and all(60_000 <= c.start and c.end <= 80_000 for c in cues)
    assert not (wav.parent / "gaps-000").exists()


def test_recognize_without_gap_fill(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_WHISPER_DROP", "3")
    whisper, wav = _fake_whisper(tmp_path, 20, gap_fill=False)
    assert not any(c.text.startswith("Recovered") for c in whisper.recognize(Chunk(0, 0, 20_000, wav)))


def test_parse_srt_lenient():
    text = "﻿1\r\n00:00:01,000 --> 00:00:02,500\r\nHello\r\nworld\r\n\r\n2\r\n00:00:03.000 --> 00:00:04.000\r\nAgain\r\n3\r\n00:00:05,000 --> 00:00:06,000\r\nNo blank line before\r\n"
    cues = parse_srt(text)
    assert [c.text for c in cues] == ["Hello\nworld", "Again", "No blank line before"]
    assert cues[0].start == 1000 and cues[0].end == 2500
    assert cues[1].start == 3000


def test_cue_body_modes():
    cue = Cue(0, 1000, "Hello", "你好")
    assert cue_body(cue, OutputMode.SOURCE) == "Hello"
    assert cue_body(cue, OutputMode.TRANSLATED) == "你好"
    assert cue_body(cue, OutputMode.BILINGUAL) == "你好\nHello"
    assert cue_body(Cue(0, 1000, "Hello"), OutputMode.TRANSLATED) == "Hello"


def test_cleaner_drops_artefacts_and_loops():
    clean = CueCleaner(max_repeats=2)
    out = [
        clean(Cue(0, 1000, " [BLANK_AUDIO] ")),
        clean(Cue(1000, 2000, "(Music)")),
        clean(Cue(2000, 3000, "...")),
        clean(Cue(3000, 4000, "Thank you.")),
        clean(Cue(4000, 5000, "Thank you!")),
        clean(Cue(5000, 6000, "thank you")),
        clean(Cue(6000, 6000, "Bye")),
        clean(Cue(7000, 8000, "(laughs)")),
    ]
    texts = [c.text if c else None for c in out]
    assert texts == [None, None, None, "Thank you.", "Thank you!", None, "Bye", "(laughs)"]
    assert out[6].end > out[6].start


def test_cleaner_drops_known_hallucinations_only():
    clean = CueCleaner()
    assert clean(Cue(0, 1000, "ご視聴ありがとうございました")) is None
    assert clean(Cue(1000, 2000, "チャンネル登録よろしくお願いします")) is None
    # common in real dialogue, must be kept
    assert clean(Cue(2000, 3000, "お疲れ様でした")) is not None
    assert clean(Cue(3000, 4000, "ありがとうございました")) is not None


def test_max_display_ms():
    assert max_display_ms("あー") == 3000  # minimum
    assert max_display_ms("今日はいい天気ですね") == 1500 + 10 * 500
    assert max_display_ms("Hello there, how are you") == 1500 + 5 * 750
    assert max_display_ms("OK 了解") == 1500 + 750 + 2 * 500


def test_cleaner_caps_stretched_cues():
    clean = CueCleaner()
    # VAD artefact: a short line stretched over minutes keeps its start, the end is capped
    stretched = clean(Cue(1_000, 200_000, "今日はいい天気ですね"))
    assert (stretched.start, stretched.end) == (1_000, 1_000 + 6_500)
    # realistic durations are untouched, including long English sentences
    long_text = "Some historians argue that without the printing press there would have been no renaissance"
    normal = clean(Cue(300_000, 308_400, long_text))
    assert (normal.start, normal.end) == (300_000, 308_400)


def test_cleaner_logs_every_decision(caplog):
    clean = CueCleaner(label="whisper[3]")
    with caplog.at_level("DEBUG", logger="v2srt.subtitle"):
        clean(Cue(0, 1000, "[BLANK_AUDIO]"))
        clean(Cue(1000, 600_000, "はい"))
    messages = [r.getMessage() for r in caplog.records]
    assert "whisper[3]" in messages[0] and "dropped: non-speech" in messages[0]
    assert "end capped" in messages[1]


def test_sequencer_trims_overlaps():
    written = []
    seq = CueSequencer(written.append)
    seq.push(Cue(0, 5000, "a"))
    seq.push(Cue(3000, 4000, "b"))
    seq.push(Cue(3100, 6000, "c"))
    seq.flush()
    assert [(c.start, c.end) for c in written] == [(0, 3000), (3000, 3300), (3100, 6000)]


def test_ordered_sink_out_of_order():
    out = []
    sink = OrderedSink(2, lambda cue: out.append(cue.text))
    sink.put(1, 0, [Cue(0, 1, "c")])
    sink.put(0, 1, [Cue(0, 1, "b")])
    assert out == []
    sink.finish_chunk(1, 1)
    sink.put(0, 0, [Cue(0, 1, "a")])
    assert out == ["a", "b"]
    sink.finish_chunk(0, 2)
    assert out == ["a", "b", "c"] and sink.done


def test_ordered_sink_empty_chunk():
    out = []
    sink = OrderedSink(2, lambda cue: out.append(cue.text))
    sink.finish_chunk(0, 0)
    sink.put(1, 0, [Cue(0, 1, "x")])
    sink.finish_chunk(1, 1)
    assert out == ["x"] and sink.done


def test_plan_boundaries():
    hour = 3_600_000
    assert plan_boundaries(hour, segment_ms=20 * 60_000, cut_ms=[], silences=None) == [1_200_000, 2_400_000]
    # short audio is not split
    assert plan_boundaries(22 * 60_000, segment_ms=20 * 60_000, cut_ms=[], silences=None) == []
    # explicit cuts win and are clamped to the audio
    assert plan_boundaries(hour, segment_ms=0, cut_ms=[hour * 2, 600_000], silences=None) == [600_000]
    # split points snap to the longest nearby silence
    silences = [(1_150_000, 1_150_500), (1_230_000, 1_232_000), (1_500_000, 1_510_000)]
    assert plan_boundaries(hour, segment_ms=20 * 60_000, cut_ms=[], silences=silences)[0] == 1_231_000


def test_settings_from_env():
    settings, sources = settings_from_env(
        {
            "V2SRT_WHISPER_MODEL": "~/models/ggml-large.bin",
            "GEMINI_API_KEY": "secret",
            "V2SRT_SEGMENT": "30m",
            "V2SRT_BILINGUAL": "yes",
            "V2SRT_CUT": "1:00:00, 2:00:00",
            "V2SRT_TRANSLATE_WORKERS": "",
        }
    )
    assert settings.whisper_model.endswith("ggml-large.bin") and "~" not in settings.whisper_model
    assert settings.gemini_api_key == "secret" and sources["gemini_api_key"] == "GEMINI_API_KEY"
    assert settings.segment_ms == 1_800_000
    assert settings.bilingual is True
    assert settings.cut_ms == [3_600_000, 7_200_000]
    assert settings.translate_workers == 4
    assert settings.translation_enabled


def test_settings_from_env_invalid():
    with pytest.raises(ConfigError, match="V2SRT_WHISPER_JOBS"):
        settings_from_env({"V2SRT_WHISPER_JOBS": "zero"})


def test_split_args():
    assert split_args('--max-len 60 --prompt "two words"') == ["--max-len", "60", "--prompt", "two words"]


def test_language_helpers():
    assert language_name("zh-CN") == "Simplified Chinese"
    assert language_name("Klingon") == "Klingon"
    assert language_name("auto") is None
    assert language_base("zh-CN") == "zh"
    assert language_base("Simplified Chinese") is None


def test_env_example_matches_template():
    from pathlib import Path

    from v2srt.config import ENV_TEMPLATE

    example = Path(__file__).parent.parent / ".env.example"
    assert example.read_text(encoding="utf-8") == ENV_TEMPLATE, "regenerate .env.example from ENV_TEMPLATE"
