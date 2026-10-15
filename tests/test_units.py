import pytest

from v2srt.config import language_base, language_name, settings_from_env, split_args
from v2srt.errors import ConfigError
from v2srt.media import plan_boundaries
from v2srt.pipeline import OrderedSink
from v2srt.subtitle import Cue, CueCleaner, CueSequencer, OutputMode, cue_body, parse_srt
from v2srt.timeutil import format_clock, format_srt_time, parse_duration
from v2srt.whisper import parse_segment_line


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
