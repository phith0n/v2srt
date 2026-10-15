from fakes import FakeClient, api_error

from v2srt.config import Settings
from v2srt.subtitle import Cue
from v2srt.translator import GeminiTranslator


def make(behaviour=None, **settings):
    t = GeminiTranslator(Settings(gemini_api_key="k", **settings), client=FakeClient(behaviour))
    t.backoff_scale = 0.001
    return t


def cues(n):
    return [Cue(i * 1000, i * 1000 + 900, f"line {i}") for i in range(n)]


def test_translates_batch_with_context_and_config():
    t = make()
    batch, previous = cues(3), [Cue(0, 1, "earlier", "之前")]
    assert t.translate(batch, previous=previous, source_language="en", title="My Show") == 0
    assert [c.translation for c in batch] == ["T:line 0", "T:line 1", "T:line 2"]
    call = t.client.models.calls[0]
    assert call["model"] == "gemini-3.8-flash"
    assert "My Show" in call["prompt"] and "earlier  =>  之前" in call["prompt"]
    assert "from English into Simplified Chinese" in call["config"].system_instruction
    assert call["config"].thinking_config.thinking_level.value == "LOW"


def test_missing_ids_are_requested_again():
    def behaviour(n, items):
        if n == 1:
            return [(items[0]["id"], "first only")]
        return None

    t = make(behaviour)
    batch = cues(3)
    assert t.translate(batch) == 0
    assert [c.translation for c in batch] == ["first only", "T:line 1", "T:line 2"]
    assert [len(c["cues"]) for c in t.client.models.calls] == [3, 2]


def test_transient_errors_are_retried():
    def behaviour(n, items):
        if n <= 2:
            raise api_error(503, "overloaded", "UNAVAILABLE")

    t = make(behaviour)
    batch = cues(2)
    assert t.translate(batch) == 0
    assert t.stats.retries == 2


def test_failing_cue_is_isolated_by_splitting():
    def behaviour(n, items):
        if any("line 2" == c["text"] for c in items):
            raise api_error(500, "internal")

    t = make(behaviour)
    batch = cues(4)
    assert t.translate(batch) == 1
    assert [c.translation for c in batch] == ["T:line 0", "T:line 1", None, "T:line 3"]


def test_fatal_error_disables_translation():
    def behaviour(n, items):
        raise api_error(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT")

    t = make(behaviour)
    assert t.translate(cues(3)) == 3
    assert t.fatal and "API key" in t.fatal
    assert len(t.client.models.calls) == 1
    t.translate(cues(2))
    assert len(t.client.models.calls) == 1  # no more requests once fatal


def test_unsupported_thinking_level_falls_back():
    def behaviour(n, items):
        if n == 1:
            raise api_error(400, "Thinking level MINIMAL is not supported for this model.", "INVALID_ARGUMENT")

    t = make(behaviour, gemini_thinking="minimal")
    assert t.translate(cues(1)) == 0
    assert t.client.models.calls[1]["config"].thinking_config is None


def test_same_language_is_not_sent():
    t = make(target_language="en")
    batch = cues(2)
    t.translate(batch, source_language="en")
    assert [c.translation for c in batch] == ["line 0", "line 1"]
    assert t.client.models.calls == []
