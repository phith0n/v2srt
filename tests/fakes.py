"""Test doubles for the Gemini client."""

import json
import re
import threading
from types import SimpleNamespace

from google.genai import errors as genai_errors

from v2srt.translator import TranslatedCue, TranslationBatch


def cues_in_prompt(prompt: str) -> list[dict]:
    """The JSON array of cues at the end of a user prompt."""
    return json.loads(prompt[prompt.rindex("\n[") + 1 :])


def api_error(code: int, message: str, status: str = "ERROR") -> genai_errors.APIError:
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"code": code, "message": message, "status": status}})


class FakeModels:
    def __init__(self, behaviour=None):
        # behaviour(call_number, cues) may raise, or return a list of (id, text) to answer with
        self.behaviour = behaviour
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def get(self, model):
        return SimpleNamespace(display_name=f"Fake {model}")

    def generate_content(self, model, contents, config):
        cues = cues_in_prompt(contents)
        with self.lock:
            self.calls.append({"model": model, "prompt": contents, "config": config, "cues": cues})
            number = len(self.calls)
        answer = None
        if self.behaviour is not None:
            answer = self.behaviour(number, cues)
        if answer is None:
            answer = [(c["id"], f"T:{c['text']}") for c in cues]
        batch = TranslationBatch(translations=[TranslatedCue(id=i, text=t) for i, t in answer])
        usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=1)
        return SimpleNamespace(parsed=batch, text=batch.model_dump_json(), usage_metadata=usage)


class FakeClient:
    def __init__(self, behaviour=None):
        self.models = FakeModels(behaviour)


def srt_texts(path) -> list[str]:
    return re.findall(r"\d+\n[\d:,]+ --> [\d:,]+\n(.+?)\n\n", path.read_text(encoding="utf-8"), re.S)
