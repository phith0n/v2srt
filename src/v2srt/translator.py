"""Subtitle translation with Gemini.

Robustness measures:
- structured JSON output validated against a schema instead of parsing free text
- retries with exponential backoff (honouring the server's ``retryDelay`` on rate limits)
- cues missing from an answer are requested again; a batch that keeps failing is split in halves to isolate
  the problematic cue, and only that cue falls back to the original text
- unrecoverable errors (bad key, unknown model, unsupported region) stop all translation at once instead of
  hammering the API, and the subtitles are still produced untranslated
"""

import logging
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from . import prompts
from .config import Settings, language_base, language_name
from .errors import TranslationError
from .subtitle import Cue

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 4
SPLIT_ATTEMPTS = 2
REQUEST_TIMEOUT_MS = 180_000

_SAFETY_CATEGORIES = (
    types.HarmCategory.HARM_CATEGORY_HARASSMENT,
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
)


class TranslatedCue(BaseModel):
    id: int
    text: str


class TranslationBatch(BaseModel):
    translations: list[TranslatedCue]


class _Retry(Exception):
    def __init__(self, message: str, delay: float | None = None, rate_limited: bool = False):
        super().__init__(message)
        self.delay = delay
        self.rate_limited = rate_limited


class _Fatal(Exception):
    pass


@dataclass
class TranslationStats:
    requests: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    failed_cues: int = 0


def make_client(settings: Settings) -> genai.Client:
    http_options = types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
    if settings.gemini_base_url:
        http_options.base_url = settings.gemini_base_url
    if settings.proxy:
        http_options.client_args = {"proxy": settings.proxy}
        http_options.async_client_args = {"proxy": settings.proxy}
    return genai.Client(api_key=settings.gemini_api_key, http_options=http_options)


def _retry_delay(error: genai_errors.APIError) -> float | None:
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", str(error.details))
    return float(m.group(1)) if m else None


def _response_text(response: Any) -> str | None:
    if response is None:
        return None
    try:
        return response.text
    except Exception:  # noqa: BLE001 - only used for logging
        return None


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text


class GeminiTranslator:
    def __init__(
        self,
        settings: Settings,
        *,
        client: Any = None,
        cancel: threading.Event | None = None,
    ):
        self.model = settings.gemini_model
        self.target_name = language_name(settings.target_language) or settings.target_language
        self.target_code = language_base(settings.target_language)
        self.context = settings.context
        self.thinking = settings.gemini_thinking
        self.client = client if client is not None else make_client(settings)
        self.cancel = cancel or threading.Event()
        self.stats = TranslationStats()
        self.fatal: str | None = None
        self.backoff_scale = 1.0
        self.recorder: Any = None  # optional object with record(dict), see logs.GeminiRecorder
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public

    def check(self) -> str:
        """Verify the API key, the model name and connectivity. Returns the model's display name."""
        try:
            model = self.client.models.get(model=self.model)
        except genai_errors.APIError as e:
            err = self._classify(e)
            raise TranslationError(str(err)) from None
        except httpx.HTTPError as e:
            raise TranslationError(
                f"cannot reach the Gemini API ({e.__class__.__name__}: {e}). "
                "If you need a proxy, set V2SRT_PROXY or HTTPS_PROXY."
            ) from None
        return getattr(model, "display_name", None) or self.model

    def should_translate(self, source_language: str | None) -> bool:
        """Skip translation when the speech already is in the target language.

        Chinese is always sent through the model: whisper mixes simplified and traditional characters, and the
        translation pass normalizes the script and fixes recognition errors.
        """
        src = language_base(source_language)
        return not (src and src == self.target_code and src != "zh")

    def translate(
        self,
        cues: Sequence[Cue],
        *,
        previous: Sequence[Cue] = (),
        source_language: str | None = None,
        title: str | None = None,
    ) -> int:
        """Fill ``cue.translation`` in place. Returns the number of cues that could not be translated."""
        if not cues:
            return 0
        if not self.should_translate(source_language):
            for cue in cues:
                cue.translation = cue.text
            return 0
        self._translate_group(list(cues), list(previous), source_language, title, MAX_ATTEMPTS)
        failed = sum(1 for c in cues if c.translation is None)
        if failed:
            with self._lock:
                self.stats.failed_cues += failed
        return failed

    # ----------------------------------------------------------------- private

    def _translate_group(
        self, cues: list[Cue], previous: list[Cue], source_language: str | None, title: str | None, attempts: int
    ) -> None:
        pending = cues
        for attempt in range(attempts):
            if self.cancel.is_set() or self.fatal:
                return
            try:
                result = self._request(pending, previous, source_language, title)
            except _Fatal as e:
                with self._lock:
                    first = self.fatal is None
                    self.fatal = self.fatal or str(e)
                if first:
                    log.error("%s\nTranslation is disabled; remaining subtitles keep the original text.", e)
                return
            except _Retry as e:
                with self._lock:
                    self.stats.retries += 1
                base = 5.0 if e.rate_limited else 2.0
                delay = e.delay if e.delay is not None else min(60.0, base * 2**attempt)
                delay = (delay + random.uniform(0, 1)) * self.backoff_scale
                if attempt + 1 < attempts:
                    log.warning("Gemini request failed (%s); retrying in %.0fs", e, delay)
                    if self.cancel.wait(delay):
                        return
                else:
                    log.warning("Gemini request failed (%s)", e)
                continue

            for i, cue in enumerate(pending, 1):
                if i in result:
                    cue.translation = result[i]
            missing = [c for c in pending if c.translation is None]
            if not missing:
                return
            log.debug("%d of %d cues missing from the answer, asking again", len(missing), len(pending))
            pending = missing

        if len(pending) > 1 and not self.fatal and not self.cancel.is_set():
            mid = len(pending) // 2
            log.debug("splitting a failing batch of %d cues", len(pending))
            self._translate_group(pending[:mid], previous, source_language, title, SPLIT_ATTEMPTS)
            self._translate_group(pending[mid:], previous + pending[:mid], source_language, title, SPLIT_ATTEMPTS)

    def _config(self, source_language: str | None) -> types.GenerateContentConfig:
        thinking = None
        if self.thinking != "default":
            thinking = types.ThinkingConfig(thinking_level=self.thinking.upper())
        return types.GenerateContentConfig(
            system_instruction=prompts.system_instruction(
                language_name(source_language), self.target_name, self.target_code
            ),
            response_mime_type="application/json",
            response_schema=TranslationBatch,
            thinking_config=thinking,
            # Films and shows contain violence, insults and swearing; a blocked answer means missing subtitles.
            safety_settings=[
                types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE)
                for c in _SAFETY_CATEGORIES
            ],
        )

    def _request(
        self, cues: list[Cue], previous: list[Cue], source_language: str | None, title: str | None
    ) -> dict[int, str]:
        prompt = prompts.user_prompt(cues, previous=previous, title=title, context=self.context)
        config = self._config(source_language)
        started = time.monotonic()
        response: Any = None
        outcome = "ok"
        try:
            response = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
            self._record_usage(response)
            return self._parse(response, len(cues))
        except genai_errors.APIError as e:
            outcome = f"error: {e}"
            raise self._classify(e) from None
        except httpx.HTTPError as e:
            outcome = f"error: {e.__class__.__name__}: {e}"
            raise _Retry(f"network error: {e.__class__.__name__}: {e}") from None
        except _Retry as e:
            outcome = f"error: {e}"
            raise
        finally:
            elapsed = time.monotonic() - started
            with self._lock:
                self.stats.requests += 1
            usage = getattr(response, "usage_metadata", None)
            tokens = {
                "input": getattr(usage, "prompt_token_count", None),
                "output": getattr(usage, "candidates_token_count", None),
                "thinking": getattr(usage, "thoughts_token_count", None),
            }
            log.debug(
                "gemini: %d cues, %.1fs, tokens in=%s out=%s thinking=%s, %s",
                len(cues), elapsed, tokens["input"], tokens["output"], tokens["thinking"], outcome,
            )
            if self.recorder is not None:
                self.recorder.record(
                    {
                        "model": self.model,
                        "thinking_level": self.thinking,
                        "source_language": source_language,
                        "cues": len(cues),
                        "elapsed": round(elapsed, 3),
                        "outcome": outcome,
                        "tokens": tokens,
                        "system_instruction": config.system_instruction,
                        "prompt": prompt,
                        "answer": _response_text(response),
                    }
                )

    def _parse(self, response: Any, count: int) -> dict[int, str]:
        parsed = getattr(response, "parsed", None)
        if not isinstance(parsed, TranslationBatch):
            text = response.text
            if not text:
                raise _Retry(f"empty answer ({self._finish_reason(response)})")
            try:
                parsed = TranslationBatch.model_validate_json(_strip_fence(text))
            except ValidationError:
                raise _Retry("answer is not valid JSON") from None

        result: dict[int, str] = {}
        for item in parsed.translations:
            text = "\n".join(line.strip() for line in item.text.strip().splitlines() if line.strip())
            if 1 <= item.id <= count and text:
                result[item.id] = text
        return result

    def _classify(self, e: genai_errors.APIError) -> Exception:
        code = e.code or 0
        message = (e.message or str(e)).strip()
        lowered = f"{message} {e.details}".lower()
        if code == 400 and "thinking" in lowered and self.thinking != "default":
            log.warning(
                "model %s does not accept thinking level %r; using the model default "
                "(set V2SRT_GEMINI_THINKING to silence this)", self.model, self.thinking,
            )
            self.thinking = "default"
            return _Retry("unsupported thinking level", delay=0)
        if code in (401, 403) or "api_key_invalid" in lowered or "api key not valid" in lowered:
            return _Fatal(f"Gemini rejected the API key ({code}): {message}")
        if code == 404:
            return _Fatal(f"Gemini model {self.model!r} was not found ({message}). Check V2SRT_GEMINI_MODEL.")
        if "location is not supported" in lowered:
            return _Fatal(f"the Gemini API is not available in your region ({message}). Set V2SRT_PROXY.")
        if code == 429:
            if "perday" in lowered:
                return _Fatal(f"the daily Gemini quota is exhausted ({message})")
            return _Retry("rate limited", delay=_retry_delay(e), rate_limited=True)
        return _Retry(f"{code} {e.status or ''} {message}".strip())

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        with self._lock:
            self.stats.prompt_tokens += usage.prompt_token_count or 0
            self.stats.output_tokens += usage.candidates_token_count or 0
            self.stats.thinking_tokens += usage.thoughts_token_count or 0

    @staticmethod
    def _finish_reason(response: Any) -> str:
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            return f"blocked: {feedback.block_reason}"
        candidates = getattr(response, "candidates", None) or []
        if candidates and getattr(candidates[0], "finish_reason", None):
            return f"finish reason: {candidates[0].finish_reason}"
        return "no candidates"
