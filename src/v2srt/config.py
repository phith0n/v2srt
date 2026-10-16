"""Settings.

Precedence, highest first:

1. command line options
2. environment variables
3. ONE configuration file: ``--env-file``, otherwise ``./.env``, otherwise the per-user ``.env``
4. built-in defaults
"""

import os
import shlex
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from dotenv import dotenv_values

from .errors import ConfigError
from .timeutil import parse_duration

APP_NAME = "v2srt"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
LOG_OFF = "off"

LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "zh-cn": "Simplified Chinese",
    "zh-hans": "Simplified Chinese",
    "zh-tw": "Traditional Chinese",
    "zh-hk": "Traditional Chinese (Hong Kong)",
    "zh-hant": "Traditional Chinese",
    "yue": "Cantonese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "pt-br": "Brazilian Portuguese",
    "ru": "Russian",
    "uk": "Ukrainian",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "tr": "Turkish",
    "ar": "Arabic",
    "he": "Hebrew",
    "fa": "Persian",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
}


def language_name(code: str | None) -> str | None:
    """Human readable language name for a language code; unknown values are returned unchanged."""
    if not code or code.lower() == "auto":
        return None
    return LANGUAGE_NAMES.get(code.lower(), code)


def language_base(code: str | None) -> str | None:
    """``zh-CN`` -> ``zh``; returns None for free-form names such as ``Simplified Chinese``."""
    if not code or code.lower() == "auto":
        return None
    base = code.lower().split("-")[0].split("_")[0]
    return base if len(base) in (2, 3) and base.isalpha() else None


def user_config_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / APP_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / APP_NAME


def find_env_file(explicit: Path | None = None) -> Path | None:
    """The single configuration file to use: ``--env-file``, else ``./.env``, else the per-user file."""
    if explicit is not None:
        if not explicit.is_file():
            raise ConfigError(f"env file not found: {explicit}")
        return explicit.resolve()
    for candidate in (Path.cwd() / ".env", user_config_dir() / ".env"):
        if candidate.is_file():
            return candidate.resolve()
    return None


def read_env_file(path: Path) -> dict[str, str]:
    try:
        values = dotenv_values(path, encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from None
    return {k: v for k, v in values.items() if v is not None}


def _bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on", "y"):
        return True
    if v in ("0", "false", "no", "off", "n", ""):
        return False
    raise ValueError(f"expected true/false, got {value!r}")


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise ValueError(f"must be at least 1, got {n}")
    return n


def _path(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value.strip().strip('"')))


def _optional_path(value: str) -> str | None:
    return _path(value) or None


def _text(value: str) -> str | None:
    return value.strip() or None


def _threshold(value: str) -> float:
    v = float(value)
    if not 0 < v < 1:
        raise ValueError("must be between 0 and 1")
    return v


def _thinking(value: str) -> str:
    v = value.strip().lower() or "default"
    if v not in ("default", "minimal", "low", "medium", "high"):
        raise ValueError("expected one of default, minimal, low, medium, high")
    return v


def split_args(value: str) -> list[str]:
    if sys.platform != "win32":
        return shlex.split(value)
    # Non-POSIX mode keeps backslashes in Windows paths intact but also keeps the quotes.
    return [a[1:-1] if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'" else a for a in shlex.split(value, posix=False)]


def _log_dir(value: str) -> str:
    v = value.strip()
    if v.lower() in ("off", "none", "false", "no", "0"):
        return LOG_OFF
    return _path(v)


def _cuts(value: str) -> list[int]:
    return sorted(parse_duration(p) for p in value.replace(",", " ").split() if p)


@dataclass
class Settings:
    # tools
    ffmpeg: str = "ffmpeg"
    ffprobe: str | None = None
    whisper_cli: str = "whisper-cli"
    whisper_server: str | None = None  # None: next to whisper-cli, then PATH
    # speech recognition
    whisper_model: str | None = None
    vad_model: str | None = None
    vad_threshold: float = 0.3
    use_vad: bool = True
    language: str = "auto"
    whisper_threads: int | None = None
    whisper_jobs: int = 1
    whisper_prompt: str | None = None
    whisper_args: list[str] = field(default_factory=list)
    segment_ms: int = 20 * 60_000
    gap_fill: bool = True
    cut_ms: list[int] = field(default_factory=list)
    audio_track: int | None = None
    # translation
    translate: bool = True
    gemini_api_key: str | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_thinking: str = "low"
    gemini_base_url: str | None = None
    proxy: str | None = None
    target_language: str = "zh-CN"
    translate_workers: int = 4
    batch_size: int = 40
    context: str | None = None
    # output
    bilingual: bool = False
    keep_source: bool = False
    overwrite: bool = False
    skip_existing: bool = False
    keep_temp: bool = False
    # logging
    log_dir: str | None = None  # None: platform default, LOG_OFF: disabled
    log_gemini: bool = False

    @property
    def translation_enabled(self) -> bool:
        return self.translate and bool(self.gemini_api_key)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Settings":
        known = {f.name for f in fields(self)}
        return replace(self, **{k: v for k, v in overrides.items() if k in known and v is not None})


@dataclass(frozen=True)
class EnvVar:
    field: str
    names: tuple[str, ...]
    parse: Callable[[str], Any]
    secret: bool = False


ENV_VARS: tuple[EnvVar, ...] = (
    EnvVar("ffmpeg", ("V2SRT_FFMPEG",), _path),
    EnvVar("ffprobe", ("V2SRT_FFPROBE",), _optional_path),
    EnvVar("whisper_cli", ("V2SRT_WHISPER_CLI",), _path),
    EnvVar("whisper_server", ("V2SRT_WHISPER_SERVER",), _optional_path),
    EnvVar("whisper_model", ("V2SRT_WHISPER_MODEL",), _optional_path),
    EnvVar("vad_model", ("V2SRT_VAD_MODEL",), _optional_path),
    EnvVar("vad_threshold", ("V2SRT_VAD_THRESHOLD",), _threshold),
    EnvVar("language", ("V2SRT_LANGUAGE",), lambda v: v.strip() or "auto"),
    EnvVar("whisper_threads", ("V2SRT_WHISPER_THREADS",), _positive_int),
    EnvVar("whisper_jobs", ("V2SRT_WHISPER_JOBS",), _positive_int),
    EnvVar("whisper_prompt", ("V2SRT_WHISPER_PROMPT",), _text),
    EnvVar("whisper_args", ("V2SRT_WHISPER_ARGS",), split_args),
    EnvVar("segment_ms", ("V2SRT_SEGMENT",), parse_duration),
    EnvVar("gap_fill", ("V2SRT_GAP_FILL",), _bool),
    EnvVar("cut_ms", ("V2SRT_CUT",), _cuts),
    EnvVar("translate", ("V2SRT_TRANSLATE",), _bool),
    EnvVar("gemini_api_key", ("V2SRT_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"), _text, secret=True),
    EnvVar("gemini_model", ("V2SRT_GEMINI_MODEL",), lambda v: v.strip() or DEFAULT_GEMINI_MODEL),
    EnvVar("gemini_thinking", ("V2SRT_GEMINI_THINKING",), _thinking),
    EnvVar("gemini_base_url", ("V2SRT_GEMINI_BASE_URL",), _text),
    EnvVar("proxy", ("V2SRT_PROXY",), _text),
    EnvVar("target_language", ("V2SRT_TARGET_LANGUAGE",), lambda v: v.strip() or "zh-CN"),
    EnvVar("translate_workers", ("V2SRT_TRANSLATE_WORKERS",), _positive_int),
    EnvVar("batch_size", ("V2SRT_BATCH_SIZE",), _positive_int),
    EnvVar("context", ("V2SRT_CONTEXT",), _text),
    EnvVar("bilingual", ("V2SRT_BILINGUAL",), _bool),
    EnvVar("keep_source", ("V2SRT_KEEP_SOURCE",), _bool),
    EnvVar("log_dir", ("V2SRT_LOG_DIR",), _log_dir),
    EnvVar("log_gemini", ("V2SRT_LOG_GEMINI",), _bool),
)


def settings_from_env(environ: Mapping[str, str] | None = None) -> tuple[Settings, dict[str, str]]:
    """Build settings from environment variables. Returns the settings and ``{field: env var name}``."""
    environ = os.environ if environ is None else environ
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for var in ENV_VARS:
        for name in var.names:
            raw = environ.get(name)
            if raw is None or raw.strip() == "":
                continue
            try:
                values[var.field] = var.parse(raw)
            except ValueError as e:
                raise ConfigError(f"invalid value for {name}: {e}") from None
            sources[var.field] = name
            break
    return Settings(**values), sources


SOURCE_DEFAULT = "default"
SOURCE_CLI = "command line"


@dataclass
class LoadedConfig:
    settings: Settings
    env_file: Path | None
    sources: dict[str, str]  # field -> human readable origin of its value


def load_config(
    env_file: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> LoadedConfig:
    """Resolve settings from command line overrides, the environment and one configuration file.

    The configuration file is read without touching ``os.environ``. Environment variables that are set but empty
    count as unset, so ``export V2SRT_X=`` does not hide the value from the file.
    """
    environ = os.environ if environ is None else environ
    path = find_env_file(env_file)
    file_values = read_env_file(path) if path else {}
    real_env = {k: v for k, v in environ.items() if v.strip()}
    settings, names = settings_from_env({**file_values, **real_env})

    sources = {}
    for field_name, var in names.items():
        origin = "environment" if var in real_env else "config file"
        sources[field_name] = f"{origin} ({var})"
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    settings = settings.with_overrides(overrides)
    for key in overrides:
        if key in sources or key in {f.name for f in fields(settings)}:
            sources[key] = SOURCE_CLI
    return LoadedConfig(settings, path, sources)


ENV_TEMPLATE = """\
# v2srt configuration.
# Precedence: command line > environment variables > this file > built-in defaults.
# Only ONE file is used: --env-file if given, otherwise ./.env, otherwise the per-user
# file created by `v2srt init --global`.
# Paths may use ~ and environment variables. Prefer absolute paths.

# ---------------------------------------------------------------- tools
# ffmpeg / whisper-cli: an executable path, the directory that contains it,
# or a command name found on PATH.
V2SRT_FFMPEG=ffmpeg
# V2SRT_FFPROBE=            # default: ffprobe next to ffmpeg, then PATH
V2SRT_WHISPER_CLI=whisper-cli
# whisper-server keeps the model loaded for gap filling (see V2SRT_GAP_FILL).
# V2SRT_WHISPER_SERVER=     # default: whisper-server next to whisper-cli, then PATH

# ------------------------------------------------------ speech recognition
# ggml model from https://huggingface.co/ggerganov/whisper.cpp (required)
V2SRT_WHISPER_MODEL=
# Silero VAD model from https://huggingface.co/ggml-org/whisper-vad (optional, recommended)
V2SRT_VAD_MODEL=
# V2SRT_VAD_THRESHOLD=0.3
# Spoken language code (en, ja, zh, ...) or auto
# V2SRT_LANGUAGE=auto
# Long audio is split into segments of about this length at silent points; 0 disables.
# V2SRT_SEGMENT=20m
# How many whisper-cli processes run in parallel on different segments.
# V2SRT_WHISPER_JOBS=1
# With VAD, re-transcribe speech regions that got no subtitles (whisper sometimes drops
# whole stretches of sparse speech). Short backchannel from this pass is discarded.
# Needs whisper-server; without it gap filling is skipped with a warning.
# V2SRT_GAP_FILL=true
# V2SRT_WHISPER_THREADS=
# Initial prompt, useful for names and jargon.
# V2SRT_WHISPER_PROMPT=
# Extra arguments passed to whisper-cli (and whisper-server) as-is, e.g. "--max-len 60 --split-on-word"
# V2SRT_WHISPER_ARGS=

# ------------------------------------------------------------ translation
# Translation is enabled when an API key is present. Get one at https://aistudio.google.com/apikey
# (GEMINI_API_KEY and GOOGLE_API_KEY are also recognized)
V2SRT_GEMINI_API_KEY=
# V2SRT_GEMINI_MODEL=gemini-3.8-flash
# Thinking level: default, minimal, low, medium, high
# V2SRT_GEMINI_THINKING=low
# Target language: a code (zh-CN, zh-TW, en, ja, ...) or a name ("Brazilian Portuguese")
# V2SRT_TARGET_LANGUAGE=zh-CN
# V2SRT_TRANSLATE_WORKERS=4
# V2SRT_BATCH_SIZE=40
# Proxy for the Gemini API (HTTPS_PROXY / ALL_PROXY are honoured too)
# V2SRT_PROXY=http://127.0.0.1:7890
# Custom API endpoint (e.g. an API gateway)
# V2SRT_GEMINI_BASE_URL=

# ----------------------------------------------------------------- output
# Translation above the original text in every cue
# V2SRT_BILINGUAL=false
# Also write the untranslated subtitles to <name>.<language>.srt
# V2SRT_KEEP_SOURCE=false

# ---------------------------------------------------------------- logging
# Every run writes a detailed log; the latest 50 are kept. `v2srt check` shows the
# default location. Set to "off" to disable.
# V2SRT_LOG_DIR=
# Also save every Gemini prompt and answer next to the log (*.gemini.jsonl).
# V2SRT_LOG_GEMINI=false
"""
