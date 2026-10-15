class V2SRTError(Exception):
    """Base class for errors that should be shown to the user without a traceback."""


class ConfigError(V2SRTError):
    """Missing or invalid configuration."""


class ToolError(V2SRTError):
    """An external tool (ffmpeg, whisper-cli) is missing or failed."""


class TranslationError(V2SRTError):
    """Translation can not continue (bad API key, unknown model, ...)."""


class Cancelled(V2SRTError):
    """The user interrupted the run."""
