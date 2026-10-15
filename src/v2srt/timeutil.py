"""Parsing and formatting of durations and timestamps. All times are integer milliseconds."""

import re

_CLOCK_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:[.,]\d+)?)$")
_UNITS_RE = re.compile(r"^(?:\d+(?:\.\d+)?(?:ms|h|m|s))+$")
_UNIT_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_MS = {"h": 3_600_000, "m": 60_000, "s": 1000, "ms": 1}


def parse_duration(text: str) -> int:
    """Parse a human friendly duration into milliseconds.

    Accepted forms: ``90`` / ``90.5`` (seconds), ``01:30:00``, ``1:30``, ``00:01:02,500``,
    ``1h30m``, ``20m``, ``45s``, ``500ms``.
    """
    s = text.strip().lower().replace(" ", "")
    if not s:
        raise ValueError("empty duration")
    try:
        value = float(s)
    except ValueError:
        pass
    else:
        if value < 0:
            raise ValueError(f"negative duration: {text!r}")
        return round(value * 1000)

    m = _CLOCK_RE.match(s)
    if m:
        hours, minutes, seconds = m.groups()
        return round(
            (int(hours or 0) * 3600 + int(minutes) * 60 + float(seconds.replace(",", "."))) * 1000
        )

    if _UNITS_RE.match(s):
        return round(sum(float(n) * _UNIT_MS[u] for n, u in _UNIT_PART_RE.findall(s)))

    raise ValueError(f"invalid duration: {text!r} (examples: 90, 20m, 1h30m, 01:30:00)")


def format_srt_time(ms: int) -> str:
    ms = max(0, int(ms))
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def format_clock(ms: int) -> str:
    """Short display form such as ``1:02:03`` or ``02:03``."""
    total = max(0, int(ms)) // 1000
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02}:{seconds:02}"
    return f"{minutes:02}:{seconds:02}"
