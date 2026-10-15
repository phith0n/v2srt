"""v2srt - generate (and translate) subtitles for any video with whisper.cpp and Gemini."""

__version__ = "2.0.0"


def main() -> None:
    from .cli import main as _main

    _main()
