"""Stand-in for whisper-cli: prints one segment per 2 seconds of audio, like the real program does."""

import os
import sys
import time
import wave


def stamp(ms: int) -> str:
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def main() -> int:
    args = sys.argv[1:]
    wav = args[args.index("-f") + 1]
    assert "-m" in args and os.path.isfile(args[args.index("-m") + 1])
    if os.environ.get("FAKE_WHISPER_FAIL"):
        print("error: failed to load model", file=sys.stderr)
        return 3

    with wave.open(wav) as w:
        duration = w.getnframes() * 1000 // w.getframerate()
    print("whisper_init_from_file_with_params_no_state: loading model", file=sys.stderr)
    print("whisper_full_with_state: auto-detected language: en (p = 0.97)", file=sys.stderr, flush=True)

    out = sys.stdout.buffer
    for n, start in enumerate(range(0, duration - 500, 2000), 1):
        end = min(start + 2000, duration)
        text = "[BLANK_AUDIO]" if n == 2 else f"Line {n} at {start}"
        out.write(f"[{stamp(start)} --> {stamp(end)}]   {text}\n".encode())
        out.flush()
        print(f"whisper_print_progress_callback: progress = {start * 100 // duration:3d}%", file=sys.stderr)
        time.sleep(float(os.environ.get("FAKE_WHISPER_DELAY", "0.005")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
