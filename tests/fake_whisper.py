"""Stand-in for whisper-cli: prints one segment per 2 seconds of audio, like the real program does.

With ``--vad`` it also reports one VAD speech segment per printed segment on stderr, in whisper.cpp's format.
Gap filling goes to whisper-server instead, see ``fake_whisper_server.py``.

Environment knobs:
- FAKE_WHISPER_FAIL: exit with an error
- FAKE_WHISPER_DELAY: seconds to sleep after every printed segment
- FAKE_WHISPER_DROP=N: with --vad, do not print every N-th segment although VAD reports speech there
"""

import os
import sys
import time
import wave


def stamp(ms: int) -> str:
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def transcribe(wav: str, args: list[str]) -> None:
    with wave.open(wav) as w:
        duration = w.getnframes() * 1000 // w.getframerate()
    print(f"main: processing '{wav}' ({duration * 16} samples, {duration / 1000:.1f} sec)", file=sys.stderr)
    print("whisper_full_with_state: auto-detected language: en (p = 0.97)", file=sys.stderr, flush=True)
    vad = "--vad" in args
    starts = range(0, max(1, duration - 500), 2000)
    if vad:
        print(f"whisper_vad_segments_from_probs: Final speech segments after filtering: {len(starts)}", file=sys.stderr)
        for i, start in enumerate(starts):
            # speech in the middle of each 2 s slot, far enough from neighbouring cues to count as uncovered
            a, b = (start + 700) / 1000, min(start + 1300, duration) / 1000
            print(
                f"whisper_vad_segments_from_probs: VAD segment {i}: start = {a:.2f}, end = {b:.2f} "
                f"(duration: {b - a:.2f})",
                file=sys.stderr,
            )
        print(f"whisper_vad: Including segment 0: {starts[0] / 1000:.2f} - 1.90 (duration: 1.90)", file=sys.stderr)

    drop = int(os.environ.get("FAKE_WHISPER_DROP", "0"))
    cues = []
    for n, start in enumerate(starts, 1):
        end = min(start + 2000, duration)
        if vad and drop and n % drop == 0:
            continue
        cues.append((start, end, "[BLANK_AUDIO]" if n == 2 else f"Line {n} at {start}"))

    out = sys.stdout.buffer
    for start, end, text in cues:
        out.write(f"[{stamp(start)} --> {stamp(end)}]   {text}\n".encode())
        out.flush()
        print(f"whisper_print_progress_callback: progress = {start * 100 // max(1, duration):3d}%", file=sys.stderr)
        time.sleep(float(os.environ.get("FAKE_WHISPER_DELAY", "0.005")))


def main() -> int:
    args = sys.argv[1:]
    assert "-m" in args and os.path.isfile(args[args.index("-m") + 1])
    if os.environ.get("FAKE_WHISPER_FAIL"):
        print("error: failed to load model", file=sys.stderr)
        return 3
    print("whisper_init_from_file_with_params_no_state: loading model", file=sys.stderr)
    for i, arg in enumerate(args):
        if arg == "-f":
            transcribe(args[i + 1], args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
