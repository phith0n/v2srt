"""Stand-in for whisper.cpp's whisper-vad-speech-segments.

"Speech" is every 10 ms frame whose peak amplitude exceeds ``--vad-threshold`` (as a fraction of full scale).
Output follows examples/vad-speech-segments/speech.cpp: log lines on stderr, then on stdout an empty line,
``Detected N speech segments:`` and one ``Speech segment i: start = <cs>, end = <cs>`` line per segment, with the
times in centiseconds.

Environment knobs:
- FAKE_VAD_FAIL: fail like a missing model (exit code 2)
- FAKE_VAD_USAGE: print the usage and exit with 0, as the real tool does for unknown arguments
- FAKE_VAD_REPLAY=path: print that file to stdout instead (a captured output of the real tool)
"""

import os
import sys
import wave
from array import array


def usage() -> None:
    print("\nusage: whisper-vad-speech-segments [options] file\n", file=sys.stderr)


def speech_segments(path: str, threshold: float) -> list[tuple[int, int]]:
    with wave.open(path) as w:
        rate = w.getframerate()
        samples = array("h", w.readframes(w.getnframes()))
    frame = rate // 100  # 10 ms = 1 cs
    limit = threshold * 32767
    segments: list[tuple[int, int]] = []
    start = None
    n_frames = (len(samples) + frame - 1) // frame
    for i in range(n_frames):
        loud = max((abs(x) for x in samples[i * frame : (i + 1) * frame]), default=0) > limit
        if loud and start is None:
            start = i
        elif not loud and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, n_frames))
    return segments


def main() -> int:
    names = {"-f": "file", "--file": "file", "-vm": "model", "--vad-model": "model", "-vt": "threshold",
             "--vad-threshold": "threshold"}  # fmt: skip
    args = sys.argv[1:]
    options = {"threshold": "0.5"}
    i = 0
    while i < len(args):
        if args[i] not in names or i + 1 >= len(args):
            print(f"error: unknown argument: {args[i]}", file=sys.stderr)
            usage()
            return 0
        options[names[args[i]]] = args[i + 1]
        i += 2
    if os.environ.get("FAKE_VAD_USAGE"):
        usage()
        return 0
    print("whisper_vad_init_from_file_with_params: loading VAD model", file=sys.stderr)
    if os.environ.get("FAKE_VAD_FAIL") or not os.path.isfile(options.get("model", "")):
        print("error: failed to initialize whisper context", file=sys.stderr)
        return 2
    if replay := os.environ.get("FAKE_VAD_REPLAY"):
        with open(replay, "rb") as fh:
            sys.stdout.buffer.write(fh.read())
        return 0
    segments = speech_segments(options["file"], float(options["threshold"]))
    print(f"whisper_vad_segments_from_probs: Final speech segments after filtering: {len(segments)}", file=sys.stderr)
    print()
    print(f"Detected {len(segments)} speech segments:")
    for n, (start, end) in enumerate(segments):
        print(f"Speech segment {n}: start = {start:.2f}, end = {end:.2f}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
