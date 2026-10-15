# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

v2srt is a CLI that transcribes video/audio locally with whisper.cpp's `whisper-cli` and translates the subtitles with Gemini (`google-genai`). It is used on Windows and should keep working on macOS/Linux. The README is in Chinese; code, comments, prompts and CLI messages are in English.

## Commands

```bash
uv sync                                    # install dependencies (incl. dev group)
uv run pytest                              # all tests
uv run pytest tests/test_units.py::test_parse_duration   # single test
uv run pytest -k translator                # by keyword
uv run v2srt --help                        # run the CLI from the checkout
uv run v2srt check                         # verify tools/models/Gemini key using the active .env
uv build                                   # wheel + sdist (uv_build backend)
```

- Pipeline tests (`tests/test_pipeline.py`) need `ffmpeg` on PATH and are skipped otherwise. No test needs a real whisper-cli or Gemini key.
- `pyproject.toml` sets ruff's line length to 120, but ruff is not a dependency and the code is not currently ruff-clean (`uvx ruff check src tests`).
- A local, git-ignored `.env` in the repo root holds the developer's real tool paths and API key.

## Architecture

Entry point `v2srt:main` → `cli.main`, which dispatches `run` (default), `init`, `check`, `clean`.

### Processing pipeline (`pipeline.py`)

```
ffmpeg → audio.wav → split at silences (media.plan_boundaries/split_wav) → one chunk per segment
  Whisper.recognize per chunk (DaemonExecutor, settings.whisper_jobs): main whisper-cli pass with --vad,
    stdout parsed line by line → CueCleaner → gap filling (whisper.py) → the chunk's cues, sorted
    → batches of batch_size, submitted once the chunk is complete
      → GeminiTranslator.translate (DaemonExecutor, settings.translate_workers)
        → OrderedSink → CueSequencer → Outputs (SrtWriter) → *.srt.part, renamed on success
```

Key invariants that span several files:

- **Streaming**: whisper-cli prints each segment to stdout as `[hh:mm:ss.mmm --> hh:mm:ss.mmm]  text`; `Whisper.transcribe` yields cues while the process runs. Language detection, progress and the VAD speech segments (`ChunkInfo`) come from **stderr**, so never add `--no-prints`. Chunk-relative timestamps are offset by `chunk.start`. Translation overlaps recognition **per chunk**: `recognize` returns a chunk's complete cue list, so the pipeline never has to retract cues it already submitted.
- **Gap filling** (`Whisper.recognize`, `V2SRT_GAP_FILL`): with `--vad`, whisper.cpp concatenates a chunk's speech and decodes it as one stream; on sparse speech it sometimes returns almost nothing for a whole chunk. After the main pass, VAD segments not touched by any kept cue (±`COVER_PAD_MS`, using capped ends) are grouped by `plan_pieces`, cut with `wave`, transcribed **without** VAD in one whisper-cli call with several `-f` and `-osrt` (model loaded once; results read from `<piece>.wav.srt`), cleaned with `CueCleaner(strict=True)` (drops backchannel, syllable loops, stock phrases) and merged. Changing this behaviour should be validated on real files with the uncovered-speech metric (VAD speech time without a cue).
- **Ordering**: batches finish out of order. `OrderedSink` is keyed by `(chunk, batch)` and only advances to the next chunk after `finish_chunk(chunk, n_batches)`; every submitted batch must eventually be `put`, even on failure (see the `finally` in `translate_batch`), or the sink stalls.
- **CueSequencer** holds back one cue to trim overlaps, so `flush()` must run before committing.
- **Outputs** write to `<name>.srt.part` and `os.replace` on success; on failure/Ctrl+C the `.part` file is kept and reported.
- **Thread pools** are `proc.DaemonExecutor` (daemon threads) rather than `ThreadPoolExecutor`, so Ctrl+C exits immediately even with HTTP requests in flight. `_wait_all` polls with a timeout to stay interruptible on Windows. Subprocesses go through `ProcessGroup` so they can all be killed on cancel.
- `.srt` inputs skip ffmpeg/whisper and go through the same sink as a single pseudo-chunk (`run_srt`).

### Translation (`translator.py`, `prompts.py`)

- Prompt and system instruction are English and language-agnostic; target-specific style notes live in `prompts._STYLE_NOTES`. Cues are sent as JSON with **batch-local ids 1..N**; preceding cues (`CONTEXT_CUES`) are context only.
- Structured output via `response_schema=TranslationBatch` (pydantic). Missing ids are re-requested; a batch that keeps failing is split in halves recursively, so only the bad cue falls back to source text.
- `_classify` maps API errors: auth/404/unsupported region/daily quota → `_Fatal` (sets `translator.fatal`, disables translation for the rest of the run); 429/5xx/network/bad JSON → `_Retry` with backoff. A 400 about thinking level drops the thinking config and retries.
- Default model `gemini-3.8-flash`, thinking level `low`. At `low` this model usually does no thinking, so `thoughts_token_count` is `None` and the reported thinking tokens are 0 — expected, not a bug.
- `should_translate` skips only when source and target base language match, except Chinese, which is always sent (normalizes simplified/traditional and fixes ASR errors).
- Tests set `translator.backoff_scale` to avoid sleeping; `translator.recorder` (a `logs.GeminiRecorder`) records full prompts/answers when `V2SRT_LOG_GEMINI=true`.

### Configuration (`config.py`)

- Precedence: CLI options > environment variables > **one** config file > defaults. The file is the first found of `--env-file`, `./.env`, `user_config_dir()/.env`; files are never merged. The file is read with `dotenv_values` and is **not** injected into `os.environ`. Empty environment variables count as unset.
- `load_config` returns `LoadedConfig` with per-field `sources` (shown by `v2srt check` and written to the run log).
- Adding a setting touches: the `Settings` field, an `EnvVar` in `ENV_VARS`, `ENV_TEMPLATE`, the regenerated `.env.example` (a test asserts it equals `ENV_TEMPLATE`; regenerate with `uv run python -c "from v2srt.config import ENV_TEMPLATE; open('.env.example','w',encoding='utf-8',newline='\n').write(ENV_TEMPLATE)"`), the README table, and optionally a CLI option. CLI options must default to `None` (or use `default=None` with `store_true`/`store_false`) because `None` means "not given" in `with_overrides`.
- `V2SRT_LOG_DIR=off` disables logging (empty values mean "unset" everywhere, so "off" is used instead).

### Logging and cleanup (`logs.py`, `cli.py`)

- Logger `v2srt` is always at DEBUG; the console `RichHandler` level is INFO (DEBUG with `-v`); `RunLog` attaches a DEBUG `FileHandler` to the root logger for `run` only. Default directory comes from `platformdirs.user_log_dir("v2srt", appauthor=False)`.
- Log files are named `YYYYmmdd-HHMMSS_<slug>.log` (+ `.gemini.jsonl`); `LOG_FILE_RE` is what `prune` and `clean` match, so the slug must not contain dots and cleanup only ever touches files matching that pattern.
- `v2srt clean` must never delete the current directory's `.env`, even when cwd is the per-user config directory. Tests for clean/logs must isolate every location: patch both `cli.default_log_dir` and `v2srt.logs.default_log_dir`, set `APPDATA`/`XDG_CONFIG_HOME`, and patch `tempfile.gettempdir` (see the `home` fixture in `tests/test_config_logs_clean.py`).

### Testing doubles

- `tests/fake_whisper.py` imitates whisper-cli's stdout/stderr; tests inject it via `Whisper(..., command_prefix=[sys.executable, fake_whisper.py])`. Env vars `FAKE_WHISPER_FAIL` / `FAKE_WHISPER_DELAY` control it.
- `tests/fakes.py` provides `FakeClient` (Gemini) whose `behaviour(call_number, cues)` callback can raise `api_error(...)` or return partial answers.

## Platform notes

- `cli.main` reconfigures stdout/stderr with `errors="replace"`: a redirected stdout on Chinese Windows is GBK and would crash on characters like `✓`.
- On Windows `NUL` reports `isatty() == True`, so interactive prompts must also handle `EOFError` (see `cmd_clean`).
- Interrupted child processes exit with `0xC000013A` on Windows / `-2` on POSIX; `ProcessGroup` maps these to `Cancelled`.
- whisper-cli receives only ASCII temp paths (`audio.wav`, `chunk-NNN.wav`) to avoid argv encoding issues with non-ASCII video names.
