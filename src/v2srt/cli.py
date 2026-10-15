"""Command line interface."""

import argparse
import glob
import logging
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.prompt import Confirm
from rich.table import Table

from . import __version__
from .config import ENV_TEMPLATE, LOG_OFF, LoadedConfig, Settings, language_name, load_config, user_config_dir
from .errors import Cancelled, ConfigError, V2SRTError
from .logs import RunLog, default_log_dir, log_files, resolve_log_dir
from .media import Media, resolve_executable, resolve_ffprobe
from .pipeline import JobResult, Pipeline, target_suffix
from .proc import ProcessGroup
from .timeutil import format_clock, parse_duration
from .translator import GeminiTranslator
from .ui import Reporter
from .whisper import Whisper

log = logging.getLogger("v2srt")
console = Console(highlight=False)

COMMANDS = ("run", "init", "check", "clean")
SUBTITLE_SUFFIXES = {".srt"}

_run_log: RunLog | None = None


# --------------------------------------------------------------------------- arguments


def _duration(value: str) -> int:
    try:
        return parse_duration(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return n


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def _env_file_option(parser: Any) -> None:
    parser.add_argument(
        "--env-file", type=Path, metavar="PATH", help="use this configuration file instead of ./.env or the per-user one"
    )


def _common(parser: Any) -> None:
    _env_file_option(parser)
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug output (including whisper-cli logs)")


def build_run_parser(prog: str = "v2srt") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Generate subtitles for videos with whisper.cpp and translate them with Gemini.",
        epilog=(
            "other commands:\n"
            "  v2srt init      create a .env file with all settings\n"
            "  v2srt check     verify tools, models and the Gemini API key\n"
            "  v2srt clean     remove the per-user configuration, logs and temporary files\n\n"
            "examples:\n"
            "  v2srt movie.mp4\n"
            "  v2srt -l ja --bilingual *.mkv\n"
            "  v2srt --to en subtitles.srt          (translate an existing subtitle file)\n\n"
            "Most options can also be set in a .env file, see `v2srt init`."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", metavar="INPUT", help="video/audio files or .srt files (wildcards allowed)")
    p.add_argument("-o", "--output", type=Path, metavar="PATH", help="output file, or directory for several inputs")
    p.add_argument("-f", "--overwrite", action="store_true", default=None, help="overwrite existing subtitle files")
    p.add_argument("--skip-existing", action="store_true", default=None, help="skip inputs whose output exists")
    p.add_argument("-V", "--version", action="version", version=f"v2srt {__version__}")

    g = p.add_argument_group("speech recognition")
    g.add_argument("-l", "--language", help="spoken language (en, ja, zh, ...), default: auto-detect")
    g.add_argument("-m", "--model", dest="whisper_model", metavar="PATH", help="whisper.cpp ggml model")
    g.add_argument("--vad-model", metavar="PATH", help="Silero VAD model for whisper.cpp")
    g.add_argument("--no-vad", dest="use_vad", action="store_false", default=None, help="do not use VAD")
    g.add_argument("--prompt", dest="whisper_prompt", metavar="TEXT", help="initial prompt (names, jargon)")
    g.add_argument(
        "--segment", dest="segment_ms", type=_duration, metavar="DURATION",
        help="split long audio into segments of about this length at silences (default 20m, 0 = never)",
    )  # fmt: skip
    g.add_argument(
        "-c", "--cut", dest="cut_ms", type=_duration, nargs="+", metavar="TIME",
        help="split at exactly these times instead, e.g. --cut 1:00:00 2:00:00",
    )  # fmt: skip
    g.add_argument("-j", "--jobs", dest="whisper_jobs", type=_positive_int, metavar="N",
                   help="whisper-cli processes running in parallel (default 1)")  # fmt: skip
    g.add_argument("--threads", dest="whisper_threads", type=_positive_int, metavar="N", help="threads per whisper-cli")
    g.add_argument("--audio-track", type=_non_negative_int, metavar="N", help="audio track to use (0 = first)")

    g = p.add_argument_group("translation")
    g.add_argument("-t", "--to", dest="target_language", metavar="LANG",
                   help="target language code or name (default zh-CN)")  # fmt: skip
    g.add_argument("--no-translate", dest="translate", action="store_false", default=None, help="only transcribe")
    g.add_argument("--bilingual", action="store_true", default=None, help="show translation and original text")
    g.add_argument("--keep-source", action="store_true", default=None,
                   help="also save the untranslated subtitles as NAME.<lang>.srt")  # fmt: skip
    g.add_argument("--context", metavar="TEXT", help="background info for the translator (show name, characters...)")
    g.add_argument("--gemini-model", metavar="NAME", help="Gemini model (default gemini-3.8-flash)")
    g.add_argument("--thinking", dest="gemini_thinking", choices=("default", "minimal", "low", "medium", "high"),
                   help="Gemini thinking level (default low)")  # fmt: skip
    g.add_argument("-w", "--workers", dest="translate_workers", type=_positive_int, metavar="N",
                   help="parallel translation requests (default 4)")  # fmt: skip
    g.add_argument("--batch-size", type=_positive_int, metavar="N", help="cues per translation request (default 40)")

    g = p.add_argument_group("misc")
    g.add_argument("-q", "--quiet", action="store_true", help="do not print subtitles while they are generated")
    g.add_argument("--keep-temp", action="store_true", default=None, help="keep temporary audio files")
    _common(g)
    return p


# --------------------------------------------------------------------------- setup


def setup_logging(verbose: bool) -> None:
    """The console shows INFO (DEBUG with -v); the per-run log file always receives DEBUG."""
    handler = RichHandler(console=console, show_time=False, show_path=False, markup=False, rich_tracebacks=True)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)
    logging.getLogger("v2srt").setLevel(logging.DEBUG)


def mask(secret: str | None) -> str | None:
    if not secret:
        return secret
    return f"{secret[:4]}…{secret[-4:]}" if len(secret) > 12 else "***"


def log_run_header(config: LoadedConfig) -> None:
    log.debug("v2srt %s, Python %s, %s", __version__, platform.python_version(), platform.platform())
    log.debug("command: %s", subprocess.list2cmdline(sys.argv))
    log.debug("working directory: %s", Path.cwd())
    log.debug("config file: %s", config.env_file or "none")
    values = asdict(config.settings)
    values["gemini_api_key"] = mask(values["gemini_api_key"])
    for key, value in values.items():
        log.debug("  %s = %r  [%s]", key, value, config.sources.get(key, "default"))


def expand_inputs(patterns: Sequence[str]) -> list[Path]:
    """Expand wildcards ourselves: the Windows shell does not do it."""
    paths: list[Path] = []
    for pattern in patterns:
        if Path(pattern).exists():
            matches = [pattern]
        elif glob.has_magic(pattern):
            matches = sorted(glob.glob(pattern, recursive=True))
            if not matches:
                raise ConfigError(f"no files match {pattern!r}")
        else:
            raise ConfigError(f"input not found: {pattern}")
        for match in matches:
            path = Path(match)
            if path.is_dir():
                raise ConfigError(f"{match} is a directory; pass files or a wildcard such as {match}/*.mp4")
            if path.resolve() not in {p.resolve() for p in paths}:
                paths.append(path)
    return paths


_LANG_SUFFIX = re.compile(r"^[a-z]{2,3}(-[a-z0-9]{2,4})?$", re.IGNORECASE)


def default_output(src: Path, settings: Settings) -> Path:
    if src.suffix.lower() in SUBTITLE_SUFFIXES:
        stem = src.stem
        base, dot, last = stem.rpartition(".")
        if dot and _LANG_SUFFIX.match(last):  # movie.en.srt -> movie.zh-CN.srt
            stem = base
        out = src.with_name(f"{stem}.{target_suffix(settings)}.srt")
        return out if out.resolve() != src.resolve() else src.with_name(f"{src.stem}.translated.srt")
    return src.with_suffix(".srt")


@dataclass
class Job:
    src: Path
    output: Path
    kind: str  # "media" or "srt"


def plan_jobs(inputs: list[Path], output: Path | None, settings: Settings) -> tuple[list[Job], list[Path]]:
    """Decide the output path of every input. Returns the jobs and the inputs that are skipped."""
    out_dir: Path | None = None
    if output is not None:
        if len(inputs) > 1 or output.is_dir() or str(output).endswith(("/", "\\")):
            out_dir = output
            out_dir.mkdir(parents=True, exist_ok=True)
        elif output.suffix.lower() != ".srt":
            raise ConfigError(f"output file must end with .srt: {output}")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)

    jobs: list[Job] = []
    skipped: list[Path] = []
    conflicts: list[Path] = []
    seen: dict[Path, Path] = {}
    for src in inputs:
        kind = "srt" if src.suffix.lower() in SUBTITLE_SUFFIXES else "media"
        target = default_output(src, settings)
        if out_dir is not None:
            target = out_dir / target.name
        elif output is not None:
            target = output
        key = target.resolve()
        if key in seen:
            raise ConfigError(f"{seen[key].name} and {src.name} would both be written to {target}")
        seen[key] = src
        if target.exists():
            if settings.skip_existing:
                skipped.append(src)
                continue
            if not settings.overwrite:
                conflicts.append(target)
        jobs.append(Job(src, target, kind))
    if conflicts:
        listing = "\n".join(f"  {c}" for c in conflicts)
        raise ConfigError(f"output already exists:\n{listing}\nuse -f/--overwrite or --skip-existing")
    return jobs, skipped


def _check_file(value: str | None, what: str, env_var: str, option: str) -> str:
    if not value:
        raise ConfigError(
            f"no {what} configured. Set {env_var} in your .env file (run `v2srt init`) or pass {option}."
        )
    if not Path(value).is_file():
        raise ConfigError(f"{what} not found: {value}")
    return value


# --------------------------------------------------------------------------- run


def cmd_run(argv: Sequence[str]) -> int:
    global _run_log
    parser = build_run_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    overrides = {k: v for k, v in vars(args).items() if k not in ("inputs", "output", "quiet", "verbose", "env_file")}
    config = load_config(args.env_file, overrides)
    settings = config.settings

    jobs, skipped = plan_jobs(expand_inputs(args.inputs), args.output, settings)
    for src in skipped:
        log.info("skipping %s (subtitles already exist)", src.name)
    if not jobs:
        return 0

    log_dir = resolve_log_dir(settings.log_dir)
    if log_dir is not None:
        _run_log = RunLog.start(log_dir, jobs[0].src.stem, record_gemini=settings.log_gemini)
    log_run_header(config)

    procs = ProcessGroup()
    media = whisper = None
    if any(j.kind == "media" for j in jobs):
        ffmpeg = resolve_executable(settings.ffmpeg, "ffmpeg", "V2SRT_FFMPEG")
        media = Media(ffmpeg, resolve_ffprobe(ffmpeg, settings.ffprobe), procs)
        whisper_cli = resolve_executable(settings.whisper_cli, "whisper-cli", "V2SRT_WHISPER_CLI")
        _check_file(settings.whisper_model, "whisper model", "V2SRT_WHISPER_MODEL", "-m/--model")
        if settings.use_vad and settings.vad_model:
            _check_file(settings.vad_model, "VAD model", "V2SRT_VAD_MODEL", "--vad-model")
        whisper = Whisper(settings, whisper_cli, procs)
        log.debug("ffmpeg: %s | ffprobe: %s | whisper-cli: %s", media.ffmpeg, media.ffprobe, whisper_cli)

    translator = None
    if settings.translation_enabled:
        translator = GeminiTranslator(settings)
        translator.recorder = _run_log.gemini if _run_log else None
        with console.status(f"Checking Gemini model {settings.gemini_model}..."):
            translator.check()
    elif settings.translate and not settings.gemini_api_key:
        if any(j.kind == "srt" for j in jobs):
            raise ConfigError("translating .srt files requires V2SRT_GEMINI_API_KEY (or GEMINI_API_KEY)")
        log.info("no Gemini API key configured, subtitles will not be translated (set V2SRT_GEMINI_API_KEY)")

    print_plan(settings, translator is not None, whisper is not None)
    pipeline = Pipeline(
        settings, Reporter(console, show_cues=not args.quiet),
        procs=procs, media=media, whisper=whisper, translator=translator,
    )  # fmt: skip

    failures = 0
    for number, job in enumerate(jobs, 1):
        counter = f"[{number}/{len(jobs)}] " if len(jobs) > 1 else ""
        console.rule(f"{counter}{escape(job.src.name)}", style="dim")
        log.debug("job %d/%d: %s -> %s", number, len(jobs), job.src, job.output)
        try:
            if job.kind == "srt":
                result = pipeline.run_srt(job.src, job.output)
            else:
                result = pipeline.run_media(job.src, job.output)
        except (Cancelled, KeyboardInterrupt):
            raise
        except (V2SRTError, OSError) as e:
            failures += 1
            log.debug("job failed", exc_info=True)
            console.print(f"[bold red]failed:[/] {escape(str(e))}")
            continue
        log.debug("job done: %s", result)
        print_result(result)

    if translator is not None and (translator.stats.requests or translator.stats.failed_cues):
        st = translator.stats
        log.debug("gemini statistics: %s", st)
        thinking = f" / {st.thinking_tokens:,} thinking" if st.thinking_tokens else ""
        console.print(
            f"[dim]Gemini: {st.requests} requests, {st.prompt_tokens:,} input / {st.output_tokens:,} output"
            f"{thinking} tokens[/]"
        )
    if len(jobs) > 1:
        style = "red" if failures else "green"
        console.print(f"[{style}]{len(jobs) - failures} of {len(jobs)} files done[/]")
    return 1 if failures else 0


def print_plan(settings: Settings, translate: bool, transcribe: bool) -> None:
    parts = []
    if transcribe:
        model = Path(settings.whisper_model or "").name
        vad = "VAD" if settings.use_vad and settings.vad_model else "no VAD"
        lang = settings.language if settings.language != "auto" else "auto-detect"
        parts.append(f"whisper [bold]{escape(model)}[/] ({vad}, {lang})")
    if translate:
        target = language_name(settings.target_language) or settings.target_language
        mode = ", bilingual" if settings.bilingual else ""
        parts.append(
            f"→ [bold]{escape(target)}[/] with {escape(settings.gemini_model)} "
            f"(thinking: {settings.gemini_thinking}{mode})"
        )
    console.print("  ".join(parts))


def print_result(result: JobResult) -> None:
    minutes, seconds = divmod(int(result.elapsed), 60)
    console.print(
        f"[green]✓[/] {result.cues} subtitles → [bold]{escape(str(result.output))}[/] "
        f"[dim]({minutes}m{seconds:02}s)[/]"
    )
    if result.source_output:
        console.print(f"  original text → {escape(str(result.source_output))}")
    if result.cues == 0:
        console.print("[yellow]  no speech was recognized[/]")
    if result.untranslated:
        console.print(f"[yellow]  {result.untranslated} subtitles could not be translated and keep the original text[/]")


# --------------------------------------------------------------------------- init / check


def cmd_init(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(prog="v2srt init", description="Create a .env configuration file.")
    p.add_argument("--global", dest="global_", action="store_true",
                   help=f"write the per-user file ({user_config_dir() / '.env'}) instead of ./.env")  # fmt: skip
    p.add_argument("-f", "--force", action="store_true", help="overwrite an existing file")
    args = p.parse_args(argv)
    path = (user_config_dir() if args.global_ else Path.cwd()) / ".env"
    if path.exists() and not args.force:
        console.print(f"[red]error:[/] {escape(str(path))} already exists (use --force to overwrite)")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ENV_TEMPLATE, encoding="utf-8")
    console.print(f"[green]✓[/] wrote {escape(str(path))}")
    local = Path.cwd() / ".env"
    if args.global_ and local.is_file() and local.resolve() != path.resolve():
        console.print(
            f"[yellow]note:[/] in this directory {escape(str(local))} is used instead "
            "(only one configuration file is read)"
        )
    console.print("Fill in V2SRT_WHISPER_MODEL (and V2SRT_GEMINI_API_KEY for translation), then run `v2srt check`.")
    return 0


def cmd_check(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(prog="v2srt check", description="Verify the configuration.")
    _common(p)
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    config = load_config(args.env_file)
    settings, sources = config.settings, config.sources

    if config.env_file:
        console.print(f"[bold]Configuration file:[/] {escape(str(config.env_file))}")
    else:
        console.print("[bold]Configuration file:[/] [yellow]none[/]")
        console.print(
            f"  looked for {escape(str(Path.cwd() / '.env'))} and {escape(str(user_config_dir() / '.env'))}; "
            "run `v2srt init` to create one"
        )

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("")
    table.add_column("item")
    table.add_column("value")
    table.add_column("source", style="dim")
    ok = True

    def row(passed: bool | None, item: str, value: str, field_name: str | None = None) -> None:
        nonlocal ok
        mark = {True: "[green]✓[/]", False: "[red]✗[/]", None: "[dim]-[/]"}[passed]
        ok = ok and passed is not False
        source = sources.get(field_name, "default") if field_name else ""
        table.add_row(mark, item, escape(value), escape(source))

    def tool(item: str, value: str, name: str, env: str, field_name: str) -> str | None:
        try:
            path = resolve_executable(value, name, env)
        except V2SRTError as e:
            row(False, item, str(e), field_name)
            return None
        row(True, item, path, field_name)
        return path

    ffmpeg = tool("ffmpeg", settings.ffmpeg, "ffmpeg", "V2SRT_FFMPEG", "ffmpeg")
    if ffmpeg:
        try:
            row(True, "ffprobe", resolve_ffprobe(ffmpeg, settings.ffprobe), "ffprobe")
        except V2SRTError as e:
            row(False, "ffprobe", str(e), "ffprobe")
    tool("whisper-cli", settings.whisper_cli, "whisper-cli", "V2SRT_WHISPER_CLI", "whisper_cli")

    for item, value, field_name, required in (
        ("whisper model", settings.whisper_model, "whisper_model", True),
        ("VAD model", settings.vad_model, "vad_model", False),
    ):
        if not value:
            row(False if required else None, item, "not set", field_name)
        elif Path(value).is_file():
            size = Path(value).stat().st_size / 1024**2
            row(True, item, f"{value} ({size:,.0f} MB)", field_name)
        else:
            row(False, item, f"{value} (file not found)", field_name)

    row(None, "language", settings.language, "language")
    seg = format_clock(settings.segment_ms) if settings.segment_ms else "off"
    row(None, "segments", f"{seg}, {settings.whisper_jobs} parallel job(s)", "segment_ms")

    if not settings.gemini_api_key:
        row(None, "translation", "disabled (no API key)", "gemini_api_key")
    elif not settings.translate:
        row(None, "translation", "disabled (V2SRT_TRANSLATE=false)", "translate")
    else:
        row(None, "API key", mask(settings.gemini_api_key) or "", "gemini_api_key")
        try:
            with console.status("Contacting Gemini..."):
                name = GeminiTranslator(settings).check()
            row(True, "Gemini model", f"{settings.gemini_model} ({name})", "gemini_model")
        except V2SRTError as e:
            row(False, "Gemini model", f"{settings.gemini_model}: {e}", "gemini_model")
        row(None, "thinking level", settings.gemini_thinking, "gemini_thinking")
        target = language_name(settings.target_language) or settings.target_language
        row(None, "target language", target, "target_language")
        if settings.proxy:
            row(None, "proxy", settings.proxy, "proxy")

    log_dir = resolve_log_dir(settings.log_dir)
    log_value = str(log_dir) if log_dir else "off"
    if log_dir and settings.log_gemini:
        log_value += " (+ Gemini requests)"
    row(None, "logs", log_value, "log_dir")

    console.print()
    console.print(table)
    console.print()
    console.print("[green]ready[/]" if ok else "[red]some checks failed[/]")
    return 0 if ok else 1


# --------------------------------------------------------------------------- clean


@dataclass
class CleanGroup:
    category: str
    location: str
    paths: list[Path]
    remove_if_empty: list[Path]  # directories removed afterwards, but only when nothing else is left in them


def _size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        return 0


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB"):
        if value < 1024:
            return f"{value:,.0f} {unit}" if unit == "B" else f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} GB"


def find_cleanable(*, config: bool, logs: bool, temp: bool, env_file: Path | None = None) -> list[CleanGroup]:
    """Everything v2srt stored outside the working directory. Never includes ``./.env``."""
    protected = (Path.cwd() / ".env").resolve()
    groups: list[CleanGroup] = []

    if config:
        env = user_config_dir() / ".env"
        if env.is_file() and env.resolve() != protected:
            groups.append(CleanGroup("config", str(env), [env], [env.parent]))

    if logs:
        default = default_log_dir()
        directories = [default]
        try:
            configured = load_config(env_file).settings.log_dir
        except V2SRTError:
            configured = None
        # Only an explicitly configured V2SRT_LOG_DIR adds a second directory.
        if configured and configured != LOG_OFF and Path(configured).resolve() != default.resolve():
            directories.append(Path(configured))
        for directory in directories:
            files = log_files(directory)  # only files named like v2srt's logs, even in a shared directory
            if files:
                parents = [directory, directory.parent] if directory == default else [directory]
                groups.append(CleanGroup("logs", f"{directory} ({len(files)} files)", files, parents))

    if temp:
        tmp = Path(tempfile.gettempdir())
        leftovers = sorted(p for p in tmp.glob("v2srt-*") if p.is_dir())
        if leftovers:
            groups.append(CleanGroup("temp", f"{tmp / 'v2srt-*'} ({len(leftovers)} folders)", leftovers, []))

    for group in groups:
        group.paths = [p for p in group.paths if p.resolve() != protected]
    return [g for g in groups if g.paths]


def cmd_clean(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(
        prog="v2srt clean",
        description="Remove what v2srt stored on this computer: the per-user configuration file, log files and "
        "leftover temporary files. The .env file in the current directory is never deleted.",
    )
    p.add_argument("--config", action="store_true", help="only the per-user configuration file")
    p.add_argument("--logs", action="store_true", help="only log files")
    p.add_argument("--temp", action="store_true", help="only leftover temporary files")
    p.add_argument("-y", "--yes", action="store_true", help="delete without asking")
    _env_file_option(p)
    args = p.parse_args(argv)
    everything = not (args.config or args.logs or args.temp)

    groups = find_cleanable(
        config=everything or args.config,
        logs=everything or args.logs,
        temp=everything or args.temp,
        env_file=args.env_file,
    )
    local_env = Path.cwd() / ".env"
    if not groups:
        console.print("Nothing to clean.")
        return 0

    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("what")
    table.add_column("location")
    table.add_column("size", justify="right")
    total = 0
    for group in groups:
        size = sum(_size(path) for path in group.paths)
        total += size
        table.add_row(group.category, escape(group.location), _human(size))
    console.print("The following will be [bold red]permanently deleted[/]:")
    console.print(table)
    if local_env.is_file():
        console.print(f"[dim]Kept: {escape(str(local_env))} (current directory)[/]")
    if any(g.category == "temp" for g in groups):
        console.print("[dim]Temporary folders are left over from interrupted runs; do not clean while v2srt runs.[/]")

    if not args.yes:
        if not sys.stdin.isatty():
            console.print("[red]error:[/] not running interactively; pass --yes to confirm")
            return 1
        try:
            confirmed = Confirm.ask(f"Delete {_human(total)}?", console=console, default=False)
        except EOFError:  # no input available (on Windows NUL even claims to be a terminal)
            console.print()
            confirmed = False
        if not confirmed:
            console.print("Nothing was deleted.")
            return 0

    errors = 0
    for group in groups:
        for path in group.paths:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError as e:
                errors += 1
                console.print(f"[red]could not delete[/] {escape(str(path))}: {escape(str(e))}")
        for directory in group.remove_if_empty:
            try:
                directory.rmdir()  # fails (harmlessly) if anything else is in there
            except OSError:
                pass
    if errors:
        console.print(f"[yellow]done, but {errors} item(s) could not be deleted[/]")
        return 1
    console.print(f"[green]✓[/] deleted {_human(total)}")
    return 0


# --------------------------------------------------------------------------- entry point


def main(argv: Sequence[str] | None = None) -> None:
    global _run_log
    # A redirected stdout on Windows uses the ANSI code page (e.g. GBK); never crash on characters it lacks.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv and argv[0] in COMMANDS and not Path(argv[0]).exists() else "run"
    if argv and argv[0] == command:
        argv = argv[1:]
    handler = {"run": cmd_run, "init": cmd_init, "check": cmd_check, "clean": cmd_clean}[command]
    try:
        code = handler(argv)
    except (KeyboardInterrupt, Cancelled):
        log.debug("interrupted by the user")
        console.print("\n[yellow]interrupted[/]")
        code = 130
    except V2SRTError as e:
        log.debug("error: %s", e)
        console.print(f"[bold red]error:[/] {escape(str(e))}")
        code = 1
    except Exception:
        log.exception("unexpected error")
        console.print("[bold red]This is a bug in v2srt.[/] Please report it together with the log file.")
        code = 1
    finally:
        if _run_log is not None:
            log.debug("exit")
            console.print(f"[dim]log: {escape(str(_run_log.path))}[/]")
            if _run_log.gemini is not None and _run_log.gemini.path.exists():
                console.print(f"[dim]Gemini requests: {escape(str(_run_log.gemini.path))}[/]")
            _run_log.close()
            _run_log = None
    sys.exit(code)
