"""The processing pipeline.

::

    ffmpeg ─► audio.wav ─► split at silences ─► whisper-cli (N parallel jobs, streamed line by line)
                                                    │  cues
                                                    ▼
                                          batches ─► Gemini (M parallel requests)
                                                    │  translated batches, in any order
                                                    ▼
                                          OrderedSink ─► SRT files (written incrementally, in order)

Recognition and translation overlap completely: a batch is translated as soon as whisper has produced it.
"""

import logging
import os
import shutil
import tempfile
import threading
import time
import wave
from collections import Counter, deque
from concurrent.futures import FIRST_EXCEPTION, Future, wait
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Callable, Iterator, Sequence

from .config import Settings, language_base
from .errors import ConfigError, V2SRTError
from .media import Chunk, Media, plan_boundaries, split_wav
from .proc import DaemonExecutor, ProcessGroup
from .subtitle import Cue, CueCleaner, CueSequencer, OutputMode, SrtWriter, parse_srt
from .timeutil import format_clock
from .translator import GeminiTranslator
from .ui import Reporter
from .whisper import Whisper

log = logging.getLogger(__name__)

CONTEXT_CUES = 8  # preceding cues sent along with every batch
FLUSH_SECONDS = 45.0  # send a partial batch when whisper is slow ...
FLUSH_MIN_CUES = 8  # ... and it has at least this many cues


class OrderedSink:
    """Collects batches that complete in any order and emits their cues in subtitle order.

    Batches are identified by ``(chunk, batch)``. The number of batches of a chunk only becomes known once the
    chunk is fully transcribed, which :meth:`finish_chunk` reports.
    """

    def __init__(self, chunks: int, emit: Callable[[Cue], None]):
        self.chunks = chunks
        self._emit = emit
        self._lock = threading.Lock()
        self._ready: dict[tuple[int, int], Sequence[Cue]] = {}
        self._batch_counts: dict[int, int] = {}
        self._chunk = 0
        self._batch = 0

    @property
    def done(self) -> bool:
        return self._chunk >= self.chunks

    def put(self, chunk: int, batch: int, cues: Sequence[Cue]) -> None:
        with self._lock:
            self._ready[(chunk, batch)] = cues
            self._drain()

    def finish_chunk(self, chunk: int, batches: int) -> None:
        with self._lock:
            self._batch_counts[chunk] = batches
            self._drain()

    def _drain(self) -> None:
        while self._chunk < self.chunks:
            cues = self._ready.pop((self._chunk, self._batch), None)
            if cues is not None:
                for cue in cues:
                    self._emit(cue)
                self._batch += 1
            elif self._batch_counts.get(self._chunk) == self._batch:
                self._chunk += 1
                self._batch = 0
            else:
                return


@dataclass
class JobResult:
    output: Path
    source_output: Path | None
    cues: int
    untranslated: int
    language: str | None
    elapsed: float


class Outputs:
    """The subtitle files of one job. Written to ``*.part`` files that are renamed when the job succeeds."""

    def __init__(self, output: Path, mode: OutputMode, source_output: Callable[[], Path] | None):
        self.output = output
        self._main = SrtWriter(self._part(output), mode)
        self._source_path_fn = source_output
        self._source: SrtWriter | None = None
        self.source_output: Path | None = None

    @staticmethod
    def _part(path: Path) -> Path:
        return path.with_name(path.name + ".part")

    def write(self, cue: Cue) -> None:
        self._main.write(cue)
        if self._source_path_fn is not None:
            if self._source is None:
                self.source_output = self._source_path_fn()
                self._source = SrtWriter(self._part(self.source_output), OutputMode.SOURCE)
            self._source.write(cue)

    @property
    def count(self) -> int:
        return self._main.count

    def commit(self) -> None:
        self.close()
        for writer, final in ((self._main, self.output), (self._source, self.source_output)):
            if writer is not None and final is not None:
                os.replace(writer.path, final)

    def close(self) -> list[Path]:
        """Close the files and return the paths of the partial files that were left behind."""
        self._main.close()
        if self._source is not None:
            self._source.close()
        return [w.path for w in (self._main, self._source) if w is not None and w.path.exists()]


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        reporter: Reporter,
        *,
        procs: ProcessGroup,
        media: Media | None = None,
        whisper: Whisper | None = None,
        translator: GeminiTranslator | None = None,
    ):
        self.settings = settings
        self.reporter = reporter
        self.procs = procs
        self.media = media
        self.whisper = whisper
        self.translator = translator

    # ------------------------------------------------------------------ jobs

    def run_media(self, src: Path, output: Path) -> JobResult:
        """Transcribe (and translate) a video or audio file."""
        if self.media is None or self.whisper is None:
            raise ConfigError("ffmpeg and whisper-cli are required to transcribe media files")
        started = time.monotonic()
        with self._workdir() as tmp, self.reporter.job():
            info = self.media.probe(src)
            if self.settings.audio_track is not None and self.settings.audio_track >= info.audio_streams:
                raise ConfigError(
                    f"{src.name} has {info.audio_streams} audio track(s); "
                    f"--audio-track must be between 0 and {info.audio_streams - 1}"
                )

            wav = tmp / "audio.wav"
            with self.reporter.stage("Audio", info.duration_ms) as update:
                self.media.extract_audio(src, wav, audio_track=self.settings.audio_track, on_progress=update)
            with wave.open(str(wav), "rb") as w:
                duration = w.getnframes() * 1000 // w.getframerate()
            if duration <= 0:
                raise V2SRTError(f"{src.name}: the audio track is empty")

            silences = None
            s = self.settings
            if not s.cut_ms and s.segment_ms > 0 and duration > s.segment_ms * 1.25:
                with self.reporter.stage("Silences", duration) as update:
                    silences = self.media.detect_silences(wav, on_progress=update)
            boundaries = plan_boundaries(duration, segment_ms=s.segment_ms, cut_ms=s.cut_ms, silences=silences)
            chunks = split_wav(wav, boundaries, tmp)
            if len(chunks) > 1:
                log.info("split into %d segments at %s", len(chunks), ", ".join(format_clock(b) for b in boundaries))

            languages: dict[int, str] = {}
            if s.language != "auto":
                languages = {c.index: s.language for c in chunks}

            def source_path() -> Path:
                lang = _majority(languages) or "orig"
                return output.with_name(f"{_stem(output)}.{lang}.srt")

            translator = self._job_translator()
            outputs = Outputs(output, self._mode(translator), source_path if s.keep_source and translator else None)
            untranslated = self._process(
                outputs,
                translator,
                title=src.stem,
                n_chunks=len(chunks),
                producers=[self._transcriber(chunk, languages) for chunk in chunks],
                duration=duration,
                languages=languages,
            )
            return JobResult(
                output=output,
                source_output=outputs.source_output,
                cues=outputs.count,
                untranslated=untranslated,
                language=_majority(languages),
                elapsed=time.monotonic() - started,
            )

    def run_srt(self, src: Path, output: Path) -> JobResult:
        """Translate an existing subtitle file."""
        started = time.monotonic()
        cues = parse_srt(_read_text(src))
        if not cues:
            raise V2SRTError(f"{src.name}: no subtitles found")
        translator = self._job_translator()
        if translator is None:
            raise ConfigError("translating a subtitle file requires a Gemini API key (set V2SRT_GEMINI_API_KEY)")

        language = None if self.settings.language == "auto" else self.settings.language

        def producer(submit: Callable[[list[Cue]], None], previous: deque) -> None:
            for batch in batched(cues, self.settings.batch_size):
                submit(list(batch))

        with self.reporter.job():
            outputs = Outputs(output, self._mode(translator), None)
            untranslated = self._process(
                outputs, translator, title=src.stem, n_chunks=1, producers=[producer], duration=None,
                languages={0: language} if language else {},
            )  # fmt: skip
        return JobResult(output, None, outputs.count, untranslated, language, time.monotonic() - started)

    # -------------------------------------------------------------- internals

    def _job_translator(self) -> GeminiTranslator | None:
        t = self.translator
        if t is not None and t.fatal:
            log.warning("translation is disabled for the rest of this run: %s", t.fatal)
            return None
        return t

    def _mode(self, translator: GeminiTranslator | None) -> OutputMode:
        if translator is None:
            return OutputMode.SOURCE
        return OutputMode.BILINGUAL if self.settings.bilingual else OutputMode.TRANSLATED

    def _transcriber(self, chunk: Chunk, languages: dict[int, str]) -> Callable:
        """A producer that streams one chunk through whisper and submits batches as they fill up."""
        whisper = self.whisper
        assert whisper is not None
        batch_size = self.settings.batch_size

        def produce(submit: Callable[[list[Cue]], None], previous: deque) -> None:
            cleaner = CueCleaner()
            batch: list[Cue] = []
            batch_started = 0.0

            def on_language(lang: str) -> None:
                languages[chunk.index] = lang
                if chunk.index == 0:
                    log.info("detected language: %s", lang)

            for raw in whisper.transcribe(
                chunk,
                on_language=on_language,
                on_progress=lambda pos: self.reporter.transcribe_progress(chunk.index, pos),
            ):
                cue = cleaner(raw)
                if cue is None:
                    continue
                self.reporter.transcribe_progress(chunk.index, cue.end - chunk.start)
                if not batch:
                    batch_started = time.monotonic()
                batch.append(cue)
                if len(batch) >= batch_size or (
                    len(batch) >= FLUSH_MIN_CUES and time.monotonic() - batch_started >= FLUSH_SECONDS
                ):
                    submit(batch)
                    batch = []
            if batch:
                submit(batch)
            self.reporter.transcribe_progress(chunk.index, chunk.duration)

        return produce

    def _process(
        self,
        outputs: Outputs,
        translator: GeminiTranslator | None,
        *,
        title: str,
        n_chunks: int,
        producers: list[Callable],
        duration: int | None,
        languages: dict[int, str],
    ) -> int:
        """Run producers (one per chunk) and translation concurrently, writing results in order."""
        s = self.settings
        sequencer = CueSequencer(lambda cue: (outputs.write(cue), self.reporter.cue(cue)))
        sink = OrderedSink(n_chunks, sequencer.push)
        untranslated = 0
        counter_lock = threading.Lock()
        translate_futures: list[Future] = []

        producer_pool = DaemonExecutor(min(s.whisper_jobs, n_chunks), "whisper")
        translate_pool = DaemonExecutor(s.translate_workers, "translate") if translator else None

        if duration is not None:
            self.reporter.transcribe_start(duration)
        if translator is not None:
            self.reporter.translate_start()

        def translate_batch(chunk: int, number: int, cues: list[Cue], previous: list[Cue]) -> None:
            nonlocal untranslated
            assert translator is not None
            failed = len(cues)
            try:
                failed = translator.translate(
                    cues, previous=previous, source_language=languages.get(chunk), title=title
                )
            except Exception:
                log.exception("unexpected error while translating")
            finally:
                with counter_lock:
                    untranslated += failed
                self.reporter.translate_done(len(cues))
                sink.put(chunk, number, cues)

        def run_producer(chunk: int, produce: Callable) -> None:
            number = 0
            previous: deque[Cue] = deque(maxlen=CONTEXT_CUES)

            def submit(cues: list[Cue]) -> None:
                nonlocal number
                context = list(previous)
                previous.extend(cues)
                if translate_pool is not None:
                    self.reporter.translate_submitted(len(cues))
                    translate_futures.append(translate_pool.submit(translate_batch, chunk, number, cues, context))
                else:
                    sink.put(chunk, number, cues)
                number += 1

            produce(submit, previous)
            sink.finish_chunk(chunk, number)

        committed = False
        futures: list[Future] = []
        try:
            futures = [producer_pool.submit(run_producer, i, p) for i, p in enumerate(producers)]
            _wait_all(futures)
            if duration is not None:
                self.reporter.transcribe_finish()
            _wait_all(list(translate_futures))
            if not sink.done:  # pragma: no cover - would be a bug
                raise V2SRTError("internal error: not all subtitles were written")
            sequencer.flush()
            outputs.commit()
            committed = True
        except BaseException:
            self.procs.cancel()
            if translator is not None:
                translator.cancel.set()
            # Give the killed whisper processes a moment to exit so their temporary files can be removed.
            wait(futures, timeout=3)
            raise
        finally:
            producer_pool.shutdown(cancel_pending=True)
            if translate_pool is not None:
                translate_pool.shutdown(cancel_pending=True)
            if not committed:
                partial = outputs.close()
                for path in partial:
                    log.warning("partial subtitles kept in %s", path)
        return untranslated

    @contextmanager
    def _workdir(self) -> Iterator[Path]:
        path = Path(tempfile.mkdtemp(prefix="v2srt-"))
        try:
            yield path
        finally:
            if self.settings.keep_temp:
                log.info("temporary files kept in %s", path)
            else:
                shutil.rmtree(path, ignore_errors=True)


def _wait_all(futures: list[Future]) -> None:
    """Wait for futures while staying responsive to Ctrl+C; re-raise the first failure."""
    pending = set(futures)
    while pending:
        done, pending = wait(pending, timeout=0.2, return_when=FIRST_EXCEPTION)
        for f in done:
            if not f.cancelled() and f.exception() is not None:
                raise f.exception()  # type: ignore[misc]


def _majority(languages: dict[int, str]) -> str | None:
    counts = Counter(v for v in languages.values() if v)
    return counts.most_common(1)[0][0] if counts else None


def _stem(path: Path) -> str:
    return path.name[: -len(".srt")] if path.name.lower().endswith(".srt") else path.stem


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def target_suffix(settings: Settings) -> str:
    """File name suffix for translated subtitle files, e.g. ``zh-CN``."""
    target = settings.target_language.strip()
    if language_base(target):
        return target
    return "".join(ch for ch in target.lower() if ch.isalnum()) or "translated"
