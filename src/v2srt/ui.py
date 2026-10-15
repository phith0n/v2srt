"""Terminal output: progress bars and the live subtitle preview."""

import threading
from contextlib import contextmanager
from typing import Callable, Iterator

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from .subtitle import Cue
from .timeutil import format_clock


class _AmountColumn(ProgressColumn):
    """``12:34 / 1:02:03`` for time based tasks, ``120 / 300`` for counters."""

    def render(self, task: Task) -> Text:
        if task.fields.get("unit") == "ms":
            total = format_clock(task.total) if task.total else "?"
            return Text(f"{format_clock(task.completed)} / {total}", style="progress.download")
        total = int(task.total) if task.total is not None else "?"
        return Text(f"{int(task.completed)} / {total}", style="progress.download")


class Reporter:
    def __init__(self, console: Console, *, show_cues: bool = True):
        self.console = console
        self.show_cues = show_cues
        self._progress: Progress | None = None
        self._lock = threading.Lock()
        self._transcribe: TaskID | None = None
        self._translate: TaskID | None = None
        self._chunk_pos: dict[int, int] = {}
        self._transcribe_total = 0
        self._translate_total = 0

    @contextmanager
    def job(self) -> Iterator[None]:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description:<12}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            _AmountColumn(),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
            expand=True,
        )
        self._progress = progress
        self._transcribe = self._translate = None
        self._chunk_pos = {}
        try:
            with progress:
                yield
        finally:
            self._progress = None

    @contextmanager
    def stage(self, description: str, total_ms: int | None) -> Iterator[Callable[[int], None]]:
        """A simple time-based stage (audio extraction, silence detection)."""
        assert self._progress is not None
        task = self._progress.add_task(description, total=total_ms or None, unit="ms")

        def update(position_ms: int) -> None:
            self._progress.update(task, completed=min(position_ms, total_ms or position_ms))

        yield update
        self._progress.update(task, total=total_ms or 1, completed=total_ms or 1)

    # ---------------------------------------------------------------- transcription

    def transcribe_start(self, duration_ms: int) -> None:
        assert self._progress is not None
        self._transcribe_total = duration_ms
        self._transcribe = self._progress.add_task("Transcribing", total=duration_ms or None, unit="ms")

    def transcribe_progress(self, chunk: int, position_ms: int) -> None:
        if self._progress is None or self._transcribe is None:
            return
        with self._lock:
            if position_ms <= self._chunk_pos.get(chunk, 0):
                return
            self._chunk_pos[chunk] = position_ms
            done = sum(self._chunk_pos.values())
        self._progress.update(self._transcribe, completed=done)

    def transcribe_finish(self) -> None:
        if self._progress is not None and self._transcribe is not None:
            total = self._transcribe_total or 1
            self._progress.update(self._transcribe, total=total, completed=total)

    # ---------------------------------------------------------------- translation

    def translate_start(self) -> None:
        assert self._progress is not None
        self._translate_total = 0
        self._translate = self._progress.add_task("Translating", total=0, unit="cues")

    def translate_submitted(self, count: int) -> None:
        if self._progress is not None and self._translate is not None:
            with self._lock:
                self._translate_total += count
                total = self._translate_total
            self._progress.update(self._translate, total=total)

    def translate_done(self, count: int) -> None:
        if self._progress is not None and self._translate is not None:
            self._progress.advance(self._translate, count)

    # ---------------------------------------------------------------- output

    def cue(self, cue: Cue) -> None:
        if not self.show_cues:
            return
        stamp = format_clock(cue.start)
        pad = " " * (len(stamp) + 2)
        source = escape(cue.text.replace("\n", " "))
        if cue.translation and cue.translation != cue.text:
            translation = escape(cue.translation.replace("\n", " "))
            self.console.print(f"[dim]{stamp}[/]  [dim]{source}[/]\n{pad}[bold cyan]{translation}[/]", highlight=False)
        else:
            self.console.print(f"[dim]{stamp}[/]  {source}", highlight=False)
