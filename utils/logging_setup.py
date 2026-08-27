"""Logging for SAGE.

`get_logger(__name__)` gives a logger under the ``sage.*`` namespace. Handlers
live on the ``sage`` root:

  * a stderr ``StreamHandler`` (INFO)         -- terminal / CLI
  * a ``_CallbackHandler`` (INFO)             -- the live UI console
  * a per-run ``FileHandler`` (DEBUG or INFO) -- ``<run_dir>/run.log``, added
    for the duration of a run by :func:`run_log_file`

Call :func:`init_logging` once at process start. Wrap a run in
``with run_log_file(run_dir, level): ...``.
"""
from __future__ import annotations

# default
import logging
import os
import sys
import threading
from contextlib import contextmanager
from typing import Callable, Iterator

ROOT = "sage"

_FILE_FORMAT = "%(asctime)s  %(levelname)-7s %(shortname)-28s %(message)s"
_TERSE_FORMAT = "%(levelname)-7s %(shortname)s  %(message)s"


class _ShortNameFormatter(logging.Formatter):
    """Adds ``%(shortname)s`` -- the logger name with the ``sage.`` prefix removed."""

    def format(self, record: logging.LogRecord) -> str:
        record.shortname = record.name.removeprefix(ROOT + ".")
        return super().format(record)


class _CallbackHandler(logging.Handler):
    """Fan a formatted record out to registered callbacks (the UI console).
    Non-blocking; callback exceptions are swallowed."""

    def __init__(self) -> None:
        super().__init__()
        self._callbacks: list[Callable[[str], None]] = []
        self._lock = threading.Lock()

    def add(self, cb: Callable[[str], None]) -> None:
        with self._lock:
            self._callbacks.append(cb)

    def remove(self, cb: Callable[[str], None]) -> None:
        with self._lock:
            if cb in self._callbacks:
                self._callbacks.remove(cb)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            return
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(msg)
            except Exception:
                pass


class _RateLimitFilter(logging.Filter):
    """Collapse a burst of identical DEBUG/WARNING lines: pass the first ``K``,
    drop the rest, keeping a count. Keys on the message *template* (before
    %-substitution) so ``"failed at %.1fmm"`` collapses across every value. INFO
    and ERROR+ are never limited. Run-scoped -- :func:`run_log_file` creates one
    and dumps the tally on exit."""

    K = 3
    _LIMITED = (logging.DEBUG, logging.WARNING)

    def __init__(self) -> None:
        super().__init__()
        self._seen: dict[tuple, int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno not in self._LIMITED:
            return True
        key = (record.name, record.levelno, record.msg)
        n = self._seen.get(key, 0)
        self._seen[key] = n + 1
        return n < self.K

    def suppressed(self) -> Iterator[tuple[str, str, int]]:
        for (name, _lvl, msg), n in list(self._seen.items()):
            if n > self.K:
                yield name, str(msg), n - self.K


_UI_HANDLER = _CallbackHandler()
_UI_HANDLER.setLevel(logging.INFO)
_UI_HANDLER.setFormatter(_ShortNameFormatter(_TERSE_FORMAT))

# The UI-console subscribe/unsubscribe API used by app.py (kept under these names).
add_console_subscriber = _UI_HANDLER.add
remove_console_subscriber = _UI_HANDLER.remove

_initialized = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith(ROOT) else f"{ROOT}.{name}")


def init_logging(terminal_level: int = logging.INFO) -> None:
    """Attach the terminal + UI handlers to the ``sage`` root. Idempotent."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    root = logging.getLogger(ROOT)
    root.setLevel(logging.DEBUG)          # the per-run file handler filters lower
    root.propagate = False

    term = logging.StreamHandler(sys.stderr)
    term.setLevel(terminal_level)
    term.setFormatter(_ShortNameFormatter(_TERSE_FORMAT))
    root.addHandler(term)
    root.addHandler(_UI_HANDLER)


@contextmanager
def run_log_file(run_dir: str, level: int = logging.DEBUG):
    """Tee ``sage`` logging into ``<run_dir>/run.log`` for the duration of a run,
    collapsing repeated lines and appending a suppression tally on exit."""
    root = logging.getLogger(ROOT)
    fh = logging.FileHandler(os.path.join(run_dir, "run.log"), encoding="utf-8", errors="replace")
    fh.setLevel(level)
    fh.setFormatter(_ShortNameFormatter(_FILE_FORMAT))

    ratelimit = _RateLimitFilter()
    fh.addFilter(ratelimit)
    _UI_HANDLER.addFilter(ratelimit)
    root.addHandler(fh)
    try:
        yield
    finally:
        for name, msg, n in ratelimit.suppressed():
            logging.getLogger(name).info("suppressed %d more like: %s", n, msg)
        root.removeHandler(fh)
        _UI_HANDLER.removeFilter(ratelimit)
        fh.close()
