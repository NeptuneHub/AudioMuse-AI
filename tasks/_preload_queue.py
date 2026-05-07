"""Bounded-concurrency queue for cache-preload requests.

Search pages fire ``/api/<feature>/cache/preload`` on DOMContentLoaded so the
user's first query is fast. When the user opens several search pages in
quick succession the requests arrive in parallel and each starts loading a
large ONNX session / Voyager index into Flask's RAM at the same time — easy
way to push the container into OOM territory.

This module serves preload work through a small thread pool:

  * Endpoints call :func:`enqueue` with a unique ``name`` and a callable
    that performs the actual blocking load.
  * A ``ThreadPoolExecutor`` (default 2 workers; tunable via the
    ``PRELOAD_QUEUE_WORKERS`` env var) drains the queue. Up to N preloads
    run in parallel, the rest queue and wait.
  * Same ``name`` enqueued again while a job is pending → silently deduped,
    second call is a no-op.
  * The endpoint returns to the browser immediately; the user keeps
    browsing while the load proceeds in the background.
  * Each job runs inside ``app.app_context()`` so functions that touch
    ``flask.g`` (e.g. :func:`app_helper.get_db`) work the same as if they
    were called from a request handler.

The pool is process-local. Gunicorn here runs with ``--workers 1`` so a
single pool serializes everything across the four request threads. With more
gunicorn workers each gets its own pool — same per-worker concurrency cap.

Tuning:
  ``PRELOAD_QUEUE_WORKERS`` (default ``2``): max parallel loads.
  Higher = faster page warmup, but peak RAM scales linearly with this number
  during the brief overlap when N indexes load at once. ``1`` = strict serial
  (lowest RAM peak). ``4`` = aggressive (good for big-RAM hosts).
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Set

logger = logging.getLogger(__name__)


def _max_workers() -> int:
    # Four workers run four preloads in parallel for fast page warmup.
    # Per-thread allocator overhead is bounded by jemalloc (LD_PRELOAD'd in
    # the container Dockerfiles) plus aggressive ``MALLOC_CONF`` decay, so
    # raising this above the old glibc-safe default of 2 no longer leaks
    # hundreds of MB of arena slack.
    try:
        n = int(os.environ.get('PRELOAD_QUEUE_WORKERS', '4'))
    except (TypeError, ValueError):
        n = 4
    return max(1, min(8, n))  # clamp to a sensible range


class PreloadQueue:
    def __init__(self, max_workers: Optional[int] = None) -> None:
        self._max_workers = max_workers if max_workers is not None else _max_workers()
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix='preload-',
        )
        self._pending: Set[str] = set()
        self._lock = threading.Lock()

    def enqueue(self, name: str, load_fn: Callable[[], None]) -> bool:
        """Submit ``load_fn`` to the preload pool.

        Returns ``True`` if the job was added, ``False`` if a job with the
        same ``name`` was already pending (deduped).
        """
        with self._lock:
            if name in self._pending:
                return False
            self._pending.add(name)
        try:
            self._executor.submit(self._run, name, load_fn)
        except RuntimeError:
            # Pool is shut down (Flask is exiting). Drop the job silently.
            with self._lock:
                self._pending.discard(name)
            return False
        return True

    def is_pending(self, name: str) -> bool:
        with self._lock:
            return name in self._pending

    def _run(self, name: str, load_fn: Callable[[], None]) -> None:
        # Each preload job needs a Flask app context so anything touching
        # ``flask.g`` (e.g. ``app_helper.get_db``) doesn't crash. Importing
        # ``flask_app`` lazily here avoids module-import-time cycles.
        try:
            from flask_app import app
        except Exception as exc:
            logger.warning("Preload '%s': could not import flask app (%s); "
                           "running without context.", name, exc)
            app = None

        try:
            logger.info("Preload: starting '%s'", name)
            if app is not None:
                with app.app_context():
                    load_fn()
            else:
                load_fn()
            logger.info("Preload: finished '%s'", name)
        except Exception:
            # Log full traceback here so silent crashes don't hide. The
            # caller's expectation is "endpoint returns 200 instantly,
            # whatever happens in the worker is best-effort"; if the load
            # fails the next user-initiated search will lazy-load it again
            # inside a real request context.
            logger.exception("Preload '%s' failed; will be retried on next "
                             "user request via lazy-load.", name)
        finally:
            with self._lock:
                self._pending.discard(name)
            # Return transient load-time slack (DB result buffers, voyager
            # build buffers, intermediate numpy arrays) to the kernel so
            # active-state RSS stays close to the live working set. Without
            # this glibc keeps the slack inside its arenas — visible as
            # +400-700 MB of unexplained "in-use" memory.
            try:
                from ._idle_unload import _malloc_trim as _trim
                import gc as _gc
                _gc.collect()
                _trim()
            except Exception:
                pass


# Module-level singleton: every preload endpoint enqueues onto this.
PRELOAD_QUEUE = PreloadQueue()
