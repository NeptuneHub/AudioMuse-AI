"""
Idle auto-unload helper for in-memory caches (voyager indexes, etc.).

Each cache that wants the "lazy-load on first use, auto-unload after N seconds
of inactivity" behaviour creates one ``IdleUnloader`` instance and calls
``touch()`` on every successful use. After ``idle_seconds`` with no activity,
the supplied ``unload_fn`` is invoked from a background thread to free RAM.

The implementation is intentionally tiny: a single timestamp + one daemon
thread per cache. No locks held during ``unload_fn`` execution.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import logging
import os
import sys
import threading
import time
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


_LIBC_MALLOC_TRIM: Optional[Callable[[int], int]] = None
_LIBC_PROBED = False


def _malloc_trim() -> None:
    """Ask glibc to return free arenas to the kernel.

    Without this, RSS stays inflated after unloading large ONNX sessions or
    voyager indexes because glibc's malloc keeps freed pages cached in its
    own arenas. ``malloc_trim(0)`` triggers a sweep that hands them back.
    This helper is Linux/glibc-specific and may be ineffective when jemalloc
    is loaded via LD_PRELOAD.
    No-op on non-Linux platforms, on jemalloc-prefixed processes, or when the
    symbol is unavailable.
    """
    if not sys.platform.startswith("linux"):
        return
    if 'jemalloc' in os.environ.get('LD_PRELOAD', ''):
        return

    global _LIBC_MALLOC_TRIM, _LIBC_PROBED
    if not _LIBC_PROBED:
        _LIBC_PROBED = True
        try:
            libc_name = ctypes.util.find_library('c') or 'libc.so.6'
            libc = ctypes.CDLL(libc_name)
            trim = libc.malloc_trim
            trim.argtypes = [ctypes.c_int]
            trim.restype = ctypes.c_int
            _LIBC_MALLOC_TRIM = trim
        except (OSError, AttributeError):
            _LIBC_MALLOC_TRIM = None
    if _LIBC_MALLOC_TRIM is not None:
        try:
            _LIBC_MALLOC_TRIM(0)
        except Exception:
            pass


class IdleUnloader:
    def __init__(self, name: str, idle_seconds: int, unload_fn: Callable[[], None]):
        self._name = name
        self._idle_seconds = max(1, int(idle_seconds))
        self._unload_fn = unload_fn
        self._expiry: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def idle_seconds(self) -> int:
        return self._idle_seconds

    def set_idle_seconds(self, idle_seconds: int) -> None:
        """Update the idle window (used on first call once config is available)."""
        self._idle_seconds = max(1, int(idle_seconds))

    def touch(self) -> None:
        """Mark the cache as just used. Resets the idle timer and starts the
        background watcher if it isn't running."""
        now = time.time()
        with self._lock:
            self._expiry = now + self._idle_seconds
            if self._thread is None or not self._thread.is_alive():
                t = threading.Thread(
                    target=self._watcher,
                    name=f"idle-unload-{self._name}",
                    daemon=True,
                )
                self._thread = t
                t.start()

    def cancel(self) -> None:
        """Stop the watcher (e.g. cache was unloaded explicitly)."""
        with self._lock:
            self._expiry = None

    def status(self) -> Dict:
        with self._lock:
            expiry = self._expiry
        if expiry is None:
            return {"active": False, "seconds_remaining": 0, "idle_seconds": self._idle_seconds}
        remaining = max(0, int(expiry - time.time()))
        return {"active": True, "seconds_remaining": remaining, "idle_seconds": self._idle_seconds}

    def _watcher(self) -> None:
        while True:
            with self._lock:
                expiry = self._expiry
            if expiry is None:
                return
            remaining = expiry - time.time()
            if remaining <= 0:
                # Re-check under lock to avoid racing with touch()
                with self._lock:
                    if self._expiry is None or self._expiry > time.time():
                        continue
                    self._expiry = None
                    self._thread = None
                try:
                    logger.info(
                        "Idle-unload: '%s' inactive for %ds, releasing memory.",
                        self._name,
                        self._idle_seconds,
                    )
                    self._unload_fn()
                except Exception as e:
                    logger.warning("Idle-unload '%s' callback failed: %s", self._name, e)
                # Drop Python-side refs, then ask glibc to return free pages
                # to the kernel — otherwise RSS stays inflated after unloading
                # large ONNX/voyager allocations.
                try:
                    gc.collect()
                except Exception:
                    pass
                _malloc_trim()
                return
            # Sleep in 1-second chunks so cancel()/touch() reactions stay quick.
            time.sleep(min(1.0, remaining))
