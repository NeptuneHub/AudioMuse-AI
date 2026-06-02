"""A logging handler that keeps the log file *newest-line-first*.

Opening ``~/Library/Logs/AudioMuse-AI/audiomuse.log`` then shows the most recent
line at the top, so the latest activity is visible immediately without scrolling
to the end.

A literal "prepend each line to the file" would be O(file size) per record and
race a ``RotatingFileHandler``'s rollover, so instead this keeps the most recent
``max_lines`` formatted lines in memory (newest first) and rewrites the file
atomically on a short debounce. Size is bounded by ``max_lines`` rather than a
byte cap, so it replaces ``RotatingFileHandler`` (no separate ``.1/.2/.3``
backups). Multiple producer threads (the supervisor's per-child ``_pump``
threads) share one handler, so all access is guarded by a lock.
"""

import logging
import os
import threading


class NewestFirstFileHandler(logging.Handler):
    """Write log records newest-first into a single capped file."""

    def __init__(self, path, max_lines=40000, flush_interval=1.0):
        super().__init__()
        self._path = path
        self._max_lines = max_lines
        self._flush_interval = flush_interval
        self._lines = []  # newest-first: index 0 is the most recent physical line
        self._lock = threading.Lock()          # guards _lines / _timer (fast, held by emit)
        self._write_lock = threading.Lock()    # serializes the disk write in _flush
        self._timer = None
        self._closed = False
        self._load_existing()

    def _load_existing(self):
        """Seed from an existing (already newest-first) file so restarts keep history."""
        try:
            with open(self._path, "r", encoding="utf-8", errors="replace") as fh:
                self._lines = fh.read().splitlines()[: self._max_lines]
        except OSError:
            self._lines = []

    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            self.handleError(record)
            return
        with self._lock:
            if self._closed:
                return
            # A record may span several physical lines (e.g. a traceback). Insert
            # the whole block at the top in its natural order, so the block reads
            # top-to-bottom while sitting above all older records.
            self._lines[0:0] = msg.split("\n")
            del self._lines[self._max_lines:]
            if self._timer is None:
                self._timer = threading.Timer(self._flush_interval, self._flush)
                self._timer.daemon = True
                self._timer.start()

    def _flush(self):
        # _write_lock serializes concurrent _flush calls (timer thread vs close())
        # so they never race on the shared ``.tmp`` path; emit() never takes it, so
        # logging stays non-blocking. Always acquire _write_lock before _lock.
        with self._write_lock:
            with self._lock:
                self._timer = None
                data = "\n".join(self._lines)
            if data:
                data += "\n"
            tmp = self._path + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(data)
                os.replace(tmp, self._path)  # atomic: readers never see a partial file
            except OSError:
                pass

    def flush(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self._flush()

    def close(self):
        with self._lock:
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self._flush()
        super().close()
