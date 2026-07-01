# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Newest-first rotating log handler for the standalone builds.

Provides a logging handler that keeps the most recent lines at the top of the
file (so the log reads newest-first) and caps the file at a maximum line count,
flushing on a timer. Lives under ``macos`` but is imported by all three
platform supervisors.

Main Features:
* Prepends new records and truncates to a bounded line count in memory.
* Batches writes on a background flush timer to limit disk churn.
"""

import logging
import os
import threading


class NewestFirstFileHandler(logging.Handler):
    def __init__(self, path, max_lines=40000, flush_interval=1.0):
        super().__init__()
        self._path = path
        self._max_lines = max_lines
        self._flush_interval = flush_interval
        self._lines = []
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._timer = None
        self._closed = False
        self._load_existing()

    def _load_existing(self):
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
            self._lines[0:0] = msg.split("\n")
            del self._lines[self._max_lines :]
            if self._timer is None:
                self._timer = threading.Timer(self._flush_interval, self._flush)
                self._timer.daemon = True
                self._timer.start()

    def _flush(self):
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
                os.replace(tmp, self._path)
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
