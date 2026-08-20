# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared idle-unload timer for expensive in-RAM resources.

The "keep a non-mmap resource loaded for a bounded window after last use, then
drop it once the window lapses" policy is the same for the GTE lyrics model,
the CLAP text model and the hyperbolic tree cache: one background thread polls
an expiry time and runs a module-specific unload action. This module owns the
thread and the lock once instead of each manager re-deriving them.

Main Features:
* IdleUnloadTimer.arm(duration, on_expire) (re)arms the window and reports
  whether a fresh timer thread was started
* IdleUnloadTimer.expiry() reads the current expiry for status reporting
* The worker thread runs on_expire exactly once, then clears the window and
  exits; an expiring action that raises still clears the window
"""

import threading
import time


class IdleUnloadTimer:
    def __init__(self):
        self._lock = threading.RLock()
        self._expiry_time = None
        self._timer_thread = None

    def lock(self):
        return self._lock

    def arm(self, duration, on_expire):
        started = False
        with self._lock:
            self._expiry_time = time.time() + duration
            if self._timer_thread is None or not self._timer_thread.is_alive():
                thread = threading.Thread(
                    target=self._worker, args=(on_expire,), daemon=True
                )
                thread.start()
                self._timer_thread = thread
                started = True
        return started

    def expiry(self):
        with self._lock:
            return self._expiry_time

    def _worker(self, on_expire):
        while True:
            with self._lock:
                expiry = self._expiry_time
                if expiry is None:
                    break
                if expiry - time.time() <= 0:
                    try:
                        on_expire()
                    finally:
                        self._expiry_time = None
                        self._timer_thread = None
                    break
                time_remaining = expiry - time.time()
            time.sleep(min(1.0, max(0.05, time_remaining)))
