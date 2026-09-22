# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Crash capture for the standalone supervisors.

A supervisor runs without a visible console (a hidden Windows console, a macOS
app bundle, a systemd unit), so an unhandled exception or a native fault in the
supervisor process itself left no trace anywhere: the app log is written by the
supervisor and stops with it. Every launcher calls capture_fatal_errors once, so
those errors land in a plain append-only file next to the app log.

Main Features:
* faulthandler writes native faults of every thread to logs/supervisor-crash.log
* Unhandled exceptions of the main thread and of any thread are appended with a
  timestamp and their traceback, then handed to the default hooks
"""

import faulthandler
import os
import sys
import threading
import time
import traceback

CRASH_LOG_NAME = "supervisor-crash.log"

_crash_log = None


def capture_fatal_errors(logs_dir):
    global _crash_log
    path = os.path.join(logs_dir, CRASH_LOG_NAME)
    _crash_log = open(path, "a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=_crash_log, all_threads=True)

    def _write(kind, exc_type, exc, tb):
        _crash_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {kind}\n")
        traceback.print_exception(exc_type, exc, tb, file=_crash_log)
        _crash_log.flush()

    def _excepthook(exc_type, exc, tb):
        _write("unhandled exception in the supervisor", exc_type, exc, tb)
        sys.__excepthook__(exc_type, exc, tb)

    def _thread_hook(args):
        name = args.thread.name if args.thread is not None else "?"
        _write(f"unhandled exception in thread {name}", args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = _excepthook
    threading.excepthook = _thread_hook
    return path
