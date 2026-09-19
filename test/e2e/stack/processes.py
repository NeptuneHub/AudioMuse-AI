# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Child-process management shared by every service the end-to-end stack starts.

Each service (gunicorn, the two queue workers, Navidrome) runs in its own session
so that a stop reaches every descendant, including the job child a worker forks,
and each writes to its own append-mode log file under the run directory that CI
uploads. wait_until is the one polling loop used by every readiness check, and
it stops early when the process being waited on has already died.

Main Features:
* ManagedProcess.start launches with start_new_session=True and stdout+stderr
  redirected to one log file; stop escalates SIGTERM to SIGKILL on the group
* tail and log_contains give readiness checks and failure messages the log text
* wait_until polls a predicate with a deadline and an optional death check
"""

import os
import signal
import subprocess
import time
from collections import deque

from .errors import StackError


class ManagedProcess:
    def __init__(self, name, argv, env, cwd, log_path):
        self.name = name
        self.argv = list(argv)
        self.env = dict(env)
        self.cwd = cwd
        self.log_path = log_path
        self.proc = None
        self.stopping = False
        self._log = None

    def start(self):
        self.stopping = False
        if self._log is not None:
            self._log.close()
        self._log = open(self.log_path, 'ab', buffering=0)
        stamp = time.strftime('%Y-%m-%d %H:%M:%S')
        self._log.write(f"=== {self.name} start {stamp}: {' '.join(self.argv)} ===\n".encode())
        self.proc = subprocess.Popen(
            self.argv,
            env=self.env,
            cwd=self.cwd,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return self.proc

    @property
    def pid(self):
        return None if self.proc is None else self.proc.pid

    @property
    def returncode(self):
        return None if self.proc is None else self.proc.poll()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def _signal_group(self, sig):
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            pass

    def stop(self, term_timeout=10.0):
        self.stopping = True
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self._signal_group(signal.SIGTERM)
            try:
                self.proc.wait(term_timeout)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)
                try:
                    self.proc.wait(5)
                except subprocess.TimeoutExpired:
                    pass
        self._signal_group(signal.SIGKILL)
        if self._log is not None:
            self._log.close()
            self._log = None

    def log_size(self):
        try:
            return os.path.getsize(self.log_path)
        except OSError:
            return 0

    def log_contains(self, needle, since=0):
        try:
            with open(self.log_path, 'rb') as handle:
                handle.seek(since)
                return needle.encode('utf-8') in handle.read()
        except OSError:
            return False

    def tail(self, lines=40):
        try:
            with open(self.log_path, 'rb') as handle:
                last = deque(handle, maxlen=lines)
        except OSError:
            return ''
        return b''.join(last).decode('utf-8', 'replace')

    def describe(self):
        state = 'running' if self.running() else f'exited with {self.returncode}'
        return f'{self.name} ({state}), last log lines:\n{self.tail()}'


def wait_until(predicate, timeout, what, interval=0.5, dead=None, detail=None):
    deadline = time.monotonic() + timeout
    last_error = None
    while True:
        try:
            value = predicate()
        except Exception as exc:
            last_error = exc
            value = None
        if value:
            return value
        if dead is not None and dead():
            extra = detail() if detail else ''
            raise StackError(f'{what}: the process died while waiting. {extra}')
        if time.monotonic() >= deadline:
            extra = detail() if detail else ''
            reason = f' (last error: {last_error})' if last_error else ''
            raise StackError(f'{what}: not ready after {timeout:.0f}s{reason}. {extra}')
        time.sleep(interval)
