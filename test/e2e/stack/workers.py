# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The queue workers and the control listener of the end-to-end stack.

Each worker is the real `python -u -m taskqueue.worker --queue <name>` (the -u is
a fork-safety requirement, as in supervisord.conf). A worker deliberately exits
after a cancel and after its job recycle limit, which supervisord would restart;
here a watchdog thread does the same and counts the restarts so a test can
assert on them. The control listener is the real `python -u -m taskqueue.control`
that turns the app's restart requests into calls on the harness control server
and acknowledges them in the database. Readiness is a log line each process
prints once it is connected.

Main Features:
* QueueWorker.start / wait_ready / stop, with restarts and dead counters; a
  stop through the control server is not respawned, a start resumes the watch
* ControlListener.start / wait_ready / stop for the restart handshake process
* listening(dsn) proves a worker holds its LISTEN connection in pg_stat_activity
"""

import sys
import threading
import time

from .paths import REPO_ROOT
from .postgres import scalar
from .processes import ManagedProcess, wait_until

READY_MARKER = 'ready; recycling after'
LISTENER_READY_MARKER = 'Listening on'
MAX_RESTARTS = 25


class _Supervised:
    def __init__(self, name, argv, env, log_path):
        self.name = name
        self.process = ManagedProcess(name, argv, env, REPO_ROOT, log_path)
        self.restarts = 0
        self.dead = False
        self._since = 0
        self._halt = threading.Event()
        self._thread = None

    def start(self):
        self._since = self.process.log_size()
        self.process.start()
        if self._thread is None or not self._thread.is_alive():
            self._halt.clear()
            self._thread = threading.Thread(target=self._watch, name=f'watch-{self.name}', daemon=True)
            self._thread.start()

    def _watch(self):
        while not self._halt.is_set():
            time.sleep(0.5)
            if self.process.stopping or self._halt.is_set():
                continue
            if self.process.running():
                continue
            code = self.process.returncode
            self.restarts += 1
            if self.restarts > MAX_RESTARTS:
                self.dead = True
                return
            time.sleep(1.0)
            if self.process.stopping or self._halt.is_set():
                continue
            self._since = self.process.log_size()
            with open(self.process.log_path, 'ab') as handle:
                handle.write(f'=== respawn {self.restarts} after exit code {code} ===\n'.encode())
            self.process.start()

    def stop(self):
        self._halt.set()
        self.process.stop(10)
        if self._thread is not None:
            self._thread.join(3)

    def running(self):
        return self.process.running()


class QueueWorker(_Supervised):
    def __init__(self, queue, env, log_path, python=None):
        self.queue = queue
        argv = [python or sys.executable, '-u', '-m', 'taskqueue.worker', '--queue', queue]
        super().__init__(f'worker-{queue}', argv, env, log_path)

    def wait_ready(self, timeout=180):
        wait_until(
            lambda: self.process.log_contains(READY_MARKER, since=self._since),
            timeout, f'worker {self.queue} ready',
            dead=lambda: self.dead, detail=self.process.describe,
        )

    def listening(self, dsn):
        count = scalar(
            dsn,
            "SELECT count(*) FROM pg_stat_activity WHERE application_name LIKE %s",
            (f'%worker-{self.queue}%listen%',),
        )
        return bool(count)


class ControlListener(_Supervised):
    def __init__(self, env, log_path, python=None):
        argv = [python or sys.executable, '-u', '-m', 'taskqueue.control']
        super().__init__('control-listener', argv, env, log_path)

    def wait_ready(self, timeout=120):
        wait_until(
            lambda: self.process.log_contains(LISTENER_READY_MARKER, since=self._since),
            timeout, 'control listener ready',
            dead=lambda: self.dead, detail=self.process.describe,
        )
