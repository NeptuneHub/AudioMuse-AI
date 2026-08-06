# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The one health loop all three native supervisors run.

Restarting an unhealthy database and respawning a child that exited is the same
work on every platform, so it lived as three copies that had already drifted:
macOS was missing both shutdown guards the other two had, which is exactly the
silent per-platform divergence ``service_roles`` was created to end. Only
``_ensure_postgres_healthy`` stays with the platform, because how the connection
parameters are discovered genuinely differs.

The mixin reads ``_health_stop``, ``_state``, ``_desired``, ``_lock``,
``_children`` and ``_log`` off the supervisor and calls its ``start_child`` and
``_ensure_postgres_healthy``; every ProcessSupervisor already defines all of them.

Main Features:
* One guarded loop: a stop request is honoured before every restart decision
* Restarts the embedded database first, then any child whose process exited
* A child that cannot be restarted is logged and retried on the next pass
"""

import threading

HEALTH_INTERVAL_SECONDS = 5


class HealthLoopMixin:
    def _claim_start(self, name):
        with self._lock:
            starting = getattr(self, '_starting_children', None)
            if starting is None:
                starting = self._starting_children = set()
            if name in starting:
                return False
            starting.add(name)
            return True

    def _release_start(self, name):
        with self._lock:
            getattr(self, '_starting_children', set()).discard(name)

    def _start_health_loop(self):
        self._health_stop.clear()
        self._health_thread = threading.Thread(
            target=self._health_loop, name="health", daemon=True
        )
        self._health_thread.start()

    def _health_loop(self):
        while not self._health_stop.wait(HEALTH_INTERVAL_SECONDS):
            if self._state != "running":
                continue
            self._ensure_postgres_healthy()
            if self._health_stop.is_set():
                return
            for name in list(self._desired):
                if self._health_stop.is_set():
                    return
                with self._lock:
                    proc = self._children.get(name)
                if proc is not None and proc.poll() is not None:
                    self._log.warning("%s exited (code %s); restarting", name, proc.returncode)
                    try:
                        self.start_child(name)
                    except Exception:
                        self._log.exception("Could not restart %s; will retry", name)
