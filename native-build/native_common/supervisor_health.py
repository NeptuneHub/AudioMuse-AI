# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The one health loop all three native supervisors run.

Restarting an unhealthy database and respawning an exited child is the same work
on every platform, so it lived as three copies that had already drifted. The
mixin reads _health_stop, _state, _desired, _lock, _children
and _log off the supervisor and calls its start_child and
_ensure_postgres_healthy.

The probe connection is held between ticks (a fresh probe every 5s would fork
~17k backends/day), but health is never inferred from its mere existence: every
tick runs a real SELECT 1. Reconnecting only on error keeps a momentarily
saturated max_connections from restarting a healthy database. The session is
autocommit so it never pins the xmin horizon.

Only unreachable-server errors are absorbed; a blanket except Exception
would turn any probe bug into an unhealthy verdict and restart a healthy
database in a loop. psycopg2 stays a deferred import so an unloadable libpq
cannot kill a supervisor before it can even open its log.

Main Features:
* One guarded loop: a stop request is honoured before every restart decision
* Restarts the embedded database first, then any child whose process exited
* A child that cannot be restarted is logged and retried on the next pass
"""

import threading

HEALTH_INTERVAL_SECONDS = 5
PROBE_CONNECT_TIMEOUT_SECONDS = 3
PROBE_KEEPALIVE_IDLE_SECONDS = 5
PROBE_KEEPALIVE_INTERVAL_SECONDS = 2
PROBE_KEEPALIVE_COUNT = 2


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

    def _probe_postgres(self, **conn_params):
        conn = self._held_probe_conn()
        if conn is not None:
            if self._run_probe(conn):
                return True
            self._close_probe_conn()
        conn = self._connect_probe(conn_params)
        if conn is None:
            return False
        if self._run_probe(conn):
            return True
        self._close_probe_conn()
        return False

    def _held_probe_conn(self):
        with self._lock:
            conn = getattr(self, '_probe_connection', None)
        if conn is None or conn.closed:
            return None
        return conn

    def _connect_probe(self, conn_params):
        import psycopg2

        conn = None
        try:
            conn = psycopg2.connect(
                connect_timeout=PROBE_CONNECT_TIMEOUT_SECONDS,
                keepalives=1,
                keepalives_idle=PROBE_KEEPALIVE_IDLE_SECONDS,
                keepalives_interval=PROBE_KEEPALIVE_INTERVAL_SECONDS,
                keepalives_count=PROBE_KEEPALIVE_COUNT,
                **conn_params,
            )
            conn.set_session(autocommit=True)
        except (psycopg2.Error, OSError):
            self._log.exception("Probe connection to embedded PostgreSQL failed")
            self._close_quietly(conn)
            return None
        except Exception:
            self._log.exception("Probe of embedded PostgreSQL hit a bug, not an unreachable server")
            self._close_quietly(conn)
            raise
        with self._lock:
            self._probe_connection = conn
        return conn

    def _run_probe(self, conn):
        import psycopg2

        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except (psycopg2.Error, OSError):
            return False
        except Exception:
            self._log.exception("Probe of embedded PostgreSQL hit a bug, not an unreachable server")
            raise

    def _close_quietly(self, conn):
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            self._log.debug("Probe connection close failed", exc_info=True)

    def _close_probe_conn(self):
        with self._lock:
            conn = getattr(self, '_probe_connection', None)
            self._probe_connection = None
        self._close_quietly(conn)

    def _start_health_loop(self):
        self._health_stop.clear()
        self._health_thread = threading.Thread(
            target=self._health_loop, name="health", daemon=True
        )
        self._health_thread.start()

    def _desired_snapshot(self):
        with self._lock:
            return tuple(self._desired)

    def _restart_if_exited(self, name):
        with self._lock:
            proc = self._children.get(name)
        if proc is None or proc.poll() is None:
            return
        self._log.warning("%s exited (code %s); restarting", name, proc.returncode)
        try:
            self.start_child(name)
        except Exception:
            self._log.exception("Could not restart %s; will retry", name)

    def _health_loop(self):
        try:
            while not self._health_stop.wait(HEALTH_INTERVAL_SECONDS):
                if self._state != "running":
                    continue
                self._ensure_postgres_healthy()
                if self._health_stop.is_set():
                    return
                for name in self._desired_snapshot():
                    if self._health_stop.is_set():
                        return
                    self._restart_if_exited(name)
        finally:
            self._close_probe_conn()
