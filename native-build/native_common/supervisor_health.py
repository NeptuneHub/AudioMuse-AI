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

The probe connection is held between ticks, the way ``taskqueue.listen`` holds
its LISTEN session, because a probe every five seconds is roughly seventeen
thousand backend forks a day on an install nobody is touching. Health is still
never inferred from the mere existence of that session: every tick runs a real
``SELECT 1`` round trip on it, and a probe that errors drops the session and
retries once on a fresh connection before the database is declared unhealthy.
Reconnecting only on error is also what keeps a momentarily saturated
``max_connections`` from restarting a database that is answering queries fine.
The session is autocommit so it can never sit idle in a transaction holding the
xmin horizon, and it is closed when the loop exits so a shutdown or a restore
never waits on it.

Opening and running that probe only ever absorbs the database and OS errors that
genuinely mean the server is unreachable. A blanket ``except Exception`` here is
a trap rather than caution: it turns any coding mistake in the probe into an
unhealthy verdict, so the supervisor restarts a perfectly healthy database in a
loop with nothing in the log naming the real cause, which is exactly how the
autocommit call silently disabled the whole probe once already. A real connection
failure is logged with its traceback and reported as unreachable; anything else
is logged as the bug it is by whichever helper hit it and only then re-raised,
because every platform ``_ensure_postgres_healthy`` absorbs that re-raise without
logging it - it has to, to keep the health thread and therefore child restarts
alive - so that log line is the only thing left naming the cause.

``psycopg2`` stays a deferred import inside the two probe helpers rather than
moving to the top of this module, which is the repository default. This mixin is
imported at module scope by all three frozen supervisors, and the supervisor is
the process that prepares the environment libpq is later loaded with: the Windows
child environment is the only place the embedded server's ``bin`` directory is
prepended to ``PATH``, and the Linux one swaps ``LD_LIBRARY_PATH`` back to the
system loader around every native load. A module-scope import would hoist that
native load into the supervisor's own import, turning a libpq that will not load
into a supervisor that dies before it can open its log file or start the
database, instead of one probe that fails and says so. Both the Windows
supervisor and its ``db_backend`` already defer psycopg2 for the same reason.

Main Features:
* One guarded loop: a stop request is honoured before every restart decision
* Restarts the embedded database first, then any child whose process exited
* A child that cannot be restarted is logged and retried on the next pass
* One held probe connection per supervisor, reconnected only when a probe fails
* Only unreachable-server errors are absorbed; a bug in the probe surfaces loudly
* psycopg2 is imported inside the probe so an unloadable libpq cannot kill boot
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
