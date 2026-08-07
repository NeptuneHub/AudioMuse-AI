# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Asking every worker container to restart, stop, start, or pre-sync plugins.

Flask cannot reach a worker container directly, so it publishes a request and
waits for the workers to say they did it.

The request and its acknowledgements are ordinary ``task_status`` rows, so they
are durable for free and the existing retention rules clean them up. The
expected acknowledgement count comes from ``pg_stat_activity``: every listener
announces itself with an ``application_name``, so Flask counts the listeners
that are alive RIGHT NOW rather than trusting a subscriber count sampled when a
message went out. Delivery is therefore never fire-and-forget: the wait ends on
a real answer, which is what lets the setup wizard's
``worker_restart_acknowledged`` mean the workers actually restarted.

The listener is its own process, not a thread inside a worker, for one specific
reason: the ``stop`` action stops the workers, and something has to still be
listening afterwards to hear the ``start`` that brings them back.

The wait is a poll rather than a notification, because the answer is a row and
not a message: a listener that acknowledges between two checks is acknowledged
all the same, while a lost notification would mean a restart nobody could
confirm. Its cadence widens as the wait goes on - it starts at
``QUEUE_CONTROL_POLL_INTERVAL_SECONDS`` and doubles up to a tenth of the budget -
because a supervisor restart resolves in whole seconds while the interval is a
quarter of one, so the flat cadence spent about 120 round trips per restart
learning nothing. That fraction alone bought a blind tail proportional to the
budget, which is why ``POLL_INTERVAL_CEILING_SECONDS`` caps it as well: the
restore waits a whole action window, a tenth of which is fifteen seconds, so an
acknowledgement that landed at t=17s went unnoticed until t=30s - once on the
stop request and again on the start request, about half a minute added to every
restore. The cap holds that dead time to a few seconds whatever the budget and
still costs a small fraction of the flat loop. The deadline itself does not move:
each sleep is clamped to the time left, so the last count still happens exactly
when the budget runs out.

The ``service_roles.declare_worker_role()`` call above the imports is ordering,
not configuration; ``service_roles`` explains why, and why it stays conditional on
``SERVICE_TYPE`` here rather than forcing the role as a real queue entrypoint
does. It has to run before ``import config``, which is why the imports below it
carry ``noqa: E402``.

Two different budgets meet here and neither may stand in for the other.
``QUEUE_CONTROL_TIMEOUT_SECONDS`` is how long a CALLER waits for the answer, and a
caller that stops waiting changes nothing about the action: the request row is
left RUNNING so the late acknowledgements still land on it.
``QUEUE_CONTROL_ACTION_WINDOW_SECONDS`` is how long the ACTION itself may still be
running, so it is what bounds the exemption below - and it is the same number
``taskqueue.maintenance`` stands its reclaim down for, deliberately. An exemption
shorter than that stand-down lets a second publisher delete the request row the
stand-down is reading, which turns the guard off in the middle of the restart it
exists to cover.

A verdict answers the CALLER; it is not a report on the fleet. The wait ends on
the first FAIL because whoever asked has to hear about it at once, and the row it
then finishes is also the marker maintenance reads, so one listener's failure
used to retract the protection from every listener still working: two replicas,
one ``supervisorctl restart`` that returns non-zero AFTER it has already killed
its workers, and the other pod's tasks were charged a worker loss they never
suffered. The verdict is therefore still written the instant it is known - the
provider-migration handshake reads it to decide whether to rotate its request -
but only SUCCESS retracts the marker, because SUCCESS is the one answer that
means every listener has finished. The finish deliberately does not rewrite
``timestamp``: the marker expires one action window after the PUBLISH, and a
refusal must not push that instant out.

Only an action that STOPS workers suspends the reclaim, and
``WORKER_STOPPING_ACTIONS`` is the same tuple that decides which actions requeue
the stopped workers' tasks below - so reclaim stands down exactly where something
else has taken responsibility for those rows. A plugin pre-sync stops nothing and
used to suspend it anyway: a pre-sync outrunning the five-second advisory budget
left its row RUNNING, and a worker that genuinely died in the window that
followed was never reclaimed. The action travels in ``sub_type_identifier`` on
the request row, which is what maintenance filters on.

The statuses these statements write are interpolated from the
``config.TASK_STATUS_*`` constants, as ``taskqueue.sql`` and
``taskqueue.maintenance`` build theirs. The request row is the row maintenance
looks for before it stands its reclaim down, so a spelling hardcoded here and
renamed in config would leave the two halves of that handshake matching
different strings and silently charge a deliberate restart as a worker loss.

Main Features:
* ``publish_control_request`` asks, waits for every live listener, and returns a real verdict
* The acknowledgement wait polls on a widening but capped cadence and still ends on the deadline
* A refusal is recorded at once and still leaves the marker standing for the rest of the window
* ``control_listener`` performs the supervisor action and records its own acknowledgement
* Expected listener count read from ``pg_stat_activity``, needing no registry
* A new handshake deletes the previous request row and its acks, exactly as a
  new task run clears the last one. Nothing else ever removed them, so repeated
  wizard saves accumulated one set of rows per save. A handshake somebody is
  still waiting on is spared: publishers run in different processes - the wizard
  in Flask, the provider migration in a worker - so a second save used to delete
  the first request's rows out from under its poll loop, which then timed out
  and reported a restart nobody had failed to do. The exemption is bounded by
  ``QUEUE_CONTROL_ACTION_WINDOW_SECONDS``, because a request that times out is
  never moved off RUNNING and an unbounded exemption would leak one set of rows
  per abandoned handshake.
* A restart or stop the CONTROL PLANE performed requeues the stopped workers'
  tasks itself, without charging a worker-loss attempt - the same warm-shutdown
  contract rq's SIGTERM gave. A wizard save is a deliberate restart, not a
  crash, and letting it burn one of the three attempts meant three saves during
  a long analysis failed the run for good. The advisory-lock probe keeps this
  exact: a worker still alive on another container keeps its task, and a real
  crash still goes through the ordinary charged reclaim.
"""

import json
import logging
import os
import time
import uuid

import service_roles

service_roles.declare_worker_role()

import config  # noqa: E402
from . import sql  # noqa: E402
from .listen import Listener  # noqa: E402

logger = logging.getLogger(__name__)

ACTION_RESTART = 'restart'
ACTION_STOP = 'stop'
ACTION_START = 'start'
ACTION_PLUGIN_SYNC = 'plugin-sync'

VALID_ACTIONS = (ACTION_RESTART, ACTION_STOP, ACTION_START, ACTION_PLUGIN_SYNC)

WORKER_STOPPING_ACTIONS = (ACTION_RESTART, ACTION_STOP)

WORKER_LISTENER_PREFIX = 'audiomuse-control-worker-'

POLL_INTERVAL_CEILING_FRACTION = 0.1
POLL_INTERVAL_CEILING_SECONDS = 3.0

_RUNNING = config.TASK_STATUS_RUNNING

_COUNT_LISTENERS = """
    SELECT count(*) FROM pg_stat_activity
    WHERE application_name LIKE %s AND datname = current_database()
"""

_INSERT_REQUEST = f"""
    INSERT INTO task_status (task_id, parent_task_id, task_type, sub_type_identifier,
                             status, progress, details, timestamp, start_time)
    VALUES (%s, NULL, %s, %s, '{_RUNNING}', 0, %s, NOW(), %s)
    ON CONFLICT (task_id) DO UPDATE SET timestamp = NOW()
"""

_INSERT_ACK = """
    INSERT INTO task_status (task_id, parent_task_id, task_type, sub_type_identifier,
                             status, progress, details, timestamp, start_time, end_time)
    VALUES (%s, %s, %s, %s, %s, 100, %s, NOW(), %s, %s)
    ON CONFLICT (task_id) DO UPDATE SET status = EXCLUDED.status, details = EXCLUDED.details
"""

_CLEAR_PREVIOUS_CONTROL_ROWS = f"""
    DELETE FROM task_status AS t
    WHERE t.task_type = %s
      AND t.task_id <> %s
      AND (t.parent_task_id IS NULL OR t.parent_task_id <> %s)
      AND NOT EXISTS (
            SELECT 1 FROM task_status AS r
            WHERE r.task_id = COALESCE(t.parent_task_id, t.task_id)
              AND r.task_type = %s
              AND r.status = '{_RUNNING}'
              AND r.timestamp > NOW() - make_interval(secs => %s))
"""

_COUNT_ACKS = """
    SELECT status, count(*) FROM task_status
    WHERE parent_task_id = %s GROUP BY status
"""

_FINISH_REQUEST = """
    UPDATE task_status SET status = %s, progress = 100, end_time = %s
    WHERE task_id = %s
"""


def listener_id():
    configured = config.AUDIO_MUSE_LISTENER_ID
    return f"{configured}:{sql.hostname()}" if configured else sql.hostname()


def new_control_request_id():
    return f"control-{uuid.uuid4()}"


def _live_worker_listeners(cur):
    cur.execute(_COUNT_LISTENERS, (WORKER_LISTENER_PREFIX + '%',))
    return int(cur.fetchone()[0])


def publish_control_request(action, request_id=None, timeout_seconds=None, conn=None):
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unknown control action: {action}")
    request_id = request_id or new_control_request_id()
    timeout = float(
        timeout_seconds if timeout_seconds is not None else config.QUEUE_CONTROL_TIMEOUT_SECONDS
    )
    owns_connection = conn is None
    if owns_connection:
        from database import connect_raw

        conn = connect_raw(application_name=f"audiomuse-control-request-{sql.hostname()}")
    try:
        with conn.cursor() as cur:
            expected = _live_worker_listeners(cur)
            if expected <= 0:
                logger.error(
                    "No worker control listener is connected; %s request %s not sent.",
                    action, request_id,
                )
                return False
            now = time.time()
            cur.execute(
                _CLEAR_PREVIOUS_CONTROL_ROWS,
                (
                    sql.CONTROL_TASK_TYPE, request_id, request_id,
                    sql.CONTROL_TASK_TYPE, config.QUEUE_CONTROL_ACTION_WINDOW_SECONDS,
                ),
            )
            cur.execute(
                _INSERT_REQUEST,
                (
                    request_id,
                    sql.CONTROL_TASK_TYPE,
                    action,
                    json.dumps({'action': action, 'expected': expected}),
                    now,
                ),
            )
            sql.notify_control(cur, {'action': action, 'request_id': request_id})
        conn.commit()
        logger.info(
            "Published %s request %s to %d worker listener(s).", action, request_id, expected
        )
        acknowledged, refused = _await_acks(conn, request_id, expected, timeout)
        if acknowledged or refused:
            with conn.cursor() as cur:
                cur.execute(
                    _FINISH_REQUEST,
                    (
                        config.TASK_STATUS_SUCCESS if acknowledged else config.TASK_STATUS_FAIL,
                        time.time(),
                        request_id,
                    ),
                )
            conn.commit()
        return acknowledged
    except Exception:
        logger.exception("Could not publish or confirm %s request %s", action, request_id)
        try:
            conn.rollback()
        except Exception:
            logger.debug("Rollback after a failed control request failed", exc_info=True)
        return False
    finally:
        if owns_connection:
            try:
                conn.close()
            except Exception:
                logger.debug("Control request connection close failed", exc_info=True)


def poll_interval_ceiling(timeout):
    return max(
        config.QUEUE_CONTROL_POLL_INTERVAL_SECONDS,
        min(timeout * POLL_INTERVAL_CEILING_FRACTION, POLL_INTERVAL_CEILING_SECONDS),
    )


def _await_acks(conn, request_id, expected, timeout):
    deadline = time.monotonic() + timeout
    interval = config.QUEUE_CONTROL_POLL_INTERVAL_SECONDS
    ceiling = poll_interval_ceiling(timeout)
    while True:
        with conn.cursor() as cur:
            cur.execute(_COUNT_ACKS, (request_id,))
            counts = {row[0]: int(row[1]) for row in (cur.fetchall() or ())}
        conn.commit()
        succeeded = counts.get(config.TASK_STATUS_SUCCESS, 0)
        failed = counts.get(config.TASK_STATUS_FAIL, 0)
        if failed:
            logger.error(
                "%d listener(s) reported a failure for control request %s.", failed, request_id
            )
            return False, True
        if succeeded >= expected:
            return True, True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "Still waiting on control request %s after %.0fs: %d of %d listener(s) "
                "acknowledged. Leaving it open for the late acknowledgements.",
                request_id, timeout, succeeded, expected,
            )
            return False, False
        time.sleep(min(interval, remaining))
        interval = min(interval * 2, ceiling)


def get_control_request_result(request_id, conn=None):
    owns_connection = conn is None
    if owns_connection:
        from database import connect_raw

        conn = connect_raw()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM task_status WHERE task_id = %s", (request_id,))
            row = cur.fetchone()
        conn.commit()
        if row is None:
            return None
        if row[0] == config.TASK_STATUS_SUCCESS:
            return True
        if row[0] in (config.TASK_STATUS_FAIL, config.TASK_STATUS_REVOKED):
            return False
        return None
    finally:
        if owns_connection:
            try:
                conn.close()
            except Exception:
                logger.debug("Control result connection close failed", exc_info=True)


class ControlListener:
    def __init__(self):
        self.identity = WORKER_LISTENER_PREFIX + sql.hostname()
        self._listener = None
        self._lid = listener_id()

    def connect(self):
        from database import connect_raw

        return connect_raw(application_name=f"audiomuse-ctlack-{sql.hostname()}")

    def start(self):
        self._listener = Listener(
            (sql.CHANNEL_CONTROL,),
            self.on_notify,
            application_name=self.identity,
            name='control-listen',
        )
        self._listener.start()

    def on_notify(self, _channel, payload):
        try:
            message = json.loads(payload)
        except (TypeError, ValueError):
            logger.exception("Ignoring an unreadable control payload")
            return
        action = message.get('action')
        request_id = message.get('request_id')
        if action not in VALID_ACTIONS or not request_id:
            logger.error("Ignoring a control request with an unknown action")
            return
        conn = self._open_ack_conn()
        stored = self._already_acknowledged(conn, request_id) if conn is not None else None
        if stored == config.TASK_STATUS_SUCCESS:
            logger.info(
                "Control request %s already succeeded on this listener; re-recording "
                "its acknowledgement instead of running %s again.", request_id, action
            )
            self._close_ack_conn(self._record_ack(conn, request_id, action, True))
            return
        if stored is not None:
            logger.warning(
                "Control request %s previously FAILED on this listener; "
                "re-running %s.", request_id, action
            )
        logger.info("Control request %s received: %s", request_id, action)
        ok = self._execute(action)
        if ok and action in WORKER_STOPPING_ACTIONS and conn is not None:
            self._requeue_tasks_of_stopped_workers(conn)
        self._close_ack_conn(self._record_ack(conn, request_id, action, ok))

    def _open_ack_conn(self):
        try:
            return self.connect()
        except Exception:
            logger.exception("Could not open a connection for control ack handling")
            return None

    def _close_ack_conn(self, conn):
        if conn is not None:
            try:
                conn.close()
            except Exception:
                logger.debug("Control ack connection close failed", exc_info=True)

    def _already_acknowledged(self, conn, request_id):
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status FROM task_status WHERE task_id = %s",
                    (f"{request_id}:{self._lid}",),
                )
                row = cur.fetchone()
            conn.commit()
            return row[0] if row else None
        except Exception:
            logger.exception(
                "Could not check for an existing acknowledgement of %s; executing "
                "to be safe", request_id
            )
            try:
                conn.rollback()
            except Exception:
                logger.debug("Rollback after ack check failed", exc_info=True)
            return None

    def _requeue_tasks_of_stopped_workers(self, conn):
        try:
            with conn.cursor() as cur:
                candidates = sql.running_tasks(cur, grace_seconds=0)
            conn.commit()
            requeued = 0
            for candidate in candidates:
                task_id = candidate['task_id']
                held = False
                try:
                    with conn.cursor() as cur:
                        if not sql.try_hold(cur, task_id):
                            continue
                        held = True
                        if sql.requeue_uncharged(cur, task_id):
                            requeued += 1
                    conn.commit()
                finally:
                    if held:
                        with conn.cursor() as cur:
                            sql.release(cur, task_id)
                        conn.commit()
            if requeued:
                with conn.cursor() as cur:
                    sql.notify_job(cur, sql.QUEUE_HIGH)
                    sql.notify_job(cur, sql.QUEUE_DEFAULT)
                conn.commit()
                logger.info(
                    "Requeued %d task(s) whose workers this control action stopped; "
                    "no restart attempt was charged.", requeued,
                )
        except Exception:
            logger.exception(
                "Could not requeue the stopped workers' tasks; the ordinary reclaim "
                "will restart them and charge an attempt"
            )
            try:
                conn.rollback()
            except Exception:
                logger.debug("Rollback after the uncharged requeue failed", exc_info=True)

    def _execute(self, action):
        import restart_manager

        try:
            if action == ACTION_RESTART:
                return bool(restart_manager.restart_supervisor_workers())
            if action == ACTION_STOP:
                return bool(restart_manager.stop_supervisor_workers())
            if action == ACTION_START:
                return bool(restart_manager.start_supervisor_workers())
            return self._dispatch_plugin_sync()
        except Exception:
            logger.exception("Control action %s failed", action)
            return False

    def _dispatch_plugin_sync(self):
        try:
            from plugin.manager import worker_presync
        except Exception:
            logger.exception("plugin-sync received but the plugin subsystem is unavailable")
            return False

        try:
            worker_presync()
        except Exception:
            logger.exception("Plugin pre-sync failed")
            return False
        return True

    def _discard_ack_conn(self, conn):
        try:
            conn.rollback()
        except Exception:
            logger.debug("Rollback before the ack retry failed", exc_info=True)
        self._close_ack_conn(conn)

    def _record_ack(self, conn, request_id, action, ok):
        status = config.TASK_STATUS_SUCCESS if ok else config.TASK_STATUS_FAIL
        now = time.time()
        params = (
            f"{request_id}:{self._lid}",
            request_id,
            sql.CONTROL_TASK_TYPE,
            self._lid,
            status,
            json.dumps({'action': action, 'listener': self._lid}),
            now,
            now,
        )
        for attempt in (1, 2):
            if conn is None:
                conn = self._open_ack_conn()
            if conn is None:
                continue
            try:
                with conn.cursor() as cur:
                    cur.execute(_INSERT_ACK, params)
                conn.commit()
                return conn
            except Exception:
                self._discard_ack_conn(conn)
                conn = None
                if attempt == 1:
                    logger.warning(
                        "Could not record the acknowledgement for %s; retrying once "
                        "on a fresh connection", request_id, exc_info=True,
                    )
                    continue
                logger.exception("Could not record the acknowledgement for %s", request_id)
                return conn
        logger.error("No connection to record acknowledgement for %s", request_id)
        return conn


def main():
    from app_logging import configure_logging

    configure_logging()
    service_type = os.environ.get('SERVICE_TYPE', '').lower()
    if service_type != 'worker':
        logger.info(
            "Control listener idle: SERVICE_TYPE=%s has no supervisor to drive.", service_type
        )
        while True:
            time.sleep(3600)
    listener = ControlListener()
    listener.start()
    logger.info("Control listener ready as %s", listener.identity)
    while True:
        time.sleep(3600)


if __name__ == '__main__':
    main()
