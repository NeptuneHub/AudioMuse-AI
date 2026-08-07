# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Restart tasks whose worker died, and finish rows whose owner cannot.

This module does NOT keep ``task_status`` small; nothing here needs to. A run
that starts empties the table, a Cancel empties it, and a run that finishes
empties it apart from its own one-line recap, so the table already holds either
the run happening now or the recap of the last one. There is no prune, no cap,
no age and no ranking anywhere in the queue.

``reclaim_orphans`` needs no heartbeat, no staleness threshold and no registry
bookkeeping: a task is orphaned when its row says RUNNING and nobody holds its
advisory lock, which Postgres answers exactly and instantly, because the lock
died with the worker's connection.

The rule it enforces is deliberately the only retry rule in the system: a task
restarts ONLY because its worker died, at most ``QUEUE_MAX_ATTEMPTS`` times, and
then fails for good. A task that raised on its own merits was already written to
FAIL by the worker and never appears here.

A worker the CONTROL PLANE stopped did not die, so reclaim stands down entirely
while a control request row is unfinished: ``taskqueue.control`` requeues those
tasks itself without charging an attempt. This has to be checked here rather
than at the one caller, because the reclaimer that would otherwise win the race
is the freshly booted worker's own grace-0 pass, which runs before the control
listener has finished restarting the fleet - on native builds the services are
restarted one at a time, so the first worker is already reclaiming while the
last is still being terminated. Three wizard saves during a long analysis failed
the run for good that way.

Unfinished means anything but SUCCESS, and the distinction is the whole point:
SUCCESS is written only once EVERY live listener has acknowledged, so it is the
one answer that means nobody is still stopping anything. A verdict of FAIL is one
listener's answer, not the fleet's - a pod whose ``supervisorctl`` returns
non-zero after it has already killed its workers answers FAIL at t=30s while its
neighbour is legitimately still stopping its three services - and reading the
marker as gone at that point charged the neighbour's rows exactly the worker-loss
attempt this window exists to prevent.

The stand-down covers only the actions that STOP workers, which is
``taskqueue.control.WORKER_STOPPING_ACTIONS`` and therefore precisely the actions
whose listener performs the uncharged requeue: standing reclaim down means
deferring to that requeue, so an action which never performs one has nothing to
defer to. A plugin pre-sync stops no worker, and suspending reclaim for it meant
a pre-sync slower than the caller's five-second budget left a row nobody would
ever finish and a worker that really died in the following window went
unreclaimed. The published action is read from the request row's
``sub_type_identifier``; a row that does not name its action at all is read as
one that stops workers, because the expensive mistake is charging an attempt that
was never a loss.

That stand-down is measured against ``QUEUE_CONTROL_ACTION_WINDOW_SECONDS``,
which is how long the ACTION may legitimately take, and NOT against
``QUEUE_CONTROL_TIMEOUT_SECONDS``, which is only how long a caller waits to hear
about it. It was the ack-wait budget once, and that is the whole bug: the request
row's timestamp is written at publish and never refreshed, a caller gives up
after 30s (5s for the wizard) while a three-worker stop legitimately takes 45-60,
and the first worker back then found the guard already expired and charged every
still-restarting worker's row an attempt. The uncharged requeue that arrives
seconds later cannot undo it, because those rows are no longer RUNNING.

The same window bounds the guard the other way: because the timestamp is never
refreshed - not by the action running, and not by the verdict that finishes the
row - a control row an abandoned, crashed or refused handshake left unfinished
suspends reclaim for exactly one action window and not one second longer.

This process runs in every worker container, and one ``pg_try_advisory_lock``
elects a single winner per cycle. Nothing here mutates state a Cancel could be
racing, so the lock needs no blocking or timeout variants: it is only about not
doing the same tidy-up N times.

The cycle is split in two only because of how often each half is worth running.
Reclaim is one indexed scan plus one ``pg_try_advisory_lock`` per RUNNING
candidate, so it runs every few seconds and a dead worker is noticed in about
that long. The slow half exists for rows no worker owns - an inline task whose
web process died, a migration killed by its own restart - and for the shared
payload a terminal row would otherwise keep alive.

The one case the advisory lock cannot answer is a Postgres restart: it frees
every lock at once. The loop answers it directly - losing its own connection is
exactly the same event, so the first cycle after any reconnect is skipped, which
is more than the couple of seconds a live worker's listener needs to come back
and retake its lock through ``Worker.ensure_hold``.

The ``service_roles.declare_worker_role()`` call above the imports is ordering,
not configuration; ``service_roles`` explains why. It has to run before ``import
config``, which is why the imports below it carry ``noqa: E402``.

Every status these statements write or compare is interpolated from the
``config.TASK_STATUS_*`` constants, exactly as ``taskqueue.sql`` builds its own,
so renaming one in config moves the stale-row sweep, the shared-payload wipe and
the control-in-flight guard with it. Hardcoding a spelling in even one of them is
silent: the Python comparisons in this module follow config, the statement does
not, and the sweep simply stops matching the rows it exists to finish.

Main Features:
* ``reclaim_orphans`` requeues or fails tasks whose worker is provably gone
* ``reclaim_orphans`` charges nothing while a control stop or restart is in flight
* One listener's refusal does not end that stand-down for the listeners still working
* An action that stops no worker never suspends reclaim at all
* ``fail_stale_inline_rows`` finishes rows whose in-process owner died
* ``recover_migration_handshakes`` resumes a migration killed by its own restart
* ``clear_terminal_shared_payloads`` drops the payload a finished row still holds
* ``run_cycle`` runs reclaim always and the slow half only when it is due
"""

import json
import logging
import time

import service_roles

service_roles.declare_worker_role()

import config  # noqa: E402
from . import control  # noqa: E402
from . import sql  # noqa: E402

logger = logging.getLogger(__name__)


INLINE_STALE_SECONDS = config.QUEUE_INLINE_STALE_SECONDS

_SUCCESS = config.TASK_STATUS_SUCCESS
_FAIL = config.TASK_STATUS_FAIL

_LIVE_IN_LIST = ','.join(f"'{status}'" for status in config.TASK_STATUS_LIVE)
_TERMINAL_IN_LIST = ','.join(f"'{status}'" for status in config.TASK_STATUS_TERMINAL)

_FAIL_STALE_INLINE_ROWS = f"""
    UPDATE task_status
    SET status = '{_FAIL}', progress = 100, end_time = %s, timestamp = NOW(),
        details = %s
    WHERE func IS NULL
      AND status IN ({_LIVE_IN_LIST})
      AND timestamp < NOW() - make_interval(secs => %s)
      AND NOT (task_status.task_id = ANY(%s))
    RETURNING task_id
"""

_PROTECTED_MIGRATION_TASKS = """
    SELECT ms.state->>'exec_task_id', ms.state->>'alignment_task_id',
           ms.state->>'restart_request_id'
    FROM migration_session AS ms
    WHERE ms.status = 'completed'
      AND lower(COALESCE(ms.state->>'restart_acknowledged', 'false'))
          NOT IN ('true', '1', 'yes')
"""


_CLEAR_TERMINAL_SHARED = f"""
    UPDATE task_status
    SET shared_token = NULL, shared_payload = NULL
    WHERE (shared_token IS NOT NULL OR shared_payload IS NOT NULL)
      AND status IN ({_TERMINAL_IN_LIST})
"""


_CONTROL_ACTION_IN_FLIGHT = f"""
    SELECT 1 FROM task_status
    WHERE task_type = %s
      AND parent_task_id IS NULL
      AND status <> '{_SUCCESS}'
      AND (sub_type_identifier IS NULL OR sub_type_identifier = ANY(%s))
      AND timestamp > NOW() - make_interval(secs => %s)
    LIMIT 1
"""


def _control_action_in_flight(cur):
    cur.execute(
        _CONTROL_ACTION_IN_FLIGHT,
        (
            sql.CONTROL_TASK_TYPE,
            list(control.WORKER_STOPPING_ACTIONS),
            config.QUEUE_CONTROL_ACTION_WINDOW_SECONDS,
        ),
    )
    return cur.fetchone() is not None


def _reclaim_one(conn, task_id):
    held = False
    try:
        with conn.cursor() as cur:
            if not sql.try_hold(cur, task_id):
                logger.debug(
                    "Task %s still holds its advisory lock; its worker is alive", task_id
                )
                return None
            held = True
            outcome = sql.requeue_or_fail(
                cur,
                task_id,
                time.time(),
                {
                    'message': (
                        "The worker running this task stopped unexpectedly. "
                        "It was restarted the allowed number of times."
                    ),
                    'error': 'worker lost',
                },
            )
        conn.commit()
        return outcome
    except Exception:
        conn.rollback()
        logger.exception("Could not reclaim %s; leaving it for the next pass", task_id)
        return None
    finally:
        if held:
            try:
                with conn.cursor() as cur:
                    sql.release(cur, task_id)
                conn.commit()
            except Exception:
                logger.debug(
                    "Could not release the reclaim probe on %s", task_id, exc_info=True
                )


def _log_reclaimed(task_id, outcome, candidate):
    if outcome == config.TASK_STATUS_NEW:
        logger.warning(
            "Task %s lost its worker; requeued (worker loss %d of an allowed %d).",
            task_id, candidate['attempts'] + 1, candidate['max_attempts'],
        )
    else:
        logger.error(
            "Task %s lost its worker %d time(s), more than the %d allowed; failed.",
            task_id, candidate['attempts'] + 1, candidate['max_attempts'],
        )


def reclaim_orphans(conn, grace_seconds=None):
    with conn.cursor() as cur:
        deferred = _control_action_in_flight(cur)
        candidates = (
            [] if deferred else sql.running_tasks(cur, grace_seconds=grace_seconds)
        )
    conn.commit()
    if deferred:
        logger.info(
            "A control-plane action is in flight; leaving the RUNNING rows to its "
            "uncharged requeue instead of charging a worker-loss attempt."
        )
        return []
    reclaimed = []
    for candidate in candidates:
        task_id = candidate['task_id']
        outcome = _reclaim_one(conn, task_id)
        if outcome is None:
            continue
        reclaimed.append((task_id, outcome))
        _log_reclaimed(task_id, outcome, candidate)
    if reclaimed:
        with conn.cursor() as cur:
            sql.notify_job(cur, sql.QUEUE_HIGH)
            sql.notify_job(cur, sql.QUEUE_DEFAULT)
        conn.commit()
    return reclaimed


def _protected_migration_task_ids(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('migration_session')")
            if cur.fetchone()[0] is None:
                return []
            cur.execute(_PROTECTED_MIGRATION_TASKS)
            protected = [
                task_id for row in (cur.fetchall() or ()) for task_id in row if task_id
            ]
        conn.commit()
        return protected
    except Exception:
        logger.exception(
            "Could not read the migration protection set; skipping the stale-row "
            "sweep entirely this cycle rather than failing the rows it protects"
        )
        try:
            conn.rollback()
        except Exception:
            logger.debug("Rollback after a failed protection read failed", exc_info=True)
        return None


def fail_stale_inline_rows(conn):
    details = json.dumps({
        'message': (
            "This task ran inside the web process and that process stopped "
            "before it finished."
        ),
        'error': 'inline task interrupted',
    })
    protected = _protected_migration_task_ids(conn)
    if protected is None:
        return []
    with conn.cursor() as cur:
        cur.execute(
            _FAIL_STALE_INLINE_ROWS,
            (time.time(), details, INLINE_STALE_SECONDS, protected),
        )
        failed = [row[0] for row in (cur.fetchall() or ())]
    conn.commit()
    if failed:
        logger.warning("Failed %d stale in-process task row(s): %s", len(failed), failed)
    return failed


def recover_migration_handshakes():
    try:
        from tasks.provider_migration_tasks import (
            recover_provider_migration_restart_handshakes,
        )

        return recover_provider_migration_restart_handshakes()
    except Exception:
        logger.exception("Provider-migration handshake recovery failed")
        return 0


def clear_terminal_shared_payloads(conn):
    with conn.cursor() as cur:
        cur.execute(_CLEAR_TERMINAL_SHARED)
        cleared = cur.rowcount
    conn.commit()
    if cleared:
        logger.info(
            "Cleared the shared payload left on %d terminal task row(s).", cleared
        )
    return cleared


def run_cycle(conn, with_retention=True):
    with conn.cursor() as cur:
        elected = sql.try_maintenance_lock(cur)
    conn.commit()
    if not elected:
        return False
    try:
        reclaim_orphans(conn)
        if with_retention:
            fail_stale_inline_rows(conn)
            recover_migration_handshakes()
            clear_terminal_shared_payloads(conn)
    finally:
        try:
            conn.rollback()
        except Exception:
            logger.debug("Rollback before the election release failed", exc_info=True)
        with conn.cursor() as cur:
            sql.release_maintenance_lock(cur)
        conn.commit()
    return True


def _drop_connection(conn):
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        logger.debug("Maintenance connection close failed", exc_info=True)


def _run_due_cycle(conn, last_retention):
    now = time.monotonic()
    due = now - last_retention >= config.QUEUE_RETENTION_SCAN_SECONDS
    if run_cycle(conn, with_retention=due) and due:
        return now
    return last_retention


def main():
    from app_logging import configure_logging

    configure_logging()
    from database import connect_raw

    identity = f"audiomuse-maintenance-{sql.hostname()}"
    logger.info(
        "Maintenance starting as %s (reclaim every %ss, retention every %ss)",
        identity, config.QUEUE_ORPHAN_SCAN_SECONDS, config.QUEUE_RETENTION_SCAN_SECONDS,
    )
    conn = None
    last_retention = float('-inf')
    settling = False
    while True:
        try:
            if conn is None or conn.closed:
                conn = connect_raw(
                    application_name=identity,
                    keepalive_idle_seconds=config.QUEUE_KEEPALIVE_IDLE_SECONDS,
                    keepalive_interval_seconds=config.QUEUE_KEEPALIVE_INTERVAL_SECONDS,
                    keepalive_count=config.QUEUE_KEEPALIVE_COUNT,
                )
                settling = True
            if settling:
                settling = False
                logger.info(
                    "Maintenance (re)connected; skipping one cycle so live workers can "
                    "retake the advisory locks a Postgres outage would have freed."
                )
            else:
                last_retention = _run_due_cycle(conn, last_retention)
        except Exception:
            logger.exception("Maintenance cycle failed")
            _drop_connection(conn)
            conn = None
        time.sleep(config.QUEUE_ORPHAN_SCAN_SECONDS)


if __name__ == '__main__':
    main()
