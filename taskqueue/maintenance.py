# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Reclaims tasks whose worker died and finishes rows whose owner cannot.

Orphan detection needs no heartbeat or registry: a task is orphaned when its row
says RUNNING and nobody holds its advisory lock, which died with the worker's
connection. A task restarts ONLY because its worker died, at most
QUEUE_MAX_ATTEMPTS times, then fails for good.

Reclaim stands down while a control stop/restart is in flight:
taskqueue.control requeues those tasks itself without charging an attempt.
The stand-down is bounded by QUEUE_CONTROL_ACTION_WINDOW_SECONDS (how long the
action may take), and only actions that STOP workers suspend it. One listener's
FAIL does not end it, because SUCCESS is the only answer that means every
listener finished.

Runs in every worker container; a single pg_try_advisory_lock elects one
winner per cycle. Reclaim runs every few seconds; the slow half (stale inline
rows, migration handshakes, terminal shared payloads) only when due. After a
Postgres restart the first cycle is skipped so live workers can retake their
locks.

Main Features:
* reclaim_orphans requeues or fails RUNNING tasks whose advisory lock died
  with their worker, deferring to an in-flight control-plane action instead
* fail_stale_inline_rows finishes task rows left RUNNING by a web process
  that stopped, skipping any protected migration handshake task
* run_cycle elects one maintenance winner per pass and runs reclaim plus the
  slower retention sweeps only when they are due
* reclaim_blob_space VACUUMs what autovacuum cannot reach, because its threshold
  counts ROWS and a table of a few huge blobs never gets near it. Plain VACUUM
  only, so readers and writers are never blocked; it stands down while a task is
  live or another session holds an old transaction (a backup's pg_dump), and a
  lock_timeout means it never queues behind anyone. run_cycle does NOT call it
* start_blob_reclaim_thread runs that hourly sweep in the web process as a
  daemon thread on its own connection; a restore stops Flask before psql
  replaces the database, so the thread is already dead before the restore takes
  its ACCESS EXCLUSIVE locks, and a failed pass reconnects for the next one
"""

import json
import logging
import os
import threading
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


def reclaim_blob_space(conn):
    import psycopg2

    previous = conn.autocommit
    try:
        conn.autocommit = True
    except Exception:
        logger.exception(
            "Could not put the connection in autocommit; skipping this blob reclaim "
            "(VACUUM cannot run inside a transaction block)"
        )
        return []
    reclaimed = []
    try:
        with conn.cursor() as cur:
            sql.begin_reclaim_session(cur)
            if sql.any_live_task(cur):
                logger.info("Blob reclaim skipped: a task is live")
                return []
            if sql.snapshot_holder_blocking_reclaim(cur):
                logger.info(
                    "Blob reclaim skipped: another session has held a snapshot for "
                    "over %ss (a backup's pg_dump looks exactly like this). VACUUM could "
                    "not remove anything newer than that snapshot anyway.",
                    config.BLOB_RECLAIM_SNAPSHOT_GRACE_SECONDS,
                )
                return []
            targets = sql.blob_tables_autovacuum_cannot_reach(cur)
            if not targets:
                logger.info("Blob reclaim passed: found no tables with reclaimable dead rows.")
                return []
        for quoted_relname, dead, total in targets:
            try:
                with conn.cursor() as cur:
                    sql.vacuum_table(cur, quoted_relname)
            except (psycopg2.errors.LockNotAvailable, psycopg2.errors.QueryCanceled):
                logger.info(
                    "Blob reclaim left %s alone: it was busy, and this sweep never queues "
                    "behind another lock. The next pass picks it up.",
                    quoted_relname,
                )
                continue
            except Exception:
                logger.exception(
                    "VACUUM %s failed; sweeping the remaining tables anyway", quoted_relname
                )
                continue
            reclaimed.append(quoted_relname)
            logger.info(
                "VACUUM %s reclaimed %d dead row(s) in a %s table. Autovacuum fires on ROW "
                "count, so a table of a few huge blobs needs ~50 rebuilds to qualify and "
                "meanwhile every replaced blob stays on disk.",
                quoted_relname, dead, total,
            )
        if reclaimed:
            logger.info(
                "Blob reclaim passed: vacuumed %d table(s): %s",
                len(reclaimed), ", ".join(reclaimed),
            )
        else:
            logger.info(
                "Blob reclaim passed: %d candidate table(s) found but none were vacuumed.",
                len(targets),
            )
    except Exception:
        logger.exception("Blob-table space reclaim failed")
    finally:
        try:
            conn.autocommit = previous
        except Exception:
            logger.exception("Restoring the connection autocommit mode failed")
    return reclaimed


def _blob_reclaim_loop(application, connect_raw, sleep, reclaim):
    try:
        sleep(120)
        conn = None
        while True:
            try:
                if conn is None or conn.closed:
                    conn = connect_raw(
                        application_name=f"audiomuse-blob-reclaim-{os.getpid()}"
                    )
                reclaim(conn)
            except Exception:
                application.logger.exception('blob space reclaim cycle failed')
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        application.logger.exception(
                            'closing the blob reclaim connection failed'
                        )
                conn = None
            sleep(config.BLOB_RECLAIM_INTERVAL_SECONDS)
    except Exception:
        application.logger.exception('blob space reclaim main loop error')


def start_blob_reclaim_thread(application):
    def _loop():
        from time import sleep
        from database import connect_raw

        _blob_reclaim_loop(application, connect_raw, sleep, reclaim_blob_space)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread


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
