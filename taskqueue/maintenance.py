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

Main Features:
* ``reclaim_orphans`` requeues or fails tasks whose worker is provably gone
* ``fail_stale_inline_rows`` finishes rows whose in-process owner died
* ``recover_migration_handshakes`` resumes a migration killed by its own restart
* ``clear_terminal_shared_payloads`` drops the payload a finished row still holds
* ``run_cycle`` runs reclaim always and the slow half only when it is due
"""

import logging
import os
import time

if os.environ.get('SERVICE_TYPE', '').lower() == 'worker':
    os.environ.setdefault('AUDIOMUSE_ROLE', 'worker')

import config  # noqa: E402
from . import sql  # noqa: E402

logger = logging.getLogger(__name__)


INLINE_STALE_SECONDS = config.QUEUE_INLINE_STALE_SECONDS

_FAIL_STALE_INLINE_ROWS = """
    UPDATE task_status
    SET status = 'FAIL', progress = 100, end_time = %s, timestamp = NOW(),
        details = %s
    WHERE func IS NULL
      AND status IN ('NEW','RUNNING')
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


_CLEAR_TERMINAL_SHARED = """
    UPDATE task_status
    SET shared_token = NULL, shared_payload = NULL
    WHERE (shared_token IS NOT NULL OR shared_payload IS NOT NULL)
      AND status IN ('SUCCESS','FAIL','REVOKED')
"""


def reclaim_orphans(conn, grace_seconds=None):
    with conn.cursor() as cur:
        candidates = sql.running_tasks(cur, grace_seconds=grace_seconds)
    conn.commit()
    reclaimed = []
    for candidate in candidates:
        task_id = candidate['task_id']
        outcome = None
        held = False
        try:
            with conn.cursor() as cur:
                if not sql.try_hold(cur, task_id):
                    logger.debug(
                        "Task %s still holds its advisory lock; its worker is alive", task_id
                    )
                    continue
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
        except Exception:
            conn.rollback()
            logger.exception("Could not reclaim %s; leaving it for the next pass", task_id)
            continue
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
        if outcome is None:
            continue
        reclaimed.append((task_id, outcome))
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
    import json

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
                now = time.monotonic()
                due = now - last_retention >= config.QUEUE_RETENTION_SCAN_SECONDS
                if run_cycle(conn, with_retention=due) and due:
                    last_retention = now
        except Exception:
            logger.exception("Maintenance cycle failed")
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    logger.debug("Maintenance connection close failed", exc_info=True)
            conn = None
        time.sleep(config.QUEUE_ORPHAN_SCAN_SECONDS)


if __name__ == '__main__':
    main()
