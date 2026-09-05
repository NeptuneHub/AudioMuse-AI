# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The queue driven against a real Postgres, because its contract IS Postgres.

Every guarantee here is a database guarantee and none of it can be proved with a
mocked cursor: ``FOR UPDATE SKIP LOCKED`` only produces one winner under genuine
concurrency, a session advisory lock only dies with a real connection, and a
partial unique index only rejects the second row when Postgres evaluates it.

Main Features:
* Two connections racing one job: exactly one claims it, the other gets nothing
* A running task's advisory lock blocks reclaim; closing its connection frees it
* Reclaim requeues while attempts remain and FAILs on the last one
* A second live main task is refused by the unique index, and a sweep still fits
* A cancelled row cannot be claimed, and a claimed row cannot be claimed twice
* A finished root keeps one recap row with no func and no payload
* A fan-out stores its shared body once, counted on the driver, not on a stand-in
* The backlog counts only NEW rows and drops a row the moment it is claimed
* The worker table lists every worker that has a live connection, claim or
  listen, from every container, plus a RUNNING row the queue still believes;
  only a RUNNING row bound to the worker makes it busy
"""

import json
import os
import select
import sys
import threading
import time

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
except Exception:  # pragma: no cover - psycopg2 is in test/requirements.txt
    psycopg2 = None

pytestmark = pytest.mark.integration

import config  # noqa: E402
from taskqueue import sql  # noqa: E402


_BASE_DDL = """
    CREATE TABLE task_status (
        id SERIAL PRIMARY KEY,
        task_id TEXT UNIQUE NOT NULL,
        parent_task_id TEXT,
        task_type TEXT,
        sub_type_identifier TEXT,
        status TEXT,
        progress INTEGER DEFAULT 0,
        details TEXT,
        timestamp TIMESTAMP DEFAULT NOW(),
        start_time DOUBLE PRECISION,
        end_time DOUBLE PRECISION
    )
"""


@pytest.fixture
def queue_db(shared_pg_dsn):
    conn = psycopg2.connect(shared_pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
        cur.execute(_BASE_DDL)
    conn.autocommit = False
    with conn.cursor() as cur:
        sql.ensure_schema(cur)
    conn.commit()
    try:
        yield conn
    finally:
        try:
            conn.rollback()
            conn.cursor_factory = None
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
        finally:
            conn.close()


def _fresh(shared_pg_dsn, application_name='audiomuse-test'):
    conn = psycopg2.connect(shared_pg_dsn, application_name=application_name)
    return conn


def _enqueue(conn, task_id, task_type='main_analysis', queue=sql.QUEUE_DEFAULT,
             parent_task_id=None, max_attempts=None):
    with conn.cursor() as cur:
        inserted = sql.insert_job(
            cur,
            task_id=task_id,
            task_type=task_type,
            func='tasks.analysis.run_analysis_task',
            args=(),
            kwargs=None,
            queue=queue,
            priority=0,
            parent_task_id=parent_task_id,
            sub_type_identifier=None,
            max_attempts=max_attempts,
            details={'message': 'queued'},
        )
    conn.commit()
    return inserted


def _row(conn, task_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status, attempts, max_attempts, func, payload, worker_id "
            "FROM task_status WHERE task_id = %s",
            (task_id,),
        )
        return cur.fetchone()


def _age_row(conn, task_id, seconds=600):
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE task_status SET timestamp = NOW() - make_interval(secs => %s) "
            "WHERE task_id = %s",
            (seconds, task_id),
        )
    conn.commit()


NOTIFY_TIMEOUT_SECONDS = 15.0


def _bare_worker(shared_pg_dsn):
    from taskqueue.worker import Worker

    worker = Worker.__new__(Worker)
    worker._shared_cache = {}
    worker._claim_txn = threading.Lock()
    worker._conn = _fresh(shared_pg_dsn)
    return worker


def _count_statement(conn, statement):
    executions = []
    base = conn.cursor_factory or psycopg2.extensions.cursor

    class _CountingCursor(base):
        def execute(self, query, params=None):
            if query == statement:
                executions.append(params)
            return super().execute(query, params)

    conn.cursor_factory = _CountingCursor
    return executions


class TestOnlyOneWorkerEverClaimsAJob:
    def test_two_connections_racing_one_job_produce_exactly_one_winner(
        self, queue_db, shared_pg_dsn
    ):
        _enqueue(queue_db, 'race-1')
        first = _fresh(shared_pg_dsn, 'audiomuse-worker-default-a-1')
        second = _fresh(shared_pg_dsn, 'audiomuse-worker-default-b-2')
        try:
            with first.cursor() as cur:
                claimed_a = sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='a')
            with second.cursor() as cur:
                claimed_b = sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='b')
            first.commit()
            second.commit()
        finally:
            first.close()
            second.close()

        assert (claimed_a is None) != (claimed_b is None)

    def test_a_claimed_job_cannot_be_claimed_again(self, queue_db, shared_pg_dsn):
        _enqueue(queue_db, 'once-1')
        conn = _fresh(shared_pg_dsn)
        try:
            with conn.cursor() as cur:
                assert sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='a')
            conn.commit()
            with conn.cursor() as cur:
                assert sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='a') is None
            conn.commit()
        finally:
            conn.close()

    def test_a_cancelled_row_is_never_claimable(self, queue_db, shared_pg_dsn):
        _enqueue(queue_db, 'gone-1')
        with queue_db.cursor() as cur:
            cur.execute(
                "UPDATE task_status SET status = %s WHERE task_id = 'gone-1'",
                (config.TASK_STATUS_REVOKED,),
            )
        queue_db.commit()
        conn = _fresh(shared_pg_dsn)
        try:
            with conn.cursor() as cur:
                assert sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='a') is None
            conn.commit()
        finally:
            conn.close()

    def test_the_high_queue_is_not_drained_by_a_default_worker(
        self, queue_db, shared_pg_dsn
    ):
        _enqueue(queue_db, 'high-1', queue=sql.QUEUE_HIGH)
        conn = _fresh(shared_pg_dsn)
        try:
            with conn.cursor() as cur:
                assert sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='a') is None
                assert sql.claim(cur, sql.QUEUE_HIGH, time.time(), worker_id='a')
            conn.commit()
        finally:
            conn.close()


class TestTheAdvisoryLockIsTheLiveness:
    def test_a_live_holder_blocks_the_reclaim_probe(self, queue_db, shared_pg_dsn):
        _enqueue(queue_db, 'held-1')
        holder = _fresh(shared_pg_dsn)
        prober = _fresh(shared_pg_dsn)
        try:
            with holder.cursor() as cur:
                sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='holder')
                sql.hold(cur, 'held-1')
            holder.commit()

            with prober.cursor() as cur:
                assert sql.try_hold(cur, 'held-1') is False
            prober.commit()
        finally:
            holder.close()
            prober.close()

    def test_closing_the_holders_connection_frees_the_lock_instantly(
        self, queue_db, shared_pg_dsn
    ):
        _enqueue(queue_db, 'dead-1')
        holder = _fresh(shared_pg_dsn)
        with holder.cursor() as cur:
            sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='holder')
            sql.hold(cur, 'dead-1')
        holder.commit()
        holder.close()

        prober = _fresh(shared_pg_dsn)
        try:
            with prober.cursor() as cur:
                assert sql.try_hold(cur, 'dead-1') is True
                sql.release(cur, 'dead-1')
            prober.commit()
        finally:
            prober.close()


class TestReclaimRestartsThreeTimesThenStops:
    def test_a_lost_worker_requeues_while_attempts_remain(
        self, queue_db, shared_pg_dsn
    ):
        from taskqueue import maintenance

        _enqueue(queue_db, 'retry-1', max_attempts=3)
        dead = _fresh(shared_pg_dsn)
        with dead.cursor() as cur:
            sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='dead')
            sql.hold(cur, 'retry-1')
        dead.commit()
        dead.close()
        _age_row(queue_db, 'retry-1')

        conn = _fresh(shared_pg_dsn)
        try:
            reclaimed = maintenance.reclaim_orphans(conn)
        finally:
            conn.close()

        assert reclaimed == [('retry-1', config.TASK_STATUS_NEW)]
        status, attempts, _max_attempts, func, _payload, _worker = _row(queue_db, 'retry-1')
        assert status == config.TASK_STATUS_NEW
        assert attempts == 1
        assert func, 'a requeued job must keep the func it will be run with'

    def test_the_final_attempt_fails_the_task_for_good(self, queue_db, shared_pg_dsn):
        from taskqueue import maintenance

        _enqueue(queue_db, 'giveup-1', max_attempts=0)
        dead = _fresh(shared_pg_dsn)
        with dead.cursor() as cur:
            sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='dead')
            sql.hold(cur, 'giveup-1')
        dead.commit()
        dead.close()
        _age_row(queue_db, 'giveup-1')

        conn = _fresh(shared_pg_dsn)
        try:
            reclaimed = maintenance.reclaim_orphans(conn)
        finally:
            conn.close()

        assert reclaimed == [('giveup-1', config.TASK_STATUS_FAIL)]
        status, _attempts, _max, _func, _payload, _worker = _row(queue_db, 'giveup-1')
        assert status == config.TASK_STATUS_FAIL

    def test_a_task_a_live_worker_is_running_is_left_alone(
        self, queue_db, shared_pg_dsn
    ):
        from taskqueue import maintenance

        _enqueue(queue_db, 'alive-1')
        holder = _fresh(shared_pg_dsn)
        conn = _fresh(shared_pg_dsn)
        try:
            with holder.cursor() as cur:
                sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='holder')
                sql.hold(cur, 'alive-1')
            holder.commit()

            assert maintenance.reclaim_orphans(conn) == []
        finally:
            holder.close()
            conn.close()

        status, attempts, _max, _func, _payload, _worker = _row(queue_db, 'alive-1')
        assert status == config.TASK_STATUS_RUNNING
        assert attempts == 0, 'attempts counts worker deaths, and this task has lost none'


class TestAdmissionIsAUniqueIndexNotALock:
    def test_a_second_live_main_task_is_refused_by_postgres(self, queue_db):
        assert _enqueue(queue_db, 'main-1', task_type='main_analysis')

        with pytest.raises(psycopg2.errors.UniqueViolation):
            _enqueue(queue_db, 'main-2', task_type='main_clustering')
        queue_db.rollback()

    def test_upgrade_retires_conflicting_live_roots_before_recreating_the_index(
        self, queue_db
    ):
        # Simulate a pre-upgrade state: no one-live-main index, and a live batch
        # root alongside a live sonic-fingerprint root (the gap this feature
        # closes). ensure_schema must retire every live main root but the newest
        # before it can build the unique index, or the CREATE would fail.
        with queue_db.cursor() as cur:
            cur.execute(f"DROP INDEX IF EXISTS {sql.MAIN_INDEX_NAME}")
        queue_db.commit()

        for task_id, task_type in (
            ('analysis-live', 'main_analysis'),
            ('fingerprint-live', 'sonic_fingerprint'),
        ):
            with queue_db.cursor() as cur:
                cur.execute(
                    "INSERT INTO task_status (task_id, task_type, status) "
                    "VALUES (%s, %s, %s)",
                    (task_id, task_type, config.TASK_STATUS_RUNNING),
                )
            queue_db.commit()

        with queue_db.cursor() as cur:
            sql.ensure_schema(cur)
        queue_db.commit()

        with queue_db.cursor() as cur:
            cur.execute(
                "SELECT task_id FROM task_status "
                "WHERE parent_task_id IS NULL AND status = %s",
                (config.TASK_STATUS_RUNNING,),
            )
            live = cur.fetchall()
        assert len(live) == 1, (
            'the upgrade retire must leave exactly one live main root'
        )

        with pytest.raises(psycopg2.errors.UniqueViolation):
            _enqueue(queue_db, 'main-3', task_type='main_clustering')
        queue_db.rollback()

    def test_a_sweep_and_a_main_task_coexist(self, queue_db):
        assert _enqueue(queue_db, 'main-1', task_type='main_analysis')
        assert _enqueue(queue_db, 'sweep-1', task_type='server_sweep')

    def test_a_child_never_counts_against_the_main_task_rule(self, queue_db):
        assert _enqueue(queue_db, 'main-1', task_type='main_analysis')
        assert _enqueue(
            queue_db, 'child-1', task_type='album_analysis', parent_task_id='main-1'
        )

    def test_a_finished_main_task_lets_the_next_one_in(self, queue_db):
        assert _enqueue(queue_db, 'main-1', task_type='main_analysis')
        with queue_db.cursor() as cur:
            cur.execute(
                "UPDATE task_status SET status = %s WHERE task_id = 'main-1'",
                (config.TASK_STATUS_SUCCESS,),
            )
        queue_db.commit()

        assert _enqueue(queue_db, 'main-2', task_type='main_clustering')


class TestAFinishedRootIsOneRow:
    def test_finishing_drops_the_func_and_the_payload(self, queue_db, shared_pg_dsn):
        _enqueue(queue_db, 'done-1')
        conn = _fresh(shared_pg_dsn)
        try:
            with conn.cursor() as cur:
                sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='w1')
                written = sql.finish_task(
                    cur, 'done-1', config.TASK_STATUS_SUCCESS,
                    {'message': 'Analysed 10 albums.'}, time.time(), worker_id='w1',
                )
            conn.commit()
        finally:
            conn.close()

        assert written == config.TASK_STATUS_SUCCESS
        status, _attempts, _max, func, payload, _worker = _row(queue_db, 'done-1')
        assert status == config.TASK_STATUS_SUCCESS
        assert func is None
        assert payload is None

    def test_a_row_claimed_by_someone_else_is_not_overwritten(
        self, queue_db, shared_pg_dsn
    ):
        _enqueue(queue_db, 'stolen-1')
        conn = _fresh(shared_pg_dsn)
        try:
            with conn.cursor() as cur:
                sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='w2')
                written = sql.finish_task(
                    cur, 'stolen-1', config.TASK_STATUS_SUCCESS,
                    {'message': 'done'}, time.time(), worker_id='w1',
                )
            conn.commit()
        finally:
            conn.close()

        assert written is None
        status, _attempts, _max, _func, _payload, worker = _row(queue_db, 'stolen-1')
        assert status == config.TASK_STATUS_RUNNING
        assert worker == 'w2'

    def test_the_recap_details_carry_the_message(self, queue_db, shared_pg_dsn):
        _enqueue(queue_db, 'recap-1')
        conn = _fresh(shared_pg_dsn)
        try:
            with conn.cursor() as cur:
                sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='w1')
                sql.finish_task(
                    cur, 'recap-1', config.TASK_STATUS_SUCCESS,
                    {'message': 'Analysed 3900 albums, 12 failed'}, time.time(),
                    worker_id='w1',
                )
            conn.commit()
        finally:
            conn.close()

        with queue_db.cursor() as cur:
            cur.execute("SELECT details FROM task_status WHERE task_id = 'recap-1'")
            details = cur.fetchone()[0]
        parsed = details if isinstance(details, dict) else json.loads(details)
        assert parsed['message'] == 'Analysed 3900 albums, 12 failed'
        assert 'log' not in parsed, 'details must never grow a log list again'

    def test_a_missing_history_table_never_undoes_the_terminal_row(
        self, queue_db, shared_pg_dsn
    ):
        _enqueue(queue_db, 'nohist-1')
        worker = _bare_worker(shared_pg_dsn)
        worker.identity = 'w-nohist'
        try:
            with worker._conn.cursor() as cur:
                sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='w-nohist')
            worker._conn.commit()
            with queue_db.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS task_history")
            queue_db.commit()
            worker.finalize(
                {'task_id': 'nohist-1'}, config.TASK_STATUS_SUCCESS, None,
                result={'message': 'Analysed 3 albums.'},
            )
        finally:
            worker._conn.close()

        status, _attempts, _max, func, _payload, _worker = _row(queue_db, 'nohist-1')
        assert status == config.TASK_STATUS_SUCCESS, (
            'record_task_history rolls back on failure; while it shared the terminal '
            "write's transaction that rollback undid the verdict, the row stayed "
            'RUNNING under a live idle worker, and every start answered 409'
        )
        assert func is None


class TestChildrenAreReapedByTheirParent:
    def test_a_terminal_child_is_deleted_and_its_result_returned_once(self, queue_db):
        _enqueue(queue_db, 'parent-1', task_type='main_clustering')
        _enqueue(
            queue_db, 'kid-1', task_type='clustering_batch', parent_task_id='parent-1'
        )
        with queue_db.cursor() as cur:
            cur.execute(
                "UPDATE task_status SET status = %s, details = %s WHERE task_id = 'kid-1'",
                (config.TASK_STATUS_SUCCESS, json.dumps({'message': 'batch done'})),
            )
        queue_db.commit()

        with queue_db.cursor() as cur:
            reaped = sql.reap_children(cur, 'parent-1')
        queue_db.commit()

        assert [row['task_id'] for row in reaped] == ['kid-1']
        with queue_db.cursor() as cur:
            second = sql.reap_children(cur, 'parent-1')
        queue_db.commit()
        assert second == [], 'a reaped child is gone, so its result cannot be counted twice'

    def test_a_live_child_survives_the_reap(self, queue_db):
        _enqueue(queue_db, 'parent-1', task_type='main_clustering')
        _enqueue(
            queue_db, 'kid-1', task_type='clustering_batch', parent_task_id='parent-1'
        )

        with queue_db.cursor() as cur:
            assert sql.reap_children(cur, 'parent-1') == []
            assert [c['task_id'] for c in sql.live_children(cur, 'parent-1')] == ['kid-1']
        queue_db.commit()


class TestReclaimWaitsOutTheGracePeriod:
    def test_a_freshly_written_row_is_not_reclaimed_even_with_no_lock_holder(
        self, queue_db, shared_pg_dsn
    ):
        from taskqueue import maintenance

        _enqueue(queue_db, 'fresh-1')
        dead = _fresh(shared_pg_dsn)
        with dead.cursor() as cur:
            sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='dead')
        dead.commit()
        dead.close()

        conn = _fresh(shared_pg_dsn)
        try:
            assert maintenance.reclaim_orphans(conn) == []
        finally:
            conn.close()

        status, attempts, _max, _func, _payload, _worker = _row(queue_db, 'fresh-1')
        assert status == config.TASK_STATUS_RUNNING
        assert attempts == 0, 'the grace protected it, so it has lost no worker'


class TestALiveWorkerSessionProtectsARowWhoseLockIsGone:
    def test_a_worker_whose_claim_connection_died_keeps_its_task(
        self, queue_db, shared_pg_dsn
    ):
        from taskqueue import maintenance

        identity = 'audiomuse-worker-default-host-11-ab12'
        _enqueue(queue_db, 'unlocked-1')
        claim_conn = _fresh(shared_pg_dsn, identity)
        with claim_conn.cursor() as cur:
            sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id=identity)
            sql.hold(cur, 'unlocked-1')
        claim_conn.commit()
        claim_conn.close()
        _age_row(queue_db, 'unlocked-1')

        listener = _fresh(shared_pg_dsn, identity + sql.WORKER_LISTEN_SUFFIX)
        conn = _fresh(shared_pg_dsn)
        try:
            assert maintenance.reclaim_orphans(conn) == []
        finally:
            listener.close()
            conn.close()

        status, attempts, _max, _func, _payload, _worker = _row(queue_db, 'unlocked-1')
        assert status == config.TASK_STATUS_RUNNING, (
            'a Postgres restart frees every advisory lock at once while the workers '
            'keep computing; the process is still there, so the task is still its own'
        )
        assert attempts == 0

    def test_a_worker_with_no_session_left_is_reclaimed_with_no_grace_at_all(
        self, queue_db, shared_pg_dsn
    ):
        from taskqueue import maintenance

        identity = 'audiomuse-worker-default-host-12-cd34'
        _enqueue(queue_db, 'gone-1')
        claim_conn = _fresh(shared_pg_dsn, identity)
        with claim_conn.cursor() as cur:
            sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id=identity)
            sql.hold(cur, 'gone-1')
        claim_conn.commit()
        claim_conn.close()

        conn = _fresh(shared_pg_dsn)
        try:
            reclaimed = maintenance.reclaim_orphans(conn, grace_seconds=0)
        finally:
            conn.close()

        assert reclaimed == [('gone-1', config.TASK_STATUS_NEW)], (
            'the boot pass runs seconds after the death it exists to repair, so a '
            'grace would guarantee it found nothing'
        )


class TestAWedgedMainTaskIsNotLeftHoldingTheQueue:
    @staticmethod
    def _held_by_a_live_worker(queue_db, shared_pg_dsn, task_id, silent_minutes,
                               task_type='main_analysis', parent_task_id=None):
        identity = f'audiomuse-worker-high-host-{abs(hash(task_id)) % 999}-ab12'
        _enqueue(
            queue_db, task_id, task_type=task_type, queue=sql.QUEUE_HIGH,
            parent_task_id=parent_task_id,
        )
        holder = _fresh(shared_pg_dsn, identity)
        with holder.cursor() as cur:
            sql.claim(cur, sql.QUEUE_HIGH, time.time(), worker_id=identity)
            sql.hold(cur, task_id)
        holder.commit()
        with queue_db.cursor() as cur:
            cur.execute(
                "UPDATE task_status SET timestamp = NOW() - make_interval(mins => %s) "
                "WHERE task_id = %s",
                (silent_minutes, task_id),
            )
        queue_db.commit()
        return holder

    def _nudge(self, shared_pg_dsn):
        from taskqueue import maintenance

        conn = _fresh(shared_pg_dsn)
        try:
            return maintenance.nudge_wedged_main_tasks(conn)
        finally:
            conn.close()

    def test_a_main_task_silent_past_the_limit_with_a_live_worker_is_cancelled(
        self, queue_db, shared_pg_dsn
    ):
        holder = self._held_by_a_live_worker(
            queue_db, shared_pg_dsn, 'wedged-1',
            config.QUEUE_WEDGED_MAIN_TASK_MINUTES + 10,
        )
        try:
            assert self._nudge(shared_pg_dsn) == ['wedged-1'], (
                'its worker still answers Postgres so reclaim will never take this '
                'row, and the one-live-main index means every other main task is '
                'locked out until somebody ends that worker'
            )
        finally:
            holder.close()

    def test_one_nudge_alone_never_terminates_a_backend(
        self, queue_db, shared_pg_dsn
    ):
        holder = self._held_by_a_live_worker(
            queue_db, shared_pg_dsn, 'firstpass-1',
            config.QUEUE_WEDGED_MAIN_TASK_MINUTES + 10,
        )
        try:
            assert self._nudge(shared_pg_dsn) == ['firstpass-1']
            with holder.cursor() as cur:
                cur.execute("SELECT 1")
                assert cur.fetchone()[0] == 1, (
                    'a worker that can hear the cancel ends itself, and killing its '
                    'connection first would take down a tree that was already going'
                )
        finally:
            holder.close()

    def test_a_task_that_ignores_the_cancel_has_its_backend_terminated(
        self, queue_db, shared_pg_dsn
    ):
        holder = self._held_by_a_live_worker(
            queue_db, shared_pg_dsn, 'stubborn-1',
            int(config.QUEUE_WEDGED_MAIN_TASK_MINUTES * 2) + 10,
        )
        try:
            assert self._nudge(shared_pg_dsn) == ['stubborn-1']
            with pytest.raises(psycopg2.Error):
                with holder.cursor() as cur:
                    cur.execute("SELECT 1")
        finally:
            holder.close()

    def test_a_heartbeat_keeps_one_long_step_from_looking_wedged(
        self, queue_db, shared_pg_dsn
    ):
        from tasks.recovery import _beat_once

        holder = self._held_by_a_live_worker(
            queue_db, shared_pg_dsn, 'building-1',
            config.QUEUE_WEDGED_MAIN_TASK_MINUTES + 10,
        )
        try:
            beat = _fresh(shared_pg_dsn)
            try:
                assert _beat_once(beat, 'building-1') is True
            finally:
                beat.close()
            assert self._nudge(shared_pg_dsn) == [], (
                'one index build is a single opaque call that writes no row while '
                'it runs, so without the heartbeat a big-library IVF or artist-GMM '
                'build is indistinguishable from a wedge and gets cancelled'
            )
        finally:
            holder.close()

    def test_a_heartbeat_stops_once_the_row_leaves_running(
        self, queue_db, shared_pg_dsn
    ):
        from tasks.recovery import _beat_once

        _enqueue(queue_db, 'gone-1', task_type='main_analysis', queue=sql.QUEUE_HIGH)
        beat = _fresh(shared_pg_dsn)
        try:
            assert _beat_once(beat, 'gone-1') is False, (
                'the row is NEW, not RUNNING: a heartbeat that kept bumping a row '
                'this process no longer owns would hold off the reclaim that does'
            )
            assert _beat_once(beat, 'never-existed') is False
        finally:
            beat.close()

    def test_a_main_task_that_reported_recently_is_left_alone(
        self, queue_db, shared_pg_dsn
    ):
        holder = self._held_by_a_live_worker(
            queue_db, shared_pg_dsn, 'busy-1',
            max(0, config.QUEUE_WEDGED_MAIN_TASK_MINUTES - 10),
        )
        try:
            assert self._nudge(shared_pg_dsn) == [], (
                'a run that is still writing progress is working, however slowly, and '
                'killing it would charge an attempt against a healthy task'
            )
        finally:
            holder.close()

    def test_a_child_row_is_never_nudged_however_silent_it_is(
        self, queue_db, shared_pg_dsn
    ):
        _enqueue(queue_db, 'root-1', task_type='main_clustering', queue=sql.QUEUE_HIGH)
        holder = self._held_by_a_live_worker(
            queue_db, shared_pg_dsn, 'kid-1',
            config.QUEUE_WEDGED_MAIN_TASK_MINUTES + 10,
            task_type='main_clustering', parent_task_id='root-1',
        )
        try:
            assert self._nudge(shared_pg_dsn) == [], (
                'a silent child belongs to the stall valve, which gives up on that '
                'batch alone; ending the worker from here would take the whole run '
                'down with it'
            )
        finally:
            holder.close()

    def test_a_task_whose_worker_is_gone_is_left_to_the_ordinary_reclaim(
        self, queue_db, shared_pg_dsn
    ):
        holder = self._held_by_a_live_worker(
            queue_db, shared_pg_dsn, 'dead-1',
            config.QUEUE_WEDGED_MAIN_TASK_MINUTES + 10,
        )
        holder.close()

        assert self._nudge(shared_pg_dsn) == [], (
            'reclaim already requeues a task whose worker died, and nudging it too '
            'would cancel a task that is about to be resumed'
        )


    def test_a_wedged_plugin_task_is_nudged_even_though_its_name_is_open_ended(
        self, queue_db, shared_pg_dsn
    ):
        holder = self._held_by_a_live_worker(
            queue_db, shared_pg_dsn, 'wedged-plugin-1',
            config.QUEUE_WEDGED_MAIN_TASK_MINUTES + 10,
            task_type='plugin.demo.sync',
        )
        try:
            assert self._nudge(shared_pg_dsn) == ['wedged-plugin-1'], (
                'get_queue_blocking_task ORs in plugin.%, so a live plugin task '
                'refuses every cron start and every manual batch start. Nothing '
                'watched it before: it is absent from NUDGE_TASK_TYPES because '
                'the namespace is open, and reclaim needs the worker to DIE'
            )
        finally:
            holder.close()

    def test_the_prefix_match_does_not_catch_a_task_type_nobody_watches(
        self, queue_db, shared_pg_dsn
    ):
        holder = self._held_by_a_live_worker(
            queue_db, shared_pg_dsn, 'planner-1',
            config.QUEUE_WEDGED_MAIN_TASK_MINUTES + 10,
            task_type='provider_migration_planner',
        )
        try:
            assert self._nudge(shared_pg_dsn) == [], (
                'the planner holds no admission index and refuses no start, and '
                'session_discard cancels a live one rather than blocking on it, '
                'so cancelling it from here would end a job nothing is waiting on'
            )
        finally:
            holder.close()


class TestARetryWaitsAndAWorkerDeathDoesNot:
    def _claim_and_hold(self, shared_pg_dsn, task_id, worker_id='w1'):
        holder = _fresh(shared_pg_dsn, worker_id)
        with holder.cursor() as cur:
            claimed = sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id=worker_id)
            if claimed is not None:
                sql.hold(cur, task_id)
        holder.commit()
        return holder, claimed

    def _next_run_at(self, conn, task_id):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT next_run_at, next_run_at > NOW() FROM task_status WHERE task_id = %s",
                (task_id,),
            )
            return cur.fetchone()

    def test_a_delayed_row_is_not_claimable_before_its_time_and_is_after(
        self, queue_db, shared_pg_dsn
    ):
        _enqueue(queue_db, 'delayed-1', max_attempts=3)
        holder, claimed = self._claim_and_hold(shared_pg_dsn, 'delayed-1')
        assert claimed is not None
        with holder.cursor() as cur:
            sql.release(cur, 'delayed-1')
            status = sql.requeue_or_fail(
                cur, 'delayed-1', time.time(), {'message': 'raised'}, delay_seconds=3600,
            )
        holder.commit()
        holder.close()
        assert status == config.TASK_STATUS_NEW

        scheduled, in_future = self._next_run_at(queue_db, 'delayed-1')
        assert scheduled is not None
        assert in_future

        too_early = _fresh(shared_pg_dsn, 'w2')
        try:
            with too_early.cursor() as cur:
                assert sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='w2') is None, (
                    'a retry that fires the instant the row is back is the same '
                    'deterministic failure three times in a row; the wait is the point'
                )
            too_early.commit()
            with queue_db.cursor() as cur:
                cur.execute(
                    "UPDATE task_status SET next_run_at = NOW() - interval '1 second' "
                    "WHERE task_id = %s",
                    ('delayed-1',),
                )
            queue_db.commit()
            with too_early.cursor() as cur:
                claimed = sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='w2')
            too_early.commit()
            assert claimed is not None
            assert claimed['task_id'] == 'delayed-1'
            assert claimed['attempts'] == 1
        finally:
            too_early.close()

    def test_a_worker_death_reclaim_is_never_delayed(self, queue_db, shared_pg_dsn):
        from taskqueue import maintenance

        _enqueue(queue_db, 'dies-1', max_attempts=3)
        holder, _claimed = self._claim_and_hold(shared_pg_dsn, 'dies-1')
        with holder.cursor() as cur:
            sql.release(cur, 'dies-1')
            sql.requeue_or_fail(
                cur, 'dies-1', time.time(), {'message': 'raised'}, delay_seconds=3600,
            )
        holder.commit()
        holder.close()
        with queue_db.cursor() as cur:
            cur.execute(
                "UPDATE task_status SET next_run_at = NOW() - interval '1 second' "
                "WHERE task_id = %s",
                ('dies-1',),
            )
        queue_db.commit()

        dead, claimed = self._claim_and_hold(shared_pg_dsn, 'dies-1', worker_id='dead')
        assert claimed is not None
        dead.close()
        _age_row(queue_db, 'dies-1')
        conn = _fresh(shared_pg_dsn)
        try:
            maintenance.reclaim_orphans(conn)
        finally:
            conn.close()

        scheduled, _in_future = self._next_run_at(queue_db, 'dies-1')
        status, attempts, _max, _func, _payload, _worker = _row(queue_db, 'dies-1')
        assert status == config.TASK_STATUS_NEW
        assert attempts == 2
        assert scheduled is None, (
            'a task whose worker died resumes at once, as it always has; the backoff '
            'is for a task that raised, and reclaim passes no delay'
        )

    def test_the_uncharged_requeue_clears_a_stale_delay(self, queue_db, shared_pg_dsn):
        _enqueue(queue_db, 'blip-1', max_attempts=3)
        with queue_db.cursor() as cur:
            cur.execute(
                "UPDATE task_status SET status = %s, worker_id = 'w1', "
                "next_run_at = NOW() + interval '1 hour' WHERE task_id = %s",
                (config.TASK_STATUS_RUNNING, 'blip-1'),
            )
            assert sql.requeue_uncharged(cur, 'blip-1', worker_id='w1')
        queue_db.commit()

        scheduled, _in_future = self._next_run_at(queue_db, 'blip-1')
        assert scheduled is None

    def test_worker_deaths_and_raises_spend_one_budget(self, queue_db, shared_pg_dsn):
        from taskqueue import maintenance

        _enqueue(queue_db, 'budget-1', max_attempts=3)
        for _death in (1, 2):
            dead, claimed = self._claim_and_hold(shared_pg_dsn, 'budget-1', worker_id='dead')
            assert claimed is not None
            dead.close()
            _age_row(queue_db, 'budget-1')
            conn = _fresh(shared_pg_dsn)
            try:
                maintenance.reclaim_orphans(conn)
            finally:
                conn.close()

        holder, claimed = self._claim_and_hold(shared_pg_dsn, 'budget-1', worker_id='w3')
        assert claimed is not None
        assert claimed['attempts'] == 2
        with holder.cursor() as cur:
            sql.release(cur, 'budget-1')
            third = sql.requeue_or_fail(
                cur, 'budget-1', time.time(), {'message': 'raised'}, delay_seconds=0,
            )
        holder.commit()
        holder.close()
        assert third == config.TASK_STATUS_NEW, 'the third bad ending still earns a restart'

        holder, claimed = self._claim_and_hold(shared_pg_dsn, 'budget-1', worker_id='w4')
        assert claimed is not None
        assert claimed['attempts'] == 3
        with holder.cursor() as cur:
            sql.release(cur, 'budget-1')
            fourth = sql.requeue_or_fail(
                cur, 'budget-1', time.time(), {'message': 'raised again'}, delay_seconds=0,
            )
        holder.commit()
        holder.close()
        assert fourth == config.TASK_STATUS_FAIL, (
            'two worker deaths and two raises are four bad endings on ONE counter; '
            'nobody may quietly give application errors a budget of their own'
        )


class TestTheSchemaUpgradeAddsTheNewestColumn:
    def test_an_install_at_the_previous_revision_gets_next_run_at_on_boot(
        self, queue_db
    ):
        with queue_db.cursor() as cur:
            cur.execute("ALTER TABLE task_status DROP COLUMN next_run_at")
        queue_db.commit()
        with queue_db.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'task_status' AND column_name = 'shared_payload'"
            )
            assert cur.fetchone() is not None, 'the previous revision already had this'

        with queue_db.cursor() as cur:
            sql.ensure_schema(cur)
        queue_db.commit()

        with queue_db.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'task_status' AND column_name = 'next_run_at'"
            )
            assert cur.fetchone() is not None, (
                'ensure_schema runs the ALTER block only when the probed column is '
                'absent; every existing install has shared_payload, so a probe left '
                'pointing at it never adds next_run_at and the first claim after the '
                'upgrade raises UndefinedColumn'
            )
        _enqueue(queue_db, 'upgraded-1')
        with queue_db.cursor() as cur:
            claimed = sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='w1')
        queue_db.commit()
        assert claimed is not None
        assert claimed['task_id'] == 'upgraded-1'


class TestAttemptsCountsWorkerDeathsAndRaisesNotClaims:
    def test_a_requeue_and_reclaim_burns_one_attempt_per_death(
        self, queue_db, shared_pg_dsn
    ):
        from taskqueue import maintenance

        _enqueue(queue_db, 'deaths-1', max_attempts=3)
        for expected in (1, 2, 3):
            dead = _fresh(shared_pg_dsn)
            with dead.cursor() as cur:
                sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='dead')
                sql.hold(cur, 'deaths-1')
            dead.commit()
            dead.close()
            _age_row(queue_db, 'deaths-1')

            conn = _fresh(shared_pg_dsn)
            try:
                maintenance.reclaim_orphans(conn)
            finally:
                conn.close()

            status, attempts, _max, _func, _payload, _worker = _row(queue_db, 'deaths-1')
            assert attempts == expected
            assert status == config.TASK_STATUS_NEW

        dead = _fresh(shared_pg_dsn)
        with dead.cursor() as cur:
            sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='dead')
            sql.hold(cur, 'deaths-1')
        dead.commit()
        dead.close()
        _age_row(queue_db, 'deaths-1')

        conn = _fresh(shared_pg_dsn)
        try:
            maintenance.reclaim_orphans(conn)
        finally:
            conn.close()

        status, _attempts, _max, _func, _payload, _worker = _row(queue_db, 'deaths-1')
        assert status == config.TASK_STATUS_FAIL, (
            'three restarts are allowed, so the fourth death is the one that gives up'
        )


class TestAFanOutIsNeverWipedByAnotherTask:
    def test_a_finished_child_of_a_live_parent_survives_another_run_starting(
        self, queue_db
    ):
        import taskqueue

        with queue_db.cursor() as cur:
            cur.execute(
                "INSERT INTO task_status (task_id, parent_task_id, task_type, "
                "status, progress, timestamp) "
                "VALUES ('parent-live', NULL, 'main_analysis', 'RUNNING', 20, NOW())"
            )
            cur.execute(
                "INSERT INTO task_status (task_id, parent_task_id, task_type, "
                "status, progress, timestamp) "
                "VALUES ('child-done', 'parent-live', 'album', 'SUCCESS', 100, NOW())"
            )
            cur.execute(
                "INSERT INTO task_status (task_id, parent_task_id, task_type, "
                "status, progress, timestamp) "
                "VALUES ('old-run', NULL, 'main_clustering', 'SUCCESS', 100, NOW())"
            )
        queue_db.commit()

        taskqueue.enqueue(
            'tasks.multiserver_sync.sweep_server',
            args=('server-1',),
            task_id='sweep-new',
            task_type='server_sweep',
            queue=sql.QUEUE_DEFAULT,
            conn=queue_db,
        )
        queue_db.commit()

        with queue_db.cursor() as cur:
            cur.execute("SELECT task_id FROM task_status")
            remaining = sorted(row[0] for row in cur.fetchall())
        queue_db.commit()

        assert 'child-done' in remaining, (
            'the analysis is still draining it, and deleting it made the parent wait '
            'for a child that no longer existed - the run hung forever'
        )
        assert 'old-run' not in remaining, 'a finished run with no live parent still goes'
        assert 'parent-live' in remaining


class TestTaskStatusNeverAccumulates:
    def test_a_new_run_leaves_only_its_own_row_whatever_was_there_before(
        self, queue_db
    ):
        with queue_db.cursor() as cur:
            for index in range(40):
                cur.execute(
                    "INSERT INTO task_status (task_id, parent_task_id, task_type, "
                    "status, progress, timestamp) "
                    "VALUES (%s, NULL, %s, 'SUCCESS', 100, NOW() - interval '1 hour')",
                    (f'done-{index}', 'main_analysis' if index % 2 else 'main_clustering'),
                )
            for index in range(40):
                cur.execute(
                    "INSERT INTO task_status (task_id, parent_task_id, task_type, "
                    "status, progress, timestamp) "
                    "VALUES (%s, NULL, %s, 'SUCCESS', 100, NOW() - interval '1 hour')",
                    (f'ctl-{index}', sql.CONTROL_TASK_TYPE),
                )
            cur.execute(
                "INSERT INTO task_status (task_id, parent_task_id, task_type, "
                "status, progress, timestamp) "
                "VALUES ('radio-1', NULL, 'alchemy_radio', 'RUNNING', 40, NOW())"
            )
            cur.execute(
                "INSERT INTO task_status (task_id, parent_task_id, task_type, "
                "status, progress, timestamp) "
                "VALUES ('radio-1-child', 'radio-1', 'album', 'RUNNING', 10, NOW())"
            )
        queue_db.commit()

        import taskqueue

        taskqueue.enqueue(
            'tasks.analysis.run_analysis_task',
            task_id='fresh-run',
            task_type='main_analysis',
            queue=sql.QUEUE_DEFAULT,
            conn=queue_db,
        )
        queue_db.commit()

        with queue_db.cursor() as cur:
            cur.execute("SELECT task_id FROM task_status")
            remaining = sorted(row[0] for row in cur.fetchall())
        queue_db.commit()

        assert remaining == ['fresh-run', 'radio-1', 'radio-1-child'], (
            'a starting run drops the 80 finished leftovers, and leaves the radio '
            f'that is still running alone: found {remaining}'
        )


class TestSecretsNeverBecomeDurable:
    def test_an_api_key_kwarg_is_not_written_into_the_row(self, queue_db):
        with queue_db.cursor() as cur:
            sql.insert_job(
                cur,
                task_id='secret-1',
                task_type='main_clustering',
                func='tasks.clustering.run_clustering_task',
                args=(),
                kwargs={'openai_api_key_param': 'sk-do-not-persist', 'runs': 5},
                queue=sql.QUEUE_HIGH,
                priority=0,
                parent_task_id=None,
                sub_type_identifier=None,
                max_attempts=None,
                details={'message': 'queued'},
            )
        queue_db.commit()

        with queue_db.cursor() as cur:
            cur.execute("SELECT payload FROM task_status WHERE task_id = 'secret-1'")
            payload = cur.fetchone()[0]
        assert 'sk-do-not-persist' not in payload
        assert 'runs' in payload

    def test_the_worker_gets_the_key_back_from_config(self, queue_db, shared_pg_dsn):
        with queue_db.cursor() as cur:
            sql.insert_job(
                cur,
                task_id='secret-2',
                task_type='main_clustering',
                func='tasks.clustering.run_clustering_task',
                args=(),
                kwargs={'openai_api_key_param': 'sk-do-not-persist'},
                queue=sql.QUEUE_DEFAULT,
                priority=0,
                parent_task_id=None,
                sub_type_identifier=None,
                max_attempts=None,
                details=None,
            )
        queue_db.commit()

        conn = _fresh(shared_pg_dsn)
        try:
            with conn.cursor() as cur:
                job = sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='w1')
            conn.commit()
        finally:
            conn.close()

        assert job['kwargs'].get('openai_api_key_param') == config.OPENAI_API_KEY


class TestTheReclaimNoticeNamesTheGenerationItTookTheTaskFrom:

    def _listen(self, shared_pg_dsn):
        conn = _fresh(shared_pg_dsn, 'audiomuse-listener-test')
        conn.set_session(autocommit=True)
        with conn.cursor() as cur:
            cur.execute('LISTEN ' + sql.CHANNEL_RECLAIM)
        return conn

    def _await_reclaim(self, conn, task_id):
        deadline = time.monotonic() + NOTIFY_TIMEOUT_SECONDS
        notices = []
        while True:
            conn.poll()
            while conn.notifies:
                decoded = sql.decode_reclaim(conn.notifies.pop(0).payload)
                if decoded and decoded['task_id'] == task_id:
                    notices.append(decoded)
            if notices:
                return notices
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return notices
            select.select([conn], [], [], remaining)

    def test_the_claim_hands_back_the_attempts_the_row_now_carries(
        self, queue_db, shared_pg_dsn
    ):
        _enqueue(queue_db, 'gen-1')
        conn = _fresh(shared_pg_dsn)
        try:
            with conn.cursor() as cur:
                job = sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='w1')
            conn.commit()
        finally:
            conn.close()

        _status, attempts, _max, _func, _payload, _worker = _row(queue_db, 'gen-1')
        assert job['attempts'] == attempts

    def test_a_requeue_notifies_the_previous_worker_and_its_attempt(
        self, queue_db, shared_pg_dsn
    ):
        from taskqueue import maintenance

        _enqueue(queue_db, 'gen-2', max_attempts=3)
        listener = self._listen(shared_pg_dsn)
        try:
            dead = _fresh(shared_pg_dsn)
            with dead.cursor() as cur:
                job = sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='worker-A')
                sql.hold(cur, 'gen-2')
            dead.commit()
            dead.close()
            _age_row(queue_db, 'gen-2')

            conn = _fresh(shared_pg_dsn)
            try:
                maintenance.reclaim_orphans(conn)
            finally:
                conn.close()

            notices = self._await_reclaim(listener, 'gen-2')
        finally:
            listener.close()

        assert notices, (
            'a requeue must announce itself, and nothing arrived on '
            f'{sql.CHANNEL_RECLAIM} within {NOTIFY_TIMEOUT_SECONDS} seconds'
        )
        assert notices[0]['worker_id'] == 'worker-A'
        assert notices[0]['attempts'] == job['attempts']

    def test_the_final_failing_reclaim_also_notifies(self, queue_db, shared_pg_dsn):
        from taskqueue import maintenance

        _enqueue(queue_db, 'gen-3', max_attempts=1)
        listener = self._listen(shared_pg_dsn)
        try:
            dead = _fresh(shared_pg_dsn)
            with dead.cursor() as cur:
                sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='worker-B')
                sql.hold(cur, 'gen-3')
            dead.commit()
            dead.close()
            _age_row(queue_db, 'gen-3')

            conn = _fresh(shared_pg_dsn)
            try:
                maintenance.reclaim_orphans(conn)
            finally:
                conn.close()

            notices = self._await_reclaim(listener, 'gen-3')
        finally:
            listener.close()

        assert notices, (
            'a give-up must announce itself too, and nothing arrived on '
            f'{sql.CHANNEL_RECLAIM} within {NOTIFY_TIMEOUT_SECONDS} seconds'
        )
        assert any(n['worker_id'] == 'worker-B' for n in notices)

    def test_the_next_claim_gets_an_attempt_no_stale_notice_can_address(
        self, queue_db, shared_pg_dsn
    ):
        from taskqueue import maintenance

        _enqueue(queue_db, 'gen-4', max_attempts=3)
        dead = _fresh(shared_pg_dsn)
        with dead.cursor() as cur:
            first = sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='worker-A')
            sql.hold(cur, 'gen-4')
        dead.commit()
        dead.close()
        _age_row(queue_db, 'gen-4')

        conn = _fresh(shared_pg_dsn)
        try:
            maintenance.reclaim_orphans(conn)
            with conn.cursor() as cur:
                second = sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='worker-C')
            conn.commit()
        finally:
            conn.close()

        assert second['attempts'] > first['attempts']


class TestReclaimFailsTheChildrenOfTerminalParentsInsteadOfRequeueingThem:
    def test_an_orphaned_child_of_a_terminal_parent_fails_with_attempts_still_left(
        self, queue_db, shared_pg_dsn
    ):
        from taskqueue import maintenance

        _enqueue(queue_db, 'parent-x', task_type='main_clustering')
        _enqueue(
            queue_db, 'kid-x', task_type='clustering_batch',
            queue=sql.QUEUE_HIGH, parent_task_id='parent-x',
        )
        dead = _fresh(shared_pg_dsn)
        with dead.cursor() as cur:
            claimed = sql.claim(cur, sql.QUEUE_HIGH, time.time(), worker_id='dead')
            sql.hold(cur, 'kid-x')
        dead.commit()
        dead.close()

        assert claimed['task_id'] == 'kid-x'
        assert claimed['attempts'] + 1 <= claimed['max_attempts']

        with queue_db.cursor() as cur:
            cur.execute(
                "UPDATE task_status SET status = %s WHERE task_id = 'parent-x'",
                (config.TASK_STATUS_FAIL,),
            )
        queue_db.commit()
        _age_row(queue_db, 'kid-x')

        conn = _fresh(shared_pg_dsn)
        try:
            reclaimed = maintenance.reclaim_orphans(conn)
        finally:
            conn.close()

        assert reclaimed == [('kid-x', config.TASK_STATUS_FAIL)]
        status, attempts = _row(queue_db, 'kid-x')[:2]
        queue_db.commit()
        assert status == config.TASK_STATUS_FAIL
        assert attempts == claimed['attempts'] + 1


class TestOneLargeInputIsStoredOnceForTheWholeFanOut:

    def test_a_child_row_carries_a_token_not_the_body(self, queue_db):
        import taskqueue

        body = 'x' * 200000
        _enqueue(queue_db, 'parent-s', task_type='main_clustering')
        taskqueue.enqueue(
            'tasks.clustering.run_clustering_batch_task',
            kwargs={'genre_to_lightweight_track_data_map_json': body, 'batch': 1},
            task_id='kid-s',
            task_type='clustering_batch',
            parent_task_id='parent-s',
            shared={'genre_to_lightweight_track_data_map_json': None},
            conn=queue_db,
        )
        queue_db.commit()

        with queue_db.cursor() as cur:
            cur.execute("SELECT payload FROM task_status WHERE task_id = 'kid-s'")
            payload = cur.fetchone()[0]
        queue_db.commit()

        assert body not in payload
        assert len(payload) < 500
        assert taskqueue.SHARED_KWARG_REF in payload

    def test_every_sibling_shares_one_stored_copy(self, queue_db):
        import taskqueue

        body = 'y' * 100000
        _enqueue(queue_db, 'parent-s', task_type='main_clustering')
        for index in range(5):
            taskqueue.enqueue(
                'tasks.clustering.run_clustering_batch_task',
                kwargs={'genre_to_lightweight_track_data_map_json': body},
                task_id=f'kid-{index}',
                task_type='clustering_batch',
                parent_task_id='parent-s',
                shared={'genre_to_lightweight_track_data_map_json': None},
                conn=queue_db,
            )
        queue_db.commit()

        with queue_db.cursor() as cur:
            cur.execute("SELECT count(*) FROM task_status WHERE shared_payload IS NOT NULL")
            assert cur.fetchone()[0] == 1
            cur.execute(
                "SELECT sum(length(payload)) FROM task_status WHERE parent_task_id = 'parent-s'"
            )
            assert cur.fetchone()[0] < len(body)
        queue_db.commit()

    def test_the_child_receives_the_byte_identical_body(self, queue_db, shared_pg_dsn):
        import taskqueue

        body = 'z' * 150000
        _enqueue(queue_db, 'parent-s', task_type='main_clustering')
        taskqueue.enqueue(
            'tasks.clustering.run_clustering_batch_task',
            kwargs={'genre_to_lightweight_track_data_map_json': body, 'batch': 7},
            task_id='kid-s',
            task_type='clustering_batch',
            parent_task_id='parent-s',
            queue=sql.QUEUE_HIGH,
            shared={'genre_to_lightweight_track_data_map_json': None},
            conn=queue_db,
        )
        queue_db.commit()

        worker = _bare_worker(shared_pg_dsn)
        try:
            with worker._conn.cursor() as cur:
                job = sql.claim(cur, sql.QUEUE_HIGH, time.time(), worker_id='w1')
            worker._conn.commit()
            restored = worker.hydrate_shared(job['kwargs'])
        finally:
            worker._conn.close()

        assert restored['genre_to_lightweight_track_data_map_json'] == body
        assert restored['batch'] == 7
        assert taskqueue.SHARED_KWARG_REF not in restored

    def test_a_worker_reads_the_body_once_for_every_sibling_batch(
        self, queue_db, shared_pg_dsn, monkeypatch
    ):
        import taskqueue

        body = 'w' * 50000
        _enqueue(queue_db, 'parent-s', task_type='main_clustering')
        for index in range(4):
            taskqueue.enqueue(
                'tasks.clustering.run_clustering_batch_task',
                kwargs={'genre_to_lightweight_track_data_map_json': body},
                task_id=f'kid-{index}',
                task_type='clustering_batch',
                parent_task_id='parent-s',
                queue=sql.QUEUE_HIGH,
                shared={'genre_to_lightweight_track_data_map_json': None},
                conn=queue_db,
            )
        queue_db.commit()

        worker = _bare_worker(shared_pg_dsn)
        reads = []
        original = sql.get_shared
        monkeypatch.setattr(sql, 'get_shared', lambda cur, task_id, token: (
            reads.append(token) or original(cur, task_id, token)
        ))
        try:
            for _ in range(4):
                with worker._conn.cursor() as cur:
                    job = sql.claim(cur, sql.QUEUE_HIGH, time.time(), worker_id='w1')
                worker._conn.commit()
                assert worker.hydrate_shared(job['kwargs'])[
                    'genre_to_lightweight_track_data_map_json'
                ] == body
        finally:
            worker._conn.close()

        assert len(reads) == 1, 'sibling batches share one token, so one read serves them all'

    def test_a_body_whose_token_no_longer_matches_is_refused(self, queue_db):
        _enqueue(queue_db, 'parent-s', task_type='main_clustering')
        with queue_db.cursor() as cur:
            token = sql.put_shared(cur, 'parent-s', 'the-body')
        queue_db.commit()

        with queue_db.cursor() as cur:
            with pytest.raises(sql.SharedPayloadUnavailable):
                sql.get_shared(cur, 'parent-s', 'a-different-token')
            assert sql.get_shared(cur, 'parent-s', token) == 'the-body'
        queue_db.commit()

    def test_clearing_is_scoped_to_the_token_that_published_it(self, queue_db):
        _enqueue(queue_db, 'parent-s', task_type='main_clustering')
        with queue_db.cursor() as cur:
            stale = sql.put_shared(cur, 'parent-s', 'first-body')
            fresh = sql.put_shared(cur, 'parent-s', 'second-body')
            assert sql.clear_shared(cur, 'parent-s', stale) == 0
            assert sql.get_shared(cur, 'parent-s', fresh) == 'second-body'
            assert sql.clear_shared(cur, 'parent-s', fresh) == 1
        queue_db.commit()


class TestTheSharedBodyIsWrittenOnceNotOncePerChild:

    def test_five_siblings_cause_exactly_one_body_write(self, queue_db):
        import taskqueue

        body = 'q' * 120000
        _enqueue(queue_db, 'parent-w', task_type='main_clustering')
        original_factory = queue_db.cursor_factory
        writes = _count_statement(queue_db, sql._PUT_SHARED)
        try:
            for index in range(5):
                taskqueue.enqueue(
                    'tasks.clustering.run_clustering_batch_task',
                    kwargs={'genre_to_lightweight_track_data_map_json': body},
                    task_id=f'kid-w{index}',
                    task_type='clustering_batch',
                    parent_task_id='parent-w',
                    queue=sql.QUEUE_HIGH,
                    shared={'genre_to_lightweight_track_data_map_json': None},
                    conn=queue_db,
                )
            queue_db.commit()
        finally:
            queue_db.cursor_factory = original_factory

        assert len(writes) == 1, (
            f"the body must be stored once for the whole fan-out, not {len(writes)} times"
        )
        token, _stored_body, owner = writes[0]
        assert owner == 'parent-w'
        assert token == sql.shared_token_for(body)

        with queue_db.cursor() as cur:
            cur.execute("SELECT shared_token FROM task_status WHERE task_id = 'parent-w'")
            assert cur.fetchone()[0] == token
            cur.execute(
                "SELECT count(*) FROM task_status WHERE parent_task_id = 'parent-w'"
            )
            assert cur.fetchone()[0] == 5
        queue_db.commit()

    def test_a_changed_body_does_replace_the_stored_one(self, queue_db):
        _enqueue(queue_db, 'parent-w', task_type='main_clustering')
        with queue_db.cursor() as cur:
            first = sql.put_shared(cur, 'parent-w', 'body-one')
            again = sql.put_shared(cur, 'parent-w', 'body-one')
            second = sql.put_shared(cur, 'parent-w', 'body-two')
        queue_db.commit()

        assert first == again
        assert second != first
        with queue_db.cursor() as cur:
            assert sql.get_shared(cur, 'parent-w', second) == 'body-two'
        queue_db.commit()


class TestTheWorkerSnapshotListsWhatTheQueueBelieves:
    def test_a_running_row_whose_worker_has_no_connection_is_still_a_busy_worker(
        self, queue_db
    ):
        _enqueue(queue_db, 'ghost-1', task_type='main_analysis', queue=sql.QUEUE_HIGH)
        with queue_db.cursor() as cur:
            claimed = sql.claim(
                cur, sql.QUEUE_HIGH, time.time(),
                worker_id='audiomuse-worker-high-ghost-host-1-beef',
            )
        queue_db.commit()
        assert claimed is not None

        with queue_db.cursor() as cur:
            workers = sql.worker_snapshot(cur)

        ghost = [w for w in workers if w['current_job_id'] == 'ghost-1']
        assert len(ghost) == 1
        assert ghost[0]['state'] == 'busy'
        assert ghost[0]['queues'] == ['high']
        assert ghost[0]['started_at'] is None

    def test_both_workers_of_one_container_are_listed_and_only_the_working_one_is_busy(
        self, queue_db, shared_pg_dsn
    ):
        host = 'audiomuse-ai-worker-0'
        high = f'audiomuse-worker-high-{host}-7-c0de'
        default = f'audiomuse-worker-default-{host}-8-f00d'
        _enqueue(queue_db, 'album-1', task_type='album_analysis', queue=sql.QUEUE_DEFAULT)
        conns = [
            _fresh(shared_pg_dsn, name)
            for name in (
                high, high + sql.WORKER_LISTEN_SUFFIX,
                default, default + sql.WORKER_LISTEN_SUFFIX,
            )
        ]
        try:
            with conns[2].cursor() as cur:
                sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id=default)
            conns[2].commit()
            with queue_db.cursor() as cur:
                workers = sql.worker_snapshot(cur)
        finally:
            for conn in conns:
                conn.close()

        mine = [w for w in workers if w['hostname'] == host]
        assert len(mine) == 2, 'a listen connection is the same worker, not a third row'
        by_queue = {w['queues'][0]: w for w in mine}
        assert by_queue['high']['state'] == 'idle'
        assert by_queue['high']['current_job_id'] is None
        assert by_queue['high']['started_at'] is not None
        assert by_queue['default']['state'] == 'busy'
        assert by_queue['default']['current_job_id'] == 'album-1'
        assert by_queue['default']['current_task_type'] == 'album_analysis'

    def test_a_worker_whose_only_connection_is_its_listener_is_still_listed(
        self, queue_db, shared_pg_dsn
    ):
        identity = 'audiomuse-worker-high-lonely-host-9-abcd'
        listener = _fresh(shared_pg_dsn, identity + sql.WORKER_LISTEN_SUFFIX)
        try:
            with queue_db.cursor() as cur:
                workers = sql.worker_snapshot(cur)
        finally:
            listener.close()

        lonely = [w for w in workers if w['hostname'] == 'lonely-host']
        assert len(lonely) == 1
        assert lonely[0]['queues'] == ['high']
        assert lonely[0]['state'] == 'idle'
        assert lonely[0]['started_at'] is not None

    def test_a_busy_worker_whose_claim_connection_dropped_is_listed_once_as_busy(
        self, queue_db, shared_pg_dsn
    ):
        identity = 'audiomuse-worker-high-dropped-host-3-d00d'
        _enqueue(queue_db, 'dropped-1', task_type='main_analysis', queue=sql.QUEUE_HIGH)
        claim_conn = _fresh(shared_pg_dsn, identity)
        with claim_conn.cursor() as cur:
            sql.claim(cur, sql.QUEUE_HIGH, time.time(), worker_id=identity)
        claim_conn.commit()
        claim_conn.close()

        listener = _fresh(shared_pg_dsn, identity + sql.WORKER_LISTEN_SUFFIX)
        try:
            with queue_db.cursor() as cur:
                workers = sql.worker_snapshot(cur)
        finally:
            listener.close()

        dropped = [w for w in workers if w['hostname'] == 'dropped-host']
        assert len(dropped) == 1, 'the listener and the RUNNING row are one worker, not two rows'
        assert dropped[0]['state'] == 'busy'
        assert dropped[0]['current_job_id'] == 'dropped-1'
        assert dropped[0]['started_at'] is not None

    def test_a_worker_with_a_stale_running_row_shows_its_newest_job(
        self, queue_db, shared_pg_dsn
    ):
        identity = 'audiomuse-worker-default-twice-host-5-beef'
        _enqueue(queue_db, 'twice-a', task_type='album_analysis')
        _enqueue(queue_db, 'twice-b', task_type='album_analysis')
        claim_conn = _fresh(shared_pg_dsn, identity)
        try:
            with claim_conn.cursor() as cur:
                first = sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id=identity)
            claim_conn.commit()
            _age_row(queue_db, first['task_id'])
            with claim_conn.cursor() as cur:
                second = sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id=identity)
            claim_conn.commit()
            with queue_db.cursor() as cur:
                workers = sql.worker_snapshot(cur)
        finally:
            claim_conn.close()

        twice = [w for w in workers if w['hostname'] == 'twice-host']
        assert len(twice) == 1
        assert twice[0]['state'] == 'busy'
        assert twice[0]['current_job_id'] == second['task_id']
        assert second['task_id'] != first['task_id']


class TestQueueBacklogCountsWhatIsActuallyWaiting:
    def test_an_empty_queue_reports_zero(self, queue_db):
        with queue_db.cursor() as cur:
            backlog = sql.queue_backlog(cur)

        by_name = {row['queue_name']: row for row in backlog}
        assert by_name[sql.QUEUE_DEFAULT]['pending_count'] == 0

    def test_pending_rows_are_counted_per_queue(self, queue_db):
        _enqueue(queue_db, 'a-1', task_type='album_analysis', queue=sql.QUEUE_DEFAULT)
        _enqueue(queue_db, 'a-2', task_type='album_analysis', queue=sql.QUEUE_DEFAULT)
        _enqueue(queue_db, 'h-1', task_type='main_analysis', queue=sql.QUEUE_HIGH)

        with queue_db.cursor() as cur:
            backlog = sql.queue_backlog(cur)

        by_name = {row['queue_name']: row for row in backlog}
        assert by_name[sql.QUEUE_DEFAULT]['pending_count'] == 2
        assert by_name[sql.QUEUE_HIGH]['pending_count'] == 1

    def test_a_claimed_row_counts_as_running_instead_of_pending(self, queue_db):
        _enqueue(queue_db, 'a-1', task_type='album_analysis', queue=sql.QUEUE_DEFAULT)
        with queue_db.cursor() as cur:
            sql.claim(cur, sql.QUEUE_DEFAULT, time.time(), worker_id='w-1')
        queue_db.commit()

        with queue_db.cursor() as cur:
            backlog = sql.queue_backlog(cur)

        by_name = {row['queue_name']: row for row in backlog}
        assert by_name[sql.QUEUE_DEFAULT]['pending_count'] == 0
        assert by_name[sql.QUEUE_DEFAULT]['running_count'] == 1, (
            'the dashboard cards read this; a queue whose only work is already '
            'claimed used to show 0 and look idle while the analysis ran'
        )

    def test_a_row_waiting_out_its_retry_delay_is_neither_pending_nor_running(
        self, queue_db
    ):
        _enqueue(queue_db, 'a-1', task_type='album_analysis', queue=sql.QUEUE_DEFAULT)
        with queue_db.cursor() as cur:
            cur.execute(
                "UPDATE task_status SET next_run_at = NOW() + interval '1 hour' "
                "WHERE task_id = %s",
                ('a-1',),
            )
        queue_db.commit()

        with queue_db.cursor() as cur:
            backlog = sql.queue_backlog(cur)

        by_name = {row['queue_name']: row for row in backlog}
        assert by_name[sql.QUEUE_DEFAULT]['pending_count'] == 0
        assert by_name[sql.QUEUE_DEFAULT]['running_count'] == 0
        assert by_name[sql.QUEUE_DEFAULT]['delayed_count'] == 1
