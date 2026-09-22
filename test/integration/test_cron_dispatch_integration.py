# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""One cron tick of every schedulable type, against a real Postgres.

The dispatch is three database acts in one transaction - claim the row for this
wall-clock minute, ask the admission guard whether a main task is already live,
write the queue job - and none of them can be proved with a mocked cursor: the
claim is an UPDATE with a predicate that must reject the second web process, the
guard reads a PARTIAL UNIQUE INDEX, and the retry bookkeeping joins cron_retry
against task_status by a task type the registry translates.

That translation is what this module is really for. app_cron and
database.cron_retry_task_already_done each held their own copy of the cron name
to queue type map, so a blocked analysis row recorded a retry under one name
while the SUCCESS that should clear it was written under another. Both now read
task_types, and a tick here proves the two ends still meet in the same rows.

Main Features:
* A due batch row reaches the queue under the queue task type the registry
  declares; the online rows (radio, sonic fingerprint, album of the week) run
  in this process through the path the registry declares and write no queue job
* A tick claims its minute once: the same minute run twice fires once
* The online rows run BESIDE a live analysis: SUCCESS, no cron_retry row, and
  the analysis stays RUNNING; a live online row never refuses a batch start
* A blocked batch row records a retry, the retry tick starts it once the blocker
  finished (clustering behind analysis and the reverse)
* The shared playlist scaffold run inline heartbeats and cancel-checks the
  inline row itself
* The startup reap never fails a row a worker owns, and an inline SUCCESS row
  keeps the log the run reported
"""

import json
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

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
import task_types  # noqa: E402
from taskqueue import sql as queue_sql  # noqa: E402

_TASK_STATUS_DDL = """
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

_CRON_DDL = """
    CREATE TABLE cron (
        id SERIAL PRIMARY KEY,
        name TEXT,
        task_type TEXT NOT NULL,
        cron_expr TEXT NOT NULL,
        enabled BOOLEAN DEFAULT FALSE,
        last_run DOUBLE PRECISION,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        options JSONB NOT NULL DEFAULT '{}'::jsonb
    )
"""

_CRON_RETRY_DDL = """
    CREATE TABLE cron_retry (
        task_type TEXT PRIMARY KEY,
        retry_until DOUBLE PRECISION,
        attempts INTEGER DEFAULT 0,
        first_blocked_at DOUBLE PRECISION,
        blocker_task_id TEXT,
        blocker_task_type TEXT
    )
"""

SCHEDULABLE = ('analysis', 'clustering', 'sonic_fingerprint', 'album_of_the_week', 'alchemy_radio')


@pytest.fixture
def cron_db(shared_pg_dsn):
    conn = psycopg2.connect(shared_pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        for table in ('cron', 'cron_retry', 'task_status'):
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        cur.execute(_TASK_STATUS_DDL)
        cur.execute(_CRON_DDL)
        cur.execute(_CRON_RETRY_DDL)
    conn.autocommit = False
    with conn.cursor() as cur:
        queue_sql.ensure_schema(cur)
    conn.commit()
    try:
        yield conn
    finally:
        try:
            conn.rollback()
            conn.autocommit = True
            with conn.cursor() as cur:
                for table in ('cron', 'cron_retry', 'task_status'):
                    cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        finally:
            conn.close()


@pytest.fixture(autouse=True)
def _fresh_cron_clock():
    import app_cron

    app_cron._cron_clock['last_minute'] = None
    yield
    app_cron._cron_clock['last_minute'] = None


def _add_row(conn, task_type, cron_expr='* * * * *', enabled=True, last_run=None):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO cron (name, task_type, cron_expr, enabled, last_run) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (f'integration {task_type}', task_type, cron_expr, enabled, last_run),
        )
        row_id = cur.fetchone()[0]
    conn.commit()
    return row_id


def _jobs(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT task_id, task_type, func, payload, queue_name FROM task_status "
            "WHERE func IS NOT NULL ORDER BY id"
        )
        return [
            {
                'task_id': row[0], 'task_type': row[1], 'func': row[2],
                'kwargs': (json.loads(row[3] or '{}') or {}).get('kwargs') or {},
                'queue': row[4],
            }
            for row in cur.fetchall()
        ]


def _rows(conn, sql_text, params=None):
    with conn.cursor() as cur:
        cur.execute(sql_text, params or ())
        return cur.fetchall()


def _inline_patches(conn, summary):
    return (
        patch('app_cron.get_db', return_value=conn),
        patch('database.get_db', return_value=conn),
        patch('tasks.radio_manager.run_radio_playlists', return_value=summary),
        patch(
            'tasks.sonic_fingerprint_manager.run_sonic_fingerprint_task',
            return_value=summary,
        ),
        patch(
            'tasks.album_creation_manager.run_album_of_the_week_task',
            return_value=summary,
        ),
    )


def _drive(conn, action, radio_summary=None):
    import database

    summary = radio_summary if radio_summary is not None else {'playlists_created': 1, 'failed': []}
    get_db, db_get_db, radio_patch, fingerprint_patch, album_patch = _inline_patches(conn, summary)
    writes = []

    def save_and_record(*args, **kwargs):
        written = database.save_task_status(*args, **kwargs)
        writes.append((args[1], args[2], written))
        return written

    with (
        get_db, db_get_db,
        radio_patch as radio,
        fingerprint_patch as fingerprint,
        album_patch as album,
        patch('app_cron.save_task_status', side_effect=save_and_record) as saved,
        patch('app_cron._INLINE_STAGGER_SECONDS', 0),
    ):
        action()
    return SimpleNamespace(
        alchemy_radio=radio,
        sonic_fingerprint=fingerprint,
        album_of_the_week=album,
        saved=saved,
        writes=writes,
    )


def _tick(conn, radio_summary=None):
    import app_cron

    return _drive(conn, app_cron.run_due_cron_jobs, radio_summary)


def _retry(conn):
    import app_cron

    return _drive(conn, app_cron.retry_due_cron_jobs)


def _insert_root(conn, task_id, task_type, status, start_time=None):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO task_status (task_id, task_type, status, start_time) "
            "VALUES (%s, %s, %s, %s)",
            (task_id, task_type, status, time.time() if start_time is None else start_time),
        )
    conn.commit()


def _set_status(conn, task_id, status):
    with conn.cursor() as cur:
        cur.execute("UPDATE task_status SET status = %s WHERE task_id = %s", (status, task_id))
    conn.commit()


class TestEveryScheduledTypeReachesTheRightPlace:
    @pytest.mark.parametrize('cron_type', SCHEDULABLE)
    def test_a_due_row_fires_exactly_once_per_minute(self, cron_db, cron_type):
        row_id = _add_row(cron_db, cron_type)

        inline = _tick(cron_db)
        first = _jobs(cron_db)
        _tick(cron_db)
        second = _jobs(cron_db)

        assert second == first, (
            'the second tick of the same wall-clock minute enqueued again; the '
            'claim predicate is the only thing stopping two web processes from '
            'double-firing every schedule'
        )
        stamped = _rows(cron_db, "SELECT last_run FROM cron WHERE id = %s", (row_id,))[0][0]
        assert stamped is not None

        if cron_type in task_types.INLINE_FLASK_TASK_TYPES:
            ran = getattr(inline, cron_type)
            assert first == [], (
                f'{cron_type} queries the in-memory index, which only Flask holds, '
                'so it must run inline and write no queue job'
            )
            assert ran.call_count == 1
            assert ran.call_args.kwargs['server_scope'] == 'all'
            statuses = _rows(
                cron_db,
                "SELECT status FROM task_status WHERE task_type = %s", (cron_type,),
            )
            assert [row[0] for row in statuses] == ['SUCCESS']
            return

        assert len(first) == 1, first
        job = first[0]
        assert job['task_type'] == task_types.CRON_TASK_TYPE_TO_QUEUE_TYPE[cron_type]
        scope = job['kwargs'].get('server_scope', job['kwargs'].get('output_server_scope'))
        assert scope == 'all', (
            'a batch task always covers every configured server; a per-schedule '
            'scope left every other server without playlists, silently'
        )
        assert not any(
            getattr(inline, name).called for name in task_types.INLINE_FLASK_TASK_TYPES
        )

    @pytest.mark.parametrize('cron_type', sorted(task_types.CRON_INLINE_TASKS))
    def test_a_playlist_builder_runs_inline_by_the_path_the_registry_holds(self, cron_db, cron_type):
        _add_row(cron_db, cron_type)
        inline = _tick(cron_db)

        ran = getattr(inline, cron_type)
        assert ran.call_count == 1
        assert _jobs(cron_db) == []
        row_id = ran.call_args.kwargs['inline_task_id']
        assert _rows(
            cron_db, "SELECT task_type, status, func FROM task_status WHERE task_id = %s",
            (row_id,),
        ) == [(cron_type, 'SUCCESS', None)], (
            'the run reports under the inline row app_cron wrote, and that row '
            'carries no func, so no worker reclaim can ever touch it'
        )

    def test_two_playlist_rows_due_together_run_one_after_the_other(self, cron_db):
        for cron_type in sorted(task_types.CRON_INLINE_TASKS):
            _add_row(cron_db, cron_type)

        inline = _tick(cron_db)

        started = sum(
            getattr(inline, cron_type).call_count
            for cron_type in task_types.CRON_INLINE_TASKS
        )
        assert started == len(task_types.CRON_INLINE_TASKS), (
            'an inline run finishes before the next row is dispatched, so both '
            'get their turn in the same tick instead of one deferring to a retry'
        )
        assert _jobs(cron_db) == [], 'neither may reach the queue any more'
        assert _rows(cron_db, "SELECT task_type FROM cron_retry") == [], (
            'nothing was blocked: they ran one after the other on this thread'
        )
        statuses = _rows(
            cron_db,
            "SELECT task_type, status FROM task_status WHERE task_type = ANY(%s)",
            (sorted(task_types.CRON_INLINE_TASKS),),
        )
        assert len(statuses) == 1, (
            'each finish collapses every other terminal root row, so the table '
            'keeps ONE recap however many ran; asserting a row per type asserts '
            'the opposite of what collapse_finished_task guarantees'
        )
        task_type, status = statuses[0]
        assert status == 'SUCCESS'
        assert task_type in task_types.CRON_INLINE_TASKS

    def test_every_online_row_runs_beside_a_live_analysis(self, cron_db):
        online = sorted(task_types.INLINE_FLASK_TASK_TYPES)
        for cron_type in online:
            _add_row(cron_db, cron_type)
        _insert_root(cron_db, 'live-analysis', 'main_analysis', 'RUNNING')

        inline = _tick(cron_db)

        for cron_type in online:
            assert getattr(inline, cron_type).call_count == 1, (
                f'{cron_type} is an online run: a live batch task must never hold '
                'it back'
            )
        terminal = [
            (task_type, status, written) for task_type, status, written in inline.writes
            if status in config.TASK_STATUS_TERMINAL
        ]
        assert sorted(terminal) == [(cron_type, 'SUCCESS', True) for cron_type in online], (
            'every online row must actually STORE its SUCCESS, not only ask for it'
        )
        assert _rows(cron_db, "SELECT task_type FROM cron_retry") == [], (
            'nothing was blocked, so nothing may wait in the batch retry'
        )
        assert _rows(
            cron_db, "SELECT status FROM task_status WHERE task_id = 'live-analysis'"
        ) == [('RUNNING',)], 'the online runs must not archive or collapse the live analysis'

    def test_a_live_online_row_never_refuses_a_manual_batch_start(self, cron_db):
        from flask import Flask

        import app_helper
        import taskqueue

        _insert_root(cron_db, 'live-fingerprint', 'sonic_fingerprint', 'RUNNING')
        app = Flask(__name__)
        with (
            app.app_context(),
            patch('database.get_db', return_value=cron_db),
            patch('app_helper.get_db', return_value=cron_db),
        ):
            response, status = app_helper.admit_and_enqueue_main_task(
                job_id='manual-analysis', task_type='main_analysis',
                busy_label='analysis', error_message='could not queue',
                enqueue=lambda: taskqueue.enqueue(
                    'tasks.analysis.run_analysis_task', args=(0, 5),
                    task_id='manual-analysis', task_type='main_analysis',
                    queue=taskqueue.QUEUE_HIGH,
                ),
            )

        assert status == 202, response.get_json()
        assert [job['task_id'] for job in _jobs(cron_db)] == ['manual-analysis']
        assert _rows(
            cron_db, "SELECT status FROM task_status WHERE task_id = 'live-fingerprint'"
        ) == [('RUNNING',)], (
            'the start archives every live root it may; an online row is '
            'self-managed, so it is neither a blocker nor archived'
        )


class TestTheRetryFindsTheRunItWasWaitingFor:
    def test_a_blocked_row_is_cleared_by_a_success_under_its_queue_task_type(self, cron_db):
        import database

        _add_row(cron_db, 'analysis')
        with cron_db.cursor() as cur:
            cur.execute(
                "INSERT INTO task_status (task_id, task_type, status, start_time) "
                "VALUES ('live-1', 'main_analysis', 'RUNNING', %s)", (time.time(),),
            )
        cron_db.commit()

        _tick(cron_db)

        assert _jobs(cron_db) == [], 'a live main task must refuse the cron start'
        pending = database.list_pending_cron_retries(conn=cron_db)
        assert [entry['task_type'] for entry in pending] == ['analysis']
        blocked_at = pending[0]['first_blocked_at']
        assert pending[0]['blocker_task_type'] == 'main_analysis'

        with patch('database.get_db', return_value=cron_db):
            assert database.cron_retry_task_already_done('analysis', blocked_at, conn=cron_db) is False
            with cron_db.cursor() as cur:
                cur.execute(
                    "INSERT INTO task_status (task_id, task_type, status, start_time) "
                    "VALUES ('done-1', 'main_analysis', 'SUCCESS', %s)", (blocked_at + 1,),
                )
            cron_db.commit()
            assert database.cron_retry_task_already_done('analysis', blocked_at, conn=cron_db) is True, (
                'the cron row says "analysis" and the run that satisfies it is '
                'written as "main_analysis"; when the two ends kept separate '
                'copies of that map a retry could never see its own run finish'
            )

    def test_a_plugin_row_keeps_its_own_name_on_both_ends(self, cron_db):
        import database

        with cron_db.cursor() as cur:
            cur.execute(
                "INSERT INTO task_status (task_id, task_type, status, start_time) "
                "VALUES ('plug-1', 'plugin.demo.sync', 'SUCCESS', 100.0)"
            )
        cron_db.commit()

        assert database.cron_retry_task_already_done('plugin.demo.sync', 50.0, conn=cron_db) is True
        assert database.cron_retry_task_already_done('plugin.demo.sync', 150.0, conn=cron_db) is False


class TestTheRetryTickStartsWhatWasBlocked:
    @pytest.mark.parametrize('cron_type,blocker_type', [
        ('clustering', 'main_analysis'),
        ('analysis', 'main_clustering'),
    ])
    def test_a_blocked_batch_row_starts_once_its_blocker_finished(
        self, cron_db, cron_type, blocker_type,
    ):
        import database

        _add_row(cron_db, cron_type)
        _insert_root(cron_db, 'blocker-1', blocker_type, 'RUNNING')

        _tick(cron_db)

        assert _jobs(cron_db) == []
        pending = database.list_pending_cron_retries(conn=cron_db)
        assert [entry['task_type'] for entry in pending] == [cron_type]
        assert pending[0]['blocker_task_id'] == 'blocker-1'

        _retry(cron_db)
        assert _jobs(cron_db) == [], 'the blocker is still live, so the retry waits'
        assert database.list_pending_cron_retries(conn=cron_db)[0]['attempts'] == 1

        _set_status(cron_db, 'blocker-1', 'SUCCESS')
        _retry(cron_db)

        queue_type = task_types.CRON_TASK_TYPE_TO_QUEUE_TYPE[cron_type]
        assert [job['task_type'] for job in _jobs(cron_db)] == [queue_type], (
            'the owner promise: a batch schedule that fired while another batch '
            'ran is started once that batch is done'
        )
        assert _rows(cron_db, "SELECT task_type FROM cron_retry") == []

    def test_the_window_is_never_extended_and_nothing_starts_after_it(self, cron_db):
        import app_cron
        import database

        _add_row(cron_db, 'clustering')
        _insert_root(cron_db, 'blocker-1', 'main_analysis', 'RUNNING')
        _tick(cron_db)
        first = database.list_pending_cron_retries(conn=cron_db)[0]
        deadline, first_blocked_at = first['retry_until'], first['first_blocked_at']

        later = time.time() + 30 * 60
        app_cron._cron_clock.update(last_minute=None, last_monotonic=None)
        with patch.object(app_cron.time, 'time', return_value=later):
            _tick(cron_db)
            _retry(cron_db)
        again = database.list_pending_cron_retries(conn=cron_db)[0]
        assert (again['retry_until'], again['first_blocked_at']) == (deadline, first_blocked_at), (
            'neither a new refused fire of the same schedule nor a retry bump may '
            'move the CRON_RETRY_MAX_MINUTES window: nothing waits forever'
        )
        assert again['attempts'] >= 1

        _set_status(cron_db, 'blocker-1', 'SUCCESS')
        with patch.object(app_cron.time, 'time', return_value=deadline + 60):
            _retry(cron_db)

        assert _jobs(cron_db) == [], (
            'the blocker finished, but after the window: the schedule expires, it '
            'is never started late'
        )
        assert _rows(cron_db, "SELECT task_type FROM cron_retry") == []
        assert _rows(
            cron_db,
            "SELECT status FROM task_status WHERE task_type = 'main_clustering'",
        ) == [(config.TASK_STATUS_FAILURE,)], 'the expiry is a visible skip'


class TestTheInlineScaffoldWatchesItsOwnRow:
    def _connect(self, dsn):
        def connect(application_name=None, **_kwargs):
            return psycopg2.connect(dsn, application_name=application_name or 'test')
        return connect

    def test_the_heartbeat_and_the_cancel_guard_watch_the_inline_row(
        self, cron_db, shared_pg_dsn, monkeypatch,
    ):
        import config
        import taskqueue
        from tasks import recovery, task_run

        _insert_root(cron_db, 'inline-1', 'sonic_fingerprint', 'RUNNING')
        cadences = []

        def fast_interval(every_minutes):
            cadences.append(every_minutes)
            return 0.05

        monkeypatch.setattr(recovery, '_heartbeat_interval_seconds', fast_interval)
        monkeypatch.setattr(task_run, 'QUEUE_CANCEL_CHECK_SECONDS', 0.0)
        monkeypatch.setattr(config, 'QUEUE_INLINE_STALE_SECONDS', 1800.0)
        connect = self._connect(shared_pg_dsn)
        monkeypatch.setattr('database.connect_raw', connect)
        monkeypatch.setattr(task_run, 'connect_raw', connect)

        def stamp():
            return _rows(cron_db, "SELECT timestamp FROM task_status WHERE task_id = 'inline-1'")[0][0]

        seen = {}

        def build_ids():
            before = stamp()
            time.sleep(0.6)
            cron_db.commit()
            seen['beat'] = stamp() > before
            _set_status(cron_db, 'inline-1', 'REVOKED')
            return ['a']

        with (
            patch('database.get_db', return_value=cron_db),
            patch('tasks.mediaserver.registry.servers_for_scope', return_value=[None, None]),
            patch('tasks.mediaserver.create_or_replace_playlist', return_value={'Id': 'p'}),
        ):
            with pytest.raises(taskqueue.TaskCancelled):
                task_run.run_playlist_task_per_server(
                    'sonic_fingerprint', 'sonic fingerprint', 'Name', 'Sonic Fingerprint',
                    build_ids, 'all', 'inline-1',
                )

        assert seen['beat'], 'the heartbeat must refresh the INLINE row, which has no claim'
        assert cadences and cadences[0] == 30.0, (
            'an inline row beats at a quarter of QUEUE_INLINE_STALE_SECONDS, never '
            'on the wedged-main cadence that outlives the stale sweep'
        )
        assert _rows(
            cron_db, "SELECT status FROM task_status WHERE task_id = 'inline-1'"
        ) == [('REVOKED',)], 'the cancel guard stopped the run before the next server'


class TestTheInlineRowOnRealPostgres:
    def test_the_startup_reap_leaves_a_live_sonic_fingerprint_row_with_a_func(self, cron_db):
        import app_cron
        import config

        _insert_root(cron_db, 'interrupted-inline', 'album_of_the_week', 'RUNNING')
        with cron_db.cursor() as cur:
            cur.execute(
                "INSERT INTO task_status (task_id, task_type, status, start_time, func) "
                "VALUES ('queued-by-a-worker', 'sonic_fingerprint', 'RUNNING', %s, %s)",
                (time.time(), task_types.CRON_INLINE_TASKS['sonic_fingerprint']),
            )
        cron_db.commit()

        with (
            patch('app_cron.get_db', return_value=cron_db),
            patch('database.get_db', return_value=cron_db),
        ):
            assert app_cron.reap_interrupted_inline_runs() == 1

        assert _rows(
            cron_db, "SELECT status FROM task_status WHERE task_id = 'queued-by-a-worker'"
        ) == [('RUNNING',)], (
            'a row with a queue func belongs to a worker (an older install mid-upgrade); '
            'the Flask startup reap must never fail it'
        )
        assert _rows(
            cron_db, "SELECT status FROM task_status WHERE task_id = 'interrupted-inline'"
        ) == [(config.TASK_STATUS_FAILURE,)]

    def test_the_success_row_keeps_the_log_the_run_reported(self, cron_db):
        import app_cron
        import database

        def run():
            database.save_task_status(
                'inline-log', 'sonic_fingerprint', 'PROGRESS', progress=50,
                details={
                    'status_message': 'Building the playlist...',
                    'log': ['[t] Building the playlist...'],
                },
            )
            return {'message': 'Created 1 playlist(s).', 'playlists_created': 1}

        with (
            patch('app_cron.get_db', return_value=cron_db),
            patch('database.get_db', return_value=cron_db),
        ):
            assert app_cron._run_inline(cron_db, 'inline-log', 'sonic_fingerprint', run, 6005) == 'ran'

        status, details = _rows(
            cron_db, "SELECT status, details FROM task_status WHERE task_id = 'inline-log'"
        )[0]
        details = json.loads(details)
        assert status == 'SUCCESS'
        assert '[t] Building the playlist...' in details['log'], (
            'the terminal write must merge into the stored details, not replace them'
        )
        assert details['log'][-1].endswith('Created 1 playlist(s).')
        assert details['playlists_created'] == 1
