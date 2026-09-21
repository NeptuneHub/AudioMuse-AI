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
* A due row of every schedulable type reaches the queue under the queue task
  type the registry declares, with the dotted path the registry declares
* The inline radio row runs in this process and writes no queue job
* A tick claims its minute once: the same minute run twice fires once
* A blocked row records a retry, and a later SUCCESS of the task type it was
  translated to is what clears it
"""

import json
import os
import sys
import time
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


def _tick(conn, radio_summary=None):
    import app_cron

    summary = radio_summary if radio_summary is not None else {'playlists_created': 1, 'failed': []}
    with (
        patch('app_cron.get_db', return_value=conn),
        patch('database.get_db', return_value=conn),
        patch('tasks.radio_manager.run_radio_playlists', return_value=summary) as radio,
    ):
        app_cron.run_due_cron_jobs()
    return radio


class TestEveryScheduledTypeReachesTheRightPlace:
    @pytest.mark.parametrize('cron_type', SCHEDULABLE)
    def test_a_due_row_fires_exactly_once_per_minute(self, cron_db, cron_type):
        row_id = _add_row(cron_db, cron_type)

        radio = _tick(cron_db)
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
            assert first == [], 'the radio runs in Flask and must write no queue job'
            assert radio.call_count == 1
            assert radio.call_args.kwargs['server_scope'] == 'all'
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
        assert not radio.called

    @pytest.mark.parametrize('cron_type', sorted(task_types.CRON_QUEUED_TASKS))
    def test_a_playlist_builder_is_queued_by_the_path_the_registry_holds(self, cron_db, cron_type):
        _add_row(cron_db, cron_type)
        _tick(cron_db)

        job = _jobs(cron_db)[0]
        assert job['func'] == task_types.CRON_QUEUED_TASKS[cron_type]
        assert job['queue'] == queue_sql.QUEUE_DEFAULT
        assert job['task_type'] == cron_type

    def test_two_playlist_rows_due_together_do_not_both_start(self, cron_db):
        for cron_type in sorted(task_types.CRON_QUEUED_TASKS):
            _add_row(cron_db, cron_type)

        _tick(cron_db)

        assert len(_jobs(cron_db)) == 1, (
            'both hold the one-live-main index, so the second is refused by the '
            'admission guard and recorded as a retry rather than queued'
        )
        blocked = _rows(cron_db, "SELECT task_type FROM cron_retry")
        assert len(blocked) == 1 and blocked[0][0] in task_types.CRON_QUEUED_TASKS


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
