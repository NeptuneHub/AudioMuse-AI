# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Task-status details roundtrip tests against a real Postgres database.

Writes and reads task-status details through app_helper on both the TEXT
and JSONB details columns to confirm each path surfaces the same dict.

Also drives the RQ janitor's orphan reap against a real task_status table,
because its whole contract is which rows survive an UPDATE: mocking the cursor
proves the SQL was assembled, never that a row was left intact. The guard that
matters most is the status exclusion, since a task writes its own SUCCESS before
RQ marks the job finished, so the reaper races every completing task.

Main Features:
* TEXT details roundtrip as a JSON string, JSONB returns a dict directly.
* Both paths surface identical content and null details yield an empty dict.
* Orphan reap on real rows: a stale PROGRESS row behind a failed RQ job becomes
  FAILURE, and a row that flips to SUCCESS mid-pass is never overwritten.
* A still-running job and a row inside the grace period are both left alone, so
  a long silent phase is not mistaken for a dead task.
"""

import copy
import json
import os
import sys
import tempfile
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

_TASK_STATUS_DDL = (
    "CREATE TABLE task_status ("
    "id SERIAL PRIMARY KEY, task_id TEXT UNIQUE NOT NULL, parent_task_id TEXT, "
    "task_type TEXT NOT NULL, sub_type_identifier TEXT, status TEXT, "
    "progress INTEGER DEFAULT 0, details TEXT, "
    "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "start_time DOUBLE PRECISION, end_time DOUBLE PRECISION)"
)

_TASK_STATUS_JSONB_DDL = _TASK_STATUS_DDL.replace("details TEXT,", "details JSONB,")

_SAMPLE_DETAILS = {
    "log": ["Analyzing album", "Done"],
    "current_album": "Album X",
    "status_message": "running",
    "nested": {"a": 1, "b": [2, 3]},
}


@pytest.fixture(scope='session')
def pg_dsn():
    if psycopg2 is None:
        pytest.skip("psycopg2 not importable")
    dsn = os.environ.get('AUDIOMUSE_TEST_DATABASE_URL')
    if dsn:
        try:
            psycopg2.connect(dsn).close()
        except Exception as e:
            pytest.skip(f"AUDIOMUSE_TEST_DATABASE_URL not reachable: {e}")
        yield dsn
        return
    try:
        import pgserver
    except Exception:
        pytest.skip(
            "No test database. Set AUDIOMUSE_TEST_DATABASE_URL to a disposable "
            "DB, or `pip install pgserver` for an ephemeral local instance."
        )
    data_dir = tempfile.mkdtemp(prefix='audiomuse_pg_')
    server = pgserver.get_server(data_dir)
    try:
        yield server.get_uri()
    finally:
        server.cleanup()


def _make_db(pg_dsn, ddl):
    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
        cur.execute(ddl)
    return conn


@pytest.fixture
def text_details_db(pg_dsn):
    conn = _make_db(pg_dsn, _TASK_STATUS_DDL)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
    conn.close()


@pytest.fixture
def jsonb_details_db(pg_dsn):
    conn = _make_db(pg_dsn, _TASK_STATUS_JSONB_DDL)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
    conn.close()


class TestTaskStatusDetailsRoundTrip:
    def test_text_details_round_trip_is_json_string(self, text_details_db, monkeypatch):
        import database
        import app_helper

        monkeypatch.setattr(database, 'get_db', lambda: text_details_db)

        database.save_task_status(
            'task-text',
            'main_analysis',
            status='PROGRESS',
            progress=42,
            details=copy.deepcopy(_SAMPLE_DETAILS),
        )
        row = database.get_task_info_from_db('task-text')
        assert row is not None
        assert isinstance(row['details'], str)

        surfaced = app_helper.coerce_db_details(row['details'])
        assert surfaced == _SAMPLE_DETAILS
        assert surfaced['nested'] == {"a": 1, "b": [2, 3]}
        assert row['task_type'] == 'main_analysis'
        assert row['progress'] == 42
        assert row['running_time_seconds'] >= 0

    def test_jsonb_details_round_trip_returns_dict_no_reparse(self, jsonb_details_db, monkeypatch):
        import database
        import app_helper

        monkeypatch.setattr(database, 'get_db', lambda: jsonb_details_db)

        database.save_task_status(
            'task-jsonb',
            'main_clustering',
            status='SUCCESS',
            progress=100,
            details=copy.deepcopy(_SAMPLE_DETAILS),
        )
        row = database.get_task_info_from_db('task-jsonb')
        assert row is not None
        assert isinstance(row['details'], dict)

        surfaced = app_helper.coerce_db_details(row['details'])
        assert surfaced is row['details']
        assert surfaced == _SAMPLE_DETAILS
        assert row['running_time_seconds'] >= 0

    def test_both_paths_surface_identical_content(
        self, text_details_db, jsonb_details_db, monkeypatch
    ):
        import database
        import app_helper

        monkeypatch.setattr(database, 'get_db', lambda: text_details_db)
        database.save_task_status(
            't-text', 'main_analysis', status='PROGRESS', details=copy.deepcopy(_SAMPLE_DETAILS)
        )
        text_surfaced = app_helper.coerce_db_details(
            database.get_task_info_from_db('t-text')['details']
        )

        monkeypatch.setattr(database, 'get_db', lambda: jsonb_details_db)
        database.save_task_status(
            't-jsonb', 'main_clustering', status='SUCCESS', details=copy.deepcopy(_SAMPLE_DETAILS)
        )
        jsonb_surfaced = app_helper.coerce_db_details(
            database.get_task_info_from_db('t-jsonb')['details']
        )

        assert text_surfaced == jsonb_surfaced == _SAMPLE_DETAILS
        assert isinstance(text_surfaced['log'], list)
        assert isinstance(jsonb_surfaced['log'], list)

    def test_null_details_surfaces_empty_dict(self, text_details_db, monkeypatch):
        import database
        import app_helper

        monkeypatch.setattr(database, 'get_db', lambda: text_details_db)

        database.save_task_status('task-null', 'main_analysis', status='PENDING', details=None)
        row = database.get_task_info_from_db('task-null')
        assert row is not None
        assert row['details'] is None
        assert app_helper.coerce_db_details(row['details']) == {}


class _FakeJob:
    def __init__(self, status, on_status=None):
        self._status = status
        self._on_status = on_status
        self.last_heartbeat = None
        self.retries_left = 0
        self.origin = 'high'

    def get_status(self, refresh=False):
        if self._on_status is not None:
            self._on_status()
        return self._status


def _insert_row(conn, task_id, status='PROGRESS', task_type='main_analysis', age_seconds=600):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO task_status (task_id, parent_task_id, task_type, status, "
            "progress, details, timestamp, start_time) VALUES "
            "(%s, NULL, %s, %s, %s, %s, NOW() - make_interval(secs => %s), %s)",
            (task_id, task_type, status, 40,
             json.dumps({'message': 'still working'}), age_seconds, time.time()),
        )


def _read_row(conn, task_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status, progress, details FROM task_status WHERE task_id = %s",
            (task_id,),
        )
        return cur.fetchone()


def _arm_reaper(monkeypatch, pg_dsn, job):
    import rq.exceptions
    import rq.job
    from tasks import multiserver_sync as sync

    monkeypatch.setattr(
        sync, 'connect_raw', lambda: psycopg2.connect(pg_dsn)
    )
    monkeypatch.setattr(sync, '_ABANDONED_FIRST_SEEN', {})
    monkeypatch.setattr(
        rq.job.Job, 'fetch', staticmethod(lambda task_id, connection=None: job)
    )

    def batch_fetch(job_ids, connection=None):
        fetched = []
        for job_id in job_ids:
            try:
                fetched.append(rq.job.Job.fetch(job_id, connection=connection))
            except rq.exceptions.NoSuchJobError:
                fetched.append(None)
        return fetched

    monkeypatch.setattr(rq.job.Job, 'fetch_many', staticmethod(batch_fetch))
    return sync


class TestOrphanReapAgainstRealRows:
    def test_stale_progress_row_behind_a_failed_rq_job_really_becomes_failure(
        self, text_details_db, pg_dsn, monkeypatch
    ):
        _insert_row(text_details_db, 'reap-failed')
        sync = _arm_reaper(monkeypatch, pg_dsn, _FakeJob('failed'))

        assert sync.reap_orphaned_tasks() == 1

        status, progress, details = _read_row(text_details_db, 'reap-failed')
        assert status == 'FAILURE'
        assert progress == 100
        assert "'failed'" in json.loads(details)['message']

    def test_a_row_that_turns_success_mid_pass_is_never_overwritten(
        self, text_details_db, pg_dsn, monkeypatch
    ):
        _insert_row(text_details_db, 'reap-race')

        def flip_to_success():
            racer = psycopg2.connect(pg_dsn)
            racer.autocommit = True
            try:
                with racer.cursor() as cur:
                    cur.execute(
                        "UPDATE task_status SET status = 'SUCCESS', progress = 100, "
                        "details = %s WHERE task_id = 'reap-race'",
                        (json.dumps({'message': 'Analysis complete.'}),),
                    )
            finally:
                racer.close()

        sync = _arm_reaper(
            monkeypatch, pg_dsn, _FakeJob('finished', on_status=flip_to_success)
        )

        sync.reap_orphaned_tasks()

        status, progress, details = _read_row(text_details_db, 'reap-race')
        assert status == 'SUCCESS'
        assert progress == 100
        assert json.loads(details)['message'] == 'Analysis complete.'

    def test_a_still_started_job_leaves_its_long_running_row_untouched(
        self, text_details_db, pg_dsn, monkeypatch
    ):
        _insert_row(text_details_db, 'reap-live')
        sync = _arm_reaper(monkeypatch, pg_dsn, _FakeJob('started'))

        assert sync.reap_orphaned_tasks() == 0

        status, progress, _details = _read_row(text_details_db, 'reap-live')
        assert status == 'PROGRESS'
        assert progress == 40

    def test_a_row_inside_the_grace_period_is_not_even_a_candidate(
        self, text_details_db, pg_dsn, monkeypatch
    ):
        _insert_row(text_details_db, 'reap-young', age_seconds=5)
        sync = _arm_reaper(monkeypatch, pg_dsn, _FakeJob('failed'))

        assert sync.reap_orphaned_tasks() == 0

        status, _progress, _details = _read_row(text_details_db, 'reap-young')
        assert status == 'PROGRESS'


class TestSweepStrengthSurvivesItsOwnProgressUpdates:
    """A sweep's task row records whether it is a FULL-REFRESH alignment, because
    recovery has to queue an equally strong replacement when the job dies: a
    matching-only sweep exits at "already aligned; nothing to do" without fetching
    whenever every track is mapped - exactly the state a provider migration leaves -
    so the artist ids it was queued to rebuild would stay empty.

    save_task_status REPLACES the details column wholesale, and recovery only ever
    fires for a job that STARTED and died (a still-queued job counts as alive and
    simply runs after the restart). So the flag has to survive the running sweep's
    own progress reports, not merely be written once at enqueue time.
    """

    def _reporter(self, monkeypatch, conn, full_refresh):
        import types
        from unittest.mock import MagicMock

        import database
        from tasks import multiserver_sync as sync

        monkeypatch.setattr(database, 'get_db', lambda: conn)
        fake_flask_app = types.ModuleType('flask_app')
        fake_flask_app.app = MagicMock()
        monkeypatch.setitem(sys.modules, 'flask_app', fake_flask_app)
        return sync, sync._make_reporter('sweep-1', 'all', full_refresh=full_refresh)

    def test_the_flag_is_still_readable_after_a_progress_report(
        self, text_details_db, monkeypatch
    ):
        import database

        sync, report = self._reporter(monkeypatch, text_details_db, True)
        database.save_task_status(
            'sweep-1', 'server_sweep', status='PENDING', progress=0,
            details={'message': 'queued', 'full_refresh': True},
        )
        report('Starting alignment for Navidrome...', 2, task_state='STARTED')

        row = database.get_task_info_from_db('sweep-1')
        assert sync._details_full_refresh(row['details']) is True, (
            "the janitor reads this row AFTER the sweep started; a wiped flag "
            "downgrades the replacement to a no-op matching-only sweep"
        )

    def test_the_flag_survives_a_jsonb_details_column_too(
        self, jsonb_details_db, monkeypatch
    ):
        import database

        sync, report = self._reporter(monkeypatch, jsonb_details_db, True)
        report('Aligning...', 10, task_state='STARTED')

        row = database.get_task_info_from_db('sweep-1')
        assert sync._details_full_refresh(row['details']) is True

    def test_a_matching_only_sweep_is_recorded_as_weak(
        self, text_details_db, monkeypatch
    ):
        import database

        sync, report = self._reporter(monkeypatch, text_details_db, False)
        report('Aligning...', 10, task_state='STARTED')

        row = database.get_task_info_from_db('sweep-1')
        assert sync._details_full_refresh(row['details']) is False
