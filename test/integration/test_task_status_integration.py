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

Main Features:
* TEXT details roundtrip as a JSON string, JSONB returns a dict directly.
* Both paths surface identical content and null details yield an empty dict.
"""

import copy
import os
import sys

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
except Exception:  # pragma: no cover - psycopg2 is in test/requirements.txt
    psycopg2 = None

pytestmark = pytest.mark.integration

_SAMPLE_DETAILS = {
    "log": ["Analyzing album", "Done"],
    "current_album": "Album X",
    "status_message": "running",
    "nested": {"a": 1, "b": [2, 3]},
}


def _make_db(shared_pg_dsn, ddl):
    conn = psycopg2.connect(shared_pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
        cur.execute(ddl)
        from taskqueue import sql as queue_sql

        queue_sql.ensure_schema(cur)
    return conn


@pytest.fixture
def text_details_db(shared_pg_dsn, task_status_ddl):
    conn = _make_db(shared_pg_dsn, task_status_ddl)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
    conn.close()


@pytest.fixture
def jsonb_details_db(shared_pg_dsn, task_status_jsonb_ddl):
    conn = _make_db(shared_pg_dsn, task_status_jsonb_ddl)
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


