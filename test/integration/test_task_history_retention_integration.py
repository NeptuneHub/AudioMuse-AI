# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Recent tasks against a real Postgres: what the history keeps and the dashboard shows.

A plugin task scheduled every minute used to fill the 10-row history in 10
minutes, so AudioMuse's own last analysis, clustering or cleaning run, a FAIL
included, dropped out of sight (#944). Each task type now keeps its own newest
runs, the table never grows past its cap, and the dashboard shows one line per
task type. Window functions and DISTINCT ON cannot be proved with a mocked
cursor, so the cases run here.

Main Features:
* A task scheduled every minute keeps only its own newest runs; an earlier
  analysis run stays
* The table never holds more than TASK_HISTORY_MAX_ROWS rows, oldest out first
* Without plugins a type keeps more than the old 10 runs
* One line per task type, the type that ran last first, with its run count
  and its newest failure; a cancelled run is not a failure
* The task_history endpoint's query lists one type's runs, newest first
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
    from psycopg2.extras import DictCursor
except Exception:  # pragma: no cover - psycopg2 is in test/requirements.txt
    psycopg2 = None

pytestmark = pytest.mark.integration

import app_dashboard  # noqa: E402
import config  # noqa: E402
import database  # noqa: E402

_TASK_HISTORY_DDL = """
    CREATE TABLE task_history (
        id SERIAL PRIMARY KEY,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        task_id TEXT,
        task_type TEXT,
        status TEXT,
        duration_seconds DOUBLE PRECISION,
        note TEXT
    )
"""

PLUGIN = 'plugin.lumae_analysis.catalog_reconcile'


@pytest.fixture
def history_db(shared_pg_dsn):
    conn = psycopg2.connect(shared_pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS task_history CASCADE")
        cur.execute(_TASK_HISTORY_DDL)
    conn.autocommit = False
    try:
        yield conn
    finally:
        try:
            conn.rollback()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS task_history CASCADE")
        finally:
            conn.close()


def _record(conn, task_type, task_id, status='SUCCESS'):
    database.record_task_history(task_id, task_type, status, note=f'{task_id} note', conn=conn)


def _record_plugin_runs(conn, count=30, statuses=None):
    for n in range(count):
        _record(conn, PLUGIN, f'plugin-{n}', (statuses or {}).get(n, 'SUCCESS'))


def _kept(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT task_id FROM task_history ORDER BY recorded_at DESC, id DESC")
        return [row[0] for row in cur.fetchall()]


def _lines(conn):
    with conn.cursor(cursor_factory=DictCursor) as cur:
        return app_dashboard._collect_task_metrics(cur)


class TestTheHistoryKeepsEachTypesOwnRuns:

    def test_a_minute_task_keeps_only_its_own_newest_runs(self, history_db, monkeypatch):
        monkeypatch.setattr(config, 'TASK_HISTORY_MAX_ROWS_PER_TYPE', 5)
        _record(history_db, 'main_analysis', 'analysis-1', 'FAIL')
        _record_plugin_runs(history_db)

        assert _kept(history_db) == [f'plugin-{n}' for n in range(29, 24, -1)] + ['analysis-1'], (
            'the minute task cannot push the analysis run, a FAIL included, out'
        )

    def test_the_table_never_holds_more_than_its_cap(self, history_db, monkeypatch):
        monkeypatch.setattr(config, 'TASK_HISTORY_MAX_ROWS', 12)
        types = ('main_analysis', 'main_clustering', PLUGIN)
        for n in range(30):
            _record(history_db, types[n % 3], f'run-{n}')

        assert _kept(history_db) == [f'run-{n}' for n in range(29, 17, -1)]

    def test_without_plugins_a_type_keeps_more_than_ten_runs(self, history_db):
        for n in range(30):
            _record(history_db, 'main_analysis', f'analysis-{n}')

        assert _kept(history_db) == [f'analysis-{n}' for n in range(29, -1, -1)]


class TestOneLinePerTaskType:

    def test_each_type_gets_one_line_the_last_to_run_first(self, history_db):
        _record(history_db, 'main_analysis', 'analysis-1', 'FAIL')
        _record(history_db, 'main_clustering', 'clustering-1')
        _record(history_db, 'main_analysis', 'analysis-2')
        _record_plugin_runs(history_db, statuses={10: 'FAIL', 20: 'REVOKED'})

        lines = _lines(history_db)

        assert [(line['task_type'], line['task_id'], line['runs']) for line in lines] == [
            (PLUGIN, 'plugin-29', 30),
            ('main_analysis', 'analysis-2', 2),
            ('main_clustering', 'clustering-1', 1),
        ]
        plugin, analysis, clustering = lines
        assert plugin['last_failure']['task_id'] == 'plugin-10', (
            'the newest FAIL, not the newer cancelled run'
        )
        assert plugin['last_failure']['note'] == 'plugin-10 note'
        assert analysis['last_failure']['task_id'] == 'analysis-1'
        assert analysis['status'] == 'SUCCESS'
        assert clustering['last_failure'] is None

    def test_a_cancelled_run_is_not_a_failure(self, history_db):
        _record(history_db, 'main_analysis', 'analysis-1', 'REVOKED')

        (line,) = _lines(history_db)

        assert line['status'] == 'REVOKED'
        assert line['last_failure'] is None

    def test_a_failed_last_run_is_also_the_last_failure(self, history_db):
        _record(history_db, 'main_analysis', 'analysis-1')
        _record(history_db, 'main_analysis', 'analysis-2', 'FAIL')

        (line,) = _lines(history_db)

        assert line['task_id'] == 'analysis-2'
        assert line['last_failure']['task_id'] == 'analysis-2'


class TestTheTaskHistoryEndpointQuery:

    def test_it_lists_one_types_runs_newest_first(self, history_db, monkeypatch):
        monkeypatch.setattr(config, 'TASK_HISTORY_MAX_ROWS_PER_TYPE', 3)
        for n in range(3):
            _record(history_db, 'main_analysis', f'analysis-{n}')
        _record_plugin_runs(history_db, count=2)
        monkeypatch.setattr(config, 'TASK_HISTORY_MAX_ROWS_PER_TYPE', 2)

        with history_db.cursor(cursor_factory=DictCursor) as cur:
            runs = app_dashboard._task_type_runs(cur, 'main_analysis')

        assert [run['task_id'] for run in runs] == ['analysis-2', 'analysis-1'], (
            'one type only, newest first, at most the per-type cap'
        )
        assert runs[0]['status'] == 'SUCCESS' and runs[0]['note'] == 'analysis-2 note'
