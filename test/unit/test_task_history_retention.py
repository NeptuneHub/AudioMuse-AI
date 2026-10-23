# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Which task_history rows record_task_history keeps after each insert.

The history used to keep the newest 10 rows of any type, so a plugin task
scheduled every minute filled it in 10 minutes and pushed AudioMuse's own last
analysis, clustering or cleaning run out of sight. Each task type now keeps its
own newest runs, so the table stays bounded: it never grows forever, the oldest
run of a type goes first. These tests pin the statement's contract; the
row-level cases run against a real Postgres in
test/integration/test_task_history_retention_integration.py.

Main Features:
* Each task type keeps its newest TASK_HISTORY_MAX_ROWS_PER_TYPE rows, so a task
  scheduled every minute cannot push another type's runs out
* There is no total cap across types: the per-type cap is the only one
* The cap is read from config on every insert, and a cap below 1 keeps one row
  instead of emptying the history
* Rows of a plugin that is no longer installed are removed on the next write,
  after the history row is committed; without a plugins table only the cap runs
"""

from unittest.mock import MagicMock

import config
import database


class _RecordingCursor:
    def __init__(self, plugins_table=False):
        self.calls = []
        self.plugins_table = plugins_table

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        self.calls.append((' '.join(statement.split()), params))

    def fetchone(self):
        if self.calls and 'to_regclass' in self.calls[-1][0]:
            return (self.plugins_table,)
        return None


def _history_calls(plugins_table=False):
    cur = _RecordingCursor(plugins_table)
    db = MagicMock()
    db.cursor.return_value = cur
    db.commit.side_effect = lambda: cur.calls.append(('COMMIT', None))

    database.record_task_history('task-1', 'main_analysis', 'SUCCESS', note='', conn=db)

    db.rollback.assert_not_called()
    return cur.calls


def _history_deletes(plugins_table=False):
    return [call for call in _history_calls(plugins_table) if call[0].startswith('DELETE FROM task_history')]


def _retention_delete():
    return _history_deletes()[0]


class TestTaskHistoryRetention:

    def test_each_task_type_keeps_its_own_newest_runs(self, monkeypatch):
        monkeypatch.setattr(config, 'TASK_HISTORY_MAX_ROWS_PER_TYPE', 7)
        sql, params = _retention_delete()

        assert (
            'ROW_NUMBER() OVER ( PARTITION BY task_type '
            'ORDER BY recorded_at DESC, id DESC ) AS type_rank'
        ) in sql, (
            'a task scheduled every minute must not push another type\'s runs '
            'out of the history (#944)'
        )
        assert 'WHERE type_rank > %s' in sql
        assert params == (7,)

    def test_there_is_no_total_cap_across_types(self):
        sql, _params = _retention_delete()

        assert 'overall_rank' not in sql
        assert not hasattr(config, 'TASK_HISTORY_MAX_ROWS'), (
            'only the per-type cap bounds the history'
        )

    def test_a_cap_below_one_keeps_a_row_instead_of_emptying_the_history(self, monkeypatch):
        monkeypatch.setattr(config, 'TASK_HISTORY_MAX_ROWS_PER_TYPE', -3)

        _sql, params = _retention_delete()

        assert params == (1,)

    def test_rows_of_a_plugin_no_longer_installed_are_removed(self):
        deletes = _history_deletes(plugins_table=True)

        assert len(deletes) == 2
        sql, params = deletes[1]
        assert 'WHERE task_type ~ %s' in sql, 'only the two shapes that name their plugin are touched'
        assert (
            "CASE WHEN split_part(task_type, '.', 2) = %s THEN split_part(task_type, '.', 3) "
            "ELSE split_part(task_type, '.', 2) END NOT IN (SELECT id FROM plugins)"
        ) in sql, 'the owner is read from the shape: plugin.<id>.<task> or plugin.audiomuse_plugins.<id>.<function>'
        assert params == (database._PLUGIN_OWNED_TASK_TYPE_RE, 'audiomuse_plugins')

    def test_the_history_row_is_committed_before_the_plugin_purge_runs(self):
        calls = [sql for sql, _params in _history_calls(plugins_table=True)]

        insert = next(i for i, sql in enumerate(calls) if sql.startswith('INSERT INTO task_history'))
        purge = [i for i, sql in enumerate(calls) if sql.startswith('DELETE FROM task_history')][1]
        assert 'COMMIT' in calls[insert:purge], (
            'a failing purge must not roll back the run that just finished'
        )

    def test_without_a_plugins_table_only_the_cap_runs(self):
        deletes = _history_deletes(plugins_table=False)

        assert len(deletes) == 1, 'a missing plugins table must not cost the history row'

    def test_the_cap_defaults_to_a_hundred_runs_per_type(self):
        assert config.TASK_HISTORY_MAX_ROWS_PER_TYPE == 100
        assert not hasattr(database, 'TASK_HISTORY_MAX_ROWS'), (
            'the cap is a config tunable now; a second copy in database.py would drift'
        )
