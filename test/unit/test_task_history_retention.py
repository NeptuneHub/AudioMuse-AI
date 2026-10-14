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
own newest runs, and the table as a whole stays bounded: it never grows forever,
the oldest rows go first. These tests pin the statement's contract; the
row-level cases run against a real Postgres in
test/integration/test_task_history_retention_integration.py.

Main Features:
* Each task type keeps its newest TASK_HISTORY_MAX_ROWS_PER_TYPE rows, so a task
  scheduled every minute cannot push another type's runs out
* The table keeps at most TASK_HISTORY_MAX_ROWS rows in total, oldest out first
* Both caps are read from config on every insert, and a cap below 1 keeps one
  row instead of emptying the history
"""

from unittest.mock import MagicMock

import config
import database


class _RecordingCursor:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        self.calls.append((' '.join(statement.split()), params))

    def fetchone(self):
        return None


def _retention_delete():
    cur = _RecordingCursor()
    db = MagicMock()
    db.cursor.return_value = cur

    database.record_task_history('task-1', 'main_analysis', 'SUCCESS', note='', conn=db)

    db.commit.assert_called_once()
    return next(call for call in cur.calls if call[0].startswith('DELETE FROM task_history'))


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
        assert 'type_rank > %s' in sql
        assert params[1] == 7

    def test_the_whole_table_stays_bounded_oldest_out_first(self, monkeypatch):
        monkeypatch.setattr(config, 'TASK_HISTORY_MAX_ROWS', 40)
        sql, params = _retention_delete()

        assert (
            'ROW_NUMBER() OVER ( ORDER BY recorded_at DESC, id DESC ) AS overall_rank'
        ) in sql
        assert 'WHERE overall_rank > %s OR type_rank > %s' in sql
        assert params[0] == 40

    def test_a_cap_below_one_keeps_a_row_instead_of_emptying_the_history(self, monkeypatch):
        monkeypatch.setattr(config, 'TASK_HISTORY_MAX_ROWS', 0)
        monkeypatch.setattr(config, 'TASK_HISTORY_MAX_ROWS_PER_TYPE', -3)

        _sql, params = _retention_delete()

        assert params == (1, 1)

    def test_the_caps_default_to_a_thousand_rows_and_a_hundred_per_type(self):
        assert config.TASK_HISTORY_MAX_ROWS == 1000
        assert config.TASK_HISTORY_MAX_ROWS_PER_TYPE == 100
        assert not hasattr(database, 'TASK_HISTORY_MAX_ROWS'), (
            'the cap is a config tunable now; a second copy in database.py would drift'
        )
