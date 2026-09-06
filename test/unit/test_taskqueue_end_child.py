# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""A parent ends the child it gave up on through the queue, and only through it.

The analysis parent and the clustering parent each wrote their victim's terminal
row through save_task_status, the generic upsert that accepts any status, and
then published the cancel as a second call. That was the last place a task
wrote a terminal row. Now taskqueue.end_child is the one way: one statement
guarded by the parent's own id, the cancel in the same transaction, and a
vocabulary of exactly two verdicts.

Main Features:
* The row write and the cancel NOTIFY share one transaction
* Only FAIL and REVOKED are verdicts a parent may hand down
* A parent must name itself; there is no ending an orphan
* A child that is already gone still gets the cancel, and the call says no row
  was written
* The statement binds the parent id, so it can only touch that parent's child
"""

from unittest.mock import MagicMock

import pytest

import config
import taskqueue
from taskqueue import sql


class _Cursor:
    def __init__(self, recorder, ended):
        self._recorder = recorder
        self._ended = ended

    def execute(self, statement, params=None):
        self._recorder.append((' '.join(statement.split()), params))

    def fetchone(self):
        return ('kid-1',) if self._ended else None

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _conn(ended=True):
    statements = []
    connection = MagicMock()
    connection.cursor.side_effect = lambda *a, **k: _Cursor(statements, ended)
    connection.statements = statements
    return connection


class TestTheQueueEndsAChildForItsParent:
    def test_the_row_write_and_the_cancel_share_one_transaction(self):
        conn = _conn()

        ended = taskqueue.end_child(
            'kid-1', 'parent-1', config.TASK_STATUS_FAIL, 'gave up', conn=conn,
        )

        assert ended is True
        kinds = [statement.split(' ', 1)[0] for statement, _ in conn.statements]
        assert kinds == ['UPDATE', 'SELECT'], (
            'the terminal row and the cancel signal must land together: a NOTIFY '
            'is delivered on commit, so a cancel outside the transaction could '
            'reach the worker for a row that then rolled back'
        )
        conn.commit.assert_not_called()

    def test_the_verdict_and_the_message_reach_the_row(self):
        conn = _conn()

        taskqueue.end_child('kid-1', 'parent-1', config.TASK_STATUS_REVOKED, 'stalled', conn=conn)

        _statement, params = conn.statements[0]
        assert params[0] == config.TASK_STATUS_REVOKED
        assert params[1] == '{"message": "stalled"}', (
            'the reap reads details.message for the failure tally, byte for byte '
            'what save_task_status used to store'
        )
        assert params[3:] == ('kid-1', 'parent-1')

    def test_the_cancel_names_the_child(self):
        conn = _conn()

        taskqueue.end_child('kid-1', 'parent-1', config.TASK_STATUS_FAIL, 'gave up', conn=conn)

        _statement, params = conn.statements[1]
        assert params == (sql.CHANNEL_CANCEL, 'kid-1')

    @pytest.mark.parametrize('status', [
        config.TASK_STATUS_SUCCESS, config.TASK_STATUS_RUNNING, config.TASK_STATUS_NEW,
    ])
    def test_only_fail_and_revoked_are_verdicts_a_parent_may_hand_down(self, status):
        conn = _conn()

        with pytest.raises(ValueError):
            taskqueue.end_child('kid-1', 'parent-1', status, 'x', conn=conn)

        assert conn.statements == []

    def test_a_parent_must_name_itself(self):
        conn = _conn()

        with pytest.raises(ValueError):
            taskqueue.end_child('kid-1', None, config.TASK_STATUS_FAIL, 'x', conn=conn)

        assert conn.statements == []

    def test_a_child_already_gone_is_still_told_to_stop_and_the_call_says_so(self):
        conn = _conn(ended=False)

        ended = taskqueue.end_child(
            'kid-1', 'parent-1', config.TASK_STATUS_FAIL, 'gave up', conn=conn,
        )

        assert ended is False
        assert any(params == (sql.CHANNEL_CANCEL, 'kid-1') for _s, params in conn.statements)

    def test_a_connection_the_queue_opens_itself_is_committed(self, monkeypatch):
        conn = _conn()
        monkeypatch.setattr('database.get_db', lambda: conn)

        taskqueue.end_child('kid-1', 'parent-1', config.TASK_STATUS_FAIL, 'gave up')

        conn.commit.assert_called_once()


class TestTheStatementBindsTheParent:
    def test_sql_end_child_binds_the_parent_and_answers_from_the_returning_row(self):
        cur = MagicMock()
        cur.fetchone.return_value = ('kid-1',)

        assert sql.end_child(
            cur, 'kid-1', 'parent-1', config.TASK_STATUS_FAIL, {'message': 'x'}, 1.0,
        ) is True

        statement, params = cur.execute.call_args.args
        assert 'parent_task_id = %s' in statement
        assert params[-2:] == ('kid-1', 'parent-1')

    def test_no_returning_row_means_nothing_was_ended(self):
        cur = MagicMock()
        cur.fetchone.return_value = None

        assert sql.end_child(
            cur, 'kid-1', 'parent-1', config.TASK_STATUS_FAIL, {'message': 'x'}, 1.0,
        ) is False
