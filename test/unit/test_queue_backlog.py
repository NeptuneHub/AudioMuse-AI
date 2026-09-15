# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""How many jobs are waiting per queue.

The dashboard used to show worker state (busy or idle) and nothing about the
backlog behind those workers, so an operator had no way to tell "the default
queue has 340 albums waiting" from "the queue is empty". ``sql.queue_backlog``
answers exactly that with one cheap aggregate against the same partial index
the claim statement already uses.

A queue with nothing waiting must still be reported as zero, not omitted: the
aggregate only produces a row for queue names that actually have a NEW row, so
the Python side has to fill in the gap or a caller iterating "one card per
queue" silently drops a queue the moment its backlog clears.

Main Features:
* Every queue name from queue_names.QUEUE_NAMES is always represented
* A queue with no pending rows reports zero
* The count comes straight off the recorded query
* Only NEW rows of the requested queue names are ever counted
"""

import queue_names
from taskqueue import sql


class _RecordingCursor:
    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def execute(self, statement, params=None):
        self.calls.append((statement, params))

    def fetchall(self):
        return self._rows


class TestEveryQueueIsAlwaysRepresented:
    def test_a_queue_with_nothing_waiting_still_reports_zero(self):
        cur = _RecordingCursor([(queue_names.QUEUE_HIGH, 1, 3, 0)])

        result = sql.queue_backlog(cur)

        by_name = {row['queue_name']: row for row in result}
        assert by_name[queue_names.QUEUE_DEFAULT]['pending_count'] == 0

    def test_every_known_queue_name_appears_exactly_once(self):
        cur = _RecordingCursor([])

        result = sql.queue_backlog(cur)

        assert sorted(row['queue_name'] for row in result) == sorted(queue_names.QUEUE_NAMES)
        assert len(result) == len(queue_names.QUEUE_NAMES)


class TestTheAggregateIsReportedAsIs:
    def test_the_pending_count_comes_from_the_query(self):
        cur = _RecordingCursor([(queue_names.QUEUE_DEFAULT, 0, 340, 0)])

        result = sql.queue_backlog(cur)

        by_name = {row['queue_name']: row for row in result}
        assert by_name[queue_names.QUEUE_DEFAULT]['pending_count'] == 340

    def test_only_the_requested_queue_names_are_bound_into_the_query(self):
        cur = _RecordingCursor([])

        sql.queue_backlog(cur)

        _statement, params = cur.calls[0]
        assert params == (list(queue_names.QUEUE_NAMES),)


class TestTheStatementOnlyCountsWaitingRows:
    def test_the_statement_filters_on_the_new_status(self):
        assert f"status = '{sql._NEW}'" in sql._QUEUE_BACKLOG

    def test_the_statement_groups_by_queue_so_one_row_never_shadows_another(self):
        assert 'GROUP BY queue_name' in sql._QUEUE_BACKLOG


class TestTheCardsSayWhatEachQueueIsDoing:
    def test_running_queued_and_delayed_are_reported_per_queue(self):
        cur = _RecordingCursor([(queue_names.QUEUE_DEFAULT, 2, 340, 1)])

        result = sql.queue_backlog(cur)

        by_name = {row['queue_name']: row for row in result}
        assert by_name[queue_names.QUEUE_DEFAULT] == {
            'queue_name': queue_names.QUEUE_DEFAULT,
            'running_count': 2,
            'pending_count': 340,
            'delayed_count': 1,
        }, (
            'the dashboard counted only NEW rows, so a running analysis whose album '
            'jobs are claimed as fast as they are queued showed 0 and 0 and looked dead'
        )
        assert by_name[queue_names.QUEUE_HIGH] == {
            'queue_name': queue_names.QUEUE_HIGH,
            'running_count': 0,
            'pending_count': 0,
            'delayed_count': 0,
        }
