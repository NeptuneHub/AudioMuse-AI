# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""What a task reporter is allowed to persist about itself.

Every progress tick appends a timestamped line to the report's own `log`. The
reporter caps that log to MAX_LOG_ENTRIES_STORED itself, in place, before it is
ever handed to save_task_status - so nothing bigger than the last 10 lines is
ever built into what gets sent toward the database, not even for the instant
before some downstream layer would trim it.

The reporter writes RUNNING and nothing else. The terminal row belongs to the
queue, which writes it from what the task returned or raised and collapses or
appends the log there (test_taskqueue_retry pins that half). A task that hands
its reporter a terminal state is the very thing the queue can no longer see past
- it used to make the queue's own verdict and retry a silent no-op - so the
reporter logs that as an error and writes RUNNING anyway.

Main Features:
* The opening write already carries a one-line log with the initial message
* Every report appends a new timestamped line to the same log
* The log handed to save_task_status never exceeds MAX_LOG_ENTRIES_STORED
  entries, capped by the reporter itself rather than left to a later layer
* Progress writes carry the current line as both message and status_message
* The DB write is throttled while running; a write passed force=True lands anyway
* Every terminal state is downgraded to RUNNING: the reporter never writes the
  row the queue owns
"""

import logging
import os
import sys
from unittest.mock import patch

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import (  # noqa: E402
    TASK_STATUS_FAIL,
    TASK_STATUS_REVOKED,
    TASK_STATUS_RUNNING,
    TASK_STATUS_SUCCESS,
)
from database import MAX_LOG_ENTRIES_STORED  # noqa: E402


def _reporter(**kwargs):
    from tasks.task_run import make_task_reporter

    return make_task_reporter('task-1', 'main_analysis', 'Started.', **kwargs)


def _reporter_with_log_snapshots(**kwargs):
    # `logs` inside the reporter is ONE mutable list, appended to (or trimmed) in
    # place and handed to save_task_status by reference every time - a mock's
    # call_args_list stores that same reference, so inspecting an EARLIER call
    # after the fact would show the list's FINAL state, not what it looked like
    # when that call actually happened. A copy of `log` taken inside the fake
    # save_task_status, at the instant of the call, is the only faithful record.
    from tasks.task_run import make_task_reporter

    snapshots = []

    def fake_save_task_status(*_args, **call_kwargs):
        snapshots.append(list(call_kwargs['details'].get('log') or []))

    patcher = patch('tasks.task_run.save_task_status', side_effect=fake_save_task_status)
    patcher.start()
    report = make_task_reporter('task-1', 'main_analysis', 'Started.', **kwargs)
    return report, snapshots, patcher


def test_the_opening_write_carries_a_one_line_log_of_the_initial_message():
    _report, snapshots, patcher = _reporter_with_log_snapshots()
    patcher.stop()

    assert len(snapshots[0]) == 1
    assert snapshots[0][0].endswith('Started.')


def test_every_progress_report_appends_to_the_same_growing_log():
    report, snapshots, patcher = _reporter_with_log_snapshots()
    report('Working on it.', 10)
    report('Still working.', 50)
    patcher.stop()

    opening_log, first_progress_log, second_progress_log = snapshots
    assert len(opening_log) == 1
    assert len(first_progress_log) == 2
    assert len(second_progress_log) == 3
    assert second_progress_log[:2] == first_progress_log
    assert second_progress_log[-1].endswith('Still working.')


def test_a_failure_still_appends_instead_of_collapsing():
    report, snapshots, patcher = _reporter_with_log_snapshots()
    report('Working on it.', 10)
    report('It broke.', 100, task_state=TASK_STATUS_FAIL)
    patcher.stop()

    assert len(snapshots[-1]) == 3
    assert snapshots[-1][-1].endswith('It broke.')


def test_a_terminal_state_still_appends_because_the_queue_collapses_the_log():
    report, snapshots, patcher = _reporter_with_log_snapshots()
    report('Working on it.', 10)
    report('All done.', 100, task_state=TASK_STATUS_SUCCESS)
    patcher.stop()

    assert len(snapshots[-1]) == 3
    assert snapshots[-1][-1].endswith('All done.'), (
        'the one-line "completed" collapse happens in the terminal row the queue '
        'writes, not here; the reporter only ever narrates'
    )


def test_the_log_handed_to_every_write_never_exceeds_the_shared_cap():
    report, snapshots, patcher = _reporter_with_log_snapshots()
    for i in range(MAX_LOG_ENTRIES_STORED + 7):
        report(f'Step {i}.', i)
    patcher.stop()

    for snapshot in snapshots:
        assert len(snapshot) <= MAX_LOG_ENTRIES_STORED
    assert len(snapshots[-1]) == MAX_LOG_ENTRIES_STORED
    assert snapshots[-1][-1].endswith(f'Step {MAX_LOG_ENTRIES_STORED + 6}.')


def test_the_opening_write_marks_the_task_running():
    with patch('tasks.task_run.save_task_status') as save:
        _reporter()

    assert save.call_args_list[0][0][2] == TASK_STATUS_RUNNING


def test_a_progress_line_is_both_message_and_status_message():
    with patch('tasks.task_run.save_task_status') as save:
        report = _reporter()
        report('Analysing album 3.', 30)

    details = save.call_args_list[-1][1]['details']
    assert details['message'] == 'Analysing album 3.'
    assert details['status_message'] == 'Analysing album 3.'


def test_throttling_skips_a_running_write():
    with patch('tasks.task_run.save_task_status') as save:
        report = _reporter(min_db_interval=3600)
        report('Still going.', 10)
        after_first = len(save.call_args_list)
        report('Still going, really.', 20)
        assert len(save.call_args_list) == after_first, 'a throttled progress write is skipped'


def test_a_forced_write_bypasses_the_throttle():
    with patch('tasks.task_run.save_task_status') as save:
        report = _reporter(min_db_interval=3600)
        report('Still going.', 10)
        after_first = len(save.call_args_list)
        report('Failed to analyze album X.', 10, error={'error_code': 'x'}, force=True)

    assert len(save.call_args_list) == after_first + 1, (
        'the one line a child writes before it raises must land whatever the throttle says'
    )
    assert 'force' not in save.call_args_list[-1][1]['details']


def test_progress_is_rescaled_into_the_phase_window():
    with patch('tasks.task_run.save_task_status') as save:
        report = _reporter(progress_base=50.0, progress_span=50.0)
        report('Half of the second half.', 50)

    assert save.call_args_list[-1][1]['progress'] == 75


@pytest.mark.parametrize('terminal', [TASK_STATUS_SUCCESS, TASK_STATUS_FAIL, TASK_STATUS_REVOKED])
def test_every_terminal_state_is_downgraded_to_running(terminal, caplog):
    with patch('tasks.task_run.save_task_status') as save:
        report = _reporter()
        with caplog.at_level(logging.ERROR, logger='tasks.task_run'):
            report('Phase done.', 100, task_state=terminal)

    assert save.call_args_list[-1][0][2] == TASK_STATUS_RUNNING, (
        'a task that wrote its own terminal row used to win the race against the '
        'queue and silently veto its verdict and its retry; the row is the queue'
    )
    assert any('belongs to the queue' in record.getMessage() for record in caplog.records)


class TestTheSharedServerLoop:
    @staticmethod
    def _loop(monkeypatch, step):
        from contextlib import nullcontext
        from tasks import task_run
        from tasks.mediaserver import registry

        servers = [{'server_id': 's1', 'name': 'One'}, {'server_id': 's2', 'name': 'Two'}]
        monkeypatch.setattr(registry, 'servers_for_scope', lambda _scope, conn=None: servers)
        monkeypatch.setattr(registry, 'bind', lambda _server, conn=None: nullcontext())
        return task_run.for_each_server_in_scope('all', step)

    def test_a_plain_error_on_one_server_names_it_and_the_loop_goes_on(self, monkeypatch):
        def step(server, name):
            if name == 'One':
                raise RuntimeError('provider down')
            return name

        _servers, results, failed = self._loop(monkeypatch, step)

        assert results == ['Two']
        assert failed == ['One']

    def test_a_declared_permanent_failure_passes_through_the_loop(self, monkeypatch):
        from taskqueue import TaskFailed

        def step(server, name):
            raise TaskFailed('unsupported media-server type')

        with pytest.raises(TaskFailed):
            self._loop(monkeypatch, step)
