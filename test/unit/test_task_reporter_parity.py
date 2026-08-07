# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""What a task reporter is allowed to persist about itself.

Every progress tick appends a timestamped line to the report's own `log`, the
same way the clustering and cleaning main tasks do. The reporter caps that log
to MAX_LOG_ENTRIES_STORED itself, in place, before it is ever handed to
save_task_status - so nothing bigger than the last 10 lines is ever built into
what gets sent toward the database, not even for the instant before some
downstream layer would trim it. A terminal SUCCESS collapses the log to one
line instead: the run is over, and the container log already has the
narration in full.

Main Features:
* The opening write already carries a one-line log with the initial message
* Every non-terminal report appends a new timestamped line to the same log
* A terminal SUCCESS collapses the log to a single "completed" line
* The log handed to save_task_status never exceeds MAX_LOG_ENTRIES_STORED
  entries, capped by the reporter itself rather than left to a later layer
* Progress writes carry the current line as both message and status_message
* The DB write is throttled while running but never for a terminal status
"""

import os
import sys
from unittest.mock import patch

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import (  # noqa: E402
    TASK_STATUS_FAIL,
    TASK_STATUS_RUNNING,
    TASK_STATUS_SUCCESS,
)
from database import MAX_LOG_ENTRIES_STORED  # noqa: E402


def _reporter(**kwargs):
    from tasks.analysis.helper import make_task_reporter

    return make_task_reporter('task-1', 'main_analysis', 'Started.', **kwargs)


def _reporter_with_log_snapshots(**kwargs):
    # `logs` inside the reporter is ONE mutable list, appended to (or trimmed) in
    # place and handed to save_task_status by reference every time - a mock's
    # call_args_list stores that same reference, so inspecting an EARLIER call
    # after the fact would show the list's FINAL state, not what it looked like
    # when that call actually happened. A copy of `log` taken inside the fake
    # save_task_status, at the instant of the call, is the only faithful record.
    from tasks.analysis.helper import make_task_reporter

    snapshots = []

    def fake_save_task_status(*_args, **call_kwargs):
        snapshots.append(list(call_kwargs['details'].get('log') or []))

    patcher = patch('tasks.analysis.helper.save_task_status', side_effect=fake_save_task_status)
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


def test_a_terminal_success_collapses_the_log_to_one_line():
    report, snapshots, patcher = _reporter_with_log_snapshots()
    report('Working on it.', 10)
    report('All done.', 100, task_state=TASK_STATUS_SUCCESS)
    patcher.stop()

    assert snapshots[-1] == ['Task completed successfully. Final status: All done.']


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
    with patch('tasks.analysis.helper.save_task_status') as save:
        _reporter()

    assert save.call_args_list[0][0][2] == TASK_STATUS_RUNNING


def test_a_progress_line_is_both_message_and_status_message():
    with patch('tasks.analysis.helper.save_task_status') as save:
        report = _reporter()
        report('Analysing album 3.', 30)

    details = save.call_args_list[-1][1]['details']
    assert details['message'] == 'Analysing album 3.'
    assert details['status_message'] == 'Analysing album 3.'


def test_throttling_skips_a_running_write_but_never_a_terminal_one():
    with patch('tasks.analysis.helper.save_task_status') as save:
        report = _reporter(min_db_interval=3600)
        report('Still going.', 10)
        after_first = len(save.call_args_list)
        report('Still going, really.', 20)
        assert len(save.call_args_list) == after_first, 'a throttled progress write is skipped'
        report('Finished.', 100, task_state=TASK_STATUS_SUCCESS)
        assert len(save.call_args_list) == after_first + 1, 'a terminal write is never throttled'


def test_progress_is_rescaled_into_the_phase_window():
    with patch('tasks.analysis.helper.save_task_status') as save:
        report = _reporter(progress_base=50.0, progress_span=50.0)
        report('Half of the second half.', 50)

    assert save.call_args_list[-1][1]['progress'] == 75


def test_downgrade_terminal_keeps_a_non_final_phase_running():
    with patch('tasks.analysis.helper.save_task_status') as save:
        report = _reporter(downgrade_terminal=True)
        report('Phase done.', 100, task_state=TASK_STATUS_SUCCESS)

    assert save.call_args_list[-1][0][2] == TASK_STATUS_RUNNING
