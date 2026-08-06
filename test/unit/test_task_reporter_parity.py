# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""What a task reporter is allowed to persist about itself.

The reporter used to accumulate a `log` list in the row, rewritten on every
progress tick and capped in three separate places. It now writes one
status_message while running and one message when it ends; the narration goes to
the container log instead.

Main Features:
* A reporter never writes a `log` key, at any status
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


def _reporter(**kwargs):
    from tasks.analysis.helper import make_task_reporter

    return make_task_reporter('task-1', 'main_analysis', 'Started.', **kwargs)


def test_no_write_at_any_status_carries_a_log_key():
    with patch('tasks.analysis.helper.save_task_status') as save:
        report = _reporter()
        report('Working on it.', 10)
        report('All done.', 100, task_state=TASK_STATUS_SUCCESS)
        report('It broke.', 100, task_state=TASK_STATUS_FAIL)

    assert save.call_args_list, 'the reporter must write something'
    for call in save.call_args_list:
        details = call[1]['details']
        assert 'log' not in details, f"a log list leaked back into {details}"


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
