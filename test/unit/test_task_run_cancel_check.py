# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The one cancel check stops a task whose own row is already terminal.

The check used to stop a task only when its row was gone or REVOKED. A parent
that gives up on a child writes that child's row as FAIL, so the child kept
running until it noticed its parent was over, holding the one default worker
its siblings needed, and when it finally returned the queue logged it as a
task that had written its own terminal row. A terminal own row means the
verdict is already in, whoever wrote it, and there is nothing left to report.

Main Features:
* Any terminal own row stops the task, not only REVOKED
* A missing own row still stops it, which is what the global cancel leaves
* A supervised child stops when its parent is terminal or gone
* A live task under a live parent goes on, and a root watches only itself
"""

import pytest

from config import TASK_STATUS_RUNNING, TASK_STATUS_TERMINAL
from taskqueue import TaskCancelled
from tasks import task_run


class TestTheCancelCheckReadsTheOwnRowAndTheParent:
    @pytest.mark.parametrize('status', TASK_STATUS_TERMINAL)
    def test_a_terminal_own_row_stops_the_task(self, status):
        with pytest.raises(TaskCancelled):
            task_run._raise_if_cancelled('t1', None, {'t1': status})

    def test_a_missing_own_row_stops_the_task(self):
        with pytest.raises(TaskCancelled):
            task_run._raise_if_cancelled('t1', None, {})

    @pytest.mark.parametrize('status', TASK_STATUS_TERMINAL)
    def test_a_terminal_parent_stops_a_supervised_child(self, status):
        with pytest.raises(TaskCancelled):
            task_run._raise_if_cancelled(
                'c1', 'p1', {'c1': TASK_STATUS_RUNNING, 'p1': status}
            )

    def test_a_missing_parent_stops_a_supervised_child(self):
        with pytest.raises(TaskCancelled):
            task_run._raise_if_cancelled('c1', 'p1', {'c1': TASK_STATUS_RUNNING})

    def test_a_live_task_under_a_live_parent_goes_on(self):
        task_run._raise_if_cancelled(
            'c1', 'p1', {'c1': TASK_STATUS_RUNNING, 'p1': TASK_STATUS_RUNNING}
        )

    def test_a_root_watches_only_itself(self):
        task_run._raise_if_cancelled('r1', None, {'r1': TASK_STATUS_RUNNING})
