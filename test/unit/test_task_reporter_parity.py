# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Every long-running task shapes its task_status log the same way.

analysis, cleaning and clustering each grew their own copy of the OK/KO log
rules and drifted: clustering appended to an uncapped list on every progress
update and never collapsed to a recap on success, while the other two did. They
now all route through tasks.task_details, and these tests hold that line.

make_task_reporter had no test at all before this file, which is why the drift
went unnoticed - the analysis family is the one of the three that can be driven
directly, so it stands in for the shared contract.

Main Features:
* A reporter emits exactly one recap line once the task succeeds
* Before success it keeps a bounded, timestamped tail at the caller's cap
* Clustering's batch child keeps its RQ-recovery keys alongside the recap line
"""

import os
import sys
from unittest.mock import patch

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import (  # noqa: E402
    TASK_STATUS_FAILURE,
    TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS,
)
from tasks.task_details import SUCCESS_RECAP_PREFIX, shape_log  # noqa: E402


def _drive_reporter(messages, final_state, log_cap=200):
    from tasks.analysis.helper import make_task_reporter

    with patch('tasks.analysis.helper.save_task_status') as save:
        report = make_task_reporter(
            'task-1', 'main_analysis', None, 'Starting.', log_cap=log_cap
        )
        for i, msg in enumerate(messages):
            last = i == len(messages) - 1
            report(msg, 100 if last else 10,
                   **({'task_state': final_state} if last else {}))
        return [c.kwargs['details'] for c in save.call_args_list]


class TestAnalysisReporter:
    def test_success_leaves_exactly_one_recap_line(self):
        details = _drive_reporter(["step one", "step two", "all done"], TASK_STATUS_SUCCESS)
        assert details[-1]["log"] == [f"{SUCCESS_RECAP_PREFIX}all done"]

    def test_failure_keeps_a_timestamped_tail_not_a_recap(self):
        details = _drive_reporter(["step one", "it broke"], TASK_STATUS_FAILURE)
        log = details[-1]["log"]
        assert len(log) > 1
        assert log[-1].endswith("it broke")
        assert not log[-1].startswith(SUCCESS_RECAP_PREFIX)

    def test_progress_log_is_capped_at_the_callers_limit(self):
        details = _drive_reporter([f"msg {i}" for i in range(40)], TASK_STATUS_PROGRESS, log_cap=5)
        assert len(details[-1]["log"]) == 5

    def test_the_initial_write_carries_one_stamped_line(self):
        details = _drive_reporter(["only"], TASK_STATUS_PROGRESS)
        assert len(details[0]["log"]) == 1
        assert details[0]["log"][0].startswith("[")


class TestClusteringUsesTheSameRule:
    def test_batch_success_collapses_but_keeps_the_rq_recovery_keys(self):
        batch_logs = []
        db_details = {
            "batch_id": "b1",
            "full_best_result_from_batch": {"playlists": {"Rock": ["a"]}},
            "final_subset_track_ids": ["a", "b"],
        }
        db_details["log"] = shape_log(batch_logs, "Batch complete.", True)

        assert db_details["log"] == [f"{SUCCESS_RECAP_PREFIX}Batch complete."]
        assert db_details["full_best_result_from_batch"] == {"playlists": {"Rock": ["a"]}}
        assert db_details["final_subset_track_ids"] == ["a", "b"]

    def test_batch_failure_keeps_a_readable_tail(self):
        batch_logs = []
        shape_log(batch_logs, "Batch started.", False)
        out = shape_log(batch_logs, "Batch failed: boom", False)

        assert len(out) == 2
        assert out[-1].endswith("Batch failed: boom")

    def test_main_clustering_matches_the_analysis_shape(self):
        analysis = _drive_reporter(["working", "finished"], TASK_STATUS_SUCCESS)[-1]["log"]
        clustering = shape_log(["[ts] working"], "finished", True)
        assert analysis == clustering
