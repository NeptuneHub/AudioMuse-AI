# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for global RQ and task-status cancellation orchestration.

Main Features:
* Verifies global cancellation delegates queue, job, database, and recap work
* Verifies queue job identifiers are normalized from queue and Redis sources
"""

from unittest.mock import Mock, call, patch

import app_helper


def test_cancel_job_delegates_independent_cleanup_steps():
    with (
        patch.object(app_helper, "_discover_rq_job_ids", return_value={"a", "b"}) as discover,
        patch.object(app_helper, "_cancel_discovered_jobs", return_value=2) as cancel_jobs,
        patch.object(app_helper, "_clear_rq_queues") as clear_queues,
        patch.object(app_helper, "_clear_task_status") as clear_status,
        patch.object(app_helper, "_save_cancel_recap") as save_recap,
    ):
        result = app_helper.cancel_job_and_children_recursive("requested", reason="user stop")

    assert result == 2
    discover.assert_called_once_with()
    cancel_jobs.assert_called_once_with({"a", "b"})
    clear_queues.assert_called_once_with()
    clear_status.assert_called_once_with()
    save_recap.assert_called_once_with("requested", "user stop")


def test_discover_rq_job_ids_combines_queue_and_started_job_keys():
    high = Mock(job_ids=["queued", None])
    high.name = "high"
    default = Mock(job_ids=None)
    default.name = "default"

    with (
        patch.object(app_helper, "rq_queue_high", high),
        patch.object(app_helper, "rq_queue_default", default),
        patch.object(
            app_helper.redis_conn, "lrange", return_value=[b"fallback"]
        ) as lrange,
        patch.object(app_helper.redis_conn, "keys", return_value=[b"rq:job:started:child"]),
    ):
        assert app_helper._discover_rq_job_ids() == {
            "queued",
            "fallback",
            "started:child",
        }

    assert lrange.call_args == call("rq:queue:default", 0, -1)
