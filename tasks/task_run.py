# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared worker task-run plumbing: claim resolution and the terminal gate.

The prologue every worker task entry point repeats - resolve the running task
id, read its live row, and decide whether it was cancelled before execution or
already finished - lives here once instead of in each task body.

Main Features:
* task_run_prologue resolves the claimed id and reads its task_status row
* terminal_skip returns the revoked/terminal dict the entry point must return,
  or None when the task should actually run
* Everything about recovering a wedged run - the stall window, who a give-up
  ends, and the heartbeat a long opaque phase needs - lives in tasks.recovery.
  It was split across the two fan-out parents and they drifted; see that module.
"""

import logging
import uuid

import taskqueue
from config import (
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
    TASK_STATUS_REVOKED,
)
from database import get_task_info_from_db

logger = logging.getLogger(__name__)


def task_run_prologue(current_task_id=None):
    claimed_task_id = taskqueue.current_task_id()
    task_id = current_task_id or claimed_task_id or str(uuid.uuid4())
    return claimed_task_id, task_id, get_task_info_from_db(task_id)


def terminal_skip(
    task_id,
    claimed_task_id,
    task_info,
    *,
    revoked_message,
    terminal_message,
    terminal_details=None,
):
    if claimed_task_id and task_info is None:
        logger.info(
            "Task %s has no live DB claim; treating it as revoked.", task_id
        )
        return {"status": TASK_STATUS_REVOKED, "message": revoked_message}
    if task_info and task_info.get('status') in (
        TASK_STATUS_SUCCESS,
        TASK_STATUS_FAILURE,
        TASK_STATUS_REVOKED,
    ):
        logger.info(
            "Task %s is already terminal (%s); skipping.",
            task_id, task_info.get('status'),
        )
        result = {"status": task_info.get('status'), "message": terminal_message}
        if terminal_details is not None:
            result["details"] = terminal_details(task_info)
        return result
    return None
