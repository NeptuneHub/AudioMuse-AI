# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Single source of truth for shaping the `log` list stored in task_status.details.

Every long-running task used to hand-roll the same two rules (collapse to a
one-line recap once the task succeeds, keep only a bounded tail otherwise), which
is how main_clustering ended up appending to an uncapped list on every progress
update. This module owns those rules so analysis, cleaning and clustering share
one implementation.

Deliberately a leaf module: it imports nothing from the project (not even
config), because test/unit/test_import_architecture.py caps the eager import
chain at 5 and this module is pulled in from both ends of that chain. Caps are
passed in by the caller rather than read from config for the same reason.

Main Features:
* `success_recap` builds the single line kept when a task finishes OK
* `append_capped` appends a timestamped entry and trims the list in place
* `shape_log` applies the OK/KO rule in one call for a task reporter
"""

import time

DEFAULT_LOG_CAP = 200

SUCCESS_RECAP_PREFIX = "Task completed successfully. Final status: "


def stamp(message):
    return f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"


def success_recap(message):
    return [f"{SUCCESS_RECAP_PREFIX}{message}"]


def append_capped(logs, message, cap=DEFAULT_LOG_CAP):
    logs.append(stamp(message))
    if cap and cap > 0 and len(logs) > cap:
        del logs[:-cap]
    return logs


def shape_log(logs, message, succeeded, cap=DEFAULT_LOG_CAP):
    if succeeded:
        return success_recap(message)
    return append_capped(logs, message, cap)
