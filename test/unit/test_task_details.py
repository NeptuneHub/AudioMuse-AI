# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared task_status log shaping rules used by analysis, cleaning and clustering.

These rules used to be copy-pasted per task, which is how main_clustering ended
up appending to an uncapped list on every progress update while the other two
capped theirs. The tests pin the two behaviours every caller now relies on: a
successful task keeps exactly one recap line, and an unfinished or failed one
keeps a bounded tail of the newest entries in the caller's own list object.

Main Features:
* Success collapses the log to a single recap line regardless of history length
* Non-success appends a timestamped entry and trims to the cap, newest kept
* The caller's list is mutated in place so a reporter keeps accumulating
"""

import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tasks.task_details import (  # noqa: E402
    DEFAULT_LOG_CAP,
    SUCCESS_RECAP_PREFIX,
    append_capped,
    shape_log,
    stamp,
    success_recap,
)


class TestSuccessRecap:
    def test_success_collapses_a_long_log_to_a_single_line(self):
        logs = [f"line {i}" for i in range(500)]
        out = shape_log(logs, "All done", True)
        assert out == [f"{SUCCESS_RECAP_PREFIX}All done"]
        assert len(out) == 1

    def test_success_recap_carries_the_final_message(self):
        assert success_recap("Playlists created") == [
            f"{SUCCESS_RECAP_PREFIX}Playlists created"
        ]

    def test_success_does_not_depend_on_prior_history(self):
        assert shape_log([], "done", True) == shape_log(["a", "b"], "done", True)


class TestBoundedTail:
    def test_non_success_appends_a_timestamped_entry(self):
        logs = []
        shape_log(logs, "Working", False)
        assert len(logs) == 1
        assert logs[0].endswith("Working")
        assert logs[0].startswith("[")

    def test_non_success_trims_to_the_cap_keeping_the_newest(self):
        logs = []
        for i in range(25):
            append_capped(logs, f"msg {i}", cap=10)
        assert len(logs) == 10
        assert logs[-1].endswith("msg 24")
        assert logs[0].endswith("msg 15")

    def test_default_cap_bounds_an_otherwise_unbounded_reporter(self):
        logs = []
        for i in range(DEFAULT_LOG_CAP + 50):
            append_capped(logs, f"msg {i}")
        assert len(logs) == DEFAULT_LOG_CAP

    def test_a_zero_or_negative_cap_disables_trimming(self):
        logs = []
        for i in range(5):
            append_capped(logs, f"msg {i}", cap=0)
        assert len(logs) == 5

    def test_non_success_mutates_the_callers_list_in_place(self):
        logs = ["existing"]
        out = shape_log(logs, "next", False)
        assert out is logs
        assert len(logs) == 2


class TestStamp:
    def test_stamp_prefixes_a_bracketed_timestamp(self):
        out = stamp("hello")
        assert out.endswith(" hello")
        assert out.startswith("[")
        assert "]" in out
