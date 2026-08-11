# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Which parentless rows may gate a main task, stated once on both sides.

Two layers answer this and they must agree. Postgres enforces "at most one live
main task" with a partial unique index over an INCLUSION list, while the start
endpoints pre-check with an EXCLUSION list. A type missing from the exclusion
list still 409s a Start even though the index would admit it - and worse,
clean_up_previous_main_tasks then REVOKES that live run when the next start goes
through. Cron-triggered sonic-fingerprint and plugin roots are exactly that case.

Main Features:
* Every self-managed type reaches the query as an exclusion, prefixes included
* Plugin roots are excluded by prefix because their type is dynamic
* Cleanup and the start gate exclude the same set, so neither can revoke the other
* No type is claimed by both the inclusion index and the exclusion list
"""

from unittest.mock import MagicMock, patch

import pytest

import database
from taskqueue import sql


def _params_of(call):
    return call[0][1]


def _query_of(call):
    return call[0][0]


def _active_main_task_call():
    cur = MagicMock()
    cur.fetchone.return_value = None
    db = MagicMock()
    db.cursor.return_value = cur
    with patch('database.get_db', return_value=db):
        assert database.get_active_main_task() is None
    return cur.execute.call_args


class TestASelfManagedRootNeverGatesAMainStart:
    @pytest.mark.parametrize('task_type', database.SELF_MANAGED_TASK_TYPES)
    def test_each_named_type_is_excluded_from_the_gate(self, task_type):
        params = _params_of(_active_main_task_call())
        excluded = next(param for param in params if isinstance(param, list))

        assert task_type in excluded

    def test_the_fingerprint_root_still_gates_a_main_start(self):
        assert 'sonic_fingerprint' not in database.SELF_MANAGED_TASK_TYPES, (
            'a running fingerprint blocked an analysis or clustering start on main; '
            'excluding it here let the two run concurrently over the same catalogue'
        )


class TestAPluginRootIsExcludedByPrefixBecauseItsTypeIsDynamic:
    def test_the_gate_matches_the_plugin_prefix(self):
        call = _active_main_task_call()

        assert 'NOT LIKE' in _query_of(call)
        assert 'plugin.%' in _params_of(call)

    def test_the_prefix_list_names_plugins(self):
        assert 'plugin.' in database.SELF_MANAGED_TASK_TYPE_PREFIXES


class TestCleanupExcludesExactlyWhatTheGateExcludes:
    def test_the_archive_pass_skips_the_same_types_and_prefixes(self):
        cur = MagicMock()
        cur.fetchall.return_value = []
        cur.fetchone.return_value = None
        db = MagicMock()
        db.cursor.return_value = cur
        with patch('database.get_db', return_value=db):
            database.clean_up_previous_main_tasks()

        archive_calls = [
            call for call in cur.execute.call_args_list
            if 'FROM task_status' in _query_of(call) and len(call[0]) > 1
            and any(isinstance(param, list) for param in _params_of(call))
        ]
        assert archive_calls, 'the archive pass must run a parameterised select'
        call = archive_calls[0]
        excluded = next(param for param in _params_of(call) if isinstance(param, list))

        assert set(database.SELF_MANAGED_TASK_TYPES) <= set(excluded), (
            'a type the start gate lets through but cleanup archives would be '
            'REVOKED mid-run by the very next Start'
        )
        assert 'plugin.%' in _params_of(call)


class TestTheTwoLayersNeverClaimTheSameType:
    def test_no_self_managed_type_is_also_a_main_task_type(self):
        overlap = set(database.SELF_MANAGED_TASK_TYPES) & set(sql.MAIN_TASK_TYPES)

        assert not overlap, (
            f'{overlap} would be admitted by the one-live-main index and refused '
            'by the endpoint pre-check at the same time'
        )

    def test_no_main_task_type_starts_with_a_self_managed_prefix(self):
        for prefix in database.SELF_MANAGED_TASK_TYPE_PREFIXES:
            offenders = [name for name in sql.MAIN_TASK_TYPES if name.startswith(prefix)]

            assert not offenders
