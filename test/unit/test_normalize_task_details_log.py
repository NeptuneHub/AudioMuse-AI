# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The log a running task accumulates is capped at write time, not dropped.

``database._normalize_task_details`` is what lets a task keep appending a
timestamped line on every progress tick without the ``details`` column growing
without bound: once the list passes ``MAX_LOG_ENTRIES_STORED``, only the tail
survives, with a note recording how many were cut and how long the log really
was. A finished task collapses to a single line instead, unless the task
itself already supplied one.

Main Features:
* A log at or under the cap survives untouched, with no storage note
* A log over the cap is cut to the last MAX_LOG_ENTRIES_STORED entries
* The storage note names both the kept count and the original length
* A successful task gets a one-line log only when it did not supply its own
* A non-dict or list-less details value is left alone
"""

import database


class TestALogUnderTheCapIsLeftAlone:
    def test_a_log_at_exactly_the_cap_keeps_every_entry_and_no_note(self):
        details = {'log': [f'line {i}' for i in range(database.MAX_LOG_ENTRIES_STORED)]}

        database._normalize_task_details(details, 'PROGRESS')

        assert len(details['log']) == database.MAX_LOG_ENTRIES_STORED
        assert 'log_storage_info' not in details

    def test_a_stale_storage_note_is_cleared_once_the_log_shrinks_back_under_the_cap(self):
        details = {'log': ['only one line now'], 'log_storage_info': 'stale note'}

        database._normalize_task_details(details, 'PROGRESS')

        assert 'log_storage_info' not in details


class TestALogOverTheCapIsCutToTheTail:
    def test_only_the_last_entries_survive(self):
        lines = [f'line {i}' for i in range(database.MAX_LOG_ENTRIES_STORED + 16)]
        details = {'log': lines}

        database._normalize_task_details(details, 'PROGRESS')

        assert details['log'] == lines[-database.MAX_LOG_ENTRIES_STORED:]

    def test_the_storage_note_names_the_kept_count_and_the_original_length(self):
        lines = [f'line {i}' for i in range(database.MAX_LOG_ENTRIES_STORED + 16)]
        details = {'log': lines}

        database._normalize_task_details(details, 'PROGRESS')

        assert str(database.MAX_LOG_ENTRIES_STORED) in details['log_storage_info']
        assert str(len(lines)) in details['log_storage_info']


class TestASuccessfulTaskGetsAOneLineRecapOnlyWhenItSuppliedNone:
    def test_a_success_with_no_log_gets_the_generic_recap(self):
        details = {}

        database._normalize_task_details(details, database.TASK_STATUS_SUCCESS)

        assert details['log'] == ['Task completed successfully.']

    def test_a_success_with_an_empty_log_list_gets_the_generic_recap(self):
        details = {'log': []}

        database._normalize_task_details(details, database.TASK_STATUS_SUCCESS)

        assert details['log'] == ['Task completed successfully.']

    def test_a_success_that_already_supplied_its_own_log_keeps_it_uncut(self):
        # A long-running task hands its own tail of recent lines to the final
        # write (see tasks/clustering.py's final_db_summary); the generic
        # success recap must not clobber it even past the cap.
        lines = [f'line {i}' for i in range(database.MAX_LOG_ENTRIES_STORED + 5)]
        details = {'log': list(lines)}

        database._normalize_task_details(details, database.TASK_STATUS_SUCCESS)

        assert details['log'] == lines

    def test_a_success_clears_any_leftover_storage_note(self):
        details = {'log': ['final line'], 'log_storage_info': 'stale note'}

        database._normalize_task_details(details, database.TASK_STATUS_SUCCESS)

        assert 'log_storage_info' not in details


class TestNonListOrNonDictInputsAreLeftAlone:
    def test_a_non_dict_details_value_does_not_crash(self):
        database._normalize_task_details('not a dict', 'PROGRESS')  # must not raise

    def test_a_details_dict_with_no_log_key_at_all_is_left_alone(self):
        details = {'progress': 40}

        database._normalize_task_details(details, 'PROGRESS')

        assert 'log' not in details

    def test_a_non_list_log_value_is_left_alone(self):
        details = {'log': 'not a list'}

        database._normalize_task_details(details, 'PROGRESS')

        assert details['log'] == 'not a list'
