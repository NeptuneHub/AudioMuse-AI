# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Task-note summary strings built by app_helper._build_task_note.

Covers the per-task-type note produced for analysis, cleaning, and clustering.
The note is built from the parent row's own details only: there is no subtask
query any more, so the builder takes no database handle at all and the tests
that used to feed it a failing connection have nothing left to assert.

Main Features:
* Analysis: always reports the parent's own tracks_analyzed, zero included
* Cleaning: first recognized numeric key wins, floats truncated, zero reported
* Clustering: sampled/cluster counts from best_params with graceful fallbacks;
  unknown task types and non-dict details yield an empty string
* The signature is pinned so a database lookup cannot creep back in
"""

import inspect

import pytest

from database import _build_task_note


class TestAnalysisNote:

    def test_albums_completed_alone_reports_zero_songs(self):
        result = _build_task_note('main_analysis', {'albums_completed': 7})
        assert result == 'Songs analyzed: 0'

    def test_total_albums_processed_alone_reports_zero_songs(self):
        result = _build_task_note('main_analysis', {'total_albums_processed': 12})
        assert result == 'Songs analyzed: 0'

    def test_reports_zero_when_nothing_to_report(self):
        assert _build_task_note('main_analysis', {}) == 'Songs analyzed: 0'


class TestCleanNote:
    @pytest.mark.parametrize(
        'key',
        [
            'tracks_deleted',
            'orphans_removed',
            'songs_cleaned',
            'tracks_removed',
            'deleted_count',
            'cleaned_tracks',
        ],
    )
    def test_each_recognized_key(self, key):
        assert _build_task_note('main_cleaning', {key: 6}) == 'Songs cleaned: 6'

    def test_first_key_wins(self):
        details = {'tracks_deleted': 2, 'orphans_removed': 9}
        assert _build_task_note('main_cleaning', details) == 'Songs cleaned: 2'

    def test_zero_is_reported(self):
        assert _build_task_note('main_cleaning', {'tracks_deleted': 0}) == 'Songs cleaned: 0'

    def test_string_values_skipped_in_favor_of_later_numeric_key(self):
        details = {'tracks_deleted': '5', 'orphans_removed': 3}
        assert _build_task_note('main_cleaning', details) == 'Songs cleaned: 3'

    def test_float_value_truncated(self):
        assert _build_task_note('main_cleaning', {'songs_cleaned': 4.7}) == 'Songs cleaned: 4'

    def test_no_recognized_keys_returns_empty_string(self):
        assert _build_task_note('main_cleaning', {'other': 1}) == ''


class TestClusterNote:
    def test_best_params_subset_size_preferred(self):
        details = {
            'best_params': {'initial_subset_size': 500},
            'sampled_songs': 1,
            'num_playlists_created': 8,
        }
        assert _build_task_note('main_clustering', details) == 'sampled: 500 | clusters: 8'

    def test_non_dict_best_params_falls_back_to_sampled_songs(self):
        details = {'best_params': 'oops', 'sampled_songs': 100}
        assert _build_task_note('main_clustering', details) == 'sampled: 100'

    def test_best_params_without_subset_size_falls_back(self):
        details = {'best_params': {}, 'num_sampled_songs': 50}
        assert _build_task_note('main_clustering', details) == 'sampled: 50'

    def test_clusters_only(self):
        assert _build_task_note('main_clustering', {'num_clusters': 4}) == 'clusters: 4'

    def test_zero_sampled_is_omitted(self):
        details = {'sampled_songs': 0, 'num_clusters': 3}
        assert _build_task_note('main_clustering', details) == 'clusters: 3'

    def test_no_data_returns_empty_string(self):
        assert _build_task_note('main_clustering', {}) == ''

    def test_non_numeric_sampled_returns_empty_string(self):
        details = {'sampled_songs': 'abc', 'num_clusters': 3}
        assert _build_task_note('main_clustering', details) == ''


class TestGeneralBehavior:
    def test_none_task_type_returns_empty_string(self):
        assert _build_task_note(None, {'tracks_deleted': 5}) == ''

    def test_unknown_task_type_returns_empty_string(self):
        assert _build_task_note('sonic_fingerprint', {'tracks_deleted': 5}) == ''

    def test_task_type_matching_is_case_insensitive(self):
        assert _build_task_note('MAIN_CLUSTERING', {'num_clusters': 2}) == 'clusters: 2'

    def test_non_dict_details_treated_as_empty(self):
        assert _build_task_note('main_cleaning', 'notadict') == ''


class TestAnalysisNoteReadsTheParentsOwnTally:

    def test_the_parents_own_track_count_is_reported(self):
        assert _build_task_note('main_analysis', {'tracks_analyzed': 4211}) == 'Songs analyzed: 4211'

    def test_the_builder_takes_no_database_handle_at_all(self):
        parameters = list(inspect.signature(_build_task_note).parameters)

        assert parameters == ['task_type', 'details_obj'], (
            'the note must not query for children that no longer exist; keeping a '
            'connection parameter is how that query creeps back in'
        )

    def test_a_zero_tally_is_still_reported_as_songs(self):
        note = _build_task_note('main_analysis', {'tracks_analyzed': 0, 'albums_completed': 12})

        assert note == 'Songs analyzed: 0'

    def test_a_float_tally_is_truncated(self):
        assert _build_task_note('main_analysis', {'tracks_analyzed': 9.9}) == 'Songs analyzed: 9'
