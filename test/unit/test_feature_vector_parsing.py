# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Label-to-column resolution and malformed-row tolerance in the clustering parsers.

Both parsers here run once per track per clustering iteration, so both trade a
linear label scan for a lookup. These tests pin the two behaviours that trade
could silently change: which column a repeated label resolves to, and what a
corrupt mood string does to the run.

Main Features:
* score_vector resolves a duplicated label to its FIRST position, matching the
  list.index() it replaced.
* A malformed or non-numeric score is skipped, never raised, so one corrupt row
  cannot fail a whole clustering batch.
* A score carrying an extra colon still reads the value between the first and
  second colon.
"""

from unittest.mock import patch

from tasks.clustering_helper import _get_track_primary_genre
from tasks.commons import score_vector


class TestDuplicateLabelResolution:
    def test_repeated_label_resolves_to_its_first_column(self):
        labels = ['rock', 'pop', 'rock']
        row = {'tempo': 120.0, 'energy': 0.5, 'mood_vector': 'rock:0.9', 'other_features': ''}

        moods = score_vector(row, labels, [])[2:]

        assert moods == [0.9, 0.0, 0.0], "a duplicated label must take its first index"

    def test_repeated_other_feature_resolves_to_its_first_column(self):
        row = {
            'tempo': 120.0,
            'energy': 0.5,
            'mood_vector': '',
            'other_features': 'danceable:0.7',
        }

        others = score_vector(row, [], ['danceable', 'happy', 'danceable'])[2:]

        assert others == [0.7, 0.0, 0.0]

    def test_distinct_label_lists_do_not_share_a_cached_index_map(self):
        row = {'tempo': 120.0, 'energy': 0.5, 'mood_vector': 'pop:0.4', 'other_features': ''}

        first = score_vector(row, ['rock', 'pop'], [])[2:]
        second = score_vector(row, ['pop', 'rock'], [])[2:]

        assert first == [0.0, 0.4]
        assert second == [0.4, 0.0]


class TestMalformedScoreTolerance:
    def test_non_numeric_score_is_skipped_not_raised(self):
        row = {
            'tempo': 120.0,
            'energy': 0.5,
            'mood_vector': 'rock:abc,pop:0.5',
            'other_features': '',
        }

        assert score_vector(row, ['rock', 'pop'], [])[2:] == [0.0, 0.5]

    def test_pair_without_a_colon_is_skipped(self):
        row = {'tempo': 120.0, 'energy': 0.5, 'mood_vector': 'rock,pop:0.5', 'other_features': ''}

        assert score_vector(row, ['rock', 'pop'], [])[2:] == [0.0, 0.5]

    @patch('tasks.clustering_helper.STRATIFIED_GENRES', ['rock', 'pop'])
    def test_corrupt_genre_score_does_not_fail_the_run(self):
        assert _get_track_primary_genre({'mood_vector': 'rock:abc'}) == '__other__'
        assert _get_track_primary_genre({'mood_vector': 'rock:'}) == '__other__'
        assert _get_track_primary_genre({'mood_vector': 'rock:abc,pop:0.5'}) == 'pop'

    @patch('tasks.clustering_helper.STRATIFIED_GENRES', ['rock', 'pop'])
    def test_extra_colon_keeps_the_value_between_the_first_two(self):
        assert _get_track_primary_genre({'mood_vector': 'rock:0.9:extra,pop:0.2'}) == 'rock'

    @patch('tasks.clustering_helper.STRATIFIED_GENRES', ['synthwave', 'vaporwave'])
    def test_genre_set_follows_a_patched_stratified_list(self):
        # Genres absent from the shipped list: a set frozen at import would miss
        # them entirely and fall through to __other__.
        assert _get_track_primary_genre({'mood_vector': 'synthwave:0.2,rock:0.9'}) == 'synthwave'

    @patch('tasks.clustering_helper.STRATIFIED_GENRES', ['rock', 'pop'])
    def test_tie_resolves_in_stratified_list_order(self):
        assert _get_track_primary_genre({'mood_vector': 'pop:0.5,rock:0.5'}) == 'rock'

    def test_absent_or_empty_mood_vector_is_other(self):
        assert _get_track_primary_genre({}) == '__other__'
        assert _get_track_primary_genre({'mood_vector': None}) == '__other__'
        assert _get_track_primary_genre({'mood_vector': ''}) == '__other__'
