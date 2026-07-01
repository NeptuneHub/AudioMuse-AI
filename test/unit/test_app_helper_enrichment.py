# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Song-feature enrichment in app_helper.attach_song_features.

Covers how score data is looked up by id and merged into result rows, including
the id-key coercion and top-genre derivation from the mood vector.

Main Features:
* Short-circuits on empty/None input or rows lacking usable ids (no DB lookup)
* Int/str id coercion so row ids match score-data keys either way
* Existing values are preserved, non-dict entries pass through, top_genre
  ignores non-stratified labels
"""

from unittest.mock import patch

from app_helper import attach_song_features


def make_score(
    item_id, album='Album', mood_vector='rock:0.9,happy:0.5', other_features='danceable:0.4'
):
    return {
        'item_id': item_id,
        'album': album,
        'mood_vector': mood_vector,
        'other_features': other_features,
    }


class TestAttachSongFeaturesShortCircuits:
    def test_empty_list_returned_without_lookup(self):
        with patch('app_helper.get_score_data_by_ids') as mock_lookup:
            assert attach_song_features([]) == []
            mock_lookup.assert_not_called()

    def test_none_returned_without_lookup(self):
        with patch('app_helper.get_score_data_by_ids') as mock_lookup:
            assert attach_song_features(None) is None
            mock_lookup.assert_not_called()

    def test_rows_without_usable_ids_skip_lookup(self):
        rows = [{'title': 'x'}, {'item_id': None}, {'item_id': ''}, 'notadict']
        with patch('app_helper.get_score_data_by_ids') as mock_lookup:
            result = attach_song_features(rows)
            assert result is rows
            assert result == [{'title': 'x'}, {'item_id': None}, {'item_id': ''}, 'notadict']
            mock_lookup.assert_not_called()


class TestAttachSongFeaturesEnrichment:
    def test_int_row_id_matches_str_score_key(self):
        rows = [{'item_id': 123}]
        with patch('app_helper.get_score_data_by_ids', return_value=[make_score('123')]):
            result = attach_song_features(rows)
        assert result[0]['album'] == 'Album'
        assert result[0]['mood_vector'] == 'rock:0.9,happy:0.5'
        assert result[0]['other_features'] == 'danceable:0.4'
        assert result[0]['top_genre'] == 'rock'

    def test_str_row_id_matches_int_score_key(self):
        rows = [{'item_id': '456'}]
        score = make_score(456, album='B', mood_vector=None, other_features=None)
        with patch('app_helper.get_score_data_by_ids', return_value=[score]):
            result = attach_song_features(rows)
        assert result[0]['album'] == 'B'
        assert result[0]['mood_vector'] is None
        assert result[0]['other_features'] is None
        assert result[0]['top_genre'] is None

    def test_existing_values_not_overwritten(self):
        rows = [{'item_id': 1, 'album': 'Keep'}]
        with patch('app_helper.get_score_data_by_ids', return_value=[make_score(1, album='New')]):
            result = attach_song_features(rows)
        assert result[0]['album'] == 'Keep'
        assert result[0]['mood_vector'] == 'rock:0.9,happy:0.5'

    def test_non_dict_entries_pass_through_untouched(self):
        rows = [{'item_id': 1}, 'plain-string', None]
        with patch('app_helper.get_score_data_by_ids', return_value=[make_score(1)]) as mock_lookup:
            result = attach_song_features(rows)
        assert result[1] == 'plain-string'
        assert result[2] is None
        assert result[0]['album'] == 'Album'
        mock_lookup.assert_called_once_with([1])

    def test_empty_score_data_leaves_rows_unenriched(self):
        rows = [{'item_id': 5, 'title': 't'}]
        with patch('app_helper.get_score_data_by_ids', return_value=[]):
            result = attach_song_features(rows)
        assert result is rows
        assert result[0] == {'item_id': 5, 'title': 't'}

    def test_row_without_matching_score_left_unchanged(self):
        rows = [{'item_id': 1}, {'item_id': 2}]
        with patch('app_helper.get_score_data_by_ids', return_value=[make_score(1)]):
            result = attach_song_features(rows)
        assert result[0]['album'] == 'Album'
        assert result[1] == {'item_id': 2}

    def test_lookup_receives_only_truthy_dict_ids(self):
        rows = [{'item_id': 1}, {'item_id': 0}, 'x', {'item_id': 2}]
        with patch('app_helper.get_score_data_by_ids', return_value=[]) as mock_lookup:
            attach_song_features(rows)
        mock_lookup.assert_called_once_with([1, 2])

    def test_custom_id_key(self):
        rows = [{'id': 9}]
        with patch(
            'app_helper.get_score_data_by_ids', return_value=[make_score(9, album='C')]
        ) as mock_lookup:
            result = attach_song_features(rows, id_key='id')
        mock_lookup.assert_called_once_with([9])
        assert result[0]['album'] == 'C'

    def test_top_genre_ignores_non_stratified_labels(self):
        rows = [{'item_id': 7}]
        score = make_score(7, mood_vector='female vocalist:0.99,metal:0.5')
        with patch('app_helper.get_score_data_by_ids', return_value=[score]):
            result = attach_song_features(rows)
        assert result[0]['top_genre'] == 'metal'
