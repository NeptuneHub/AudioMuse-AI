import json
from unittest.mock import MagicMock

import pytest

from app_helper import _build_task_note


def make_db(rows):
    db = MagicMock()
    cur = MagicMock()
    cur.fetchall.return_value = rows
    db.cursor.return_value.__enter__.return_value = cur
    return db, cur


class TestAnalysisNote:
    def test_sums_tracks_analyzed_from_subtasks(self):
        db, cur = make_db(
            [
                (json.dumps({'tracks_analyzed': 10}),),
                (json.dumps({'tracks_analyzed': 5}),),
            ]
        )
        result = _build_task_note('main_analysis', {'_task_id': 'task-1'}, db)
        assert result == 'Songs analyzed: 15'
        assert cur.execute.call_args[0][1] == ('task-1',)

    def test_queries_with_empty_string_when_task_id_missing(self):
        db, cur = make_db([])
        _build_task_note('main_analysis', {}, db)
        assert cur.execute.call_args[0][1] == ('',)

    def test_skips_invalid_subtask_details(self):
        db, _ = make_db(
            [
                (None,),
                ('not json',),
                (json.dumps({'tracks_analyzed': '3'}),),
                (json.dumps([1, 2]),),
                (json.dumps({'tracks_analyzed': 4}),),
            ]
        )
        assert _build_task_note('main_analysis', {}, db) == 'Songs analyzed: 4'

    def test_float_track_counts_are_truncated_per_row(self):
        db, _ = make_db(
            [
                (json.dumps({'tracks_analyzed': 2.0}),),
                (json.dumps({'tracks_analyzed': 3.5}),),
            ]
        )
        assert _build_task_note('main_analysis', {}, db) == 'Songs analyzed: 5'

    def test_falls_back_to_albums_completed(self):
        db, _ = make_db([])
        result = _build_task_note('main_analysis', {'albums_completed': 7}, db)
        assert result == 'Albums analyzed: 7'

    def test_falls_back_to_total_albums_processed(self):
        db, _ = make_db([])
        result = _build_task_note('main_analysis', {'total_albums_processed': 12}, db)
        assert result == 'Albums analyzed: 12'

    def test_returns_empty_string_when_nothing_to_report(self):
        db, _ = make_db([])
        assert _build_task_note('main_analysis', {}, db) == ''

    def test_db_error_falls_back_to_album_details(self):
        db = MagicMock()
        db.cursor.side_effect = RuntimeError('no db')
        result = _build_task_note('main_analysis', {'albums_completed': 3}, db)
        assert result == 'Albums analyzed: 3'

    def test_db_error_without_albums_returns_empty_string(self):
        db = MagicMock()
        db.cursor.side_effect = RuntimeError('no db')
        assert _build_task_note('main_analysis', {}, db) == ''


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
        result = _build_task_note('main_cleaning', {key: 6}, MagicMock())
        assert result == 'Songs cleaned: 6'

    def test_first_key_wins(self):
        details = {'tracks_deleted': 2, 'orphans_removed': 9}
        assert _build_task_note('main_cleaning', details, MagicMock()) == 'Songs cleaned: 2'

    def test_zero_is_reported(self):
        result = _build_task_note('main_cleaning', {'tracks_deleted': 0}, MagicMock())
        assert result == 'Songs cleaned: 0'

    def test_string_values_skipped_in_favor_of_later_numeric_key(self):
        details = {'tracks_deleted': '5', 'orphans_removed': 3}
        assert _build_task_note('main_cleaning', details, MagicMock()) == 'Songs cleaned: 3'

    def test_float_value_truncated(self):
        result = _build_task_note('main_cleaning', {'songs_cleaned': 4.7}, MagicMock())
        assert result == 'Songs cleaned: 4'

    def test_no_recognized_keys_returns_empty_string(self):
        assert _build_task_note('main_cleaning', {'other': 1}, MagicMock()) == ''


class TestClusterNote:
    def test_best_params_subset_size_preferred(self):
        details = {
            'best_params': {'initial_subset_size': 500},
            'sampled_songs': 1,
            'num_playlists_created': 8,
        }
        result = _build_task_note('main_clustering', details, MagicMock())
        assert result == 'sampled: 500 | clusters: 8'

    def test_non_dict_best_params_falls_back_to_sampled_songs(self):
        details = {'best_params': 'oops', 'sampled_songs': 100}
        assert _build_task_note('main_clustering', details, MagicMock()) == 'sampled: 100'

    def test_best_params_without_subset_size_falls_back(self):
        details = {'best_params': {}, 'num_sampled_songs': 50}
        assert _build_task_note('main_clustering', details, MagicMock()) == 'sampled: 50'

    def test_clusters_only(self):
        assert (
            _build_task_note('main_clustering', {'num_clusters': 4}, MagicMock()) == 'clusters: 4'
        )

    def test_zero_sampled_is_omitted(self):
        details = {'sampled_songs': 0, 'num_clusters': 3}
        assert _build_task_note('main_clustering', details, MagicMock()) == 'clusters: 3'

    def test_no_data_returns_empty_string(self):
        assert _build_task_note('main_clustering', {}, MagicMock()) == ''

    def test_non_numeric_sampled_returns_empty_string(self):
        details = {'sampled_songs': 'abc', 'num_clusters': 3}
        assert _build_task_note('main_clustering', details, MagicMock()) == ''


class TestGeneralBehavior:
    def test_none_task_type_returns_empty_string(self):
        assert _build_task_note(None, {'tracks_deleted': 5}, MagicMock()) == ''

    def test_unknown_task_type_returns_empty_string(self):
        assert _build_task_note('sonic_fingerprint', {'tracks_deleted': 5}, MagicMock()) == ''

    def test_task_type_matching_is_case_insensitive(self):
        result = _build_task_note('MAIN_CLUSTERING', {'num_clusters': 2}, MagicMock())
        assert result == 'clusters: 2'

    def test_non_dict_details_treated_as_empty(self):
        assert _build_task_note('main_cleaning', 'notadict', MagicMock()) == ''
