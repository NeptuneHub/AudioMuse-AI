# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Playlist post-processing filters in clustering_postprocessing.

Covers the filters that trim and deduplicate clustering output before it becomes
playlists: size threshold, title/artist dedup, distance filtering, and diversity.

Main Features:
* Minimum-size filter drops small playlists and keeps exact-threshold ones
* Title/artist dedup normalizes remastered/explicit markers and is case-insensitive
* Distance filtering drops near-duplicate vectors, falls back to title/artist dedup
  when vectors are missing, and diverse Top-N selection picks the largest first
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from tasks.clustering_postprocessing import (
    apply_minimum_size_filter_to_clustering_result,
    apply_title_artist_deduplication,
    apply_distance_filtering_direct,
    select_diverse_playlists_with_genre_coverage,
)


class TestMinimumSizeFilter:
    def test_filters_out_small_playlists(self):
        best_result = {
            "named_playlists": {
                "Big Playlist": [("song1", "Title 1", "Artist 1")] * 25,
                "Small Playlist": [("song2", "Title 2", "Artist 2")] * 5,
                "Medium Playlist": [("song3", "Title 3", "Artist 3")] * 20,
            }
        }

        filtered = apply_minimum_size_filter_to_clustering_result(best_result, min_size=20)

        assert "Big Playlist" in filtered["named_playlists"]
        assert "Medium Playlist" in filtered["named_playlists"]
        assert "Small Playlist" not in filtered["named_playlists"]

    def test_keeps_all_when_above_threshold(self):
        best_result = {
            "named_playlists": {
                "Playlist 1": [("s1", "T1", "A1")] * 25,
                "Playlist 2": [("s2", "T2", "A2")] * 30,
            }
        }

        filtered = apply_minimum_size_filter_to_clustering_result(best_result, min_size=20)

        assert len(filtered["named_playlists"]) == 2

    @pytest.mark.parametrize(
        "result_without_playlists",
        [None, {}, {"named_playlists": {}}],
    )
    def test_returns_the_caller_object_itself_when_there_are_no_playlists_to_filter(
        self, result_without_playlists
    ):
        filtered = apply_minimum_size_filter_to_clustering_result(
            result_without_playlists, min_size=10
        )

        assert filtered is result_without_playlists

    def test_returns_a_new_result_with_an_empty_playlist_map_when_every_playlist_is_too_small(self):
        songs = [("s1", "T1", "A1")] * 3
        best_result = {
            "named_playlists": {"Tiny": songs},
            "playlist_centroids": {"Tiny": np.array([1.0, 0.0])},
        }

        filtered = apply_minimum_size_filter_to_clustering_result(best_result, min_size=10)

        assert filtered is not best_result
        assert filtered["named_playlists"] == {}
        assert filtered["playlist_centroids"] == {}
        assert best_result["named_playlists"] == {"Tiny": songs}

    def test_min_size_zero_keeps_all(self):
        best_result = {
            "named_playlists": {
                "Empty": [],
                "One Song": [("s1", "T1", "A1")],
            }
        }

        filtered = apply_minimum_size_filter_to_clustering_result(best_result, min_size=0)

        assert len(filtered["named_playlists"]) == 2

    def test_preserves_exact_size_playlists(self):
        best_result = {
            "named_playlists": {
                "Exact Size": [("s1", "T1", "A1")] * 20,
                "Too Small": [("s2", "T2", "A2")] * 19,
            }
        }

        filtered = apply_minimum_size_filter_to_clustering_result(best_result, min_size=20)

        assert "Exact Size" in filtered["named_playlists"]
        assert "Too Small" not in filtered["named_playlists"]


class TestTitleArtistDeduplication:
    def test_removes_duplicate_titles(self):
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'item_id': 's1', 'title': 'Bohemian Rhapsody', 'author': 'Queen'},
            {'item_id': 's2', 'title': 'Bohemian Rhapsody', 'author': 'Queen'},
        ]
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor

        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]

        filtered = apply_title_artist_deduplication(song_results, mock_db)

        assert len(filtered) == 1
        assert filtered[0]['item_id'] == 's1'

    def test_normalizes_remastered_versions(self):
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'item_id': 's1', 'title': 'Stairway to Heaven', 'author': 'Led Zeppelin'},
            {
                'item_id': 's2',
                'title': 'Stairway to Heaven (Remastered 2014)',
                'author': 'Led Zeppelin',
            },
        ]
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor

        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]

        filtered = apply_title_artist_deduplication(song_results, mock_db)

        assert len(filtered) == 1

    def test_normalizes_explicit_markers(self):
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'item_id': 's1', 'title': 'Song Title', 'author': 'Artist'},
            {'item_id': 's2', 'title': 'Song Title [Explicit]', 'author': 'Artist'},
        ]
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor

        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]

        filtered = apply_title_artist_deduplication(song_results, mock_db)

        assert len(filtered) == 1

    def test_keeps_different_artists(self):
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'item_id': 's1', 'title': 'Hello', 'author': 'Adele'},
            {'item_id': 's2', 'title': 'Hello', 'author': 'Lionel Richie'},
        ]
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor

        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]

        filtered = apply_title_artist_deduplication(song_results, mock_db)

        assert len(filtered) == 2

    def test_returns_empty_for_an_empty_song_list_without_opening_a_cursor(self):
        mock_db = MagicMock()

        filtered = apply_title_artist_deduplication([], mock_db)

        assert filtered == []
        mock_db.cursor.assert_not_called()

    def test_skips_songs_with_no_score_row_and_keeps_the_ones_that_have_one(self):
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'item_id': 's2', 'title': 'Kept Song', 'author': 'Artist'},
        ]
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor

        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]

        filtered = apply_title_artist_deduplication(song_results, mock_db)

        assert filtered == [{'item_id': 's2'}]
        assert mock_cursor.execute.call_args[0][1] == (['s1', 's2'],)


class TestDistanceFilteringDirect:
    @patch('config.IVF_METRIC', 'euclidean')
    @patch('config.DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN', 0.1)
    @patch('config.DUPLICATE_DISTANCE_CHECK_LOOKBACK', 5)
    @patch('tasks.clustering_postprocessing.get_vectors_from_database')
    def test_filters_close_vectors(self, mock_get_vectors):
        mock_db = MagicMock()
        mock_get_vectors.return_value = {
            's1': np.array([1.0, 0.0, 0.0]),
            's2': np.array([1.01, 0.0, 0.0]),
        }
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'item_id': 's1', 'title': 'Song 1', 'author': 'Artist'},
            {'item_id': 's2', 'title': 'Song 2', 'author': 'Artist'},
        ]
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor

        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]

        filtered = apply_distance_filtering_direct(song_results, mock_db)

        assert len(filtered) == 1
        assert filtered[0]['item_id'] == 's1'

    @patch('config.IVF_METRIC', 'euclidean')
    @patch('config.DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN', 0.1)
    @patch('config.DUPLICATE_DISTANCE_CHECK_LOOKBACK', 5)
    @patch('tasks.clustering_postprocessing.get_vectors_from_database')
    def test_keeps_distant_vectors(self, mock_get_vectors):
        mock_db = MagicMock()
        mock_get_vectors.return_value = {
            's1': np.array([1.0, 0.0, 0.0]),
            's2': np.array([0.0, 1.0, 0.0]),
        }
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'item_id': 's1', 'title': 'Song 1', 'author': 'Artist'},
            {'item_id': 's2', 'title': 'Song 2', 'author': 'Artist'},
        ]
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor

        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]

        filtered = apply_distance_filtering_direct(song_results, mock_db)

        assert len(filtered) == 2

    @patch('tasks.clustering_postprocessing.apply_title_artist_deduplication')
    @patch('tasks.clustering_postprocessing.get_vectors_from_database')
    def test_fallback_to_title_artist_when_no_vectors(self, mock_get_vectors, mock_title_dedup):
        mock_db = MagicMock()
        mock_get_vectors.return_value = {}
        mock_title_dedup.return_value = [{'item_id': 's1'}]

        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]

        filtered = apply_distance_filtering_direct(song_results, mock_db)

        mock_title_dedup.assert_called_once()
        assert filtered == [{'item_id': 's1'}]

    @patch('config.DUPLICATE_DISTANCE_CHECK_LOOKBACK', 0)
    def test_skips_filtering_when_lookback_zero(self):
        mock_db = MagicMock()
        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]

        filtered = apply_distance_filtering_direct(song_results, mock_db)

        assert filtered == song_results


class TestSelectDiversePlaylists:
    def test_selects_diverse_playlists(self):
        best_result = {
            "playlist_to_centroid_vector_map": {
                "Playlist 1": np.array([1.0, 0.0, 0.0]),
                "Playlist 2": np.array([0.0, 1.0, 0.0]),
                "Playlist 3": np.array([1.1, 0.1, 0.0]),
            },
            "named_playlists": {
                "Playlist 1": [("s1", "T1", "A1")] * 20,
                "Playlist 2": [("s2", "T2", "A2")] * 20,
                "Playlist 3": [("s3", "T3", "A3")] * 20,
            },
            "playlist_centroids": {
                "Playlist 1": np.array([1.0, 0.0, 0.0]),
                "Playlist 2": np.array([0.0, 1.0, 0.0]),
                "Playlist 3": np.array([1.1, 0.1, 0.0]),
            },
        }

        selected = select_diverse_playlists_with_genre_coverage(best_result, limit=2)

        assert len(selected["named_playlists"]) == 2

    def test_returns_all_when_limit_exceeds_available(self):
        best_result = {
            "playlist_to_centroid_vector_map": {
                "Playlist 1": np.array([1.0, 0.0]),
                "Playlist 2": np.array([0.0, 1.0]),
            },
            "named_playlists": {
                "Playlist 1": [("s1", "T1", "A1")],
                "Playlist 2": [("s2", "T2", "A2")],
            },
            "playlist_centroids": {
                "Playlist 1": np.array([1.0, 0.0]),
                "Playlist 2": np.array([0.0, 1.0]),
            },
        }

        selected = select_diverse_playlists_with_genre_coverage(best_result, limit=5)

        assert len(selected["named_playlists"]) == 2

    def test_starts_with_largest_playlist(self):
        best_result = {
            "playlist_to_centroid_vector_map": {
                "Small": np.array([1.0, 0.0]),
                "Large": np.array([0.0, 1.0]),
                "Medium": np.array([0.5, 0.5]),
            },
            "named_playlists": {
                "Small": [("s1", "T1", "A1")] * 10,
                "Large": [("s2", "T2", "A2")] * 50,
                "Medium": [("s3", "T3", "A3")] * 25,
            },
            "playlist_centroids": {
                "Small": np.array([1.0, 0.0]),
                "Large": np.array([0.0, 1.0]),
                "Medium": np.array([0.5, 0.5]),
            },
        }

        selected = select_diverse_playlists_with_genre_coverage(best_result, limit=2)

        assert "Large" in selected["named_playlists"]


class TestEdgeCases:
    def test_title_dedup_with_unicode(self):
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'item_id': 's1', 'title': 'Café del Mar', 'author': 'Artist'},
            {'item_id': 's2', 'title': 'Café del Mar (Remastered)', 'author': 'Artist'},
        ]
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor

        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]

        filtered = apply_title_artist_deduplication(song_results, mock_db)

        assert len(filtered) == 1

    @patch('config.DUPLICATE_DISTANCE_CHECK_LOOKBACK', 5)
    @patch('tasks.clustering_postprocessing.get_vectors_from_database')
    def test_distance_filtering_with_missing_vectors(self, mock_get_vectors):
        mock_db = MagicMock()
        mock_get_vectors.return_value = {
            's1': np.array([1.0, 0.0, 0.0]),
        }
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'item_id': 's1', 'title': 'Song 1', 'author': 'Artist'},
            {'item_id': 's2', 'title': 'Song 2', 'author': 'Artist'},
        ]
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor

        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]

        filtered = apply_distance_filtering_direct(song_results, mock_db)

        assert len(filtered) == 2
