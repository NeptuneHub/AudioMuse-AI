"""Unit tests for tasks/clustering_postprocessing.py

Tests core duplicate filtering and playlist selection logic:
- Minimum size filtering
- Title/artist deduplication
- Distance-based duplicate detection
- Diversity-based playlist selection
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from tasks.clustering_postprocessing import (
    apply_minimum_size_filter_to_clustering_result,
    apply_title_artist_deduplication,
    apply_distance_filtering_direct,
    select_top_n_diverse_playlists
)


class TestMinimumSizeFilter:
    """Test minimum playlist size filtering"""

    def test_filters_out_small_playlists(self):
        """Test that playlists smaller than min_size are removed"""
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
        """Test that all playlists are kept when above threshold"""
        best_result = {
            "named_playlists": {
                "Playlist 1": [("s1", "T1", "A1")] * 25,
                "Playlist 2": [("s2", "T2", "A2")] * 30,
            }
        }
        
        filtered = apply_minimum_size_filter_to_clustering_result(best_result, min_size=20)
        
        assert len(filtered["named_playlists"]) == 2

    def test_handles_empty_playlists(self):
        """Test handling of empty playlist dict"""
        best_result = {"named_playlists": {}}
        
        filtered = apply_minimum_size_filter_to_clustering_result(best_result, min_size=10)
        
        assert filtered["named_playlists"] == {}

    def test_handles_none_result(self):
        """Test handling of None input"""
        filtered = apply_minimum_size_filter_to_clustering_result(None, min_size=10)
        
        assert filtered is None

    def test_min_size_zero_keeps_all(self):
        """Test that min_size=0 keeps all playlists"""
        best_result = {
            "named_playlists": {
                "Empty": [],
                "One Song": [("s1", "T1", "A1")],
            }
        }
        
        filtered = apply_minimum_size_filter_to_clustering_result(best_result, min_size=0)
        
        assert len(filtered["named_playlists"]) == 2

    def test_preserves_exact_size_playlists(self):
        """Test that playlists exactly at min_size are kept"""
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
    """Test title/artist based deduplication"""

    def test_removes_duplicate_titles(self):
        """Test removal of exact title/artist duplicates"""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'item_id': 's1', 'title': 'Bohemian Rhapsody', 'author': 'Queen'},
            {'item_id': 's2', 'title': 'Bohemian Rhapsody', 'author': 'Queen'},  # Duplicate
        ]
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor
        
        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]
        
        filtered = apply_title_artist_deduplication(song_results, mock_db)
        
        assert len(filtered) == 1
        assert filtered[0]['item_id'] == 's1'

    def test_normalizes_remastered_versions(self):
        """Test that (Remastered) versions are detected as duplicates"""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'item_id': 's1', 'title': 'Stairway to Heaven', 'author': 'Led Zeppelin'},
            {'item_id': 's2', 'title': 'Stairway to Heaven (Remastered 2014)', 'author': 'Led Zeppelin'},
        ]
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor
        
        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]
        
        filtered = apply_title_artist_deduplication(song_results, mock_db)
        
        # Should detect as duplicate after normalization
        assert len(filtered) == 1

    def test_normalizes_explicit_markers(self):
        """Test that [Explicit] markers are removed during normalization"""
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
        """Test that same title by different artists are kept"""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'item_id': 's1', 'title': 'Hello', 'author': 'Adele'},
            {'item_id': 's2', 'title': 'Hello', 'author': 'Lionel Richie'},
        ]
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor
        
        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]
        
        filtered = apply_title_artist_deduplication(song_results, mock_db)
        
        # Different artists, should keep both
        assert len(filtered) == 2

    def test_handles_empty_list(self):
        """Test handling of empty song list"""
        mock_db = MagicMock()
        
        filtered = apply_title_artist_deduplication([], mock_db)
        
        assert filtered == []

    def test_handles_missing_song_details(self):
        """Test handling when song details are not found in DB"""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []  # No results from DB
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor
        
        song_results = [{'item_id': 's1'}]
        
        filtered = apply_title_artist_deduplication(song_results, mock_db)
        
        # Should skip songs without details
        assert len(filtered) == 0


class TestDistanceFilteringDirect:
    """Test vector distance-based duplicate filtering"""

    @patch('config.VOYAGER_METRIC', 'euclidean')
    @patch('config.DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN', 0.1)
    @patch('config.DUPLICATE_DISTANCE_CHECK_LOOKBACK', 5)
    @patch('tasks.clustering_postprocessing.get_vectors_from_database')
    def test_filters_close_vectors(self, mock_get_vectors):
        """Test that songs with close vectors are filtered"""
        mock_db = MagicMock()
        # Two very similar vectors
        mock_get_vectors.return_value = {
            's1': np.array([1.0, 0.0, 0.0]),
            's2': np.array([1.01, 0.0, 0.0])  # Very close to s1
        }
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'item_id': 's1', 'title': 'Song 1', 'author': 'Artist'},
            {'item_id': 's2', 'title': 'Song 2', 'author': 'Artist'},
        ]
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor
        
        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]
        
        filtered = apply_distance_filtering_direct(song_results, mock_db)
        
        # Second song should be filtered as too close
        assert len(filtered) == 1
        assert filtered[0]['item_id'] == 's1'

    @patch('config.VOYAGER_METRIC', 'euclidean')
    @patch('config.DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN', 0.1)
    @patch('config.DUPLICATE_DISTANCE_CHECK_LOOKBACK', 5)
    @patch('tasks.clustering_postprocessing.get_vectors_from_database')
    def test_keeps_distant_vectors(self, mock_get_vectors):
        """Test that songs with distant vectors are kept"""
        mock_db = MagicMock()
        mock_get_vectors.return_value = {
            's1': np.array([1.0, 0.0, 0.0]),
            's2': np.array([0.0, 1.0, 0.0])  # Orthogonal - far from s1
        }
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'item_id': 's1', 'title': 'Song 1', 'author': 'Artist'},
            {'item_id': 's2', 'title': 'Song 2', 'author': 'Artist'},
        ]
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor
        
        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]
        
        filtered = apply_distance_filtering_direct(song_results, mock_db)
        
        # Both should be kept
        assert len(filtered) == 2

    @patch('tasks.clustering_postprocessing.apply_title_artist_deduplication')
    @patch('tasks.clustering_postprocessing.get_vectors_from_database')
    def test_fallback_to_title_artist_when_no_vectors(self, mock_get_vectors, mock_title_dedup):
        """Test fallback to title/artist matching when no vectors available"""
        mock_db = MagicMock()
        mock_get_vectors.return_value = {}  # No vectors
        mock_title_dedup.return_value = [{'item_id': 's1'}]
        
        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]
        
        filtered = apply_distance_filtering_direct(song_results, mock_db)
        
        # Should call title/artist deduplication
        mock_title_dedup.assert_called_once()
        assert filtered == [{'item_id': 's1'}]

    @patch('config.DUPLICATE_DISTANCE_CHECK_LOOKBACK', 0)
    def test_skips_filtering_when_lookback_zero(self):
        """Test that filtering is skipped when lookback is 0"""
        mock_db = MagicMock()
        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]
        
        filtered = apply_distance_filtering_direct(song_results, mock_db)
        
        # Should return original list unchanged
        assert filtered == song_results


class TestSelectTopNDiversePlaylists:
    """Test diversity-based playlist selection"""

    def test_selects_diverse_playlists(self):
        """Test that diverse playlists are prioritized"""
        best_result = {
            "playlist_to_centroid_vector_map": {
                "Playlist 1": np.array([1.0, 0.0, 0.0]),
                "Playlist 2": np.array([0.0, 1.0, 0.0]),  # Orthogonal to 1
                "Playlist 3": np.array([1.1, 0.1, 0.0]),  # Very close to 1
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
            }
        }
        
        selected = select_top_n_diverse_playlists(best_result, n=2)
        
        # Should prefer diverse playlists 1 and 2 over similar 3
        assert len(selected["named_playlists"]) == 2

    def test_returns_all_when_n_exceeds_available(self):
        """Test that all playlists are returned when N >= available"""
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
            }
        }
        
        selected = select_top_n_diverse_playlists(best_result, n=5)
        
        # Should return original result unchanged
        assert len(selected["named_playlists"]) == 2

    def test_starts_with_largest_playlist(self):
        """Test that selection starts with the largest playlist"""
        best_result = {
            "playlist_to_centroid_vector_map": {
                "Small": np.array([1.0, 0.0]),
                "Large": np.array([0.0, 1.0]),
                "Medium": np.array([0.5, 0.5]),
            },
            "named_playlists": {
                "Small": [("s1", "T1", "A1")] * 10,
                "Large": [("s2", "T2", "A2")] * 50,  # Largest
                "Medium": [("s3", "T3", "A3")] * 25,
            },
            "playlist_centroids": {
                "Small": np.array([1.0, 0.0]),
                "Large": np.array([0.0, 1.0]),
                "Medium": np.array([0.5, 0.5]),
            }
        }
        
        selected = select_top_n_diverse_playlists(best_result, n=2)
        
        # Should include the largest playlist
        assert "Large" in selected["named_playlists"]


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_minimum_size_filter_with_missing_key(self):
        """Test that missing 'named_playlists' key is handled"""
        best_result = {}  # Missing key
        
        filtered = apply_minimum_size_filter_to_clustering_result(best_result, min_size=10)
        
        # Should handle gracefully
        assert filtered is not None

    def test_title_dedup_with_unicode(self):
        """Test title deduplication with unicode characters"""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'item_id': 's1', 'title': 'Café del Mar', 'author': 'Artist'},
            {'item_id': 's2', 'title': 'Café del Mar (Remastered)', 'author': 'Artist'},
        ]
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor
        
        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]
        
        filtered = apply_title_artist_deduplication(song_results, mock_db)
        
        # Should normalize and detect as duplicate
        assert len(filtered) == 1

    @patch('config.DUPLICATE_DISTANCE_CHECK_LOOKBACK', 5)
    @patch('tasks.clustering_postprocessing.get_vectors_from_database')
    def test_distance_filtering_with_missing_vectors(self, mock_get_vectors):
        """Test that songs missing vectors are kept without distance check"""
        mock_db = MagicMock()
        mock_get_vectors.return_value = {
            's1': np.array([1.0, 0.0, 0.0]),
            # s2 has no vector
        }
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'item_id': 's1', 'title': 'Song 1', 'author': 'Artist'},
            {'item_id': 's2', 'title': 'Song 2', 'author': 'Artist'},
        ]
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor
        
        song_results = [{'item_id': 's1'}, {'item_id': 's2'}]
        
        filtered = apply_distance_filtering_direct(song_results, mock_db)
        
        # Both should be kept (s2 kept without distance check)
        assert len(filtered) == 2
