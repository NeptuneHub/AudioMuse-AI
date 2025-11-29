"""Unit tests for clustering modules

Tests cover clustering helper functions, parameter generation, post-processing,
and duplicate filtering with minimal mocking.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from collections import defaultdict


class TestParameterMutation:
    """Tests for parameter mutation functions in clustering_helper"""

    def test_mutate_param_integer_within_bounds(self):
        """Test integer parameter mutation stays within bounds"""
        from tasks.clustering_helper import _mutate_param
        
        # Run multiple times to test randomness
        for _ in range(10):
            result = _mutate_param(50, min_val=0, max_val=100, delta=10, is_float=False)
            assert 0 <= result <= 100
            assert isinstance(result, int)

    def test_mutate_param_integer_at_min_boundary(self):
        """Test integer mutation at minimum boundary"""
        from tasks.clustering_helper import _mutate_param
        
        result = _mutate_param(0, min_val=0, max_val=100, delta=10, is_float=False)
        assert 0 <= result <= 100

    def test_mutate_param_integer_at_max_boundary(self):
        """Test integer mutation at maximum boundary"""
        from tasks.clustering_helper import _mutate_param
        
        result = _mutate_param(100, min_val=0, max_val=100, delta=10, is_float=False)
        assert 0 <= result <= 100

    def test_mutate_param_float_within_bounds(self):
        """Test float parameter mutation stays within bounds"""
        from tasks.clustering_helper import _mutate_param
        
        for _ in range(10):
            result = _mutate_param(0.5, min_val=0.0, max_val=1.0, delta=0.1, is_float=True)
            assert 0.0 <= result <= 1.0
            assert isinstance(result, float)

    def test_mutate_param_float_precision(self):
        """Test float mutation maintains reasonable precision"""
        from tasks.clustering_helper import _mutate_param
        
        result = _mutate_param(0.5, min_val=0.0, max_val=1.0, delta=0.05, is_float=True)
        # Should be close to original value
        assert abs(result - 0.5) <= 0.1  # Within delta range

    def test_mutate_param_clipping_low(self):
        """Test that values below min are clipped"""
        from tasks.clustering_helper import _mutate_param
        
        # Even with large delta, should clip to min
        result = _mutate_param(5, min_val=10, max_val=100, delta=1, is_float=False)
        assert result == 10

    def test_mutate_param_clipping_high(self):
        """Test that values above max are clipped"""
        from tasks.clustering_helper import _mutate_param
        
        # Even with large delta, should clip to max
        result = _mutate_param(95, min_val=0, max_val=90, delta=1, is_float=False)
        assert result == 90


class TestDataPreparationAndScaling:
    """Tests for data preparation and scaling functions"""

    def test_prepare_and_scale_data_with_features(self):
        """Test data scaling with feature vectors"""
        from tasks.clustering_helper import _prepare_and_scale_data
        
        # Create sample feature data
        X_feat = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        X_embed = None
        
        scaled_data, scaler = _prepare_and_scale_data(X_feat, X_embed, use_embeddings=False)
        
        # Verify scaling occurred
        assert scaled_data is not None
        assert scaler is not None
        assert scaled_data.shape == X_feat.shape
        
        # Verify data is standardized (mean ~0, std ~1)
        assert np.abs(scaled_data.mean(axis=0)).max() < 0.1  # Near zero mean
        assert np.abs(scaled_data.std(axis=0) - 1.0).max() < 0.1  # Near unit variance

    def test_prepare_and_scale_data_with_embeddings(self):
        """Test data scaling with embedding vectors"""
        from tasks.clustering_helper import _prepare_and_scale_data
        
        X_feat = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_embed = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]
        ])
        
        scaled_data, scaler = _prepare_and_scale_data(X_feat, X_embed, use_embeddings=True)
        
        assert scaled_data is not None
        assert scaled_data.shape == X_embed.shape
        # Should use embeddings, not features
        assert scaled_data.shape[1] == 4

    def test_prepare_and_scale_data_returns_none_for_empty(self):
        """Test returns None for empty data"""
        from tasks.clustering_helper import _prepare_and_scale_data
        
        X_feat = np.array([])
        X_embed = None
        
        result = _prepare_and_scale_data(X_feat, X_embed, use_embeddings=False)
        
        assert result == (None, None)

    def test_prepare_and_scale_data_returns_none_for_zero_rows(self):
        """Test returns None for data with zero rows"""
        from tasks.clustering_helper import _prepare_and_scale_data
        
        X_feat = np.empty((0, 5))
        X_embed = None
        
        result = _prepare_and_scale_data(X_feat, X_embed, use_embeddings=False)
        
        assert result == (None, None)


class TestFeatureCentroidCalculation:
    """Tests for feature centroid calculation from embeddings"""

    def test_get_feature_centroid_for_embedding_cluster_basic(self):
        """Test centroid calculation for a cluster"""
        from tasks.clustering_helper import _get_feature_centroid_for_embedding_cluster
        
        labels = np.array([0, 0, 1, 1, 0])
        X_feat = np.array([
            [1.0, 2.0, 3.0],
            [1.5, 2.5, 3.5],
            [5.0, 6.0, 7.0],
            [5.5, 6.5, 7.5],
            [2.0, 3.0, 4.0]
        ])
        
        # Get centroid for cluster 0 (indices 0, 1, 4)
        centroid = _get_feature_centroid_for_embedding_cluster(0, labels, X_feat)
        
        assert centroid is not None
        assert centroid.shape == (3,)
        
        # Verify it's the mean of cluster 0 features
        expected_centroid = np.mean(X_feat[[0, 1, 4]], axis=0)
        np.testing.assert_array_almost_equal(centroid, expected_centroid)

    def test_get_feature_centroid_for_single_member_cluster(self):
        """Test centroid for cluster with single member"""
        from tasks.clustering_helper import _get_feature_centroid_for_embedding_cluster
        
        labels = np.array([0, 1, 2])
        X_feat = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        
        centroid = _get_feature_centroid_for_embedding_cluster(1, labels, X_feat)
        
        # For single member, centroid should equal that member's features
        np.testing.assert_array_almost_equal(centroid, X_feat[1])

    def test_get_feature_centroid_for_empty_cluster(self):
        """Test centroid calculation for empty cluster returns None"""
        from tasks.clustering_helper import _get_feature_centroid_for_embedding_cluster
        
        labels = np.array([0, 0, 1, 1])
        X_feat = np.array([[1.0], [2.0], [3.0], [4.0]])
        
        # Request centroid for non-existent cluster 5
        result = _get_feature_centroid_for_embedding_cluster(5, labels, X_feat)
        
        assert result is None

    def test_get_feature_centroid_maintains_dimensionality(self):
        """Test that centroid has correct dimensionality"""
        from tasks.clustering_helper import _get_feature_centroid_for_embedding_cluster
        
        labels = np.array([0, 0, 0, 1, 1])
        # High-dimensional features
        X_feat = np.random.rand(5, 50)
        
        centroid = _get_feature_centroid_for_embedding_cluster(0, labels, X_feat)
        
        assert centroid.shape == (50,)


class TestTrackPrimaryGenre:
    """Tests for track genre classification"""

    @patch('tasks.clustering_helper.STRATIFIED_GENRES', ['rock', 'pop', 'jazz', 'metal'])
    def test_get_track_primary_genre_with_mood_vector(self):
        """Test genre extraction from mood vector"""
        from tasks.clustering_helper import _get_track_primary_genre
        
        track = {
            'mood_vector': 'rock:0.8,pop:0.2,jazz:0.1'
        }
        
        genre = _get_track_primary_genre(track)
        
        assert genre == 'rock'

    def test_get_track_primary_genre_with_no_mood_vector(self):
        """Test returns __other__ when no mood vector"""
        from tasks.clustering_helper import _get_track_primary_genre
        
        track = {
            'title': 'Some Song'
        }
        
        genre = _get_track_primary_genre(track)
        
        assert genre == '__other__'

    def test_get_track_primary_genre_with_empty_mood_vector(self):
        """Test returns __other__ for empty mood vector"""
        from tasks.clustering_helper import _get_track_primary_genre
        
        track = {
            'mood_vector': ''
        }
        
        genre = _get_track_primary_genre(track)
        
        assert genre == '__other__'

    def test_get_track_primary_genre_with_none_mood_vector(self):
        """Test returns __other__ for None mood vector"""
        from tasks.clustering_helper import _get_track_primary_genre
        
        track = {
            'mood_vector': None
        }
        
        genre = _get_track_primary_genre(track)
        
        assert genre == '__other__'


class TestGenreMapPreparation:
    """Tests for genre map creation from database rows"""

    @patch('tasks.clustering.STRATIFIED_GENRES', ['rock', 'pop', 'jazz', 'metal'])
    def test_prepare_genre_map_basic(self):
        """Test creating genre map from rows"""
        from tasks.clustering import _prepare_genre_map
        
        rows = [
            {'item_id': '1', 'mood_vector': 'rock:0.9,pop:0.1'},
            {'item_id': '2', 'mood_vector': 'rock:0.8,jazz:0.2'},
            {'item_id': '3', 'mood_vector': 'pop:0.9,rock:0.1'},
            {'item_id': '4', 'mood_vector': 'jazz:0.7,rock:0.3'}
        ]
        
        genre_map = _prepare_genre_map(rows)
        
        assert 'rock' in genre_map
        assert 'pop' in genre_map
        assert 'jazz' in genre_map
        assert len(genre_map['rock']) == 2  # tracks 1 and 2
        assert len(genre_map['pop']) == 1   # track 3
        assert len(genre_map['jazz']) == 1  # track 4

    def test_prepare_genre_map_with_no_mood_vector(self):
        """Test that tracks without mood_vector are skipped (not added to map)"""
        from tasks.clustering import _prepare_genre_map
        
        rows = [
            {'item_id': '1', 'mood_vector': ''},
            {'item_id': '2', 'mood_vector': None},
            {'item_id': '3', 'title': 'Song'}
        ]
        
        genre_map = _prepare_genre_map(rows)
        
        # Function only processes rows with valid mood_vector
        # Rows without mood_vector are skipped
        assert len(genre_map) == 0

    def test_prepare_genre_map_empty_input(self):
        """Test genre map with empty input"""
        from tasks.clustering import _prepare_genre_map
        
        genre_map = _prepare_genre_map([])
        
        assert isinstance(genre_map, defaultdict)
        assert len(genre_map) == 0


class TestTargetSongsCalculation:
    """Tests for calculating target songs per genre"""

    @patch('tasks.clustering.STRATIFIED_GENRES', ['rock', 'pop', 'jazz', 'metal'])
    def test_calculate_target_songs_per_genre_basic(self):
        """Test target calculation with normal distribution"""
        from tasks.clustering import _calculate_target_songs_per_genre
        
        genre_map = {
            'rock': [{'id': i} for i in range(100)],
            'pop': [{'id': i} for i in range(50)],
            'jazz': [{'id': i} for i in range(150)],
            'metal': [{'id': i} for i in range(75)]
        }
        
        target = _calculate_target_songs_per_genre(genre_map, percentile=50, min_songs=10)
        
        # 50th percentile of [50, 75, 100, 150] should be around 87.5
        assert 70 <= target <= 100
        assert isinstance(target, int)

    def test_calculate_target_songs_respects_minimum(self):
        """Test that minimum songs constraint is enforced"""
        from tasks.clustering import _calculate_target_songs_per_genre
        
        genre_map = {
            'rock': [{'id': 1}],  # Very small counts
            'pop': [{'id': 2}]
        }
        
        target = _calculate_target_songs_per_genre(genre_map, percentile=50, min_songs=100)
        
        # Should use min_songs when calculated value is lower
        assert target == 100

    @patch('tasks.clustering.STRATIFIED_GENRES', ['rock', 'pop', 'jazz', 'metal'])
    def test_calculate_target_songs_high_percentile(self):
        """Test calculation with high percentile"""
        from tasks.clustering import _calculate_target_songs_per_genre
        
        genre_map = {
            'rock': [{'id': i} for i in range(100)],
            'pop': [{'id': i} for i in range(200)],
            'jazz': [{'id': i} for i in range(50)]
        }
        
        target = _calculate_target_songs_per_genre(genre_map, percentile=90, min_songs=10)
        
        # 90th percentile should be closer to 200
        assert target >= 150

    def test_calculate_target_songs_empty_genre_map(self):
        """Test with empty genre map returns minimum"""
        from tasks.clustering import _calculate_target_songs_per_genre
        
        genre_map = {}
        
        target = _calculate_target_songs_per_genre(genre_map, percentile=50, min_songs=20)
        
        assert target == 20


class TestSanitizeForJson:
    """Tests for JSON sanitization function"""

    def test_sanitize_numpy_array(self):
        """Test numpy array conversion to list"""
        from tasks.clustering import _sanitize_for_json
        
        obj = np.array([1.0, 2.0, 3.0])
        result = _sanitize_for_json(obj)
        
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_sanitize_numpy_integers(self):
        """Test numpy integer conversion"""
        from tasks.clustering import _sanitize_for_json
        
        obj = {
            'int8': np.int8(42),
            'int16': np.int16(100),
            'int32': np.int32(1000),
            'int64': np.int64(10000)
        }
        
        result = _sanitize_for_json(obj)
        
        for key, val in result.items():
            assert isinstance(val, int)
            assert not isinstance(val, np.integer)

    def test_sanitize_numpy_floats(self):
        """Test numpy float conversion"""
        from tasks.clustering import _sanitize_for_json
        
        obj = {
            'float32': np.float32(3.14),
            'float64': np.float64(2.718)
        }
        
        result = _sanitize_for_json(obj)
        
        for key, val in result.items():
            assert isinstance(val, float)
            assert not isinstance(val, np.floating)

    def test_sanitize_numpy_bool(self):
        """Test numpy bool conversion"""
        from tasks.clustering import _sanitize_for_json
        
        obj = {'flag': np.bool_(True)}
        result = _sanitize_for_json(obj)
        
        assert isinstance(result['flag'], bool)
        assert result['flag'] is True

    def test_sanitize_nested_structures(self):
        """Test sanitization of nested dictionaries and lists"""
        from tasks.clustering import _sanitize_for_json
        
        obj = {
            'array': np.array([1, 2, 3]),
            'nested': {
                'float': np.float64(1.5),
                'list': [np.int32(5), np.int32(10)]
            }
        }
        
        result = _sanitize_for_json(obj)
        
        assert isinstance(result['array'], list)
        assert isinstance(result['nested']['float'], float)
        assert all(isinstance(x, int) for x in result['nested']['list'])

    def test_sanitize_preserves_native_types(self):
        """Test that native Python types are preserved"""
        from tasks.clustering import _sanitize_for_json
        
        obj = {
            'string': 'hello',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'list': [1, 2, 3],
            'none': None
        }
        
        result = _sanitize_for_json(obj)
        
        assert result == obj


class TestGetVectorsFromDatabase:
    """Tests for vector retrieval from database"""

    def test_get_vectors_from_database_basic(self):
        """Test fetching vectors from database"""
        from tasks.clustering_postprocessing import get_vectors_from_database
        
        # Mock database connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
        
        # Mock query results - embedding needs to be bytes (bytea from DB)
        vector1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        vector2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        
        mock_cursor.fetchall.return_value = [
            {'item_id': 'song1', 'embedding': vector1.tobytes()},
            {'item_id': 'song2', 'embedding': vector2.tobytes()}
        ]
        
        item_ids = ['song1', 'song2']
        result = get_vectors_from_database(item_ids, mock_conn)
        
        assert len(result) == 2
        assert 'song1' in result
        assert 'song2' in result
        np.testing.assert_array_almost_equal(result['song1'], vector1)
        np.testing.assert_array_almost_equal(result['song2'], vector2)

    def test_get_vectors_from_database_empty(self):
        """Test with no results"""
        from tasks.clustering_postprocessing import get_vectors_from_database
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
        
        mock_cursor.fetchall.return_value = []
        
        result = get_vectors_from_database(['song1'], mock_conn)
        
        assert len(result) == 0


class TestTitleArtistDeduplication:
    """Tests for title/artist based deduplication"""

    def test_title_artist_deduplication_removes_exact_duplicates(self):
        """Test removal of exact title/artist duplicates"""
        from tasks.clustering_postprocessing import apply_title_artist_deduplication
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
        
        mock_cursor.fetchall.return_value = [
            {'item_id': 'song1', 'title': 'Song A', 'author': 'Artist X'},
            {'item_id': 'song2', 'title': 'Song A', 'author': 'Artist X'},  # Duplicate
            {'item_id': 'song3', 'title': 'Song B', 'author': 'Artist Y'}
        ]
        
        songs = [{'item_id': 'song1'}, {'item_id': 'song2'}, {'item_id': 'song3'}]
        result = apply_title_artist_deduplication(songs, mock_conn)
        
        # Should keep only 2 songs (song1 and song3)
        assert len(result) == 2
        result_ids = [s['item_id'] for s in result]
        assert 'song1' in result_ids
        assert 'song3' in result_ids

    def test_title_artist_deduplication_case_insensitive(self):
        """Test deduplication is case insensitive"""
        from tasks.clustering_postprocessing import apply_title_artist_deduplication
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
        
        mock_cursor.fetchall.return_value = [
            {'item_id': 'song1', 'title': 'Song A', 'author': 'Artist X'},
            {'item_id': 'song2', 'title': 'SONG A', 'author': 'ARTIST X'},  # Case different
        ]
        
        songs = [{'item_id': 'song1'}, {'item_id': 'song2'}]
        result = apply_title_artist_deduplication(songs, mock_conn)
        
        # Should keep only 1 song
        assert len(result) == 1

    def test_title_artist_deduplication_removes_remastered_versions(self):
        """Test removal of remastered/explicit version markers"""
        from tasks.clustering_postprocessing import apply_title_artist_deduplication
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
        
        mock_cursor.fetchall.return_value = [
            {'item_id': 'song1', 'title': 'Song A', 'author': 'Artist X'},
            {'item_id': 'song2', 'title': 'Song A (Remastered)', 'author': 'Artist X'},
            {'item_id': 'song3', 'title': 'Song A [Explicit]', 'author': 'Artist X'},
        ]
        
        songs = [{'item_id': 'song1'}, {'item_id': 'song2'}, {'item_id': 'song3'}]
        result = apply_title_artist_deduplication(songs, mock_conn)
        
        # Should keep only 1 song (all are same after cleanup)
        assert len(result) == 1

    def test_title_artist_deduplication_preserves_different_songs(self):
        """Test that different songs are preserved"""
        from tasks.clustering_postprocessing import apply_title_artist_deduplication
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
        
        mock_cursor.fetchall.return_value = [
            {'item_id': 'song1', 'title': 'Song A', 'author': 'Artist X'},
            {'item_id': 'song2', 'title': 'Song B', 'author': 'Artist X'},  # Different title
            {'item_id': 'song3', 'title': 'Song A', 'author': 'Artist Y'},  # Different artist
        ]
        
        songs = [{'item_id': 'song1'}, {'item_id': 'song2'}, {'item_id': 'song3'}]
        result = apply_title_artist_deduplication(songs, mock_conn)
        
        # Should keep all 3 songs
        assert len(result) == 3

    def test_title_artist_deduplication_empty_input(self):
        """Test with empty input"""
        from tasks.clustering_postprocessing import apply_title_artist_deduplication
        
        mock_conn = Mock()
        result = apply_title_artist_deduplication([], mock_conn)
        
        assert len(result) == 0


class TestMinimumSizeFilter:
    """Tests for minimum playlist size filtering"""

    def test_minimum_size_filter_removes_small_playlists(self):
        """Test removal of playlists below size threshold"""
        from tasks.clustering_postprocessing import apply_minimum_size_filter_to_clustering_result
        
        best_result = {
            'named_playlists': {
                'Large Playlist': [{'item_id': f'song{i}'} for i in range(25)],
                'Medium Playlist': [{'item_id': f'song{i}'} for i in range(20)],
                'Small Playlist': [{'item_id': f'song{i}'} for i in range(15)],
            },
            'playlist_centroids': {
                'Large Playlist': [0.5, 0.5, 0.5],
                'Medium Playlist': [0.3, 0.3, 0.3],
                'Small Playlist': [0.1, 0.1, 0.1],
            }
        }
        
        result = apply_minimum_size_filter_to_clustering_result(best_result, min_size=20)
        
        # Should keep only Large and Medium playlists
        assert len(result['named_playlists']) == 2
        assert 'Large Playlist' in result['named_playlists']
        assert 'Medium Playlist' in result['named_playlists']
        assert 'Small Playlist' not in result['named_playlists']

    def test_minimum_size_filter_preserves_large_playlists(self):
        """Test that playlists above threshold are preserved"""
        from tasks.clustering_postprocessing import apply_minimum_size_filter_to_clustering_result
        
        best_result = {
            'named_playlists': {
                'Playlist A': [{'item_id': f'song{i}'} for i in range(50)],
                'Playlist B': [{'item_id': f'song{i}'} for i in range(100)],
            },
            'playlist_centroids': {
                'Playlist A': [0.5],
                'Playlist B': [0.6],
            }
        }
        
        result = apply_minimum_size_filter_to_clustering_result(best_result, min_size=20)
        
        assert len(result['named_playlists']) == 2
        assert len(result['named_playlists']['Playlist A']) == 50
        assert len(result['named_playlists']['Playlist B']) == 100

    def test_minimum_size_filter_updates_centroids(self):
        """Test that centroids are updated to match filtered playlists"""
        from tasks.clustering_postprocessing import apply_minimum_size_filter_to_clustering_result
        
        best_result = {
            'named_playlists': {
                'Keep': [{'item_id': f'song{i}'} for i in range(25)],
                'Remove': [{'item_id': f'song{i}'} for i in range(5)],
            },
            'playlist_centroids': {
                'Keep': [0.5, 0.5],
                'Remove': [0.1, 0.1],
            }
        }
        
        result = apply_minimum_size_filter_to_clustering_result(best_result, min_size=20)
        
        assert 'Keep' in result['playlist_centroids']
        assert 'Remove' not in result['playlist_centroids']

    def test_minimum_size_filter_empty_input(self):
        """Test with empty result"""
        from tasks.clustering_postprocessing import apply_minimum_size_filter_to_clustering_result
        
        best_result = None
        result = apply_minimum_size_filter_to_clustering_result(best_result, min_size=20)
        
        assert result is None

    def test_minimum_size_filter_all_playlists_removed(self):
        """Test when all playlists are below threshold"""
        from tasks.clustering_postprocessing import apply_minimum_size_filter_to_clustering_result
        
        best_result = {
            'named_playlists': {
                'Small 1': [{'item_id': 'song1'}],
                'Small 2': [{'item_id': 'song2'}],
            },
            'playlist_centroids': {
                'Small 1': [0.1],
                'Small 2': [0.2],
            }
        }
        
        result = apply_minimum_size_filter_to_clustering_result(best_result, min_size=50)
        
        # Should return result with empty playlists
        assert len(result['named_playlists']) == 0


class TestSelectTopNDiversePlaylists:
    """Tests for selecting most diverse playlists"""

    def test_select_top_n_diverse_basic(self):
        """Test selecting N most diverse playlists"""
        from tasks.clustering_postprocessing import select_top_n_diverse_playlists
        
        # Create 5 playlists with distinct centroid vectors
        best_result = {
            'named_playlists': {
                f'Playlist {i}': [{'item_id': f'song{j}'} for j in range(20)]
                for i in range(5)
            },
            'playlist_centroids': {
                f'Playlist {i}': [float(i), float(i)]
                for i in range(5)
            },
            'playlist_to_centroid_vector_map': {
                f'Playlist {i}': np.array([float(i), float(i)])
                for i in range(5)
            }
        }
        
        result = select_top_n_diverse_playlists(best_result, n=3)
        
        # Should select 3 playlists
        assert len(result['named_playlists']) == 3
        assert len(result['playlist_centroids']) == 3
        assert len(result['playlist_to_centroid_vector_map']) == 3

    def test_select_top_n_diverse_preserves_largest_first(self):
        """Test that selection starts with largest playlist"""
        from tasks.clustering_postprocessing import select_top_n_diverse_playlists
        
        best_result = {
            'named_playlists': {
                'Small': [{'item_id': f'song{i}'} for i in range(10)],
                'Large': [{'item_id': f'song{i}'} for i in range(100)],
                'Medium': [{'item_id': f'song{i}'} for i in range(50)],
            },
            'playlist_centroids': {
                'Small': [1.0, 1.0],
                'Large': [2.0, 2.0],
                'Medium': [3.0, 3.0],
            },
            'playlist_to_centroid_vector_map': {
                'Small': np.array([1.0, 1.0]),
                'Large': np.array([2.0, 2.0]),
                'Medium': np.array([3.0, 3.0]),
            }
        }
        
        result = select_top_n_diverse_playlists(best_result, n=2)
        
        # Large should always be selected (it's the largest)
        assert 'Large' in result['named_playlists']

    def test_select_top_n_skips_when_n_too_large(self):
        """Test returns original when N >= available playlists"""
        from tasks.clustering_postprocessing import select_top_n_diverse_playlists
        
        best_result = {
            'named_playlists': {
                'P1': [{'item_id': 'song1'}],
                'P2': [{'item_id': 'song2'}],
            },
            'playlist_centroids': {
                'P1': [1.0],
                'P2': [2.0],
            },
            'playlist_to_centroid_vector_map': {
                'P1': np.array([1.0]),
                'P2': np.array([2.0]),
            }
        }
        
        result = select_top_n_diverse_playlists(best_result, n=10)
        
        # Should return original (2 playlists)
        assert len(result['named_playlists']) == 2

    def test_select_top_n_skips_when_n_zero(self):
        """Test returns original when N is 0"""
        from tasks.clustering_postprocessing import select_top_n_diverse_playlists
        
        best_result = {
            'named_playlists': {'P1': [{'item_id': 'song1'}]},
            'playlist_centroids': {'P1': [1.0]},
            'playlist_to_centroid_vector_map': {'P1': np.array([1.0])},
        }
        
        result = select_top_n_diverse_playlists(best_result, n=0)
        
        assert result == best_result

    def test_select_top_n_empty_result(self):
        """Test with empty result"""
        from tasks.clustering_postprocessing import select_top_n_diverse_playlists
        
        best_result = {
            'named_playlists': {},
            'playlist_centroids': {},
            'playlist_to_centroid_vector_map': {},
        }
        
        result = select_top_n_diverse_playlists(best_result, n=5)
        
        assert result == best_result


class TestClusterNaming:
    """Tests for cluster naming function"""

    def test_name_cluster_basic(self):
        """Test cluster naming with basic centroid"""
        from tasks.clustering_helper import _name_cluster
        
        # Create a centroid vector [tempo, energy, mood1, mood2, mood3]
        centroid = np.array([0.8, 0.6, 0.9, 0.1, 0.2])
        mood_labels = ['rock', 'pop', 'jazz']
        
        # No scaling (direct interpretation)
        name, details = _name_cluster(
            centroid, 
            pca_model=None, 
            pca_enabled=False, 
            mood_labels=mood_labels, 
            scaler=None
        )
        
        assert isinstance(name, str)
        assert 'Fast' in name  # High tempo (0.8)
        assert isinstance(details, dict)
        assert 'rock' in details

    def test_name_cluster_slow_tempo(self):
        """Test cluster naming identifies slow tempo"""
        from tasks.clustering_helper import _name_cluster
        
        centroid = np.array([0.2, 0.4, 0.5, 0.3, 0.2])
        mood_labels = ['chill', 'relaxed', 'ambient']
        
        name, _ = _name_cluster(centroid, None, False, mood_labels, None)
        
        assert 'Slow' in name

    def test_name_cluster_medium_tempo(self):
        """Test cluster naming identifies medium tempo"""
        from tasks.clustering_helper import _name_cluster
        
        centroid = np.array([0.5, 0.5, 0.4, 0.4, 0.2])
        mood_labels = ['pop', 'dance', 'electronic']
        
        name, _ = _name_cluster(centroid, None, False, mood_labels, None)
        
        assert 'Medium' in name

    def test_name_cluster_top_moods_in_name(self):
        """Test that top moods appear in cluster name"""
        from tasks.clustering_helper import _name_cluster
        
        # High rock and pop, low jazz
        centroid = np.array([0.6, 0.5, 0.9, 0.8, 0.1])
        mood_labels = ['rock', 'pop', 'jazz']
        
        name, details = _name_cluster(centroid, None, False, mood_labels, None)
        
        # Should contain top mood (capitalized in function)
        assert 'Rock' in name or 'Pop' in name
        
        # Details should have all moods
        assert len(details) == 3

    def test_name_cluster_returns_correct_structure(self):
        """Test that naming returns tuple of (name, details)"""
        from tasks.clustering_helper import _name_cluster
        
        centroid = np.array([0.5, 0.5, 0.4, 0.4, 0.3])
        mood_labels = ['mood1', 'mood2', 'mood3']
        
        result = _name_cluster(centroid, None, False, mood_labels, None)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        name, details = result
        assert isinstance(name, str)
        assert isinstance(details, dict)