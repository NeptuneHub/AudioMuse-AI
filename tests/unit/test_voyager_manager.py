# tests/unit/test_voyager_manager.py
"""
Unit tests for tasks/voyager_manager.py

Tests cover:
- Distance calculation functions (Euclidean and Cosine)
- String normalization and song comparison
- Mood feature parsing
- Vector caching
- Deduplication logic
- Index loading and querying
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# DISTANCE CALCULATION TESTS - Pure functions, no mocking needed
# =============================================================================

class TestDirectEuclideanDistance:
    """Test Euclidean distance calculation"""

    def test_identical_vectors_return_zero(self):
        """Identical vectors should have distance 0"""
        from tasks.voyager_manager import _get_direct_euclidean_distance
        
        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        dist = _get_direct_euclidean_distance(v1, v2)
        
        assert dist == 0.0

    def test_known_distance(self):
        """Test with known Euclidean distance"""
        from tasks.voyager_manager import _get_direct_euclidean_distance
        
        v1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        
        dist = _get_direct_euclidean_distance(v1, v2)
        
        # sqrt(3^2 + 4^2) = 5
        assert abs(dist - 5.0) < 1e-5

    def test_none_vector_returns_inf(self):
        """None vectors should return infinity"""
        from tasks.voyager_manager import _get_direct_euclidean_distance
        
        v1 = np.array([1.0, 2.0], dtype=np.float32)
        
        assert _get_direct_euclidean_distance(None, v1) == float('inf')
        assert _get_direct_euclidean_distance(v1, None) == float('inf')
        assert _get_direct_euclidean_distance(None, None) == float('inf')

    def test_handles_different_dtypes(self):
        """Should handle different numpy dtypes"""
        from tasks.voyager_manager import _get_direct_euclidean_distance
        
        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        v2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        dist = _get_direct_euclidean_distance(v1, v2)
        
        assert dist == 0.0


class TestDirectCosineDistance:
    """Test Cosine distance calculation (1 - cosine_similarity)"""

    def test_identical_vectors_return_zero(self):
        """Identical vectors should have cosine distance 0"""
        from tasks.voyager_manager import _get_direct_cosine_distance
        
        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        dist = _get_direct_cosine_distance(v1, v2)
        
        assert abs(dist) < 1e-5  # Should be ~0

    def test_orthogonal_vectors_return_one(self):
        """Orthogonal vectors should have cosine distance 1"""
        from tasks.voyager_manager import _get_direct_cosine_distance
        
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        
        dist = _get_direct_cosine_distance(v1, v2)
        
        assert abs(dist - 1.0) < 1e-5

    def test_opposite_vectors_return_two(self):
        """Opposite vectors should have cosine distance 2"""
        from tasks.voyager_manager import _get_direct_cosine_distance
        
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0], dtype=np.float32)
        
        dist = _get_direct_cosine_distance(v1, v2)
        
        assert abs(dist - 2.0) < 1e-5

    def test_none_vector_returns_inf(self):
        """None vectors should return infinity"""
        from tasks.voyager_manager import _get_direct_cosine_distance
        
        v1 = np.array([1.0, 2.0], dtype=np.float32)
        
        assert _get_direct_cosine_distance(None, v1) == float('inf')
        assert _get_direct_cosine_distance(v1, None) == float('inf')

    def test_zero_vector_returns_inf(self):
        """Zero vector should return infinity (division by zero)"""
        from tasks.voyager_manager import _get_direct_cosine_distance
        
        v1 = np.array([0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 1.0], dtype=np.float32)
        
        dist = _get_direct_cosine_distance(v1, v2)
        
        assert dist == float('inf')

    def test_parallel_vectors_different_magnitude(self):
        """Parallel vectors with different magnitudes should have distance ~0"""
        from tasks.voyager_manager import _get_direct_cosine_distance
        
        v1 = np.array([1.0, 1.0], dtype=np.float32)
        v2 = np.array([10.0, 10.0], dtype=np.float32)
        
        dist = _get_direct_cosine_distance(v1, v2)
        
        assert abs(dist) < 1e-5  # Same direction = similar


class TestGetDirectDistance:
    """Test the public distance helper that picks metric based on config"""

    @patch('tasks.voyager_manager.VOYAGER_METRIC', 'angular')
    def test_uses_cosine_for_angular_metric(self):
        """Should use cosine distance when metric is 'angular'"""
        from tasks.voyager_manager import get_direct_distance
        
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        
        dist = get_direct_distance(v1, v2)
        
        # Orthogonal vectors have cosine distance = 1
        assert abs(dist - 1.0) < 1e-5

    @patch('tasks.voyager_manager.VOYAGER_METRIC', 'euclidean')
    def test_uses_euclidean_for_euclidean_metric(self):
        """Should use Euclidean distance when metric is 'euclidean'"""
        from tasks.voyager_manager import get_direct_distance
        
        v1 = np.array([0.0, 0.0], dtype=np.float32)
        v2 = np.array([3.0, 4.0], dtype=np.float32)
        
        dist = get_direct_distance(v1, v2)
        
        # sqrt(3^2 + 4^2) = 5
        assert abs(dist - 5.0) < 1e-5


# =============================================================================
# STRING NORMALIZATION AND COMPARISON TESTS
# =============================================================================

class TestNormalizeString:
    """Test string normalization for song comparison"""

    def test_lowercase_and_strip(self):
        """Should lowercase and strip whitespace"""
        from tasks.voyager_manager import _normalize_string
        
        assert _normalize_string("  Hello World  ") == "hello world"
        assert _normalize_string("UPPERCASE") == "uppercase"
        assert _normalize_string("  mixed CASE  ") == "mixed case"

    def test_empty_string(self):
        """Should handle empty strings"""
        from tasks.voyager_manager import _normalize_string
        
        assert _normalize_string("") == ""
        assert _normalize_string("   ") == ""

    def test_none_returns_empty(self):
        """Should handle None"""
        from tasks.voyager_manager import _normalize_string
        
        assert _normalize_string(None) == ""


class TestIsSameSong:
    """Test song comparison logic"""

    def test_exact_match(self):
        """Exact match should return True"""
        from tasks.voyager_manager import _is_same_song
        
        assert _is_same_song("Song Title", "Artist", "Song Title", "Artist") is True

    def test_case_insensitive_match(self):
        """Case-insensitive match should return True"""
        from tasks.voyager_manager import _is_same_song
        
        assert _is_same_song("SONG TITLE", "ARTIST", "song title", "artist") is True
        assert _is_same_song("Song Title", "Artist Name", "song title", "artist name") is True

    def test_whitespace_insensitive(self):
        """Whitespace differences should still match"""
        from tasks.voyager_manager import _is_same_song
        
        assert _is_same_song("  Song Title  ", "  Artist  ", "Song Title", "Artist") is True

    def test_different_title_returns_false(self):
        """Different titles should return False"""
        from tasks.voyager_manager import _is_same_song
        
        assert _is_same_song("Song A", "Artist", "Song B", "Artist") is False

    def test_different_artist_returns_false(self):
        """Different artists should return False"""
        from tasks.voyager_manager import _is_same_song
        
        assert _is_same_song("Song", "Artist A", "Song", "Artist B") is False

    def test_empty_fields(self):
        """Empty fields should be handled"""
        from tasks.voyager_manager import _is_same_song
        
        assert _is_same_song("", "", "", "") is True
        assert _is_same_song("Song", "", "Song", "") is True
        assert _is_same_song("", "Artist", "", "Artist") is True


# =============================================================================
# MOOD FEATURE PARSING TESTS
# =============================================================================

class TestParseMoodFeatures:
    """Test mood feature string parsing"""

    def test_parses_valid_format(self):
        """Should parse comma-separated key:value pairs"""
        from tasks.voyager_manager import _parse_mood_features
        
        features_str = "danceable:0.5,aggressive:0.2,happy:0.8"
        
        result = _parse_mood_features(features_str)
        
        assert result['danceable'] == 0.5
        assert result['aggressive'] == 0.2
        assert result['happy'] == 0.8

    def test_handles_whitespace(self):
        """Should handle whitespace around keys and values"""
        from tasks.voyager_manager import _parse_mood_features
        
        features_str = " danceable : 0.5 , aggressive : 0.2 "
        
        result = _parse_mood_features(features_str)
        
        assert result['danceable'] == 0.5
        assert result['aggressive'] == 0.2

    def test_empty_string_returns_empty_dict(self):
        """Empty string should return empty dict"""
        from tasks.voyager_manager import _parse_mood_features
        
        result = _parse_mood_features("")
        
        assert result == {}

    def test_invalid_format_returns_empty_dict(self):
        """Invalid format should return empty dict without crashing"""
        from tasks.voyager_manager import _parse_mood_features
        
        # Missing colon
        result = _parse_mood_features("danceable0.5")
        assert result == {}

    def test_non_numeric_value_returns_empty(self):
        """Non-numeric values should cause empty dict return"""
        from tasks.voyager_manager import _parse_mood_features
        
        result = _parse_mood_features("danceable:notanumber")
        
        # Should fail gracefully
        assert result == {}


# =============================================================================
# VECTOR CACHING TESTS
# =============================================================================

class TestGetVectorById:
    """Test vector retrieval by item ID"""

    @patch('tasks.voyager_manager.voyager_index', None)
    def test_returns_none_when_index_not_loaded(self):
        """Should return None if index not loaded"""
        from tasks.voyager_manager import get_vector_by_id, _get_cached_vector
        
        # Clear the cache first
        _get_cached_vector.cache_clear()
        
        result = get_vector_by_id('some-item-id')
        
        assert result is None

    @patch('tasks.voyager_manager.reverse_id_map', {'item-123': 0})
    @patch('tasks.voyager_manager.voyager_index')
    def test_returns_vector_when_found(self, mock_index):
        """Should return vector when item exists"""
        from tasks.voyager_manager import _get_cached_vector
        
        # Clear the cache first
        _get_cached_vector.cache_clear()
        
        expected_vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mock_index.get_vector.return_value = expected_vector
        
        result = _get_cached_vector('item-123')
        
        np.testing.assert_array_equal(result, expected_vector)
        mock_index.get_vector.assert_called_once_with(0)

    @patch('tasks.voyager_manager.reverse_id_map', {})
    @patch('tasks.voyager_manager.voyager_index', Mock())
    def test_returns_none_when_item_not_in_map(self):
        """Should return None if item not in reverse_id_map"""
        from tasks.voyager_manager import _get_cached_vector
        
        # Clear the cache first
        _get_cached_vector.cache_clear()
        
        result = _get_cached_vector('unknown-item')
        
        assert result is None


# =============================================================================
# INDEX LOADING TESTS
# =============================================================================

class TestLoadVoyagerIndex:
    """Test Voyager index loading from database"""

    @patch('tasks.voyager_manager.voyager_index', Mock())
    def test_skips_reload_if_already_loaded(self):
        """Should skip reload if index already in memory"""
        from tasks.voyager_manager import load_voyager_index_for_querying
        
        with patch('app_helper.get_db') as mock_get_db:
            load_voyager_index_for_querying(force_reload=False)
            
            # get_db should not be called if index already loaded
            mock_get_db.assert_not_called()

    @patch('tasks.voyager_manager.voyager_index', None)
    @patch('tasks.voyager_manager.id_map', None)
    @patch('tasks.voyager_manager.reverse_id_map', None)
    def test_loads_index_from_database(self):
        """Should load index from database when not in memory"""
        import json
        import io
        
        with patch('app_helper.get_db') as mock_get_db:
            # Mock database connection and cursor
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_get_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Mock the database record - use minimal binary data
            fake_id_map = {0: 'item-1', 1: 'item-2'}
            mock_cursor.fetchone.return_value = (
                b'fake_binary_data',  # index_data
                json.dumps(fake_id_map),  # id_map_json
                128  # embedding_dimension
            )
            
            with patch('tasks.voyager_manager.voyager') as mock_voyager:
                with patch('tasks.voyager_manager.EMBEDDING_DIMENSION', 128):
                    mock_loaded_index = Mock()
                    mock_voyager.Index.load.return_value = mock_loaded_index
                    
                    from tasks.voyager_manager import load_voyager_index_for_querying
                    load_voyager_index_for_querying(force_reload=True)
                    
                    # Verify index was loaded
                    mock_voyager.Index.load.assert_called_once()

    @patch('tasks.voyager_manager.voyager_index', None)
    @patch('tasks.voyager_manager.id_map', None)
    def test_handles_missing_index_gracefully(self):
        """Should handle missing index in database"""
        with patch('app_helper.get_db') as mock_get_db:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_get_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchone.return_value = None  # No index found
            
            from tasks.voyager_manager import load_voyager_index_for_querying
            
            # Should not raise
            load_voyager_index_for_querying(force_reload=True)


# =============================================================================
# NEIGHBOR SEARCH TESTS
# =============================================================================

class TestFindNearestNeighborsById:
    """Test nearest neighbor search functionality"""

    @patch('tasks.voyager_manager.voyager_index', None)
    def test_raises_when_index_not_loaded(self):
        """Should raise RuntimeError if index not loaded"""
        from tasks.voyager_manager import find_nearest_neighbors_by_id
        
        with pytest.raises(RuntimeError, match="Voyager index is not loaded"):
            find_nearest_neighbors_by_id('item-123', n=10)

    @patch('tasks.voyager_manager.voyager_index', Mock())
    @patch('tasks.voyager_manager.id_map', None)
    def test_raises_when_id_map_not_loaded(self):
        """Should raise RuntimeError if id_map not loaded"""
        from tasks.voyager_manager import find_nearest_neighbors_by_id
        
        with pytest.raises(RuntimeError, match="Voyager index is not loaded"):
            find_nearest_neighbors_by_id('item-123', n=10)

    @patch('tasks.voyager_manager.voyager_index', Mock())
    @patch('tasks.voyager_manager.id_map', {0: 'item-1'})
    @patch('tasks.voyager_manager.reverse_id_map', None)
    def test_raises_when_reverse_id_map_not_loaded(self):
        """Should raise RuntimeError if reverse_id_map not loaded"""
        from tasks.voyager_manager import find_nearest_neighbors_by_id
        
        with pytest.raises(RuntimeError, match="Voyager index is not loaded"):
            find_nearest_neighbors_by_id('item-123', n=10)


class TestFindNearestNeighborsByVector:
    """Test nearest neighbor search by vector"""

    @patch('tasks.voyager_manager.voyager_index', None)
    def test_raises_when_index_not_loaded(self):
        """Should raise RuntimeError if index not loaded"""
        from tasks.voyager_manager import find_nearest_neighbors_by_vector
        
        query_vec = np.array([1.0, 2.0], dtype=np.float32)
        
        with pytest.raises(RuntimeError, match="Voyager index is not loaded"):
            find_nearest_neighbors_by_vector(query_vec, n=10)

    @patch('tasks.voyager_manager.voyager_index', Mock())
    @patch('tasks.voyager_manager.id_map', None)
    def test_raises_when_id_map_not_loaded(self):
        """Should raise RuntimeError if id_map not loaded"""
        from tasks.voyager_manager import find_nearest_neighbors_by_vector
        
        query_vec = np.array([1.0, 2.0], dtype=np.float32)
        
        with pytest.raises(RuntimeError, match="Voyager index is not loaded"):
            find_nearest_neighbors_by_vector(query_vec, n=10)


# =============================================================================
# PLAYLIST CREATION TESTS
# =============================================================================

class TestCreatePlaylistFromIds:
    """Test playlist creation functionality"""

    @patch('tasks.voyager_manager.create_instant_playlist')
    def test_calls_mediaserver_create_playlist(self, mock_create):
        """Should call mediaserver create_instant_playlist"""
        from tasks.voyager_manager import create_playlist_from_ids
        
        mock_create.return_value = {'Id': 'playlist-123', 'Name': 'Test Playlist'}
        
        result = create_playlist_from_ids('Test Playlist', ['track-1', 'track-2'])
        
        assert result == 'playlist-123'
        mock_create.assert_called_once_with('Test Playlist', ['track-1', 'track-2'], user_creds=None)

    @patch('tasks.voyager_manager.create_instant_playlist')
    def test_raises_on_creation_failure(self, mock_create):
        """Should raise exception if playlist creation fails"""
        from tasks.voyager_manager import create_playlist_from_ids
        
        mock_create.return_value = None
        
        with pytest.raises(Exception, match="Playlist creation failed"):
            create_playlist_from_ids('Test Playlist', ['track-1'])

    @patch('tasks.voyager_manager.create_instant_playlist')
    def test_raises_on_missing_playlist_id(self, mock_create):
        """Should raise exception if response has no Id"""
        from tasks.voyager_manager import create_playlist_from_ids
        
        mock_create.return_value = {'Name': 'Test'}  # Missing 'Id'
        
        with pytest.raises(Exception, match="did not include a playlist ID"):
            create_playlist_from_ids('Test Playlist', ['track-1'])

    @patch('tasks.voyager_manager.create_instant_playlist')
    def test_passes_user_credentials(self, mock_create):
        """Should pass user credentials to mediaserver"""
        from tasks.voyager_manager import create_playlist_from_ids
        
        mock_create.return_value = {'Id': 'playlist-123'}
        user_creds = {'user_id': 'user1', 'token': 'token123'}
        
        create_playlist_from_ids('Test', ['track-1'], user_creds=user_creds)
        
        mock_create.assert_called_once_with('Test', ['track-1'], user_creds=user_creds)


# =============================================================================
# SEARCH TESTS
# =============================================================================

class TestSearchTracksByTitleAndArtist:
    """Test track search functionality"""

    def test_searches_with_title_only(self):
        """Should search by title when no artist provided"""
        with patch('app_helper.get_db') as mock_get_db:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_get_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = [
                {'item_id': 'item-1', 'title': 'Test Song', 'author': 'Artist 1'}
            ]
            mock_cursor.__iter__ = Mock(return_value=iter([]))
            
            from tasks.voyager_manager import search_tracks_by_title_and_artist
            
            results = search_tracks_by_title_and_artist('Test Song', '', limit=10)
            
            # Verify query was executed
            mock_cursor.execute.assert_called_once()
            # The query should search both title and author when only title provided
            call_args = mock_cursor.execute.call_args
            assert 'ILIKE' in call_args[0][0]

    def test_returns_empty_for_no_query(self):
        """Should return empty list when no search terms"""
        with patch('app_helper.get_db') as mock_get_db:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_get_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            from tasks.voyager_manager import search_tracks_by_title_and_artist
            
            results = search_tracks_by_title_and_artist('', '')
            
            assert results == []


class TestGetItemIdByTitleAndArtist:
    """Test item ID lookup by title/artist"""

    def test_finds_exact_match(self):
        """Should find item with exact title/artist match"""
        with patch('app_helper.get_db') as mock_get_db:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_get_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchone.return_value = {'item_id': 'found-item'}
            
            from tasks.voyager_manager import get_item_id_by_title_and_artist
            
            result = get_item_id_by_title_and_artist('Song Title', 'Artist Name')
            
            assert result == 'found-item'

    def test_returns_none_when_not_found(self):
        """Should return None when no match found"""
        with patch('app_helper.get_db') as mock_get_db:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_get_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchone.return_value = None
            
            from tasks.voyager_manager import get_item_id_by_title_and_artist
            
            result = get_item_id_by_title_and_artist('Unknown', 'Unknown')
            
            assert result is None


# =============================================================================
# CLEANUP TESTS
# =============================================================================

class TestCleanupResources:
    """Test resource cleanup functionality"""

    @patch('tasks.voyager_manager._shutdown_thread_pool')
    @patch('tasks.voyager_manager._get_cached_vector')
    def test_clears_cache_and_shuts_down_pool(self, mock_cache, mock_shutdown):
        """Should clear cache and shutdown thread pool"""
        mock_cache.cache_clear = Mock()
        
        from tasks.voyager_manager import cleanup_resources
        
        cleanup_resources()
        
        mock_shutdown.assert_called_once()
        mock_cache.cache_clear.assert_called_once()


# =============================================================================
# MAX DISTANCE TESTS
# =============================================================================

class TestGetMaxDistanceForId:
    """Test max distance calculation"""

    @patch('tasks.voyager_manager.voyager_index', None)
    def test_raises_when_index_not_loaded(self):
        """Should raise RuntimeError if index not loaded"""
        from tasks.voyager_manager import get_max_distance_for_id
        
        with pytest.raises(RuntimeError, match="Voyager index is not loaded"):
            get_max_distance_for_id('item-123')

    @patch('tasks.voyager_manager.voyager_index')
    @patch('tasks.voyager_manager.id_map', {0: 'item-1', 1: 'item-2', 2: 'item-3'})
    @patch('tasks.voyager_manager.reverse_id_map', {'item-1': 0, 'item-2': 1, 'item-3': 2})
    def test_finds_farthest_item(self, mock_index):
        """Should find the farthest item from target"""
        mock_index.get_vector.return_value = np.array([1.0, 2.0], dtype=np.float32)
        mock_index.query.return_value = (
            [0, 1, 2],  # voyager IDs
            [0.0, 0.5, 1.5]  # distances - item-3 (id=2) is farthest
        )
        mock_index.__len__ = Mock(return_value=3)
        
        from tasks.voyager_manager import get_max_distance_for_id
        
        result = get_max_distance_for_id('item-1')
        
        assert result['max_distance'] == 1.5
        assert result['farthest_item_id'] == 'item-3'

    @patch('tasks.voyager_manager.voyager_index')
    @patch('tasks.voyager_manager.id_map', {0: 'item-1'})
    @patch('tasks.voyager_manager.reverse_id_map', {'item-1': 0})
    def test_single_item_index(self, mock_index):
        """Should handle single-item index"""
        mock_index.get_vector.return_value = np.array([1.0], dtype=np.float32)
        mock_index.query.return_value = ([0], [0.0])  # Only self
        mock_index.__len__ = Mock(return_value=1)
        
        from tasks.voyager_manager import get_max_distance_for_id
        
        result = get_max_distance_for_id('item-1')
        
        assert result['max_distance'] == 0.0
        assert result['farthest_item_id'] is None

    @patch('tasks.voyager_manager.voyager_index')
    @patch('tasks.voyager_manager.id_map', {})
    @patch('tasks.voyager_manager.reverse_id_map', {})
    def test_unknown_item_returns_none(self, mock_index):
        """Should return None for unknown item"""
        from tasks.voyager_manager import get_max_distance_for_id
        
        result = get_max_distance_for_id('unknown-item')
        
        assert result is None
