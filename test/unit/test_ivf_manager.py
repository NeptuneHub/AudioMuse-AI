import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestDirectEuclideanDistance:
    def test_identical_vectors_return_zero(self):
        from tasks.ivf_manager import _get_direct_euclidean_distance

        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        dist = _get_direct_euclidean_distance(v1, v2)

        assert dist == 0.0

    def test_known_distance(self):
        from tasks.ivf_manager import _get_direct_euclidean_distance

        v1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([3.0, 4.0, 0.0], dtype=np.float32)

        dist = _get_direct_euclidean_distance(v1, v2)

        assert abs(dist - 5.0) < 1e-5

    def test_none_vector_returns_inf(self):
        from tasks.ivf_manager import _get_direct_euclidean_distance

        v1 = np.array([1.0, 2.0], dtype=np.float32)

        assert _get_direct_euclidean_distance(None, v1) == float('inf')
        assert _get_direct_euclidean_distance(v1, None) == float('inf')
        assert _get_direct_euclidean_distance(None, None) == float('inf')

    def test_handles_different_dtypes(self):
        from tasks.ivf_manager import _get_direct_euclidean_distance

        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        v2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        dist = _get_direct_euclidean_distance(v1, v2)

        assert dist == 0.0


class TestDirectCosineDistance:
    def test_identical_vectors_return_zero(self):
        from tasks.ivf_manager import _get_direct_cosine_distance

        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        dist = _get_direct_cosine_distance(v1, v2)

        assert abs(dist) < 1e-5

    def test_orthogonal_vectors_return_one(self):
        from tasks.ivf_manager import _get_direct_cosine_distance

        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)

        dist = _get_direct_cosine_distance(v1, v2)

        assert abs(dist - 1.0) < 1e-5

    def test_opposite_vectors_return_two(self):
        from tasks.ivf_manager import _get_direct_cosine_distance

        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0], dtype=np.float32)

        dist = _get_direct_cosine_distance(v1, v2)

        assert abs(dist - 2.0) < 1e-5

    def test_none_vector_returns_inf(self):
        from tasks.ivf_manager import _get_direct_cosine_distance

        v1 = np.array([1.0, 2.0], dtype=np.float32)

        assert _get_direct_cosine_distance(None, v1) == float('inf')
        assert _get_direct_cosine_distance(v1, None) == float('inf')

    def test_zero_vector_returns_inf(self):
        from tasks.ivf_manager import _get_direct_cosine_distance

        v1 = np.array([0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 1.0], dtype=np.float32)

        dist = _get_direct_cosine_distance(v1, v2)

        assert dist == float('inf')

    def test_parallel_vectors_different_magnitude(self):
        from tasks.ivf_manager import _get_direct_cosine_distance

        v1 = np.array([1.0, 1.0], dtype=np.float32)
        v2 = np.array([10.0, 10.0], dtype=np.float32)

        dist = _get_direct_cosine_distance(v1, v2)

        assert abs(dist) < 1e-5


class TestGetDirectDistance:
    @patch('tasks.ivf_manager.IVF_METRIC', 'angular')
    def test_uses_cosine_for_angular_metric(self):
        from tasks.ivf_manager import get_direct_distance

        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)

        dist = get_direct_distance(v1, v2)

        assert abs(dist - 1.0) < 1e-5

    @patch('tasks.ivf_manager.IVF_METRIC', 'euclidean')
    def test_uses_euclidean_for_euclidean_metric(self):
        from tasks.ivf_manager import get_direct_distance

        v1 = np.array([0.0, 0.0], dtype=np.float32)
        v2 = np.array([3.0, 4.0], dtype=np.float32)

        dist = get_direct_distance(v1, v2)

        assert abs(dist - 5.0) < 1e-5


class TestNormalizeString:
    def test_lowercase_and_strip(self):
        from tasks.ivf_manager import _normalize_string

        assert _normalize_string("  Hello World  ") == "hello world"
        assert _normalize_string("UPPERCASE") == "uppercase"
        assert _normalize_string("  mixed CASE  ") == "mixed case"

    def test_empty_string(self):
        from tasks.ivf_manager import _normalize_string

        assert _normalize_string("") == ""
        assert _normalize_string("   ") == ""

    def test_none_returns_empty(self):
        from tasks.ivf_manager import _normalize_string

        assert _normalize_string(None) == ""


class TestIsSameSong:
    def test_exact_match(self):
        from tasks.ivf_manager import _is_same_song

        assert _is_same_song("Song Title", "Artist", "Song Title", "Artist") is True

    def test_case_insensitive_match(self):
        from tasks.ivf_manager import _is_same_song

        assert _is_same_song("SONG TITLE", "ARTIST", "song title", "artist") is True
        assert _is_same_song("Song Title", "Artist Name", "song title", "artist name") is True

    def test_whitespace_insensitive(self):
        from tasks.ivf_manager import _is_same_song

        assert _is_same_song("  Song Title  ", "  Artist  ", "Song Title", "Artist") is True

    def test_different_title_returns_false(self):
        from tasks.ivf_manager import _is_same_song

        assert _is_same_song("Song A", "Artist", "Song B", "Artist") is False

    def test_different_artist_returns_false(self):
        from tasks.ivf_manager import _is_same_song

        assert _is_same_song("Song", "Artist A", "Song", "Artist B") is False

    def test_empty_fields(self):
        from tasks.ivf_manager import _is_same_song

        assert _is_same_song("", "", "", "") is True
        assert _is_same_song("Song", "", "Song", "") is True
        assert _is_same_song("", "Artist", "", "Artist") is True


class TestParseMoodFeatures:
    def test_parses_valid_format(self):
        from tasks.ivf_manager import _parse_mood_features

        features_str = "danceable:0.5,aggressive:0.2,happy:0.8"

        result = _parse_mood_features(features_str)

        assert result['danceable'] == 0.5
        assert result['aggressive'] == 0.2
        assert result['happy'] == 0.8

    def test_handles_whitespace(self):
        from tasks.ivf_manager import _parse_mood_features

        features_str = " danceable : 0.5 , aggressive : 0.2 "

        result = _parse_mood_features(features_str)

        assert result['danceable'] == 0.5
        assert result['aggressive'] == 0.2

    def test_empty_string_returns_empty_dict(self):
        from tasks.ivf_manager import _parse_mood_features

        result = _parse_mood_features("")

        assert result == {}

    def test_invalid_format_returns_empty_dict(self):
        from tasks.ivf_manager import _parse_mood_features

        result = _parse_mood_features("danceable0.5")
        assert result == {}

    def test_non_numeric_value_returns_empty(self):
        from tasks.ivf_manager import _parse_mood_features

        result = _parse_mood_features("danceable:notanumber")

        assert result == {}


class TestGetVectorById:
    @patch('tasks.ivf_manager.ivf_index', None)
    def test_returns_none_when_index_not_loaded(self):
        from tasks.ivf_manager import get_vector_by_id

        result = get_vector_by_id('some-item-id')

        assert result is None

    @patch('tasks.ivf_manager.reverse_id_map', {'item-123': 0})
    @patch('tasks.ivf_manager.ivf_index')
    def test_returns_vector_when_found(self, mock_index):
        from tasks.ivf_manager import _get_cached_vector

        expected_vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mock_index.get_vector.return_value = expected_vector

        result = _get_cached_vector('item-123')

        np.testing.assert_array_equal(result, expected_vector)
        mock_index.get_vector.assert_called_once_with(0)

    @patch('tasks.ivf_manager.reverse_id_map', {})
    @patch('tasks.ivf_manager.ivf_index', Mock())
    def test_returns_none_when_item_not_in_map(self):
        from tasks.ivf_manager import _get_cached_vector

        result = _get_cached_vector('unknown-item')

        assert result is None

    def test_prefers_primed_exact_f32_over_index_int8(self):
        import tasks.ivf_manager as im

        exact = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_index = Mock()
        mock_index.get_vector.return_value = np.array([9.0, 9.0, 9.0], dtype=np.float32)
        with (
            patch.object(im, 'ivf_index', mock_index),
            patch.object(im, 'reverse_id_map', {'item-123': 0}),
        ):
            im._prime_request_f32({'item-123': exact})
            try:
                np.testing.assert_array_equal(im._get_cached_vector('item-123'), exact)
                mock_index.get_vector.assert_not_called()
            finally:
                im._clear_request_f32()
            np.testing.assert_array_equal(
                im._get_cached_vector('item-123'), np.array([9.0, 9.0, 9.0], dtype=np.float32)
            )


class TestLoadIVFIndex:
    @patch('tasks.ivf_manager.ivf_index', Mock())
    def test_skips_reload_if_already_loaded(self):
        from tasks.ivf_manager import load_ivf_index_for_querying

        with patch('app_helper.get_db') as mock_get_db:
            load_ivf_index_for_querying(force_reload=False)

            mock_get_db.assert_not_called()

    @patch('tasks.ivf_manager.ivf_index', None)
    @patch('tasks.ivf_manager.id_map', None)
    @patch('tasks.ivf_manager.reverse_id_map', None)
    def test_loads_index_from_database(self):
        import tasks.ivf_manager as vm

        with patch('app_helper.get_db') as mock_get_db:
            mock_get_db.return_value = Mock()
            with patch('tasks.paged_ivf.load_paged_ivf_index') as mock_load:
                mock_index = Mock()
                mock_load.return_value = (mock_index, {0: 'item-1'}, {'item-1': 0})

                from tasks.ivf_manager import load_ivf_index_for_querying

                load_ivf_index_for_querying(force_reload=True)

                mock_load.assert_called_once()
                assert vm.ivf_index is mock_index

    @patch('tasks.ivf_manager.ivf_index', None)
    @patch('tasks.ivf_manager.id_map', None)
    def test_handles_missing_index_gracefully(self):
        with patch('app_helper.get_db') as mock_get_db:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_get_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchone.return_value = None

            from tasks.ivf_manager import load_ivf_index_for_querying

            load_ivf_index_for_querying(force_reload=True)


class TestFindNearestNeighborsById:
    @patch('tasks.ivf_manager.ivf_index', None)
    def test_raises_when_index_not_loaded(self):
        from tasks.ivf_manager import find_nearest_neighbors_by_id

        with pytest.raises(RuntimeError, match="IVF index is not loaded"):
            find_nearest_neighbors_by_id('item-123', n=10)

    @patch('tasks.ivf_manager.ivf_index', Mock())
    @patch('tasks.ivf_manager.id_map', None)
    def test_raises_when_id_map_not_loaded(self):
        from tasks.ivf_manager import find_nearest_neighbors_by_id

        with pytest.raises(RuntimeError, match="IVF index is not loaded"):
            find_nearest_neighbors_by_id('item-123', n=10)

    @patch('tasks.ivf_manager.ivf_index', Mock())
    @patch('tasks.ivf_manager.id_map', {0: 'item-1'})
    @patch('tasks.ivf_manager.reverse_id_map', None)
    def test_raises_when_reverse_id_map_not_loaded(self):
        from tasks.ivf_manager import find_nearest_neighbors_by_id

        with pytest.raises(RuntimeError, match="IVF index is not loaded"):
            find_nearest_neighbors_by_id('item-123', n=10)


class TestFindNearestNeighborsByVector:
    @patch('tasks.ivf_manager.ivf_index', None)
    def test_raises_when_index_not_loaded(self):
        from tasks.ivf_manager import find_nearest_neighbors_by_vector

        query_vec = np.array([1.0, 2.0], dtype=np.float32)

        with pytest.raises(RuntimeError, match="IVF index is not loaded"):
            find_nearest_neighbors_by_vector(query_vec, n=10)

    @patch('tasks.ivf_manager.ivf_index', Mock())
    @patch('tasks.ivf_manager.id_map', None)
    def test_raises_when_id_map_not_loaded(self):
        from tasks.ivf_manager import find_nearest_neighbors_by_vector

        query_vec = np.array([1.0, 2.0], dtype=np.float32)

        with pytest.raises(RuntimeError, match="IVF index is not loaded"):
            find_nearest_neighbors_by_vector(query_vec, n=10)


class TestCreatePlaylistFromIds:
    @patch('tasks.mediaserver.create_instant_playlist')
    def test_calls_mediaserver_create_playlist(self, mock_create):
        from tasks.ivf_manager import create_playlist_from_ids

        mock_create.return_value = {'Id': 'playlist-123', 'Name': 'Test Playlist'}

        result = create_playlist_from_ids('Test Playlist', ['track-1', 'track-2'])

        assert result == 'playlist-123'
        mock_create.assert_called_once_with(
            'Test Playlist', ['track-1', 'track-2'], user_creds=None
        )

    @patch('tasks.mediaserver.create_instant_playlist')
    def test_raises_on_creation_failure(self, mock_create):
        from tasks.ivf_manager import create_playlist_from_ids

        mock_create.return_value = None

        with pytest.raises(Exception, match="Playlist creation failed"):
            create_playlist_from_ids('Test Playlist', ['track-1'])

    @patch('tasks.mediaserver.create_instant_playlist')
    def test_raises_on_missing_playlist_id(self, mock_create):
        from tasks.ivf_manager import create_playlist_from_ids

        mock_create.return_value = {'Name': 'Test'}

        with pytest.raises(Exception, match="did not include a playlist ID"):
            create_playlist_from_ids('Test Playlist', ['track-1'])

    @patch('tasks.mediaserver.create_instant_playlist')
    def test_passes_user_credentials(self, mock_create):
        from tasks.ivf_manager import create_playlist_from_ids

        mock_create.return_value = {'Id': 'playlist-123'}
        user_creds = {'user_id': 'user1', 'token': 'token123'}

        create_playlist_from_ids('Test', ['track-1'], user_creds=user_creds)

        mock_create.assert_called_once_with('Test', ['track-1'], user_creds=user_creds)


class TestSearchTracksByTitleAndArtist:
    def test_search_with_single_keyword(self):
        with patch('app_helper.get_db') as mock_get_db:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_get_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = [
                {'item_id': 'item-1', 'title': 'Test Song', 'author': 'Artist 1'},
                {'item_id': 'item-2', 'title': 'Song 2', 'author': 'Test Artist'},
            ]

            from tasks.ivf_manager import search_tracks_unified

            results = search_tracks_unified("test", limit=10)

            mock_cursor.execute.assert_called_once()

            query, params = mock_cursor.execute.call_args[0]
            assert query.count("search_u LIKE") == 1
            assert "AND search_u LIKE" not in query
            assert query.count("CASE WHEN lower(unaccent(title))") == 1
            assert query.count("CASE WHEN lower(unaccent(author))") == 1
            assert query.count("CASE WHEN lower(unaccent(album))") == 1
            assert "%test%" in params
            assert len(results) == 2
            assert {r['item_id'] for r in results} == {"item-1", "item-2"}

    def test_search_with_two_keywords(self):
        with patch('app_helper.get_db') as mock_get_db:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_get_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = [
                {'item_id': 'item-1', 'title': 'Test Song', 'author': 'Artist 1'},
                {'item_id': 'item-2', 'title': 'Song 2', 'author': 'Test Artist'},
            ]

            from tasks.ivf_manager import search_tracks_unified

            results = search_tracks_unified("test song", limit=10)

            mock_cursor.execute.assert_called_once()

            query, params = mock_cursor.execute.call_args[0]
            assert query.count("search_u LIKE") == 2
            assert query.count("AND search_u LIKE") == 1
            assert query.count("CASE WHEN lower(unaccent(title))") == 2
            assert query.count("CASE WHEN lower(unaccent(author))") == 2
            assert query.count("CASE WHEN lower(unaccent(album))") == 2
            assert "%test%" in params
            assert "%song%" in params
            assert len(results) == 2
            assert {r['item_id'] for r in results} == {"item-1", "item-2"}

    def test_returns_empty_for_no_query(self):
        with patch('app_helper.get_db') as mock_get_db:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_get_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor

            from tasks.ivf_manager import search_tracks_unified

            results = search_tracks_unified('')

            assert results == []


class TestGetItemIdByTitleAndArtist:
    def test_finds_exact_match(self):
        with patch('app_helper.get_db') as mock_get_db:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_get_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchone.return_value = {'item_id': 'found-item'}

            from tasks.ivf_manager import get_item_id_by_title_and_artist

            result = get_item_id_by_title_and_artist('Song Title', 'Artist Name')

            assert result == 'found-item'

    def test_returns_none_when_not_found(self):
        with patch('app_helper.get_db') as mock_get_db:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_get_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchone.return_value = None

            from tasks.ivf_manager import get_item_id_by_title_and_artist

            result = get_item_id_by_title_and_artist('Unknown', 'Unknown')

            assert result is None


class TestCleanupResources:
    @patch('tasks.ivf_manager._shutdown_thread_pool')
    def test_shuts_down_pool(self, mock_shutdown):
        from tasks.ivf_manager import cleanup_resources

        cleanup_resources()

        mock_shutdown.assert_called_once()


class TestGetMaxDistanceForId:
    @pytest.fixture(autouse=True)
    def _clear_max_distance_cache(self):
        from tasks.ivf_manager import _max_distance_cache

        _max_distance_cache.clear()
        yield

    @patch('tasks.ivf_manager.ivf_index', None)
    def test_raises_when_index_not_loaded(self):
        from tasks.ivf_manager import get_max_distance_for_id

        with pytest.raises(RuntimeError, match="IVF index is not loaded"):
            get_max_distance_for_id('item-123')

    @patch('tasks.ivf_manager.ivf_index')
    @patch('tasks.ivf_manager.id_map', {0: 'item-1', 1: 'item-2', 2: 'item-3'})
    @patch('tasks.ivf_manager.reverse_id_map', {'item-1': 0, 'item-2': 1, 'item-3': 2})
    def test_finds_farthest_item(self, mock_index):
        mock_index.get_max_distance.return_value = (1.5, 2)
        mock_index.__len__ = Mock(return_value=3)

        from tasks.ivf_manager import get_max_distance_for_id

        result = get_max_distance_for_id('item-1')

        assert result['max_distance'] == 1.5
        assert result['farthest_item_id'] == 'item-3'

    @patch('tasks.ivf_manager.ivf_index')
    @patch('tasks.ivf_manager.id_map', {0: 'item-1'})
    @patch('tasks.ivf_manager.reverse_id_map', {'item-1': 0})
    def test_single_item_index(self, mock_index):
        mock_index.get_max_distance.return_value = (0.0, None)
        mock_index.__len__ = Mock(return_value=1)

        from tasks.ivf_manager import get_max_distance_for_id

        result = get_max_distance_for_id('item-1')

        assert result['max_distance'] == 0.0
        assert result['farthest_item_id'] is None

    @patch('tasks.ivf_manager.ivf_index')
    @patch('tasks.ivf_manager.id_map', {})
    @patch('tasks.ivf_manager.reverse_id_map', {})
    def test_unknown_item_returns_none(self, mock_index):
        from tasks.ivf_manager import get_max_distance_for_id

        result = get_max_distance_for_id('unknown-item')

        assert result is None


class TestResultCache:
    def test_lru_eviction_and_clear(self):
        from tasks.ivf_manager import _ResultCache

        c = _ResultCache(100, 2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("cc", 3)
        assert c.get("a") is None
        assert c.get("b") == 2
        assert c.get("cc") == 3
        c.clear()
        assert c.get("b") is None

    def test_ttl_expiry(self):
        import time
        from tasks.ivf_manager import _ResultCache

        c = _ResultCache(0.05, 10)
        c.put("k", 42)
        assert c.get("k") == 42
        time.sleep(0.1)
        assert c.get("k") is None

    def test_sweep_expired_drops_only_expired_entries(self):
        import time
        from tasks.ivf_manager import _ResultCache

        c = _ResultCache(0.05, 10)
        c.put("old", 1)
        time.sleep(0.1)
        c.put("fresh", 2)
        c.sweep_expired()
        assert c.get("old") is None
        assert c.get("fresh") == 2

    def test_sweep_expired_noop_when_ttl_zero(self):
        from tasks.ivf_manager import _ResultCache

        c = _ResultCache(0, 10)
        c.sweep_expired()

    def test_disabled_when_ttl_zero(self):
        from tasks.ivf_manager import _ResultCache

        c = _ResultCache(0, 10)
        c.put("k", 1)
        assert c.get("k") is None
