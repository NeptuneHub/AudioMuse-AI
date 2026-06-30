import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from tasks.sonic_fingerprint_manager import generate_sonic_fingerprint


class TestGenerateSonicFingerprint:
    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_generates_fingerprint_with_recent_songs(
        self, mock_ivf, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': 's1'}, {'Id': 's2'}, {'Id': 's3'}]

        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0, 0.0, 0.0])},
            {'item_id': 's2', 'embedding_vector': np.array([0.0, 1.0, 0.0])},
            {'item_id': 's3', 'embedding_vector': np.array([0.0, 0.0, 1.0])},
        ]

        mock_last_played.return_value = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        mock_ivf.return_value = [
            {'item_id': 's4', 'distance': 0.1},
            {'item_id': 's5', 'distance': 0.2},
        ]

        result = generate_sonic_fingerprint(num_neighbors=5)

        assert len(result) == 5
        assert result[0]['item_id'] in ['s1', 's2', 's3']
        assert result[0]['distance'] == 0.0

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    def test_returns_empty_when_no_top_songs(self, mock_top_songs):
        mock_top_songs.return_value = []

        result = generate_sonic_fingerprint(num_neighbors=10)

        assert result == []

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    def test_returns_empty_when_no_embeddings(self, mock_get_tracks, mock_top_songs):
        mock_top_songs.return_value = [{'Id': 's1'}, {'Id': 's2'}]
        mock_get_tracks.return_value = []

        result = generate_sonic_fingerprint(num_neighbors=10)

        assert result == []

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_weights_recent_songs_higher(
        self, mock_ivf, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': 's1'}, {'Id': 's2'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0, 0.0])},
            {'item_id': 's2', 'embedding_vector': np.array([0.0, 1.0])},
        ]

        def last_played_side_effect(song_id, user_creds=None):
            if song_id == 's1':
                return (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            else:
                return (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()

        mock_last_played.side_effect = last_played_side_effect
        mock_ivf.return_value = []

        result = generate_sonic_fingerprint(num_neighbors=2)

        assert len(result) == 2

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    def test_truncates_when_seed_songs_exceed_desired_size(
        self, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': f's{i}'} for i in range(10)]
        mock_get_tracks.return_value = [
            {'item_id': f's{i}', 'embedding_vector': np.random.rand(128)} for i in range(10)
        ]
        mock_last_played.return_value = None

        result = generate_sonic_fingerprint(num_neighbors=5)

        assert len(result) == 5

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_deduplicates_ivf_results(
        self, mock_ivf, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': 's1'}, {'Id': 's2'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0, 0.0])},
            {'item_id': 's2', 'embedding_vector': np.array([0.0, 1.0])},
        ]
        mock_last_played.return_value = None

        mock_ivf.return_value = [
            {'item_id': 's1', 'distance': 0.05},
            {'item_id': 's3', 'distance': 0.1},
        ]

        result = generate_sonic_fingerprint(num_neighbors=5)

        item_ids = [r['item_id'] for r in result]
        assert item_ids.count('s1') == 1
        assert 's2' in item_ids
        assert 's3' in item_ids

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_seed_songs_have_zero_distance(
        self, mock_ivf, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [{'item_id': 's1', 'embedding_vector': np.array([1.0])}]
        mock_last_played.return_value = None
        mock_ivf.return_value = [{'item_id': 's2', 'distance': 0.5}]

        result = generate_sonic_fingerprint(num_neighbors=2)

        assert result[0]['item_id'] == 's1'
        assert result[0]['distance'] == 0.0
        assert result[1]['item_id'] == 's2'
        assert result[1]['distance'] == 0.5

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_handles_invalid_last_played_date(
        self, mock_ivf, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [{'item_id': 's1', 'embedding_vector': np.array([1.0])}]
        mock_last_played.return_value = "invalid-date"
        mock_ivf.return_value = []

        result = generate_sonic_fingerprint(num_neighbors=1)

        assert len(result) == 1

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_passes_user_credentials(
        self, mock_ivf, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        user_creds = {'user_id': 'test_user', 'token': 'test_token'}
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [{'item_id': 's1', 'embedding_vector': np.array([1.0])}]
        mock_last_played.return_value = None
        mock_ivf.return_value = []

        generate_sonic_fingerprint(num_neighbors=1, user_creds=user_creds)

        from unittest.mock import ANY

        mock_top_songs.assert_called_with(limit=ANY, user_creds=user_creds)
        mock_last_played.assert_called_with('s1', user_creds=user_creds)

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_uses_config_default_for_num_neighbors(
        self, mock_ivf, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [{'item_id': 's1', 'embedding_vector': np.array([1.0])}]
        mock_last_played.return_value = None
        mock_ivf.return_value = []

        result = generate_sonic_fingerprint()

        assert isinstance(result, list)


class TestWeightedAverageCalculation:
    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_calculates_weighted_average_correctly(
        self, mock_ivf, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': 's1'}, {'Id': 's2'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0, 0.0])},
            {'item_id': 's2', 'embedding_vector': np.array([0.0, 1.0])},
        ]
        recent_time = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        mock_last_played.return_value = recent_time
        mock_ivf.return_value = []

        generate_sonic_fingerprint(num_neighbors=5)

        assert mock_ivf.called
        call_args = mock_ivf.call_args
        query_vector = call_args[1]['query_vector']

        assert query_vector.shape == (2,)
        assert query_vector[0] > 0
        assert query_vector[1] > 0

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_skips_songs_without_embeddings(
        self, mock_ivf, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': 's1'}, {'Id': 's2'}]
        mock_get_tracks.return_value = [{'item_id': 's1', 'embedding_vector': np.array([1.0, 0.0])}]
        mock_last_played.return_value = None
        mock_ivf.return_value = []

        result = generate_sonic_fingerprint(num_neighbors=1)

        assert len(result) == 1
        assert result[0]['item_id'] == 's1'

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    def test_returns_empty_when_all_embeddings_invalid(
        self, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [{'item_id': 's1', 'embedding_vector': np.array([])}]
        mock_last_played.return_value = None

        result = generate_sonic_fingerprint(num_neighbors=1)

        assert result == []


class TestTimestampParsing:
    def test_parse_iso_format_with_z(self):
        timestamp_str = "2024-01-15T10:30:00Z"

        parsed = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

        assert parsed.year == 2024
        assert parsed.month == 1
        assert parsed.day == 15

    def test_parse_iso_format_with_microseconds(self):
        timestamp_str = "2024-01-15T10:30:00.123456Z"

        parsed = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

        assert parsed.microsecond == 123456

    def test_truncate_long_microseconds(self):
        timestamp_str = "2024-01-15T10:30:00.1234567890Z"

        if '.' in timestamp_str and timestamp_str.endswith('Z'):
            dot_index = timestamp_str.rfind('.')
            z_index = timestamp_str.rfind('Z')
            if z_index > dot_index and (z_index - dot_index - 1) > 6:
                timestamp_str = timestamp_str[: dot_index + 7] + 'Z'

        assert timestamp_str == "2024-01-15T10:30:00.123456Z"

        parsed = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        assert parsed.microsecond == 123456

    def test_calculate_days_since_played(self):
        now = datetime.now(timezone.utc)
        last_played = now - timedelta(days=15)

        days_since = (now - last_played).days

        assert days_since == 15

    def test_days_calculation_fractional(self):
        now = datetime.now(timezone.utc)
        last_played = now - timedelta(days=15, hours=18)

        days_since = (now - last_played).days

        assert days_since == 15

    def test_future_date_negative_days(self):
        now = datetime.now(timezone.utc)
        last_played = now + timedelta(days=5)

        days_since = (now - last_played).days

        assert days_since == -5


class TestIVFIntegration:
    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_ivf_called_with_correct_parameters(
        self, mock_ivf, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [{'item_id': 's1', 'embedding_vector': np.array([1.0])}]
        mock_last_played.return_value = None
        mock_ivf.return_value = []

        generate_sonic_fingerprint(num_neighbors=10)

        mock_ivf.assert_called_once()
        call_kwargs = mock_ivf.call_args[1]
        assert call_kwargs['n'] == 9
        assert call_kwargs['eliminate_duplicates'] is True

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_handles_ivf_exception(
        self, mock_ivf, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [{'item_id': 's1', 'embedding_vector': np.array([1.0])}]
        mock_last_played.return_value = None
        mock_ivf.side_effect = Exception("IVF error")

        result = generate_sonic_fingerprint(num_neighbors=5)

        assert result == []

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_combines_seed_and_ivf_results_correctly(
        self, mock_ivf, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': 's1'}, {'Id': 's2'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0])},
            {'item_id': 's2', 'embedding_vector': np.array([0.5])},
        ]
        mock_last_played.return_value = None
        mock_ivf.return_value = [
            {'item_id': 's3', 'distance': 0.1},
            {'item_id': 's4', 'distance': 0.2},
            {'item_id': 's5', 'distance': 0.3},
        ]

        result = generate_sonic_fingerprint(num_neighbors=5)

        assert len(result) == 5
        seed_ids = {result[0]['item_id'], result[1]['item_id']}
        assert seed_ids == {'s1', 's2'}
        vec_ids = {result[2]['item_id'], result[3]['item_id'], result[4]['item_id']}
        assert vec_ids == {'s3', 's4', 's5'}


class TestEdgeCases:
    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    def test_handles_tracks_with_partial_embeddings(
        self, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': 's1'}, {'Id': 's2'}, {'Id': 's3'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0])},
            {'item_id': 's2'},
            {'item_id': 's3', 'embedding_vector': np.array([0.5])},
        ]
        mock_last_played.return_value = None

        result = generate_sonic_fingerprint(num_neighbors=2)

        assert len(result) == 2
        item_ids = {r['item_id'] for r in result}
        assert item_ids == {'s1', 's3'}

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    def test_handles_empty_embedding_vectors(
        self, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [{'item_id': 's1', 'embedding_vector': np.array([])}]
        mock_last_played.return_value = None

        result = generate_sonic_fingerprint(num_neighbors=5)

        assert result == []

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    def test_handles_microseconds_truncation(
        self, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [{'item_id': 's1', 'embedding_vector': np.array([1.0])}]
        mock_last_played.return_value = "2024-01-15T10:30:00.12345678901234Z"

        result = generate_sonic_fingerprint(num_neighbors=1)

        assert len(result) == 1

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_respects_total_desired_size_limit(
        self, mock_ivf, mock_last_played, mock_get_tracks, mock_top_songs
    ):
        mock_top_songs.return_value = [{'Id': f's{i}'} for i in range(5)]
        mock_get_tracks.return_value = [
            {'item_id': f's{i}', 'embedding_vector': np.random.rand(10)} for i in range(5)
        ]
        mock_last_played.return_value = None
        mock_ivf.return_value = [{'item_id': f'v{i}', 'distance': 0.1 * i} for i in range(20)]

        result = generate_sonic_fingerprint(num_neighbors=10)

        assert len(result) == 10
