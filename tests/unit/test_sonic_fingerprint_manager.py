"""Unit tests for tasks/sonic_fingerprint_manager.py

Tests sonic fingerprint generation with actual function calls:
- Weighted average calculation from top played songs
- Recency-based weighting (exponential decay)
- Playlist size limiting and deduplication
- Integration with media server and Voyager
"""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from tasks.sonic_fingerprint_manager import generate_sonic_fingerprint


class TestGenerateSonicFingerprint:
    """Test the main generate_sonic_fingerprint function"""

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_generates_fingerprint_with_recent_songs(self, mock_voyager, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test generating fingerprint from recently played songs"""
        # Mock top played songs
        mock_top_songs.return_value = [
            {'Id': 's1'},
            {'Id': 's2'},
            {'Id': 's3'}
        ]
        
        # Mock track embeddings
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0, 0.0, 0.0])},
            {'item_id': 's2', 'embedding_vector': np.array([0.0, 1.0, 0.0])},
            {'item_id': 's3', 'embedding_vector': np.array([0.0, 0.0, 1.0])}
        ]
        
        # Mock last played times (all recent)
        mock_last_played.return_value = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        
        # Mock Voyager results
        mock_voyager.return_value = [
            {'item_id': 's4', 'distance': 0.1},
            {'item_id': 's5', 'distance': 0.2}
        ]
        
        # Generate fingerprint
        result = generate_sonic_fingerprint(num_neighbors=5)
        
        # Should return seed songs + Voyager results
        assert len(result) == 5
        # Seed songs should come first with distance 0.0
        assert result[0]['item_id'] in ['s1', 's2', 's3']
        assert result[0]['distance'] == 0.0

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    def test_returns_empty_when_no_top_songs(self, mock_top_songs):
        """Test handling of empty top played songs"""
        mock_top_songs.return_value = []
        
        result = generate_sonic_fingerprint(num_neighbors=10)
        
        assert result == []

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    def test_returns_empty_when_no_embeddings(self, mock_get_tracks, mock_top_songs):
        """Test handling when no tracks have embeddings"""
        mock_top_songs.return_value = [{'Id': 's1'}, {'Id': 's2'}]
        mock_get_tracks.return_value = []
        
        result = generate_sonic_fingerprint(num_neighbors=10)
        
        assert result == []

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_weights_recent_songs_higher(self, mock_voyager, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test that recently played songs get higher weights"""
        mock_top_songs.return_value = [{'Id': 's1'}, {'Id': 's2'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0, 0.0])},
            {'item_id': 's2', 'embedding_vector': np.array([0.0, 1.0])}
        ]
        
        # s1 played 1 day ago, s2 played 60 days ago
        def last_played_side_effect(song_id, user_creds=None):
            if song_id == 's1':
                return (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            else:
                return (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        
        mock_last_played.side_effect = last_played_side_effect
        mock_voyager.return_value = []
        
        result = generate_sonic_fingerprint(num_neighbors=2)
        
        # Should successfully generate (verifies weighting logic runs)
        assert len(result) == 2

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    def test_truncates_when_seed_songs_exceed_desired_size(self, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test truncation when seed songs exceed desired playlist size"""
        mock_top_songs.return_value = [{'Id': f's{i}'} for i in range(10)]
        mock_get_tracks.return_value = [
            {'item_id': f's{i}', 'embedding_vector': np.random.rand(128)} for i in range(10)
        ]
        mock_last_played.return_value = None
        
        result = generate_sonic_fingerprint(num_neighbors=5)
        
        # Should truncate to 5
        assert len(result) == 5

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_deduplicates_voyager_results(self, mock_voyager, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test that duplicate songs from Voyager are filtered out"""
        mock_top_songs.return_value = [{'Id': 's1'}, {'Id': 's2'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0, 0.0])},
            {'item_id': 's2', 'embedding_vector': np.array([0.0, 1.0])}
        ]
        mock_last_played.return_value = None
        
        # Voyager returns s1 again (duplicate) and s3 (new)
        mock_voyager.return_value = [
            {'item_id': 's1', 'distance': 0.05},  # Duplicate!
            {'item_id': 's3', 'distance': 0.1}
        ]
        
        result = generate_sonic_fingerprint(num_neighbors=5)
        
        # Should have s1, s2 (seeds) and s3 (new from Voyager), but not duplicate s1
        item_ids = [r['item_id'] for r in result]
        assert item_ids.count('s1') == 1  # Only once
        assert 's2' in item_ids
        assert 's3' in item_ids

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_seed_songs_have_zero_distance(self, mock_voyager, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test that seed songs are marked with distance 0.0"""
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0])}
        ]
        mock_last_played.return_value = None
        mock_voyager.return_value = [{'item_id': 's2', 'distance': 0.5}]
        
        result = generate_sonic_fingerprint(num_neighbors=2)
        
        # First result should be seed with distance 0.0
        assert result[0]['item_id'] == 's1'
        assert result[0]['distance'] == 0.0
        # Second should be from Voyager with real distance
        assert result[1]['item_id'] == 's2'
        assert result[1]['distance'] == 0.5

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_handles_invalid_last_played_date(self, mock_voyager, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test handling of unparseable last played dates"""
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0])}
        ]
        # Invalid date format
        mock_last_played.return_value = "invalid-date"
        mock_voyager.return_value = []
        
        result = generate_sonic_fingerprint(num_neighbors=1)
        
        # Should still work, using fallback weight (0.5)
        assert len(result) == 1

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_passes_user_credentials(self, mock_voyager, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test that user credentials are passed to media server functions"""
        user_creds = {'user_id': 'test_user', 'token': 'test_token'}
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0])}
        ]
        mock_last_played.return_value = None
        mock_voyager.return_value = []
        
        generate_sonic_fingerprint(num_neighbors=1, user_creds=user_creds)
        
        # Verify user_creds were passed
        from unittest.mock import ANY
        mock_top_songs.assert_called_with(limit=ANY, user_creds=user_creds)
        mock_last_played.assert_called_with('s1', user_creds=user_creds)

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_uses_config_default_for_num_neighbors(self, mock_voyager, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test that config default is used when num_neighbors not specified"""
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0])}
        ]
        mock_last_played.return_value = None
        mock_voyager.return_value = []
        
        # Call without num_neighbors (should use config default)
        result = generate_sonic_fingerprint()
        
        # Should complete without error
        assert isinstance(result, list)


class TestWeightedAverageCalculation:
    """Test weighted average calculation logic within generate_sonic_fingerprint"""

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_calculates_weighted_average_correctly(self, mock_voyager, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test that weighted average is calculated and passed to Voyager"""
        mock_top_songs.return_value = [{'Id': 's1'}, {'Id': 's2'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0, 0.0])},
            {'item_id': 's2', 'embedding_vector': np.array([0.0, 1.0])}
        ]
        # Equal recent times = equal weights
        recent_time = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        mock_last_played.return_value = recent_time
        mock_voyager.return_value = []
        
        # Request more neighbors than seed songs to ensure Voyager is called
        generate_sonic_fingerprint(num_neighbors=5)
        
        # Voyager should be called with averaged vector
        assert mock_voyager.called
        call_args = mock_voyager.call_args
        query_vector = call_args[1]['query_vector']
        
        # With equal weights, should be roughly [0.5, 0.5] (normalized)
        assert query_vector.shape == (2,)
        assert query_vector[0] > 0
        assert query_vector[1] > 0

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_skips_songs_without_embeddings(self, mock_voyager, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test that songs without embeddings are skipped"""
        mock_top_songs.return_value = [{'Id': 's1'}, {'Id': 's2'}]
        # s2 has no embedding
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0, 0.0])}
        ]
        mock_last_played.return_value = None
        mock_voyager.return_value = []
        
        result = generate_sonic_fingerprint(num_neighbors=1)
        
        # Should only include s1
        assert len(result) == 1
        assert result[0]['item_id'] == 's1'

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    def test_returns_empty_when_all_embeddings_invalid(self, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test handling when all embeddings are empty/invalid"""
        mock_top_songs.return_value = [{'Id': 's1'}]
        # Empty embedding vector
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([])}
        ]
        mock_last_played.return_value = None
        
        result = generate_sonic_fingerprint(num_neighbors=1)
        
        assert result == []


class TestTimestampParsing:
    """Test timestamp parsing logic for last played times"""

    def test_parse_iso_format_with_z(self):
        """Test parsing ISO format with Z suffix"""
        timestamp_str = "2024-01-15T10:30:00Z"
        
        parsed = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
        assert parsed.year == 2024
        assert parsed.month == 1
        assert parsed.day == 15

    def test_parse_iso_format_with_microseconds(self):
        """Test parsing ISO format with microseconds"""
        timestamp_str = "2024-01-15T10:30:00.123456Z"
        
        parsed = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
        assert parsed.microsecond == 123456

    def test_truncate_long_microseconds(self):
        """Test truncating microseconds longer than 6 digits"""
        # Simulate the truncation logic from the code
        timestamp_str = "2024-01-15T10:30:00.1234567890Z"
        
        # Truncation logic
        if '.' in timestamp_str and timestamp_str.endswith('Z'):
            dot_index = timestamp_str.rfind('.')
            z_index = timestamp_str.rfind('Z')
            if z_index > dot_index and (z_index - dot_index - 1) > 6:
                timestamp_str = timestamp_str[:dot_index+7] + 'Z'
        
        # Should be truncated to 6 digits
        assert timestamp_str == "2024-01-15T10:30:00.123456Z"
        
        parsed = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        assert parsed.microsecond == 123456

    def test_calculate_days_since_played(self):
        """Test calculating days since last played"""
        now = datetime.now(timezone.utc)
        last_played = now - timedelta(days=15)
        
        days_since = (now - last_played).days
        
        assert days_since == 15

    def test_days_calculation_fractional(self):
        """Test that fractional days are truncated to integers"""
        now = datetime.now(timezone.utc)
        last_played = now - timedelta(days=15, hours=18)
        
        days_since = (now - last_played).days
        
        # Should be 15, not 16 (truncated)
        assert days_since == 15

    def test_future_date_negative_days(self):
        """Test handling of future dates (negative days)"""
        now = datetime.now(timezone.utc)
        last_played = now + timedelta(days=5)
        
        days_since = (now - last_played).days
        
        # Should be negative (5 days in the future)
        assert days_since == -5


class TestVoyagerIntegration:
    """Test Voyager integration within sonic fingerprint"""

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_voyager_called_with_correct_parameters(self, mock_voyager, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test that Voyager is called with correct n and eliminate_duplicates"""
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0])}
        ]
        mock_last_played.return_value = None
        mock_voyager.return_value = []
        
        generate_sonic_fingerprint(num_neighbors=10)
        
        # With 1 seed song and 10 desired, should ask for 9 neighbors
        mock_voyager.assert_called_once()
        call_kwargs = mock_voyager.call_args[1]
        assert call_kwargs['n'] == 9
        assert call_kwargs['eliminate_duplicates'] is True

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_handles_voyager_exception(self, mock_voyager, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test handling of Voyager exceptions"""
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0])}
        ]
        mock_last_played.return_value = None
        mock_voyager.side_effect = Exception("Voyager error")
        
        result = generate_sonic_fingerprint(num_neighbors=5)
        
        # Should return empty list on error
        assert result == []

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_combines_seed_and_voyager_results_correctly(self, mock_voyager, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test correct combination of seed songs and Voyager results"""
        mock_top_songs.return_value = [{'Id': 's1'}, {'Id': 's2'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0])},
            {'item_id': 's2', 'embedding_vector': np.array([0.5])}
        ]
        mock_last_played.return_value = None
        mock_voyager.return_value = [
            {'item_id': 's3', 'distance': 0.1},
            {'item_id': 's4', 'distance': 0.2},
            {'item_id': 's5', 'distance': 0.3}
        ]
        
        result = generate_sonic_fingerprint(num_neighbors=5)
        
        # Should have 2 seeds + 3 from Voyager = 5 total
        assert len(result) == 5
        # First two should be seeds
        seed_ids = {result[0]['item_id'], result[1]['item_id']}
        assert seed_ids == {'s1', 's2'}
        # Last three should be from Voyager
        voyager_ids = {result[2]['item_id'], result[3]['item_id'], result[4]['item_id']}
        assert voyager_ids == {'s3', 's4', 's5'}


class TestEdgeCases:
    """Test edge cases and error handling"""

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    def test_handles_tracks_with_partial_embeddings(self, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test handling when some tracks have embeddings and some don't"""
        mock_top_songs.return_value = [{'Id': 's1'}, {'Id': 's2'}, {'Id': 's3'}]
        # Only s1 and s3 have valid embeddings
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0])},
            {'item_id': 's2'},  # No embedding
            {'item_id': 's3', 'embedding_vector': np.array([0.5])}
        ]
        mock_last_played.return_value = None
        
        result = generate_sonic_fingerprint(num_neighbors=2)
        
        # Should use s1 and s3 only
        assert len(result) == 2
        item_ids = {r['item_id'] for r in result}
        assert item_ids == {'s1', 's3'}

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    def test_handles_empty_embedding_vectors(self, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test handling of zero-size embedding vectors"""
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([])}  # Empty!
        ]
        mock_last_played.return_value = None
        
        result = generate_sonic_fingerprint(num_neighbors=5)
        
        # Should return empty
        assert result == []

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    def test_handles_microseconds_truncation(self, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test handling of timestamps with excessive microsecond precision"""
        mock_top_songs.return_value = [{'Id': 's1'}]
        mock_get_tracks.return_value = [
            {'item_id': 's1', 'embedding_vector': np.array([1.0])}
        ]
        # Timestamp with more than 6 decimal places
        mock_last_played.return_value = "2024-01-15T10:30:00.12345678901234Z"
        
        result = generate_sonic_fingerprint(num_neighbors=1)
        
        # Should handle truncation and parse successfully
        assert len(result) == 1

    @patch('tasks.sonic_fingerprint_manager.get_top_played_songs')
    @patch('app_helper.get_tracks_by_ids')
    @patch('tasks.sonic_fingerprint_manager.get_last_played_time')
    @patch('tasks.sonic_fingerprint_manager.find_nearest_neighbors_by_vector')
    def test_respects_total_desired_size_limit(self, mock_voyager, mock_last_played, mock_get_tracks, mock_top_songs):
        """Test that result does not exceed total_desired_size"""
        mock_top_songs.return_value = [{'Id': f's{i}'} for i in range(5)]
        mock_get_tracks.return_value = [
            {'item_id': f's{i}', 'embedding_vector': np.random.rand(10)} for i in range(5)
        ]
        mock_last_played.return_value = None
        # Voyager returns many results
        mock_voyager.return_value = [
            {'item_id': f'v{i}', 'distance': 0.1 * i} for i in range(20)
        ]
        
        result = generate_sonic_fingerprint(num_neighbors=10)
        
        # Should have exactly 10 results
        assert len(result) == 10



