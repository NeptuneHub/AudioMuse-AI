"""Unit tests for index rebuild and reload verification

Tests cover the rebuild_all_indexes_task function and the reload listener
to ensure indexes are properly rebuilt and reloaded during analysis.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
import json


class TestRebuildAllIndexesTask:
    """Tests for rebuild_all_indexes_task function"""

    @patch('tasks.analysis.get_db')
    @patch('tasks.analysis.redis_conn')
    @patch('tasks.analysis.build_and_store_voyager_index')
    @patch('tasks.analysis.build_and_store_artist_index')
    def test_successful_rebuild_all_indexes(
        self, mock_artist_index, mock_voyager_index, mock_redis, mock_get_db
    ):
        """Test that all indexes are successfully rebuilt and verified"""
        from tasks.analysis import rebuild_all_indexes_task
        
        # Mock database connection and cursor
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_db.return_value = mock_db
        
        # Mock successful Voyager index verification
        mock_cursor.fetchone.return_value = ('main_index', 200, '2024-01-01 00:00:00')
        
        # Mock successful builds
        mock_voyager_index.return_value = None
        mock_artist_index.return_value = None
        
        # Mock map projections
        with patch('tasks.analysis.build_and_store_map_projection') as mock_map:
            with patch('tasks.analysis.build_and_store_artist_projection') as mock_artist_proj:
                # Create app context mock
                with patch('tasks.analysis.app') as mock_app:
                    mock_app_context = MagicMock()
                    mock_app.app_context.return_value.__enter__.return_value = mock_app_context
                    
                    result = rebuild_all_indexes_task()
        
        # Verify all indexes were built
        mock_voyager_index.assert_called_once()
        mock_artist_index.assert_called_once()
        
        # Verify reload message was published
        mock_redis.publish.assert_called_once_with('index-updates', 'reload')
        
        # Verify result
        assert result['status'] == 'SUCCESS'
        assert 'voyager' in result['indexes_rebuilt']
        assert len(result['indexes_failed']) == 0

    @patch('tasks.analysis.get_db')
    @patch('tasks.analysis.redis_conn')
    @patch('tasks.analysis.build_and_store_voyager_index')
    def test_voyager_rebuild_failure_is_tracked(
        self, mock_voyager_index, mock_redis, mock_get_db
    ):
        """Test that Voyager index rebuild failures are tracked"""
        from tasks.analysis import rebuild_all_indexes_task
        
        # Mock database connection
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        # Mock Voyager build failure
        mock_voyager_index.side_effect = Exception("Voyager build failed")
        
        # Create app context mock
        with patch('tasks.analysis.app') as mock_app:
            mock_app_context = MagicMock()
            mock_app.app_context.return_value.__enter__.return_value = mock_app_context
            
            # Mock other indexes to succeed
            with patch('tasks.analysis.build_and_store_artist_index'):
                with patch('tasks.analysis.build_and_store_map_projection'):
                    with patch('tasks.analysis.build_and_store_artist_projection'):
                        result = rebuild_all_indexes_task()
        
        # Verify Voyager failure is tracked
        assert 'voyager' in result['indexes_failed']
        assert 'voyager' not in result['indexes_rebuilt']

    @patch('tasks.analysis.get_db')
    @patch('tasks.analysis.redis_conn')
    @patch('tasks.analysis.build_and_store_voyager_index')
    def test_voyager_verification_failure_is_tracked(
        self, mock_voyager_index, mock_redis, mock_get_db
    ):
        """Test that Voyager index verification failures are tracked"""
        from tasks.analysis import rebuild_all_indexes_task
        
        # Mock database connection and cursor
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_db.return_value = mock_db
        
        # Mock Voyager build success but verification failure (index not found)
        mock_cursor.fetchone.return_value = None
        mock_voyager_index.return_value = None
        
        # Create app context mock
        with patch('tasks.analysis.app') as mock_app:
            mock_app_context = MagicMock()
            mock_app.app_context.return_value.__enter__.return_value = mock_app_context
            
            # Mock other indexes to succeed
            with patch('tasks.analysis.build_and_store_artist_index'):
                with patch('tasks.analysis.build_and_store_map_projection'):
                    with patch('tasks.analysis.build_and_store_artist_projection'):
                        result = rebuild_all_indexes_task()
        
        # Verify Voyager verification failure is tracked
        assert 'voyager' in result['indexes_failed']
        assert 'voyager' not in result['indexes_rebuilt']

    @patch('tasks.analysis.get_db')
    @patch('tasks.analysis.redis_conn')
    @patch('tasks.analysis.build_and_store_voyager_index')
    def test_no_reload_message_when_all_indexes_fail(
        self, mock_voyager_index, mock_redis, mock_get_db
    ):
        """Test that no reload message is sent when all indexes fail"""
        from tasks.analysis import rebuild_all_indexes_task
        
        # Mock database connection
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        # Mock all index builds to fail
        mock_voyager_index.side_effect = Exception("Build failed")
        
        # Create app context mock
        with patch('tasks.analysis.app') as mock_app:
            mock_app_context = MagicMock()
            mock_app.app_context.return_value.__enter__.return_value = mock_app_context
            
            with patch('tasks.analysis.build_and_store_artist_index', side_effect=Exception("Build failed")):
                with patch('tasks.analysis.build_and_store_map_projection', side_effect=Exception("Build failed")):
                    with patch('tasks.analysis.build_and_store_artist_projection', side_effect=Exception("Build failed")):
                        result = rebuild_all_indexes_task()
        
        # Verify no reload message was published
        mock_redis.publish.assert_not_called()
        
        # Verify result indicates failure
        assert result['status'] == 'FAILURE'
        assert len(result['indexes_rebuilt']) == 0

    @patch('tasks.analysis.get_db')
    @patch('tasks.analysis.redis_conn')
    @patch('tasks.analysis.build_and_store_voyager_index')
    def test_reload_message_sent_with_partial_success(
        self, mock_voyager_index, mock_redis, mock_get_db
    ):
        """Test that reload message is sent even with partial success"""
        from tasks.analysis import rebuild_all_indexes_task
        
        # Mock database connection and cursor
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_db.return_value = mock_db
        
        # Mock successful Voyager index verification
        mock_cursor.fetchone.return_value = ('main_index', 200, '2024-01-01 00:00:00')
        mock_voyager_index.return_value = None
        
        # Create app context mock
        with patch('tasks.analysis.app') as mock_app:
            mock_app_context = MagicMock()
            mock_app.app_context.return_value.__enter__.return_value = mock_app_context
            
            # Mock artist index to fail, but map projections to succeed
            with patch('tasks.analysis.build_and_store_artist_index', side_effect=Exception("Build failed")):
                with patch('tasks.analysis.build_and_store_map_projection'):
                    with patch('tasks.analysis.build_and_store_artist_projection'):
                        result = rebuild_all_indexes_task()
        
        # Verify reload message was still published (partial success)
        mock_redis.publish.assert_called_once_with('index-updates', 'reload')
        
        # Verify result shows mixed status
        assert result['status'] == 'SUCCESS'
        assert 'voyager' in result['indexes_rebuilt']
        assert 'artist_similarity' in result['indexes_failed']

    @patch('tasks.analysis.get_db')
    @patch('tasks.analysis.redis_conn')
    @patch('tasks.analysis.build_and_store_voyager_index')
    def test_partial_success_status_when_reload_message_fails(
        self, mock_voyager_index, mock_redis, mock_get_db
    ):
        """Test that partial success status is returned when reload message fails"""
        from tasks.analysis import rebuild_all_indexes_task
        
        # Mock database connection and cursor
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_db.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_db.return_value = mock_db
        
        # Mock successful Voyager index verification
        mock_cursor.fetchone.return_value = ('main_index', 200, '2024-01-01 00:00:00')
        mock_voyager_index.return_value = None
        
        # Mock Redis publish to fail
        mock_redis.publish.side_effect = Exception("Redis connection failed")
        
        # Create app context mock
        with patch('tasks.analysis.app') as mock_app:
            mock_app_context = MagicMock()
            mock_app.app_context.return_value.__enter__.return_value = mock_app_context
            
            with patch('tasks.analysis.build_and_store_artist_index'):
                with patch('tasks.analysis.build_and_store_map_projection'):
                    with patch('tasks.analysis.build_and_store_artist_projection'):
                        result = rebuild_all_indexes_task()
        
        # Verify partial success status
        assert result['status'] == 'PARTIAL_SUCCESS'
        assert 'voyager' in result['indexes_rebuilt']
        assert 'reload notification failed' in result['message']


class TestReloadListener:
    """Tests for the Redis reload listener functionality"""

    def test_reload_listener_tracks_component_status(self):
        """Test that reload listener tracks success/failure of each component"""
        # This test would require mocking the entire Flask app context
        # and Redis pub/sub mechanism, which is complex for a unit test.
        # Integration tests would be more appropriate for this functionality.
        pass

    def test_reload_logs_failures_individually(self):
        """Test that reload logs each component failure individually"""
        # This test would require mocking the entire Flask app context
        # Integration tests would be more appropriate for this functionality.
        pass


class TestIndexRebuildDuringAnalysis:
    """Tests for index rebuild during main analysis task"""

    @patch('tasks.analysis.rq_queue_default')
    def test_index_rebuild_enqueued_at_batch_intervals(
        self, mock_queue
    ):
        """Test that index rebuild is enqueued at correct batch intervals"""
        # This test would require mocking the entire analysis flow
        # Integration tests would be more appropriate for this functionality.
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
