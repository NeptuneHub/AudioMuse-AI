"""
Unit tests for memory cleanup in tasks/analysis.py
Tests the finally blocks that ensure cleanup on all code paths.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np


class TestAnalyzeTrackMemoryCleanup:
    """Test memory cleanup in analyze_track function."""
    
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    @patch('tasks.analysis.librosa')
    @patch('tasks.analysis.ort')
    @patch('tasks.analysis.cleanup_onnx_session')
    @patch('tasks.analysis.cleanup_cuda_memory')
    def test_cleanup_on_inference_error(
        self, mock_cuda_cleanup, mock_session_cleanup, 
        mock_ort, mock_librosa, mock_load_audio
    ):
        """Test that cleanup happens when inference fails."""
        from tasks.analysis import analyze_track
        
        # Setup mocks
        mock_load_audio.return_value = (np.random.randn(16000), 16000)
        mock_librosa.beat.beat_track.return_value = (120.0, None)
        mock_librosa.feature.rms.return_value = np.array([[0.5]])
        mock_librosa.feature.chroma_stft.return_value = np.random.randn(12, 100)
        mock_librosa.feature.melspectrogram.return_value = np.random.randn(96, 500)
        
        # Mock ONNX sessions
        mock_embedding_sess = MagicMock()
        mock_prediction_sess = MagicMock()
        mock_ort.InferenceSession.side_effect = [mock_embedding_sess, mock_prediction_sess]
        mock_ort.get_available_providers.return_value = ['CPUExecutionProvider']
        
        # Make inference fail with non-OOM error
        mock_embedding_sess.run.side_effect = RuntimeError("Model error")
        
        # Call analyze_track (should fail but cleanup should happen)
        result = analyze_track(
            "/tmp/test.mp3",
            ["happy", "sad"],
            {
                "embedding": "/tmp/embedding.onnx",
                "prediction": "/tmp/prediction.onnx",
                "danceable": "/tmp/danceable.onnx",
                "aggressive": "/tmp/aggressive.onnx",
                "happy": "/tmp/happy.onnx",
                "party": "/tmp/party.onnx",
                "relaxed": "/tmp/relaxed.onnx",
                "sad": "/tmp/sad.onnx"
            }
        )
        
        # Should return None due to error
        assert result == (None, None)
        
        # Verify cleanup was called in finally block
        assert mock_session_cleanup.call_count >= 2  # embedding and prediction
        assert mock_cuda_cleanup.called
    
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    @patch('tasks.analysis.librosa')
    @patch('tasks.analysis.ort')
    def test_no_cleanup_with_album_sessions(
        self, mock_ort, mock_librosa, mock_load_audio
    ):
        """Test that cleanup is skipped when using album-level sessions."""
        from tasks.analysis import analyze_track
        
        # Setup mocks
        mock_load_audio.return_value = (np.random.randn(16000), 16000)
        mock_librosa.beat.beat_track.return_value = (120.0, None)
        mock_librosa.feature.rms.return_value = np.array([[0.5]])
        mock_librosa.feature.chroma_stft.return_value = np.random.randn(12, 100)
        mock_librosa.feature.melspectrogram.return_value = np.random.randn(96, 500)
        
        # Pre-loaded sessions
        mock_embedding_sess = MagicMock()
        mock_prediction_sess = MagicMock()
        mock_embedding_sess.run.return_value = [np.random.randn(10, 200)]
        mock_prediction_sess.run.return_value = [np.random.randn(10, 2)]
        
        # Create onnx_sessions dict with all required models
        onnx_sessions = {
            'embedding': mock_embedding_sess,
            'prediction': mock_prediction_sess,
            'danceable': MagicMock(),
            'aggressive': MagicMock(),
            'happy': MagicMock(),
            'party': MagicMock(),
            'relaxed': MagicMock(),
            'sad': MagicMock()
        }
        
        # Configure secondary model mocks
        for key in ['danceable', 'aggressive', 'happy', 'party', 'relaxed', 'sad']:
            onnx_sessions[key].run.return_value = [np.random.randn(10, 2)]
        
        # Call with pre-loaded sessions
        with patch('tasks.analysis.cleanup_onnx_session') as mock_cleanup:
            result = analyze_track(
                "/tmp/test.mp3",
                ["happy", "sad"],
                {
                    "embedding": "/tmp/embedding.onnx",
                    "prediction": "/tmp/prediction.onnx",
                    "danceable": "/tmp/danceable.onnx",
                    "aggressive": "/tmp/aggressive.onnx",
                    "happy": "/tmp/happy.onnx",
                    "party": "/tmp/party.onnx",
                    "relaxed": "/tmp/relaxed.onnx",
                    "sad": "/tmp/sad.onnx"
                },
                onnx_sessions=onnx_sessions
            )
            
            # Cleanup should NOT be called for main sessions (album reuse)
            # It should only be called for secondary models if loaded per-song
            # Since we provided all sessions, cleanup should not happen
            assert mock_cleanup.call_count == 0


class TestAnalyzeAlbumMemoryCleanup:
    """Test memory cleanup in analyze_album_task function."""
    
    @patch('tasks.analysis.get_tracks_from_album')
    @patch('tasks.analysis.download_track')
    @patch('tasks.analysis.analyze_track')
    @patch('tasks.analysis.get_db')
    @patch('tasks.analysis.ort')
    @patch('tasks.analysis.cleanup_onnx_session')
    @patch('tasks.analysis.cleanup_cuda_memory')
    @patch('tasks.analysis.save_task_status')
    @patch('tasks.analysis.get_task_info_from_db')
    @patch('tasks.analysis.redis_conn')
    @patch('tasks.analysis.get_current_job')
    def test_cleanup_on_database_error(
        self, mock_get_job, mock_redis, mock_get_task_info, mock_save_task,
        mock_cuda_cleanup, mock_session_cleanup, mock_ort, mock_get_db,
        mock_analyze, mock_download, mock_get_tracks
    ):
        """Test that cleanup happens when database error occurs."""
        from tasks.analysis import analyze_album_task
        from psycopg2 import OperationalError
        
        # Setup mocks
        mock_get_job.return_value = None
        mock_get_tracks.return_value = [
            {'Id': '1', 'Name': 'Track 1', 'AlbumArtist': 'Artist 1', 'ArtistId': 'artist1'}
        ]
        mock_download.return_value = "/tmp/track.mp3"
        
        # Mock database to raise error
        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.side_effect = OperationalError("Connection failed")
        
        # Mock ONNX sessions
        mock_ort.get_available_providers.return_value = ['CPUExecutionProvider']
        mock_session = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        
        # Try to analyze album (should fail with database error)
        with pytest.raises(OperationalError):
            analyze_album_task("album_123", "Test Album", 5, None)
        
        # Verify cleanup was called even though error occurred
        # Note: cleanup will be called if sessions were loaded before error
        assert mock_cuda_cleanup.called
    
    @patch('tasks.analysis.get_tracks_from_album')
    @patch('tasks.analysis.cleanup_cuda_memory')
    @patch('tasks.analysis.save_task_status')
    @patch('tasks.analysis.get_task_info_from_db')
    @patch('tasks.analysis.get_current_job')
    @patch('tasks.analysis.get_db')
    @patch('tasks.analysis.unload_clap_model')
    @patch('tasks.analysis.is_clap_model_loaded')
    @patch('tasks.analysis.unload_mulan_model')
    @patch('tasks.analysis.is_mulan_model_loaded')
    def test_cleanup_all_models_in_finally(
        self, mock_mulan_loaded, mock_mulan_unload, mock_clap_loaded, 
        mock_clap_unload, mock_get_db, mock_get_job, mock_get_task_info,
        mock_save_task, mock_cuda_cleanup, mock_get_tracks
    ):
        """Test that all models are cleaned up in finally block."""
        from tasks.analysis import analyze_album_task
        
        # Setup mocks
        mock_get_job.return_value = None
        mock_get_tracks.return_value = []  # Empty track list
        mock_get_db.return_value = MagicMock()
        
        # Simulate models being loaded
        mock_clap_loaded.return_value = True
        mock_mulan_loaded.return_value = True
        
        # Call function (should complete successfully)
        result = analyze_album_task("album_123", "Empty Album", 5, None)
        
        # Verify all cleanup functions were called
        assert mock_cuda_cleanup.called
        assert mock_clap_unload.called
        assert mock_mulan_unload.called
    
    @patch('tasks.analysis.get_tracks_from_album')
    @patch('tasks.analysis.download_track')
    @patch('tasks.analysis.analyze_track')
    @patch('tasks.analysis.get_db')
    @patch('tasks.analysis.ort')
    @patch('tasks.analysis.cleanup_onnx_session')
    @patch('tasks.analysis.cleanup_cuda_memory')
    @patch('tasks.analysis.save_task_status')
    @patch('tasks.analysis.get_task_info_from_db')
    @patch('tasks.analysis.get_current_job')
    @patch('tasks.analysis.save_track_analysis_and_embedding')
    @patch('tasks.analysis.os.remove')
    def test_cleanup_onnx_sessions_on_success(
        self, mock_remove, mock_save_track, mock_get_job, mock_get_task_info,
        mock_save_task, mock_cuda_cleanup, mock_session_cleanup, mock_ort,
        mock_get_db, mock_analyze, mock_download, mock_get_tracks
    ):
        """Test that ONNX sessions are cleaned up after successful album analysis."""
        from tasks.analysis import analyze_album_task
        
        # Setup mocks
        mock_get_job.return_value = None
        mock_get_tracks.return_value = [
            {'Id': '1', 'Name': 'Track 1', 'AlbumArtist': 'Artist 1', 'ArtistId': 'artist1'}
        ]
        mock_download.return_value = "/tmp/track.mp3"
        
        # Mock database
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = []  # No existing tracks
        mock_get_db.return_value = mock_conn
        
        # Mock ONNX sessions
        mock_ort.get_available_providers.return_value = ['CPUExecutionProvider']
        mock_session = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        
        # Mock analyze_track to return results
        mock_analyze.return_value = (
            {
                'tempo': 120.0,
                'key': 'C',
                'scale': 'major',
                'moods': {'happy': 0.8},
                'energy': 0.7,
                'danceable': 0.6,
                'aggressive': 0.3,
                'happy': 0.8,
                'party': 0.5,
                'relaxed': 0.4,
                'sad': 0.2
            },
            np.random.randn(200)
        )
        
        # Call function
        with patch('tasks.analysis.is_clap_available', return_value=False):
            with patch('tasks.analysis.MULAN_ENABLED', False):
                result = analyze_album_task("album_123", "Test Album", 5, None)
        
        # Verify session cleanup was called for all loaded sessions
        # Should be called 8 times (embedding, prediction, and 6 secondary models)
        assert mock_session_cleanup.call_count >= 8
        
        # Verify CUDA cleanup was called
        assert mock_cuda_cleanup.called
