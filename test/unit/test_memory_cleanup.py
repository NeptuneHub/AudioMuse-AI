# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Model and CUDA memory cleanup around track and album analysis.

Verifies that analyze_track and analyze_album_task release ONNX sessions and
CUDA memory on the success, inference-error and database-error paths.

Main Features:
* ONNX sessions and CUDA memory are freed when inference raises
* Externally supplied album sessions are not cleaned up by the callee
* analyze_album_task runs comprehensive cleanup and CLAP unload in finally
* Database failure re-raises while still tearing down loaded models
"""

import sys
from unittest.mock import MagicMock, patch

if "jwt" not in sys.modules:
    sys.modules["jwt"] = MagicMock()

_pg_connect_patcher = patch("psycopg2.connect", return_value=MagicMock())
_pg_connect_patcher.start()

import pytest
import numpy as np


class TestAnalyzeTrackMemoryCleanup:
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    @patch('tasks.analysis_helper.librosa')
    @patch('tasks.analysis.create_onnx_session')
    @patch('tasks.analysis.cleanup_onnx_session')
    @patch('tasks.analysis.cleanup_cuda_memory')
    def test_cleanup_on_inference_error(
        self,
        mock_cuda_cleanup,
        mock_session_cleanup,
        mock_create_sess,
        mock_librosa,
        mock_load_audio,
    ):
        from tasks.analysis import analyze_track

        mock_load_audio.return_value = (np.random.randn(16000), 16000)
        mock_librosa.beat.beat_track.return_value = (120.0, None)
        mock_librosa.feature.rms.return_value = np.array([[0.5]])
        mock_librosa.feature.chroma_stft.return_value = np.random.randn(12, 100)
        mock_librosa.feature.melspectrogram.return_value = np.random.randn(96, 500)

        mock_embedding_sess = MagicMock()
        mock_prediction_sess = MagicMock()
        mock_create_sess.side_effect = [mock_embedding_sess, mock_prediction_sess]

        mock_embedding_sess.run.side_effect = RuntimeError("Model error")

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
                "sad": "/tmp/sad.onnx",
            },
        )

        assert result == (None, None)

        assert mock_session_cleanup.call_count >= 2
        assert mock_cuda_cleanup.called

    @patch('tasks.analysis.robust_load_audio_with_fallback')
    @patch('tasks.analysis.librosa')
    @patch('tasks.analysis_helper.ort')
    def test_no_cleanup_with_album_sessions(self, mock_ort, mock_librosa, mock_load_audio):
        from tasks.analysis import analyze_track

        mock_load_audio.return_value = (np.random.randn(16000), 16000)
        mock_librosa.beat.beat_track.return_value = (120.0, None)
        mock_librosa.feature.rms.return_value = np.array([[0.5]])
        mock_librosa.feature.chroma_stft.return_value = np.random.randn(12, 100)
        mock_librosa.feature.melspectrogram.return_value = np.random.randn(96, 500)

        mock_embedding_sess = MagicMock()
        mock_prediction_sess = MagicMock()
        mock_embedding_sess.run.return_value = [np.random.randn(10, 200)]
        mock_prediction_sess.run.return_value = [np.random.randn(10, 2)]

        onnx_sessions = {
            'embedding': mock_embedding_sess,
            'prediction': mock_prediction_sess,
            'danceable': MagicMock(),
            'aggressive': MagicMock(),
            'happy': MagicMock(),
            'party': MagicMock(),
            'relaxed': MagicMock(),
            'sad': MagicMock(),
        }

        for key in ['danceable', 'aggressive', 'happy', 'party', 'relaxed', 'sad']:
            onnx_sessions[key].run.return_value = [np.random.randn(10, 2)]

        with patch('tasks.analysis.cleanup_onnx_session') as mock_cleanup:
            analyze_track(
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
                    "sad": "/tmp/sad.onnx",
                },
                onnx_sessions=onnx_sessions,
            )

            assert mock_cleanup.call_count == 0


class TestAnalyzeAlbumMemoryCleanup:
    @patch('tasks.analysis.get_tracks_from_album')
    @patch('tasks.analysis.download_track')
    @patch('tasks.analysis.analyze_track')
    @patch('tasks.analysis_helper.get_db')
    @patch('tasks.analysis_helper.ort')
    @patch('tasks.analysis.cleanup_onnx_session')
    @patch('tasks.memory_utils.cleanup_cuda_memory')
    @patch('app_helper.save_task_status')
    @patch('app_helper.get_task_info_from_db')
    @patch('app_helper.redis_conn')
    @patch('tasks.analysis.get_current_job')
    def test_cleanup_on_database_error(
        self,
        mock_get_job,
        mock_redis,
        mock_get_task_info,
        mock_save_task,
        mock_cuda_cleanup,
        mock_session_cleanup,
        mock_ort,
        mock_get_db,
        mock_analyze,
        mock_download,
        mock_get_tracks,
    ):
        from tasks.analysis import analyze_album_task
        from psycopg2 import OperationalError

        mock_get_job.return_value = None
        mock_get_tracks.return_value = [
            {'Id': '1', 'Name': 'Track 1', 'AlbumArtist': 'Artist 1', 'ArtistId': 'artist1'}
        ]
        mock_download.return_value = "/tmp/track.mp3"

        mock_get_db.side_effect = OperationalError("Connection failed")

        mock_ort.get_available_providers.return_value = ['CPUExecutionProvider']
        mock_session = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session

        with pytest.raises(OperationalError):
            analyze_album_task("album_123", "Test Album", 5, None)

    @patch('tasks.analysis.get_tracks_from_album')
    @patch('tasks.analysis.comprehensive_memory_cleanup')
    @patch('app_helper.save_task_status')
    @patch('app_helper.get_task_info_from_db')
    @patch('tasks.analysis.get_current_job')
    @patch('app_helper.get_db')
    @patch('tasks.clap_analyzer.unload_clap_model')
    @patch('tasks.clap_analyzer.is_clap_model_loaded')
    def test_cleanup_all_models_in_finally(
        self,
        mock_clap_loaded,
        mock_clap_unload,
        mock_get_db,
        mock_get_job,
        mock_get_task_info,
        mock_save_task,
        mock_memory_cleanup,
        mock_get_tracks,
    ):
        from tasks.analysis import analyze_album_task

        mock_get_job.return_value = None
        mock_get_tracks.return_value = []
        mock_get_db.return_value = MagicMock()

        mock_clap_loaded.return_value = True

        analyze_album_task("album_123", "Empty Album", 5, None)

        assert mock_memory_cleanup.called
        assert mock_clap_unload.called

    @patch('tasks.analysis.get_tracks_from_album')
    @patch('tasks.analysis.download_track')
    @patch('tasks.analysis.analyze_track')
    @patch('app_helper.get_db')
    @patch('tasks.analysis_helper.create_onnx_session')
    @patch('tasks.analysis_helper.cleanup_onnx_session')
    @patch('tasks.analysis.cleanup_cuda_memory')
    @patch('app_helper.save_task_status')
    @patch('app_helper.get_task_info_from_db')
    @patch('tasks.analysis.get_current_job')
    @patch('app_helper.save_track_analysis_and_embedding')
    @patch('tasks.analysis.os.remove')
    def test_cleanup_onnx_sessions_on_success(
        self,
        mock_remove,
        mock_save_track,
        mock_get_job,
        mock_get_task_info,
        mock_save_task,
        mock_cuda_cleanup,
        mock_session_cleanup,
        mock_create_sess,
        mock_get_db,
        mock_analyze,
        mock_download,
        mock_get_tracks,
    ):
        from tasks.analysis import analyze_album_task

        mock_get_job.return_value = None
        mock_get_tracks.return_value = [
            {'Id': '1', 'Name': 'Track 1', 'AlbumArtist': 'Artist 1', 'ArtistId': 'artist1'}
        ]
        mock_download.return_value = "/tmp/track.mp3"

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = []
        mock_get_db.return_value = mock_conn

        mock_session = MagicMock()
        mock_create_sess.return_value = mock_session

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
                'sad': 0.2,
            },
            np.random.randn(200),
            np.random.randn(16000),
            16000,
        )

        with patch('tasks.clap_analyzer.is_clap_available', return_value=False), \
                patch('tasks.analysis._ah.run_lyrics_for_track', return_value=True):
            analyze_album_task("album_123", "Test Album", 5, None)

        assert mock_session_cleanup.call_count >= 2

        assert mock_cuda_cleanup.called
