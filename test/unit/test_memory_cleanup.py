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
* Session recycle empties the old dict and frees old GPU sessions before allocating new ones
* The web process's idle heap trim holds off while requests keep arriving, fires
  once the process goes quiet, and stays off when its config window is 0
* The idle heap trim returns free heap to the OS without dropping a single loaded
  index: every startup cache is still populated and identical after it runs
* The heap release resolves its symbol once out of the running image - malloc_trim
  on Linux, malloc_zone_pressure_relief on macOS - with no subprocess; Windows gets
  no trim and never reaches ctypes, and an unresolvable symbol is logged and cached
* The in-flight counter is paired on teardown_request, not teardown_appcontext:
  a background app context and a request rejected before our before_request ran
  both leave a live request's count alone, and wiring it either of the two wrong
  ways is shown to steal that live request's count
"""

import gc
import sys
import weakref
from unittest.mock import MagicMock, patch

if "jwt" not in sys.modules:
    sys.modules["jwt"] = MagicMock()

_pg_conn = MagicMock()
_pg_conn.cursor.return_value.rowcount = 1
_pg_conn.cursor.return_value.__enter__.return_value = _pg_conn.cursor.return_value

import pytest
import numpy as np


@pytest.fixture(autouse=True, scope='module')
def _fake_pg_connect():
    patcher = patch("psycopg2.connect", return_value=_pg_conn)
    patcher.start()
    yield
    patcher.stop()


class _FakeSession:
    pass


class TestMusicnnSessionRecycleFreesGpuBeforeAlloc:
    def test_cleanup_musicnn_sessions_empties_dict_and_drops_every_reference(self):
        from tasks.analysis.song import cleanup_musicnn_sessions

        sessions = {'embedding': _FakeSession(), 'prediction': _FakeSession()}
        refs = [weakref.ref(s) for s in sessions.values()]

        cleanup_musicnn_sessions(sessions, context="recycle")
        gc.collect()

        assert sessions == {}
        assert all(r() is None for r in refs)

    def test_ensure_musicnn_sessions_releases_old_gpu_sessions_before_loading_new(self):
        from tasks.analysis import song
        from tasks.memory_utils import SessionRecycler

        old_sessions = {'embedding': _FakeSession(), 'prediction': _FakeSession()}
        old_ref = weakref.ref(old_sessions['embedding'])
        observed = {}

        def fake_load(model_paths):
            gc.collect()
            observed['old_alive_when_new_allocated'] = old_ref() is not None
            return {'embedding': _FakeSession(), 'prediction': _FakeSession()}

        recycler = SessionRecycler(recycle_interval=1)
        recycler.increment()

        with patch.object(song, 'load_musicnn_sessions', side_effect=fake_load), \
                patch.object(song, 'comprehensive_memory_cleanup', return_value={}):
            new_sessions = song.ensure_musicnn_sessions(
                old_sessions,
                {'embedding': 'e.onnx', 'prediction': 'p.onnx'},
                recycler,
                "Album",
            )

        assert observed['old_alive_when_new_allocated'] is False
        assert new_sessions is not old_sessions
        assert old_ref() is None


class TestAnalyzeTrackMemoryCleanup:
    @patch('tasks.analysis.song.robust_load_audio_with_fallback')
    @patch('tasks.analysis.song._estimate_key_scale')
    @patch('tasks.analysis.song._estimate_energy')
    @patch('tasks.analysis.song._estimate_tempo')
    @patch('tasks.analysis.song.librosa.feature.melspectrogram')
    @patch('tasks.analysis.song.create_onnx_session')
    @patch('tasks.analysis.song.cleanup_onnx_session')
    @patch('tasks.analysis.song.cleanup_cuda_memory')
    def test_cleanup_on_inference_error(
        self,
        mock_cuda_cleanup,
        mock_session_cleanup,
        mock_create_sess,
        mock_mel,
        mock_tempo,
        mock_energy,
        mock_key_scale,
        mock_load_audio,
    ):
        from tasks.analysis import analyze_track

        mock_load_audio.return_value = (np.random.randn(16000), 16000)
        mock_tempo.return_value = 120.0
        mock_energy.return_value = 0.5
        mock_key_scale.return_value = ('C', 'major')
        mock_mel.return_value = np.random.randn(96, 500)

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

    @patch('tasks.analysis.song.robust_load_audio_with_fallback')
    @patch('tasks.analysis.song._estimate_key_scale')
    @patch('tasks.analysis.song._estimate_energy')
    @patch('tasks.analysis.song._estimate_tempo')
    @patch('tasks.analysis.song.librosa.feature.melspectrogram')
    @patch('tasks.onnx_utils.ort')
    def test_no_cleanup_with_album_sessions(
        self, mock_ort, mock_mel, mock_tempo, mock_energy, mock_key_scale, mock_load_audio
    ):
        from tasks.analysis import analyze_track

        mock_load_audio.return_value = (np.random.randn(16000), 16000)
        mock_tempo.return_value = 120.0
        mock_energy.return_value = 0.5
        mock_key_scale.return_value = ('C', 'major')
        mock_mel.return_value = np.random.randn(96, 500)

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

        with patch('tasks.analysis.song.cleanup_onnx_session') as mock_cleanup:
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
    @patch('tasks.analysis.album.get_tracks_from_album')
    @patch('tasks.analysis.album.download_track')
    @patch('tasks.analysis.album.analyze_track')
    @patch('tasks.analysis.album.decode_audio_once',
           new=lambda path: (np.ones(16000, dtype=np.float32), 16000))
    @patch('tasks.analysis.helper.get_db')
    @patch('tasks.onnx_utils.ort')
    @patch('tasks.analysis.song.cleanup_onnx_session')
    @patch('tasks.memory_utils.cleanup_cuda_memory')
    @patch('database.save_task_status')
    @patch('database.get_task_info_from_db')
    @patch('tasks.analysis.album.taskqueue.current_task_id')
    def test_cleanup_on_database_error(
        self,
        mock_get_job,
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

    @patch('tasks.analysis.album.get_tracks_from_album')
    @patch('tasks.analysis.album.comprehensive_memory_cleanup')
    @patch('tasks.task_run.save_task_status')
    @patch('tasks.task_run.get_task_info_from_db')
    @patch('tasks.analysis.album.taskqueue.current_task_id')
    @patch('tasks.analysis.helper.get_db')
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

    @patch('tasks.analysis.album.get_tracks_from_album')
    @patch('tasks.analysis.album.download_track')
    @patch('tasks.analysis.album.analyze_track')
    @patch('tasks.analysis.album.decode_audio_once',
           new=lambda path: (np.ones(16000, dtype=np.float32), 16000))
    @patch('tasks.analysis.helper.get_db')
    @patch('tasks.analysis.song.create_onnx_session')
    @patch('tasks.analysis.song.cleanup_onnx_session')
    @patch('tasks.analysis.album.cleanup_cuda_memory')
    @patch('tasks.task_run.save_task_status')
    @patch('tasks.task_run.get_task_info_from_db')
    @patch('tasks.analysis.album.taskqueue.current_task_id')
    @patch('tasks.analysis.song.save_track_analysis_and_embedding')
    @patch('tasks.analysis.album.os.remove')
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
                patch('tasks.analysis.album._ah.run_lyrics_for_track', return_value=True):
            analyze_album_task("album_123", "Test Album", 5, None)

        assert mock_session_cleanup.call_count >= 2

        assert mock_cuda_cleanup.called


class TestIdleHeapTrim:
    def _armed(self, monkeypatch, window):
        import config
        from tasks import memory_utils

        monkeypatch.setattr(config, 'FLASK_IDLE_HEAP_TRIM_SECONDS', window, raising=False)
        from tasks.idle_unload import IdleUnloadTimer

        monkeypatch.setattr(
            memory_utils, '_IDLE_TRIM_TIMER', IdleUnloadTimer(), raising=False
        )
        monkeypatch.setattr(memory_utils, '_ACTIVE_REQUESTS', 0, raising=False)
        fired = []
        monkeypatch.setattr(
            memory_utils,
            'release_memory_to_os',
            lambda *args, **kwargs: fired.append(1) or True,
        )
        return memory_utils, fired

    def test_a_zero_window_disables_the_trim(self, monkeypatch):
        memory_utils, fired = self._armed(monkeypatch, 0)

        assert memory_utils.arm_idle_heap_trim() is False
        assert fired == []

    def test_a_burst_of_requests_keeps_re_arming_without_trimming(self, monkeypatch):
        memory_utils, fired = self._armed(monkeypatch, 2.0)

        class FakeTimer:
            def __init__(self):
                self.arms = 0
                self.on_expire = None

            def arm(self, duration, on_expire):
                self.arms += 1
                self.on_expire = on_expire

        monkeypatch.setattr(memory_utils, '_IDLE_TRIM_TIMER', FakeTimer(), raising=False)

        assert memory_utils.arm_idle_heap_trim() is True
        for _ in range(5):
            memory_utils.arm_idle_heap_trim()

        assert memory_utils._IDLE_TRIM_TIMER.arms == 6
        assert memory_utils._IDLE_TRIM_TIMER.on_expire is memory_utils._idle_heap_trim
        assert fired == []

    def test_the_heap_is_trimmed_once_the_process_goes_idle(self, monkeypatch):
        import time

        memory_utils, fired = self._armed(monkeypatch, 0.3)

        memory_utils.arm_idle_heap_trim()
        deadline = time.time() + 5
        while not fired and time.time() < deadline:
            time.sleep(0.05)

        assert fired == [1]

    def test_the_trim_skips_while_a_request_is_in_flight(self, monkeypatch):
        memory_utils, fired = self._armed(monkeypatch, 0.3)
        monkeypatch.setattr(memory_utils, '_ACTIVE_REQUESTS', 1, raising=False)

        memory_utils._idle_heap_trim()

        assert fired == []

    def test_request_bookkeeping_round_trips(self, monkeypatch):
        memory_utils, _ = self._armed(monkeypatch, 0)
        monkeypatch.setattr(memory_utils, '_ACTIVE_REQUESTS', 0, raising=False)

        memory_utils.note_request_started()
        assert memory_utils._ACTIVE_REQUESTS == 1

        memory_utils.note_request_finished()
        assert memory_utils._ACTIVE_REQUESTS == 0

    def test_the_trim_never_unloads_a_loaded_index(self, monkeypatch):
        import database
        import tasks.artist_gmm_manager as artist_mgr
        import tasks.clap_text_search as clap_search
        import tasks.ivf_manager as ivf_mgr
        import tasks.lyrics_manager as lyrics_mgr
        import tasks.sem_grove_manager as sem_grove
        from tasks import memory_utils

        sentinel_index = object()
        sentinel_ids = {0: 'song-1'}
        monkeypatch.setattr(ivf_mgr, 'ivf_index', sentinel_index, raising=False)
        monkeypatch.setattr(ivf_mgr, 'id_map', sentinel_ids, raising=False)
        monkeypatch.setattr(artist_mgr, 'artist_map', {0: 'Artist'}, raising=False)
        monkeypatch.setattr(
            artist_mgr, 'artist_gmm_params', {'Artist': {'weights': [1.0]}}, raising=False
        )
        monkeypatch.setattr(database, 'MAP_PROJECTION_CACHE', {'id_map': ['a']}, raising=False)
        clap_search._CLAP_INDEX_CACHE['index'] = sentinel_index
        clap_search._CLAP_INDEX_CACHE['loaded'] = True
        lyrics_mgr._LYRICS_INDEX_CACHE['index'] = sentinel_index
        lyrics_mgr._LYRICS_INDEX_CACHE['loaded'] = True
        sem_grove._SEM_GROVE_CACHE['loaded'] = True
        monkeypatch.setattr(memory_utils, '_ACTIVE_REQUESTS', 0, raising=False)

        released = []
        real_release = memory_utils.release_memory_to_os
        monkeypatch.setattr(
            memory_utils,
            'release_memory_to_os',
            lambda *args, **kwargs: released.append(1) or real_release(*args, **kwargs),
            raising=False,
        )

        memory_utils._idle_heap_trim()

        assert released == [1]
        assert ivf_mgr.ivf_index is sentinel_index
        assert ivf_mgr.id_map == sentinel_ids
        assert artist_mgr.artist_map == {0: 'Artist'}
        assert artist_mgr.artist_gmm_params == {'Artist': {'weights': [1.0]}}
        assert database.MAP_PROJECTION_CACHE == {'id_map': ['a']}
        assert clap_search._CLAP_INDEX_CACHE['index'] is sentinel_index
        assert clap_search._CLAP_INDEX_CACHE['loaded'] is True
        assert lyrics_mgr._LYRICS_INDEX_CACHE['index'] is sentinel_index
        assert lyrics_mgr._LYRICS_INDEX_CACHE['loaded'] is True
        assert sem_grove._SEM_GROVE_CACHE['loaded'] is True

    def test_windows_gets_no_trim_and_never_reaches_ctypes(self, monkeypatch):
        import ctypes
        import platform

        from tasks import memory_utils

        monkeypatch.setattr(memory_utils, '_HEAP_TRIM', None, raising=False)
        monkeypatch.setattr(platform, 'system', lambda: 'Windows')

        def _explode(*args, **kwargs):
            raise AssertionError('ctypes must not be touched on Windows')

        monkeypatch.setattr(ctypes, 'CDLL', _explode)

        assert memory_utils._resolve_heap_trim() is False
        assert memory_utils.release_memory_to_os() is False
        assert memory_utils.arm_idle_heap_trim() is False

    def test_each_platform_asks_for_its_own_heap_release_symbol(self, monkeypatch):
        import ctypes
        import platform

        from tasks import memory_utils

        asked = []

        class _Image:
            def __getattr__(self, name):
                asked.append(name)
                return MagicMock()

        monkeypatch.setattr(ctypes, 'CDLL', lambda *a, **k: _Image())
        for system, symbol in (
            ('Linux', 'malloc_trim'),
            ('Darwin', 'malloc_zone_pressure_relief'),
        ):
            asked.clear()
            monkeypatch.setattr(memory_utils, '_HEAP_TRIM', None, raising=False)
            monkeypatch.setattr(platform, 'system', lambda s=system: s)

            assert memory_utils.release_memory_to_os() is True
            assert asked == [symbol]

    def test_the_symbol_is_resolved_once_and_needs_no_subprocess(self, monkeypatch):
        import subprocess

        from tasks import memory_utils

        monkeypatch.setattr(memory_utils, '_HEAP_TRIM', None, raising=False)
        spawned = []
        real_popen = subprocess.Popen

        class _Recorder(real_popen):
            def __init__(self, *args, **kwargs):
                spawned.append(args[0] if args else kwargs.get('args'))
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(subprocess, 'Popen', _Recorder)
        for _ in range(3):
            memory_utils.release_memory_to_os()

        assert spawned == []

    def test_an_unresolvable_symbol_is_logged_and_cached_as_unavailable(self, monkeypatch):
        import ctypes
        import platform

        from tasks import memory_utils

        monkeypatch.setattr(memory_utils, '_HEAP_TRIM', None, raising=False)
        monkeypatch.setattr(platform, 'system', lambda: 'Linux')
        builds = []

        class _Empty:
            def __getattr__(self, name):
                raise AttributeError(name)

        def _build(*args, **kwargs):
            builds.append(1)
            return _Empty()

        monkeypatch.setattr(ctypes, 'CDLL', _build)

        assert memory_utils.release_memory_to_os() is False
        assert memory_utils.release_memory_to_os() is False
        assert builds == [1]



class TestIdleTrimRequestWiring:
    def _wired_app(self, monkeypatch, decrement_on='request', guarded=True):
        from flask import Flask, g, jsonify, request

        from tasks import memory_utils

        monkeypatch.setattr(memory_utils, '_ACTIVE_REQUESTS', 0, raising=False)
        monkeypatch.setattr(memory_utils, 'arm_idle_heap_trim', lambda: True)

        app = Flask(__name__)

        def barrier():
            if request.path == '/blocked':
                return jsonify({'error': 'Setup required'}), 403

        app.before_request(barrier)

        @app.before_request
        def _start():
            g._heap_trim_counted = True
            memory_utils.note_request_started()

        def _end(exc=None):
            if guarded and not g.pop('_heap_trim_counted', False):
                return
            memory_utils.note_request_finished()

        if decrement_on == 'app_context':
            app.teardown_appcontext(_end)
        else:
            app.teardown_request(_end)

        @app.route('/ok')
        def _ok():
            return 'ok'

        @app.route('/blocked')
        def _blocked():
            return 'never reached'

        return app, memory_utils

    def test_a_served_request_returns_the_counter_to_zero(self, monkeypatch):
        app, memory_utils = self._wired_app(monkeypatch)

        assert app.test_client().get('/ok').status_code == 200
        assert memory_utils._ACTIVE_REQUESTS == 0

    def test_a_background_app_context_never_steals_an_in_flight_request(self, monkeypatch):
        app, memory_utils = self._wired_app(monkeypatch)
        memory_utils.note_request_started()

        with app.app_context():
            pass

        assert memory_utils._ACTIVE_REQUESTS == 1

    def test_a_request_rejected_before_our_hook_never_steals_an_in_flight_request(
        self, monkeypatch
    ):
        app, memory_utils = self._wired_app(monkeypatch)
        memory_utils.note_request_started()

        assert app.test_client().get('/blocked').status_code == 403
        assert memory_utils._ACTIVE_REQUESTS == 1

    def test_decrementing_on_the_app_context_would_steal_a_live_request(self, monkeypatch):
        app, memory_utils = self._wired_app(
            monkeypatch, decrement_on='app_context', guarded=False
        )
        memory_utils.note_request_started()

        with app.app_context():
            pass

        assert memory_utils._ACTIVE_REQUESTS == 0

    def test_an_unguarded_decrement_would_steal_a_barrier_rejected_request(self, monkeypatch):
        app, memory_utils = self._wired_app(monkeypatch, guarded=False)
        memory_utils.note_request_started()

        assert app.test_client().get('/blocked').status_code == 403
        assert memory_utils._ACTIVE_REQUESTS == 0

    def test_the_g_guard_alone_already_survives_a_background_app_context(self, monkeypatch):
        app, memory_utils = self._wired_app(monkeypatch, decrement_on='app_context')
        memory_utils.note_request_started()

        with app.app_context():
            pass

        assert memory_utils._ACTIVE_REQUESTS == 1
