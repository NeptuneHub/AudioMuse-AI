# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the setup wizard naming preview that runs on a worker.

The web process only enqueues a naming_preview job and reads its row back, and the
worker clusters a sample of the library once and names every playlist. These tests
pin that it stays a read-only side job: registered like the dry-run planner, never
blocking a real task, never erasing the last task recap nor adding a history entry,
never sampling more than 10,000 songs, only issuing SELECT statements for its data,
and naming through the same helper and duplicate-suffix rule as a real run.

Main Features:
* The job type is registered in every queue registry and is non-blocking.
* Starting refuses while a preview is live, drops only old preview rows and enqueues
  with clear_finished=False, max_attempts=0 on the high queue.
* The row is mapped to the wizard states, with a generic message for any failure
  that is not an expected preview problem.
* A finishing preview neither collapses the table nor records task history.
* Clustering is one K-Means pass at a fixed K followed by the real filters, and
  naming goes through _try_ai_name_playlist with the selected style.
"""

from unittest.mock import MagicMock

import pytest

import config
import database
import task_types
import taskqueue
from tasks import naming_preview, recovery

AI_CONFIG = {
    'provider': 'OLLAMA',
    'ollama_url': 'http://localhost:11434/api/generate',
    'ollama_model': 'test-model',
    'openai_url': '', 'openai_model': '', 'openai_key': '',
    'gemini_key': '', 'gemini_model': '',
    'mistral_key': '', 'mistral_model': '',
}


class _Cursor:
    def __init__(self, log, rows=None):
        self.log = log
        self.rows = list(rows or [])
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.log.append((' '.join(sql.split()), params))

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchall(self):
        return self.rows

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Connection:
    def __init__(self, rows=None):
        self.log = []
        self.rows = rows
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, cursor_factory=None):
        return _Cursor(self.log, self.rows)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class TestRegistration:
    def test_the_task_function_is_allowed(self):
        assert naming_preview.PREVIEW_TASK_FUNC in taskqueue.ALLOWED_FUNCS
        assert taskqueue.resolve_func(naming_preview.PREVIEW_TASK_FUNC) is (
            naming_preview.run_naming_preview_task
        )

    def test_the_task_type_is_a_non_blocking_self_managed_side_job(self):
        entry = next(e for e in task_types.ALL if e.name == naming_preview.PREVIEW_TASK_TYPE)
        assert entry.queue == 'high' and entry.restarts == 0
        assert not entry.holds_main_index and not entry.blocks_starts
        assert naming_preview.PREVIEW_TASK_TYPE in database.NON_BLOCKING_TASK_TYPES
        assert naming_preview.PREVIEW_TASK_TYPE not in config.QUEUE_BLOCKING_TASK_TYPES
        assert naming_preview.PREVIEW_TASK_TYPE not in task_types.NON_WORKER_TASK_TYPES

    def test_the_recovery_table_has_a_stance(self):
        assert naming_preview.PREVIEW_TASK_TYPE in recovery.RECOVERY

    def test_it_is_hidden_from_the_dashboard_and_the_cancel_history(self, monkeypatch):
        import app_helper
        from pathlib import Path

        app_source = (Path(app_helper.__file__).parent / 'app.py').read_text(encoding='utf-8')
        assert 'from tasks.naming_preview import PREVIEW_TASK_TYPE' in app_source
        assert 'MIGRATION_PLANNER_TASK_TYPE, PREVIEW_TASK_TYPE)' in app_source
        recorded = []
        monkeypatch.setattr(app_helper, '_record_one_cancellation', lambda *a: recorded.append(a))
        app_helper._record_cancel_history(
            [{'task_id': 'p', 'task_type': naming_preview.PREVIEW_TASK_TYPE}], set(), 0, 'x'
        )
        assert recorded == []


class TestTheRecapIsNeverTouched:
    def test_a_finished_preview_does_not_collapse_the_table(self):
        db = _Connection()
        deleted = database.collapse_finished_task(
            db, 'p1', naming_preview.PREVIEW_TASK_TYPE, None, config.TASK_STATUS_SUCCESS
        )
        assert deleted == 0 and db.log == []

    def test_a_finished_preview_records_no_task_history(self, monkeypatch):
        recorded = []
        monkeypatch.setattr(database, 'record_task_history', lambda *a, **k: recorded.append(a))
        database._maybe_record_task_history(
            _Connection(), 'p1', naming_preview.PREVIEW_TASK_TYPE, config.TASK_STATUS_SUCCESS,
            None, {}, 0.0,
        )
        assert recorded == []

    def test_a_real_task_still_collapses_and_records_history(self, monkeypatch):
        recorded = []
        monkeypatch.setattr(database, 'record_task_history', lambda *a, **k: recorded.append(a))
        db = _Connection()
        database._maybe_record_task_history(
            db, 'm1', 'main_clustering', config.TASK_STATUS_SUCCESS, None, {}, 0.0,
        )
        database.collapse_finished_task(
            db, 'm1', 'main_clustering', None, config.TASK_STATUS_SUCCESS
        )
        assert recorded
        assert any(sql.startswith('DELETE FROM task_status WHERE task_id <> %s') for sql, _p in db.log)


class TestEnqueueFlag:
    def _enqueue(self, monkeypatch, **kwargs):
        calls = []
        fake_sql = MagicMock()
        fake_sql.take_start_lock.side_effect = lambda cur: calls.append('lock')
        fake_sql.clear_task_status.side_effect = lambda cur: calls.append('clear')
        fake_sql.insert_job.side_effect = lambda cur, **kw: calls.append('insert') or True
        fake_sql.notify_job.side_effect = lambda cur, queue: calls.append('notify')
        monkeypatch.setattr(taskqueue, '_with_cursor', lambda action, conn: action(fake_sql, MagicMock()))
        taskqueue.enqueue(
            naming_preview.PREVIEW_TASK_FUNC, task_id='p1',
            task_type=naming_preview.PREVIEW_TASK_TYPE, **kwargs
        )
        return calls

    def test_a_root_enqueue_still_clears_finished_rows_by_default(self, monkeypatch):
        assert self._enqueue(monkeypatch) == ['lock', 'clear', 'insert', 'notify']

    def test_clear_finished_false_skips_only_the_clear(self, monkeypatch):
        assert self._enqueue(monkeypatch, clear_finished=False) == ['lock', 'insert', 'notify']


class TestStartPreview:
    def test_a_live_preview_refuses_a_second_one(self, monkeypatch):
        db = _Connection(rows=[(1,)])
        enqueued = []
        monkeypatch.setattr(database, 'get_db', lambda: db)
        monkeypatch.setattr(taskqueue, 'enqueue', lambda *a, **k: enqueued.append(k))
        assert naming_preview.start_preview('title', 'Name it.') is None
        assert enqueued == [] and db.rollbacks == 1 and db.commits == 0

    def test_a_start_drops_old_preview_rows_and_enqueues_a_side_job(self, monkeypatch):
        db = _Connection()
        enqueued = []
        monkeypatch.setattr(database, 'get_db', lambda: db)
        monkeypatch.setattr(taskqueue, 'enqueue', lambda func, **k: enqueued.append((func, k)))
        task_id = naming_preview.start_preview('title', 'Name it.')
        assert task_id and db.commits == 1
        delete = [(sql, p) for sql, p in db.log if sql.startswith('DELETE')]
        assert delete and delete[0][1][0] == naming_preview.PREVIEW_TASK_TYPE
        assert set(delete[0][1][1]) == set(config.TASK_STATUS_TERMINAL)
        func, kwargs = enqueued[0]
        assert func == naming_preview.PREVIEW_TASK_FUNC
        assert kwargs['task_id'] == task_id
        assert kwargs['task_type'] == naming_preview.PREVIEW_TASK_TYPE
        assert kwargs['kwargs'] == {'mode': 'title', 'instructions': 'Name it.'}
        assert kwargs['queue'] == taskqueue.QUEUE_HIGH
        assert kwargs['max_attempts'] == 0
        assert kwargs['clear_finished'] is False
        assert kwargs['conn'] is db

    def test_no_secret_is_put_in_the_job_arguments(self, monkeypatch):
        db = _Connection()
        enqueued = []
        monkeypatch.setattr(database, 'get_db', lambda: db)
        monkeypatch.setattr(taskqueue, 'enqueue', lambda func, **k: enqueued.append(k))
        naming_preview.start_preview('concept', None)
        assert set(enqueued[0]['kwargs']) == {'mode', 'instructions'}


class TestPreviewStatus:
    def _status(self, monkeypatch, row):
        db = _Connection(rows=[row] if row else [])
        monkeypatch.setattr(database, 'get_db', lambda: db)
        return naming_preview.preview_status()

    def test_no_row_is_idle(self, monkeypatch):
        assert self._status(monkeypatch, None)['status'] == 'idle'

    def test_a_queued_row_is_waiting_for_a_worker(self, monkeypatch):
        state = self._status(monkeypatch, {'task_id': 't', 'status': config.TASK_STATUS_NEW,
                                           'details': {'message': 'x'}})
        assert state['status'] == 'running'
        assert state['message'] == naming_preview.PREVIEW_WAITING_MESSAGE

    def test_a_running_row_shows_progress_and_titles(self, monkeypatch):
        details = {'message': 'Naming 2 playlists...', 'mode': 'title', 'total': 2,
                   'song_count': 500, 'titles': [{'title': 'A'}]}
        state = self._status(monkeypatch, {'task_id': 't', 'status': config.TASK_STATUS_RUNNING,
                                           'details': details})
        assert state['status'] == 'running'
        assert state['message'] == 'Naming 2 playlists...'
        assert state['done'] == 1 and state['total'] == 2 and state['song_count'] == 500

    def test_a_finished_row_is_done_with_its_own_message(self, monkeypatch):
        details = {'message': 'queue summary', 'preview_message': 'Preview complete: 1 playlist titles.',
                   'titles': [{'title': 'A'}], 'total': 1}
        state = self._status(monkeypatch, {'task_id': 't', 'status': config.TASK_STATUS_SUCCESS,
                                           'details': details})
        assert state['status'] == 'done'
        assert state['message'] == 'Preview complete: 1 playlist titles.'

    def test_an_expected_problem_is_shown(self, monkeypatch):
        details = {'message': 'Traceback secret', 'preview_error': 'Not enough analyzed songs.'}
        state = self._status(monkeypatch, {'task_id': 't', 'status': config.TASK_STATUS_FAIL,
                                           'details': details})
        assert state['status'] == 'failed' and state['message'] == 'Not enough analyzed songs.'

    def test_any_other_failure_is_generic(self, monkeypatch):
        state = self._status(monkeypatch, {'task_id': 't', 'status': config.TASK_STATUS_FAIL,
                                           'details': '{"message": "KeyError secret"}'})
        assert state['message'] == naming_preview.PREVIEW_FAILED_MESSAGE

    def test_a_cancelled_preview_says_so(self, monkeypatch):
        state = self._status(monkeypatch, {'task_id': 't', 'status': config.TASK_STATUS_REVOKED,
                                           'details': {}})
        assert state['status'] == 'failed'
        assert state['message'] == naming_preview.PREVIEW_CANCELLED_MESSAGE

    def test_malformed_titles_become_an_empty_list(self, monkeypatch):
        state = self._status(monkeypatch, {'task_id': 't', 'status': config.TASK_STATUS_RUNNING,
                                           'details': {'titles': 'nope'}})
        assert state['titles'] == [] and state['done'] == 0


class TestWorkerTask:
    def test_the_task_reports_and_returns_the_titles(self, monkeypatch):
        reports = []
        monkeypatch.setattr(taskqueue, 'current_task_id', lambda: 'job-1')
        monkeypatch.setattr(naming_preview, '_report', lambda task_id, state, **f: (state.update(f), reports.append(task_id)))
        monkeypatch.setattr(naming_preview, '_cluster_sample', lambda task_id, state: {'named_playlists': {}})
        monkeypatch.setattr(naming_preview, 'worker_ai_config', lambda: AI_CONFIG)
        received = {}

        def fake_name(task_id, state, result, mode, instructions, ai_config):
            received.update(mode=mode, instructions=instructions, provider=ai_config['provider'])
            state['titles'].append({'title': 'A'})
            state['total'] = 1

        monkeypatch.setattr(naming_preview, '_name_playlists', fake_name)
        summary = naming_preview.run_naming_preview_task(mode=' TITLE ', instructions='Edited.')
        assert received == {'mode': 'title', 'instructions': 'Edited.', 'provider': 'OLLAMA'}
        assert summary['titles'] == [{'title': 'A'}]
        assert summary['preview_message'] == 'Preview complete: 1 playlist titles.'
        assert reports == ['job-1']

    def test_the_worker_uses_its_own_server_side_credentials(self, monkeypatch):
        monkeypatch.setattr(config, 'AI_MODEL_PROVIDER', 'gemini')
        monkeypatch.setattr(config, 'GEMINI_API_KEY', 'server-key')
        ai_config = naming_preview.worker_ai_config()
        assert ai_config['provider'] == 'GEMINI' and ai_config['gemini_key'] == 'server-key'

    def test_report_writes_a_running_row(self, monkeypatch):
        saved = []
        monkeypatch.setattr(
            database, 'save_task_status',
            lambda task_id, task_type, status, progress=0, details=None: saved.append(
                (task_id, task_type, status, progress, details)
            ),
        )
        state = {'mode': 'title', 'message': '', 'song_count': 0, 'done': 0, 'total': 4, 'titles': [1, 2]}
        naming_preview._report('job-1', state, message='Naming...')
        task_id, task_type, status, progress, details = saved[0]
        assert (task_id, task_type, status) == ('job-1', naming_preview.PREVIEW_TASK_TYPE, config.TASK_STATUS_RUNNING)
        assert progress == 52 and details['message'] == 'Naming...'

    def test_report_without_a_task_id_touches_nothing(self, monkeypatch):
        monkeypatch.setattr(database, 'save_task_status', lambda *a, **k: pytest.fail('no write'))
        state = {'total': 0, 'titles': []}
        naming_preview._report(None, state, message='x')
        assert state['message'] == 'x'

    def test_fail_records_the_problem_then_raises(self, monkeypatch):
        reported = {}
        monkeypatch.setattr(naming_preview, '_report', lambda task_id, state, **f: reported.update(f))
        with pytest.raises(naming_preview.PreviewUnavailable):
            naming_preview._fail('job-1', {}, 'Not enough analyzed songs.')
        assert reported['preview_error'] == 'Not enough analyzed songs.'


class TestClusterCount:
    @pytest.fixture(autouse=True)
    def limits(self, monkeypatch):
        monkeypatch.setattr(config, 'CLUSTERING_MAX_PLAYLIST_SONGS', 200)
        monkeypatch.setattr(config, 'MIN_PLAYLIST_SIZE_FOR_TOP_N', 20)
        monkeypatch.setattr(config, 'NUM_CLUSTERS_MAX', 100)

    def test_a_large_sample_uses_the_configured_maximum(self, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 0)
        assert naming_preview.preview_cluster_count(10000) == 100

    def test_a_small_sample_is_bounded_by_playlist_size(self, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 0)
        assert naming_preview.preview_cluster_count(400) == 10

    def test_k_never_reaches_the_song_count(self, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 0)
        assert naming_preview.preview_cluster_count(3) == 2


class TestSampling:
    def test_the_sample_is_capped_and_read_only(self, monkeypatch):
        from tasks import clustering, clustering_helper

        db = _Connection(rows=[{'item_id': 'x', 'mood_vector': 'rock:1'}])
        monkeypatch.setattr(database, 'get_db', lambda: db)
        monkeypatch.setattr(clustering, '_prepare_genre_map', lambda rows: {'rock': rows})
        monkeypatch.setattr(clustering, '_calculate_target_songs_per_genre', lambda *a: 50)
        monkeypatch.setattr(
            clustering_helper,
            '_get_stratified_song_subset',
            lambda genre_map, target: [{'item_id': str(i)} for i in range(12000)],
        )
        item_ids, genre_map = naming_preview._sample_item_ids()
        assert len(item_ids) == naming_preview.PREVIEW_MAX_SONGS == 10000
        assert len(set(item_ids)) == 10000
        assert genre_map == {'rock': [{'item_id': 'x', 'mood_vector': 'rock:1'}]}
        assert db.log and all(sql.upper().startswith('SELECT') for sql, _p in db.log)


class TestClusterSample:
    @pytest.fixture
    def patched(self, monkeypatch):
        from tasks import clustering_helper, clustering_postprocessing

        calls = {}
        monkeypatch.setattr(config, 'MIN_PLAYLIST_SIZE_FOR_TOP_N', 2)
        monkeypatch.setattr(naming_preview, '_report', lambda task_id, state, **f: state.update(f))
        monkeypatch.setattr(
            naming_preview,
            '_sample_item_ids',
            lambda: (['a', 'b', 'c', 'd'], {'rock': [1, 2], '__other__': [3]}),
        )
        monkeypatch.setattr(naming_preview, 'preview_cluster_count', lambda count: 3)

        def fake_iteration(**kwargs):
            calls['iteration'] = kwargs
            return calls.get('result', {'named_playlists': {'Rock_automatic': [('a', 'T', 'A')]}})

        def fake_min_size(result, min_size, log_prefix=''):
            calls['min_size'] = min_size
            return result

        def fake_top_n(result, limit, primary_genre_counts=None):
            calls['top_n'] = (limit, primary_genre_counts)
            return result

        monkeypatch.setattr(clustering_helper, '_perform_single_clustering_iteration', fake_iteration)
        monkeypatch.setattr(
            clustering_postprocessing, 'apply_minimum_size_filter_to_clustering_result', fake_min_size
        )
        monkeypatch.setattr(
            clustering_postprocessing, 'select_diverse_playlists_with_genre_coverage', fake_top_n
        )
        return calls

    def test_one_kmeans_pass_at_a_fixed_k(self, patched, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 0)
        state = {}
        result = naming_preview._cluster_sample('job-1', state)
        assert result['named_playlists']
        iteration = patched['iteration']
        assert iteration['clustering_method'] == 'kmeans'
        assert iteration['num_clusters_min_max'] == (3, 3)
        assert iteration['elite_solutions_params_list'] == []
        assert iteration['item_ids_for_subset'] == ['a', 'b', 'c', 'd']
        assert patched['min_size'] == 2
        assert 'top_n' not in patched
        assert state['song_count'] == 4

    def test_the_diverse_top_n_selection_runs_when_configured(self, patched, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 8)
        naming_preview._cluster_sample('job-1', {})
        assert patched['top_n'] == (8, {'rock': 2})

    def test_too_few_songs_fail_with_a_readable_problem(self, patched, monkeypatch):
        monkeypatch.setattr(config, 'MIN_PLAYLIST_SIZE_FOR_TOP_N', 50)
        state = {}
        with pytest.raises(naming_preview.PreviewUnavailable):
            naming_preview._cluster_sample('job-1', state)
        assert 'Not enough analyzed songs' in state['preview_error']
        assert 'iteration' not in patched

    def test_no_playlists_fail_with_a_readable_problem(self, patched, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 0)
        patched['result'] = {'fitness_score': -1.0}
        state = {}
        with pytest.raises(naming_preview.PreviewUnavailable):
            naming_preview._cluster_sample('job-1', state)
        assert 'no playlists' in state['preview_error']


class TestNamePlaylists:
    @pytest.mark.parametrize('mode, prompt', [('concept', None), ('title', 'Edited.')])
    def test_every_playlist_goes_through_the_real_naming_call(self, monkeypatch, mode, prompt):
        from tasks import clustering_helper

        answers = {
            'A_automatic': 'Velvet Light',
            'B_automatic': 'Velvet Light',
            'C_automatic': 'C_automatic',
        }
        calls = []
        reports = []

        def fake_name(original_name, songs, centroids, provider, *positional,
                      primary_genre=None, naming_mode=None, title_prompt=None):
            calls.append({
                'name': original_name,
                'provider': provider,
                'positional': len(positional),
                'avoid': list(positional[9]),
                'centroids': centroids,
                'primary_genre': primary_genre,
                'naming_mode': naming_mode,
                'title_prompt': title_prompt,
            })
            if original_name == 'D_automatic':
                raise RuntimeError('provider exploded')
            return answers[original_name]

        monkeypatch.setattr(clustering_helper, '_try_ai_name_playlist', fake_name)
        monkeypatch.setattr(
            naming_preview, '_report',
            lambda task_id, state, **f: (state.update(f), reports.append(len(state['titles']))),
        )
        centroids = {'A_automatic': {'rock': 0.9}}
        result = {
            'named_playlists': {
                'A_automatic': [('a1', 'Song 1', 'Artist 1'), ('a2', 'Song 2', None)],
                'B_automatic': [('b1', 'Song 3', 'Artist 3')],
                'C_automatic': [('c1', 'Song 4', 'Artist 4')],
                'D_automatic': [('d1', 'Song 5', 'Artist 5')],
                'Empty_automatic': [],
            },
            'playlist_centroids': centroids,
            'playlist_primary_genres': {'A_automatic': 'rock'},
        }
        state = naming_preview._initial_state(mode, '')
        naming_preview._name_playlists('job-1', state, result, mode, prompt, AI_CONFIG)

        assert state['total'] == 4 and state['done'] == 4
        assert reports == [0, 1, 2, 3, 4]
        assert [t['title'] for t in state['titles']] == [
            'Velvet Light', 'Velvet Light (2)', 'C_automatic', 'D_automatic',
        ]
        assert [t['tag_name_kept'] for t in state['titles']] == [False, False, True, True]
        assert state['titles'][0]['sample'] == ['Song 1 - Artist 1', 'Song 2 - Unknown Artist']
        assert state['titles'][0]['song_count'] == 2
        assert [call['name'] for call in calls] == [
            'A_automatic', 'B_automatic', 'C_automatic', 'D_automatic',
        ]
        assert all(call['naming_mode'] == mode for call in calls)
        assert all(call['title_prompt'] == prompt for call in calls)
        assert all(call['provider'] == 'OLLAMA' and call['positional'] == 10 for call in calls)
        assert all(call['centroids'] is centroids for call in calls)
        assert [call['primary_genre'] for call in calls] == ['rock', None, None, None]
        assert [call['avoid'] for call in calls] == [
            [],
            ['Velvet Light'],
            ['Velvet Light', 'Velvet Light (2)'],
            ['Velvet Light', 'Velvet Light (2)', 'C_automatic'],
        ]
