# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the setup wizard naming preview that runs on the default worker.

The web process admits and enqueues a naming_preview side job and reads its row
back; the default-queue worker clusters a sample of the library and names every
playlist. These tests pin the queue model (one batch at a time, never the high
queue), the side-job rules derived from task_types, restart and cancel safety, the
calibration retry, and naming through the same helper as a real run.

Main Features:
* The type is a side job on the default queue: it blocks and is blocked by every
  batch start, is watched by the wedged nudge, and is never restarted.
* Starting takes the main-task start lock, refuses while any batch task or another
  preview runs with the parallel-batch rule in the message, and never uses high.
* Every side-job rule (recap, cancel history, history, blocking gate, enqueue
  clear) reads SIDE_JOB_TASK_TYPES instead of a hand-kept list, and a running
  preview still shows on the dashboard that it blocks.
* A running server sweep refuses the preview, as it refuses every batch start.
* A restarted preview fails instead of re-running, progress writes only UPDATE a
  RUNNING row, and expected problems end as TaskFailed.
* The K bound never divides by zero and is halved while too few playlists survive.
"""

from contextlib import contextmanager

import pytest

import config
import database
import task_types
import taskqueue
from tasks import naming_preview, recovery


class _Cursor:
    def __init__(self, conn):
        self.conn = conn
        self.rowcount = 0

    def execute(self, sql, params=None):
        text = ' '.join(sql.split())
        self.conn.log.append((text, params))
        if self.conn.fail_on and self.conn.fail_on in text:
            raise RuntimeError('database error')

    def fetchone(self):
        return self.conn.rows.pop(0) if self.conn.rows else None

    def fetchall(self):
        rows, self.conn.rows = self.conn.rows, []
        return rows

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Connection:
    def __init__(self, rows=None, fail_on=None):
        self.log = []
        self.rows = list(rows or [])
        self.fail_on = fail_on
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, cursor_factory=None):
        return _Cursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _entry():
    return next(e for e in task_types.ALL if e.name == naming_preview.PREVIEW_TASK_TYPE)


class TestRegistration:
    def test_the_task_function_is_allowed(self):
        assert naming_preview.PREVIEW_TASK_FUNC in taskqueue.ALLOWED_FUNCS
        assert taskqueue.resolve_func(naming_preview.PREVIEW_TASK_FUNC) is (
            naming_preview.run_naming_preview_task
        )

    def test_the_type_is_a_default_queue_side_job_that_blocks_batch_starts(self):
        entry = _entry()
        assert entry.queue == 'default' and entry.queue != 'high'
        assert entry.side_job and entry.blocks_starts and entry.watched_by_nudge
        assert entry.restarts == 0 and not entry.holds_main_index
        assert naming_preview.PREVIEW_TASK_TYPE in task_types.SIDE_JOB_TASK_TYPES
        assert naming_preview.PREVIEW_TASK_TYPE in task_types.NUDGE_TASK_TYPES
        assert naming_preview.PREVIEW_TASK_TYPE not in database.NON_BLOCKING_TASK_TYPES
        assert naming_preview.PREVIEW_TASK_TYPE not in task_types.NON_WORKER_TASK_TYPES
        assert naming_preview.PREVIEW_TASK_TYPE not in config.QUEUE_BLOCKING_TASK_TYPES

    def test_the_recovery_table_has_a_stance(self):
        stances = recovery.RECOVERY[naming_preview.PREVIEW_TASK_TYPE]
        assert stances[recovery.MAIN_ROW_SILENT].applicable

    def test_the_dashboard_and_cancel_history_read_the_side_job_list(self, monkeypatch):
        import app_helper
        from pathlib import Path

        app_source = (Path(app_helper.__file__).parent / 'app.py').read_text(encoding='utf-8')
        assert 'HIDDEN_ACTIVE_TASK_TYPES + task_types.SIDE_JOB_TASK_TYPES' in app_source
        active = app_source.split('def get_active_tasks_endpoint', 1)[1].split('\ndef ', 1)[0]
        assert '(non_terminal_statuses, HIDDEN_ACTIVE_TASK_TYPES)' in active, (
            'the preview refuses dashboard starts, so the dashboard must list it and offer its Stop'
        )
        last = app_source.split('def get_last_overall_task_status_endpoint', 1)[1].split('\ndef ', 1)[0]
        assert '(NON_USER_TASK_TYPES,)' in last
        recorded = []
        monkeypatch.setattr(app_helper, '_record_one_cancellation', lambda *a: recorded.append(a))
        app_helper._record_cancel_history(
            [{'task_id': 'p', 'task_type': naming_preview.PREVIEW_TASK_TYPE},
             {'task_id': 'm', 'task_type': 'main_analysis'}], set(), 0, 'x'
        )
        assert [row[0]['task_id'] for row in recorded] == ['m']

    def test_a_finished_preview_never_becomes_the_dashboard_status(self):
        from pathlib import Path

        root = Path(naming_preview.__file__).resolve().parents[1]
        app_source = (root / 'app.py').read_text(encoding='utf-8')
        active = app_source.split('def get_active_tasks_endpoint', 1)[1].split('\ndef ', 1)[0]
        assert "task_item['side_job'] = task_item.get('task_type') in task_types.SIDE_JOB_TASK_TYPES" in active
        script = (root / 'static' / 'script.js').read_text(encoding='utf-8')
        check = script[script.index('async function checkActiveTasks'):]
        check = check[:check.index('\nfunction ')]
        side = check.index('previousDetails.side_job')
        assert side < check.index('getTaskStatusEndpointUrl'), (
            'a finished preview must reload the real last task before any final-status popup'
        )
        assert 'fetchAndDisplayOverallLastTask()' in check[side:check.index('getTaskStatusEndpointUrl')]


class TestSideJobRules:
    def test_a_finished_side_job_does_not_collapse_the_table(self):
        db = _Connection()
        deleted = database.collapse_finished_task(
            db, 'p1', naming_preview.PREVIEW_TASK_TYPE, None, config.TASK_STATUS_SUCCESS
        )
        assert deleted == 0 and db.log == []

    def test_a_finished_side_job_records_no_task_history(self, monkeypatch):
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
        database.collapse_finished_task(db, 'm1', 'main_clustering', None, config.TASK_STATUS_SUCCESS)
        assert recorded
        assert any(sql.startswith('DELETE FROM task_status WHERE task_id <> %s') for sql, _p in db.log)

    def test_the_blocking_gate_sees_side_jobs_and_every_main_type(self):
        db = _Connection()
        database.get_queue_blocking_task(conn=db)
        sql, params = db.log[0]
        blocking_types = params[1]
        assert naming_preview.PREVIEW_TASK_TYPE in blocking_types
        assert set(config.QUEUE_BLOCKING_TASK_TYPES) <= set(blocking_types)

    def _enqueue_calls(self, monkeypatch, task_type, **kwargs):
        from unittest.mock import MagicMock

        calls = []
        fake_sql = MagicMock()
        fake_sql.take_start_lock.side_effect = lambda cur: calls.append('lock')
        fake_sql.clear_task_status.side_effect = lambda cur: calls.append('clear')
        fake_sql.insert_job.side_effect = lambda cur, **kw: calls.append(('insert', kw['queue'], kw['max_attempts'])) or True
        fake_sql.notify_job.side_effect = lambda cur, queue: calls.append('notify')
        monkeypatch.setattr(taskqueue, '_with_cursor', lambda action, conn: action(fake_sql, MagicMock()))
        taskqueue.enqueue('tasks.analysis.run_analysis_task' if task_type != naming_preview.PREVIEW_TASK_TYPE
                          else naming_preview.PREVIEW_TASK_FUNC,
                          task_id='t1', task_type=task_type, **kwargs)
        return calls

    def test_a_main_root_enqueue_still_clears_finished_rows(self, monkeypatch):
        calls = self._enqueue_calls(monkeypatch, 'main_analysis', queue=taskqueue.QUEUE_HIGH)
        assert calls[:2] == ['lock', 'clear']

    def test_a_side_job_enqueue_keeps_the_recap(self, monkeypatch):
        calls = self._enqueue_calls(monkeypatch, naming_preview.PREVIEW_TASK_TYPE, queue=taskqueue.QUEUE_DEFAULT)
        assert 'clear' not in calls and calls[0] == 'lock'
        assert ('insert', taskqueue.QUEUE_DEFAULT, 0) in calls


class TestStartPreview:
    @pytest.fixture
    def patched(self, monkeypatch):
        state = {'events': [], 'blocking': None, 'enqueued': []}
        db = _Connection()

        @contextmanager
        def fake_lock(conn=None):
            state['events'].append('lock')
            yield
            state['events'].append('unlock')

        def fake_gate(conn=None):
            state['events'].append('gate')
            return state['blocking']

        def fake_active(task_type=None, conn=None):
            state['events'].append('active:%s' % task_type)
            return state.get('sweep')

        monkeypatch.setattr(database, 'get_db', lambda: db)
        monkeypatch.setattr(database, 'main_task_start_lock', fake_lock)
        monkeypatch.setattr(database, 'get_queue_blocking_task', fake_gate)
        monkeypatch.setattr(database, 'get_active_main_task', fake_active)
        monkeypatch.setattr(taskqueue, 'enqueue', lambda func, **k: (
            state['events'].append('enqueue'), state['enqueued'].append((func, k))
        ))
        state['db'] = db
        return state

    def test_another_batch_task_refuses_the_preview_with_the_parallel_rule(self, patched):
        patched['blocking'] = {'task_id': 'a1', 'task_type': 'main_analysis', 'status': 'RUNNING'}
        task_id, message = naming_preview.start_preview('title', 'Name it.')
        assert task_id is None
        assert 'main_analysis' in message and 'never runs two batch tasks in parallel' in message
        assert patched['enqueued'] == [] and patched['db'].rollbacks == 1
        assert patched['events'] == ['lock', 'gate', 'unlock']

    def test_a_running_server_sweep_refuses_the_preview_like_every_batch_start(self, patched):
        patched['sweep'] = {'task_id': 's1', 'task_type': 'server_sweep', 'status': 'RUNNING'}
        task_id, message = naming_preview.start_preview('concept', None)
        assert task_id is None
        assert 'server_sweep' in message and 'never runs two batch tasks in parallel' in message
        assert patched['enqueued'] == [] and patched['db'].rollbacks == 1

    def test_a_live_preview_refuses_a_second_one(self, patched):
        patched['blocking'] = {'task_id': 'p0', 'task_type': naming_preview.PREVIEW_TASK_TYPE, 'status': 'NEW'}
        task_id, message = naming_preview.start_preview('concept', None)
        assert task_id is None and message == naming_preview.PREVIEW_RUNNING_MESSAGE
        assert patched['enqueued'] == []

    def test_a_start_is_admitted_under_the_lock_on_the_default_queue(self, patched):
        task_id, message = naming_preview.start_preview('title', 'Name it.')
        assert task_id and message is None
        assert patched['events'] == ['lock', 'gate', 'active:server_sweep', 'enqueue', 'unlock']
        func, kwargs = patched['enqueued'][0]
        assert func == naming_preview.PREVIEW_TASK_FUNC
        assert kwargs['queue'] == taskqueue.QUEUE_DEFAULT
        assert kwargs['queue'] != taskqueue.QUEUE_HIGH
        assert 'max_attempts' not in kwargs
        assert kwargs['task_type'] == naming_preview.PREVIEW_TASK_TYPE
        assert kwargs['kwargs'] == {'mode': 'title', 'instructions': 'Name it.'}
        assert kwargs['conn'] is patched['db'] and patched['db'].commits == 1
        deletes = [(sql, p) for sql, p in patched['db'].log if sql.startswith('DELETE')]
        assert deletes and deletes[0][1][0] == naming_preview.PREVIEW_TASK_TYPE


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
        assert state['status'] == 'running' and state['message'] == naming_preview.PREVIEW_WAITING_MESSAGE

    def test_a_running_row_shows_progress_and_titles(self, monkeypatch):
        details = {'message': 'Naming 2 playlists...', 'mode': 'title', 'total': 2,
                   'song_count': 500, 'titles': [{'title': 'A'}]}
        state = self._status(monkeypatch, {'task_id': 't', 'status': config.TASK_STATUS_RUNNING,
                                           'details': details})
        assert state['status'] == 'running' and state['task_id'] == 't'
        assert state['done'] == 1 and state['total'] == 2 and state['song_count'] == 500

    def test_a_finished_row_is_done_with_its_own_message(self, monkeypatch):
        details = {'message': 'queue summary', 'preview_message': 'Preview complete: 1 playlist titles.',
                   'titles': [{'title': 'A'}], 'total': 1}
        state = self._status(monkeypatch, {'task_id': 't', 'status': config.TASK_STATUS_SUCCESS,
                                           'details': details})
        assert state['status'] == 'done' and state['message'] == 'Preview complete: 1 playlist titles.'

    def test_an_expected_problem_is_shown(self, monkeypatch):
        details = {'message': 'Traceback secret', 'preview_error': 'Not enough analyzed songs.'}
        state = self._status(monkeypatch, {'task_id': 't', 'status': config.TASK_STATUS_FAIL,
                                           'details': details})
        assert state['status'] == 'failed' and state['message'] == 'Not enough analyzed songs.'

    def test_any_other_failure_is_generic(self, monkeypatch):
        state = self._status(monkeypatch, {'task_id': 't', 'status': config.TASK_STATUS_FAIL,
                                           'details': '{"message": "KeyError secret"}'})
        assert state['message'] == naming_preview.PREVIEW_FAILED_MESSAGE

    def test_a_preview_whose_worker_died_says_it_was_interrupted(self, monkeypatch):
        import taskqueue

        details = {'message': 'The worker running this task stopped unexpectedly.',
                   'error': taskqueue.WORKER_LOST_ERROR}
        state = self._status(monkeypatch, {'task_id': 't', 'status': config.TASK_STATUS_FAIL,
                                           'details': details})
        assert state['status'] == 'failed'
        assert state['message'] == naming_preview.PREVIEW_INTERRUPTED_MESSAGE

    def test_a_cancelled_preview_says_so(self, monkeypatch):
        state = self._status(monkeypatch, {'task_id': 't', 'status': config.TASK_STATUS_REVOKED,
                                           'details': {}})
        assert state['status'] == 'failed' and state['message'] == naming_preview.PREVIEW_CANCELLED_MESSAGE

    def test_malformed_titles_become_an_empty_list(self, monkeypatch):
        state = self._status(monkeypatch, {'task_id': 't', 'status': config.TASK_STATUS_RUNNING,
                                           'details': {'titles': 'nope'}})
        assert state['titles'] == [] and state['done'] == 0


class TestWorkerTask:
    def test_an_expected_problem_is_a_task_failed(self):
        assert issubclass(naming_preview.PreviewUnavailable, taskqueue.TaskFailed)

    def test_the_task_reports_and_returns_the_titles(self, monkeypatch):
        reports = []
        monkeypatch.setattr(taskqueue, 'current_task_id', lambda: 'job-1')
        monkeypatch.setattr(naming_preview, '_already_started', lambda task_id: False)
        monkeypatch.setattr(naming_preview, '_report', lambda task_id, state, **f: (state.update(f), reports.append(dict(state))))
        monkeypatch.setattr(naming_preview, '_cluster_sample', lambda task_id, state: {'named_playlists': {}})
        received = {}

        def fake_name(task_id, state, result, mode, instructions):
            received.update(mode=mode, instructions=instructions)
            state['titles'].append({'title': 'A'})
            state['total'] = 1

        monkeypatch.setattr(naming_preview, '_name_playlists', fake_name)
        summary = naming_preview.run_naming_preview_task(mode=' TITLE ', instructions='Edited.')
        assert received == {'mode': 'title', 'instructions': 'Edited.'}
        assert summary['titles'] == [{'title': 'A'}]
        assert summary['preview_message'] == 'Preview complete: 1 playlist titles.'
        assert reports[0]['started'] is True

    def test_a_restarted_preview_fails_instead_of_running_again(self, monkeypatch):
        reported = {}
        monkeypatch.setattr(taskqueue, 'current_task_id', lambda: 'job-1')
        monkeypatch.setattr(naming_preview, '_already_started', lambda task_id: True)
        monkeypatch.setattr(naming_preview, '_report', lambda task_id, state, **f: reported.update(f))
        monkeypatch.setattr(naming_preview, '_cluster_sample', lambda *a: pytest.fail('must not cluster again'))
        with pytest.raises(naming_preview.PreviewUnavailable):
            naming_preview.run_naming_preview_task(mode='title', instructions='Edited.')
        assert reported['preview_error'] == naming_preview.PREVIEW_INTERRUPTED_MESSAGE

    @pytest.mark.parametrize('rows, expected', [
        ([({'started': True, 'titles': []},)], True),
        ([('{"message": "Waiting"}',)], False),
        ([], False),
    ])
    def test_already_started_reads_the_row_marker(self, monkeypatch, rows, expected):
        db = _Connection(rows=rows)
        monkeypatch.setattr(database, 'get_db', lambda: db)
        assert naming_preview._already_started('job-1') is expected

    def test_report_only_updates_a_running_row(self, monkeypatch):
        db = _Connection()
        monkeypatch.setattr(database, 'get_db', lambda: db)
        state = {'mode': 'title', 'message': '', 'song_count': 0, 'done': 0, 'total': 4, 'titles': [1, 2]}
        naming_preview._report('job-1', state, message='Naming...')
        sql, params = db.log[0]
        assert sql.startswith('UPDATE task_status SET details = %s')
        assert 'INSERT' not in sql and 'status = %s' in sql
        assert params[1] == 52 and params[2] == 'job-1' and params[3] == config.TASK_STATUS_RUNNING
        assert '"message": "Naming..."' in params[0] and db.commits == 1

    def test_a_failed_report_rolls_back_and_does_not_raise(self, monkeypatch):
        db = _Connection(fail_on='UPDATE')
        monkeypatch.setattr(database, 'get_db', lambda: db)
        naming_preview._report('job-1', {'total': 0, 'titles': []})
        assert db.rollbacks == 1

    def test_report_without_a_task_id_touches_nothing(self, monkeypatch):
        monkeypatch.setattr(database, 'get_db', lambda: pytest.fail('no write'))
        state = {'total': 0, 'titles': []}
        naming_preview._report(None, state, message='x')
        assert state['message'] == 'x'

    def test_the_ai_config_has_the_keys_a_clustering_run_builds(self, monkeypatch):
        from tasks import clustering_helper

        monkeypatch.setattr(config, 'AI_MODEL_PROVIDER', 'gemini')
        monkeypatch.setattr(config, 'GEMINI_API_KEY', 'server-key')
        built = naming_preview._ai_config()
        assert built['provider'] == 'GEMINI' and built['gemini_key'] == 'server-key'
        captured = {}
        monkeypatch.setattr(clustering_helper, '_name_playlist_with_ai_config',
                            lambda name, songs, centroids, ai_config, **k: captured.update(ai_config))
        clustering_helper._try_ai_name_playlist('n', [], {}, *range(10))
        assert set(captured) == set(built)


class TestLimits:
    @pytest.fixture(autouse=True)
    def limits(self, monkeypatch):
        monkeypatch.setattr(config, 'CLUSTERING_MAX_PLAYLIST_SONGS', 200)
        monkeypatch.setattr(config, 'MIN_PLAYLIST_SIZE_FOR_TOP_N', 20)
        monkeypatch.setattr(config, 'NUM_CLUSTERS_MAX', 100)
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 0)

    def test_a_large_sample_uses_the_configured_maximum(self):
        assert naming_preview.preview_limits(10000) == (100, 100)

    def test_a_small_sample_is_bounded_by_playlist_size(self):
        assert naming_preview.preview_limits(400) == (10, 10)

    def test_k_never_reaches_the_song_count(self):
        assert naming_preview.preview_limits(3) == (2, 2)

    def test_top_n_sets_how_many_playlists_are_needed(self, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 8)
        assert naming_preview.preview_limits(10000) == (100, 8)

    def test_zero_divisors_do_not_raise(self, monkeypatch):
        monkeypatch.setattr(config, 'CLUSTERING_MAX_PLAYLIST_SONGS', 0)
        monkeypatch.setattr(config, 'MIN_PLAYLIST_SIZE_FOR_TOP_N', 0)
        clusters, needed = naming_preview.preview_limits(500)
        assert 2 <= clusters <= 499 and needed >= 1


class TestSampling:
    def test_the_sample_is_capped_read_only_and_ends_its_transaction(self, monkeypatch):
        from tasks import clustering, clustering_helper

        db = _Connection(rows=[{'item_id': 'x', 'mood_vector': 'rock:1'}])
        monkeypatch.setattr(database, 'get_db', lambda: db)
        monkeypatch.setattr(clustering, '_prepare_genre_map', lambda rows: {'rock': rows})
        monkeypatch.setattr(clustering, '_calculate_target_songs_per_genre', lambda *a: 50)
        monkeypatch.setattr(
            clustering_helper, '_get_stratified_song_subset',
            lambda genre_map, target: [{'item_id': str(i)} for i in range(12000)],
        )
        item_ids, genre_map = naming_preview._sample_item_ids()
        assert len(item_ids) == naming_preview.PREVIEW_MAX_SONGS == 10000
        assert len(set(item_ids)) == 10000
        assert db.log and all(sql.upper().startswith('SELECT') for sql, _p in db.log)
        assert db.commits == 1


class TestClusterSample:
    @pytest.fixture
    def patched(self, monkeypatch):
        from tasks import clustering_postprocessing

        calls = {'k': [], 'results': []}
        monkeypatch.setattr(config, 'MIN_PLAYLIST_SIZE_FOR_TOP_N', 2)
        monkeypatch.setattr(config, 'CLUSTERING_CALIBRATION_MAX_TRIES', 3)
        monkeypatch.setattr(naming_preview, '_report', lambda task_id, state, **f: state.update(f))
        monkeypatch.setattr(
            naming_preview, '_sample_item_ids',
            lambda: (['a', 'b', 'c', 'd'], {'rock': [1, 2], '__other__': [3]}),
        )
        monkeypatch.setattr(naming_preview, 'preview_limits', lambda count: (8, 3))

        def fake_once(item_ids, clusters):
            calls['k'].append(clusters)
            if calls['results']:
                return calls['results'].pop(0)
            return {'named_playlists': {'A': [1, 2], 'B': [1, 2], 'C': [1, 2]}}

        def fake_min_size(result, min_size, log_prefix=''):
            calls['min_size'] = (min_size, result)
            return result

        def fake_top_n(result, limit, primary_genre_counts=None):
            calls['top_n'] = (limit, primary_genre_counts)
            return result

        monkeypatch.setattr(naming_preview, '_cluster_once', fake_once)
        monkeypatch.setattr(clustering_postprocessing, 'apply_minimum_size_filter_to_clustering_result', fake_min_size)
        monkeypatch.setattr(clustering_postprocessing, 'select_diverse_playlists_with_genre_coverage', fake_top_n)
        return calls

    def test_one_pass_when_enough_playlists_survive(self, patched, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 0)
        state = {}
        naming_preview._cluster_sample('job-1', state)
        assert patched['k'] == [8] and 'top_n' not in patched
        assert state['song_count'] == 4

    def test_k_is_halved_while_too_few_playlists_survive(self, patched, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 0)
        weak = {'named_playlists': {'A': [1, 2], 'B': [1]}}
        strong = {'named_playlists': {'A': [1, 2], 'B': [1, 2], 'C': [1, 2, 3]}}
        patched['results'] = [weak, strong]
        naming_preview._cluster_sample('job-1', {})
        assert patched['k'] == [8, 4]
        assert patched['min_size'][1] is strong

    def test_the_best_pass_is_kept_when_tries_run_out(self, patched, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 0)
        one = {'named_playlists': {'A': [1, 2]}}
        two = {'named_playlists': {'A': [1, 2], 'B': [1, 2]}}
        none = {'named_playlists': {'A': [1]}}
        patched['results'] = [one, two, none]
        naming_preview._cluster_sample('job-1', {})
        assert patched['k'] == [8, 4, 3]
        assert patched['min_size'][1] is two

    def test_the_diverse_top_n_selection_runs_when_configured(self, patched, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 8)
        naming_preview._cluster_sample('job-1', {})
        assert patched['top_n'] == (8, {'rock': 2})

    def test_too_few_songs_fail_with_a_readable_problem(self, patched, monkeypatch):
        monkeypatch.setattr(config, 'MIN_PLAYLIST_SIZE_FOR_TOP_N', 50)
        state = {}
        with pytest.raises(naming_preview.PreviewUnavailable):
            naming_preview._cluster_sample('job-1', state)
        assert 'Not enough analyzed songs' in state['preview_error'] and patched['k'] == []

    def test_no_playlists_fail_with_a_readable_problem(self, patched, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 0)
        patched['results'] = [{'fitness_score': -1.0}] * 3
        state = {}
        with pytest.raises(naming_preview.PreviewUnavailable):
            naming_preview._cluster_sample('job-1', state)
        assert 'no playlists' in state['preview_error']

    def test_cluster_once_runs_kmeans_at_a_fixed_k(self, monkeypatch):
        from tasks import clustering_helper

        captured = {}
        monkeypatch.setattr(clustering_helper, '_perform_single_clustering_iteration',
                            lambda **kwargs: captured.update(kwargs) or {'ok': True})
        assert naming_preview._cluster_once(['a', 'b'], 5) == {'ok': True}
        assert captured['clustering_method'] == 'kmeans'
        assert captured['num_clusters_min_max'] == (5, 5)
        assert captured['elite_solutions_params_list'] == []


class TestNamePlaylists:
    @pytest.mark.parametrize('mode, prompt', [('concept', None), ('title', 'Edited.')])
    def test_every_playlist_goes_through_the_real_naming_call(self, monkeypatch, mode, prompt):
        from tasks import clustering_helper

        answers = {'A_automatic': 'Velvet Light', 'B_automatic': 'Velvet Light', 'C_automatic': 'C_automatic'}
        calls, reports = [], []
        db = _Connection()

        def fake_name(original_name, songs, centroids, ai_config, avoid_names,
                      primary_genre=None, naming_mode=None, title_prompt=None):
            calls.append({
                'name': original_name, 'provider': ai_config['provider'],
                'avoid': list(avoid_names), 'centroids': centroids, 'primary_genre': primary_genre,
                'naming_mode': naming_mode, 'title_prompt': title_prompt,
            })
            if original_name == 'D_automatic':
                raise RuntimeError('provider exploded')
            return answers[original_name]

        monkeypatch.setattr(config, 'AI_MODEL_PROVIDER', 'ollama')
        monkeypatch.setattr(database, 'get_db', lambda: db)
        monkeypatch.setattr(clustering_helper, '_name_playlist_with_ai_config', fake_name)
        monkeypatch.setattr(naming_preview, '_report',
                            lambda task_id, state, **f: (state.update(f), reports.append(len(state['titles']))))
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
        naming_preview._name_playlists('job-1', state, result, mode, prompt)

        assert state['total'] == 4 and state['done'] == 4
        assert reports == [0, 1, 2, 3, 4]
        assert [t['title'] for t in state['titles']] == ['Velvet Light', 'Velvet Light (2)', 'C_automatic', 'D_automatic']
        assert [t['tag_name_kept'] for t in state['titles']] == [False, False, True, True]
        assert state['titles'][0]['sample'] == ['Song 1 - Artist 1', 'Song 2 - Unknown Artist']
        assert all(c['naming_mode'] == mode and c['title_prompt'] == prompt for c in calls)
        assert all(c['provider'] == 'OLLAMA' for c in calls)
        assert all(c['centroids'] is centroids for c in calls)
        assert [c['primary_genre'] for c in calls] == ['rock', None, None, None]
        assert [c['avoid'] for c in calls] == [
            [], ['Velvet Light'], ['Velvet Light', 'Velvet Light (2)'],
            ['Velvet Light', 'Velvet Light (2)', 'C_automatic'],
        ]
        assert db.rollbacks == 1
