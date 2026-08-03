# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Cron scheduler dispatch and the sonic-fingerprint task it enqueues.

Exercises run_due_cron_jobs and the RQ task behind the sonic-fingerprint row.
Batch rows enqueue so a slow media server cannot swallow a scheduling window; the
alchemy radio runs inline in Flask, the only process holding the similarity index.

Main Features:
* The sonic-fingerprint row enqueues its task rather than running it inline
* The alchemy-radio row runs inline in Flask, never on a worker, and records
  STARTED then SUCCESS (or FAILURE, without leaving the row STARTED forever)
* Empty fingerprint results skip both playlist upsert and the legacy fallback
* Non-empty results upsert under the constant cron playlist name via item_ids
* NotImplementedError from the backend falls back to a timestamped legacy playlist
* A live main task blocks a cron analysis/clustering start, as the manual endpoints do
* A failed enqueue leaves FAILURE, never a PENDING row that would 409 every later start
"""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch


def _make_cron_row(task_type='sonic_fingerprint'):
    return {
        'id': 1,
        'name': 'Sonic Fingerprint',
        'task_type': task_type,
        'cron_expr': '* * * * *',
        'enabled': True,
        'last_run': 0,
    }


def _setup_db_mock(task_type='sonic_fingerprint'):
    cur = MagicMock()
    cur.fetchall.return_value = [_make_cron_row(task_type)]
    # The row is claimed for its minute with an UPDATE ... WHERE last_run < %s;
    # rowcount == 1 means this tick won the claim.
    cur.rowcount = 1
    db = MagicMock()
    db.cursor.return_value = cur
    return db, cur


def _run_fingerprint_task():
    """Drive the task the sonic-fingerprint cron row enqueues, on the legacy default.

    Outside an RQ job get_current_job() is None, so the task writes no task_status
    row and needs no database.
    """
    from tasks.sonic_fingerprint_manager import run_sonic_fingerprint_task

    with patch('tasks.mediaserver.registry.servers_for_scope', return_value=[None]):
        return run_sonic_fingerprint_task(server_scope='all')


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_sonic_fingerprint_row_enqueues_instead_of_running_inline(mock_get_db, _matches):
    from app_cron import run_due_cron_jobs

    db, _cur = _setup_db_mock()
    mock_get_db.return_value = db

    with (
        patch('app_cron.save_task_status'),
        patch('app_cron.rq_queue_default') as queue,
        patch('tasks.sonic_fingerprint_manager.generate_sonic_fingerprint') as gen,
    ):
        run_due_cron_jobs()

    gen.assert_not_called()
    queue.enqueue.assert_called_once()
    assert (
        queue.enqueue.call_args[0][0]
        == 'tasks.sonic_fingerprint_manager.run_sonic_fingerprint_task'
    )
    assert queue.enqueue.call_args[1]['kwargs'] == {'server_scope': 'all'}


def test_sonic_fingerprint_task_skips_on_empty_results():
    with (
        patch('tasks.sonic_fingerprint_manager.generate_sonic_fingerprint', return_value=[]) as gen,
        patch('tasks.mediaserver.create_or_replace_playlist') as upsert,
        patch('tasks.ivf_manager.create_playlist_from_ids') as legacy,
    ):
        summary = _run_fingerprint_task()

    gen.assert_called_once()
    upsert.assert_not_called()
    legacy.assert_not_called()
    assert summary['playlists_created'] == 0


def test_sonic_fingerprint_task_calls_upsert_with_constant_name():
    from config import SONIC_FINGERPRINT_CRON_PLAYLIST_NAME

    fp = [{'item_id': 'a'}, {'item_id': 'b'}, {'item_id': 'c'}]

    with (
        patch('tasks.sonic_fingerprint_manager.generate_sonic_fingerprint', return_value=fp),
        patch(
            'tasks.mediaserver.create_or_replace_playlist', return_value={'Id': 'pl-x'}
        ) as upsert,
        patch('tasks.ivf_manager.create_playlist_from_ids') as legacy,
    ):
        summary = _run_fingerprint_task()

    upsert.assert_called_once_with(SONIC_FINGERPRINT_CRON_PLAYLIST_NAME, ['a', 'b', 'c'])
    legacy.assert_not_called()
    assert summary['playlists_created'] == 1


def test_sonic_fingerprint_task_falls_back_for_unsupported_backend():
    fp = [{'item_id': 'a'}]

    with (
        patch('tasks.sonic_fingerprint_manager.generate_sonic_fingerprint', return_value=fp),
        patch('tasks.mediaserver.create_or_replace_playlist', side_effect=NotImplementedError),
        patch('tasks.ivf_manager.create_playlist_from_ids', return_value='legacy-id') as legacy,
    ):
        _run_fingerprint_task()

    legacy.assert_called_once()
    legacy_name = legacy.call_args[0][0]
    assert legacy_name.startswith('Sonic Fingerprint (Cron ')
    assert legacy.call_args[0][1] == ['a']


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_alchemy_radio_row_runs_inline_in_flask_never_on_a_worker(mock_get_db, _matches):
    from app_cron import run_due_cron_jobs
    from config import TASK_STATUS_STARTED, TASK_STATUS_SUCCESS

    db, _cur = _setup_db_mock(task_type='alchemy_radio')
    mock_get_db.return_value = db

    summary = {'playlists_created': 2, 'failed': []}
    with (
        patch('app_cron.save_task_status') as save,
        patch('app_cron.rq_queue_default') as queue,
        patch('app_cron.rq_queue_high') as queue_high,
        patch('tasks.radio_manager.run_radio_playlists', return_value=summary) as run,
    ):
        run_due_cron_jobs()

    queue.enqueue.assert_not_called()
    queue_high.enqueue.assert_not_called()
    run.assert_called_once()
    assert run.call_args.kwargs['server_scope'] == 'all'
    assert callable(run.call_args.kwargs['report'])
    statuses = [c[0][2] for c in save.call_args_list]
    assert statuses == [TASK_STATUS_STARTED, TASK_STATUS_SUCCESS]
    assert save.call_args_list[-1][1]['details'] == summary


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_failed_inline_radio_run_is_recorded_as_failure_without_a_traceback(
    mock_get_db, _matches
):
    from app_cron import run_due_cron_jobs
    from config import TASK_STATUS_FAILURE

    db, _cur = _setup_db_mock(task_type='alchemy_radio')
    mock_get_db.return_value = db

    with (
        patch('app_cron.save_task_status') as save,
        patch(
            'tasks.radio_manager.run_radio_playlists',
            side_effect=RuntimeError('internal detail that must stay in logs'),
        ),
    ):
        run_due_cron_jobs()

    last_call = save.call_args_list[-1]
    assert last_call[0][2] == TASK_STATUS_FAILURE
    assert 'internal detail' not in last_call[1]['details']['error']
    db.rollback.assert_called_once()


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_the_inline_radio_row_heartbeats_progress_into_its_own_task_row(mock_get_db, _matches):
    """The row has no RQ job behind it, so a heartbeat is the only thing separating a
    long run from a dead one. Without it the janitor fails a perfectly healthy run
    after its 120s orphan grace, and the row then flips back to SUCCESS."""
    from app_cron import run_due_cron_jobs
    from config import TASK_STATUS_PROGRESS

    db, _cur = _setup_db_mock(task_type='alchemy_radio')
    mock_get_db.return_value = db

    def _fake_run(server_scope='all', report=None):
        report('Radio 1 of 2', 50.0)
        return {'playlists_created': 1, 'failed': []}

    with (
        patch('app_cron.save_task_status') as save,
        patch('tasks.radio_manager.run_radio_playlists', side_effect=_fake_run),
    ):
        run_due_cron_jobs()

    progress_calls = [c for c in save.call_args_list if c[0][2] == TASK_STATUS_PROGRESS]
    assert len(progress_calls) == 1
    assert progress_calls[0][1]['progress'] == 50
    assert progress_calls[0][1]['details']['status_message'] == 'Radio 1 of 2'


@patch('app_cron.get_db')
def test_an_inline_run_interrupted_by_a_restart_is_failed_when_the_cron_thread_starts(
    mock_get_db,
):
    """Only this process writes an inline run's final status, so a restart mid-run left
    a STARTED row nothing ever resolved. It is failed at startup, where no inline run
    can be live: skipping that occurrence is fine, outliving the process is not."""
    from app_cron import reap_interrupted_inline_runs
    from config import TASK_STATUS_FAILURE, TASK_STATUS_SUCCESS, TASK_STATUS_REVOKED

    cur = MagicMock()
    cur.fetchall.return_value = [{'task_id': 'radio-1', 'task_type': 'alchemy_radio'}]
    db = MagicMock()
    db.cursor.return_value = cur
    mock_get_db.return_value = db

    with patch('app_cron.save_task_status') as save:
        assert reap_interrupted_inline_runs() == 1

    select_params = cur.execute.call_args[0][1]
    assert select_params[0] == ['alchemy_radio']
    assert set(select_params[1:]) == {
        TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED,
    }
    assert save.call_args[0][:3] == ('radio-1', 'alchemy_radio', TASK_STATUS_FAILURE)
    assert 'restart' in save.call_args[1]['details']['error']


@patch('app_cron.get_db')
def test_startup_reap_writes_nothing_when_no_inline_run_was_interrupted(mock_get_db):
    from app_cron import reap_interrupted_inline_runs

    cur = MagicMock()
    cur.fetchall.return_value = []
    db = MagicMock()
    db.cursor.return_value = cur
    mock_get_db.return_value = db

    with patch('app_cron.save_task_status') as save:
        assert reap_interrupted_inline_runs() == 0

    save.assert_not_called()


def test_a_radio_row_can_never_gate_a_start_because_only_flask_can_finish_it():
    """The radio shares nothing with a batch run - it reads the index and upserts a
    playlist - so it never needed the main-task mutex, and holding it meant one
    interrupted run 409'd every Start Analysis/Clustering until a human intervened."""
    import database

    cur = MagicMock()
    cur.fetchone.return_value = None
    db = MagicMock()
    db.cursor.return_value = cur

    with patch('database.get_db', return_value=db):
        assert database.get_active_main_task() is None

    excluded = cur.execute.call_args[0][1][-1]
    assert 'alchemy_radio' in excluded
    assert 'alchemy_radio' in database.SELF_MANAGED_TASK_TYPES


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_cron_analysis_does_not_start_a_second_run_while_one_is_live(mock_get_db, _matches):
    """The manual endpoints 409 on a live main task; cron must refuse too."""
    from app_cron import run_due_cron_jobs

    db, _cur = _setup_db_mock(task_type='analysis')
    mock_get_db.return_value = db

    active = {'task_id': 'live-1', 'task_type': 'main_analysis', 'status': 'PROGRESS'}
    with (
        patch('app_cron.get_active_main_task', return_value=active),
        patch('app_cron.save_task_status') as save,
        patch('app_cron.rq_queue_high') as queue,
    ):
        run_due_cron_jobs()

    queue.enqueue.assert_not_called()
    save.assert_not_called()


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_failed_analysis_enqueue_is_recorded_as_failure_not_left_pending(mock_get_db, _matches):
    """A PENDING row with no job behind it would 409-block every later manual start."""
    from app_cron import run_due_cron_jobs
    from config import TASK_STATUS_FAILURE

    db, _cur = _setup_db_mock(task_type='analysis')
    mock_get_db.return_value = db

    queue = MagicMock()
    queue.enqueue.side_effect = RuntimeError("redis is down")
    with (
        patch('app_cron.get_active_main_task', return_value=None),
        patch('app_cron.save_task_status') as save,
        patch('app_cron.rq_queue_high', queue),
        # The start lock talks to the real connection, not app_cron's mocked one.
        patch('app_cron.main_task_start_lock', return_value=nullcontext()),
        patch('app_cron.clean_up_previous_main_tasks'),
        patch('app_cron.prune_task_status_history'),
    ):
        run_due_cron_jobs()

    assert save.call_args_list[-1][0][2] == TASK_STATUS_FAILURE


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_plugin_branch_always_runs_against_all_servers(mock_get_db, _matches):
    """Batch work always covers EVERY server, so a plugin schedule is enqueued
    with scope 'all' even if an old row still carries a narrower one: a stale
    'default' option must not quietly keep skipping the other servers."""
    from app_cron import run_due_cron_jobs

    row = _make_cron_row(task_type='plugin.demo.sync')
    row['options'] = {'server_scope': 'default'}
    cur = MagicMock()
    cur.fetchall.return_value = [row]
    cur.rowcount = 1
    db = MagicMock()
    db.cursor.return_value = cur
    mock_get_db.return_value = db

    plugin_manager = MagicMock()
    plugin_manager.get_cron_task.return_value = {
        'dotted': 'audiomuse_plugins.demo.tasks.sync', 'queue': 'default',
    }
    fake_plugin_module = MagicMock()
    fake_plugin_module.plugin_manager = plugin_manager

    with patch.dict('sys.modules', {'plugin.manager': fake_plugin_module}), \
            patch('app_cron.save_task_status'), \
            patch('app_cron.rq_queue_default') as queue:
        run_due_cron_jobs()

    assert queue.enqueue.called
    kwargs = queue.enqueue.call_args.kwargs
    assert kwargs['args'] == ('audiomuse_plugins.demo.tasks.sync',)
    assert kwargs['kwargs'] == {'server_scope': 'all'}


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_plugin_branch_defaults_to_all_servers(mock_get_db, _matches):
    from app_cron import run_due_cron_jobs

    row = _make_cron_row(task_type='plugin.demo.sync')
    cur = MagicMock()
    cur.fetchall.return_value = [row]
    cur.rowcount = 1
    db = MagicMock()
    db.cursor.return_value = cur
    mock_get_db.return_value = db

    plugin_manager = MagicMock()
    plugin_manager.get_cron_task.return_value = {
        'dotted': 'audiomuse_plugins.demo.tasks.sync', 'queue': 'default',
    }
    fake_plugin_module = MagicMock()
    fake_plugin_module.plugin_manager = plugin_manager

    with patch.dict('sys.modules', {'plugin.manager': fake_plugin_module}), \
            patch('app_cron.save_task_status'), \
            patch('app_cron.rq_queue_default') as queue:
        run_due_cron_jobs()

    assert queue.enqueue.call_args.kwargs['kwargs'] == {'server_scope': 'all'}
