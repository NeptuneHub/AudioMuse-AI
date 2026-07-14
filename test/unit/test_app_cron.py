# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Cron scheduler dispatch and the sonic-fingerprint task it enqueues.

Exercises run_due_cron_jobs and the RQ task behind the sonic-fingerprint row.
Every cron branch enqueues: nothing runs inline on the poll thread, so a slow
media server cannot swallow a scheduling window.

Main Features:
* The sonic-fingerprint row enqueues its task rather than running it inline
* Empty fingerprint results skip both playlist upsert and the legacy fallback
* Non-empty results upsert under the constant cron playlist name via item_ids
* NotImplementedError from the backend falls back to a timestamped legacy playlist
* A live main task blocks a cron analysis/clustering start, as the manual endpoints do
* A failed enqueue leaves FAILURE, never a PENDING row that would 409 every later start
"""

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
    ):
        run_due_cron_jobs()

    assert save.call_args_list[-1][0][2] == TASK_STATUS_FAILURE


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_plugin_branch_forwards_the_schedules_server_scope(mock_get_db, _matches):
    """A plugin schedule runs per server, like every other scheduled task."""
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
    assert kwargs['kwargs'] == {'server_scope': 'default'}


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
