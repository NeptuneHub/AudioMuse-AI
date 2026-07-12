# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Cron scheduler dispatch for the sonic-fingerprint job.

Exercises run_due_cron_jobs when a due row selects the sonic-fingerprint task,
asserting how generated fingerprints flow into playlist creation.

Main Features:
* Empty fingerprint results skip both playlist upsert and the legacy fallback
* Non-empty results upsert under the constant cron playlist name via item_ids
* NotImplementedError from the backend falls back to a timestamped legacy playlist
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


def _setup_db_mock():
    cur = MagicMock()
    cur.fetchall.return_value = [_make_cron_row()]
    db = MagicMock()
    db.cursor.return_value = cur
    return db, cur


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_sonic_fingerprint_branch_skips_on_empty_results(mock_get_db, _matches):
    from app_cron import run_due_cron_jobs

    db, _cur = _setup_db_mock()
    mock_get_db.return_value = db

    with (
        patch('tasks.sonic_fingerprint_manager.generate_sonic_fingerprint', return_value=[]) as gen,
        patch('tasks.mediaserver.create_or_replace_playlist') as upsert,
        patch('tasks.ivf_manager.create_playlist_from_ids') as legacy,
    ):
        run_due_cron_jobs()

    gen.assert_called_once()
    upsert.assert_not_called()
    legacy.assert_not_called()


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_sonic_fingerprint_branch_calls_upsert_with_constant_name(mock_get_db, _matches):
    from app_cron import run_due_cron_jobs
    from config import SONIC_FINGERPRINT_CRON_PLAYLIST_NAME

    db, _cur = _setup_db_mock()
    mock_get_db.return_value = db

    fp = [{'item_id': 'a'}, {'item_id': 'b'}, {'item_id': 'c'}]

    with (
        patch('tasks.sonic_fingerprint_manager.generate_sonic_fingerprint', return_value=fp),
        patch(
            'tasks.mediaserver.create_or_replace_playlist', return_value={'Id': 'pl-x'}
        ) as upsert,
        patch('tasks.ivf_manager.create_playlist_from_ids') as legacy,
    ):
        run_due_cron_jobs()

    upsert.assert_called_once_with(SONIC_FINGERPRINT_CRON_PLAYLIST_NAME, ['a', 'b', 'c'])
    legacy.assert_not_called()


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_sonic_fingerprint_branch_falls_back_for_unsupported_backend(mock_get_db, _matches):
    from app_cron import run_due_cron_jobs

    db, _cur = _setup_db_mock()
    mock_get_db.return_value = db

    fp = [{'item_id': 'a'}]

    with (
        patch('tasks.sonic_fingerprint_manager.generate_sonic_fingerprint', return_value=fp),
        patch('tasks.mediaserver.create_or_replace_playlist', side_effect=NotImplementedError),
        patch('tasks.ivf_manager.create_playlist_from_ids', return_value='legacy-id') as legacy,
    ):
        run_due_cron_jobs()

    legacy.assert_called_once()
    legacy_name = legacy.call_args[0][0]
    assert legacy_name.startswith('Sonic Fingerprint (Cron ')
    assert legacy.call_args[0][1] == ['a']


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_plugin_branch_forwards_the_schedules_server_scope(mock_get_db, _matches):
    """A plugin schedule runs per server, like every other scheduled task."""
    from app_cron import run_due_cron_jobs

    row = _make_cron_row(task_type='plugin.demo.sync')
    row['options'] = {'server_scope': 'default'}
    cur = MagicMock()
    cur.fetchall.return_value = [row]
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
