# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only

"""Setup saves report whether the new configuration is actually active.

Saving settings publishes a worker restart. Reporting success before the workers
acknowledged it told the user their change was live while every worker was still
running the old configuration.

Main Features:
* A save reports success only once the workers acknowledged the restart
* A failed or unacknowledged WORKER restart is surfaced instead of swallowed
* A deliberately declined Flask restart is an opt-out, not a failed save
"""

from unittest.mock import MagicMock

import app_setup


def _save_setup(
    monkeypatch, *, worker_result=True, flask_result=True, worker_error=None
):
    monkeypatch.setattr(app_setup, '_get_allowed_setup_keys', lambda: {'MEDIASERVER_TYPE'})
    monkeypatch.setattr(app_setup, '_has_admin_user', lambda: True)
    monkeypatch.setattr(app_setup.config, 'AUTH_ENABLED', False)
    monkeypatch.setattr(app_setup.config, 'MEDIASERVER_TYPE', 'jellyfin')
    monkeypatch.setattr(
        app_setup.config, 'MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE', {'jellyfin': []}
    )
    monkeypatch.setattr(app_setup.config, 'MEDIASERVER_FIELDS_BY_TYPE', {'jellyfin': []})
    monkeypatch.setattr(app_setup.config, 'MEDIASERVER_CONFIG_KEYS', set())
    monkeypatch.setattr(app_setup.config, 'MUSIC_LIBRARIES', '')
    monkeypatch.setattr(app_setup.setup_manager, '_is_valid_server_config', lambda _cfg: True)
    monkeypatch.setattr(app_setup.setup_manager, 'delete_config_values', MagicMock())
    monkeypatch.setattr(app_setup.setup_manager, 'save_config_values', MagicMock())
    monkeypatch.setattr(app_setup.config, 'refresh_config', MagicMock())

    from tasks.mediaserver import registry

    monkeypatch.setattr(registry, 'save_default_server_settings', MagicMock())
    publish = MagicMock(
        return_value=worker_result,
        side_effect=worker_error,
    )
    schedule = MagicMock(return_value=flask_result)
    monkeypatch.setattr(app_setup.restart_manager, 'publish_restart_request', publish)
    monkeypatch.setattr(app_setup.restart_manager, 'schedule_flask_restart', schedule)

    with app_setup.app.test_request_context(
        '/api/setup', method='POST', json={'config': {'MEDIASERVER_TYPE': 'jellyfin'}}
    ):
        response = app_setup.setup_api()
    return response, publish, schedule


def test_setup_success_requires_worker_ack_and_local_restart(monkeypatch):
    response, publish, schedule = _save_setup(monkeypatch)

    assert response.status_code == 200
    assert response.get_json()['status'] == 'ok'
    assert response.get_json()['worker_restart_acknowledged'] is True
    assert response.get_json()['flask_restart_scheduled'] is True
    publish.assert_called_once_with(
        timeout_seconds=app_setup.restart_manager.CONTROL_ACK_ADVISORY_TIMEOUT_SECONDS
    )
    schedule.assert_called_once_with()


def test_the_setup_save_never_waits_the_full_background_ack_deadline(monkeypatch):
    import restart_manager

    response, publish, _schedule = _save_setup(monkeypatch)

    assert response.status_code == 200
    waited = publish.call_args.kwargs['timeout_seconds']
    assert waited <= 5.0
    assert waited <= restart_manager.CONTROL_ACK_TIMEOUT_SECONDS


def test_setup_reports_partial_with_a_warning_when_workers_do_not_ack(monkeypatch):
    response, _publish, schedule = _save_setup(monkeypatch, worker_result=False)

    body = response.get_json()
    assert response.status_code == 200
    assert body['status'] == 'partial'
    assert body['worker_restart_acknowledged'] is False
    assert 'saved' in body['warning'].lower()
    # The committed config must still be reloaded by the local Flask process.
    schedule.assert_called_once_with()


def test_a_declined_flask_restart_is_an_opt_out_not_a_failed_save(monkeypatch):
    # schedule_flask_restart returns False only for DISABLE_FLASK_RESTART or a
    # non-flask SERVICE_TYPE; a real failure raises. Treating the opt-out as an
    # incomplete restart made those deployments answer 503 on every save.
    response, _publish, _schedule = _save_setup(monkeypatch, flask_result=False)

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['flask_restart_scheduled'] is False
    assert payload['worker_restart_acknowledged'] is True
    assert payload['status'] == 'ok'


def test_setup_publish_exception_is_a_partial_save_not_an_error_response(
    monkeypatch,
):
    response, publish, schedule = _save_setup(
        monkeypatch, worker_error=RuntimeError('listener disappeared')
    )

    assert response.status_code == 200
    assert response.get_json()['worker_restart_acknowledged'] is False
    publish.assert_called_once_with(
        timeout_seconds=app_setup.restart_manager.CONTROL_ACK_ADVISORY_TIMEOUT_SECONDS
    )
    schedule.assert_called_once_with()
