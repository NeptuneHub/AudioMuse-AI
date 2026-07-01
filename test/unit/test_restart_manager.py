# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Guard conditions that decide whether a Flask restart may be scheduled.

Covers restart_manager's gating so only a Flask service with restart enabled
arms the restart timer, using a mocked timer to observe the decision.

Main Features:
* Returns False when the service type is unset or is a worker
* Returns False when the Flask restart flag is disabled (case-insensitive)
* Guards pass only for a Flask service with restart explicitly enabled
"""

from unittest.mock import MagicMock

import pytest

import restart_manager


@pytest.fixture
def mock_timer(monkeypatch):
    timer_cls = MagicMock()
    monkeypatch.setattr(restart_manager.threading, 'Timer', timer_cls)
    return timer_cls


def test_returns_false_when_service_type_unset(monkeypatch, mock_timer):
    monkeypatch.delenv('SERVICE_TYPE', raising=False)
    monkeypatch.delenv('DISABLE_FLASK_RESTART', raising=False)

    assert restart_manager.schedule_flask_restart() is False
    mock_timer.assert_not_called()


def test_returns_false_when_service_type_is_worker(monkeypatch, mock_timer):
    monkeypatch.setenv('SERVICE_TYPE', 'worker')
    monkeypatch.delenv('DISABLE_FLASK_RESTART', raising=False)

    assert restart_manager.schedule_flask_restart() is False
    mock_timer.assert_not_called()


def test_returns_false_when_flask_restart_disabled(monkeypatch, mock_timer):
    monkeypatch.setenv('SERVICE_TYPE', 'flask')
    monkeypatch.setenv('DISABLE_FLASK_RESTART', 'true')

    assert restart_manager.schedule_flask_restart() is False
    mock_timer.assert_not_called()


def test_disable_guard_is_case_insensitive(monkeypatch, mock_timer):
    monkeypatch.setenv('SERVICE_TYPE', 'FLASK')
    monkeypatch.setenv('DISABLE_FLASK_RESTART', 'TRUE')

    assert restart_manager.schedule_flask_restart() is False
    mock_timer.assert_not_called()


def test_guards_pass_for_flask_service_with_restart_enabled(monkeypatch, mock_timer):
    monkeypatch.setenv('SERVICE_TYPE', 'flask')
    monkeypatch.setenv('DISABLE_FLASK_RESTART', 'false')

    assert restart_manager.schedule_flask_restart() is True
    mock_timer.assert_called_once_with(2.5, restart_manager._restart_flask_program)
    mock_timer.return_value.start.assert_called_once_with()
