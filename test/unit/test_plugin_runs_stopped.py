# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Uninstall and restart stop plugin runs first.

An uninstalled plugin's run kept going, and a running plugin run held a database
lock that a restarting Flask waited for. Uninstall now stops that plugin's runs
before anything is removed, and Apply stops every plugin run before the restart
is published.

Main Features:
* Uninstall stops the plugin's runs, scheduled and queued with enqueue, before removing it
* A stop that fails refuses the uninstall, so nothing keeps running for a removed plugin
* Apply stops every plugin run before the workers are told to restart
"""

from unittest.mock import MagicMock

from flask import Flask

import plugin.blueprint as blueprint


def _call(view, json_body=None):
    with Flask(__name__).test_request_context(method='POST', json=json_body or {}):
        return view()


class TestUninstall:
    def test_the_plugin_runs_are_stopped_before_it_is_removed(self, monkeypatch):
        order = []
        monkeypatch.setattr(
            blueprint, 'stop_plugin_tasks',
            lambda prefixes, reason: order.append(('stop', prefixes)),
        )
        monkeypatch.setattr(
            blueprint.plugin_manager, 'uninstall',
            lambda plugin_id, purge_data=False: order.append(('uninstall', plugin_id)),
        )

        response = _call(blueprint.api_uninstall, {'id': 'gone'})

        assert response.get_json()['status'] == 'ok'
        assert order == [
            ('stop', ('plugin.gone.', 'plugin.audiomuse_plugins.gone.')),
            ('uninstall', 'gone'),
        ], 'its scheduled and enqueued runs are stopped before the plugin is removed'

    def test_a_failed_stop_refuses_the_uninstall(self, monkeypatch):
        uninstall = MagicMock()
        monkeypatch.setattr(
            blueprint, 'stop_plugin_tasks', MagicMock(side_effect=RuntimeError('database down'))
        )
        monkeypatch.setattr(blueprint.plugin_manager, 'uninstall', uninstall)

        _response, status = _call(blueprint.api_uninstall, {'id': 'gone'})

        assert status >= 500
        assert not uninstall.called, 'a plugin whose runs could not be stopped stays installed'


class TestApply:
    def test_every_plugin_run_is_stopped_before_the_restart(self, monkeypatch):
        publish = MagicMock(return_value=True)
        published_before_stop = []
        monkeypatch.setattr(blueprint.restart_manager, 'publish_restart_request', publish)
        monkeypatch.setattr(
            blueprint.restart_manager, 'schedule_flask_restart', MagicMock(return_value=True)
        )
        monkeypatch.setattr(
            blueprint, 'stop_plugin_tasks_before_restart',
            lambda: published_before_stop.append(publish.called),
        )

        response = _call(blueprint.api_apply)

        assert response.get_json()['status'] == 'ok'
        assert published_before_stop == [False], (
            'a running plugin run must be stopped before the restart, or the new Flask '
            'waits for its database lock'
        )
