# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The plugin manager's read side on an instance without plugins.

Installing needs the catalog repository on the network and applying publishes
a worker restart, so only the listing endpoints are driven here.

Main Features:
* the installed list is empty with no restart pending
* the repository list carries the default catalog
* settings of an unknown plugin are 404 for GET and POST
"""

import pytest

pytestmark = pytest.mark.e2e


def test_installed_list_is_empty(stack, api):
    body = api.json('GET', '/api/plugins/installed')
    assert body['plugins'] == []
    assert not body.get('restart_pending')
    assert isinstance(body.get('pip_supported'), bool)


def test_repositories_have_a_default(stack, api):
    body = api.json('GET', '/api/plugins/repos')
    assert isinstance(body.get('repos'), list) and body['repos']
    assert body.get('default')


def test_unknown_plugin_settings(stack, api):
    assert api.get('/api/plugins/settings/nope').status_code == 404
    assert api.post('/api/plugins/settings/nope', json={}).status_code == 404
