# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The web process of the end-to-end stack: app.py under gunicorn, as in production.

The command line is the one in deployment/supervisord.conf plus a short
graceful timeout, run from the repo root so gunicorn loads ./gunicorn.conf.py on
its own. The port is 8000 because config.FLASK_BIND_PORT is not tunable.
Readiness is GET /api/health (exempt from the setup and auth barrier) followed
by GET /api/config, whose 403 would mean the environment did not satisfy the
media-server configuration and the wizard is still gating every request.

Main Features:
* FlaskServer.start / wait_ready / stop around one ManagedProcess
* a 403 on /api/config fails the stack immediately with the log tail
"""

import sys

import requests

from .errors import StackError
from .paths import REPO_ROOT
from .processes import ManagedProcess, wait_until

FLASK_PORT = 8000
GUNICORN_ARGS = [
    '--bind', f'0.0.0.0:{FLASK_PORT}',
    '--workers', '1',
    '--threads', '4',
    '--worker-class', 'gthread',
    '--keep-alive', '5',
    '--timeout', '300',
    '--graceful-timeout', '10',
    '--capture-output',
    '--error-logfile', '-',
    '--access-logfile', '-',
    'app:app',
]


class FlaskServer:
    def __init__(self, env, log_path, python=None):
        self.base_url = f'http://127.0.0.1:{FLASK_PORT}'
        argv = [python or sys.executable, '-m', 'gunicorn', *GUNICORN_ARGS]
        self.process = ManagedProcess('flask', argv, env, REPO_ROOT, log_path)

    def start(self):
        self.process.start()

    def stop(self):
        self.process.stop(15)

    def running(self):
        return self.process.running()

    def _healthy(self):
        response = requests.get(self.base_url + '/api/health', timeout=5)
        return response.status_code == 200 and response.json().get('status') == 'ok'

    def wait_ready(self, timeout=240):
        wait_until(
            self._healthy, timeout, 'Flask /api/health',
            dead=lambda: not self.process.running(), detail=self.process.describe,
        )
        response = requests.get(self.base_url + '/api/config', timeout=15)
        if response.status_code == 403:
            raise StackError(
                'Flask answers 403 on /api/config: setup is still required, so the '
                'navidrome environment did not satisfy SetupManager._is_valid_server_config. '
                + self.process.describe()
            )
        if response.status_code != 200:
            raise StackError(
                f'Flask answered {response.status_code} on /api/config: {response.text[:300]}. '
                + self.process.describe()
            )
