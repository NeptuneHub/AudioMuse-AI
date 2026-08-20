# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The supervised services, the roles they run, and how a role is dispatched.

One table, because renaming a service used to mean editing six files with silent
failures. A control request names a SERVICE; supervisors map it to a ROLE; the
container maps it to a supervisord program. WORKER_SERVICES is what
restart/stop/start acts on (excludes the restart listener); WORKER_ROLES drives
build_child_env and includes it so a maintenance child never runs Flask's schema
bootstrap. run_role is the single dispatcher, and declare_worker_role the one
SERVICE_TYPE to AUDIOMUSE_ROLE shim - ordering, not configuration: it runs before
import config, stays conditional so Flask keeps its own role, and queue
entrypoints pass force=True. Lives at the repository root so restart_manager can
import it.

Main Features:
* ROLE_OF and QUEUE_OF_ROLE map a supervised SERVICE to its ROLE and, for
  workers, the queue it drains
* declare_worker_role sets AUDIOMUSE_ROLE before config is imported so a
  maintenance or worker child never runs Flask's schema bootstrap
* run_role dispatches a role to Flask, a queue worker, maintenance, or the
  restart listener
* role_from_argv/command_from_argv/serve_flask are what all three native
  launchers used to keep as their own identical copies
"""

import os
import runpy
import sys

import queue_names

ROLE_FLASK = 'flask'
ROLE_WORKER_HIGH = 'worker-high'
ROLE_WORKER_DEFAULT = 'worker-default'
ROLE_MAINTENANCE = 'maintenance'
ROLE_RESTART_LISTENER = 'restart-listener'

SERVICE_FLASK = 'flask'
SERVICE_WORKER_HIGH = 'queue-worker-high'
SERVICE_WORKER_DEFAULT = 'queue-worker-default'
SERVICE_MAINTENANCE = 'queue-maintenance'
SERVICE_RESTART_LISTENER = 'config-restart-listener'

ROLE_OF = {
    SERVICE_FLASK: ROLE_FLASK,
    SERVICE_WORKER_HIGH: ROLE_WORKER_HIGH,
    SERVICE_WORKER_DEFAULT: ROLE_WORKER_DEFAULT,
    SERVICE_MAINTENANCE: ROLE_MAINTENANCE,
    SERVICE_RESTART_LISTENER: ROLE_RESTART_LISTENER,
}

BOOT_ORDER = [
    SERVICE_FLASK,
    SERVICE_WORKER_HIGH,
    SERVICE_WORKER_DEFAULT,
    SERVICE_MAINTENANCE,
    SERVICE_RESTART_LISTENER,
]

FLASK_SERVICES = [SERVICE_FLASK]

WORKER_SERVICES = [SERVICE_WORKER_DEFAULT, SERVICE_WORKER_HIGH, SERVICE_MAINTENANCE]

WORKER_ROLES = frozenset({
    ROLE_WORKER_HIGH,
    ROLE_WORKER_DEFAULT,
    ROLE_MAINTENANCE,
    ROLE_RESTART_LISTENER,
})

QUEUE_OF_ROLE = {
    ROLE_WORKER_HIGH: queue_names.QUEUE_HIGH,
    ROLE_WORKER_DEFAULT: queue_names.QUEUE_DEFAULT,
}

WORKER_MODULE = 'taskqueue.worker'
MAINTENANCE_MODULE = 'taskqueue.maintenance'

ROLE_FLAG_PREFIX = '--role='

FLASK_BIND_HOST = '0.0.0.0'
FLASK_BIND_PORT = 8000
FLASK_THREADS = 8
FLASK_MAX_REQUEST_BODY_BYTES = 6 * 1024 * 1024 * 1024
FLASK_CHANNEL_TIMEOUT_SECONDS = 300

ROLE_ENV = 'AUDIOMUSE_ROLE'
SERVICE_TYPE_ENV = 'SERVICE_TYPE'
WORKER_ENV_VALUE = 'worker'


def role_from_argv():
    for arg in sys.argv[1:]:
        if arg.startswith(ROLE_FLAG_PREFIX):
            return arg.split('=', 1)[1]
    return None


def command_from_argv():
    for arg in sys.argv[1:]:
        if not arg.startswith('-'):
            return arg
    return None


def serve_flask():
    import waitress
    import app as app_module

    waitress.serve(
        app_module.app,
        host=FLASK_BIND_HOST,
        port=FLASK_BIND_PORT,
        threads=FLASK_THREADS,
        max_request_body_size=FLASK_MAX_REQUEST_BODY_BYTES,
        channel_timeout=FLASK_CHANNEL_TIMEOUT_SECONDS,
    )


def declare_worker_role(force=False):
    if force:
        os.environ[ROLE_ENV] = WORKER_ENV_VALUE
        return True
    if os.environ.get(SERVICE_TYPE_ENV, '').lower() != WORKER_ENV_VALUE:
        return False
    os.environ.setdefault(ROLE_ENV, WORKER_ENV_VALUE)
    return True


def run_role(role, run_flask):
    sys.argv = [arg for arg in sys.argv if not arg.startswith(ROLE_FLAG_PREFIX)]
    if role == ROLE_FLASK:
        run_flask()
        return
    queue = QUEUE_OF_ROLE.get(role)
    if queue is not None:
        sys.argv = [sys.argv[0], '--queue', queue]
        runpy.run_module(WORKER_MODULE, run_name='__main__')
        return
    if role == ROLE_MAINTENANCE:
        runpy.run_module(MAINTENANCE_MODULE, run_name='__main__')
        return
    if role == ROLE_RESTART_LISTENER:
        from taskqueue import control

        control.main()
        return
    raise SystemExit(f"Unknown role: {role}")
