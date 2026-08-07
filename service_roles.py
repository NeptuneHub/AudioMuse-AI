# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The supervised services, the roles they run, and how a role is dispatched.

One table, because renaming a service used to mean editing six files and the
failure was silent every time. A control request names a SERVICE
(``queue-worker-high``); the native supervisors map that to a ROLE
(``worker-high``) and re-exec themselves with ``--role=``; the container maps it
straight to a supervisord program. When ``restart_manager.WORKER_SERVICES`` and
the native ``ROLE_OF`` drifted apart, ``dispatch_control`` simply returned False
for an unknown name - so the tray's restart, the setup wizard's save and the
backup restore all reported "the workers did not restart" with nothing in any
log, on exactly the platforms whose copy had been missed.

The lists are deliberately not the same set, and both distinctions matter.
``WORKER_SERVICES`` is what a restart/stop/start control request acts on and
excludes the restart listener, because something has to still be listening after
a stop in order to hear the start that brings the workers back. ``WORKER_ROLES``
is what ``build_child_env`` treats as a worker for environment purposes and DOES
include the listener, because ``taskqueue.control.main`` idles out unless
``SERVICE_TYPE=worker``, and because a maintenance child that is handed the
Flask environment runs config's schema-bootstrapping branch in a process that
must never bootstrap the schema.

``run_role`` is here rather than copied into three launchers for the same
reason: it was byte-identical in all of them, including the ``sys.argv`` rewrite
that hands the queue name to ``taskqueue.worker``. Only the Flask server differs
per platform, so it arrives as a callable.

``declare_worker_role`` is here for that same reason and for one more. Every
queue-side entrypoint had grown its own spelling of the ``SERVICE_TYPE`` to
``AUDIOMUSE_ROLE`` shim - a conditional two-liner in two of them, a bare
assignment in a third, a ``setdefault`` buried in a function in a fourth - and
the drift was invisible because nothing here fails loudly. It is ORDERING, not
configuration: ``config`` decides at IMPORT time whether it is running
Flask-side and bootstraps the schema when it decides that it is, so an
entrypoint that imports config before declaring its role runs Flask's DDL from
inside a worker container. The call therefore stays ABOVE the config import,
which is why the imports below it carry ``noqa: E402``.

It stays CONDITIONAL by default because Flask imports ``taskqueue.control`` in
order to publish a control request and must keep its own role. A process that IS
a queue entrypoint - ``python -m taskqueue.worker`` is nothing else, whatever
``SERVICE_TYPE`` says or fails to say - passes ``force=True`` instead, so a bare
metal run with no ``SERVICE_TYPE`` in the environment is still a worker.

This sits at the repository root rather than under ``native-build/native_common``
because ``restart_manager`` is one of its consumers and runs in the container,
where ``native-build/`` is not on ``sys.path`` at all. The native supervisors and
launchers reach the root perfectly well - they already run ``taskqueue.worker``
through it - so root is the only directory both halves share.

Main Features:
* ``ROLE_OF`` / ``BOOT_ORDER`` drive the three native supervisors
* ``WORKER_SERVICES`` / ``FLASK_SERVICES`` drive restart_manager's supervisorctl calls
* ``WORKER_ROLES`` drives each platform's ``build_child_env``
* ``run_role`` is the one role dispatcher, taking the platform's Flask runner
* ``declare_worker_role`` is the one SERVICE_TYPE to AUDIOMUSE_ROLE shim, and it
  runs before config is imported in every queue-side entrypoint
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

ROLE_ENV = 'AUDIOMUSE_ROLE'
SERVICE_TYPE_ENV = 'SERVICE_TYPE'
WORKER_ENV_VALUE = 'worker'


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
