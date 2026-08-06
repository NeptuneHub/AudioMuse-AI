# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Stopping, starting and restarting the supervised Flask and worker processes.

The "how do I reach the other container" half now lives in ``taskqueue.control``,
which broadcasts over Postgres and collects durable acknowledgements. What is
left here is the local half: actually driving supervisorctl (containers) or the
control socket (native builds), plus the delayed self-restart Flask arms for
itself.

Main Features:
* ``publish_*`` delegate to ``taskqueue.control`` and keep their boolean contract.
* supervisorctl-driven actions over the known Flask and worker service names.
* ``run_supervisorctl_detail`` returns WHY an action failed, so the restore log
  can say more than "did not confirm it stopped".
* On native builds (control socket/host:port set), dispatches there instead of supervisorctl.
"""

import json
import logging
import os
import socket
import subprocess
import threading

import config
import service_roles

SUPERVISORCTL_CMD = os.environ.get('SUPERVISORCTL_CMD', '/usr/bin/supervisorctl')
SUPERVISOR_CONF = os.environ.get('SUPERVISOR_CONF', '/etc/supervisor/conf.d/supervisord.conf')
logger = logging.getLogger(__name__)

CONTROL_ACK_ADVISORY_TIMEOUT_SECONDS = config.QUEUE_CONTROL_ADVISORY_TIMEOUT_SECONDS

FLASK_SERVICE = service_roles.FLASK_SERVICES
WORKER_SERVICES = service_roles.WORKER_SERVICES


def new_control_request_id():
    from taskqueue.control import new_control_request_id as _new_id

    return _new_id()


def get_control_request_result(action, request_id):
    from taskqueue.control import get_control_request_result as _result

    if not _action_matches(request_id, action):
        return False
    return _result(request_id)


def _action_matches(request_id, action):
    try:
        from database import connect_raw

        conn = connect_raw()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT details FROM task_status WHERE task_id = %s", (request_id,)
                )
                row = cur.fetchone()
            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                logger.debug("Action-check connection close failed", exc_info=True)
        if row is None or not row[0]:
            return True
        try:
            recorded = json.loads(row[0]).get('action')
        except (TypeError, ValueError):
            return True
        if recorded is not None and recorded != action:
            logger.error(
                'Control request ID %s belongs to action %r, not %r',
                request_id, recorded, action,
            )
            return False
        return True
    except Exception:
        logger.exception(
            "Could not validate the action of control request %s; assuming it matches",
            request_id,
        )
        return True


def publish_control_request(action, request_id=None, timeout_seconds=None):
    from taskqueue.control import publish_control_request as _publish

    return _publish(action, request_id=request_id, timeout_seconds=timeout_seconds)


def publish_restart_request(request_id=None, timeout_seconds=None):
    from taskqueue import control

    return publish_control_request(control.ACTION_RESTART, request_id, timeout_seconds)


def publish_stop_request(request_id=None, timeout_seconds=None):
    from taskqueue import control

    return publish_control_request(control.ACTION_STOP, request_id, timeout_seconds)


def publish_start_request(request_id=None, timeout_seconds=None):
    from taskqueue import control

    return publish_control_request(control.ACTION_START, request_id, timeout_seconds)


def publish_plugin_sync_request(request_id=None, timeout_seconds=None):
    from taskqueue import control

    return publish_control_request(control.ACTION_PLUGIN_SYNC, request_id, timeout_seconds)


CONTROL_IPC_TIMEOUT_SECONDS = config.CONTROL_IPC_TIMEOUT_SECONDS


def _control_endpoint():
    host = config.AUDIOMUSE_CONTROL_HOST
    port = config.AUDIOMUSE_CONTROL_PORT
    if host and port:
        if not str(port).isdigit():
            logger.error('Invalid AUDIOMUSE_CONTROL_PORT %r; expected an integer', port)
            return None
        return socket.AF_INET, (str(host), int(port)), f'{host}:{port}'
    if config.AUDIOMUSE_CONTROL_SOCKET:
        return socket.AF_UNIX, config.AUDIOMUSE_CONTROL_SOCKET, config.AUDIOMUSE_CONTROL_SOCKET
    return None


def _use_control_ipc():
    return _control_endpoint() is not None


def _send_control(arguments):
    if not arguments:
        return False

    endpoint = _control_endpoint()
    if endpoint is None:
        logger.error(
            'Neither AUDIOMUSE_CONTROL_SOCKET nor AUDIOMUSE_CONTROL_HOST/PORT set; cannot dispatch %s',
            arguments,
        )
        return False
    family, address, label = endpoint

    payload = json.dumps({'action': arguments[0], 'services': list(arguments[1:])}).encode('utf-8')
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(CONTROL_IPC_TIMEOUT_SECONDS)
            sock.connect(address)
            sock.sendall(payload + b'\n')
            response = sock.recv(1024).strip()
    except Exception:
        logger.exception('Failed to send control command %s to %s', arguments, label)
        return False
    if response == b'ok':
        logger.info('Control command succeeded: %s', arguments)
        return True
    logger.error('Control server rejected %s: %s', arguments, response)
    return False


def run_supervisorctl_detail(arguments):
    if _use_control_ipc():
        ok = _send_control(arguments)
        return ok, ('control server accepted' if ok else 'control server rejected the command')
    cmd = [SUPERVISORCTL_CMD, '-c', SUPERVISOR_CONF] + arguments
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        detail = stderr or stdout or '(no output)'
        if result.returncode != 0:
            lines = [line.strip().lower() for line in (stdout + '\n' + stderr).splitlines()
                     if line.strip()]
            action = arguments[0] if arguments else ''
            idempotent = (
                action == 'start' and lines and all('started' in line for line in lines)
            ) or (
                action == 'stop' and lines
                and all(('stopped' in line or 'not running' in line) for line in lines)
            )
            if idempotent:
                logger.info(
                    'supervisorctl %s was already satisfied: %s', action, stdout or stderr
                )
                return True, detail
            logger.error('supervisorctl failed (%s): %s', result.returncode, detail)
            return False, f'exit {result.returncode}: {detail}'
        logger.info('supervisorctl succeeded: %s', stdout)
        return True, detail
    except FileNotFoundError:
        logger.exception('supervisorctl command not found at %s', SUPERVISORCTL_CMD)
        return False, f'supervisorctl not found at {SUPERVISORCTL_CMD}'
    except subprocess.TimeoutExpired:
        logger.exception('supervisorctl timed out after 30s: %s', cmd)
        return False, 'supervisorctl timed out after 30s'
    except Exception as exc:
        logger.exception('Failed to run supervisorctl command: %s', cmd)
        return False, f'{exc.__class__.__name__}: {exc}'


def _run_supervisorctl(arguments):
    ok, _detail = run_supervisorctl_detail(arguments)
    return ok


def stop_local_flask_service_detail():
    logger.info('Stopping supervised Flask service')
    return run_supervisorctl_detail(['stop'] + FLASK_SERVICE)


def start_local_flask_service():
    logger.info('Starting supervised Flask service')
    return _run_supervisorctl(['start'] + FLASK_SERVICE)


def stop_supervisor_workers():
    logger.info('Stopping supervised worker services: %s', WORKER_SERVICES)
    return _run_supervisorctl(['stop'] + WORKER_SERVICES)


def start_supervisor_workers():
    logger.info('Starting supervised worker services: %s', WORKER_SERVICES)
    return _run_supervisorctl(['start'] + WORKER_SERVICES)


def _spawn_supervisorctl(arguments):
    if _use_control_ipc():
        return _send_control(arguments)
    cmd = [SUPERVISORCTL_CMD, '-c', SUPERVISOR_CONF] + arguments
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info('Spawned detached supervisorctl: %s', ' '.join(cmd))
        return True
    except FileNotFoundError:
        logger.exception('supervisorctl command not found at %s', SUPERVISORCTL_CMD)
        return False
    except Exception:
        logger.exception('Failed to spawn supervisorctl command: %s', cmd)
        return False


def _restart_flask_program():
    logger.info('Restarting supervised Flask program via supervisorctl')
    return _spawn_supervisorctl(['restart'] + FLASK_SERVICE)


def schedule_flask_restart(delay_seconds=2.5):
    if os.environ.get('SERVICE_TYPE', '').lower() != 'flask':
        return False

    if os.environ.get('DISABLE_FLASK_RESTART', 'false').lower() == 'true':
        return False

    timer = threading.Timer(delay_seconds, _restart_flask_program)
    timer.daemon = True
    timer.start()
    return True


def restart_supervisor_workers():
    if os.environ.get('SERVICE_TYPE', '').lower() != 'worker':
        logger.info('SERVICE_TYPE is not worker; skipping supervised worker restart')
        return True

    logger.info('Restarting supervised worker programs via supervisorctl')
    return _run_supervisorctl(['restart'] + WORKER_SERVICES)
