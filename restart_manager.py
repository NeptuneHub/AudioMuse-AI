# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Supervisor control plane for restarting Flask and RQ worker processes.

Publishes control requests onto the Redis restart channel that
``restart_listener`` consumes, and provides the supervisorctl-backed helpers
that actually stop, start, and restart the managed services.

Main Features:
* ``publish_*`` helpers broadcast restart/stop/start requests to workers.
* supervisorctl-driven actions over the known Flask and worker service names.
* On native builds (control socket/host:port set), dispatches there instead of supervisorctl.
"""

import json
import logging
import os
import socket
import subprocess
import threading

from redis import Redis
import config

RESTART_CHANNEL = os.environ.get('AUDIO_MUSE_CONFIG_RESTART_CHANNEL', 'audiomuse:config_restart')
SUPERVISORCTL_CMD = os.environ.get('SUPERVISORCTL_CMD', '/usr/bin/supervisorctl')
SUPERVISOR_CONF = os.environ.get('SUPERVISOR_CONF', '/etc/supervisor/conf.d/supervisord.conf')
logger = logging.getLogger(__name__)

FLASK_SERVICE = ['flask']
WORKER_SERVICES = ['rq-worker-default', 'rq-worker-high', 'rq-janitor']
ALL_SUPERVISOR_SERVICES = FLASK_SERVICE + WORKER_SERVICES


def publish_control_request(action):
    try:
        redis_conn = Redis.from_url(
            config.REDIS_URL,
            socket_connect_timeout=5,
            socket_timeout=5,
            health_check_interval=30,
            retry_on_timeout=True,
            decode_responses=True,
        )
        redis_conn.publish(RESTART_CHANNEL, action)
        return True
    except Exception as exc:
        logger.exception('Could not publish %s request to Redis: %s', action, exc)
        return False


def publish_restart_request():
    return publish_control_request('restart')


def publish_stop_request():
    return publish_control_request('stop')


def publish_start_request():
    return publish_control_request('start')


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
            sock.settimeout(15)
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


def _run_supervisorctl(arguments):
    if _use_control_ipc():
        return _send_control(arguments)
    cmd = [SUPERVISORCTL_CMD, '-c', SUPERVISOR_CONF] + arguments
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode != 0:
            logger.error('supervisorctl failed (%s): %s', result.returncode, stderr or stdout)
            return False
        logger.info('supervisorctl succeeded: %s', stdout)
        return True
    except FileNotFoundError:
        logger.exception('supervisorctl command not found at %s', SUPERVISORCTL_CMD)
        return False
    except subprocess.TimeoutExpired:
        logger.exception('supervisorctl timed out after 30s: %s', cmd)
        return False
    except Exception:
        logger.exception('Failed to run supervisorctl command: %s', cmd)
        return False


def stop_local_flask_service():
    logger.info('Stopping supervised Flask service')
    return _run_supervisorctl(['stop'] + FLASK_SERVICE)


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
    return _spawn_supervisorctl(['restart', 'flask'])


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
