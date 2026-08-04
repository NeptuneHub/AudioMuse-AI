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
* ``publish_*`` helpers broadcast restart/stop/start and plugin-sync requests to workers.
* The ACK wait runs on the caller's thread; request handlers whose restart is only
  advisory pass ``CONTROL_ACK_ADVISORY_TIMEOUT_SECONDS`` so they cannot occupy a
  gunicorn thread for the full background deadline.
* supervisorctl-driven actions over the known Flask and worker service names.
* On native builds (control socket/host:port set), dispatches there instead of supervisorctl.
"""

import json
import logging
import os
import socket
import subprocess
import threading
import time
import uuid

import config
from taskqueue import new_redis_connection

RESTART_CHANNEL = os.environ.get('AUDIO_MUSE_CONFIG_RESTART_CHANNEL', 'audiomuse:config_restart')
SUPERVISORCTL_CMD = os.environ.get('SUPERVISORCTL_CMD', '/usr/bin/supervisorctl')
SUPERVISOR_CONF = os.environ.get('SUPERVISOR_CONF', '/etc/supervisor/conf.d/supervisord.conf')
logger = logging.getLogger(__name__)

CONTROL_REQUEST_PREFIX = 'audiomuse:worker_control:request:'
CONTROL_RESULT_PREFIX = 'audiomuse:worker_control:result:'
CONTROL_RESULT_TTL_SECONDS = max(
    60, int(os.environ.get('AUDIO_MUSE_CONTROL_RESULT_TTL_SECONDS', '86400'))
)
CONTROL_ACK_TIMEOUT_SECONDS = max(
    5.0, float(os.environ.get('AUDIO_MUSE_CONTROL_ACK_TIMEOUT_SECONDS', '15'))
)
CONTROL_ACK_ADVISORY_TIMEOUT_SECONDS = min(5.0, CONTROL_ACK_TIMEOUT_SECONDS)
_configured_ipc_timeout = float(
    os.environ.get('AUDIO_MUSE_CONTROL_IPC_TIMEOUT_SECONDS', CONTROL_ACK_TIMEOUT_SECONDS - 2)
)
CONTROL_IPC_TIMEOUT_SECONDS = max(
    3.0, min(_configured_ipc_timeout, CONTROL_ACK_TIMEOUT_SECONDS - 2)
)
CONTROL_RETRY_LEASE_SECONDS = max(
    CONTROL_ACK_TIMEOUT_SECONDS,
    float(os.environ.get('AUDIO_MUSE_CONTROL_RETRY_LEASE_SECONDS', '90')),
)
CONTROL_MAX_DELIVERY_ATTEMPTS = max(
    1, int(os.environ.get('AUDIO_MUSE_CONTROL_MAX_DELIVERY_ATTEMPTS', '3'))
)

_ACK_POLL_START_SECONDS = 0.1

_ACK_POLL_MAX_SECONDS = 2.0

_REGISTER_CONTROL_REQUEST_LUA = """
local now = tonumber(redis.call('TIME')[1])
if redis.call('EXISTS', KEYS[1]) == 1 then
    local recorded_action = redis.call('HGET', KEYS[1], 'action')
    if recorded_action ~= ARGV[3] then
        return -2
    end
    local outcome = redis.call('HGET', KEYS[1], 'outcome')
    if outcome == '1' then
        return -4
    elseif outcome == '0' then
        return -3
    end
    local previous_expected = tonumber(redis.call('HGET', KEYS[1], 'expected') or '-1')
    local lease_until = tonumber(redis.call('HGET', KEYS[1], 'lease_until') or '0')
    if previous_expected > 0 and now < lease_until then
        return previous_expected
    end
    local delivery_attempts = tonumber(redis.call('HGET', KEYS[1], 'delivery_attempts') or '0')
    if previous_expected > 0 and delivery_attempts >= tonumber(ARGV[6]) then
        redis.call('HSET', KEYS[1], 'outcome', '0')
        redis.call('EXPIRE', KEYS[1], ARGV[4])
        return -3
    end

    local next_attempt = tonumber(redis.call('HGET', KEYS[1], 'attempt') or '1') + 1
    redis.call('HSET', KEYS[1], 'attempt', next_attempt)
    local retry_expected = redis.call('PUBLISH', ARGV[1], ARGV[2])
    local expected_floor = tonumber(redis.call('HGET', KEYS[1], 'expected_floor') or '0')
    if retry_expected > expected_floor then
        expected_floor = retry_expected
    end
    if previous_expected == 0 and retry_expected > 0 then
        delivery_attempts = 1
    elseif previous_expected > 0 then
        delivery_attempts = delivery_attempts + 1
    end
    redis.call('HSET', KEYS[1],
        'expected', expected_floor,
        'expected_floor', expected_floor,
        'delivery_attempts', delivery_attempts,
        'lease_until', now + tonumber(ARGV[5]))
    redis.call('EXPIRE', KEYS[1], ARGV[4])
    return expected_floor
end
redis.call('HSET', KEYS[1], 'action', ARGV[3], 'attempt', 1)
redis.call('EXPIRE', KEYS[1], ARGV[4])
local expected = redis.call('PUBLISH', ARGV[1], ARGV[2])
local delivery_attempts = 0
if expected > 0 then
    delivery_attempts = 1
end
redis.call('HSET', KEYS[1],
    'expected', expected,
    'expected_floor', expected,
    'delivery_attempts', delivery_attempts,
    'lease_until', now + tonumber(ARGV[5]))
return expected
"""

FLASK_SERVICE = ['flask']
WORKER_SERVICES = ['rq-worker-default', 'rq-worker-high', 'rq-janitor']


def new_control_request_id():
    return uuid.uuid4().hex


def control_request_key(request_id):
    return f'{CONTROL_REQUEST_PREFIX}{{{request_id}}}:meta'


def control_result_key(request_id):
    return f'{CONTROL_RESULT_PREFIX}{{{request_id}}}:listeners'


def control_attempt_result_key(request_id, attempt):
    return f'{CONTROL_RESULT_PREFIX}{{{request_id}}}:attempt:{attempt}'


def _hash_value(values, key):
    return values.get(key, values.get(key.encode('utf-8')))


def get_control_request_result(action, request_id, redis_conn=None):
    owns_connection = redis_conn is None
    try:
        if owns_connection:
            redis_conn = new_redis_connection(
                socket_connect_timeout=5,
                socket_timeout=5,
                decode_responses=True,
            )

        request = redis_conn.hgetall(control_request_key(request_id))
        if not request:
            return None
        recorded_action = _hash_value(request, 'action')
        if isinstance(recorded_action, bytes):
            recorded_action = recorded_action.decode('utf-8', errors='replace')
        if recorded_action != action:
            logger.error(
                'Control request ID %s belongs to action %r, not %r',
                request_id,
                recorded_action,
                action,
            )
            return False

        outcome = _hash_value(request, 'outcome')
        if isinstance(outcome, bytes):
            outcome = outcome.decode('utf-8', errors='replace')
        if outcome == '1':
            return True
        if outcome == '0':
            return False

        raw_expected = _hash_value(request, 'expected')
        if raw_expected is None:
            return None
        expected = int(raw_expected)
        if expected <= 0:
            return False

        raw_attempt = _hash_value(request, 'attempt')
        if raw_attempt is None:
            return None
        attempt = int(raw_attempt)
        results = redis_conn.hgetall(control_attempt_result_key(request_id, attempt))
        if len(results) < expected:
            return None

        completed_ok = True
        for raw_result in results.values():
            if isinstance(raw_result, bytes):
                raw_result = raw_result.decode('utf-8', errors='replace')
            try:
                result = json.loads(raw_result)
            except (TypeError, ValueError):
                logger.exception(
                    'Invalid ACK for control request %s: %r', request_id, raw_result
                )
                completed_ok = False
                break
            if result.get('action') != action or result.get('ok') is not True:
                completed_ok = False
                break
        try:
            redis_conn.hset(
                control_request_key(request_id),
                'outcome',
                '1' if completed_ok else '0',
            )
            redis_conn.expire(control_request_key(request_id), CONTROL_RESULT_TTL_SECONDS)
        except Exception:
            logger.warning(
                'Could not cache outcome for control request %s', request_id, exc_info=True
            )
        return completed_ok
    except Exception:
        logger.exception('Could not read %s request %s result from Redis', action, request_id)
        return None
    finally:
        if owns_connection and redis_conn is not None:
            try:
                redis_conn.close()
            except Exception:
                pass


def publish_control_request(action, request_id=None, timeout_seconds=CONTROL_ACK_TIMEOUT_SECONDS):
    supplied_request_id = request_id is not None
    request_id = request_id or new_control_request_id()
    redis_conn = None
    try:
        redis_conn = new_redis_connection(
            socket_connect_timeout=5,
            socket_timeout=5,
            decode_responses=True,
        )
        if supplied_request_id:
            existing_result = get_control_request_result(
                action, request_id, redis_conn=redis_conn
            )
            if existing_result is True:
                return True
            if existing_result is False:
                request = redis_conn.hgetall(control_request_key(request_id))
                recorded_action = _hash_value(request, 'action') if request else None
                if isinstance(recorded_action, bytes):
                    recorded_action = recorded_action.decode('utf-8', errors='replace')
                raw_expected = _hash_value(request, 'expected') if request else None
                if recorded_action == action and raw_expected is not None and int(raw_expected) > 0:
                    return False

        payload = json.dumps({'action': action, 'request_id': request_id})
        expected = int(redis_conn.eval(
            _REGISTER_CONTROL_REQUEST_LUA,
            1,
            control_request_key(request_id),
            RESTART_CHANNEL,
            payload,
            action,
            CONTROL_RESULT_TTL_SECONDS,
            CONTROL_RETRY_LEASE_SECONDS,
            CONTROL_MAX_DELIVERY_ATTEMPTS,
        ))
        if expected == -2:
            logger.error('Control request ID %s was reused for a different action', request_id)
            return False
        if expected == -4:
            return True
        if expected == -3:
            logger.error('Control request %s has a durable negative outcome', request_id)
            return False
        if expected <= 0:
            logger.error(
                'Could not deliver %s request %s: no worker restart listener is subscribed',
                action,
                request_id,
            )
            return False

        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        poll_interval = _ACK_POLL_START_SECONDS
        while True:
            result = get_control_request_result(action, request_id, redis_conn=redis_conn)
            if result is not None:
                return result
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error(
                    'Timed out waiting for %d worker ACK(s) for %s request %s',
                    expected,
                    action,
                    request_id,
                )
                return False
            poll_interval = min(poll_interval * 2, _ACK_POLL_MAX_SECONDS)
            time.sleep(min(poll_interval, remaining))
    except Exception:
        logger.exception('Could not publish or confirm %s request %s', action, request_id)
        return False
    finally:
        if redis_conn is not None:
            try:
                redis_conn.close()
            except Exception:
                pass


def publish_restart_request(request_id=None, timeout_seconds=CONTROL_ACK_TIMEOUT_SECONDS):
    return publish_control_request('restart', request_id=request_id, timeout_seconds=timeout_seconds)


def publish_plugin_sync_request(request_id=None, timeout_seconds=CONTROL_ACK_TIMEOUT_SECONDS):
    return publish_control_request('plugin-sync', request_id=request_id, timeout_seconds=timeout_seconds)


def publish_stop_request(request_id=None, timeout_seconds=CONTROL_ACK_TIMEOUT_SECONDS):
    return publish_control_request('stop', request_id=request_id, timeout_seconds=timeout_seconds)


def publish_start_request(request_id=None, timeout_seconds=CONTROL_ACK_TIMEOUT_SECONDS):
    return publish_control_request('start', request_id=request_id, timeout_seconds=timeout_seconds)


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


def _run_supervisorctl(arguments):
    if _use_control_ipc():
        return _send_control(arguments)
    cmd = [SUPERVISORCTL_CMD, '-c', SUPERVISOR_CONF] + arguments
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
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
                return True
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
