# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Long-running worker-side listener for restart/stop/start control requests.

Subscribes to the Redis restart channel and, on worker containers only, drives
the supervisor actions defined in ``restart_manager`` in response to published
``restart``/``stop``/``start`` messages, reconnecting on failure. Also handles a
``plugin-sync`` signal, pre-installing plugin code and pip dependencies into this
worker's own volume so the apply restart reloads fast.

Main Features:
* Redis pub/sub loop with automatic reconnect, keepalive and health checks
  via the shared taskqueue connection factory.
* Only worker-role processes subscribe, so a Redis publish count identifies
  listeners that can actually perform the requested supervisor action.
* Records a durable, per-listener result after supervisor actions complete;
  plugin-sync records acceptance once its long-running background thread starts.
"""

import json
import logging
import os
import socket
import threading
import time

from app_logging import configure_logging
from taskqueue import new_redis_connection
from restart_manager import (
    CONTROL_RESULT_TTL_SECONDS,
    RESTART_CHANNEL,
    control_attempt_result_key,
    control_request_key,
    control_result_key,
    restart_supervisor_workers,
    stop_supervisor_workers,
    start_supervisor_workers,
)

logger = logging.getLogger(__name__)
configure_logging()

_PERSIST_RESULT_LUA = """
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('HSET', KEYS[2], ARGV[1], ARGV[2])
redis.call('EXPIRE', KEYS[1], ARGV[3])
redis.call('EXPIRE', KEYS[2], ARGV[3])
redis.call('EXPIRE', KEYS[3], ARGV[3])
return 1
"""

try:
    from plugin.manager import worker_presync
except Exception:
    worker_presync = None
    logger.exception('plugin.manager import failed; plugin-sync signals will be ignored')


def _dispatch_plugin_sync():
    if worker_presync is None:
        logger.warning('plugin-sync received but the plugin subsystem is unavailable')
        return False

    def _run():
        try:
            worker_presync()
        except Exception:
            logger.exception('Plugin-sync handling on this worker failed')

    try:
        threading.Thread(target=_run, name='plugin-sync', daemon=True).start()
    except Exception:
        logger.exception('Could not start plugin-sync background thread')
        return False
    return True


def listener_id():
    hostname = socket.gethostname()
    configured = os.environ.get('AUDIO_MUSE_WORKER_ID') or os.environ.get('WORKER_ID')
    return f'{configured}:{hostname}' if configured else hostname


def _execute_action(action):
    if action == 'restart':
        logger.info('Restart request received, restarting worker processes...')
        return bool(restart_supervisor_workers())
    if action == 'stop':
        logger.info('Stop request received, stopping worker processes...')
        return bool(stop_supervisor_workers())
    if action == 'start':
        logger.info('Start request received, starting worker processes...')
        return bool(start_supervisor_workers())
    if action == 'plugin-sync':
        logger.info('Plugin sync request received; syncing plugins for this worker...')
        return _dispatch_plugin_sync()
    logger.error('Unknown worker control action %r', action)
    return False


def _persist_pending_result(redis_conn, pending):
    redis_conn.eval(
        _PERSIST_RESULT_LUA,
        3,
        pending['canonical_result_key'],
        pending['attempt_result_key'],
        pending['request_key'],
        pending['worker_id'],
        pending['result'],
        CONTROL_RESULT_TTL_SECONDS,
    )


def flush_pending_results(redis_conn, pending_results):
    for pending_id, pending in tuple(pending_results.items()):
        _persist_pending_result(redis_conn, pending)
        del pending_results[pending_id]
        logger.info(
            'Persisted delayed %s ACK for request %s after Redis reconnected',
            pending['action'],
            pending['request_id'],
        )


def handle_control_message(redis_conn, payload, pending_results=None):
    if pending_results is None:
        pending_results = {}
    if isinstance(payload, bytes):
        payload = payload.decode('utf-8', errors='replace')
    try:
        request = json.loads(payload)
    except (TypeError, ValueError):
        return _execute_action(payload)

    if not isinstance(request, dict):
        logger.error('Invalid worker control payload: %r', request)
        return False
    action = request.get('action')
    request_id = request.get('request_id')
    if not action or not request_id:
        logger.error('Worker control payload lacks action/request_id: %r', request)
        return False

    worker_id = listener_id()
    request_key = control_request_key(request_id)
    request_meta = redis_conn.hgetall(request_key)

    def _meta_value(name):
        return request_meta.get(name, request_meta.get(name.encode('utf-8')))

    recorded_action = _meta_value('action')
    if isinstance(recorded_action, bytes):
        recorded_action = recorded_action.decode('utf-8', errors='replace')
    raw_attempt = _meta_value('attempt')
    if recorded_action != action or raw_attempt is None:
        logger.error(
            'Control request %s metadata is missing or does not match action %r',
            request_id,
            action,
        )
        return False
    attempt = int(raw_attempt)
    canonical_result_key = control_result_key(request_id)
    attempt_result_key = control_attempt_result_key(request_id, attempt)
    pending_id = (str(request_id), worker_id, attempt)
    if pending_id in pending_results:
        pending = pending_results[pending_id]
        _persist_pending_result(redis_conn, pending)
        del pending_results[pending_id]
        return bool(pending['ok'])
    existing = redis_conn.hget(canonical_result_key, worker_id)
    if existing is not None:
        logger.info(
            'Control request %s was already handled by listener %s; copying its ACK to attempt %s',
            request_id,
            worker_id,
            attempt,
        )
        try:
            if isinstance(existing, bytes):
                existing = existing.decode('utf-8', errors='replace')
            existing_result = json.loads(existing)
            ok = existing_result.get('action') == action and existing_result.get('ok') is True
        except (TypeError, ValueError):
            ok = False
        result = existing
    else:
        try:
            ok = _execute_action(action)
        except Exception:
            logger.exception('Unhandled failure processing %s request %s', action, request_id)
            ok = False

        result = json.dumps({
            'action': action,
            'listener_id': worker_id,
            'ok': bool(ok),
        })
    pending = {
        'action': action,
        'attempt': attempt,
        'attempt_result_key': attempt_result_key,
        'canonical_result_key': canonical_result_key,
        'ok': bool(ok),
        'request_id': request_id,
        'request_key': request_key,
        'result': result,
        'worker_id': worker_id,
    }
    pending_results[pending_id] = pending
    _persist_pending_result(redis_conn, pending)
    del pending_results[pending_id]
    if ok:
        logger.info('Worker %s completed %s request %s', worker_id, action, request_id)
    else:
        logger.error('Worker %s failed %s request %s', worker_id, action, request_id)
    return bool(ok)


def main():
    channel = os.environ.get('AUDIO_MUSE_CONFIG_RESTART_CHANNEL', RESTART_CHANNEL)
    if os.environ.get('SERVICE_TYPE', '').lower() != 'worker':
        logger.info('SERVICE_TYPE is not worker; restart listener will remain idle')
        while True:
            time.sleep(3600)

    logger.info('Starting restart listener on channel: %s', channel)
    pending_results = {}

    while True:
        redis_conn = None
        pubsub = None
        try:
            redis_conn = new_redis_connection(
                socket_connect_timeout=5,
                socket_timeout=5,
                decode_responses=True,
            )
            pubsub = redis_conn.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe(channel)
            flush_pending_results(redis_conn, pending_results)
            logger.info('Subscribed to restart channel. Waiting for restart messages...')

            while True:
                message = pubsub.get_message(timeout=1.0)
                if not message:
                    continue
                if message.get('type') != 'message':
                    continue
                payload = message.get('data')
                logger.info('Control listener received request: %s', payload)
                handle_control_message(redis_conn, payload, pending_results)
        except Exception:
            logger.exception('Restart listener connection error, retrying in 5 seconds')
            time.sleep(5)
        finally:
            try:
                if pubsub is not None:
                    pubsub.close()
            except Exception:
                pass
            try:
                if redis_conn is not None:
                    redis_conn.close()
            except Exception:
                pass


if __name__ == '__main__':
    main()
