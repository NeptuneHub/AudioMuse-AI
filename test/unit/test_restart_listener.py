# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only

"""Worker-side execution and durable ACK behaviour for control requests.

A restart is only useful if a WORKER acted on it. The Flask container subscribes
to the same channel and ignores every signal, so counting publish subscribers can
never prove delivery; the worker listener records a durable result instead.

Main Features:
* A result is recorded only after the action ran, and repeats are deduplicated
* An explicit failure is persisted rather than silently dropped
* The ACK write retries after a Redis reconnect without re-running the action
"""

import json
from unittest.mock import MagicMock

import pytest

import restart_listener
import restart_manager


class _ListenerRedis:
    def __init__(self):
        self.values = {}
        self.expired = []

    def hget(self, key, field):
        return self.values.get(key, {}).get(field)

    def hgetall(self, key):
        return self.values.get(key, {})

    def hset(self, key, field, value):
        self.values.setdefault(key, {})[field] = value

    def eval(self, _script, numkeys, *args):
        keys = args[:numkeys]
        field, value, ttl = args[numkeys:]
        for key in keys[:2]:
            self.hset(key, field, value)
        for key in keys:
            self.expire(key, ttl)
        return 1

    def expire(self, key, ttl):
        self.expired.append((key, ttl))


def _payload(action='restart', request_id='req-1'):
    return json.dumps({'action': action, 'request_id': request_id})


def _prepare_request(redis_conn, action='restart', request_id='req-1', attempt=1):
    redis_conn.values[restart_manager.control_request_key(request_id)] = {
        'action': action,
        'attempt': attempt,
        'expected': 1,
    }


def test_listener_records_result_only_after_action_and_deduplicates(monkeypatch):
    redis_conn = _ListenerRedis()
    _prepare_request(redis_conn)
    order = []

    def execute(action):
        order.append(('action', action))
        return True

    monkeypatch.setenv('AUDIO_MUSE_WORKER_ID', 'worker-a')
    monkeypatch.setattr(restart_listener.socket, 'gethostname', lambda: 'pod-1')
    monkeypatch.setattr(restart_listener, '_execute_action', execute)
    original_hset = redis_conn.hset

    def record_hset(key, field, value):
        order.append(('ack', key))
        original_hset(key, field, value)

    redis_conn.hset = record_hset

    assert restart_listener.handle_control_message(redis_conn, _payload()) is True
    assert order[0] == ('action', 'restart')
    assert order[1][0] == 'ack'
    stored = redis_conn.values[restart_manager.control_result_key('req-1')]['worker-a:pod-1']
    assert json.loads(stored)['ok'] is True
    assert (
        restart_manager.control_result_key('req-1'),
        restart_manager.CONTROL_RESULT_TTL_SECONDS,
    ) in redis_conn.expired

    # A reconnect/re-delivery sees the stable listener field and does not act twice.
    assert restart_listener.handle_control_message(redis_conn, _payload()) is True
    assert order.count(('action', 'restart')) == 1


def test_listener_persists_explicit_failure(monkeypatch):
    redis_conn = _ListenerRedis()
    _prepare_request(redis_conn, action='stop', request_id='req-failed')
    monkeypatch.setenv('AUDIO_MUSE_WORKER_ID', 'worker-b')
    monkeypatch.setattr(restart_listener.socket, 'gethostname', lambda: 'pod-2')
    monkeypatch.setattr(restart_listener, '_execute_action', lambda _action: False)

    assert restart_listener.handle_control_message(
        redis_conn, _payload(action='stop', request_id='req-failed')
    ) is False
    stored = redis_conn.values[restart_manager.control_result_key('req-failed')]['worker-b:pod-2']
    assert json.loads(stored) == {
        'action': 'stop',
        'listener_id': 'worker-b:pod-2',
        'ok': False,
    }


def test_ack_write_retries_after_redis_reconnect_without_reexecuting(monkeypatch):
    pending = {}
    execute = MagicMock(return_value=True)
    monkeypatch.setenv('AUDIO_MUSE_WORKER_ID', 'worker-c')
    monkeypatch.setattr(restart_listener.socket, 'gethostname', lambda: 'pod-3')
    monkeypatch.setattr(restart_listener, '_execute_action', execute)

    class _DroppedRedis(_ListenerRedis):
        def hset(self, _key, _field, _value):
            raise ConnectionError('Redis dropped after supervisor action')

    dropped = _DroppedRedis()
    _prepare_request(dropped, request_id='req-reconnect')
    with pytest.raises(ConnectionError):
        restart_listener.handle_control_message(
            dropped, _payload(request_id='req-reconnect'), pending
        )
    execute.assert_called_once_with('restart')
    assert pending

    reconnected = _ListenerRedis()
    _prepare_request(reconnected, request_id='req-reconnect')
    restart_listener.flush_pending_results(reconnected, pending)
    assert pending == {}
    stored = reconnected.values[
        restart_manager.control_result_key('req-reconnect')
    ]['worker-c:pod-3']
    assert json.loads(stored)['ok'] is True

    # Even a duplicate delivery after reconnect reads the ACK and does not act.
    assert restart_listener.handle_control_message(
        reconnected, _payload(request_id='req-reconnect'), pending
    ) is True
    execute.assert_called_once_with('restart')


def test_republished_attempt_copies_canonical_ack_without_reexecuting(monkeypatch):
    redis_conn = _ListenerRedis()
    _prepare_request(redis_conn, request_id='req-republish', attempt=2)
    monkeypatch.setenv('AUDIO_MUSE_WORKER_ID', 'worker-d')
    monkeypatch.setattr(restart_listener.socket, 'gethostname', lambda: 'pod-4')
    execute = MagicMock(side_effect=AssertionError('ACKed action must not run twice'))
    monkeypatch.setattr(restart_listener, '_execute_action', execute)
    worker_id = 'worker-d:pod-4'
    redis_conn.values[restart_manager.control_result_key('req-republish')] = {
        worker_id: json.dumps({
            'action': 'restart', 'listener_id': worker_id, 'ok': True,
        })
    }

    assert restart_listener.handle_control_message(
        redis_conn, _payload(request_id='req-republish')
    ) is True
    execute.assert_not_called()
    attempt_result = redis_conn.values[
        restart_manager.control_attempt_result_key('req-republish', 2)
    ][worker_id]
    assert json.loads(attempt_result)['ok'] is True


def test_flask_role_never_opens_or_subscribes_to_redis(monkeypatch):
    monkeypatch.setenv('SERVICE_TYPE', 'flask')
    connect = MagicMock(side_effect=AssertionError('Flask must not subscribe'))
    monkeypatch.setattr(restart_listener, 'new_redis_connection', connect)

    class _IdleLoopReached(Exception):
        pass

    monkeypatch.setattr(
        restart_listener.time,
        'sleep',
        lambda _seconds: (_ for _ in ()).throw(_IdleLoopReached()),
    )
    with pytest.raises(_IdleLoopReached):
        restart_listener.main()
    connect.assert_not_called()


def test_plugin_sync_acknowledges_background_thread_start_without_running_inline(monkeypatch):
    presync = MagicMock()
    started = []

    class _Thread:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            started.append(self.target)

    monkeypatch.setattr(restart_listener, 'worker_presync', presync)
    monkeypatch.setattr(restart_listener.threading, 'Thread', _Thread)

    assert restart_listener._execute_action('plugin-sync') is True
    presync.assert_not_called()
    assert len(started) == 1


def test_plugin_sync_thread_start_failure_is_negative_ack(monkeypatch):
    thread = MagicMock()
    thread.start.side_effect = RuntimeError('cannot start thread')
    monkeypatch.setattr(restart_listener, 'worker_presync', lambda: None)
    monkeypatch.setattr(restart_listener.threading, 'Thread', lambda **_kwargs: thread)

    assert restart_listener._execute_action('plugin-sync') is False
