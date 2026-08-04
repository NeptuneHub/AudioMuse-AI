# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Restart scheduling and durable worker-control acknowledgement tests.

Covers restart_manager's gating so only a Flask service with restart enabled
arms the restart timer, using a mocked timer to observe the decision.

Main Features:
* Returns False when the service type is unset or is a worker
* Returns False when the Flask restart flag is disabled (case-insensitive)
* Guards pass only for a Flask service with restart explicitly enabled
"""

import json
import socket
from unittest.mock import MagicMock

import pytest

import restart_manager


class _ControlRedis:
    def __init__(self, subscribers=1, auto_ack=True):
        self.subscribers = subscribers
        self.auto_ack = auto_ack
        self.requests = {}
        self.results = {}
        self.publish_count = 0
        self.closed = False
        self.lease_expired = False

    def eval(
        self, _script, _numkeys, request_key, _channel, payload, action,
        _ttl, _lease, max_attempts,
    ):
        existing = self.requests.get(request_key)
        if existing:
            if existing['action'] != action:
                return -2
            if existing.get('outcome') == '1':
                return -4
            if existing.get('outcome') == '0':
                return -3
            if existing['expected'] > 0 and not self.lease_expired:
                return existing['expected']
            if (
                existing['expected'] > 0
                and existing.get('delivery_attempts', 1) >= int(max_attempts)
            ):
                existing['outcome'] = '0'
                return -3
            attempt = existing.get('attempt', 1) + 1
            delivery_attempts = existing.get('delivery_attempts', 0)
        else:
            attempt = 1
            delivery_attempts = 0
        self.publish_count += 1
        delivered = self.subscribers
        prior_floor = existing.get('expected_floor', 0) if existing else 0
        expected = max(prior_floor, delivered)
        if delivered > 0:
            delivery_attempts = 1 if not prior_floor else delivery_attempts + 1
        elif prior_floor:
            delivery_attempts += 1
        self.requests[request_key] = {
            'action': action,
            'attempt': attempt,
            'delivery_attempts': delivery_attempts,
            'expected': expected,
            'expected_floor': expected,
        }
        if delivered and self.auto_ack:
            request_id = json.loads(payload)['request_id']
            result = json.dumps({'action': action, 'ok': True})
            self.results[restart_manager.control_result_key(request_id)] = {
                'worker-1': result,
            }
            self.results[restart_manager.control_attempt_result_key(request_id, attempt)] = {
                'worker-1': json.dumps({'action': action, 'ok': True}),
            }
        return expected

    def hgetall(self, key):
        return self.requests.get(key, self.results.get(key, {}))

    def hset(self, key, field, value):
        self.requests.setdefault(key, {})[field] = value

    def expire(self, _key, _ttl):
        return True

    def close(self):
        self.closed = True


@pytest.fixture
def mock_timer(monkeypatch):
    timer_cls = MagicMock()
    monkeypatch.setattr(restart_manager.threading, 'Timer', timer_cls)
    return timer_cls


def test_returns_false_when_service_type_unset(monkeypatch, mock_timer):
    monkeypatch.delenv('SERVICE_TYPE', raising=False)
    monkeypatch.delenv('DISABLE_FLASK_RESTART', raising=False)

    assert restart_manager.schedule_flask_restart() is False
    mock_timer.assert_not_called()


def test_returns_false_when_service_type_is_worker(monkeypatch, mock_timer):
    monkeypatch.setenv('SERVICE_TYPE', 'worker')
    monkeypatch.delenv('DISABLE_FLASK_RESTART', raising=False)

    assert restart_manager.schedule_flask_restart() is False
    mock_timer.assert_not_called()


def test_returns_false_when_flask_restart_disabled(monkeypatch, mock_timer):
    monkeypatch.setenv('SERVICE_TYPE', 'flask')
    monkeypatch.setenv('DISABLE_FLASK_RESTART', 'true')

    assert restart_manager.schedule_flask_restart() is False
    mock_timer.assert_not_called()


def test_disable_guard_is_case_insensitive(monkeypatch, mock_timer):
    monkeypatch.setenv('SERVICE_TYPE', 'FLASK')
    monkeypatch.setenv('DISABLE_FLASK_RESTART', 'TRUE')

    assert restart_manager.schedule_flask_restart() is False
    mock_timer.assert_not_called()


def test_guards_pass_for_flask_service_with_restart_enabled(monkeypatch, mock_timer):
    monkeypatch.setenv('SERVICE_TYPE', 'flask')
    monkeypatch.setenv('DISABLE_FLASK_RESTART', 'false')

    assert restart_manager.schedule_flask_restart() is True
    mock_timer.assert_called_once_with(2.5, restart_manager._restart_flask_program)
    mock_timer.return_value.start.assert_called_once_with()


class TestWorkerControlAcknowledgements:
    def test_publish_waits_for_durable_action_result(self, monkeypatch):
        redis_conn = _ControlRedis(subscribers=1, auto_ack=True)
        monkeypatch.setattr(restart_manager, 'new_redis_connection', lambda **_kwargs: redis_conn)

        assert restart_manager.publish_restart_request(
            request_id='request-1', timeout_seconds=0
        ) is True
        assert redis_conn.publish_count == 1
        assert redis_conn.closed is True

    def test_request_with_no_listener_can_retry_same_id_safely(self, monkeypatch):
        redis_conn = _ControlRedis(subscribers=0, auto_ack=True)
        monkeypatch.setattr(restart_manager, 'new_redis_connection', lambda **_kwargs: redis_conn)

        assert restart_manager.publish_stop_request(
            request_id='recoverable', timeout_seconds=0
        ) is False
        redis_conn.subscribers = 1
        assert restart_manager.publish_stop_request(
            request_id='recoverable', timeout_seconds=0
        ) is True
        assert redis_conn.publish_count == 2

        # Once a listener received it, later calls only read the stored ACK.
        assert restart_manager.publish_stop_request(
            request_id='recoverable', timeout_seconds=0
        ) is True
        assert redis_conn.publish_count == 2

    def test_nonzero_delivery_is_not_republished_before_its_lease(self, monkeypatch):
        redis_conn = _ControlRedis(subscribers=1, auto_ack=False)
        monkeypatch.setattr(restart_manager, 'new_redis_connection', lambda **_kwargs: redis_conn)

        assert restart_manager.publish_start_request(
            request_id='pending', timeout_seconds=0
        ) is False
        assert restart_manager.publish_start_request(
            request_id='pending', timeout_seconds=0
        ) is False
        assert redis_conn.publish_count == 1

    def test_missing_ack_is_republished_after_lease_and_can_recover(self, monkeypatch):
        redis_conn = _ControlRedis(subscribers=1, auto_ack=False)
        monkeypatch.setattr(restart_manager, 'new_redis_connection', lambda **_kwargs: redis_conn)

        assert restart_manager.publish_restart_request(
            request_id='listener-crashed', timeout_seconds=0
        ) is False
        redis_conn.lease_expired = True
        redis_conn.auto_ack = True
        assert restart_manager.publish_restart_request(
            request_id='listener-crashed', timeout_seconds=0
        ) is True
        assert redis_conn.publish_count == 2

    def test_recovery_never_lowers_expected_when_a_replica_disappears(self, monkeypatch):
        redis_conn = _ControlRedis(subscribers=2, auto_ack=False)
        monkeypatch.setattr(restart_manager, 'new_redis_connection', lambda **_kwargs: redis_conn)

        assert restart_manager.publish_restart_request(
            request_id='replica-gone', timeout_seconds=0
        ) is False
        redis_conn.lease_expired = True
        redis_conn.subscribers = 1
        redis_conn.auto_ack = True
        assert restart_manager.publish_restart_request(
            request_id='replica-gone', timeout_seconds=0
        ) is False
        request = redis_conn.requests[restart_manager.control_request_key('replica-gone')]
        assert request['expected'] == 2

    def test_missing_ack_becomes_durable_false_after_bounded_rounds(self, monkeypatch):
        redis_conn = _ControlRedis(subscribers=1, auto_ack=False)
        monkeypatch.setattr(restart_manager, 'new_redis_connection', lambda **_kwargs: redis_conn)

        for _attempt in range(restart_manager.CONTROL_MAX_DELIVERY_ATTEMPTS):
            assert restart_manager.publish_restart_request(
                request_id='bounded', timeout_seconds=0
            ) is False
            redis_conn.lease_expired = True

        assert restart_manager.publish_restart_request(
            request_id='bounded', timeout_seconds=0
        ) is False
        assert restart_manager.get_control_request_result('restart', 'bounded') is False
        assert redis_conn.publish_count == restart_manager.CONTROL_MAX_DELIVERY_ATTEMPTS

    def test_all_targeted_listeners_must_ack_and_any_failure_is_false(self):
        request_id = 'multiple-workers'
        redis_conn = _ControlRedis(subscribers=2, auto_ack=False)
        redis_conn.requests[restart_manager.control_request_key(request_id)] = {
            'action': 'restart',
            'attempt': 1,
            'expected': 2,
        }
        attempt_key = restart_manager.control_attempt_result_key(request_id, 1)
        redis_conn.results[attempt_key] = {
            'worker-a': json.dumps({'action': 'restart', 'ok': True}),
        }
        assert restart_manager.get_control_request_result(
            'restart', request_id, redis_conn=redis_conn
        ) is None

        redis_conn.results[attempt_key]['worker-b'] = json.dumps(
            {'action': 'restart', 'ok': False}
        )
        assert restart_manager.get_control_request_result(
            'restart', request_id, redis_conn=redis_conn
        ) is False

    def test_redis_read_error_is_pending_not_a_negative_ack(self, monkeypatch):
        redis_conn = MagicMock()
        redis_conn.hgetall.side_effect = RuntimeError('temporary Redis blip')
        monkeypatch.setattr(restart_manager, 'new_redis_connection', lambda **_kwargs: redis_conn)

        assert restart_manager.get_control_request_result('restart', 'request-2') is None


def test_native_control_timeout_allows_completion_inside_ack_deadline(monkeypatch):
    seen = {}

    class _Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def settimeout(self, timeout):
            seen['timeout'] = timeout

        def connect(self, address):
            seen['address'] = address

        def sendall(self, payload):
            seen['payload'] = payload

        def recv(self, _size):
            return b'ok'

    monkeypatch.setattr(
        restart_manager,
        '_control_endpoint',
        lambda: (socket.AF_INET, ('127.0.0.1', 12345), 'test-control'),
    )
    monkeypatch.setattr(restart_manager.socket, 'socket', lambda *_args: _Socket())

    assert restart_manager._send_control(['restart', 'rq-worker-default']) is True
    assert 0 < seen['timeout'] < restart_manager.CONTROL_ACK_TIMEOUT_SECONDS
    assert restart_manager.CONTROL_ACK_TIMEOUT_SECONDS <= 30


def test_the_advisory_ack_bound_cannot_occupy_a_gunicorn_thread_for_long():
    assert restart_manager.CONTROL_ACK_ADVISORY_TIMEOUT_SECONDS <= 5.0
    assert (
        restart_manager.CONTROL_ACK_ADVISORY_TIMEOUT_SECONDS
        <= restart_manager.CONTROL_ACK_TIMEOUT_SECONDS
    )


def test_publish_control_request_never_sleeps_past_a_caller_supplied_ack_bound(monkeypatch):
    clock = {'now': 1000.0}
    waited = []

    class _Redis:
        def close(self):
            pass

        def eval(self, *_args):
            return 1

        def hgetall(self, _key):
            return {}

    def _sleep(seconds):
        waited.append(seconds)
        clock['now'] += seconds

    monkeypatch.setattr(restart_manager, 'new_redis_connection', lambda **_kw: _Redis())
    monkeypatch.setattr(restart_manager.time, 'monotonic', lambda: clock['now'])
    monkeypatch.setattr(restart_manager.time, 'sleep', _sleep)

    assert restart_manager.publish_control_request('restart', timeout_seconds=4.0) is False
    assert sum(waited) <= 4.0, (
        'the wait occupies one of gunicorn 4 threads, so it must never outlive the '
        'deadline the request handler asked for'
    )
    assert waited, 'it must actually poll rather than give up immediately'


@pytest.mark.parametrize(
    'action,output',
    [
        ('start', 'rq-worker-default: ERROR (already started)\nrq-janitor: started'),
        ('stop', 'rq-worker-default: ERROR (not running)\nrq-janitor: stopped'),
    ],
)
def test_supervisor_start_stop_replays_accept_already_satisfied_state(
    action, output, monkeypatch
):
    monkeypatch.setattr(restart_manager, '_use_control_ipc', lambda: False)
    monkeypatch.setattr(
        restart_manager.subprocess,
        'run',
        lambda *_a, **_k: type('Result', (), {
            'returncode': 1, 'stdout': output, 'stderr': '',
        })(),
    )

    assert restart_manager._run_supervisorctl(
        [action, 'rq-worker-default', 'rq-janitor']
    ) is True
