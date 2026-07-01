# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""RQ queue, registry and Redis connection configuration in taskqueue.

Covers how taskqueue wires queues, registries and the Redis connection, including
the platform-specific death-penalty class and socket-option selection by URL scheme.

Main Features:
* Base registry and queues use the platform death-penalty class, inherited by registries
* Redis socket options drop keepalive for unix sockets and keep it for tcp/tls
* Embedded-Redis argv builds the expected binary flags and URL
* Queue names, connection kwargs and default timeout are as configured; app_helper re-exports handles
"""

import config
import taskqueue
from rq.registry import BaseRegistry
from rq.timeouts import get_default_death_penalty_class


def test_base_registry_uses_platform_death_penalty():
    assert BaseRegistry.death_penalty_class is get_default_death_penalty_class()


def test_queues_use_platform_death_penalty():
    expected = get_default_death_penalty_class()
    assert taskqueue.rq_queue_high.death_penalty_class is expected
    assert taskqueue.rq_queue_default.death_penalty_class is expected


def test_registries_built_from_queues_inherit_platform_death_penalty():
    expected = get_default_death_penalty_class()
    for queue in (taskqueue.rq_queue_high, taskqueue.rq_queue_default):
        assert queue.started_job_registry.death_penalty_class is expected
        assert queue.finished_job_registry.death_penalty_class is expected
        assert queue.failed_job_registry.death_penalty_class is expected


def test_redis_socket_options_unix_url_omits_keepalive():
    assert taskqueue.redis_socket_options('unix:///tmp/r.sock') == {}


def test_redis_socket_options_tcp_url_keeps_keepalive():
    assert taskqueue.redis_socket_options('redis://h:6379/0') == {'socket_keepalive': True}


def test_redis_socket_options_tls_url_keeps_keepalive():
    assert taskqueue.redis_socket_options('rediss://h:6380/0') == {'socket_keepalive': True}


def test_build_embedded_redis_argv_binary_flags_and_url():
    argv, url = taskqueue.build_embedded_redis_argv(
        '/usr/bin/redis-server', '/var/lib/audiomuse/redis.sock', '/data'
    )
    assert argv[0] == '/usr/bin/redis-server'
    for flag, value in (
        ('--unixsocket', '/var/lib/audiomuse/redis.sock'),
        ('--unixsocketperm', '700'),
        ('--port', '0'),
        ('--save', ''),
        ('--appendonly', 'no'),
        ('--dir', '/data'),
    ):
        idx = argv.index(flag)
        assert argv[idx + 1] == value
    assert url == 'unix:///var/lib/audiomuse/redis.sock?db=0'


def test_redis_conn_connection_kwargs():
    kwargs = taskqueue.redis_conn.connection_pool.connection_kwargs
    assert kwargs['socket_connect_timeout'] == 30
    assert kwargs['socket_timeout'] == 60
    assert kwargs['health_check_interval'] == 30
    assert kwargs['retry_on_timeout'] is True
    expected_keepalive = not str(config.REDIS_URL).startswith('unix://')
    assert ('socket_keepalive' in kwargs) == expected_keepalive


def test_queue_names_connection_and_default_timeout():
    assert taskqueue.rq_queue_high.name == 'high'
    assert taskqueue.rq_queue_default.name == 'default'
    for queue in (taskqueue.rq_queue_high, taskqueue.rq_queue_default):
        assert queue.connection is taskqueue.redis_conn
        assert queue._default_timeout == -1


def test_app_helper_reexports_taskqueue_handles():
    import app_helper

    assert app_helper.redis_conn is taskqueue.redis_conn
    assert app_helper.rq_queue_high is taskqueue.rq_queue_high
    assert app_helper.rq_queue_default is taskqueue.rq_queue_default
