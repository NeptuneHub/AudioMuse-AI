# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Redis connection and RQ queue definitions shared across the app.

Creates the process-wide Redis connection and the ``high`` and ``default`` RQ
queues that workers and enqueuers import, and re-exports the RQ primitives so
callers depend on this module rather than on ``rq`` directly.

Main Features:
* Configures the Redis connection with keepalive and Unix-socket awareness.
* Builds the argv for the embedded Redis server used by the standalone build.
"""

from redis import Redis
from rq import Queue, get_current_job
from rq.job import Job
from rq.exceptions import NoSuchJobError
from rq.command import send_stop_job_command
from rq.registry import BaseRegistry
from rq.timeouts import get_default_death_penalty_class

import config

_death_penalty_class = get_default_death_penalty_class()
BaseRegistry.death_penalty_class = _death_penalty_class

__all__ = [
    "redis_conn",
    "rq_queue_high",
    "rq_queue_default",
    "Job",
    "NoSuchJobError",
    "send_stop_job_command",
    "get_current_job",
    "build_embedded_redis_argv",
    "redis_socket_options",
]


def redis_socket_options(url):
    return {} if str(url).startswith("unix://") else {"socket_keepalive": True}


redis_conn = Redis.from_url(
    config.REDIS_URL,
    socket_connect_timeout=30,
    socket_timeout=60,
    health_check_interval=30,
    retry_on_timeout=True,
    **redis_socket_options(config.REDIS_URL),
)

rq_queue_high = Queue(
    'high', connection=redis_conn, default_timeout=-1, death_penalty_class=_death_penalty_class
)
rq_queue_default = Queue(
    'default', connection=redis_conn, default_timeout=-1, death_penalty_class=_death_penalty_class
)


def build_embedded_redis_argv(server_binary, socket_path, data_dir):
    argv = [
        server_binary,
        "--unixsocket",
        socket_path,
        "--unixsocketperm",
        "700",
        "--port",
        "0",
        "--save",
        "",
        "--appendonly",
        "no",
        "--dir",
        data_dir,
    ]
    redis_url = f"unix://{socket_path}?db=0"
    return argv, redis_url
