"""Centralized task-queue / Redis access, dispatched on ``config.QUEUE_TYPE``.

Companion to :mod:`database`, in the same spirit as ``tasks/mediaserver.py``: one
module owns the Redis connection and the two RQ queues so the rest of the code
never builds them itself. ``app_helper`` re-exports the handles below, so every
module that does ``from app_helper import redis_conn, rq_queue_high`` is untouched.

``redis`` (default) and ``embedded`` both talk to a *real* Redis via
``config.REDIS_URL`` -- RQ relies on Lua ``EVAL``, pub/sub and job registries, so a
genuine server is required. The only difference is who starts it: with ``embedded``
the macOS supervisor launches the bundled ``redis-server`` binary (see
:func:`build_embedded_redis_argv`) and exports its socket URL as ``REDIS_URL``
before the app and workers boot.
"""

from redis import Redis
from rq import Queue, get_current_job
from rq.job import Job
from rq.exceptions import NoSuchJobError
from rq.command import send_stop_job_command

import config

__all__ = [
    "redis_conn",
    "rq_queue_high",
    "rq_queue_default",
    "Job",
    "NoSuchJobError",
    "send_stop_job_command",
    "get_current_job",
    "build_embedded_redis_argv",
]

redis_conn = Redis.from_url(
    config.REDIS_URL,
    socket_connect_timeout=30,
    socket_timeout=60,
    socket_keepalive=True,
    health_check_interval=30,
    retry_on_timeout=True
)

rq_queue_high = Queue('high', connection=redis_conn, default_timeout=-1)
rq_queue_default = Queue('default', connection=redis_conn, default_timeout=-1)


def build_embedded_redis_argv(server_binary, socket_path, data_dir):
    """Return ``(argv, redis_url)`` for launching a private embedded Redis.

    Used only by the standalone (macOS) supervisor when ``QUEUE_TYPE`` is
    ``embedded``. The supervisor owns the resulting process (spawn, log capture,
    group shutdown). TCP is disabled (``--port 0``) so the instance is reachable
    only over the unix socket and never collides with a Redis the user already
    runs on 6379; persistence is off because RQ state is transient queue data.
    """
    argv = [
        server_binary,
        "--unixsocket", socket_path,
        "--unixsocketperm", "700",
        "--port", "0",
        "--save", "",
        "--appendonly", "no",
        "--dir", data_dir,
    ]
    redis_url = f"unix://{socket_path}?db=0"
    return argv, redis_url
