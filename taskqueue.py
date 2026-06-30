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
