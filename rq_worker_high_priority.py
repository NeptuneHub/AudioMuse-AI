# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Entrypoint for the high-priority RQ worker process.

Mirrors ``rq_worker`` but listens only on the ``high`` queue with tighter TTLs,
using an even lower math-library thread cap so latency-sensitive jobs run
promptly alongside the heavier default-queue worker.

Main Features:
* Caps math-library threads (cpu_count // 3) and pins passive OpenMP waiting.
* Uses ``SimpleWorker`` on Windows and restarts after ``RQ_MAX_JOBS_HIGH`` jobs.
"""

import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ['AUDIOMUSE_ROLE'] = 'worker'

_cpu_count = os.cpu_count() or 1
_max_threads = max(1, _cpu_count // 3)
for _env_key in (
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
):
    os.environ[_env_key] = str(_max_threads)
os.environ.setdefault('GOMP_SPINCOUNT', '0')
os.environ.setdefault('OMP_WAIT_POLICY', 'passive')
print(f"High-priority worker CPU thread cap = {_max_threads} (cpu_count // 3, min 1)")

from rq import SimpleWorker, Worker

WorkerClass = SimpleWorker if sys.platform == 'win32' else Worker

try:
    from app_helper import redis_conn
    from app_logging import configure_logging
    from config import APP_VERSION, RQ_MAX_JOBS_HIGH, RQ_LOGGING_LEVEL
except ImportError as e:
    print(f"Error importing from app.py: {e}")
    print("Please ensure app.py is in the Python path and does not have top-level errors.")
    sys.exit(1)

configure_logging()
logger = logging.getLogger(__name__)

queues_to_listen = ['high']

if __name__ == '__main__':
    logger.info(
        f"HIGH PRIORITY RQ Worker starting. Version: {APP_VERSION}. Listening ONLY on queues: {queues_to_listen}"
    )
    logger.info(f"Using Redis connection: {redis_conn.connection_pool.connection_kwargs}")

    worker = WorkerClass(
        queues_to_listen, connection=redis_conn, worker_ttl=30, job_monitoring_interval=10
    )

    max_jobs_before_restart = RQ_MAX_JOBS_HIGH

    logging_level = RQ_LOGGING_LEVEL
    logger.info(f"RQ Worker logging level set to: {logging_level}")
    logger.info(f"Worker will restart after {max_jobs_before_restart} jobs to prevent memory leaks")

    try:
        worker.work(logging_level=logging_level, max_jobs=max_jobs_before_restart)
    except Exception:
        logger.exception("High Priority RQ Worker failed to start or encountered an error")
        sys.exit(1)
