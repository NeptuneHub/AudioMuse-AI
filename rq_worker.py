# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Entrypoint for the default-priority RQ worker process.

Configures the worker role and BLAS/OpenMP thread caps before importing heavy
numeric libraries, then runs a worker on the ``default`` queue; the companion
of ``rq_worker_high_priority`` for the ``high`` queue.

Main Features:
* Caps math-library threads (cpu_count // 2) and pins passive OpenMP waiting.
* Takes its worker class from ``rq_heartbeat_worker`` (a heartbeating SimpleWorker on
  Windows, the forking Worker elsewhere) and restarts after ``RQ_MAX_JOBS`` to limit leaks.
"""

import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ['AUDIOMUSE_ROLE'] = 'worker'

_cpu_count = os.cpu_count() or 2
_max_lyrics_threads = max(2, _cpu_count // 2)
for _env_key in (
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
):
    os.environ[_env_key] = str(_max_lyrics_threads)
os.environ.setdefault('GOMP_SPINCOUNT', '0')
os.environ.setdefault('OMP_WAIT_POLICY', 'passive')
print(f"Default worker CPU thread cap = {_max_lyrics_threads} (cpu_count // 2, min 2)")

from rq_heartbeat_worker import WorkerClass

try:
    from app_helper import redis_conn
    from app_logging import configure_logging
    from config import APP_VERSION, TEMP_DIR, RQ_MAX_JOBS, RQ_LOGGING_LEVEL
except ImportError as e:
    print(f"Error importing worker dependencies: {e}")
    sys.exit(1)

try:
    os.makedirs(TEMP_DIR, exist_ok=True)
except OSError as e:
    print(f"Warning: Could not create TEMP_DIR '{TEMP_DIR}': {e}")
    print(
        "Note: This may be expected in some test/CI environments, but could lead to task failures in production."
    )

configure_logging()
logger = logging.getLogger(__name__)

queues_to_listen = ['default']


if __name__ == '__main__':
    logger.info(
        f"DEFAULT RQ Worker starting. Version: {APP_VERSION}. Listening on queues: {queues_to_listen}"
    )
    logger.info(f"Using Redis connection: {redis_conn.connection_pool.connection_kwargs}")

    try:
        from plugin.manager import boot as plugin_boot

        plugin_boot('worker')
    except Exception:
        logger.exception('Plugin subsystem worker boot failed; continuing without plugins')

    worker = WorkerClass(
        queues_to_listen, connection=redis_conn, worker_ttl=120, job_monitoring_interval=30
    )

    max_jobs_before_restart = RQ_MAX_JOBS

    logging_level = RQ_LOGGING_LEVEL
    logger.info(f"RQ Worker logging level set to: {logging_level}")
    logger.info(f"Worker will restart after {max_jobs_before_restart} jobs to prevent memory leaks")

    try:
        worker.work(logging_level=logging_level, max_jobs=max_jobs_before_restart)
    except Exception:
        logger.exception("RQ Worker failed to start or encountered an error")
        sys.exit(1)
