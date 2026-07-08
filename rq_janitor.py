# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Standalone process that reaps stale RQ job and worker registries.

Runs an infinite loop cleaning the started, finished, and failed job registries
of the high and default queues so orphaned jobs (from crashed or restarted
workers) do not accumulate, and prunes dead worker registrations whose Redis
keys expired (hard-killed containers never deregister, leaving ghost rows in
the dashboard's Queue Workers table); a sibling to the worker entrypoints.

Main Features:
* Periodic cleanup of started/finished/failed registries every 10 seconds.
* Dead worker registrations removed via clean_worker_registry per queue.
* Logs only when something is actually reaped, and survives per-iteration errors.
"""

import os
import sys
import time
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rq.worker_registration import clean_worker_registry
    from app_helper import rq_queue_high, rq_queue_default, redis_conn
    from app_logging import configure_logging
except ImportError as e:
    print(f"Error importing from app.py: {e}")
    print("Please ensure app.py is in the Python path and does not have top-level errors.")
    sys.exit(1)

configure_logging()
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("RQ Janitor process starting. Cleaning registries every 10 seconds.")
    queues_to_clean = [rq_queue_high, rq_queue_default]
    while True:
        try:
            for queue in queues_to_clean:
                started_registry = queue.started_job_registry
                started_before = started_registry.count
                started_registry.cleanup()
                started_after = started_registry.count
                started_cleaned = started_before - started_after
                if started_cleaned > 0:
                    logger.info(
                        "Janitor cleaned %d orphaned jobs from '%s' started_job_registry.",
                        started_cleaned,
                        queue.name,
                    )

                finished_registry = queue.finished_job_registry
                finished_before = finished_registry.count
                finished_registry.cleanup()
                finished_after = finished_registry.count
                finished_cleaned = finished_before - finished_after
                if finished_cleaned > 0:
                    logger.info(
                        "Janitor cleaned %d expired finished jobs from '%s' finished_job_registry.",
                        finished_cleaned,
                        queue.name,
                    )

                failed_registry = queue.failed_job_registry
                failed_before = failed_registry.count
                failed_registry.cleanup()
                failed_after = failed_registry.count
                failed_cleaned = failed_before - failed_after
                if failed_cleaned > 0:
                    logger.info(
                        "Janitor cleaned %d expired failed jobs from '%s' failed_job_registry.",
                        failed_cleaned,
                        queue.name,
                    )

            workers_before = redis_conn.scard('rq:workers')
            for queue in queues_to_clean:
                clean_worker_registry(queue)
            workers_removed = workers_before - redis_conn.scard('rq:workers')
            if workers_removed > 0:
                logger.info(
                    "Janitor removed %d dead worker registrations.", workers_removed
                )
        except Exception:
            logger.exception("Error in RQ Janitor loop")

        time.sleep(10)
