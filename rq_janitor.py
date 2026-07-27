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
* Zombie worker hashes reaped: a job-count increment landing after the worker
  key expired recreates it without a TTL, leaving a permanent ghost row with no
  heartbeat - any registration without a last_heartbeat is deregistered.
* TTL-less worker keys expired: any rq:worker:* hash left without a TTL (a
  heartbeat or increment raced the key's expiry, including keys accumulated by
  releases before the re-registration mixin) gets the standard worker TTL, so a
  live worker re-asserts its own TTL on the next beat while a dead one
  evaporates and clean_worker_registry then drops its registration.
* Interrupted server-alignment sweeps recovered: a sweep whose RQ job died (e.g.
  killed by a worker restart) or vanished from Redis entirely is revoked and
  replaced with a fresh alignment covering every server.
* Orphaned main tasks reconciled: a task_status row is committed BEFORE its job
  is enqueued, so a Redis outage can leave a PENDING row with nothing behind it;
  a worker crash can instead leave the job parked in a terminal RQ registry while
  the row stays in PROGRESS. Rows past a grace period whose RQ job is missing or
  terminal are finalized; tasks that run inside the Flask process have no RQ job
  by design, so those are judged by how long ago they last reported progress.
* Logs only when something is actually reaped, and survives per-iteration errors.
"""

import os
import sys
import time
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rq import Worker
    from rq.defaults import DEFAULT_WORKER_TTL
    from rq.worker_registration import clean_worker_registry
    from app_helper import rq_queue_high, rq_queue_default, redis_conn
    from app_logging import configure_logging
    from tasks.multiserver_sync import recover_abandoned_sweeps, reap_orphaned_tasks
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

            zombies_removed = 0
            for worker in Worker.all(connection=redis_conn):
                try:
                    heartbeat = worker.last_heartbeat
                except Exception:
                    heartbeat = None
                if heartbeat is not None:
                    continue
                try:
                    worker.register_death()
                except Exception:
                    try:
                        redis_conn.srem('rq:workers', worker.key)
                        redis_conn.delete(worker.key)
                    except Exception:
                        logger.exception(
                            "Janitor could not remove zombie worker %s", worker.key
                        )
                        continue
                zombies_removed += 1
            if zombies_removed > 0:
                logger.info(
                    "Janitor removed %d zombie worker entries (hash without heartbeat).",
                    zombies_removed,
                )

            ghost_ttls_fixed = 0
            cursor = 0
            while True:
                cursor, worker_keys = redis_conn.scan(
                    cursor, match=Worker.redis_worker_namespace_prefix + '*', count=100
                )
                for worker_key in worker_keys:
                    if redis_conn.ttl(worker_key) == -1:
                        redis_conn.expire(worker_key, DEFAULT_WORKER_TTL + 60)
                        ghost_ttls_fixed += 1
                if cursor == 0:
                    break
            if ghost_ttls_fixed > 0:
                logger.info(
                    "Janitor gave %d TTL-less worker keys a %ds expiry.",
                    ghost_ttls_fixed,
                    DEFAULT_WORKER_TTL + 60,
                )
        except Exception:
            logger.exception("Error in RQ Janitor loop")

        try:
            recover_abandoned_sweeps()
        except Exception:
            logger.exception("Janitor sweep recovery failed")

        try:
            reap_orphaned_tasks()
        except Exception:
            logger.exception("Janitor orphaned-task reaping failed")

        time.sleep(10)
