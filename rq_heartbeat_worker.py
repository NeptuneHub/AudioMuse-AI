# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Worker class selection, with a heartbeat that keeps long jobs alive on Windows
and re-registers a worker that a transient Redis outage silently deregistered.

RQ's forking ``Worker`` refreshes a running job's started-registry score from its
monitor loop. ``SimpleWorker`` (the only option on Windows, which cannot fork) has no
monitor loop, so that score is written once as ``now + DEFAULT_WORKER_TTL`` (420s) and
then goes stale even for ``job_timeout=-1``, and the janitor's
``started_registry.cleanup()`` re-queues or fails every job running longer than seven
minutes. The Windows heartbeat must NOT be replaced by returning -1 from
get_heartbeat_ttl: Execution.save would then EXPIRE the key with a negative TTL,
deleting it the instant the job starts.

Issue #784: when Redis is unreachable longer than the worker-key TTL the key expires and
clean_worker_registry SREMs the worker from ``rq:workers``; register() runs once at
birth, so the live worker keeps taking jobs while invisible to the dashboard - no rq
release through 2.10 re-adds it. The mixin re-runs register() after every heartbeat,
clears any ``death`` a janitor stamped on a worker that had not actually gone away, and
re-EXPIREs the key so it regains a TTL even on rq 2.7, whose heartbeat recreates an
expired key without one (the noavx2 image stays on that pin). The repair runs on the
worker's own pipeline only: rq's maintain_heartbeats reads the pipeline it hands to
heartbeat positionally, so queueing extra commands there would make rq inspect the
wrong result. Since rq 2.9 maintain_heartbeats itself rebuilds a recreated key's hash
content; on rq 2.7 the hash stays partial until restart, which is accepted.

The Windows beat thread shares a lock with execution preparation and cleanup, so a beat
can neither see an execution before its transaction commits nor outlive the cleanup that
deletes it, and execute_job does not return until the thread exits, so a blocked beat
cannot cross into the next job. A beat that finds no execution refreshes the worker key
alone rather than returning empty-handed; there is no execution to beat, but the worker
key is the worker's own and is always safe to renew.

Main Features:
* ReregisterOnHeartbeatMixin: SADDs the worker back into rq:workers after every
  heartbeat, clears false death stamps, and re-EXPIREs the worker key.
* HeartbeatSimpleWorker: runs perform_job on the main thread while a daemon thread
  calls maintain_heartbeats, giving SimpleWorker the liveness signal the forking
  worker gets for free; also carries the re-registration mixin.
* The beat thread is serialized with execution setup and cleanup, keeps the worker key
  alive across both, and logs genuine Redis failures instead of dying.
* WorkerClass: HeartbeatSimpleWorker on win32, a re-registering forking Worker elsewhere.
"""

import logging
import sys
import threading

from rq import SimpleWorker, Worker, worker_registration
from rq.exceptions import StopRequested

logger = logging.getLogger(__name__)


class ReregisterOnHeartbeatMixin:
    def heartbeat(self, timeout=None, pipeline=None):
        timeout = timeout or self.worker_ttl + 60
        super().heartbeat(timeout, pipeline)
        try:
            with self.connection.pipeline() as repair:
                worker_registration.register(self, repair)
                repair.hdel(self.key, 'death')
                repair.expire(self.key, timeout)
                repair.execute()
        except StopRequested:
            raise
        except Exception:
            logger.exception("Worker %s: re-registration on heartbeat failed", self.name)


class ReregisteringWorker(ReregisterOnHeartbeatMixin, Worker):
    pass


class HeartbeatSimpleWorker(ReregisterOnHeartbeatMixin, SimpleWorker):
    def _refresh_job_heartbeat(self, job, stop, heartbeat_lock):
        with heartbeat_lock:
            if stop.is_set():
                return
            try:
                if self.execution is None:
                    self.heartbeat(self.job_monitoring_interval + 60)
                else:
                    self.maintain_heartbeats(job)
            except Exception:
                logger.exception("Heartbeat refresh failed for job %s", job.id)

    def _heartbeat_loop(self, job, stop, heartbeat_lock):
        interval = max(1, int(self.job_monitoring_interval))
        while not stop.wait(interval):
            self._refresh_job_heartbeat(job, stop, heartbeat_lock)

    def prepare_execution(self, job):
        heartbeat_lock = getattr(self, '_execution_heartbeat_lock', None)
        if heartbeat_lock is None:
            return super().prepare_execution(job)

        with heartbeat_lock:
            return super().prepare_execution(job)

    def cleanup_execution(self, job, pipeline):
        stop = getattr(self, '_execution_heartbeat_stop', None)
        if stop is not None:
            stop.set()

        heartbeat_lock = getattr(self, '_execution_heartbeat_lock', None)
        if heartbeat_lock is None:
            return super().cleanup_execution(job, pipeline)

        with heartbeat_lock:
            return super().cleanup_execution(job, pipeline)

    def execute_job(self, job, queue):
        stop = threading.Event()
        heartbeat_lock = threading.Lock()
        self._execution_heartbeat_stop = stop
        self._execution_heartbeat_lock = heartbeat_lock

        beater = threading.Thread(
            target=self._heartbeat_loop,
            args=(job, stop, heartbeat_lock),
            name=f"rq-heartbeat-{job.id}",
            daemon=True,
        )
        beater.start()
        try:
            return super().execute_job(job, queue)
        finally:
            stop.set()
            beater.join()
            if getattr(self, '_execution_heartbeat_stop', None) is stop:
                self._execution_heartbeat_stop = None
            if getattr(self, '_execution_heartbeat_lock', None) is heartbeat_lock:
                self._execution_heartbeat_lock = None


WorkerClass = HeartbeatSimpleWorker if sys.platform == 'win32' else ReregisteringWorker
