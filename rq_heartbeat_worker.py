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

Issue #784: when Redis is unreachable longer than the worker-key TTL the key expires,
clean_worker_registry SREMs the worker from ``rq:workers``, and only the heartbeat
recreates it - register() ran once at birth, so the live worker keeps taking jobs while
invisible to the dashboard. The mixin re-runs register() on every heartbeat and rebuilds
the identity hash when the key came back partial, clearing any ``death`` the janitor
stamped on a worker that had not actually gone away.

Four rq 2.7.0 details this override depends on, each anchored by a test: the repair runs
on its own pipeline, never the one heartbeat was handed, because maintain_heartbeats
reads its results by FIXED position (``results[7]`` is job.heartbeat's HSET); the two
heartbeat commands are HSET then EXPIRE, the reverse of rq's order, so an expired key
cannot be recreated without a TTL; last_heartbeat is recorded on the instance, which rq
only does in refresh() (issue #799); and StopRequested is re-raised because it carries
the warm-shutdown request out of the signal handler.

The Windows beat thread shares a lock with execution preparation and cleanup, so a beat
can neither see an execution before its transaction commits nor outlive the cleanup that
deletes it, and execute_job does not return until the thread exits, so a blocked beat
cannot cross into the next job. A beat that finds no execution refreshes the worker key
alone rather than returning empty-handed; there is no execution to beat, but the worker
key is the worker's own and is always safe to renew.

Main Features:
* ReregisterOnHeartbeatMixin: SADDs the worker back into rq:workers on each heartbeat
  and restores identity/state/current job if the key expired and came back partial.
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
from rq.utils import now, utcformat

logger = logging.getLogger(__name__)


class ReregisterOnHeartbeatMixin:
    def heartbeat(self, timeout=None, pipeline=None):
        timeout = timeout or self.worker_ttl + 60
        heartbeat_at = now()
        self.last_heartbeat = heartbeat_at

        if pipeline is None:
            with self.connection.pipeline() as heartbeat_pipeline:
                heartbeat_pipeline.hset(self.key, 'last_heartbeat', utcformat(heartbeat_at))
                heartbeat_pipeline.expire(self.key, timeout)
                heartbeat_pipeline.execute()
        else:
            pipeline.hset(self.key, 'last_heartbeat', utcformat(heartbeat_at))
            pipeline.expire(self.key, timeout)

        self.log.debug(
            'Worker %s: sent heartbeat to prevent worker timeout. '
            'Next one should arrive in %s seconds.',
            self.name,
            timeout,
        )

        try:
            identity_missing = not self.connection.hexists(self.key, 'birth')
            with self.connection.pipeline() as repair:
                if identity_missing:
                    repair.hset(self.key, mapping=self._identity_mapping())
                    repair.hdel(self.key, 'death')
                worker_registration.register(self, repair)
                repair.expire(self.key, timeout)
                repair.execute()
        except StopRequested:
            raise
        except Exception:
            logger.exception("Worker %s: re-registration on heartbeat failed", self.name)

    def _identity_mapping(self):
        stamp = utcformat(getattr(self, 'last_heartbeat', None) or now())
        birth_date = getattr(self, 'birth_date', None)
        mapping = {
            'birth': utcformat(birth_date) if birth_date else stamp,
            'last_heartbeat': stamp,
            'queues': ','.join(self.queue_names()),
            'pid': getattr(self, 'pid', None) or 0,
            'hostname': getattr(self, 'hostname', None) or '',
            'ip_address': getattr(self, 'ip_address', None) or '',
            'version': getattr(self, 'version', None) or '',
            'python_version': getattr(self, 'python_version', None) or '',
            'state': self.get_state(),
        }
        job_id = getattr(getattr(self, 'execution', None), 'job_id', None)
        if job_id:
            mapping['current_job'] = job_id
        return mapping


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
