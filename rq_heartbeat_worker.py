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
monitor loop, so the RQ janitor can tell a live job from a dead one. ``SimpleWorker``
(the only option on Windows, which cannot fork) runs the job in-process and has no
monitor loop, so it never refreshes that score: it is written once as
``now + DEFAULT_WORKER_TTL`` (420s) and then goes stale, even for ``job_timeout=-1``.
The janitor's ``started_registry.cleanup()`` then expires the entry and either
re-queues the job or fails it, so no analysis, clustering, cleaning or sweep running
longer than seven minutes could ever complete on Windows. The Windows heartbeat must
NOT be replaced by returning -1 from get_heartbeat_ttl: Execution.save would then
EXPIRE the key with a negative TTL, deleting it the instant the job starts.

Issue #784: if Redis is unreachable longer than the worker-key TTL, the key expires,
clean_worker_registry SREMs the worker from ``rq:workers``, and on reconnect only the
heartbeat (HSET last_heartbeat + EXPIRE) recreates the key - register() ran once at
birth, so the live worker keeps taking jobs while invisible to the dashboard. The
mixin re-runs register() on every heartbeat (idempotent SADD) and rebuilds the
identity hash if the key came back as a partial, so a recovered worker rejoins.
The re-registration always runs on the worker's OWN pipeline, never on the pipeline
passed to heartbeat: rq 2.7.0's maintain_heartbeats reads its results by FIXED position
(``results[7]`` is job.heartbeat's HSET), so injecting a command would make RQ inspect
the wrong result and potentially delete a RUNNING job's key. This override keeps the
same two heartbeat commands but writes HSET before EXPIRE. RQ 2.7.0 does the reverse,
so an expired key can otherwise be recreated without a TTL. It also records
last_heartbeat on the instance; rq 2.7.0 only creates that attribute in refresh(),
which caused issue #799.

The recovery transaction restores a missing identity before atomically registering
the worker and setting its TTL. The handler re-raises StopRequested because it carries
RQ's warm-shutdown request out of the signal handler and must not be swallowed by a
blanket except.

The Windows beat thread shares a lock with execution preparation and cleanup. Without
it, a beat can see an execution before its creation transaction commits, cleanup can
delete an execution before an already-started heartbeat recreates it, or an old job's
blocked beat thread can resume after the next job replaces self.execution. The stop
event is set before cleanup takes the lock, and execute_job does not return until the
thread exits, so heartbeat work cannot cross an execution boundary.

Main Features:
* ReregisterOnHeartbeatMixin: SADDs the worker back into rq:workers on each heartbeat
  and restores hostname/birth/queues if the key expired and was recreated partial.
* HeartbeatSimpleWorker: runs perform_job on the main thread while a daemon thread
  calls maintain_heartbeats, giving SimpleWorker the liveness signal the forking
  worker gets for free; also carries the re-registration mixin.
* The beat thread is serialized with execution cleanup and logs genuine Redis failures.
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
            if self.connection.hexists(self.key, 'death'):
                return
            identity_missing = not self.connection.hexists(self.key, 'birth')
            with self.connection.pipeline() as repair:
                if identity_missing:
                    repair.hset(self.key, mapping=self._identity_mapping())
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
        }
        state = self.get_state()
        if state:
            mapping['state'] = state
        return mapping


class ReregisteringWorker(ReregisterOnHeartbeatMixin, Worker):
    pass


class HeartbeatSimpleWorker(ReregisterOnHeartbeatMixin, SimpleWorker):
    def _refresh_job_heartbeat(self, job, stop, heartbeat_lock):
        with heartbeat_lock:
            if stop.is_set() or self.execution is None:
                return
            try:
                self.maintain_heartbeats(job)
            except StopRequested:
                raise
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
