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
The re-registration always runs on the worker's OWN connection, never on the
pipeline a heartbeat was given: rq 2.7.0's maintain_heartbeats reads its pipeline
results by FIXED position (``results[7]`` is job.heartbeat's HSET), so any command
injected into that pipeline shifts the slot onto an EXPIRE - which returns 1 for a
live key - and RQ would delete the RUNNING job's key on every monitor beat. The
identity mapping is built here (the same fields register_birth writes) because
Worker.serialize does not exist on rq 2.7.0.

Main Features:
* ReregisterOnHeartbeatMixin: SADDs the worker back into rq:workers on each heartbeat
  and restores hostname/birth/queues if the key expired and was recreated partial.
* HeartbeatSimpleWorker: runs perform_job on the main thread while a daemon thread
  calls maintain_heartbeats, giving SimpleWorker the liveness signal the forking
  worker gets for free; also carries the re-registration mixin.
* WorkerClass: HeartbeatSimpleWorker on win32, a re-registering forking Worker elsewhere.
"""

import logging
import sys
import threading

from rq import SimpleWorker, Worker, worker_registration
from rq.utils import now, utcformat

logger = logging.getLogger(__name__)


class ReregisterOnHeartbeatMixin:
    def heartbeat(self, timeout=None, pipeline=None):
        super().heartbeat(timeout=timeout, pipeline=pipeline)
        try:
            worker_registration.register(self)
            if self.connection and self.key:
                if not self.connection.hexists(self.key, 'birth'):
                    self.connection.hset(self.key, mapping=self._identity_mapping())
                    worker_ttl = getattr(self, 'worker_ttl', 420)
                    self.connection.expire(self.key, worker_ttl + 60)
        except Exception:
            logger.exception("Worker %s: re-registration on heartbeat failed", self.name)

    def _identity_mapping(self):
        stamp = utcformat(getattr(self, 'last_heartbeat', None) or now())
        birth_date = getattr(self, 'birth_date', None)
        try:
            queue_str = ','.join(self.queue_names())
        except Exception:
            queue_str = ''
        return {
            'birth': utcformat(birth_date) if birth_date else stamp,
            'last_heartbeat': stamp,
            'queues': queue_str,
            'pid': getattr(self, 'pid', None) or 0,
            'hostname': getattr(self, 'hostname', None) or '',
            'ip_address': getattr(self, 'ip_address', None) or '',
            'version': getattr(self, 'version', None) or '',
            'python_version': getattr(self, 'python_version', None) or '',
        }


class ReregisteringWorker(ReregisterOnHeartbeatMixin, Worker):
    pass


class HeartbeatSimpleWorker(ReregisterOnHeartbeatMixin, SimpleWorker):
    def execute_job(self, job, queue):
        stop = threading.Event()

        def beat():
            interval = max(1, int(self.job_monitoring_interval))
            while not stop.wait(interval):
                try:
                    self.maintain_heartbeats(job)
                except Exception:
                    logger.exception("Heartbeat refresh failed for job %s", job.id)

        beater = threading.Thread(
            target=beat, name=f"rq-heartbeat-{job.id}", daemon=True
        )
        beater.start()
        try:
            return super().execute_job(job, queue)
        finally:
            stop.set()


WorkerClass = HeartbeatSimpleWorker if sys.platform == 'win32' else ReregisteringWorker
