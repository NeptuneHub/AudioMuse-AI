# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Worker class selection, with a heartbeat that keeps long jobs alive on Windows.

RQ's forking ``Worker`` refreshes a running job's started-registry score from its
monitor loop, so the RQ janitor can tell a live job from a dead one. ``SimpleWorker``
(the only option on Windows, which cannot fork) runs the job in-process and has no
monitor loop, so it never refreshes that score: it is written once as
``now + DEFAULT_WORKER_TTL`` (420s) and then goes stale, even for ``job_timeout=-1``.
The janitor's ``started_registry.cleanup()`` then expires the entry and either
re-queues the job or fails it, so no analysis, clustering, cleaning or sweep running
longer than seven minutes could ever complete on Windows.

Main Features:
* HeartbeatSimpleWorker: runs perform_job on the main thread while a daemon thread
  calls maintain_heartbeats, giving SimpleWorker the liveness signal the forking
  worker gets for free.
* WorkerClass: HeartbeatSimpleWorker on win32, the stock forking Worker elsewhere.
"""

import logging
import sys
import threading

from rq import SimpleWorker, Worker

logger = logging.getLogger(__name__)


class HeartbeatSimpleWorker(SimpleWorker):
    """SimpleWorker that keeps refreshing the started-registry score while it works.

    The heartbeat must NOT be replaced by returning -1 from get_heartbeat_ttl:
    Execution.save would then call Redis EXPIRE with a negative TTL, which deletes
    the key immediately and makes the job look dead the instant it starts.
    """

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


WorkerClass = HeartbeatSimpleWorker if sys.platform == 'win32' else Worker
