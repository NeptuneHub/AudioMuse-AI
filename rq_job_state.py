# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The ONE place that decides whether an RQ job is alive, terminal or abandoned.

Every caller that used to hand-roll its own status list now asks here, because
each hand-rolled list omitted a different status and every omission became a
stuck task: the analysis monitor forgot ``deferred``, the /api/status merge and
the cancel-all sweep forgot ``stopped``, and the orphan reaper treated a mere
``Job.fetch`` success as proof of life even though RQ keeps failed, stopped,
canceled and finished jobs in their terminal registries.

The status vectors below are PRIVATE on purpose. Callers must never re-read them
or compare status strings themselves, because that is exactly how the copies
drifted apart in the first place; they ask the predicates and ``probe_job``
instead, so adding a status to RQ means editing one tuple in this file.

Two vocabularies live here and must not be merged. ``_*_STATUSES`` hold RQ's OWN
status strings, the literal values RQ writes into Redis, which are inputs we can
only match against. ``_STATE_*`` are this module's verdicts, the answers it
returns, and three of them deliberately correspond to no RQ status at all:
``missing`` means the job record does not exist, ``unknown`` means the probe
itself failed, and ``abandoned`` means RQ still claims ``started`` while the
worker behind it is provably gone. The remaining two, ``alive`` and ``terminal``,
are names for a whole tuple rather than members of one. Putting a verdict inside
a status tuple would therefore assert that RQ can report a job as literally
"alive", and would make ``is_alive_status('alive')`` answer True for a status RQ
never emits.

The hard case is a job whose worker was SIGKILLed mid-run: RQ leaves it in
``started``. ``rq_janitor`` already runs ``StartedJobRegistry.cleanup`` every 10
seconds, and RQ scores every started-registry entry finitely from the moment the
execution is created (heartbeat TTL: monitoring interval + 60s on the forking
worker, the 420s worker TTL on SimpleWorker), so that ladder retires the ordinary
Linux crash on its own about 90 seconds after the last beat. Two gaps remain, and
they are why this module judges liveness by the job's own ``last_heartbeat``
instead of the registry score: on the Windows standalone the SimpleWorker's 420s
score outlives the 300s heartbeat threshold, so only the heartbeat path acts
inside that window; and rq's cleanup runs failure callbacks under a Unix-only
death penalty (``signal.SIGALRM``), so on Windows a single expired job WITH a
failure callback makes the whole ladder raise before it retires anything, forever
- the heartbeat path runs no callbacks and still recovers those jobs.

Abandoned is deliberately its own answer and NOT a synonym for dead: an abandoned
main task must be RESTARTED, not buried. ``retries_left`` (from the
``Retry(max=N)`` the start endpoints already pass) is what bounds the restarts,
so a job that keeps killing its worker fails for real once its budget is spent
instead of looping forever.

The same ``retries_left`` is why cancelling needs ``forbid_retries``. Neither
rq's cleanup nor its abandoned-job path looks at a job's status before retrying,
and neither knows anything about the REVOKED row in Postgres, so a cancelled job
that still carried a retry budget was requeued by the janitor and re-ran against a
row that ``save_task_status`` refuses to move off REVOKED, i.e. hours of invisible
work. Zeroing the budget at cancel time is what makes "cancelled, failed and
completed tasks never restart" hold, and requeueing here drops the dead
started-registry entry first, exactly as rq's own failure path does, so cleanup
cannot re-process the stale entry and double-run the job.

Main Features:
* ``probe_job``: one Redis round trip, returning the job's state plus its raw RQ
  status, never raising. ``probe_jobs_many`` batches that over one pipelined
  fetch, falling back to per-job probes so one corrupt record cannot poison the
  whole batch.
* ``is_alive`` / ``is_terminal`` / ``is_abandoned`` / ``is_missing`` /
  ``is_unknown``: the functions callers ask about a probed state.
* ``is_alive_status`` / ``is_terminal_status`` / ``is_cancelled_status`` /
  ``is_running_status``: the same questions for callers that already hold a job
  or a status string.
* ``status_value``: normalizes enum, plain str, bytes and ``'JobStatus.QUEUED'``
  reprs to one lowercase token, across the rq versions this project pins.
* ``retry_abandoned_job``: restarts a job whose worker died, bounded by the job's
  own remaining retry budget. Returns a verdict (``RETRY_REQUEUED`` /
  ``RETRY_RECOVERED`` / ``RETRY_NO_BUDGET`` / ``RETRY_MISSING`` /
  ``RETRY_ERROR``) so the janitor can tell a spent budget from a transient error
  instead of failing live work on a Redis blip.
* ``unknown`` is a first-class answer: a Redis outage is NOT a dead job, so
  callers leave those rows alone instead of mass-failing live work.
"""

import logging
from datetime import datetime, timezone

from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq.queue import Queue

import config

logger = logging.getLogger(__name__)

_ALIVE_STATUSES = ('created', 'queued', 'started', 'deferred', 'scheduled')
_TERMINAL_STATUSES = ('canceled', 'failed', 'finished', 'stopped')
_CANCELLED_STATUSES = ('canceled', 'stopped')

_STATE_ALIVE = 'alive'
_STATE_TERMINAL = 'terminal'
_STATE_ABANDONED = 'abandoned'
_STATE_MISSING = 'missing'
_STATE_UNKNOWN = 'unknown'

_RUNNING_STATUS = 'started'

RETRY_REQUEUED = 'requeued'
RETRY_RECOVERED = 'recovered'
RETRY_NO_BUDGET = 'no-budget'
RETRY_MISSING = 'missing-job'
RETRY_ERROR = 'error'


def status_value(status):
    if status is None:
        return ''
    value = getattr(status, 'value', None)
    if value is None:
        value = status
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode('utf-8', 'replace')
    return str(value).strip().lower().rsplit('.', 1)[-1]


def is_alive_status(status):
    return status_value(status) in _ALIVE_STATUSES


def is_terminal_status(status):
    return status_value(status) in _TERMINAL_STATUSES


def is_cancelled_status(status):
    return status_value(status) in _CANCELLED_STATUSES


def is_running_status(status):
    return status_value(status) == _RUNNING_STATUS


def is_alive(state):
    return state == _STATE_ALIVE


def is_terminal(state):
    return state == _STATE_TERMINAL


def is_abandoned(state):
    return state == _STATE_ABANDONED


def is_missing(state):
    return state == _STATE_MISSING


def is_unknown(state):
    return state == _STATE_UNKNOWN


def heartbeat_age_seconds(job):
    last = getattr(job, 'last_heartbeat', None)
    if last is None:
        return None
    try:
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - last).total_seconds())
    except Exception:
        logger.debug("Unusable last_heartbeat on job", exc_info=True)
        return None


def _classify(status, job=None, abandoned_after=None):
    value = status_value(status)
    if value in _TERMINAL_STATUSES:
        return _STATE_TERMINAL, value
    if value not in _ALIVE_STATUSES:
        return _STATE_UNKNOWN, value
    if value != _RUNNING_STATUS or job is None:
        return _STATE_ALIVE, value
    if abandoned_after is None:
        abandoned_after = config.RQ_JOB_ABANDONED_SECONDS
    if abandoned_after <= 0:
        return _STATE_ALIVE, value
    age = heartbeat_age_seconds(job)
    if age is not None and age > abandoned_after:
        return _STATE_ABANDONED, value
    return _STATE_ALIVE, value


def probe_job(job_id, connection, abandoned_after=None):
    try:
        job = Job.fetch(job_id, connection=connection)
        status = job.get_status(refresh=False)
    except NoSuchJobError:
        return _STATE_MISSING, ''
    except Exception:
        logger.debug("Could not probe RQ job %s", job_id, exc_info=True)
        return _STATE_UNKNOWN, ''
    return _classify(status, job=job, abandoned_after=abandoned_after)


def probe_jobs_many(job_ids, connection, abandoned_after=None):
    if not job_ids:
        return {}
    try:
        jobs = Job.fetch_many(job_ids, connection=connection)
    except Exception:
        logger.debug("Batch job probe failed; probing one by one.", exc_info=True)
        return {
            job_id: probe_job(job_id, connection, abandoned_after=abandoned_after)
            for job_id in job_ids
        }
    results = {}
    for job_id, job in zip(job_ids, jobs):
        if job is None:
            results[job_id] = (_STATE_MISSING, '')
            continue
        try:
            status = job.get_status(refresh=False)
        except Exception:
            logger.debug("Could not read status of job %s", job_id, exc_info=True)
            results[job_id] = (_STATE_UNKNOWN, '')
            continue
        results[job_id] = _classify(status, job=job, abandoned_after=abandoned_after)
    return results


def forbid_retries(job_id, connection):
    try:
        job = Job.fetch(job_id, connection=connection)
    except NoSuchJobError:
        return False
    except Exception:
        logger.debug("Could not fetch job %s to clear its retries", job_id, exc_info=True)
        return False
    if not job.retries_left:
        return False
    try:
        job.retries_left = 0
        job.save(include_meta=False, include_result=False)
    except Exception:
        logger.exception("Could not clear the retry budget of job %s", job_id)
        return False
    logger.info(
        "Cleared the retry budget of job %s so nothing can requeue it.", job_id
    )
    return True


def _drop_started_registry_entries(job, connection):
    try:
        registry_key = job.started_job_registry.key
        members = connection.zrange(registry_key, 0, -1)
    except Exception:
        logger.debug(
            "Could not read the started registry for job %s", job.id, exc_info=True
        )
        return
    prefix = f'{job.id}:'
    stale = []
    for raw in members or ():
        member = (
            raw.decode('utf-8', 'replace')
            if isinstance(raw, (bytes, bytearray)) else str(raw)
        )
        if member == job.id or member.startswith(prefix):
            stale.append(member)
    if not stale:
        return
    try:
        connection.zrem(registry_key, *stale)
    except Exception:
        logger.debug(
            "Could not drop started-registry entries for job %s", job.id, exc_info=True
        )


def retry_abandoned_job(job_id, connection, abandoned_after=None):
    try:
        job = Job.fetch(job_id, connection=connection)
        status = job.get_status(refresh=False)
    except NoSuchJobError:
        return RETRY_MISSING
    except Exception:
        logger.debug("Could not fetch job %s to retry it", job_id, exc_info=True)
        return RETRY_ERROR
    state, _value = _classify(status, job=job, abandoned_after=abandoned_after)
    if not is_abandoned(state):
        logger.info(
            "Job %s recovered on its own (now %s); not requeueing it.", job_id, status
        )
        return RETRY_RECOVERED
    if not job.retries_left:
        return RETRY_NO_BUDGET
    try:
        queue = Queue(job.origin, connection=connection)
        _drop_started_registry_entries(job, connection)
        with connection.pipeline() as pipeline:
            job.retry(queue, pipeline)
            pipeline.execute()
    except Exception:
        logger.exception("Could not requeue abandoned job %s", job_id)
        return RETRY_ERROR
    logger.warning(
        "Requeued abandoned job %s on queue %s; %s retry attempt(s) left.",
        job_id, job.origin, job.retries_left,
    )
    return RETRY_REQUEUED
