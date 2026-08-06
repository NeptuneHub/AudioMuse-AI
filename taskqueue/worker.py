# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""One worker process: claim a task, run it, finish it, repeat.

Run as ``python -m taskqueue.worker --queue high`` or ``--queue default``. Two
of these per container and N containers all point at the same Postgres and need
no coordination between them, because the claim is a single UPDATE whose
subquery takes ``FOR UPDATE SKIP LOCKED``: every worker races, exactly one wins,
and the losers move on to the next row instead of blocking.

The job runs IN THIS PROCESS. That is why cancelling means ending the process
(see ``taskqueue.process.stop_hard``) and why the same mechanism recycles the
worker after ``QUEUE_MAX_JOBS`` jobs, which is this project's guard against a
long-lived process accumulating leaks. Running in-process also keeps the worker
warm: the ONNX models are loaded once and stay loaded for the life of the
worker rather than being re-loaded per album.

Liveness is the advisory lock this process holds on the task it is running, on
its own connection, for exactly as long as it runs it. Nothing is written every
few seconds and nothing times out; if this process dies the connection drops and
Postgres releases the lock immediately, so no heartbeat, job-state mirror or
registry bookkeeping is needed to notice.

That lock is only as durable as the connection under it, which is why
``ensure_hold`` exists. Whatever drops the listener - a Postgres restart, a
failover, a recycled PgBouncer server connection - drops the idle claim
connection too, and the lock goes with it while this process is still computing.
Reclaim would then find an unlocked RUNNING row and hand a live task to somebody
else, so the lock is RETAKEN the moment the listener comes back rather than
covered by a grace period long enough to outlast the outage. The claim
connection also carries the same short TCP keepalives the listener does, on the
server side as well as the client side, because it is Postgres that releases the
lock and a worker whose host vanished without a FIN/RST otherwise kept it for
the better part of two hours.

The thread caps at the bottom of this module must be applied BEFORE numpy, ONNX
or BLAS are imported anywhere, so they run at module import and every heavy
import is deferred into ``main``. The frozen launchers import only stdlib and
``native_common.frozen_children`` before dispatching here, so this ordering
holds in the native builds too.

Main Features:
* Claim/drain loop that keeps pulling while work exists, and blocks on LISTEN when it does not
* A cancel notification for the task this worker holds ends the process tree in about 50ms
* Terminal rows are a safety net: a task normally writes its own, and a child row is deleted
* Boot reclaims tasks whose worker died, bounded by QUEUE_MAX_ATTEMPTS
* Schema bootstrap runs under the SAME advisory lock init_db takes, so a worker
  and Flask booting together cannot run the DDL migration concurrently
* A database connectivity error never becomes a terminal row: the tasks re-raise
  those deliberately, the row stays RUNNING, and reclaim requeues it once the
  advisory lock drops. Writing FAIL there let a two-second Postgres blip kill a
  job for good, because nothing ever claims or reclaims a FAIL row again.

The boot reclaim passes a grace of zero on purpose. A process that has just
started is provably not the one that abandoned any RUNNING row, and the advisory
lock plus the worker's own live session in ``pg_stat_activity`` already answer
liveness exactly - so waiting out the normal grace only guaranteed that the pass
found nothing, which is precisely the case it was written for.
"""

import logging
import os
import sys
import threading
import time

import queue_names

_QUEUE_FLAG = '--queue'


def _queue_from_argv(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    for index, arg in enumerate(argv):
        if arg == _QUEUE_FLAG and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(_QUEUE_FLAG + '='):
            return arg.split('=', 1)[1]
    return queue_names.QUEUE_DEFAULT


def _apply_thread_caps(queue):
    cpu_count = os.cpu_count() or 2
    if queue == queue_names.QUEUE_HIGH:
        cap = max(1, cpu_count // 3)
    else:
        cap = max(2, cpu_count // 2)
    for key in (
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    ):
        os.environ[key] = str(cap)
    os.environ.setdefault('GOMP_SPINCOUNT', '0')
    os.environ.setdefault('OMP_WAIT_POLICY', 'passive')
    return cap


_UNPARSED_QUEUE = _queue_from_argv()
if _UNPARSED_QUEUE not in queue_names.QUEUE_NAMES:
    raise SystemExit(
        f"Unknown queue {_UNPARSED_QUEUE!r}; expected one of {queue_names.QUEUE_NAMES}"
    )
QUEUE = _UNPARSED_QUEUE
os.environ['AUDIOMUSE_ROLE'] = 'worker'
THREAD_CAP = _apply_thread_caps(QUEUE)
print(f"{QUEUE} worker CPU thread cap = {THREAD_CAP}")

import config  # noqa: E402
from . import sql  # noqa: E402
from .listen import Listener  # noqa: E402
from .process import stop_hard, sweep_stale_temp_dirs  # noqa: E402

logger = logging.getLogger(__name__)


APPLICATION_NAME_LIMIT = 63


def build_identity(queue, hostname, pid, suffix_len=None, token=None):
    if suffix_len is None:
        suffix_len = len(sql.WORKER_LISTEN_SUFFIX)
    prefix = f"{sql.WORKER_IDENTITY_PREFIX}{queue}-"
    tail = f"-{pid}-{token or os.urandom(2).hex()}"
    budget = APPLICATION_NAME_LIMIT - suffix_len - len(prefix) - len(tail)
    if budget < 1:
        return f"{prefix}{tail.lstrip('-')}"[:APPLICATION_NAME_LIMIT - suffix_len]
    safe_hostname = hostname.encode('utf-8')[:budget].decode('utf-8', 'ignore')
    return f"{prefix}{safe_hostname}{tail}"


class Worker:
    def __init__(self, queue):
        self.queue = queue
        self.identity = build_identity(queue, sql.hostname(), os.getpid())
        self.max_jobs = (
            config.QUEUE_MAX_JOBS_HIGH if queue == sql.QUEUE_HIGH else config.QUEUE_MAX_JOBS
        )
        self._wake = threading.Event()
        self._held_task_id = None
        self._held_parent_id = None
        self._held_attempts = None
        self._conn = None
        self._listener = None
        self._jobs_done = 0
        self._shared_cache = {}
        self._claim_txn = threading.Lock()

    def reconnect(self):
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            logger.debug("Closing the dropped worker connection failed", exc_info=True)
        self._conn = None
        time.sleep(config.QUEUE_RECONNECT_DELAY_SECONDS)
        try:
            self.connect()
        except Exception:
            logger.exception("Could not reconnect; will retry")

    def connect(self):
        from database import connect_raw

        self._conn = connect_raw(
            application_name=self.identity,
            keepalive_idle_seconds=config.QUEUE_KEEPALIVE_IDLE_SECONDS,
            keepalive_interval_seconds=config.QUEUE_KEEPALIVE_INTERVAL_SECONDS,
            keepalive_count=config.QUEUE_KEEPALIVE_COUNT,
        )
        return self._conn

    def on_notify(self, channel, payload):
        if channel == sql.CHANNEL_JOB:
            if payload == self.queue:
                self._wake.set()
            return
        if channel == sql.CHANNEL_RECLAIM:
            self.on_reclaimed(payload)
            return
        if channel != sql.CHANNEL_CANCEL:
            return
        held = self._held_task_id
        if held is None:
            return
        if payload in (sql.CANCEL_ALL, held, self._held_parent_id):
            stop_hard(f"task {held} was cancelled")

    def on_reclaimed(self, payload):
        notice = sql.decode_reclaim(payload)
        if notice is None:
            return
        held = self._held_task_id
        if held is None or notice['task_id'] != held:
            return
        if notice['worker_id'] != self.identity or notice['attempts'] != self._held_attempts:
            return
        stop_hard(f"task {held} was reclaimed while this worker was still running it")

    def on_listener_ready(self, conn):
        with self._claim_txn:
            held = self._held_task_id
            if held is None:
                return
            with conn.cursor() as cur:
                owner = sql.owner_of(cur, held)
            if owner is None:
                stop_hard(f"task {held} no longer exists; this worker must not continue it")
                return
            if owner['worker_id'] != self.identity or owner['status'] == config.TASK_STATUS_NEW:
                stop_hard(f"task {held} was taken from this worker while it was not listening")
                return
            self.ensure_hold(held)

    def ensure_hold(self, task_id):
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
            self._conn.commit()
            return True
        except Exception:
            logger.warning(
                "The claim connection for %s is gone; retaking its lock", task_id, exc_info=True
            )
        self._safe_rollback()
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            logger.debug("Closing the dead claim connection failed", exc_info=True)
        self._conn = None
        try:
            self.connect()
            with self._conn.cursor() as cur:
                retaken = sql.try_hold(cur, task_id)
            self._conn.commit()
        except Exception:
            logger.exception("Could not reopen the claim connection for %s", task_id)
            return False
        if not retaken:
            stop_hard(f"task {task_id} was reclaimed while this worker's connection was down")
        return True

    def start_listener(self):
        self._listener = Listener(
            (sql.CHANNEL_JOB, sql.CHANNEL_CANCEL, sql.CHANNEL_RECLAIM),
            self.on_notify,
            application_name=f"{self.identity}{sql.WORKER_LISTEN_SUFFIX}",
            name=f"listen-{self.queue}",
            on_ready=self.on_listener_ready,
        )
        self._listener.start()

    def claim(self):
        with self._claim_txn:
            try:
                with self._conn.cursor() as cur:
                    job = sql.claim(cur, self.queue, time.time(), worker_id=self.identity)
                    if job is not None:
                        sql.hold(cur, job['task_id'])
                        self._held_task_id = job['task_id']
                        self._held_parent_id = job['parent_task_id']
                        self._held_attempts = job['attempts']
                self._conn.commit()
                return job
            except Exception:
                self._clear_held()
                self._safe_rollback()
                raise

    def _clear_held(self):
        self._held_task_id = None
        self._held_parent_id = None
        self._held_attempts = None

    def run_forever(self):
        while True:
            try:
                job = self.claim()
            except Exception:
                logger.exception(
                    "Claim failed; reconnecting in %ss", config.QUEUE_RECONNECT_DELAY_SECONDS
                )
                self.reconnect()
                continue
            if job is None:
                self._shared_cache = {}
                self._wake.wait(config.QUEUE_POLL_INTERVAL_SECONDS)
                self._wake.clear()
                continue
            try:
                self.run_job(job)
            except Exception:
                logger.exception("Bookkeeping for %s failed; reconnecting", job['task_id'])
                self.reconnect()
            self._jobs_done += 1
            if self.max_jobs and self._jobs_done >= self.max_jobs:
                stop_hard(f"recycling after {self._jobs_done} jobs")

    def run_job(self, job):
        from . import resolve_func, set_current_task_id

        task_id = job['task_id']
        set_current_task_id(task_id)
        logger.info(
            "Running %s (%s) after %d worker loss(es) of an allowed %d",
            task_id, job['func'], job['attempts'], job['max_attempts'],
        )
        started = time.time()
        result = None
        try:
            self.hydrate_config()
            func = resolve_func(job['func'])
            result = func(*job['args'], **self.hydrate_shared(job['kwargs']))
        except Exception as exc:
            logger.exception("Task %s raised", task_id)
            outcome, summary = config.TASK_STATUS_FAIL, _error_summary(exc)
            if _is_connectivity_error(exc):
                logger.warning(
                    "Task %s hit a database connectivity error; leaving its row for "
                    "the queue to requeue instead of failing it.", task_id,
                )
                outcome = None
        else:
            outcome, summary = config.TASK_STATUS_SUCCESS, None
        try:
            with self._claim_txn:
                if outcome is not None:
                    self.finalize(job, outcome, summary, result=result)
                set_current_task_id(None)
                self._clear_held()
                try:
                    with self._conn.cursor() as cur:
                        sql.release(cur, task_id)
                    self._conn.commit()
                except Exception:
                    logger.exception("Could not release the hold on %s", task_id)
        finally:
            logger.info("Finished %s in %.1fs", task_id, time.time() - started)

    def finalize(self, job, status, error, result=None):
        task_id = job['task_id']
        for attempt in (1, 2):
            try:
                if self._conn is None or self._conn.closed:
                    logger.warning(
                        "The claim connection dropped while %s ran; reconnecting to finish it",
                        task_id,
                    )
                    self.connect()
                with self._conn.cursor() as cur:
                    row = sql.current_row(cur, task_id)
                    if row is None or row['status'] != config.TASK_STATUS_RUNNING:
                        return
                    details = {'message': _final_message(status, error)}
                    if error:
                        details['error'] = error
                    if isinstance(result, dict):
                        details['final_summary_details'] = result
                    written = sql.finish_task(
                        cur, task_id, status, details, time.time(), worker_id=self.identity
                    )
                if written is None:
                    logger.error(
                        "Refusing to finish %s: the row is no longer this worker's. It was "
                        "reclaimed and restarted elsewhere while this process was still on it.",
                        task_id,
                    )
            except Exception:
                self._safe_rollback()
                if attempt == 1:
                    logger.warning(
                        "Could not write the terminal row for %s; retrying once on a "
                        "fresh connection", task_id, exc_info=True,
                    )
                    try:
                        if self._conn is not None:
                            self._conn.close()
                    except Exception:
                        logger.debug("Closing the dead claim connection failed", exc_info=True)
                    self._conn = None
                    continue
                logger.exception("Could not write the terminal row for %s", task_id)
            else:
                self._safe_commit()
            return

    def _safe_rollback(self):
        try:
            if self._conn is not None and not self._conn.closed:
                self._conn.rollback()
        except Exception:
            logger.debug("Rollback on the worker connection failed", exc_info=True)

    def _safe_commit(self):
        try:
            self._conn.commit()
        except Exception:
            logger.exception("Could not commit the worker connection")
            self._safe_rollback()

    def hydrate_shared(self, kwargs):
        from . import SHARED_KWARG_REF

        ref = kwargs.get(SHARED_KWARG_REF)
        if not ref:
            return kwargs
        restored = {key: value for key, value in kwargs.items() if key != SHARED_KWARG_REF}
        owner = ref['owner']
        for name, token in ref['tokens'].items():
            restored[name] = self.shared_body(owner, token)
        return restored

    def shared_body(self, owner, token):
        cached = self._shared_cache.get(token)
        if cached is not None:
            return cached
        with self._conn.cursor() as cur:
            body = sql.get_shared(cur, owner, token)
        self._conn.commit()
        if len(body) <= config.QUEUE_SHARED_CACHE_MAX_BYTES:
            self._shared_cache = {token: body}
        else:
            self._shared_cache = {}
            logger.info(
                "Shared payload %s is %d bytes; reading it per job instead of caching it.",
                token, len(body),
            )
        return body

    def hydrate_config(self):
        try:
            from tasks.setup_manager import hydrate_worker_config

            hydrate_worker_config()
        except Exception:
            logger.exception("Could not refresh the worker configuration; using what is loaded")

    def ensure_schema(self):
        from database import _SCHEMA_ADVISORY_LOCK

        with self._conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (_SCHEMA_ADVISORY_LOCK,))
            try:
                sql.ensure_schema(cur)
            finally:
                cur.execute("SELECT pg_advisory_unlock(%s)", (_SCHEMA_ADVISORY_LOCK,))
        self._conn.commit()

    def reclaim_orphans(self):
        from .maintenance import reclaim_orphans

        return reclaim_orphans(self._conn, grace_seconds=0)


def _final_message(status, error):
    if status == config.TASK_STATUS_SUCCESS:
        return "Task completed successfully."
    return error or "Task failed. Check the container logs for details."


def _is_connectivity_error(exc):
    try:
        from psycopg2 import InterfaceError, OperationalError
    except Exception:
        return False
    return isinstance(exc, (OperationalError, InterfaceError))


def _error_summary(exc):
    text = str(exc).strip() or exc.__class__.__name__
    return text[:500]


def main():
    from app_logging import configure_logging

    configure_logging()
    from config import APP_VERSION, TEMP_DIR

    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
    except OSError:
        logger.warning("Could not create TEMP_DIR %s", TEMP_DIR)

    worker = Worker(QUEUE)
    logger.info("Worker %s starting (AudioMuse-AI %s)", worker.identity, APP_VERSION)

    from tasks.setup_manager import hydrate_worker_config

    hydrate_worker_config()

    try:
        from plugin.manager import boot as plugin_boot

        plugin_boot('worker')
    except Exception:
        logger.exception("Plugin subsystem worker boot failed; continuing without plugins")

    try:
        from numeric_bootstrap import warmup_scipy_longdouble

        warmup_scipy_longdouble()
    except Exception:
        logger.exception("Numeric warmup failed; continuing")

    sweep_stale_temp_dirs(TEMP_DIR)
    worker.connect()
    worker.ensure_schema()
    worker.reclaim_orphans()
    worker.start_listener()
    logger.info("Worker %s ready; recycling after %s jobs.", worker.identity, worker.max_jobs)
    worker.run_forever()


if __name__ == '__main__':
    main()
