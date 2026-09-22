# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""One worker process: claim a task, run it, finish it, repeat.

Run as python -m taskqueue.worker --queue high|default. The claim is a single
UPDATE whose subquery takes FOR UPDATE SKIP LOCKED, so N workers racing need no
coordination. Liveness is the advisory lock on the task's own connection: it
dies with the process, so no heartbeat is needed, and ensure_hold retakes it as
soon as the listener reconnects, before reclaim can move a running task. Thread
caps must be applied BEFORE numpy/ONNX/BLAS import, so heavy imports are
deferred into main; declare_worker_role runs before import config likewise.

Main Features:
* Claim/drain loop blocking on LISTEN when idle, recycling after QUEUE_MAX_JOBS
* In the container the job runs in a FORKED CHILD that reports over a pipe and
  leaves through os._exit, giving the OS back every byte it allocated. Only the
  parent touches the claim connection and the lock, config is hydrated before
  the fork, and the child binds itself to the parent's death (PR_SET_PDEATHSIG,
  or a getppid watchdog) so no orphan writes after reclaim. A cancel ends the
  whole tree; a stopping worker claims nothing
* Frozen native builds run the job in-process (macOS cannot fork and keep CoreML
  alive, Windows cannot fork) and unload its models afterwards
* The claim connection is re-checked at every listener poll: a dropped Postgres
  is reopened and the lock retaken; a lock gone to a reclaim ends this worker
* Boot reclaims orphans bounded by QUEUE_MAX_ATTEMPTS; a lost connection
  (SQLSTATE 08, 57Pxx) requeues uncharged UNCHARGED_REQUEUE_LIMIT times
* The queue writes EVERY terminal row and decides EVERY retry (taskqueue.retry):
  a raise is requeued with backoff, TaskFailed fails at once, TaskCancelled is
  revoked, a killed child is retried. The requeue waits until the hold is
  dropped, whose notice would otherwise end this very worker
* The terminal row (task message plus log) is COMMITTED before task_history and
  the recap collapse, on their own transactions
* A payload that does not match the signature, a func outside ALLOWED_FUNCS and
  a vanished shared payload are permanent failures, never retries
* Every FAIL carries a record from error.error_manager, and a child that died
  without reporting is diagnosed from its signal: SIGKILL is ERR_OUT_OF_MEMORY
  when /dev/kmsg names this pid, else the cgroup oom counter decides; SIGSEGV
  and friends are ERR_PROCESS_CRASHED, anything else ERR_JOB_PROCESS_DIED
"""

import errno
import inspect
import logging
import os
import pickle
import re
import signal
import sys
import threading
import time

import queue_names
import service_roles

from cpu_budget import detect_cpu_count
from error import error_manager
from error.error_dictionary import (
    ERR_DB_CONNECTION,
    ERR_JOB_PROCESS_DIED,
    ERR_OUT_OF_MEMORY,
    ERR_PROCESS_CRASHED,
    ERR_WORKER_LOST,
    UNKNOWN_ERROR_CODE,
)

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
    if queue == queue_names.QUEUE_HIGH:
        cpu_count, source = detect_cpu_count(
            os.cpu_count() or 1, 1, label='High-priority worker'
        )
        cap = max(1, cpu_count // 3)
    else:
        cpu_count, source = detect_cpu_count(
            os.cpu_count() or 2, 2, label='Default worker'
        )
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
    print(f"{queue} worker CPU thread cap = {cap} ({cpu_count} CPUs from {source})")
    return cap


_UNPARSED_QUEUE = _queue_from_argv()
if _UNPARSED_QUEUE not in queue_names.QUEUE_NAMES:
    raise SystemExit(
        f"Unknown queue {_UNPARSED_QUEUE!r}; expected one of {queue_names.QUEUE_NAMES}"
    )
QUEUE = _UNPARSED_QUEUE
service_roles.declare_worker_role(force=True)
THREAD_CAP = _apply_thread_caps(QUEUE)

import config  # noqa: E402
from . import error_summary as _error_summary, failure_record  # noqa: E402
from . import ERROR_AT_CLAIM_UNREAD as _UNREAD  # noqa: E402
from . import terminal_details as _terminal_details  # noqa: E402
from . import retry  # noqa: E402
from . import sql  # noqa: E402
from .errors import TaskCancelled, TaskFailed  # noqa: E402
from .listen import Listener  # noqa: E402
from .process import stop_hard, stopping_reason, sweep_stale_temp_dirs  # noqa: E402

logger = logging.getLogger(__name__)


APPLICATION_NAME_LIMIT = 63

UNCHARGED_REQUEUE_LIMIT = 3

_OPTIONAL_JOB_MODELS = (
    ('tasks.clap_analyzer', 'is_clap_model_loaded', 'unload_clap_model'),
    ('lyrics', 'is_lyrics_loaded', 'unload_lyrics_models'),
)


def build_identity(queue, hostname, pid):
    suffix_len = len(sql.WORKER_LISTEN_SUFFIX)
    prefix = f"{sql.WORKER_IDENTITY_PREFIX}{queue}-"
    tail = f"-{pid}-{os.urandom(2).hex()}"
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
        self._abandoned = []
        self._uncharged = {}
        self._claim_txn = threading.Lock()
        self._fork_jobs = hasattr(os, 'fork') and not getattr(sys, 'frozen', False)

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
        with self._claim_txn:
            held = self._held_task_id
            if held is None:
                return
            if payload in (sql.CANCEL_ALL, held, self._held_parent_id):
                stop_hard(f"task {held} was cancelled")

    def on_reclaimed(self, payload):
        notice = sql.decode_reclaim(payload)
        if notice is None:
            return
        with self._claim_txn:
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
                row = sql.current_row(cur, held)
            if row is None:
                stop_hard(f"task {held} no longer exists; this worker must not continue it")
                return
            if row['worker_id'] != self.identity or row['status'] == config.TASK_STATUS_NEW:
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

    def on_listener_idle(self):
        with self._claim_txn:
            held = self._held_task_id
            if held is not None:
                self.ensure_hold(held)

    def start_listener(self):
        self._listener = Listener(
            (sql.CHANNEL_JOB, sql.CHANNEL_CANCEL, sql.CHANNEL_RECLAIM),
            self.on_notify,
            application_name=f"{self.identity}{sql.WORKER_LISTEN_SUFFIX}",
            name=f"listen-{self.queue}",
            on_ready=self.on_listener_ready,
            on_idle=self.on_listener_idle,
        )
        self._listener.start()

    def claim(self):
        with self._claim_txn:
            if stopping_reason() is not None:
                return None
            try:
                with self._conn.cursor() as cur:
                    job = sql.claim(cur, self.queue, time.time(), worker_id=self.identity)
                    if job is not None:
                        job['error_at_claim'] = sql.current_details(cur, job['task_id']).get('error')
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

    def _forget_abandoned(self, task_id):
        self._uncharged.pop(task_id, None)
        logger.info(
            "Abandoned task %s is no longer this worker's RUNNING row; "
            "leaving it exactly as it is.", task_id,
        )
        return False

    def _requeue_charging_an_attempt(self, cur, task_id):
        if not self._still_mine(sql.current_row(cur, task_id)):
            return self._forget_abandoned(task_id)
        status = sql.requeue_or_fail(
            cur, task_id, time.time(),
            _terminal_details(
                config.TASK_STATUS_FAIL, _LOST_CONNECTION_SUMMARY,
                error_manager.build(ERR_DB_CONNECTION, _LOST_CONNECTION_SUMMARY),
            ),
        )
        if status == config.TASK_STATUS_NEW:
            logger.error(
                "Task %s has already been put back %d times for a lost database "
                "connection; this retry costs a worker-loss attempt.",
                task_id, UNCHARGED_REQUEUE_LIMIT,
            )
            return True
        self._uncharged.pop(task_id, None)
        if status is not None:
            logger.error(
                "Task %s ran out of worker-loss attempts while the database stayed "
                "unreachable; its row is now %s.", task_id, status,
            )
            return False
        return self._forget_abandoned(task_id)

    def _put_abandoned_back(self, cur, task_id):
        free_passes_used = self._uncharged.get(task_id, 0)
        if free_passes_used >= UNCHARGED_REQUEUE_LIMIT:
            return self._requeue_charging_an_attempt(cur, task_id)
        if not sql.requeue_uncharged(cur, task_id, worker_id=self.identity):
            return self._forget_abandoned(task_id)
        self._uncharged[task_id] = free_passes_used + 1
        logger.warning(
            "Task %s was abandoned to a lost database connection; it is queued "
            "again with no worker-loss attempt charged (%d of %d free retries).",
            task_id, free_passes_used + 1, UNCHARGED_REQUEUE_LIMIT,
        )
        return True

    def _wait_out_repeated_loss(self):
        already_lost = max(
            (self._uncharged.get(task_id, 0) for task_id in self._abandoned), default=0
        )
        if already_lost < 1:
            return
        delay = min(
            config.QUEUE_RECONNECT_DELAY_SECONDS * (2 ** (already_lost - 1)),
            config.QUEUE_POLL_INTERVAL_SECONDS,
        )
        logger.warning(
            "Waiting %.1fs before putting %d abandoned row(s) back; the database "
            "connection has already been lost %d time(s) on the same work.",
            delay, len(self._abandoned), already_lost,
        )
        time.sleep(delay)

    def requeue_abandoned(self):
        if not self._abandoned:
            return
        self._wait_out_repeated_loss()
        still_abandoned = []
        requeued = 0
        for task_id in self._abandoned:
            try:
                with self._claim_txn:
                    with self._conn.cursor() as cur:
                        put_back = self._put_abandoned_back(cur, task_id)
                    self._conn.commit()
            except Exception:
                logger.warning(
                    "Could not put abandoned task %s back on the queue yet; retrying "
                    "on the next loop", task_id, exc_info=True,
                )
                self._safe_rollback()
                still_abandoned.append(task_id)
                continue
            if put_back:
                requeued += 1
        self._abandoned = still_abandoned
        if not requeued:
            return
        try:
            with self._claim_txn:
                with self._conn.cursor() as cur:
                    sql.notify_job(cur, sql.QUEUE_HIGH)
                    sql.notify_job(cur, sql.QUEUE_DEFAULT)
                self._conn.commit()
        except Exception:
            logger.exception(
                "Could not wake the queues after requeueing an abandoned task"
            )
            self._safe_rollback()

    def run_forever(self):
        while True:
            self.requeue_abandoned()
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
        from . import set_current_task_id

        task_id = job['task_id']
        set_current_task_id(task_id)
        logger.info(
            "Running %s (%s), attempt %d; %d restart(s) allowed",
            task_id, job['func'], job['attempts'] + 1, job['max_attempts'],
        )
        started = time.time()
        outcome, summary, result = self._execute(job)
        if outcome is None:
            if task_id not in self._abandoned:
                self._abandoned.append(task_id)
        else:
            self._uncharged.pop(task_id, None)
        verdict = retry.decide(job, outcome)
        if outcome == retry.FAIL_RETRYABLE and verdict != retry.RETRY and stopping_reason() is None:
            _log_retry_verdict(job, summary, config.TASK_STATUS_FAIL, 0.0)
        try:
            with self._claim_txn:
                if outcome is not None and verdict != retry.RETRY:
                    self.finalize(job, retry.row_status(outcome), summary, result=result)
                set_current_task_id(None)
                self._clear_held()
                if verdict == retry.RETRY:
                    self._requeue_for_retry(job, summary, result)
                try:
                    with self._conn.cursor() as cur:
                        sql.release(cur, task_id)
                    self._conn.commit()
                except Exception:
                    logger.exception("Could not release the hold on %s", task_id)
        finally:
            logger.info("Finished %s in %.1fs", task_id, time.time() - started)

    def _execute(self, job):
        task_id = job['task_id']
        try:
            kwargs = self.hydrate_shared(job['kwargs'])
        except sql.SharedPayloadUnavailable as exc:
            return self._failure(job, TaskFailed(str(exc)))
        except Exception as exc:
            _log_raised(task_id, exc)
            return self._failure(job, exc)
        if self._fork_jobs:
            return self._run_in_child(job, kwargs)
        try:
            return self._attempt(job, kwargs)
        finally:
            self._unload_job_models()

    def _attempt(self, job, kwargs, hydrate=True):
        task_id = job['task_id']
        try:
            if hydrate:
                self.hydrate_config()
            func = _callable_for(job, kwargs)
            result = func(*job['args'], **kwargs)
        except Exception as exc:
            _log_raised(task_id, exc)
            return self._failure(job, exc)
        return config.TASK_STATUS_SUCCESS, None, result

    def _failure(self, job, exc):
        task_id = job['task_id']
        if _is_connectivity_error(exc):
            logger.warning(
                "Task %s lost its database connection; putting its row back "
                "on the queue instead of failing it.", task_id,
            )
            return None, _error_summary(exc), None
        if isinstance(exc, TaskCancelled):
            logger.info("Task %s stopped at its cancel check: %s", task_id, exc)
            return retry.REVOKED_BY_TASK, _error_summary(exc), None
        record = _error_record(job, exc)
        if isinstance(exc, TaskFailed):
            logger.error(
                "Task %s failed permanently and will not be retried: %s", task_id, exc
            )
            return retry.FAIL_PERMANENT, _error_summary(exc), record
        return retry.FAIL_RETRYABLE, _error_summary(exc), record

    def _requeue_for_retry(self, job, summary, record=None):
        task_id = job['task_id']
        delay = retry.backoff_seconds(job['attempts'] + 1)
        details = _terminal_details(config.TASK_STATUS_FAIL, summary, record)
        for attempt in (1, 2):
            try:
                if self._conn is None or self._conn.closed:
                    self.connect()
                with self._conn.cursor() as cur:
                    row = sql.current_row(cur, task_id)
                    if not self._still_mine(row):
                        (logger.error if stopping_reason() is None else logger.info)(
                            "Not retrying %s: its row is no longer this worker's RUNNING "
                            "row (%s), so something else already decided its fate.",
                            task_id, row and row['status'],
                        )
                        self._safe_rollback()
                        return
                    status = sql.requeue_or_fail(
                        cur, task_id, time.time(), details, delay_seconds=delay,
                    )
            except Exception:
                self._safe_rollback()
                if attempt == 1:
                    logger.warning(
                        "Could not requeue %s for a retry; retrying once on a fresh "
                        "connection", task_id, exc_info=True,
                    )
                    self._drop_claim_conn()
                    continue
                logger.exception(
                    "Could not requeue %s for a retry; its row stays RUNNING for "
                    "reclaim to pick up", task_id,
                )
                return
            self._safe_commit()
            _log_retry_verdict(job, summary, status, delay)
            return

    def _still_mine(self, row):
        return (
            row is not None
            and row['status'] == config.TASK_STATUS_RUNNING
            and row['worker_id'] in (None, self.identity)
        )

    def _run_in_child(self, job, kwargs):
        task_id = job['task_id']
        self.hydrate_config()
        try:
            read_fd, write_fd = os.pipe()
        except OSError as exc:
            logger.exception("Could not open the report pipe for %s", task_id)
            return retry.FAIL_RETRYABLE, _error_summary(exc), None
        parent_pid = os.getpid()
        oom_baseline = _oom_baseline()
        try:
            pid = os.fork()
        except OSError as exc:
            os.close(read_fd)
            os.close(write_fd)
            logger.exception("Could not fork the job process for %s", task_id)
            return retry.FAIL_RETRYABLE, _error_summary(exc), None
        if pid == 0:
            self._child_main(job, kwargs, read_fd, write_fd, parent_pid)
        oom_baseline['pid'] = pid
        os.close(write_fd)
        payload = b''
        killed_by_worker = False
        try:
            with os.fdopen(read_fd, 'rb') as pipe:
                payload = pipe.read()
        except Exception:
            logger.exception("Reading the job process report for %s failed", task_id)
            try:
                os.kill(pid, signal.SIGKILL)
                killed_by_worker = True
            except Exception:
                logger.debug("SIGKILL to the job process failed", exc_info=True)
        try:
            _, status = os.waitpid(pid, 0)
        except OSError:
            logger.exception("Could not reap the job process for %s", task_id)
            status = 0
        return self._child_outcome(
            task_id, status, payload, oom_baseline, killed_by_worker=killed_by_worker,
        )

    def _child_main(self, job, kwargs, read_fd, write_fd, parent_pid):
        exit_code = 1
        try:
            _bind_to_parent_death(parent_pid)
            os.close(read_fd)
            _close_inherited_sockets(self)
            payload = _encode_outcome(self._attempt(job, kwargs, hydrate=False))
            with os.fdopen(write_fd, 'wb') as pipe:
                pipe.write(payload)
            exit_code = 0
        except BaseException:
            try:
                logger.exception(
                    "The job process for %s could not report back", job['task_id']
                )
            except BaseException:
                pass
        finally:
            os._exit(exit_code)

    def _child_outcome(self, task_id, status, payload, oom_baseline=None,
                       killed_by_worker=False):
        if payload:
            try:
                outcome = pickle.loads(payload)
            except Exception:
                logger.exception("Could not decode the job process report for %s", task_id)
            else:
                if isinstance(outcome, tuple) and len(outcome) == 3:
                    return outcome
                logger.error("The job process report for %s is malformed", task_id)
        reason = stopping_reason()
        if reason is not None:
            logger.info("Task %s: its job process was stopped by this worker (%s)", task_id, reason)
            summary = f"The job process was stopped by this worker: {reason}"
            return retry.FAIL_RETRYABLE, summary, error_manager.build(ERR_WORKER_LOST, summary)
        if killed_by_worker:
            summary = (
                "The worker killed its job process because the job's report could not be "
                "read. Check the container logs for details."
            )
            record = error_manager.build(ERR_JOB_PROCESS_DIED, summary)
        else:
            summary, record = _child_death(status, oom_baseline)
        logger.error("Task %s: %s", task_id, summary)
        return retry.FAIL_RETRYABLE, summary, record

    def _unload_job_models(self):
        if not self._unload_resident_models():
            return
        try:
            from tasks.memory_utils import release_memory_to_os

            release_memory_to_os()
        except Exception:
            logger.debug("Worker job-end heap trim failed", exc_info=True)

    def _unload_resident_models(self):
        if 'tasks.analysis.song' in sys.modules:
            try:
                from tasks.analysis.song import cleanup_optional_models

                cleanup_optional_models(context="worker job end")
            except Exception:
                logger.debug("Worker job-end optional-model cleanup failed", exc_info=True)
            return True
        resident = False
        for module_name, is_loaded_name, unload_name in _OPTIONAL_JOB_MODELS:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            resident = True
            try:
                if getattr(module, is_loaded_name)():
                    getattr(module, unload_name)()
            except Exception:
                logger.debug(
                    "Worker job-end unload of %s failed", module_name, exc_info=True
                )
        return resident

    def _drop_claim_conn(self):
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            logger.debug("Closing the dead claim connection failed", exc_info=True)
        self._conn = None

    def _write_terminal_row(self, task_id, status, error, result, error_at_claim=_UNREAD):
        with self._conn.cursor() as cur:
            row = sql.current_row(cur, task_id)
            if row is None or row['status'] != config.TASK_STATUS_RUNNING:
                _report_foreign_terminal(task_id, status, row)
                return None
            previous = sql.current_details(cur, task_id)
            details = _terminal_details(
                status, error, result, previous=previous, error_at_claim=error_at_claim,
            )
            written = sql.finish_task(
                cur, task_id, status, details, time.time(), worker_id=self.identity,
            )
        if written is None:
            logger.error(
                "Refusing to finish %s: the row is no longer this worker's. It was "
                "reclaimed and restarted elsewhere while this process was still on it.",
                task_id,
            )
            return None
        return row, details

    def _record_and_collapse(self, task_id, row, status, details):
        try:
            from database import record_root_recap

            record_root_recap(
                self._conn, task_id, row.get('task_type'), row.get('parent_task_id'),
                status, details,
            )
        except Exception:
            logger.exception(
                "Finished %s as %s but could not record its history or collapse the "
                "table; the recap row itself is already committed", task_id, status,
            )
            self._safe_rollback()

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
                recap = self._write_terminal_row(
                    task_id, status, error, result,
                    error_at_claim=job.get('error_at_claim', _UNREAD),
                )
                self._conn.commit()
            except Exception:
                self._safe_rollback()
                if attempt == 1:
                    logger.warning(
                        "Could not write the terminal row for %s; retrying once on a "
                        "fresh connection", task_id, exc_info=True,
                    )
                    self._drop_claim_conn()
                    continue
                logger.exception("Could not write the terminal row for %s", task_id)
                return
            if recap is not None:
                self._record_and_collapse(task_id, recap[0], status, recap[1])
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
        with self._claim_txn:
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


def _callable_for(job, kwargs):
    from . import UnknownTaskFunction, resolve_func

    try:
        func = resolve_func(job['func'])
    except UnknownTaskFunction as exc:
        raise TaskFailed(
            f"{job['func']} is not a task function this worker may run; no retry "
            "can change that"
        ) from exc
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func
    try:
        signature.bind(*job['args'], **kwargs)
    except TypeError as exc:
        raise TaskFailed(
            f"{job['func']} cannot be called with the stored arguments ({exc}); no "
            "retry can change that"
        ) from exc
    return func


def _log_raised(task_id, exc):
    if isinstance(exc, (TaskFailed, TaskCancelled)):
        return
    logger.exception("Task %s raised", task_id)


def _log_retry_verdict(job, summary, status, delay):
    attempt = job['attempts'] + 1
    if status == config.TASK_STATUS_NEW:
        logger.warning(
            "Task %s failed on attempt %d (%s); restart %d of %d runs in %.0fs.",
            job['task_id'], attempt, summary, attempt, job['max_attempts'], delay,
        )
        return
    logger.error(
        "Task %s failed on attempt %d (%s) with no restart left of %d; "
        "the queue gave it %s.",
        job['task_id'], attempt, summary, job['max_attempts'], status,
    )


def _report_foreign_terminal(task_id, status, row):
    if row is None:
        logger.info(
            "Not finishing %s as %s: its row is gone, which is what a cancel does.",
            task_id, status,
        )
        return
    if row['status'] == config.TASK_STATUS_REVOKED:
        logger.info(
            "Not finishing %s as %s: it was revoked while it ran.", task_id, status,
        )
        return
    if row['status'] == status:
        logger.info(
            "Not finishing %s again: its row is already %s, so the earlier write "
            "landed even though its acknowledgement did not.", task_id, status,
        )
        return
    if row['parent_task_id'] is not None and row['status'] == config.TASK_STATUS_FAIL:
        logger.info(
            "Not finishing %s as %s: its parent gave up on it and ended it as %s "
            "before it returned.", task_id, status, row['status'],
        )
        return
    logger.error(
        "Not finishing %s as %s: its row is already %s. A task wrote its own "
        "terminal row, so the queue could not record this attempt's verdict or "
        "retry it; the task must return or raise instead.",
        task_id, status, row['status'],
    )


def _error_record(job, exc):
    from . import TASK_FUNC_ERROR_CODES

    return failure_record(exc, TASK_FUNC_ERROR_CODES.get(job.get('func'), UNKNOWN_ERROR_CODE))


_LOST_CONNECTION_SUMMARY = (
    "The database connection was lost repeatedly while this task ran. "
    "Check the container logs for details."
)

_LOST_CONNECTION_ERROR_NAMES = (
    'ConnectionException',
    'ConnectionDoesNotExist',
    'ConnectionFailure',
    'SqlclientUnableToEstablishSqlconnection',
    'SqlserverRejectedEstablishmentOfSqlconnection',
    'TransactionResolutionUnknown',
    'ProtocolViolation',
    'AdminShutdown',
    'CrashShutdown',
    'CannotConnectNow',
    'DatabaseDropped',
    'IdleSessionTimeout',
    'TooManyConnections',
)

_LOST_CONNECTION_SQLSTATE_CLASS = '08'

_LOST_CONNECTION_SQLSTATES = frozenset({
    '53300', '57P01', '57P02', '57P03', '57P04', '57P05',
})


def _lost_connection_types():
    from psycopg2 import InterfaceError, errors

    found = [InterfaceError]
    for name in _LOST_CONNECTION_ERROR_NAMES:
        error_type = getattr(errors, name, None)
        if isinstance(error_type, type):
            found.append(error_type)
    try:
        from database import ConnectionLostError

        found.append(ConnectionLostError)
    except Exception:
        logger.debug("database.ConnectionLostError is unavailable", exc_info=True)
    return tuple(found)


def _is_connectivity_error(exc):
    try:
        from psycopg2 import OperationalError

        lost = _lost_connection_types()
    except Exception:
        return False
    if isinstance(exc, lost):
        return True
    if not isinstance(exc, OperationalError):
        return False
    sqlstate = getattr(exc, 'pgcode', None)
    if sqlstate is None:
        return type(exc) is OperationalError
    return (
        str(sqlstate).startswith(_LOST_CONNECTION_SQLSTATE_CLASS)
        or sqlstate in _LOST_CONNECTION_SQLSTATES
    )


def _close_inherited_sockets(worker):
    conns = [worker._conn]
    listener = getattr(worker, "_listener", None)
    if listener is not None:
        conns.append(getattr(listener, "_conn", None))
    for conn in conns:
        if conn is None:
            continue
        try:
            fd = conn.fileno()
        except Exception:
            continue
        if not isinstance(fd, int) or fd < 0:
            continue
        try:
            os.close(fd)
        except Exception:
            pass


def _encode_outcome(outcome):
    status, summary, result = outcome
    if not isinstance(result, dict):
        result = None
    try:
        return pickle.dumps((status, summary, result))
    except Exception:
        logger.exception(
            "The task result could not be pickled; reporting the outcome without it"
        )
        return pickle.dumps((status, summary, None))


_PR_SET_PDEATHSIG = 1
_PARENT_WATCHDOG_INTERVAL = 1.0


def _bind_to_parent_death(parent_pid):
    if sys.platform == 'linux':
        if not _bind_linux_pdeathsig(parent_pid):
            _watch_parent_death(parent_pid)
    else:
        _watch_parent_death(parent_pid)


def _bind_linux_pdeathsig(parent_pid):
    try:
        import ctypes
        import ctypes.util

        libc_name = ctypes.util.find_library('c')
        if not libc_name:
            return False
        ctypes.CDLL(libc_name, use_errno=True).prctl(
            _PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0
        )
    except Exception:
        logger.debug("Could not bind the job process to the worker's death", exc_info=True)
        return False
    if os.getppid() != parent_pid:
        os._exit(1)
    return True


def _watch_parent_death(parent_pid):
    def _watch():
        while True:
            try:
                if os.getppid() != parent_pid:
                    os._exit(1)
            except Exception:
                pass
            time.sleep(_PARENT_WATCHDOG_INTERVAL)

    threading.Thread(
        target=_watch, name='worker-parent-watchdog', daemon=True
    ).start()


_OOM_KILL_COUNTER_FILES = (
    '/sys/fs/cgroup/memory.events',
    '/sys/fs/cgroup/memory/memory.oom_control',
)

_NATIVE_CRASH_SIGNALS = ('SIGSEGV', 'SIGBUS', 'SIGABRT', 'SIGFPE')

_INITIAL_PID_NAMESPACE_INODE = 0xEFFFFFFC

_KERNEL_OOM_VICTIM = re.compile(r'(?:Killed process |oom-kill:.*\bpid=)(\d+)')


def _own_cgroup_counter_files():
    files = []
    try:
        with open('/proc/self/cgroup', encoding='ascii') as membership:
            for line in membership:
                hierarchy, controllers, path = line.rstrip('\n').split(':', 2)
                path = path.rstrip('/')
                if hierarchy == '0' and not controllers:
                    files.append(f'/sys/fs/cgroup{path}/memory.events')
                elif 'memory' in controllers.split(','):
                    files.append(f'/sys/fs/cgroup/memory{path}/memory.oom_control')
    except (OSError, ValueError):
        return _OOM_KILL_COUNTER_FILES
    return tuple(dict.fromkeys(files + list(_OOM_KILL_COUNTER_FILES)))


def _oom_kill_count():
    for path in _own_cgroup_counter_files():
        try:
            with open(path, encoding='ascii') as counters:
                for line in counters:
                    name, _, value = line.partition(' ')
                    if name == 'oom_kill':
                        return int(value)
        except (OSError, ValueError):
            continue
    return None


def _monotonic_usec():
    clock = getattr(time, 'CLOCK_MONOTONIC', None)
    if clock is None:
        return None
    return time.clock_gettime_ns(clock) // 1000


def _oom_baseline():
    return {'kills': _oom_kill_count(), 'since_usec': _monotonic_usec(), 'pid': None}


def _kernel_oom_victims(since_usec):
    if since_usec is None:
        return None
    try:
        if os.stat('/proc/self/ns/pid').st_ino != _INITIAL_PID_NAMESPACE_INODE:
            return None
        kmsg = os.open('/dev/kmsg', os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None
    victims = set()
    try:
        while True:
            try:
                record = os.read(kmsg, 8192)
            except BlockingIOError:
                break
            except OSError as exc:
                if exc.errno == errno.EPIPE:
                    continue
                break
            if not record:
                break
            header, _, message = record.decode('utf-8', 'replace').partition(';')
            fields = header.split(',')
            try:
                logged_usec = int(fields[2])
            except (IndexError, ValueError):
                continue
            match = _KERNEL_OOM_VICTIM.search(message)
            if match and logged_usec >= since_usec:
                victims.add(int(match.group(1)))
    finally:
        os.close(kmsg)
    return victims


def _signal_name(signum):
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


_SIGKILL_VERDICTS = {
    'victim': (
        ERR_OUT_OF_MEMORY,
        "by the kernel out-of-memory killer: the system ran out of memory while the job ran.",
    ),
    'bystander': (
        ERR_JOB_PROCESS_DIED,
        "but the kernel out-of-memory killer ended a different process (pid {victims}) "
        "while it ran, so something else stopped this job.",
    ),
    'counted': (
        ERR_OUT_OF_MEMORY,
        "while the kernel recorded an out-of-memory kill in this container: the container "
        "ran out of memory while the job ran. The counter covers the whole container, so "
        "the kill is attributed to this job without naming the victim process.",
    ),
    'not_counted': (
        ERR_JOB_PROCESS_DIED,
        "and the kernel recorded no out-of-memory kill while it ran. A "
        "userspace memory killer (systemd-oomd, earlyoom, a Kubernetes eviction) or a "
        "manual kill stopped it.",
    ),
    'unconfirmed': (
        ERR_JOB_PROCESS_DIED,
        "before it could report back. This is most often an out-of-memory kill, but the "
        "kernel's out-of-memory records are not readable here to confirm it.",
    ),
}


def _sigkill_verdict(baseline):
    pid = baseline.get('pid')
    victims = _kernel_oom_victims(baseline.get('since_usec')) if pid is not None else None
    if victims and pid in victims:
        return 'victim', victims
    if victims:
        return 'bystander', victims
    kills_before = baseline.get('kills')
    kills_after = _oom_kill_count()
    if kills_before is not None and kills_after is not None:
        return ('counted' if kills_after > kills_before else 'not_counted'), victims
    return ('unconfirmed' if victims is None else 'not_counted'), victims


def _killed_child_death(signum, baseline):
    baseline = baseline or {}
    verdict, victims = _sigkill_verdict(baseline)
    code, clause = _SIGKILL_VERDICTS[verdict]
    process = f"The job process (pid {baseline['pid']})" if verdict == 'victim' else "The job process"
    summary = (
        f"{process} was killed on signal {signum} (SIGKILL) "
        f"{clause.format(victims=', '.join(str(victim) for victim in sorted(victims or ())))} "
        "Check the container logs for details."
    )
    return summary, error_manager.build(code, summary)


def _child_death(status, oom_baseline=None):
    code = os.waitstatus_to_exitcode(status)
    if code >= 0:
        summary = (
            f"The job process exited with code {code} without reporting back. "
            "Check the container logs for details."
        )
        return summary, error_manager.build(ERR_JOB_PROCESS_DIED, summary)
    signum = -code
    name = _signal_name(signum)
    if name == 'SIGKILL':
        return _killed_child_death(signum, oom_baseline)
    if name == 'SIGILL':
        summary = (
            f"The job process crashed on signal {signum} (SIGILL), an illegal CPU "
            "instruction: a native library such as the model runtime uses an instruction "
            "set this CPU does not have. This is not an out-of-memory condition."
        )
        return summary, error_manager.build(ERR_PROCESS_CRASHED, summary)
    if name in _NATIVE_CRASH_SIGNALS:
        summary = (
            f"The job process crashed on signal {signum} ({name}) inside native code, "
            "most often the model runtime during inference. This is a crash, not an "
            "out-of-memory condition. Check the container logs for details."
        )
        return summary, error_manager.build(ERR_PROCESS_CRASHED, summary)
    summary = (
        f"The job process died on signal {signum} ({name}) before it could report back. "
        "Check the container logs for details."
    )
    return summary, error_manager.build(ERR_JOB_PROCESS_DIED, summary)


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
    if worker._fork_jobs:
        logger.info(
            "Jobs run in a forked child process; job memory returns to the OS at job end."
        )
    else:
        logger.info(
            "Jobs run in the worker process; analysis models are unloaded after each job."
        )
    logger.info("Worker %s ready; recycling after %s jobs.", worker.identity, worker.max_jobs)
    worker.run_forever()


if __name__ == '__main__':
    main()
