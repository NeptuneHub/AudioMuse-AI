# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""One worker process: claim a task, run it, finish it, repeat.

Run as python -m taskqueue.worker --queue high or --queue default. The
claim is a single UPDATE whose subquery takes FOR UPDATE SKIP LOCKED, so N
workers racing need no coordination.

Where the OS can fork (Linux, macOS, every container - detected by os.fork
existing) the job runs in a FORKED CHILD that reports its outcome over a pipe
and exits,
so all the memory the job allocated - ONNX models, the transformers stack,
numpy buffers - is returned to the OS the moment the job ends, exactly like
the old RQ fork-per-job worker. The child inherits the claim by copy but only
the parent ever touches the claim connection and the advisory lock; the child
opens its own database connections through the task's app context and leaves
through os._exit, so the parent's sockets are never written or closed by the
child. A child that dies without reporting (OOM kill, segfault) fails the job
with its signal or exit code instead of taking the whole worker down. The
child binds itself to the parent's death (PR_SET_PDEATHSIG on Linux; a
getppid() watchdog thread on other POSIX platforms), so a worker killed
uncleanly cannot leave an orphaned job still writing after the task's
advisory lock died with the parent and reclaim handed the row to another
worker. Config is hydrated in the parent before forking so a first-success
refresh latches in the worker instead of being thrown away with the child.

Windows cannot fork, so there the job runs in the worker process itself (the
shape the old SimpleWorker had) and any heavy analysis models the job loaded
are unloaded again when it finishes (_unload_job_models): the sessions are
dropped and the heap trimmed, keeping an idle worker at the library floor
instead of the loaded-model footprint. Either way cancelling means ending the
worker's process tree, which includes the job child, and the worker recycles
after QUEUE_MAX_JOBS.

Liveness is the advisory lock held on the task's own connection; if the process
dies the lock dies with it, so no heartbeat is needed. ensure_hold retakes
the lock the moment the listener reconnects after a Postgres restart or failover,
before reclaim can hand the still-running task elsewhere.

Thread caps at the bottom of this module must be applied BEFORE numpy/ONNX/BLAS
import, so heavy imports are deferred into main, and
service_roles.declare_worker_role runs before import config for the same
ordering reason.

Main Features:
* Claim/drain loop that blocks on LISTEN when no work exists
* Fork-per-job on POSIX gives every byte of job memory back to the OS;
  Windows runs the job in-process and unloads the models after each job
* A cancel notification ends the process tree in about 50ms
* Boot reclaims orphaned tasks bounded by QUEUE_MAX_ATTEMPTS
* A lost connection (SQLSTATE class 08, 57Pxx, InterfaceError) requeues the row
  without charging an attempt, up to UNCHARGED_REQUEUE_LIMIT free passes,
  then charges and fails as usual
"""

import logging
import os
import pickle
import signal
import sys
import threading
import time

import queue_names
import service_roles

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
service_roles.declare_worker_role(force=True)
THREAD_CAP = _apply_thread_caps(QUEUE)
print(f"{QUEUE} worker CPU thread cap = {THREAD_CAP}")

import config  # noqa: E402
from . import sql  # noqa: E402
from .listen import Listener  # noqa: E402
from .process import stop_hard, sweep_stale_temp_dirs  # noqa: E402

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
        self._fork_jobs = hasattr(os, 'fork')

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

    def _forget_abandoned(self, task_id):
        self._uncharged.pop(task_id, None)
        logger.info(
            "Abandoned task %s is no longer this worker's RUNNING row; "
            "leaving it exactly as it is.", task_id,
        )
        return False

    def _requeue_charging_an_attempt(self, cur, task_id):
        row = sql.current_row(cur, task_id)
        if (
            row is None
            or row['status'] != config.TASK_STATUS_RUNNING
            or row['worker_id'] not in (None, self.identity)
        ):
            return self._forget_abandoned(task_id)
        status = sql.requeue_or_fail(
            cur, task_id, time.time(),
            _terminal_details(config.TASK_STATUS_FAIL, _LOST_CONNECTION_SUMMARY, None),
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
            "Running %s (%s) after %d worker loss(es) of an allowed %d",
            task_id, job['func'], job['attempts'], job['max_attempts'],
        )
        started = time.time()
        outcome, summary, result = self._execute(job)
        if outcome is None:
            if task_id not in self._abandoned:
                self._abandoned.append(task_id)
        else:
            self._uncharged.pop(task_id, None)
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

    def _execute(self, job):
        task_id = job['task_id']
        try:
            kwargs = self.hydrate_shared(job['kwargs'])
        except Exception as exc:
            logger.exception("Task %s raised", task_id)
            return self._failure(task_id, exc)
        if self._fork_jobs:
            return self._run_in_child(job, kwargs)
        try:
            return self._attempt(job, kwargs)
        finally:
            self._unload_job_models()

    def _attempt(self, job, kwargs):
        from . import resolve_func

        task_id = job['task_id']
        try:
            self.hydrate_config()
            func = resolve_func(job['func'])
            result = func(*job['args'], **kwargs)
        except Exception as exc:
            logger.exception("Task %s raised", task_id)
            return self._failure(task_id, exc)
        return config.TASK_STATUS_SUCCESS, None, result

    def _failure(self, task_id, exc):
        if _is_connectivity_error(exc):
            logger.warning(
                "Task %s lost its database connection; putting its row back "
                "on the queue instead of failing it.", task_id,
            )
            return None, _error_summary(exc), None
        return config.TASK_STATUS_FAIL, _error_summary(exc), None

    def _run_in_child(self, job, kwargs):
        task_id = job['task_id']
        self.hydrate_config()
        try:
            read_fd, write_fd = os.pipe()
        except OSError as exc:
            logger.exception("Could not open the report pipe for %s", task_id)
            return config.TASK_STATUS_FAIL, _error_summary(exc), None
        parent_pid = os.getpid()
        try:
            pid = os.fork()
        except OSError as exc:
            os.close(read_fd)
            os.close(write_fd)
            logger.exception("Could not fork the job process for %s", task_id)
            return config.TASK_STATUS_FAIL, _error_summary(exc), None
        if pid == 0:
            self._child_main(job, kwargs, read_fd, write_fd, parent_pid)
        os.close(write_fd)
        payload = b''
        try:
            with os.fdopen(read_fd, 'rb') as pipe:
                payload = pipe.read()
        except Exception:
            logger.exception("Reading the job process report for %s failed", task_id)
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                logger.debug("SIGKILL to the job process failed", exc_info=True)
        try:
            _, status = os.waitpid(pid, 0)
        except OSError:
            logger.exception("Could not reap the job process for %s", task_id)
            status = 0
        return self._child_outcome(task_id, status, payload)

    def _child_main(self, job, kwargs, read_fd, write_fd, parent_pid):
        exit_code = 1
        try:
            _bind_to_parent_death(parent_pid)
            os.close(read_fd)
            payload = _encode_outcome(self._attempt(job, kwargs))
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

    def _child_outcome(self, task_id, status, payload):
        if payload:
            try:
                outcome = pickle.loads(payload)
            except Exception:
                logger.exception("Could not decode the job process report for %s", task_id)
            else:
                if isinstance(outcome, tuple) and len(outcome) == 3:
                    return outcome
                logger.error("The job process report for %s is malformed", task_id)
        summary = _child_death_summary(status)
        logger.error("Task %s: %s", task_id, summary)
        return config.TASK_STATUS_FAIL, summary, None

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

    def _write_terminal_row(self, task_id, status, error, result):
        with self._conn.cursor() as cur:
            row = sql.current_row(cur, task_id)
            if row is None or row['status'] != config.TASK_STATUS_RUNNING:
                return False
            written = sql.finish_task(
                cur, task_id, status, _terminal_details(status, error, result),
                time.time(), worker_id=self.identity,
            )
        if written is None:
            logger.error(
                "Refusing to finish %s: the row is no longer this worker's. It was "
                "reclaimed and restarted elsewhere while this process was still on it.",
                task_id,
            )
        return True

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
                if not self._write_terminal_row(task_id, status, error, result):
                    return
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


def _final_message(status, error):
    if status == config.TASK_STATUS_SUCCESS:
        return "Task completed successfully."
    return error or "Task failed. Check the container logs for details."


def _terminal_details(status, error, result):
    details = {'message': _final_message(status, error)}
    if error:
        details['error'] = error
    if isinstance(result, dict):
        details['final_summary_details'] = result
    return details


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


def _error_summary(exc):
    text = str(exc).strip() or exc.__class__.__name__
    return text[:500]


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
        _bind_linux_pdeathsig(parent_pid)
    else:
        _watch_parent_death(parent_pid)


def _bind_linux_pdeathsig(parent_pid):
    try:
        import ctypes
        import ctypes.util

        libc_name = ctypes.util.find_library('c')
        if not libc_name:
            return
        ctypes.CDLL(libc_name, use_errno=True).prctl(
            _PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0
        )
    except Exception:
        logger.debug("Could not bind the job process to the worker's death", exc_info=True)
        return
    if os.getppid() != parent_pid:
        os._exit(1)


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


def _child_death_summary(status):
    code = os.waitstatus_to_exitcode(status)
    if code < 0:
        return (
            f"The job process died on signal {-code} before it could report back. "
            "This usually means the system ran out of memory. "
            "Check the container logs for details."
        )
    return (
        f"The job process exited with code {code} without reporting back. "
        "Check the container logs for details."
    )


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
