# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Everything a worker task shares: the prologue, the cancel check, the reporter.

The contract with the queue is one sentence: the queue writes the terminal row
and decides every retry; the task returns a summary or raises. A task never
writes SUCCESS, FAIL or REVOKED on its own row. It raises TaskFailed for an
error no retry can fix, TaskCancelled from its cancel check, and anything else
for a failure the queue should try again. The message it wants on the dashboard
recap goes in the dict it returns.

Before this module held them, every task carried its own copy of the three
things below and they drifted: seven progress reporters, four cancellation
mechanisms of which four tasks had none past their first line, and a sweep that
caught every exception, wrote FAILURE itself and returned normally, so the
queue recorded SUCCESS and never retried it.

Main Features:
* task_run_prologue / terminal_skip: resolve the claimed id and refuse to rerun
  a row that is already terminal, before any work
* make_cancel_check / cancel_guard: the ONE cooperative cancellation. It reads
  the task's own row and its parent's on a dedicated autocommit connection,
  throttled to QUEUE_CANCEL_CHECK_SECONDS, and raises TaskCancelled. A read
  that fails never cancels: a database blip is not a cancel. cancel_guard is
  the form to reach for; make_cancel_check is the same check for a body that
  already owns a finally block for other cleanup (the album task, the analysis
  phase), where a second with-block would only re-indent hundreds of lines.
  A parent is passed only by a supervised child (an album, a batch) that has
  nothing to report to once its parent is over. A task that merely carries
  lineage on its row, like the alignment a migration queues, watches its own
  row alone: its parent finishes first by design
* make_task_reporter: the ONE progress reporter. It writes RUNNING and only
  RUNNING; a terminal state handed to it is logged as an error and downgraded,
  because that row belongs to the queue. It keeps the capped log, the
  progress window, the write throttle, and can merge a live details dict on
  every write for a task that persists its resume state on its own row
* for_each_server_in_scope: the shared per-server loop for a task that runs
  the same step against every server and reports which ones failed. A
  TaskFailed raised by the step is the task's own verdict and passes through
"""

import logging
import time
import uuid
from contextlib import contextmanager

import taskqueue
from taskqueue import TaskCancelled, TaskFailed
from config import (
    QUEUE_CANCEL_CHECK_SECONDS,
    TASK_STATUS_RUNNING,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
    TASK_STATUS_REVOKED,
    TASK_STATUS_TERMINAL,
)
from database import (
    MAX_LOG_ENTRIES_STORED,
    connect_raw,
    get_task_info_from_db,
    save_task_status,
)
from psycopg2 import OperationalError

from error import error_manager
from error.error_dictionary import ERR_DB_CONNECTION

logger = logging.getLogger(__name__)

__all__ = (
    'TaskCancelled', 'TaskFailed',
    'task_run_prologue', 'terminal_skip',
    'make_cancel_check', 'cancel_guard',
    'make_task_reporter', 'for_each_server_in_scope',
)


def task_run_prologue(current_task_id=None):
    claimed_task_id = taskqueue.current_task_id()
    task_id = current_task_id or claimed_task_id or str(uuid.uuid4())
    return claimed_task_id, task_id, get_task_info_from_db(task_id)


def terminal_skip(
    task_id,
    claimed_task_id,
    task_info,
    *,
    revoked_message,
    terminal_message,
    terminal_details=None,
):
    if claimed_task_id and task_info is None:
        logger.info(
            "Task %s has no live DB claim; treating it as revoked.", task_id
        )
        return {"status": TASK_STATUS_REVOKED, "message": revoked_message}
    if task_info and task_info.get('status') in (
        TASK_STATUS_SUCCESS,
        TASK_STATUS_FAILURE,
        TASK_STATUS_REVOKED,
    ):
        logger.info(
            "Task %s is already terminal (%s); skipping.",
            task_id, task_info.get('status'),
        )
        result = {"status": task_info.get('status'), "message": terminal_message}
        if terminal_details is not None:
            result["details"] = terminal_details(task_info)
        return result
    return None


def _open_check_connection():
    try:
        conn = connect_raw(application_name='audiomuse-cancel-check')
        conn.autocommit = True
        return conn
    except Exception:
        logger.exception(
            "Could not open the cancel-check connection; this run will not notice "
            "a cancel until the next check that can"
        )
        return None


def _read_task_statuses(conn, task_ids):
    return taskqueue.task_statuses(task_ids, conn=conn)


def _close_quietly(conn):
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        logger.debug("Closing the cancel-check connection failed", exc_info=True)


def _statuses_or_none(state, task_id, watched):
    if not state['opened']:
        state['opened'] = True
        state['conn'] = _open_check_connection()
    conn = state['conn']
    if conn is None:
        state['opened'] = False
        return None
    try:
        statuses = _read_task_statuses(conn, watched)
    except Exception:
        logger.exception(
            "Cancel check for %s could not read task_status; assuming the run "
            "is live and reopening the connection next time", task_id,
        )
        _close_quietly(conn)
        state['conn'] = None
        state['opened'] = False
        return None
    return statuses if isinstance(statuses, dict) else None


def _raise_if_cancelled(task_id, parent_task_id, statuses):
    if task_id and task_id not in statuses:
        raise TaskCancelled(
            f"task {task_id} has no task_status row any more; it was cancelled"
        )
    if task_id and statuses.get(task_id) == TASK_STATUS_REVOKED:
        raise TaskCancelled(f"task {task_id} was revoked")
    if not parent_task_id:
        return
    parent = statuses.get(parent_task_id)
    if parent is None or parent in TASK_STATUS_TERMINAL:
        raise TaskCancelled(
            f"parent {parent_task_id} of {task_id} is "
            f"{parent or 'gone'}, so this child has nothing to report to"
        )


def make_cancel_check(task_id, parent_task_id=None, every_seconds=None,
                      clock=time.monotonic):
    interval = QUEUE_CANCEL_CHECK_SECONDS if every_seconds is None else float(every_seconds)
    watched = [task for task in (task_id, parent_task_id) if task]
    state = {'last': float('-inf'), 'conn': None, 'opened': False}

    def check(force=False):
        if not watched:
            return
        now = clock()
        if not force and now - state['last'] < interval:
            return
        state['last'] = now
        statuses = _statuses_or_none(state, task_id, watched)
        if statuses is not None:
            _raise_if_cancelled(task_id, parent_task_id, statuses)

    def close():
        _close_quietly(state['conn'])
        state['conn'] = None

    return check, close


@contextmanager
def cancel_guard(task_id, parent_task_id=None, every_seconds=None):
    check, close = make_cancel_check(task_id, parent_task_id, every_seconds)
    try:
        yield check
    finally:
        close()


def make_task_reporter(task_id, task_type, initial_message,
                       parent_task_id=None, sub_type_identifier=None,
                       base_details=None, prefix=None,
                       progress_base=0.0, progress_span=100.0,
                       min_db_interval=0.0, details_source=None,
                       clock=time.monotonic):
    base = dict(base_details or {})
    state = {'progress': 0, 'last_db': float('-inf')}
    label = prefix or f"{task_type}-{task_id}"
    logs = [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {initial_message}"]

    def _live():
        if details_source is None:
            return {}
        live = details_source()
        return dict(live) if isinstance(live, dict) else {}

    try:
        save_task_status(
            task_id, task_type, TASK_STATUS_RUNNING,
            parent_task_id=parent_task_id, sub_type_identifier=sub_type_identifier,
            progress=int(progress_base),
            details={
                **base, **_live(),
                "message": initial_message, "status_message": initial_message,
                "log": list(logs),
            },
        )
    except OperationalError as e:
        error_manager.from_exception(e, code=ERR_DB_CONNECTION, logger=logger)
        raise

    def report(message, progress, **kwargs):
        state['progress'] = progress
        logger.info(f"[{label}] {message}")
        task_state = kwargs.pop('task_state', TASK_STATUS_RUNNING)
        if task_state in TASK_STATUS_TERMINAL:
            logger.error(
                "[%s] a task asked its reporter to write %s; that row belongs to the "
                "queue, so this write is downgraded to RUNNING. Return or raise instead.",
                label, task_state,
            )
            task_state = TASK_STATUS_RUNNING
        details = {**base, **_live(), **kwargs, "message": message, "status_message": message}
        logs.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")
        if len(logs) > MAX_LOG_ENTRIES_STORED:
            del logs[:-MAX_LOG_ENTRIES_STORED]
        details["log"] = logs
        scaled = int(progress_base + (progress or 0) * progress_span / 100.0)
        now = clock()
        if min_db_interval and now - state['last_db'] < min_db_interval:
            return True
        state['last_db'] = now
        return save_task_status(
            task_id, task_type, task_state,
            parent_task_id=parent_task_id, sub_type_identifier=sub_type_identifier,
            progress=scaled, details=details,
        )

    report.state = state
    return report


def for_each_server_in_scope(scope, step, *, on_server=None, cancel=None):
    from .mediaserver import registry

    servers = registry.servers_for_scope(scope)
    results = []
    failed = []
    for index, server in enumerate(servers):
        if cancel is not None:
            cancel()
        name = server['name'] if server else 'default server'
        if on_server is not None:
            on_server(index, len(servers), server, name)
        try:
            with registry.bind(server):
                results.append(step(server, name))
        except (TaskCancelled, TaskFailed, OperationalError):
            raise
        except Exception:
            logger.exception(
                "%s failed on %s; continuing with the remaining servers", scope, name
            )
            failed.append(name)
    return servers, results, failed
