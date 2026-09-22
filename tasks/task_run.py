# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Everything a worker task shares: the prologue, the cancel check, the reporter.

The contract with the queue is one sentence: the queue writes the terminal row
and decides every retry, the task returns a summary or raises. A task never
writes SUCCESS, FAIL or REVOKED on its own row. It raises TaskFailed for an
error no retry can fix, TaskCancelled from its cancel check, and anything else
for a failure worth retrying; the recap line goes in the dict it returns.
The one terminal row a task writes is its own child's, through
taskqueue.end_child. Before this module every task carried its own copy of the
three things below, and they drifted.

Main Features:
* task_run_prologue resolves the claimed id and the id to report under; it
  reads no row
* make_cancel_check / cancel_guard is the ONE cooperative cancellation: it reads
  the task's own row and its parent's on a dedicated autocommit connection,
  throttled to QUEUE_CANCEL_CHECK_SECONDS, and raises TaskCancelled. A failed
  read never cancels. Every task calls
  it once with force=True BEFORE its first report, so a row a cancel wiped is
  never written to again. A parent is passed only by a supervised child that has
  nothing to report to once its parent is over; a task that merely carries
  lineage watches its own row alone
* make_task_reporter is the ONE progress reporter. It writes RUNNING and only
  RUNNING - a terminal state handed to it is logged as an error and downgraded,
  because that row belongs to the queue - and keeps the capped log, the progress
  window and the throttle. force=True skips it for the one line
  that must land: the failure a child records before it raises
* for_each_server_in_scope is the shared per-server loop: a TaskFailed from the
  step is the task's verdict and passes through, and so does a lost database
  connection (OperationalError or InterfaceError): swallowing that would march
  the loop over every remaining server on a dead connection
* run_playlist_task_per_server is the WHOLE body of a scheduled one-playlist-
  per-server task. The caller passes its label, playlist names and a callable
  returning the ids; the prologue, forced cancel check, reporter, heartbeat, the empty result that
  PRESERVES the previous playlist, the dated-playlist fallback and the rule that
  only a failure on EVERY server fails the task live here once. Inline in Flask
  (inline_task_id) app_cron writes the terminal row and the heartbeat beats
  inside QUEUE_INLINE_STALE_SECONDS
"""

import logging
import time
import uuid
from contextlib import contextmanager

import config
import taskqueue
from taskqueue import TaskCancelled, TaskFailed
from config import (
    QUEUE_CANCEL_CHECK_SECONDS,
    TASK_STATUS_RUNNING,
    TASK_STATUS_TERMINAL,
)
from database import (
    MAX_LOG_ENTRIES_STORED,
    connect_raw,
    save_task_status,
)
from psycopg2 import InterfaceError, OperationalError

from error import error_manager
from error.error_dictionary import ERR_DB_CONNECTION

logger = logging.getLogger(__name__)

__all__ = (
    'TaskCancelled', 'TaskFailed',
    'task_run_prologue',
    'make_cancel_check', 'cancel_guard',
    'make_task_reporter', 'for_each_server_in_scope',
    'run_playlist_task_per_server',
)


def task_run_prologue(current_task_id=None):
    claimed_task_id = taskqueue.current_task_id()
    return claimed_task_id, current_task_id or claimed_task_id or str(uuid.uuid4())


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
    own = statuses.get(task_id) if task_id else None
    if own in TASK_STATUS_TERMINAL:
        raise TaskCancelled(
            f"task {task_id} is already {own}: it was revoked, or its parent gave "
            "up on it, and it has nothing left to report"
        )
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
        force = kwargs.pop('force', False)
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
        if min_db_interval and not force and now - state['last_db'] < min_db_interval:
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
        except (TaskCancelled, TaskFailed, OperationalError, InterfaceError):
            raise
        except Exception:
            logger.exception(
                "The step failed on %s (scope %s); continuing with the remaining servers",
                name, scope,
            )
            failed.append(name)
    return servers, results, failed


_INLINE_BEAT_FLOOR_SECONDS = 120.0


def _playlist_heartbeat_cadence(inline):
    from .recovery import slow_step_budget_minutes

    wedged_minutes = config.QUEUE_WEDGED_MAIN_TASK_MINUTES
    if not inline:
        return None, slow_step_budget_minutes(wedged_minutes)
    stale_seconds = max(float(config.QUEUE_INLINE_STALE_SECONDS), _INLINE_BEAT_FLOOR_SECONDS)
    stale_minutes = stale_seconds / 60.0
    return stale_minutes, slow_step_budget_minutes(wedged_minutes or stale_minutes)


def run_playlist_task_per_server(task_type, label, playlist_name, fallback_name,
                                 build_ids, server_scope="all", inline_task_id=None):
    from flask_app import app

    from .mediaserver import create_or_replace_playlist
    from .ivf_manager import create_playlist_from_ids

    with app.app_context():
        from .recovery import row_heartbeat

        claimed_task_id, task_id = task_run_prologue(inline_task_id)
        watched_task_id = claimed_task_id or inline_task_id
        every_minutes, stop_after_minutes = _playlist_heartbeat_cadence(
            inline=claimed_task_id is None and bool(inline_task_id)
        )
        created = [0]
        current = ['resolving the server scope']

        def build(_server, server_name):
            current[0] = f"the {label} for {server_name}"
            with row_heartbeat(
                watched_task_id, lambda: current[0],
                every_minutes=every_minutes,
                stop_after_minutes=stop_after_minutes,
            ):
                track_ids = build_ids()
                if not track_ids:
                    logger.warning(
                        "The %s came out empty on %s; preserving the previous playlist.",
                        label, server_name,
                    )
                    return None
                try:
                    if create_or_replace_playlist(playlist_name, track_ids) is None:
                        raise RuntimeError(
                            f"Media server reported failure upserting the {label} playlist"
                        )
                    name = playlist_name
                except NotImplementedError:
                    name = f"{fallback_name} (Cron {time.strftime('%Y-%m-%d')})"
                    create_playlist_from_ids(name, track_ids)
                created[0] += 1
                logger.info(
                    "The %s playlist '%s' was upserted on %s with %d tracks.",
                    label, name, server_name, len(track_ids),
                )
                return name

        def on_server(index, total, _server, server_name):
            report(
                f"Building the {label} for {server_name} ({index + 1}/{total})...",
                int(100 * index / max(1, total)),
            )

        with cancel_guard(watched_task_id) as cancel:
            cancel(force=True)
            report = make_task_reporter(
                task_id, task_type, f"Building the {label} playlist...",
                prefix=f"{fallback_name.title().replace(' ', '')}-{task_id}",
            )
            servers, _results, failed = for_each_server_in_scope(
                server_scope, build, on_server=on_server, cancel=cancel,
            )

        if failed and len(failed) == len(servers):
            raise RuntimeError(
                f"The {label} failed on every server: " + ", ".join(failed)
            )
        message = f"Created {created[0]} {label} playlist(s)."
        if failed:
            message += f" Failed on: {', '.join(failed)}."
        summary = {
            "message": message,
            "servers_enabled": len(servers),
            "playlists_created": created[0],
            "failed": failed,
        }
        report(message, 100)
        logger.info("The %s run finished: %s", label, summary)
        return summary
