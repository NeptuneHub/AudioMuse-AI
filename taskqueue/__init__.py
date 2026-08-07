# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The public face of the Postgres task queue: enqueue, cancel, and who am I.

Enqueueing is an INSERT that runs in the CALLER's transaction, and the NOTIFY
that wakes a worker only fires when Postgres commits it. A start path therefore
has exactly one step that can fail, and it fails atomically: either the row and
its wake-up both commit or neither does. There is no window in which a task row
exists without a queued job, or a queued job without a row, so no code is needed
to detect or repair that half-state.

Admission is likewise enforced by the database rather than by a lock: a partial
unique index allows one live main task, so a concurrent second start raises
``TaskAlreadyRunning`` rather than racing a check-then-act sequence.

``ALLOWED_FUNCS`` is a security boundary, not bookkeeping. ``func`` is read back
out of a database row and then called, so it is matched against this frozen set
before ``importlib`` ever sees it; a dotted path that is not listed is refused
rather than imported.

Every import below the constants is deliberately deferred into a function body.
``taskqueue`` sits one level above ``config`` in test_import_architecture.py's
eager graph and several long chains run through it, so importing its own
submodules at module level would push the deepest chain past MAX_CHAIN.

Main Features:
* ``enqueue`` writes the job row and the wake-up notification in one transaction
* ``TaskAlreadyRunning`` carries the 409 message the start endpoints return
* ``request_cancel`` / ``request_cancel_all`` publish the real-time stop signal
* ``current_task_id`` tells a running task which row is its own
* ``reap_finished_children`` deletes a parent's finished children and returns their outcomes

A root enqueue clears the FINISHED rows before inserting itself - that is the
whole retention policy, no cap and no age. It deliberately never touches NEW or
RUNNING rows: an unconditional wipe also deleted work that was genuinely still
running (a radio, a sweep, a plugin task), and a missing row IS the cancellation
signal, so starting one task silently killed another. The wipe sits INSIDE the
enqueue savepoint so a refused start undoes it.
"""

import importlib
import logging

import queue_names

logger = logging.getLogger(__name__)

QUEUE_HIGH = queue_names.QUEUE_HIGH
QUEUE_DEFAULT = queue_names.QUEUE_DEFAULT
PRIORITY_FRONT = queue_names.PRIORITY_FRONT
CANCEL_ALL = queue_names.CANCEL_ALL


ALLOWED_FUNCS = frozenset((
    'tasks.analysis.run_analysis_task',
    'tasks.analysis.analyze_album_task',
    'tasks.analysis.rebuild_all_indexes_task',
    'tasks.cleaning.identify_and_clean_orphaned_albums_task',
    'tasks.clustering.run_clustering_task',
    'tasks.clustering.run_clustering_batch_task',
    'tasks.multiserver_sync.sweep_server',
    'tasks.multiserver_sync.sweep_all_secondary_servers',
    'tasks.sonic_fingerprint_manager.run_sonic_fingerprint_task',
    'tasks.provider_migration_tasks.execute_provider_migration',
    'tasks.provider_migration_tasks.dry_run_provider_migration',
    'tasks.provider_migration_tasks.source_refresh_provider_migration',
    'tasks.provider_migration_tasks.resume_provider_migration_restart',
    'plugin.manager.run_plugin_task',
))

_current_task_id = None


class TaskAlreadyRunning(RuntimeError):
    def __init__(self, message=None):
        message = message or (
            "Another task is already running. Wait for it to finish, or cancel it first."
        )
        super().__init__(message)
        self.user_message = message
        self.status_code = 409


class UnknownTaskFunction(RuntimeError):
    pass


class TaskNotQueued(RuntimeError):
    pass


def current_task_id():
    return _current_task_id


def set_current_task_id(task_id):
    global _current_task_id
    _current_task_id = task_id


def resolve_func(dotted):
    if dotted not in ALLOWED_FUNCS:
        raise UnknownTaskFunction(f"{dotted} is not an allowed task function")
    module_name, _, attribute = dotted.rpartition('.')
    return getattr(importlib.import_module(module_name), attribute)


def _connection(conn):
    if conn is not None:
        return conn, False
    from database import get_db

    return get_db(), True


def _with_cursor(action, conn):
    from . import sql

    db, owns_transaction = _connection(conn)
    cur = db.cursor()
    try:
        result = action(sql, cur)
    finally:
        cur.close()
    if owns_transaction:
        db.commit()
    return result


def take_start_lock(conn=None):
    _with_cursor(lambda sql, cur: sql.take_start_lock(cur), conn)


SHARED_KWARG_REF = '__audiomuse_shared__'


def put_shared_payload(owner_task_id, body, conn=None, token=None):
    return _with_cursor(
        lambda sql, cur: sql.put_shared(cur, owner_task_id, body, token=token), conn
    )


def clear_shared_payload(owner_task_id, token, conn=None):
    return _with_cursor(
        lambda sql, cur: sql.clear_shared(cur, owner_task_id, token), conn
    )


def _check_shared(shared, kwargs, parent_task_id):
    if parent_task_id is None:
        raise ValueError('shared needs a parent_task_id to hang the payload on')
    for name, token in shared.items():
        if name not in kwargs and token is None:
            raise ValueError(
                f"shared kwarg {name!r} has neither a body in kwargs nor a token"
            )


def _publish_shared(sql, cur, parent_task_id, shared, kwargs):
    refs = {}
    for name, token in shared.items():
        if name in kwargs:
            refs[name] = sql.put_shared(cur, parent_task_id, kwargs.pop(name), token=token)
        elif token is not None:
            refs[name] = token
    if refs:
        kwargs[SHARED_KWARG_REF] = {'owner': parent_task_id, 'tokens': refs}


def enqueue(func, args=(), kwargs=None, *, task_id, task_type, queue=QUEUE_DEFAULT,
            priority=0, parent_task_id=None, sub_type_identifier=None,
            max_attempts=None, details=None, conn=None, shared=None):
    import psycopg2

    if func not in ALLOWED_FUNCS:
        raise UnknownTaskFunction(f"{func} is not an allowed task function")

    kwargs = dict(kwargs or {})
    if shared:
        _check_shared(shared, kwargs, parent_task_id)

    def _write(sql, cur):
        if parent_task_id is None:
            sql.take_start_lock(cur)
        cur.execute("SAVEPOINT audiomuse_enqueue")
        try:
            if parent_task_id is None:
                sql.clear_task_status(cur)
            if shared:
                _publish_shared(sql, cur, parent_task_id, shared, kwargs)
            inserted = sql.insert_job(
                cur,
                task_id=task_id,
                task_type=task_type,
                func=func,
                args=args,
                kwargs=kwargs,
                queue=queue,
                priority=priority,
                parent_task_id=parent_task_id,
                sub_type_identifier=sub_type_identifier,
                max_attempts=max_attempts,
                details=details,
            )
        except psycopg2.errors.UniqueViolation as exc:
            cur.execute("ROLLBACK TO SAVEPOINT audiomuse_enqueue")
            raise TaskAlreadyRunning() from exc
        if not inserted:
            cur.execute("ROLLBACK TO SAVEPOINT audiomuse_enqueue")
            raise TaskNotQueued(
                f"task {task_id} already exists and cannot be re-queued"
            )
        cur.execute("RELEASE SAVEPOINT audiomuse_enqueue")
        sql.notify_job(cur, queue)

    _with_cursor(_write, conn)
    logger.info("Queued %s task %s on the %s queue.", task_type, task_id, queue)
    return task_id


def reap_finished_children(parent_task_id, conn=None):
    return _with_cursor(lambda sql, cur: sql.reap_children(cur, parent_task_id), conn)


def live_children(parent_task_id, conn=None):
    return _with_cursor(lambda sql, cur: sql.live_children(cur, parent_task_id), conn)


def worker_snapshot(conn=None):
    return _with_cursor(lambda sql, cur: sql.worker_snapshot(cur), conn)


def queue_backlog(conn=None):
    return _with_cursor(lambda sql, cur: sql.queue_backlog(cur), conn)


def request_cancel(task_id, conn=None):
    _with_cursor(lambda sql, cur: sql.notify_cancel(cur, str(task_id)), conn)


def request_cancel_all(conn=None):
    _with_cursor(lambda sql, cur: sql.notify_cancel(cur, CANCEL_ALL), conn)


def publish_event(event, conn=None):
    _with_cursor(lambda sql, cur: sql.notify_event(cur, event), conn)
