# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Every statement the Postgres task queue runs, and nothing else.

The queue IS the task_status row: the same row carries the payload a worker
must run and the status the UI reads. Six columns and one partial index turn
the existing table into a queue.

Two design points are load-bearing. First, the claim is ONE statement - an
UPDATE whose subquery takes FOR UPDATE SKIP LOCKED - so N workers race with no
coordination and exactly one wins. Second, liveness is a session advisory lock
rather than a heartbeat: a dead process releases the lock when its connection
drops. Every function takes a cursor the caller owns and commits, keeping this
a near-leaf module (imports only config) under the MAX_CHAIN cap.

Main Features:
* ensure_schema adds the queue columns, indexes and the one-time migration
* insert_job / claim / finish_child / requeue_or_fail move a row through its life
* hold / try_hold / release are the advisory-lock liveness primitives
* reap_children deletes finished children; notify_* publish to workers/Flask
* blob_tables_autovacuum_cannot_reach / vacuum_table sweep the tables whose dead
  rows autovacuum will not collect, because its threshold counts ROWS and these
  hold a few enormous TOASTed blobs; begin_reclaim_session caps that sweep with
  a lock_timeout so it can never queue behind a reader, writer or restore
"""

import hashlib
import json
import logging
import socket
import zlib

import config
import queue_names

logger = logging.getLogger(__name__)

LOCK_CLASS = 0x41554449
MAINTENANCE_LOCK_CLASS = 0x4155444A
MAINTENANCE_LOCK_KEY = 1

CHANNEL_JOB = 'audiomuse_job'
CHANNEL_CANCEL = 'audiomuse_cancel'
CHANNEL_EVENT = 'audiomuse_event'
CHANNEL_CONTROL = 'audiomuse_control'
CHANNEL_RECLAIM = 'audiomuse_reclaim'

_NOTIFY = "SELECT pg_notify(%s, %s)"

QUEUE_HIGH = queue_names.QUEUE_HIGH
QUEUE_DEFAULT = queue_names.QUEUE_DEFAULT
CANCEL_ALL = queue_names.CANCEL_ALL

CONTROL_TASK_TYPE = 'worker_control'

LIVE_STATUSES = tuple(config.TASK_STATUS_LIVE)
TERMINAL_STATUSES = tuple(config.TASK_STATUS_TERMINAL)

_NEW = config.TASK_STATUS_NEW
_RUNNING = config.TASK_STATUS_RUNNING
_FAIL = config.TASK_STATUS_FAIL
_REVOKED = config.TASK_STATUS_REVOKED


def _status_list(statuses, separator=','):
    return separator.join(f"'{status}'" for status in statuses)


_LIVE_STATUS_SQL = _status_list(LIVE_STATUSES, ', ')
_LIVE_IN_LIST = _status_list(LIVE_STATUSES)
_TERMINAL_IN_LIST = _status_list(TERMINAL_STATUSES)

_ADD_COLUMNS = """
    ALTER TABLE task_status
      ADD COLUMN IF NOT EXISTS func         TEXT,
      ADD COLUMN IF NOT EXISTS payload      TEXT,
      ADD COLUMN IF NOT EXISTS queue_name   TEXT,
      ADD COLUMN IF NOT EXISTS priority     INTEGER DEFAULT 0,
      ADD COLUMN IF NOT EXISTS attempts     INTEGER DEFAULT 0,
      ADD COLUMN IF NOT EXISTS max_attempts INTEGER DEFAULT 3,
      ADD COLUMN IF NOT EXISTS worker_id    TEXT,
      ADD COLUMN IF NOT EXISTS shared_token   TEXT,
      ADD COLUMN IF NOT EXISTS shared_payload TEXT
"""

_SET_FILLFACTOR = "ALTER TABLE task_status SET (fillfactor = 70)"

CLAIM_INDEX_NAME = 'idx_task_status_claim'

_CLAIM_INDEX = """
    CREATE INDEX IF NOT EXISTS {name}
      ON task_status (queue_name, priority DESC, id) WHERE status = '{new}'
""".format(name=CLAIM_INDEX_NAME, new=_NEW)

PARENT_INDEX_NAME = 'idx_task_status_parent'

PARENT_INDEX_SQL = """
    CREATE INDEX IF NOT EXISTS {}
      ON task_status (parent_task_id) WHERE parent_task_id IS NOT NULL
""".format(PARENT_INDEX_NAME)

def _index_name(prefix, seed):
    return "{}_{:x}".format(prefix, zlib.crc32(seed.encode()))


LIVE_INDEX_PREFIX = 'idx_task_status_live'

LIVE_INDEX_NAME = _index_name(LIVE_INDEX_PREFIX, ','.join(LIVE_STATUSES))

_LIVE_INDEX = """
    CREATE INDEX IF NOT EXISTS {name}
      ON task_status (status) WHERE status IN ({live})
""".format(name=LIVE_INDEX_NAME, live=_LIVE_STATUS_SQL)

MAIN_TASK_TYPES = (
    'main_analysis', 'main_clustering', 'cleaning', 'provider_migration',
    'sonic_fingerprint',
)

MAIN_INDEX_PREFIX = 'idx_task_status_one_live_main'

MAIN_INDEX_NAME = _index_name(
    MAIN_INDEX_PREFIX, '|'.join((','.join(MAIN_TASK_TYPES), ','.join(LIVE_STATUSES)))
)

_DROP_STALE_INDEXES = """
    DO $do$
    DECLARE stale text;
    BEGIN
        FOR stale IN
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'task_status'
              AND indexname LIKE %s
              AND indexname <> %s
        LOOP
            EXECUTE format('DROP INDEX IF EXISTS %%I', stale);
        END LOOP;
    END
    $do$
"""

_ONE_LIVE_MAIN_INDEX = """
    CREATE UNIQUE INDEX IF NOT EXISTS {name}
      ON task_status ((parent_task_id IS NULL))
      WHERE parent_task_id IS NULL
        AND status IN ({statuses})
        AND task_type IN ({types})
""".format(
    name=MAIN_INDEX_NAME,
    statuses=_LIVE_STATUS_SQL,
    types=', '.join(f"'{name}'" for name in MAIN_TASK_TYPES),
)


def _retire_surplus_main_live_roots(cur):
    cur.execute(
        "SELECT task_id FROM task_status "
        "WHERE parent_task_id IS NULL AND status IN ({live}) "
        "AND task_type IN ({types}) ORDER BY id DESC".format(
            live=_LIVE_IN_LIST,
            types=', '.join(f"'{name}'" for name in MAIN_TASK_TYPES),
        )
    )
    rows = [row[0] for row in cur.fetchall()]
    if len(rows) <= 1:
        return True
    running = []
    for task_id in rows:
        if try_hold(cur, task_id):
            release(cur, task_id)
        else:
            running.append(task_id)
    if len(running) > 1:
        logger.warning(
            "Queue schema: %d main tasks are still executing; deferring the "
            "one-live-main index build to a later startup.", len(running),
        )
        return False
    keep = running[0] if running else rows[0]
    revoke = [task_id for task_id in rows if task_id != keep]
    cur.execute(
        "UPDATE task_status SET status=%s, progress=100 WHERE task_id = ANY(%s)",
        (_REVOKED, revoke),
    )
    return True

SWEEP_TASK_TYPE = 'server_sweep'

SWEEP_INDEX_PREFIX = 'idx_task_status_one_live_sweep'

SWEEP_INDEX_NAME = _index_name(
    SWEEP_INDEX_PREFIX, '|'.join((SWEEP_TASK_TYPE, ','.join(LIVE_STATUSES)))
)

_ONE_LIVE_SWEEP_INDEX = """
    CREATE UNIQUE INDEX IF NOT EXISTS {name}
      ON task_status ((parent_task_id IS NULL))
      WHERE parent_task_id IS NULL
        AND status IN ({statuses})
        AND task_type = '{sweep}'
""".format(name=SWEEP_INDEX_NAME, statuses=_LIVE_STATUS_SQL, sweep=SWEEP_TASK_TYPE)

# PENDING, STARTED, PROGRESS and FAILURE are the spellings this table held before
# the queue existed. They are history, not vocabulary, so they stay literal here:
# config's aliases of those names point at the CURRENT spellings and would make
# every one of these statements a no-op.
_MIGRATE_STATUSES = (
    f"UPDATE task_status SET status='{_NEW}' WHERE status='PENDING'",
    f"UPDATE task_status SET status='{_RUNNING}' WHERE status IN ('STARTED','PROGRESS')",
    f"UPDATE task_status SET status='{_FAIL}' WHERE status='FAILURE'",
    f"""
    DO $do$
    BEGIN
        IF to_regclass('task_history') IS NOT NULL THEN
            EXECUTE 'UPDATE task_history SET status=''{_FAIL}'' WHERE status=''FAILURE''';
        END IF;
    END
    $do$
    """,
)

_DROP_LEGACY_CHILDREN = "DELETE FROM task_status WHERE parent_task_id IS NOT NULL"

_RETIRE_SURPLUS_LIVE_ROOTS = """
    UPDATE task_status SET status='{revoked}', progress=100
    WHERE task_id IN (
        SELECT task_id FROM (
            SELECT task_id, row_number() OVER (
                       PARTITION BY (task_type = '{sweep}') ORDER BY id DESC) AS rank
            FROM task_status
            WHERE parent_task_id IS NULL
              AND status IN ({live})
              AND task_type NOT IN ('alchemy_radio','{control}')
        ) ranked WHERE ranked.rank > 1
    )
""".format(
    revoked=_REVOKED, live=_LIVE_IN_LIST, sweep=SWEEP_TASK_TYPE, control=CONTROL_TASK_TYPE,
)

_CREATE_BASE_TABLE = """
    CREATE TABLE IF NOT EXISTS task_status (
        id SERIAL PRIMARY KEY,
        task_id TEXT UNIQUE NOT NULL,
        parent_task_id TEXT,
        task_type TEXT NOT NULL,
        sub_type_identifier TEXT,
        status TEXT,
        progress INTEGER DEFAULT 0,
        details TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        start_time DOUBLE PRECISION,
        end_time DOUBLE PRECISION
    )
"""

_PROBE_NEWEST_COLUMN = (
    "SELECT 1 FROM information_schema.columns "
    "WHERE table_name = 'task_status' AND column_name = 'shared_payload'"
)

_PROBE_FILLFACTOR = (
    "SELECT 1 FROM pg_class WHERE relname = 'task_status' "
    "AND reloptions @> ARRAY['fillfactor=70']"
)


def ensure_schema(cur):
    cur.execute(_CREATE_BASE_TABLE)

    cur.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'task_status' AND column_name = 'func'"
    )
    first_time = cur.fetchone() is None

    cur.execute(_PROBE_NEWEST_COLUMN)
    if cur.fetchone() is None:
        cur.execute(_ADD_COLUMNS)

    cur.execute(_PROBE_FILLFACTOR)
    if cur.fetchone() is None:
        cur.execute(_SET_FILLFACTOR)

    if first_time:
        for statement in _MIGRATE_STATUSES:
            cur.execute(statement)
        cur.execute(_DROP_LEGACY_CHILDREN)
        cur.execute(_RETIRE_SURPLUS_LIVE_ROOTS)

    if _index_missing(cur, CLAIM_INDEX_NAME):
        cur.execute(_CLAIM_INDEX)
    if _index_missing(cur, PARENT_INDEX_NAME):
        cur.execute(PARENT_INDEX_SQL)
    if _index_missing(cur, MAIN_INDEX_NAME):
        # The one-live-main index now also covers sonic_fingerprint. An upgrade
        # can leave a live batch root and a live fingerprint side by side (the
        # old index never admitted the fingerprint), so retire every live main
        # root but the newest before the unique index is (re)created. A row a
        # live worker is still executing (its per-task advisory lock is held) is
        # never revoked; if two are executing the build is deferred to a later
        # boot rather than force-revoking a running task.
        if _retire_surplus_main_live_roots(cur):
            cur.execute(_ONE_LIVE_MAIN_INDEX)
            cur.execute(_DROP_STALE_INDEXES, (MAIN_INDEX_PREFIX + '%', MAIN_INDEX_NAME))
    if _index_missing(cur, SWEEP_INDEX_NAME):
        cur.execute(_ONE_LIVE_SWEEP_INDEX)
        cur.execute(_DROP_STALE_INDEXES, (SWEEP_INDEX_PREFIX + '%', SWEEP_INDEX_NAME))
    if _index_missing(cur, LIVE_INDEX_NAME):
        cur.execute(_LIVE_INDEX)
        cur.execute(_DROP_STALE_INDEXES, (LIVE_INDEX_PREFIX + '%', LIVE_INDEX_NAME))
    return first_time


def _index_missing(cur, index_name):
    cur.execute(
        "SELECT 1 FROM pg_indexes WHERE tablename = 'task_status' AND indexname = %s",
        (index_name,),
    )
    return cur.fetchone() is None


_INSERT_JOB = f"""
    INSERT INTO task_status (task_id, parent_task_id, task_type, sub_type_identifier,
                             status, func, payload, queue_name, priority,
                             attempts, max_attempts, progress, details, timestamp, start_time)
    VALUES (%s, %s, %s, %s, '{_NEW}', %s, %s, %s, %s, 0, %s, 0, %s, NOW(), NULL)
    ON CONFLICT (task_id) DO UPDATE SET
        func = EXCLUDED.func,
        payload = EXCLUDED.payload,
        queue_name = EXCLUDED.queue_name,
        priority = EXCLUDED.priority,
        max_attempts = EXCLUDED.max_attempts,
        parent_task_id = COALESCE(EXCLUDED.parent_task_id, task_status.parent_task_id),
        sub_type_identifier = COALESCE(EXCLUDED.sub_type_identifier,
                                       task_status.sub_type_identifier),
        details = COALESCE(EXCLUDED.details, task_status.details),
        status = '{_NEW}',
        timestamp = NOW()
    WHERE task_status.func IS NULL
      AND task_status.status IN ({_LIVE_IN_LIST})
    RETURNING task_id
"""


def strip_secret_kwargs(kwargs):
    kept = {}
    stripped = []
    for key, value in dict(kwargs or {}).items():
        if key in config.QUEUE_SECRET_KWARGS:
            stripped.append(key)
        else:
            kept[key] = value
    return kept, stripped


def restore_secret_kwargs(kwargs, stripped):
    restored = dict(kwargs or {})
    for key in stripped or ():
        if key in restored:
            continue
        config_name = key[:-len('_param')].upper() if key.endswith('_param') else key.upper()
        restored[key] = getattr(config, config_name, None)
    return restored


def insert_job(cur, task_id, task_type, func, args=None, kwargs=None, queue=QUEUE_DEFAULT,
               priority=0, parent_task_id=None, sub_type_identifier=None,
               max_attempts=None, details=None):
    clean_kwargs, stripped = strip_secret_kwargs(kwargs)
    payload = json.dumps({
        'args': list(args or ()),
        'kwargs': clean_kwargs,
        'stripped': stripped,
    })
    cur.execute(
        _INSERT_JOB,
        (
            task_id,
            parent_task_id,
            task_type,
            sub_type_identifier,
            func,
            payload,
            queue,
            priority,
            config.QUEUE_MAX_ATTEMPTS if max_attempts is None else int(max_attempts),
            json.dumps(details) if details is not None else None,
        ),
    )
    return cur.fetchone() is not None


def notify_job(cur, queue):
    cur.execute(_NOTIFY, (CHANNEL_JOB, queue))


def notify_cancel(cur, task_id):
    cur.execute(_NOTIFY, (CHANNEL_CANCEL, str(task_id)))


def notify_event(cur, event):
    cur.execute(_NOTIFY, (CHANNEL_EVENT, str(event)))


def notify_control(cur, payload):
    cur.execute(_NOTIFY, (CHANNEL_CONTROL, json.dumps(payload)))


_CLAIM = f"""
    UPDATE task_status SET status='{_RUNNING}',
                           worker_id = %s,
                           start_time = COALESCE(start_time, %s),
                           timestamp = NOW()
    WHERE task_id = (
        SELECT task_id FROM task_status
        WHERE status='{_NEW}' AND queue_name = %s
        ORDER BY priority DESC, id
        FOR UPDATE SKIP LOCKED
        LIMIT 1)
    RETURNING task_id, task_type, parent_task_id, func, payload, attempts, max_attempts
"""


def claim(cur, queue, now, worker_id=None):
    cur.execute(_CLAIM, (worker_id, now, queue))
    row = cur.fetchone()
    if row is None:
        return None
    payload = row[4]
    try:
        decoded = json.loads(payload) if isinstance(payload, str) else (payload or {})
    except (TypeError, ValueError):
        decoded = {}
    return {
        'task_id': row[0],
        'task_type': row[1],
        'parent_task_id': row[2],
        'func': row[3],
        'args': list(decoded.get('args') or ()),
        'kwargs': restore_secret_kwargs(decoded.get('kwargs'), decoded.get('stripped')),
        'attempts': row[5],
        'max_attempts': row[6],
    }


def hold(cur, task_id):
    cur.execute("SELECT pg_advisory_lock(%s, hashtext(%s))", (LOCK_CLASS, str(task_id)))


def try_hold(cur, task_id):
    cur.execute("SELECT pg_try_advisory_lock(%s, hashtext(%s))", (LOCK_CLASS, str(task_id)))
    return bool(cur.fetchone()[0])


def release(cur, task_id):
    cur.execute("SELECT pg_advisory_unlock(%s, hashtext(%s))", (LOCK_CLASS, str(task_id)))


START_LOCK_KEY = 5512740318664902


def take_start_lock(cur):
    cur.execute("SELECT pg_advisory_xact_lock(%s)", (START_LOCK_KEY,))


def try_maintenance_lock(cur):
    cur.execute(
        "SELECT pg_try_advisory_lock(%s, %s)", (MAINTENANCE_LOCK_CLASS, MAINTENANCE_LOCK_KEY)
    )
    return bool(cur.fetchone()[0])


def release_maintenance_lock(cur):
    cur.execute(
        "SELECT pg_advisory_unlock(%s, %s)", (MAINTENANCE_LOCK_CLASS, MAINTENANCE_LOCK_KEY)
    )


def begin_reclaim_session(cur):
    cur.execute(
        "SELECT set_config('lock_timeout', %s, false)", (config.BLOB_RECLAIM_LOCK_TIMEOUT,)
    )
    cur.execute(
        "SELECT set_config('statement_timeout', %s, false)",
        (config.BLOB_RECLAIM_STATEMENT_TIMEOUT,),
    )


_ANY_LIVE_TASK = f"SELECT 1 FROM task_status WHERE status IN ({_LIVE_IN_LIST}) LIMIT 1"


def any_live_task(cur):
    cur.execute(_ANY_LIVE_TASK)
    return cur.fetchone() is not None


_OLD_SNAPSHOT_HOLDER = """
    SELECT 1 FROM pg_stat_activity
    WHERE datname = current_database()
      AND pid <> pg_backend_pid()
      AND backend_xmin IS NOT NULL
      AND xact_start < now() - make_interval(secs => %s)
    LIMIT 1
"""


def snapshot_holder_blocking_reclaim(cur, grace_seconds=None):
    grace = (
        config.BLOB_RECLAIM_SNAPSHOT_GRACE_SECONDS
        if grace_seconds is None
        else float(grace_seconds)
    )
    cur.execute(_OLD_SNAPSHOT_HOLDER, (grace,))
    return cur.fetchone() is not None


_BLOB_TABLES_AUTOVACUUM_CANNOT_REACH = """
    SELECT quote_ident(s.schemaname) || '.' || quote_ident(s.relname), s.n_dead_tup,
           pg_size_pretty(pg_total_relation_size(s.relid))
    FROM pg_stat_user_tables AS s
    WHERE s.n_dead_tup > 0
      AND s.n_dead_tup < current_setting('autovacuum_vacuum_threshold')::numeric
                         + current_setting('autovacuum_vacuum_scale_factor')::numeric
                           * greatest(s.n_live_tup, 0)
      AND (
          EXISTS (
              SELECT 1 FROM pg_attribute AS a
              WHERE a.attrelid = s.relid
                AND a.attnum > 0 AND NOT a.attisdropped
                AND a.atttypid = 'bytea'::regtype
          )
          OR pg_total_relation_size(s.relid) >= %s
      )
    ORDER BY pg_total_relation_size(s.relid) ASC
"""


def blob_tables_autovacuum_cannot_reach(cur, min_bytes=None):
    floor_bytes = config.BLOB_RECLAIM_MIN_BYTES if min_bytes is None else int(min_bytes)
    cur.execute(_BLOB_TABLES_AUTOVACUUM_CANNOT_REACH, (floor_bytes,))
    return [
        (quoted_relname, int(dead), str(total))
        for quoted_relname, dead, total in (cur.fetchall() or ())
    ]


def vacuum_table(cur, quoted_relname):
    cur.execute('VACUUM ' + quoted_relname)


_RUNNING_TASKS = f"""
    SELECT task_id, attempts, max_attempts, task_type
    FROM task_status AS t
    WHERE t.status='{_RUNNING}' AND t.func IS NOT NULL
      AND t.timestamp < NOW() - make_interval(secs => %s)
      AND NOT EXISTS (SELECT 1 FROM pg_stat_activity AS a
                      WHERE a.datname = current_database()
                        AND a.application_name IN (t.worker_id, t.worker_id || %s))
"""


def running_tasks(cur, grace_seconds=None):
    grace = (
        config.QUEUE_ORPHAN_GRACE_SECONDS if grace_seconds is None else float(grace_seconds)
    )
    cur.execute(_RUNNING_TASKS, (grace, WORKER_LISTEN_SUFFIX))
    return [
        {'task_id': row[0], 'attempts': row[1], 'max_attempts': row[2], 'task_type': row[3]}
        for row in (cur.fetchall() or ())
    ]


_REQUEUE_OR_FAIL = f"""
    WITH prev AS (
        SELECT task_id, attempts, worker_id,
               (parent_task_id IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM task_status AS p
                    WHERE p.task_id = task_status.parent_task_id
                      AND p.status IN ({_LIVE_IN_LIST}))) AS parent_gone
        FROM task_status
        WHERE task_id = %s AND status='{_RUNNING}'
    ), reclaimed AS (
        UPDATE task_status AS t
        SET attempts = t.attempts + 1,
            status = CASE WHEN NOT prev.parent_gone AND t.attempts + 1 <= t.max_attempts
                          THEN '{_NEW}' ELSE '{_FAIL}' END,
            details = CASE WHEN NOT prev.parent_gone AND t.attempts + 1 <= t.max_attempts
                           THEN t.details ELSE %s END,
            progress = CASE WHEN NOT prev.parent_gone AND t.attempts + 1 <= t.max_attempts
                            THEN t.progress ELSE 100 END,
            end_time = CASE WHEN NOT prev.parent_gone AND t.attempts + 1 <= t.max_attempts
                            THEN NULL ELSE %s END,
            timestamp = NOW()
        FROM prev
        WHERE t.task_id = prev.task_id AND t.status='{_RUNNING}'
        RETURNING t.status
    )
    SELECT reclaimed.status,
           pg_notify(%s, prev.task_id || %s || COALESCE(prev.worker_id, '') || %s
                     || prev.attempts::text)
    FROM reclaimed, prev
"""

# The ASCII unit separator, written as an escape so it stays visible: a raw
# U+001F byte in the source is invisible in every editor and diff, and an
# editor that normalises it silently breaks decode_reclaim's three-way split.
RECLAIM_SEPARATOR = '\x1f'


_REQUEUE_UNCHARGED = f"""
    UPDATE task_status SET status='{_NEW}', worker_id=NULL, timestamp=NOW()
    WHERE task_id = %s AND status='{_RUNNING}'
      AND (%s IS NULL OR worker_id IS NULL OR worker_id = %s)
    RETURNING task_id
"""


def requeue_uncharged(cur, task_id, worker_id=None):
    cur.execute(_REQUEUE_UNCHARGED, (task_id, worker_id, worker_id))
    return cur.fetchone() is not None


def requeue_or_fail(cur, task_id, now, failure_details):
    cur.execute(
        _REQUEUE_OR_FAIL,
        (
            task_id, json.dumps(failure_details), now,
            CHANNEL_RECLAIM, RECLAIM_SEPARATOR, RECLAIM_SEPARATOR,
        ),
    )
    row = cur.fetchone()
    return row[0] if row else None


def decode_reclaim(payload):
    parts = str(payload or '').split(RECLAIM_SEPARATOR)
    if len(parts) != 3:
        return None
    try:
        attempts = int(parts[2])
    except (TypeError, ValueError):
        return None
    return {'task_id': parts[0], 'worker_id': parts[1], 'attempts': attempts}


_PUT_SHARED = f"""
    UPDATE task_status SET shared_token = %s, shared_payload = %s
    WHERE task_id = %s AND status IN ({_LIVE_IN_LIST})
    RETURNING task_id
"""

_GET_SHARED = """
    SELECT shared_payload FROM task_status
    WHERE task_id = %s AND shared_token = %s
"""

_CLEAR_SHARED = """
    UPDATE task_status SET shared_token = NULL, shared_payload = NULL
    WHERE task_id = %s AND shared_token = %s
"""


def shared_token_for(body):
    return hashlib.sha256(body.encode('utf-8')).hexdigest()[:32]


_SHARED_TOKEN = "SELECT shared_token FROM task_status WHERE task_id = %s"


def put_shared(cur, task_id, body, token=None):
    token = token or shared_token_for(body)
    cur.execute(_SHARED_TOKEN, (task_id,))
    existing = cur.fetchone()
    if existing is not None and existing[0] == token:
        return token
    cur.execute(_PUT_SHARED, (token, body, task_id))
    if cur.fetchone() is None:
        raise SharedPayloadUnavailable(
            f"task {task_id} is not live, so a shared payload cannot be attached"
        )
    return token


def get_shared(cur, task_id, token):
    cur.execute(_GET_SHARED, (task_id, token))
    row = cur.fetchone()
    if row is None or row[0] is None:
        raise SharedPayloadUnavailable(
            f"the shared payload for {task_id} is gone or no longer matches its token"
        )
    return row[0]


def clear_shared(cur, task_id, token):
    cur.execute(_CLEAR_SHARED, (task_id, token))
    return cur.rowcount


class SharedPayloadUnavailable(RuntimeError):
    pass


_CURRENT_STATUS = (
    "SELECT status, task_type, parent_task_id, worker_id FROM task_status WHERE task_id = %s"
)


def current_row(cur, task_id):
    cur.execute(_CURRENT_STATUS, (task_id,))
    row = cur.fetchone()
    if row is None:
        return None
    return {
        'status': row[0],
        'task_type': row[1],
        'parent_task_id': row[2],
        'worker_id': row[3],
    }


_FINISH_TASK = f"""
    UPDATE task_status
    SET status = %s, progress = 100, details = %s,
        end_time = COALESCE(end_time, %s), timestamp = NOW(),
        func = NULL, payload = NULL,
        shared_token = NULL, shared_payload = NULL
    WHERE task_id = %s AND status = '{_RUNNING}'
      AND (%s IS NULL OR worker_id IS NULL OR worker_id = %s)
    RETURNING status
"""


def finish_task(cur, task_id, status, details, now, worker_id=None):
    cur.execute(
        _FINISH_TASK, (status, json.dumps(details), now, task_id, worker_id, worker_id)
    )
    row = cur.fetchone()
    return row[0] if row else None


_REAP_CHILDREN = f"""
    DELETE FROM task_status
    WHERE parent_task_id = %s AND status IN ({_TERMINAL_IN_LIST})
    RETURNING task_id, status, sub_type_identifier, details
"""


def reap_children(cur, parent_task_id):
    cur.execute(_REAP_CHILDREN, (parent_task_id,))
    reaped = []
    for task_id, status, sub_type_identifier, details in (cur.fetchall() or ()):
        try:
            decoded = json.loads(details) if isinstance(details, str) else (details or {})
        except (TypeError, ValueError):
            decoded = {}
        reaped.append({
            'task_id': task_id,
            'status': status,
            'sub_type_identifier': sub_type_identifier,
            'details': decoded,
        })
    return reaped


_LIVE_CHILDREN = f"""
    SELECT task_id, sub_type_identifier, progress FROM task_status
    WHERE parent_task_id = %s AND status IN ({_LIVE_IN_LIST})
"""


def live_children(cur, parent_task_id):
    cur.execute(_LIVE_CHILDREN, (parent_task_id,))
    return [
        {'task_id': row[0], 'sub_type_identifier': row[1], 'progress': row[2]}
        for row in (cur.fetchall() or ())
    ]


# Finished, and NOT a child some parent is still draining. A fan-out leaves its
# album or batch children terminal until the parent's next reap, so a wipe that
# only looked at the status deleted them out from under it - and the parent then
# waited for children that no longer existed. Both wipes share this one predicate
# because getting it right in one of them and not the other hangs a run just the
# same.
TERMINAL_AND_NOT_A_LIVE_PARENTS_CHILD = (
    f"status IN ({_TERMINAL_IN_LIST}) "
    "AND (parent_task_id IS NULL "
    "     OR NOT EXISTS (SELECT 1 FROM task_status AS live "
    "                    WHERE live.task_id = task_status.parent_task_id "
    f"                      AND live.status IN ({_LIVE_IN_LIST})))"
)

_CLEAR_TASK_STATUS = (
    "DELETE FROM task_status WHERE " + TERMINAL_AND_NOT_A_LIVE_PARENTS_CHILD
)


def clear_task_status(cur):
    cur.execute(_CLEAR_TASK_STATUS)
    return cur.rowcount


_WORKER_SNAPSHOT = f"""
    SELECT a.application_name, a.backend_start, t.task_id, t.task_type
    FROM pg_stat_activity AS a
    LEFT JOIN task_status AS t
      ON t.status = '{_RUNNING}' AND t.worker_id = a.application_name
    WHERE a.application_name LIKE %s AND a.application_name NOT LIKE %s
      AND a.datname = current_database()
    ORDER BY a.application_name
"""

WORKER_IDENTITY_PREFIX = 'audiomuse-worker-'
WORKER_LISTEN_SUFFIX = '-listen'


def parse_worker_identity(application_name):
    if not application_name or not application_name.startswith(WORKER_IDENTITY_PREFIX):
        return None, None
    body = application_name[len(WORKER_IDENTITY_PREFIX):]
    queue, _, rest = body.partition('-')
    hostname = rest.rsplit('-', 2)[0] if rest.count('-') >= 2 else ''
    return queue or None, hostname or None


def worker_snapshot(cur):
    from tz_helper import to_local_str

    cur.execute(
        _WORKER_SNAPSHOT,
        (WORKER_IDENTITY_PREFIX + '%', '%' + WORKER_LISTEN_SUFFIX),
    )
    seen = set()
    workers = []
    for application_name, backend_start, task_id, task_type in (cur.fetchall() or ()):
        if application_name in seen:
            continue
        seen.add(application_name)
        queue, hostname = parse_worker_identity(application_name)
        workers.append({
            'hostname': hostname or application_name,
            'queues': [queue] if queue else [],
            'state': 'busy' if task_id else 'idle',
            'current_job_id': task_id,
            'current_task_type': task_type,
            'started_at': to_local_str(backend_start) if backend_start else None,
        })
    return workers


_QUEUE_BACKLOG = f"""
    SELECT queue_name, COUNT(*)
    FROM task_status
    WHERE status = '{_NEW}' AND queue_name = ANY(%s)
    GROUP BY queue_name
"""


def queue_backlog(cur):
    cur.execute(_QUEUE_BACKLOG, (list(queue_names.QUEUE_NAMES),))
    found = dict(cur.fetchall())
    return [
        {
            'queue_name': name,
            'pending_count': found.get(name, 0),
        }
        for name in queue_names.QUEUE_NAMES
    ]


def hostname():
    try:
        return socket.gethostname()
    except Exception:
        return 'unknown'
