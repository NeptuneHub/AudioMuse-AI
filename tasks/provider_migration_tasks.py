# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Orchestrate the media-provider migration as queued tasks.

Drives the multi-step migration flow whose dry-run and source-refresh phases run
as queued tasks polled by the UI; delegates track matching to
provider_migration_matcher and reuses the app's core routines under an app
context.

Main Features:
* The centralized catalogue is NEVER touched: score rows, their canonical ids
  and embeddings survive. A provider swap only rewrites track_server_map for the
  default server, so no analysis is lost and the indexes stay valid (no rebuild).
* Under an advisory lock, repoints the default server's mappings, refreshes
  metadata, clears the old provider's artist ids, and points the default row at
  the target.
* Chromaprint fingerprints are carried onto the target's track ids in the same
  transaction rather than left keyed by dead ids.
* Queues a full-refresh alignment of the migrated server, its task row written
  INSIDE the migration transaction so the intent survives a crash; the root task
  stays non-terminal until every worker acknowledges the restart.
"""

import json
import logging
import time
import uuid

import taskqueue

from config import (
    TASK_STATUS_NEW,
    TASK_STATUS_RUNNING,
    TASK_STATUS_PENDING,
    TASK_STATUS_STARTED,
    TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
    TASK_STATUS_REVOKED,
    QUEUE_WEDGED_MAIN_TASK_MINUTES,
)
from sanitization import sanitize_string_for_db as _sanitize_text

from .recovery import row_heartbeat, slow_step_budget_minutes

logger = logging.getLogger(__name__)


_ADVISORY_LOCK_KEY = 7421536190082003

_MIG_TMP_PREFIX = '__audiomuse_mig_tmp__'

# The orphan snapshot lives in migration_session.state, which the wizard
# rewrites key by key precisely to avoid multi-MB blobs. Cap it and record the
# real total rather than persisting one row per unbound song forever.
_ORPHAN_REPORT_MAX_ROWS = 5000

# Upper bound on the post-commit restart handshake. The migration itself is
# already durable by then; this only bounds how long one worker is held.
_XACT_LOCK_SQL = "SELECT pg_advisory_xact_lock(%s)"

_RESTART_HANDSHAKE_MAX_SECONDS = 900

_RESTART_PUBLISH_ATTEMPTS = 3

_RESTART_PUBLISH_RETRY_SECONDS = 2

# Once the catalogue commit is durable the execute root is the admission barrier.
_RESTART_HANDSHAKE_RETRY_SECONDS = 5

_RESTART_RECOVERY_TASK_KEY = 'restart_recovery_task_id'


def find_fk(cur, table, column, ref_table='score', ref_column='item_id'):
    cur.execute(
        """
        SELECT tc.constraint_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_name = kcu.table_name
        JOIN information_schema.constraint_column_usage ccu
          ON tc.constraint_name = ccu.constraint_name
        WHERE tc.table_name = %s
          AND tc.constraint_type = 'FOREIGN KEY'
          AND kcu.column_name = %s
          AND ccu.table_name = %s
          AND ccu.column_name = %s
        LIMIT 1
        """,
        (table, column, ref_table, ref_column),
    )
    row = cur.fetchone()
    return row[0] if row else None


# The migration itself no longer rewrites score.item_id, so it has no need to drop
# these. The one-time fingerprint canonicalization DOES relabel item_ids (that is its
# whole purpose) and imports both from here.
def _drop_fk_constraints(cur, fk_embedding, fk_clap_embedding, lyrics_exists, fk_lyrics_embedding):
    if fk_embedding:
        cur.execute(f"ALTER TABLE embedding DROP CONSTRAINT IF EXISTS {fk_embedding}")
    if fk_clap_embedding:
        cur.execute(f"ALTER TABLE clap_embedding DROP CONSTRAINT IF EXISTS {fk_clap_embedding}")
    if lyrics_exists and fk_lyrics_embedding:
        cur.execute(
            f"ALTER TABLE lyrics_embedding DROP CONSTRAINT IF EXISTS {fk_lyrics_embedding}"
        )


def _readd_fk_constraints(cur, fk_embedding, fk_clap_embedding, lyrics_exists, fk_lyrics_embedding):
    for table, name in (
        ('embedding', fk_embedding),
        ('clap_embedding', fk_clap_embedding),
        ('lyrics_embedding', fk_lyrics_embedding if lyrics_exists else None),
    ):
        if not name:
            continue
        cur.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {name}")
        cur.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {name} "
            f"FOREIGN KEY (item_id) REFERENCES score(item_id) ON DELETE CASCADE"
        )


def _get_dedicated_conn():
    from database import connect_raw

    return connect_raw(application_name='audiomuse-provider-migration')


MIGRATION_TASK_TYPE = 'provider_migration'

MIGRATION_PLANNER_TASK_TYPE = 'provider_migration_planner'


def _migration_task_id():
    return taskqueue.current_task_id()


def _report_migration(task_id, status, progress, message, details=None):
    if not task_id:
        return True
    try:
        from flask_app import app
        from database import get_db, save_task_status

        payload = {'message': message, 'status_message': message}
        if details:
            payload.update(details)
        with app.app_context():
            db = get_db()
            with db.cursor() as cur:
                cur.execute(
                    "SELECT status FROM task_status WHERE task_id = %s FOR UPDATE",
                    (task_id,),
                )
                existing = cur.fetchone()
            if not existing:
                db.rollback()
                logger.error(
                    "provider migration: refusing to write %s for job %s; it has no "
                    "task_status claim, so the run was cancelled",
                    status,
                    task_id,
                )
                return 'missing'
            if existing and existing[0] == TASK_STATUS_REVOKED:
                db.commit()
                logger.warning(
                    "provider migration: task %s is REVOKED; refusing %s update",
                    task_id,
                    status,
                )
                return 'revoked'
            if (
                existing
                and existing[0] == TASK_STATUS_SUCCESS
                and status != TASK_STATUS_SUCCESS
            ):
                db.commit()
                logger.warning(
                    "provider migration: preserving SUCCESS for delayed retry %s "
                    "instead of writing %s",
                    task_id,
                    status,
                )
                return 'success'
            save_task_status(
                task_id, MIGRATION_TASK_TYPE, status, progress=progress, details=payload
            )
        return True
    except Exception:
        logger.exception("Could not record provider-migration task status")
        return False


def _restart_request_id(task_id, session_id):
    stable = task_id or f'session-{session_id}'
    return f'provider-migration-restart:{stable}'


def _completed_summary(state):
    post = (state or {}).get('post_migration') or {}
    return {
        'ok': True,
        'matched': int(post.get('matched') or 0),
        'index_rebuild_needed': False,
        'already_applied': True,
    }


def _persist_completed_restart_state(conn, session_id, request_id, *, acknowledged=None):
    cur = conn.cursor()
    if acknowledged is None:
        cur.execute(
            "UPDATE migration_session SET state = jsonb_set("
            "COALESCE(state, '{}'::jsonb), '{restart_request_id}', %s::jsonb, true) "
            "WHERE id = %s AND status = 'completed' RETURNING id",
            (json.dumps(request_id), session_id),
        )
    else:
        cur.execute(
            "UPDATE migration_session SET state = jsonb_set("
            "jsonb_set(COALESCE(state, '{}'::jsonb), '{restart_request_id}', "
            "%s::jsonb, true), '{restart_acknowledged}', %s::jsonb, true) "
            "WHERE id = %s AND status = 'completed' RETURNING id",
            (json.dumps(request_id), json.dumps(bool(acknowledged)), session_id),
        )
    if cur.fetchone() is None:
        conn.rollback()
        raise RuntimeError(
            f'completed migration session {session_id} disappeared before restart ACK'
        )
    conn.commit()


def _finalize_restart_handshake(
    conn,
    session_id,
    request_id,
    task_id,
    message,
    *,
    details=None,
):
    if not task_id:
        _persist_completed_restart_state(
            conn, session_id, request_id, acknowledged=True
        )
        return True

    taskqueue.take_start_lock(conn=conn)
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT status, state FROM migration_session WHERE id = %s FOR UPDATE",
            (session_id,),
        )
        row = cur.fetchone()
        state = _decoded_state(row[1]) if row else {}
        if (
            not row
            or row[0] != 'completed'
            or state.get('restart_request_id') != request_id
        ):
            raise RuntimeError(
                f'provider migration session {session_id} no longer owns restart '
                f'request {request_id}'
            )

        cur.execute(
            "UPDATE migration_session SET state = jsonb_set("
            "COALESCE(state, '{}'::jsonb), '{restart_acknowledged}', 'true'::jsonb, true) "
            "WHERE id = %s AND status = 'completed' "
            "AND state->>'restart_request_id' = %s RETURNING id",
            (session_id, request_id),
        )
        if cur.fetchone() is None:
            raise RuntimeError(
                f'provider migration restart request {request_id} was superseded '
                'before SUCCESS'
            )

        payload = {'message': message, 'status_message': message}
        if details:
            payload.update(details)
        now = time.time()
        cur.execute(
            "INSERT INTO task_status "
            "(task_id, task_type, status, progress, details, timestamp, start_time, "
            "end_time) VALUES (%s, %s, %s, 100, %s, NOW(), %s, %s) "
            "ON CONFLICT (task_id) DO UPDATE SET task_type = EXCLUDED.task_type, "
            "status = EXCLUDED.status, progress = 100, details = EXCLUDED.details, "
            "timestamp = NOW(), end_time = EXCLUDED.end_time",
            (
                task_id,
                MIGRATION_TASK_TYPE,
                TASK_STATUS_SUCCESS,
                json.dumps(payload),
                now,
                now,
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    try:
        from database import collapse_finished_task, record_task_history

        duration = None
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT start_time, end_time FROM task_status WHERE task_id = %s", (task_id,)
                )
                row = cur.fetchone()
            if row is not None:
                started, ended = (
                    (row['start_time'], row['end_time'])
                    if isinstance(row, dict) else (row[0], row[1])
                )
                if started is not None and ended is not None:
                    duration = max(0.0, float(ended) - float(started))
        except Exception:
            logger.debug("Could not compute the migration duration", exc_info=True)
        record_task_history(
            task_id,
            MIGRATION_TASK_TYPE,
            TASK_STATUS_SUCCESS,
            duration_seconds=duration,
            details=payload,
            conn=conn,
        )
        collapse_finished_task(conn, task_id, MIGRATION_TASK_TYPE, None, TASK_STATUS_SUCCESS)
    except Exception:
        logger.exception(
            "Provider migration %s committed SUCCESS but could not finalize its "
            "task row; history and cleanup will lag until the next run",
            session_id,
        )


def _restart_request_result(request_id):
    try:
        import restart_manager

        return restart_manager.get_control_request_result('restart', request_id)
    except Exception:
        logger.warning(
            "provider migration: could not read restart ACK for %s",
            request_id,
            exc_info=True,
        )
        return None


def _await_worker_restart(
    conn,
    session_id,
    request_id,
    *,
    alignment_task_id=None,
):
    current_request_id = request_id
    first_attempt = True
    deadline = time.monotonic() + _RESTART_HANDSHAKE_MAX_SECONDS
    while True:
        try:
            if first_attempt:
                acknowledged = _post_commit_reload(
                    alignment_task_id,
                    restart_request_id=current_request_id,
                )
                first_attempt = False
            else:
                acknowledged = _publish_restart_with_retries(current_request_id)
        except Exception:
            acknowledged = False
            logger.exception(
                "provider migration: restart handshake attempt %s failed",
                current_request_id,
            )

        if acknowledged is True:
            return current_request_id

        if _restart_request_result(current_request_id) is False:
            try:
                import restart_manager

                replacement = restart_manager.new_control_request_id()
                _persist_completed_restart_state(
                    conn,
                    session_id,
                    replacement,
                    acknowledged=False,
                )
                current_request_id = replacement
            except Exception:
                logger.exception(
                    "provider migration: could not rotate failed restart request %s; "
                    "will retry the durable request",
                    current_request_id,
                )

        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"provider migration session {session_id} was APPLIED and committed, "
                f"but no worker acknowledged the restart within "
                f"{_RESTART_HANDSHAKE_MAX_SECONDS}s (request {current_request_id}). "
                f"THE WORKERS ARE STILL ON THE OLD PROVIDER - restart them by hand. "
                f"Releasing this worker so other jobs can run."
            )

        logger.error(
            "provider migration session %s remains in restart_handshake; retrying "
            "request %s in %ss",
            session_id,
            current_request_id,
            _RESTART_HANDSHAKE_RETRY_SECONDS,
        )
        time.sleep(_RESTART_HANDSHAKE_RETRY_SECONDS)


def execute_provider_migration(session_id):
    logger.info("provider migration: starting session %s", session_id)

    task_id = _migration_task_id()
    start_claim = _report_migration(
        task_id, TASK_STATUS_STARTED, 0, "Provider migration started..."
    )
    if start_claim in ('missing', 'revoked') or start_claim is False:
        raise RuntimeError(
            f'provider migration task {task_id} has no live execution claim '
            f'({start_claim})'
        )
    try:
        conn = _get_dedicated_conn()
        try:
            conn.autocommit = False
        except Exception:
            pass

        cur = conn.cursor()

        cur.execute(_XACT_LOCK_SQL, (_ADVISORY_LOCK_KEY,))

        session = _load_session(cur, session_id)
        target_type = session['target_type']
        target_creds = session['target_creds']
        state = session['state']

        if session['status'] == 'completed':
            summary = _completed_summary(state)
            request_id = state.get('restart_request_id') or _restart_request_id(
                task_id, session_id
            )
            logger.warning(
                "provider migration: session %s is already applied; this run is a "
                "retry of a job that committed before restart acknowledgement",
                session_id,
            )
            if not state.get('restart_request_id'):
                _persist_completed_restart_state(conn, session_id, request_id)

            try:
                if not state.get('restart_acknowledged'):
                    acknowledged = _restart_request_result(request_id)
                    if acknowledged is not True:
                        request_id = _await_worker_restart(
                            conn,
                            session_id,
                            request_id,
                            alignment_task_id=state.get('alignment_task_id'),
                        )

                _finalize_restart_handshake(
                    conn,
                    session_id,
                    request_id,
                    task_id,
                    "Provider migration applied and worker restart acknowledged.",
                    details=summary,
                )
            except Exception:
                logger.exception(
                    "PROVIDER MIGRATION SESSION %s IS COMMITTED BUT THE WORKER RESTART "
                    "HANDSHAKE DID NOT COMPLETE. The catalogue swap is durable; the "
                    "workers must be restarted by hand so they pick up the new provider.",
                    session_id,
                )
                _report_migration(
                    task_id, TASK_STATUS_SUCCESS, 100,
                    "Provider migration was applied. The workers did not confirm their "
                    "restart, so restart AudioMuse by hand.",
                    details=summary,
                )
            return summary

        if session['status'] != 'dry_run_ready':
            raise RuntimeError(
                f"Cannot execute migration: session {session_id} is in status "
                f"'{session['status']}', expected 'dry_run_ready'"
            )

        mapping = _merge_mapping(state)
        if not mapping:
            raise RuntimeError(
                f"Refusing to execute migration session {session_id}: the mapping is "
                f"empty, which would unbind every song from the default server. Re-run "
                f"the dry run."
            )
        new_meta = _load_new_meta_from_table(cur, session_id)
        selected_libraries = state.get('selected_libraries')
        logger.info(
            "provider migration: %d songs will be repointed at the new provider; "
            "the catalogue itself is not touched", len(mapping),
        )

        alignment_task_id = str(uuid.uuid4())
        restart_request_id = _restart_request_id(task_id, session_id)
        try:
            with row_heartbeat(
                task_id,
                'the provider-migration transaction, which rewrites every catalogue '
                'id in one statement sequence and writes no row until it commits',
                stop_after_minutes=slow_step_budget_minutes(
                    QUEUE_WEDGED_MAIN_TASK_MINUTES
                ),
            ):
                index_rebuild_needed = _run_migration_transaction(
                    cur=cur,
                    mapping=mapping,
                    new_meta=new_meta,
                    target_type=target_type,
                    target_creds=target_creds,
                    session_id=session_id,
                    selected_libraries=selected_libraries,
                    alignment_task_id=alignment_task_id,
                    exec_task_id=task_id,
                    restart_request_id=restart_request_id,
                )
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise

        summary = {
            'ok': True,
            'matched': len(mapping),
            'index_rebuild_needed': bool(index_rebuild_needed),
        }
        _report_migration(
            task_id, TASK_STATUS_SUCCESS, 100,
            f"Provider migration complete: {len(mapping)} tracks repointed.",
            details=summary,
        )
        try:
            restart_request_id = _await_worker_restart(
                conn,
                session_id,
                restart_request_id,
                alignment_task_id=alignment_task_id,
            )
            _finalize_restart_handshake(
                conn,
                session_id,
                restart_request_id,
                task_id,
                f"Provider migration complete: {len(mapping)} tracks repointed; "
                "worker restart acknowledged.",
                details=summary,
            )
        except Exception:
            logger.exception(
                "PROVIDER MIGRATION SESSION %s IS COMMITTED BUT THE WORKER RESTART "
                "HANDSHAKE DID NOT COMPLETE. The catalogue swap is durable; the "
                "workers must be restarted by hand so they pick up the new provider.",
                session_id,
            )
            _report_migration(
                task_id, TASK_STATUS_SUCCESS, 100,
                f"Provider migration complete: {len(mapping)} tracks repointed. The "
                "workers did not confirm their restart, so restart AudioMuse by hand.",
                details=summary,
            )
        return summary
    except Exception:
        logger.exception("Provider migration failed for session %s", session_id)
        _report_migration(
            task_id, TASK_STATUS_FAILURE, 100,
            "Provider migration failed; check the container logs.",
        )
        raise


def _load_session(cur, session_id):
    cur.execute(
        "SELECT id, target_type, target_creds, state, status FROM migration_session WHERE id = %s",
        (session_id,),
    )
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"migration_session {session_id} not found")
    _id, target_type, target_creds_json, state_json, status = row
    try:
        creds = (
            json.loads(target_creds_json)
            if isinstance(target_creds_json, str)
            else target_creds_json
        )
    except Exception:
        creds = {}
    try:
        state = json.loads(state_json) if isinstance(state_json, str) else state_json
    except Exception:
        state = {}
    return {
        'id': _id,
        'target_type': target_type,
        'target_creds': creds,
        'state': state or {},
        'status': status,
    }


def build_mapping(state):
    dry = (state.get('dry_run') or {}).get('matches') or {}
    manual = state.get('manual_matches') or {}
    manual_unmatches = set(state.get('manual_unmatches') or [])

    merged = {}
    for old_id, new_id in dry.items():
        if old_id in manual_unmatches:
            continue
        merged[old_id] = new_id
    merged.update(manual)

    seen_new = {}
    deduped = {}
    dropped = []
    for old_id, new_id in merged.items():
        key = str(new_id)
        if key in seen_new:
            dropped.append((old_id, new_id, seen_new[key]))
            continue
        seen_new[key] = old_id
        deduped[old_id] = new_id
    return deduped, dropped


def _merge_mapping(state):
    deduped, dropped = build_mapping(state)
    if dropped:
        logger.warning(
            "provider migration: dropped %d mapping(s) that collided on "
            "new_id (multiple source rows pointed at the same target id); "
            "those rows will be orphaned on execute. First 10: %s",
            len(dropped),
            dropped[:10],
        )
    return deduped


def _migration_target_meta_exists(cur):
    cur.execute("SELECT to_regclass('public.migration_target_meta')")
    return cur.fetchone()[0] is not None


def _load_new_meta_from_table(cur, session_id):
    if not _migration_target_meta_exists(cur):
        logger.warning(
            "provider migration: migration_target_meta does not exist; item ids "
            "will be rewritten but the target's path/title/artist/album will not "
            "be applied to the catalogue"
        )
        return {}
    cur.execute(
        "SELECT new_id, path, title, artist, album, album_artist, year "
        "FROM migration_target_meta WHERE session_id = %s",
        (session_id,),
    )
    out = {}
    for r in cur.fetchall() or []:
        out[r[0]] = {
            'path': r[1],
            'title': r[2],
            'artist': r[3],
            'album': r[4],
            'album_artist': r[5],
            'year': r[6],
        }
    if not out:
        logger.warning(
            "provider migration: session %s has no target metadata rows; the "
            "catalogue keeps the SOURCE provider's metadata (re-run the dry run "
            "to collect it again)",
            session_id,
        )
    return out


def _populate_migration_map_table(cur, mapping):
    cur.execute(
        "CREATE TEMP TABLE item_id_migration_map ("
        " old_id TEXT PRIMARY KEY, "
        " new_id TEXT NOT NULL UNIQUE"
        ") ON COMMIT DROP"
    )
    _rows = list(mapping.items())
    for i in range(0, len(_rows), 1000):
        chunk = _rows[i : i + 1000]
        placeholders = ",".join(["(%s,%s)"] * len(chunk))
        flat = []
        for old_id, new_id in chunk:
            flat.append(old_id)
            flat.append(_sanitize_text(new_id) if isinstance(new_id, str) else new_id)
        cur.execute(
            "INSERT INTO item_id_migration_map (old_id, new_id) VALUES " + placeholders,  # nosec B608 - %s-placeholder string only; values are bound params
            flat,
        )
    cur.execute("ANALYZE item_id_migration_map")


def _stage_unsignable_items(cur):
    cur.execute(
        "CREATE TEMP TABLE migration_unsignable_items ("
        " item_id TEXT PRIMARY KEY"
        ") ON COMMIT DROP"
    )
    cur.execute(
        "INSERT INTO migration_unsignable_items (item_id) "
        "SELECT DISTINCT t.item_id FROM track_server_map t, music_servers s "
        "WHERE s.is_default AND t.server_id = s.server_id "
        "AND t.match_tier = 'analysis'"
    )
    return cur.rowcount


def _stage_chromaprint_carry(cur):
    cur.execute("SELECT to_regclass('public.chromaprint')")
    if cur.fetchone()[0] is None:
        return False
    cur.execute(
        "CREATE TEMP TABLE migration_chromaprint_carry ("
        " provider_track_id TEXT PRIMARY KEY, "
        " new_id TEXT NOT NULL UNIQUE"
        ") ON COMMIT DROP"
    )
    cur.execute(
        "INSERT INTO migration_chromaprint_carry (provider_track_id, new_id) "
        "SELECT DISTINCT ON (m.new_id) c.provider_track_id, m.new_id "
        "FROM chromaprint c "
        "JOIN music_servers s ON s.is_default AND c.server_id = s.server_id "
        "JOIN track_server_map t "
        "  ON t.server_id = c.server_id AND t.provider_track_id = c.provider_track_id "
        "JOIN item_id_migration_map m ON m.old_id = t.item_id "
        "ORDER BY m.new_id, (c.fingerprint IS NULL), c.provider_track_id"
    )
    return True


def _repoint_chromaprint(cur, staged):
    if not staged:
        return
    cur.execute(
        "DELETE FROM chromaprint c USING music_servers s "
        "WHERE s.is_default AND c.server_id = s.server_id "
        "AND NOT EXISTS (SELECT 1 FROM migration_chromaprint_carry k "
        "WHERE k.provider_track_id = c.provider_track_id)"
    )
    dropped = cur.rowcount
    cur.execute(
        "UPDATE chromaprint c SET provider_track_id = %s || k.new_id, updated_at = now() "
        "FROM migration_chromaprint_carry k, music_servers s "
        "WHERE s.is_default AND c.server_id = s.server_id "
        "AND c.provider_track_id = k.provider_track_id",
        (_MIG_TMP_PREFIX,),
    )
    carried = cur.rowcount
    cur.execute(
        "UPDATE chromaprint c "
        "SET provider_track_id = substr(c.provider_track_id, %s) "
        "FROM music_servers s "
        "WHERE s.is_default AND c.server_id = s.server_id "
        "AND c.provider_track_id LIKE %s",
        (len(_MIG_TMP_PREFIX) + 1, _MIG_TMP_PREFIX + '%'),
    )
    if carried or dropped:
        logger.info(
            "provider migration: carried %d Chromaprint fingerprint(s) onto the new "
            "provider ids, dropped %d that no longer map to anything",
            carried, dropped,
        )


def _apply_new_meta(cur, new_meta):
    if not new_meta:
        return
    cur.execute(
        "CREATE TEMP TABLE migration_new_meta ("
        " new_id TEXT PRIMARY KEY, "
        " new_path TEXT, new_title TEXT, new_artist TEXT, "
        " new_album TEXT, new_album_artist TEXT, new_year INTEGER"
        ") ON COMMIT DROP"
    )
    _metas = list(new_meta.items())
    for i in range(0, len(_metas), 500):
        chunk = _metas[i : i + 500]
        placeholders = ",".join(["(%s,%s,%s,%s,%s,%s,%s)"] * len(chunk))
        flat = []
        for new_id, meta in chunk:
            flat.extend(
                (
                    _sanitize_text(new_id),
                    _sanitize_text(meta.get('path')),
                    _sanitize_text(meta.get('title')),
                    _sanitize_text(meta.get('artist')),
                    _sanitize_text(meta.get('album')),
                    _sanitize_text(meta.get('album_artist')),
                    meta.get('year'),
                )
            )
        cur.execute(
            "INSERT INTO migration_new_meta "
            "(new_id, new_path, new_title, new_artist, new_album, new_album_artist, new_year) "
            "VALUES " + placeholders,  # nosec B608 - %s-placeholder string only; values are bound params
            flat,
        )
    cur.execute(
        "UPDATE score s SET "
        "  title        = COALESCE(n.new_title,        s.title), "
        "  author       = COALESCE(n.new_artist,       s.author), "
        "  album        = COALESCE(n.new_album,        s.album), "
        "  album_artist = COALESCE(n.new_album_artist, s.album_artist), "
        "  year         = COALESCE(n.new_year,         s.year) "
        "FROM migration_new_meta n, item_id_migration_map m "
        "WHERE m.new_id = n.new_id AND s.item_id = m.old_id"
    )
    cur.execute(
        "UPDATE track_server_map t SET file_path = n.new_path, updated_at = now() "
        "FROM migration_new_meta n, item_id_migration_map m, music_servers s "
        "WHERE m.new_id = n.new_id AND t.item_id = m.old_id "
        "AND s.is_default AND t.server_id = s.server_id "
        "AND n.new_path IS NOT NULL"
    )


def _clear_default_server_artist_map(cur):
    cur.execute("SELECT to_regclass('public.artist_server_map')")
    if cur.fetchone()[0] is not None:
        cur.execute(
            "DELETE FROM artist_server_map a USING music_servers s "
            "WHERE s.is_default AND a.server_id = s.server_id"
        )
        if cur.rowcount:
            logger.info(
                "provider migration: cleared %d stale artist id(s) of the default server",
                cur.rowcount,
            )


def _stage_post_migration_alignment(cur, task_id):
    if not task_id:
        return
    cur.execute("SELECT to_regclass('public.task_status')")
    if cur.fetchone()[0] is None:
        return
    from tasks.multiserver_sync import insert_pending_sweep_row

    insert_pending_sweep_row(
        cur, task_id,
        'Aligning the migrated server: rebuilding artist ids and file paths.',
    )


def _restart_handshake_details(session_id, restart_request_id):
    message = 'Provider swap committed; waiting for worker restart acknowledgement.'
    return {
        'message': message,
        'status_message': message,
        'phase': 'restart_handshake',
        'provider_migration_committed': True,
        'migration_session_id': session_id,
        'restart_request_id': restart_request_id,
    }


def _stage_restart_handshake(cur, task_id, session_id, restart_request_id):
    if not task_id:
        return
    taskqueue.take_start_lock(conn=cur.connection)

    cur.execute(
        "SELECT status FROM task_status WHERE task_id = %s FOR UPDATE",
        (task_id,),
    )
    row = cur.fetchone()
    live_statuses = (
        TASK_STATUS_PENDING,
        TASK_STATUS_STARTED,
        TASK_STATUS_PROGRESS,
    )
    if not row or row[0] not in live_statuses:
        raise RuntimeError(
            f'provider migration execution claim {task_id} was cancelled or lost '
            'before commit'
        )

    details = _restart_handshake_details(session_id, restart_request_id)
    cur.execute(
        "UPDATE task_status SET status = %s, progress = 99, details = %s, "
        "timestamp = NOW(), end_time = NULL WHERE task_id = %s AND status IN %s "
        "RETURNING task_id",
        (
            TASK_STATUS_STARTED,
            json.dumps(details),
            task_id,
            live_statuses,
        ),
    )
    if cur.fetchone() is None:
        raise RuntimeError(
            f'provider migration execution claim {task_id} changed before commit'
        )


def _report_playlists_bound_to_default_server(cur):
    cur.execute("SELECT to_regclass('public.playlist')")
    if cur.fetchone()[0] is None:
        return
    cur.execute("SAVEPOINT mig_playlist_report")
    try:
        cur.execute(
            "SELECT count(DISTINCT p.playlist_name) FROM playlist p, music_servers s "
            "WHERE s.is_default AND p.server_id = s.server_id"
        )
        row = cur.fetchone()
        cur.execute("RELEASE SAVEPOINT mig_playlist_report")
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT mig_playlist_report")
        logger.exception(
            "provider migration: could not count the stored playlists (reporting only)"
        )
        return
    stored = row[0] if row else 0
    if stored:
        logger.warning(
            "provider migration: %d stored playlist(s) are still recorded against the "
            "migrated server. Their songs survived and re-translate to the new provider "
            "ids, but the playlists themselves live on the OLD provider: re-run "
            "clustering to create them on the target.",
            stored,
        )


def _run_migration_transaction(
    cur,
    mapping,
    new_meta,
    target_type,
    target_creds,
    session_id,
    selected_libraries=None,
    alignment_task_id=None,
    exec_task_id=None,
    restart_request_id=None,
):
    cur.execute(_XACT_LOCK_SQL, (_ADVISORY_LOCK_KEY,))

    _populate_migration_map_table(cur, mapping)
    _stage_unsignable_items(cur)
    chromaprint_staged = _stage_chromaprint_carry(cur)

    cur.execute(
        "DELETE FROM track_server_map t USING music_servers s "
        "WHERE s.is_default AND t.server_id = s.server_id "
        "AND NOT EXISTS (SELECT 1 FROM item_id_migration_map m WHERE m.old_id = t.item_id) "
        "RETURNING t.item_id, t.provider_track_id, t.file_path"
    )
    orphans = {}
    for item_id, provider_id, file_path in cur.fetchall() or []:
        orphans.setdefault(item_id, {
            'item_id': item_id,
            'old_id': provider_id or '',
            'old_path': file_path or '',
        })
    orphan_rows = list(orphans.values())

    cur.execute(
        "DELETE FROM track_server_map t USING music_servers s "
        "WHERE s.is_default AND t.server_id = s.server_id "
        "AND t.ctid <> (SELECT min(t2.ctid) FROM track_server_map t2 "
        "WHERE t2.item_id = t.item_id AND t2.server_id = t.server_id)"
    )

    cur.execute(
        "UPDATE track_server_map t SET provider_track_id = %s || m.new_id, "
        "match_tier = CASE WHEN EXISTS (SELECT 1 FROM migration_unsignable_items u "
        "  WHERE u.item_id = t.item_id) THEN 'analysis' ELSE 'default' END, "
        "updated_at = now() "
        "FROM item_id_migration_map m, music_servers s "
        "WHERE s.is_default AND t.server_id = s.server_id AND t.item_id = m.old_id",
        (_MIG_TMP_PREFIX,),
    )
    cur.execute(
        "UPDATE track_server_map t "
        "SET provider_track_id = substr(t.provider_track_id, %s) "
        "FROM music_servers s "
        "WHERE s.is_default AND t.server_id = s.server_id "
        "AND t.provider_track_id LIKE %s",
        (len(_MIG_TMP_PREFIX) + 1, _MIG_TMP_PREFIX + '%'),
    )

    _repoint_chromaprint(cur, chromaprint_staged)
    _apply_new_meta(cur, new_meta)
    _clear_default_server_artist_map(cur)
    _report_playlists_bound_to_default_server(cur)

    _write_provider_to_default_server(
        cur, target_type, target_creds, selected_libraries=selected_libraries
    )
    _purge_media_keys_from_app_config(cur)
    _stage_post_migration_alignment(cur, alignment_task_id)
    _stage_restart_handshake(
        cur, exec_task_id, session_id, restart_request_id
    )

    cur.execute(
        "UPDATE migration_session SET status = 'completed', completed_at = NOW(), "
        "state = %s::jsonb WHERE id = %s RETURNING id",
        (
            json.dumps(
                {
                    'post_migration': {
                        'orphans': orphan_rows[:_ORPHAN_REPORT_MAX_ROWS],
                        'orphan_total': len(orphan_rows),
                        'matched': len(mapping or {}),
                    },
                    'exec_task_id': exec_task_id,
                    'alignment_task_id': alignment_task_id,
                    'restart_request_id': restart_request_id,
                    'restart_acknowledged': False,
                }
            ),
            session_id,
        ),
    )
    if cur.fetchone() is None:
        raise RuntimeError(
            f"migration_session {session_id} disappeared before the completion "
            f"marker could be written; refusing to commit the repoint"
        )
    if _migration_target_meta_exists(cur):
        cur.execute("DELETE FROM migration_target_meta WHERE session_id = %s", (session_id,))

    return False


def _cleaned_libraries_value(selected_libraries):
    cleaned = [_sanitize_text(str(name)).strip() for name in (selected_libraries or [])]
    cleaned = [name for name in cleaned if name and ',' not in name]
    return ','.join(cleaned)


def _write_provider_to_default_server(cur, target_type, target_creds, selected_libraries=None):
    import uuid as _uuid

    from psycopg2.extras import Json

    cur.execute("SELECT to_regclass('public.music_servers') IS NOT NULL")
    if not cur.fetchone()[0]:
        return
    creds = Json(dict(target_creds or {}))
    libraries = _cleaned_libraries_value(selected_libraries)
    cur.execute(
        "UPDATE music_servers SET server_type = %s, creds = %s, music_libraries = %s, "
        "track_count = NULL, updated_at = now() WHERE is_default",
        (target_type, creds, libraries),
    )
    if cur.rowcount:
        logger.info(
            "provider migration: music_servers default row now targets '%s'",
            target_type,
        )
        return
    cur.execute(
        "INSERT INTO music_servers "
        "(server_id, name, server_type, creds, music_libraries, is_default) "
        "VALUES (%s, %s, %s, %s, %s, TRUE)",
        (_uuid.uuid4().hex, (target_type or 'media server').capitalize(),
         target_type, creds, libraries),
    )
    logger.warning(
        "provider migration: no default server existed; created one for '%s'",
        target_type,
    )


def _purge_media_keys_from_app_config(cur):
    from database import purge_media_keys_from_app_config

    removed = purge_media_keys_from_app_config(cur)
    if removed:
        logger.info(
            "provider migration: removed %d legacy media-server key(s) from app_config",
            removed,
        )


def _post_commit_reload(alignment_task_id=None, *, restart_request_id=None):
    try:
        from tasks.mediaserver import registry

        registry.invalidate_server_cache()
    except Exception as e:
        logger.warning("registry cache invalidation failed: %s", e)
    try:
        import config

        config.refresh_config()
    except Exception as e:
        logger.warning("config.refresh_config() failed: %s", e)
    _enqueue_post_migration_alignment(alignment_task_id)
    return _publish_restart_with_retries(restart_request_id)


def _publish_restart_with_retries(request_id):
    if not request_id:
        raise ValueError('a durable restart request id is required')
    for attempt in range(1, _RESTART_PUBLISH_ATTEMPTS + 1):
        try:
            import restart_manager

            if restart_manager.publish_restart_request(request_id=request_id) is True:
                return True
        except Exception as e:
            logger.warning("restart_manager.publish_restart_request() failed: %s", e)
        if attempt < _RESTART_PUBLISH_ATTEMPTS:
            logger.warning(
                "provider migration: restart request not delivered (attempt %d of "
                "%d); retrying in %ss",
                attempt, _RESTART_PUBLISH_ATTEMPTS, _RESTART_PUBLISH_RETRY_SECONDS,
            )
            time.sleep(_RESTART_PUBLISH_RETRY_SECONDS)
    logger.error(
        "PROVIDER MIGRATION: WORKER RESTART %s WAS NOT ACKNOWLEDGED after %d "
        "attempts. The catalogue is migrated and the registry points at the new "
        "provider, but the migration task remains active until workers confirm "
        "their restart. RESTART AUDIOMUSE MANUALLY if the listeners cannot recover.",
        request_id,
        _RESTART_PUBLISH_ATTEMPTS,
    )
    return False


def _enqueue_post_migration_alignment(alignment_task_id=None):
    if not alignment_task_id:
        logger.info(
            "provider migration: this run has no alignment id of its own (a retry of "
            "an already applied session); the alignment row its first run committed "
            "is left to the janitor rather than queueing a second one"
        )
        return
    try:
        from tasks import multiserver_sync

        task_id = multiserver_sync.enqueue_server_alignment(
            message='Aligning the migrated server: rebuilding artist ids and file paths.',
            task_id=alignment_task_id,
            parent_task_id=_migration_task_id(),
        )
        if task_id:
            logger.info(
                "provider migration: queued alignment %s of the migrated server; it "
                "rebuilds the artist ids the swap cleared and refreshes the file paths",
                task_id,
            )
        else:
            logger.warning(
                "provider migration: no alignment was queued; run 'Align' from the "
                "music servers page so artist ids are rebuilt for the new provider"
            )
    except Exception:
        logger.exception(
            "provider migration: could not queue the post-migration alignment; run "
            "'Align' from the music servers page to rebuild the artist ids"
        )


def _decoded_state(raw_state):
    if isinstance(raw_state, str):
        try:
            return json.loads(raw_state) or {}
        except Exception:
            return {}
    return raw_state or {}


def recover_provider_migration_restart_handshakes():
    conn = _get_dedicated_conn()
    try:
        try:
            conn.autocommit = False
        except Exception:
            pass
        cur = conn.cursor()
        cur.execute("SELECT to_regclass('migration_session')")
        if cur.fetchone()[0] is None:
            conn.rollback()
            return 0
        cur.execute(
            "SELECT id, state FROM migration_session WHERE status = 'completed' "
            "AND state ? 'restart_request_id' "
            "AND NOT COALESCE((state->>'restart_acknowledged')::boolean, false)"
        )
        candidates = cur.fetchall() or []
        conn.rollback()
        reserved = 0

        for session_id, raw_state in candidates:
            state = _decoded_state(raw_state)
            root_task_id = state.get('exec_task_id')
            request_id = state.get('restart_request_id')
            current_recovery_id = state.get(_RESTART_RECOVERY_TASK_KEY)
            if not root_task_id or not request_id:
                logger.error(
                    "provider migration session %s has an incomplete restart tombstone",
                    session_id,
                )
                continue

            authority_id = current_recovery_id or root_task_id
            cur.execute(
                "SELECT status FROM task_status WHERE task_id = %s", (authority_id,)
            )
            authority_row = cur.fetchone()
            authority_status = authority_row[0] if authority_row else None
            if authority_status in (TASK_STATUS_NEW, TASK_STATUS_RUNNING):
                continue

            recovery_id = str(uuid.uuid4())
            try:
                taskqueue.take_start_lock(conn=conn)
                cur.execute(
                    "SELECT pg_try_advisory_xact_lock(%s)", (_ADVISORY_LOCK_KEY,)
                )
                if not cur.fetchone()[0]:
                    conn.rollback()
                    continue
                cur.execute(
                    "SELECT state FROM migration_session WHERE id = %s "
                    "AND status = 'completed' FOR UPDATE",
                    (session_id,),
                )
                locked = cur.fetchone()
                locked_state = _decoded_state(locked[0]) if locked else {}
                if (
                    not locked
                    or locked_state.get('restart_acknowledged') is True
                    or locked_state.get('exec_task_id') != root_task_id
                    or locked_state.get('restart_request_id') != request_id
                    or locked_state.get(_RESTART_RECOVERY_TASK_KEY)
                    != current_recovery_id
                ):
                    conn.rollback()
                    continue

                details = json.dumps(
                    _restart_handshake_details(session_id, request_id)
                )
                cur.execute(
                    "INSERT INTO task_status "
                    "(task_id, task_type, status, progress, details, timestamp, "
                    "start_time, end_time) VALUES (%s, %s, %s, 99, %s, NOW(), %s, NULL) "
                    "ON CONFLICT (task_id) DO UPDATE SET status = EXCLUDED.status, "
                    "progress = 99, details = EXCLUDED.details, timestamp = NOW(), "
                    "end_time = NULL",
                    (
                        root_task_id,
                        MIGRATION_TASK_TYPE,
                        TASK_STATUS_STARTED,
                        details,
                        time.time(),
                    ),
                )
                cur.execute(
                    "UPDATE migration_session SET state = jsonb_set("
                    "COALESCE(state, '{}'::jsonb), %s, %s::jsonb, true) "
                    "WHERE id = %s AND status = 'completed' "
                    "AND NOT COALESCE((state->>'restart_acknowledged')::boolean, false) "
                    "RETURNING id",
                    (
                        [_RESTART_RECOVERY_TASK_KEY],
                        json.dumps(recovery_id),
                        session_id,
                    ),
                )
                if cur.fetchone() is None:
                    conn.rollback()
                    continue

                taskqueue.enqueue(
                    'tasks.provider_migration_tasks.resume_provider_migration_restart',
                    args=(session_id, root_task_id),
                    task_id=recovery_id,
                    task_type=MIGRATION_TASK_TYPE,
                    queue=taskqueue.QUEUE_HIGH,
                    parent_task_id=root_task_id,
                    details={'message': 'Resuming the provider-migration restart handshake.'},
                    conn=conn,
                )
                conn.commit()
                reserved += 1
            except Exception:
                conn.rollback()
                logger.exception(
                    "Could not reserve restart recovery for provider session %s",
                    session_id,
                )
        return reserved
    finally:
        try:
            conn.close()
        except Exception:
            logger.debug("Could not close provider restart-recovery DB connection", exc_info=True)


def resume_provider_migration_restart(session_id, root_task_id):
    recovery_id = _migration_task_id()
    conn = _get_dedicated_conn()
    try:
        try:
            conn.autocommit = False
        except Exception:
            pass
        cur = conn.cursor()
        cur.execute(_XACT_LOCK_SQL, (_ADVISORY_LOCK_KEY,))
        session = _load_session(cur, session_id)
        state = session['state']
        if session['status'] != 'completed':
            raise RuntimeError(
                f'provider migration session {session_id} is not completed'
            )
        if state.get('exec_task_id') != root_task_id:
            raise RuntimeError(
                f'provider migration root {root_task_id} no longer owns session {session_id}'
            )
        if recovery_id and state.get(_RESTART_RECOVERY_TASK_KEY) != recovery_id:
            raise RuntimeError(
                f'provider restart recovery {recovery_id} was superseded'
            )
        request_id = state.get('restart_request_id')
        if not request_id:
            raise RuntimeError(
                f'provider migration session {session_id} has no restart request id'
            )
        conn.commit()

        if not state.get('restart_acknowledged'):
            if _restart_request_result(request_id) is not True:
                request_id = _await_worker_restart(
                    conn,
                    session_id,
                    request_id,
                    alignment_task_id=state.get('alignment_task_id'),
                )

        summary = _completed_summary(state)
        _finalize_restart_handshake(
            conn,
            session_id,
            request_id,
            root_task_id,
            'Provider migration applied and worker restart acknowledged.',
            details=summary,
        )
        return summary
    finally:
        try:
            conn.close()
        except Exception:
            logger.debug("Could not close provider restart recovery connection", exc_info=True)


def dry_run_provider_migration(session_id, allow_title_artist_only=False):
    from flask_app import app

    with app.app_context():
        import app_provider_migration
        job_id = _migration_task_id()
        try:
            app_provider_migration._validate_planner_worker_claim(
                session_id, 'dry_run_task_id', job_id
            )
            return app_provider_migration.run_dry_run_core(
                session_id, allow_title_artist_only=allow_title_artist_only
            )
        finally:
            app_provider_migration._clear_planner_claim(
                session_id, 'dry_run_task_id', job_id
            )


def source_refresh_provider_migration(session_id):
    from flask_app import app

    with app.app_context():
        import app_provider_migration
        job_id = _migration_task_id()
        try:
            app_provider_migration._validate_planner_worker_claim(
                session_id, 'source_refresh_task_id', job_id
            )
            return app_provider_migration.run_source_refresh_core(session_id)
        finally:
            app_provider_migration._clear_planner_claim(
                session_id, 'source_refresh_task_id', job_id
            )
