# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Provider-migration Flask blueprint (migration_bp) for switching media servers.

Single add-on entry point for switching the active media-server provider on a
running install: a wizard page at ``/provider-migration`` plus the backing REST
API under ``/api/migration/*``. Target-provider probing runs through
``tasks.provider_probe`` and the long migration through the high-priority queue.

Main Features:
* Full wizard flow: session start, probe test, library select, album search,
  source-path refresh, dry-run, manual match/skip, finalize, and execute, with
  status polling for the async queue jobs.
* Target credentials stay in ``migration_session.target_creds`` (never read
  from ``config``), so the live provider keeps working throughout; a successful
  execute writes the new settings to ``app_config`` and restarts via
  ``restart_manager``. ``provider_probe`` is lazily imported to avoid loading
  ``tasks/__init__.py`` at module import.
"""

import csv
import io
import json
import logging
import uuid

from flask import Blueprint, jsonify, render_template, request
from psycopg2 import sql as pgsql

# App-level singletons (the DB connection and the task queue). Importing here keeps
# the blueprint file self-contained - the rest of the app doesn't need to hand
# anything in.
from app_helper import cancel_job_and_children_recursive
from app_logging import sanitize_log_value
from config import TASK_STATUS_PENDING, TASK_STATUS_FAILURE
from database import (
    GLOBAL_CANCEL_EPOCH_KEY,
    NON_BLOCKING_TASK_TYPES,
    get_app_config_value,
    get_db,
    get_active_main_task,
    save_task_status,
    main_task_start_lock,
)
from tasks.provider_migration_tasks import (
    MIGRATION_TASK_TYPE,
    MIGRATION_PLANNER_TASK_TYPE,
    _ADVISORY_LOCK_KEY,
)
import config
import taskqueue
from database import coerce_db_details
from ssrf_guard import validate_outbound_url
from tasks.mediaserver.helper import detect_path_format as _detect_path_format

logger = logging.getLogger(__name__)

migration_bp = Blueprint('migration_bp', __name__)


# ---------------------------------------------------------------------------
# Lazy provider_probe import - keeps the _import_module bypass test happy
# because we don't trigger ``tasks/__init__.py`` at module-load time.
# ---------------------------------------------------------------------------


class _LazyProbe:

    _real = None

    def _load(self):
        if self._real is None:
            import importlib

            self._real = importlib.import_module('tasks.provider_probe')
        return self._real

    def __getattr__(self, name):
        return getattr(self._load(), name)


provider_probe = _LazyProbe()


# ---------------------------------------------------------------------------
# Supported target providers (what the tool knows how to talk to)
# ---------------------------------------------------------------------------

_SUPPORTED_TARGETS = frozenset({'jellyfin', 'navidrome', 'emby', 'lyrion', 'plex', 'ampache'})

# A planner id in session state is a durable reservation, not just a cache of the
# job's current status. The claim and the queue row are one transaction, so a
# later 'this task no longer exists' reading can reconcile the reservation with
# no pre-enqueue race to worry about.
_PLANNER_TASK_KEYS = ('dry_run_task_id', 'source_refresh_task_id')
_MIGRATION_TASK_KEYS = _PLANNER_TASK_KEYS + ('exec_task_id',)


class _PlanningClaimError(RuntimeError):

    def __init__(self, message, status_code=409):
        super().__init__(message)
        self.status_code = status_code
        self.user_message = message


def _task_is_the_execute_job(task_id):
    safe_task_id = sanitize_log_value(task_id)
    for attempt in (1, 2):
        try:
            db = get_db()
            with db.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM migration_session WHERE state->>'exec_task_id' = %s LIMIT 1",
                    (str(task_id),),
                )
                return cur.fetchone() is not None
        except Exception:
            # get_db() hands back the SAME cached connection, so the retry only has
            # a chance if this aborted transaction is cleared first.
            try:
                get_db().rollback()
            except Exception:
                logger.debug("Rollback before the classify retry failed", exc_info=True)
            if attempt == 1:
                logger.warning(
                    "Could not classify migration task %s; retrying once",
                    safe_task_id, exc_info=True,
                )
                continue
            logger.exception("Retry also failed for migration task %s", safe_task_id)
    return False


def _task_statuses_by_id(task_ids):
    from database import get_task_statuses

    ids = [str(task_id) for task_id in task_ids if task_id]
    if not ids:
        return {}
    statuses = get_task_statuses(ids)
    return {task_id: statuses.get(task_id) for task_id in ids}


def _task_is_live(status):
    return status in (config.TASK_STATUS_NEW, config.TASK_STATUS_RUNNING)


def _clear_stale_planner_reservation(cur, session_id, key, job_id):
    try:
        cur.execute("SAVEPOINT planner_reconcile")
        cur.execute(
            "UPDATE migration_session SET state = state - %s "
            "WHERE id = %s AND state->>%s = %s",
            (key, session_id, key, job_id),
        )
        cur.execute("RELEASE SAVEPOINT planner_reconcile")
    except Exception:
        logger.exception(
            "Could not clear the stale %s reservation on migration session %s; "
            "the next control retries it",
            key, session_id,
        )
        try:
            cur.execute("ROLLBACK TO SAVEPOINT planner_reconcile")
            cur.execute("RELEASE SAVEPOINT planner_reconcile")
        except Exception:
            logger.exception(
                "Could not roll back the planner reconcile savepoint for session %s",
                session_id,
            )


def _session_state(raw_state):
    if isinstance(raw_state, str):
        try:
            return json.loads(raw_state) or {}
        except (TypeError, ValueError):
            return {}
    return raw_state or {}


def _migration_job_in_flight(cur, keys=_MIGRATION_TASK_KEYS):
    # Dry-run and source-refresh write no task row and hold no long-lived DB lock,
    # so their durable reservation on the session is the only record that one is
    # in flight. A read failure keeps the reservation live; a definitive 'gone' is
    # reconciled only after this control acquired the same advisory lock that
    # covered claim plus enqueue.
    cur.execute(
        "SELECT id, state FROM migration_session"
    )
    rows = [(row[0], _session_state(row[1])) for row in (cur.fetchall() or [])]
    reservations = [
        (session_id, key, state.get(key))
        for session_id, state in rows
        for key in keys
        if state.get(key)
    ]
    job_ids = [reservation[2] for reservation in reservations]
    if not job_ids:
        return False
    try:
        jobs = _task_statuses_by_id(job_ids)
        for session_id, key, job_id in reservations:
            job = jobs.get(job_id)
            stale_planner = key in _PLANNER_TASK_KEYS and (
                not _task_is_live(job)
            )
            if stale_planner:
                # This predicate runs on the CALLER's cursor, inside transactions
                # that may end in a 409 and be rolled back. A SAVEPOINT keeps a
                # failed reconciliation from poisoning that transaction, and the
                # clear is retried on the next control either way.
                _clear_stale_planner_reservation(cur, session_id, key, job_id)
                continue
            # Execute has a task_status NEW/RUNNING claim as its second
            # authority; a successful missing probe is reconciled by that gate,
            # while a completed missing job is exactly what makes its compact
            # tombstone safe to prune.
            if _task_is_live(job):
                return True
    except Exception:
        logger.exception(
            "COULD NOT CHECK WHETHER A MIGRATION JOB IS STILL RUNNING. Assuming one is, "
            "so a live dry run is never deleted; start the session again once the database is back"
        )
        return True
    return False


def _live_planner_job_id(cur):
    # PLANNER keys only, never exec_task_id: a dry-run/source-refresh is a pure
    # fetch with no external side effects, so Discard can cancel it. An active
    # EXECUTE writes to the target server, so it stays behind the hard
    # _no_migration_executing block instead of ever being auto-cancelled here.
    # Session-agnostic on purpose, like _migration_job_in_flight: the shared
    # migration advisory lock treats planning as a system-wide singleton.
    cur.execute("SELECT state FROM migration_session")
    states = [_session_state(row[0]) for row in (cur.fetchall() or [])]
    job_ids = [
        state.get(key)
        for state in states
        for key in _PLANNER_TASK_KEYS
        if state.get(key)
    ]
    if not job_ids:
        return None
    jobs = _task_statuses_by_id(job_ids)
    for job_id in job_ids:
        if _task_is_live(jobs.get(job_id)):
            return job_id
    return None


def _completed_sessions_safe_to_prune(cur):
    cur.execute(
        "SELECT id, state->>'exec_task_id', "
        "COALESCE((state->>'restart_acknowledged')::boolean, false) "
        "FROM migration_session "
        "WHERE status = 'completed'"
    )
    rows = cur.fetchall() or []
    job_ids = [row[1] for row in rows if row[1]]
    if not job_ids:
        return []
    try:
        jobs = _task_statuses_by_id(job_ids)
    except Exception:
        logger.exception(
            "COULD NOT CHECK COMPLETED MIGRATION RETRIES. Keeping every completion "
            "tombstone so a delayed retry can still prove the swap was applied"
        )
        return []
    return [
        session_id
        for session_id, job_id, restart_acknowledged in rows
        if restart_acknowledged
        and job_id
        and not _task_is_live(jobs.get(job_id))
    ]


def _restart_handshake_pending(cur):
    cur.execute(
        "SELECT EXISTS (SELECT 1 FROM migration_session "
        "WHERE status = 'completed' AND state ? 'restart_request_id' "
        "AND NOT COALESCE((state->>'restart_acknowledged')::boolean, false))"
    )
    return bool(cur.fetchone()[0])


def _no_migration_executing(cur):
    # The advisory lock only covers the window in which a WORKER is running. An
    # execute that is merely QUEUED holds nothing, so the session it is about to
    # read was still deletable between the enqueue and the worker picking it up.
    # The execute endpoint writes a PENDING provider_migration row under the same
    # lock, so that row is the missing half of the guard.
    if get_active_main_task(task_type=MIGRATION_TASK_TYPE):
        return False
    # The task row is normally the admission barrier, but this transactionally
    # committed session marker is the final authority if an external janitor or
    # operator incorrectly terminalized that row.  Legacy completed rows have no
    # restart_request_id and do not participate in the new handshake.
    cur.execute("SELECT pg_try_advisory_xact_lock(%s)", (_ADVISORY_LOCK_KEY,))
    if not cur.fetchone()[0]:
        return False
    if _restart_handshake_pending(cur):
        return False
    return True


def _global_cancel_epoch(db):
    raw = get_app_config_value(GLOBAL_CANCEL_EPOCH_KEY, default='0', conn=db)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _claim_and_enqueue_planner(
    session_id,
    state_key,
    func_name,
    job_args,
    *,
    claim_status='in_progress',
):
    db = get_db()
    cancel_epoch = _global_cancel_epoch(db)
    with main_task_start_lock():
        if _global_cancel_epoch(db) != cancel_epoch:
            raise _PlanningClaimError(
                'A global cancellation completed while this planner was waiting. '
                'Start the planner again if you still want to run it.'
            )
        return _claim_and_enqueue_planner_locked(
            db,
            session_id,
            state_key,
            func_name,
            job_args,
            claim_status=claim_status,
        )


def _claim_and_enqueue_planner_locked(
    db,
    session_id,
    state_key,
    func_name,
    job_args,
    *,
    claim_status='in_progress',
):
    if state_key not in _PLANNER_TASK_KEYS:
        raise ValueError(f'unsupported planner state key: {state_key}')

    with db.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_xact_lock(%s)", (_ADVISORY_LOCK_KEY,))
        if not cur.fetchone()[0]:
            raise _PlanningClaimError(
                'A migration is already running. Wait for it to finish.'
            )
        cur.execute(
            "SELECT status, (id = (SELECT MAX(id) FROM migration_session)), "
            "state->>'dry_run_task_id', state->>'source_refresh_task_id' "
            "FROM migration_session WHERE id = %s FOR UPDATE",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            raise _PlanningClaimError('session not found', 404)
        status, is_current, dry_run_id, refresh_id = row
        if status in ('completed', 'failed'):
            raise _PlanningClaimError('cannot plan against a finished migration session')
        if not is_current:
            raise _PlanningClaimError(
                'This is not the current migration session. Start again from the wizard.'
            )
        if get_active_main_task(task_type=MIGRATION_TASK_TYPE):
            raise _PlanningClaimError(
                'A provider migration is queued or running. Wait for it to finish.'
            )
        if _restart_handshake_pending(cur):
            raise _PlanningClaimError(
                'The previous provider swap is still waiting for worker restart '
                'acknowledgement.'
            )

        claims = {
            'dry_run_task_id': dry_run_id,
            'source_refresh_task_id': refresh_id,
        }
        existing_ids = [job_id for job_id in claims.values() if job_id]
        try:
            jobs = _task_statuses_by_id(existing_ids)
        except Exception as exc:
            # With a persisted claim and no readable status, allowing another
            # planner would create two writers for the same plan.
            if existing_ids:
                raise _PlanningClaimError(
                    'Could not verify the existing migration planner job. Try again '
                    'when the database is available.',
                    503,
                ) from exc
            jobs = {}

        for key, existing_id in claims.items():
            if not existing_id or key == state_key:
                continue
            other_job = jobs.get(existing_id)
            if _task_is_live(other_job):
                raise _PlanningClaimError(
                    'Another migration planner job is reserved or running. Wait for '
                    'it to finish.',
                )

        existing_id = claims[state_key]
        existing_job = jobs.get(existing_id) if existing_id else None
        if existing_id and _task_is_live(existing_job):
            db.commit()
            return existing_id, True

        # A missing same-kind job is the recoverable ambiguous-enqueue case: reuse
        # its deterministic id.  A conclusively terminal job gets a fresh id.
        job_id = (
            existing_id
            if existing_id and existing_job is None
            else str(uuid.uuid4())
        )
        # A DRY RUN rebuilds the plan, so demoting to in_progress is right. A
        # SOURCE-PATH REFRESH only fills in overrides; demoting it threw away a
        # finalized dry_run_ready and Execute then refused the migration, forcing
        # the user to redo the whole dry run.
        cur.execute(
            "UPDATE migration_session SET "
            "state = jsonb_set(COALESCE(state, '{}'::jsonb), %s, %s::jsonb, true), "
            "status = COALESCE(%s, status) "
            "WHERE id = %s AND status NOT IN ('completed', 'failed') "
            "AND id = (SELECT MAX(id) FROM migration_session) RETURNING id",
            ([state_key], json.dumps(job_id), claim_status, session_id),
        )
        if cur.fetchone() is None:
            raise _PlanningClaimError(
                'The migration session changed before the planner could be reserved.'
            )

        # The reservation and the queue row are the same transaction, so there
        # is no outcome to resolve: either both committed or neither did.
        # max_attempts=0: a dry-run/source-refresh re-fetches the ENTIRE source or
        # target catalogue with no checkpointing, so the normal worker-death retry
        # (attempts+1 <= max_attempts, i.e. max_attempts=1 would still allow one
        # silent retry) would restart a multi-minute fetch from scratch while
        # leaving the session claimed - Discard blocked - for the whole extra
        # attempt. Failing on the first worker death is more useful than an
        # invisible do-over; it does not affect the task's normal first run.
        taskqueue.enqueue(
            func_name,
            args=tuple(job_args),
            task_id=job_id,
            task_type=MIGRATION_PLANNER_TASK_TYPE,
            queue=taskqueue.QUEUE_HIGH,
            max_attempts=0,
            details={'message': 'Migration planner queued.'},
            conn=db,
        )

    db.commit()
    return job_id, False


def _validate_planner_worker_claim(session_id, state_key, job_id):
    if not job_id:
        return
    db = get_db()
    with db.cursor() as cur:
        # The enqueueing request holds this transaction lock until its claim
        # commits.  A very fast worker must wait here rather than read the old
        # MVCC snapshot and incorrectly reject a valid just-enqueued job.
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_ADVISORY_LOCK_KEY,))
        cur.execute(
            "SELECT status, (id = (SELECT MAX(id) FROM migration_session)), "
            "state->>%s FROM migration_session WHERE id = %s",
            (state_key, session_id),
        )
        row = cur.fetchone()
    if not row:
        db.rollback()
        raise RuntimeError(f'migration session {session_id} no longer exists')
    status, is_current, claimed_job_id = row
    if status in ('completed', 'failed') or not is_current or claimed_job_id != job_id:
        db.rollback()
        raise RuntimeError(
            f'migration planner job {job_id} no longer owns current session {session_id}'
        )
    db.commit()


def _clear_planner_claim(session_id, state_key, job_id):
    if not job_id:
        return
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "UPDATE migration_session SET state = state - %s "
            "WHERE id = %s AND state->>%s = %s",
            (state_key, session_id, state_key, job_id),
        )
    db.commit()


# ---------------------------------------------------------------------------
# SSRF guard for the user-supplied media-server URL. Delegates to the shared
# ``ssrf_guard.validate_outbound_url`` (allows LAN/loopback, blocks non-HTTP(S)
# schemes and link-local/cloud-metadata). A missing url is allowed and left to
# the downstream probe.
# ---------------------------------------------------------------------------


def _validate_probe_url(creds):
    url = (creds or {}).get('url')
    if not url:
        return True, None
    return validate_outbound_url(url)


# ---------------------------------------------------------------------------
# Source path sanity check - matching tiers 1 (path) and 2 (path tail) need
# absolute filesystem paths in ``score.file_path``. If the user's current
# provider stored garbage (Navidrome without Report Real Path, Lyrion stream
# URIs, etc.), we can re-probe the current provider to get real paths and
# apply them to ``old_rows`` before matching.
# ---------------------------------------------------------------------------

_SOURCE_PATH_SAMPLE_SIZE = 100


def _sample_score_file_paths(limit=_SOURCE_PATH_SAMPLE_SIZE):
    from tasks.mediaserver import registry

    db = get_db()
    default = registry.get_default_server(db)
    default_id = default['server_id'] if default else None
    if default_id is None:
        return []
    with db.cursor() as cur:
        cur.execute(
            "SELECT file_path FROM track_server_map "
            "WHERE server_id = %s AND file_path IS NOT NULL LIMIT %s",
            (default_id, limit),
        )
        rows = cur.fetchall() or []
    return [r[0] for r in rows]


def _detect_source_path_format():
    samples = _sample_score_file_paths()
    tracks = [{'path': p} for p in samples]
    return _detect_path_format(tracks)


def _current_provider_creds():
    import config as cfg

    t = (getattr(cfg, 'MEDIASERVER_TYPE', '') or '').lower()
    fields = cfg.MEDIASERVER_FIELDS_BY_TYPE.get(t)
    if not fields:
        return None, {}
    creds = {}
    for field in fields:
        key = cfg.MEDIASERVER_CRED_KEY_BY_FIELD.get(field)
        if key:
            creds[key] = getattr(cfg, field, '')
    return t, creds


def _overrides_by_catalogue_id(by_provider_id):
    if not by_provider_id:
        return {}
    from tasks.mediaserver import registry

    canonical_of = registry.canonical_input_ids(list(by_provider_id.keys()))
    overrides = {}
    for provider_id in sorted(by_provider_id):
        catalogue_id = canonical_of.get(provider_id, provider_id)
        if catalogue_id not in overrides:
            overrides[catalogue_id] = by_provider_id[provider_id]
    return overrides


def _apply_source_path_overrides(old_rows, overrides):
    if not overrides:
        return old_rows
    for r in old_rows:
        real = overrides.get(r.get('item_id'))
        if real:
            r['file_path'] = real
    return old_rows


# ---------------------------------------------------------------------------
# Routes - wizard page
# ---------------------------------------------------------------------------


@migration_bp.route('/provider-migration')
def provider_migration_page():
    """
    Provider migration wizard page.
    ---
    tags:
      - Provider Migration
    summary: HTML wizard for migrating analysis state between media-server providers (Jellyfin/Emby/Navidrome/Lyrion).
    description: Resumes any in-flight session so a page refresh lands on the right step.
    responses:
      200:
        description: Wizard HTML rendered with `active_session_id` if a non-terminal session exists.
    """
    # Look up an in-flight migration so a page refresh can resume the wizard
    # at the right step instead of creating a brand new session.
    active_session_id = None
    try:
        db = get_db()
        with db.cursor() as cur:
            cur.execute(
                "SELECT id FROM migration_session "
                "WHERE status NOT IN ('completed', 'failed') "
                "ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
        if row:
            active_session_id = row[0]
    except Exception as e:
        logger.warning(
            "provider_migration_page: failed to look up active session: %s",
            e,
            exc_info=True,
        )
        active_session_id = None

    return render_template(
        'provider_migration.html',
        title='Provider Migration',
        active='provider_migration',
        active_session_id=active_session_id,
    )


# ---------------------------------------------------------------------------
# Routes - session CRUD
# ---------------------------------------------------------------------------


@migration_bp.route('/api/migration/session/start', methods=['POST'])
def session_start():
    """
    Start a new migration session.
    ---
    tags:
      - Provider Migration
    summary: Create a `migration_session` row and prune any already-terminal sessions.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [target_type, target_creds]
            properties:
              target_type:
                type: string
                enum: [jellyfin, emby, navidrome, lyrion, plex, ampache]
              target_creds:
                type: object
                additionalProperties: true
    responses:
      200:
        description: Session id returned.
        content:
          application/json:
            schema:
              type: object
              properties:
                session_id:
                  type: integer
      400:
        description: Unsupported target_type.
      409:
        description: A migration is queued or executing.
    """
    payload = request.get_json(silent=True) or {}
    target_type = (payload.get('target_type') or '').lower()
    target_creds = payload.get('target_creds') or {}

    if target_type not in _SUPPORTED_TARGETS:
        return jsonify({'error': f'target_type must be one of {sorted(_SUPPORTED_TARGETS)}'}), 400

    ok, reason = _validate_probe_url(target_creds)
    if not ok:
        return jsonify({'error': f'target_creds url is not allowed: {reason}'}), 400

    import config

    source_type = config.MEDIASERVER_TYPE or ''

    db = get_db()
    with db.cursor() as guard:
        # Refusing beats the old behaviour of creating a session anyway: that one
        # could not be used while a migration held the provider, and every repeated
        # click added another row plus its staging metadata.
        # Acquire the shared migration lock BEFORE reconciling reservations.  A
        # retried ambiguous planner may be re-enqueueing the same id under that
        # lock; inspecting/clearing it first could otherwise erase the fresh claim
        # immediately after its transaction commits.
        if not _no_migration_executing(guard) or _migration_job_in_flight(guard):
            return jsonify(
                {
                    'error': 'A migration job is queued or running. Wait for it to '
                             'finish before starting a new one.'
                }
            ), 409
    with db.cursor() as cur:
        # Starting a session discards every idle/abandoned session, and ON DELETE
        # CASCADE takes its potentially huge migration_target_meta rows with it.
        # Completed rows are different: they are compact retry tombstones.  Delete
        # those only after the queue conclusively has no live row for the
        # persisted exec id.  This is bounded by live retry jobs rather than an
        # arbitrary "last N" that could erase the marker a delayed self-restart
        # retry needs.
        prunable_completed = _completed_sessions_safe_to_prune(cur)
        if prunable_completed:
            cur.execute(
                "DELETE FROM migration_session WHERE status <> 'completed' "
                "OR id = ANY(%s)",
                (prunable_completed,),
            )
        else:
            cur.execute(
                "DELETE FROM migration_session WHERE status <> 'completed'"
            )
        cur.execute(
            "INSERT INTO migration_session "
            "(source_type, target_type, target_creds, state, status) "
            "VALUES (%s, %s, %s, %s, 'in_progress') RETURNING id",
            (source_type, target_type, json.dumps(target_creds), json.dumps({})),
        )
        row = cur.fetchone()
    db.commit()
    return jsonify({'session_id': row[0]})


def _source_provider_id_map(canonical_ids):
    # Map source-catalogue canonical item_ids to the default (source) server's
    # provider ids so a migration response never exposes an internal fp_ id.
    # Unmapped ids are omitted (fail closed); a registry error yields an empty map.
    ids = [str(i) for i in canonical_ids if i]
    if not ids:
        return {}
    from tasks.mediaserver import registry
    try:
        return registry.translate_ids(ids, None)
    except Exception:
        logger.exception("Migration source id translation failed")
        return {}


def _translate_state_source_ids(state):
    # Rewrite the canonical old_id keys in a session state's manual_matches /
    # manual_unmatches to the source server's provider ids for the API response.
    manual_matches = state.get('manual_matches')
    manual_unmatches = state.get('manual_unmatches')
    ids = []
    if isinstance(manual_matches, dict):
        ids += list(manual_matches.keys())
    if isinstance(manual_unmatches, list):
        ids += manual_unmatches
    mapping = _source_provider_id_map(ids)
    if isinstance(manual_matches, dict):
        state['manual_matches'] = {
            mapping[k]: v for k, v in manual_matches.items() if k in mapping
        }
    if isinstance(manual_unmatches, list):
        state['manual_unmatches'] = [mapping[i] for i in manual_unmatches if i in mapping]


@migration_bp.route('/api/migration/session/<int:session_id>', methods=['GET'])
def session_get(session_id):
    """
    Inspect a migration session.
    ---
    tags:
      - Provider Migration
    summary: Return current status and JSON state for a session.
    parameters:
      - name: session_id
        in: path
        required: true
        schema: { type: integer }
    responses:
      200:
        description: Session summary.
        content:
          application/json:
            schema:
              type: object
              properties:
                id:
                  type: integer
                source_type:
                  type: string
                target_type:
                  type: string
                status:
                  type: string
                  enum: [in_progress, dry_run_ready, completed, failed]
                state:
                  type: object
      404:
        description: Session not found.
    """
    db = get_db()
    with db.cursor() as cur:
        # Strip the two heavy keys the wizard UI never reads (the per-row
        # auto-match map and the source-path override map). Server-side ``#-``
        # keeps a 100k-entry, tens-of-MB blob from being shipped to the browser
        # on every step-4 render.
        # ``post_migration.orphans`` holds raw canonical ids, which may be internal
        # fp_ ids, plus the on-disk paths. Those must never reach any API response,
        # and the wizard never renders them: the CSV endpoint is the only consumer
        # and it maps them to provider ids itself. The key stripped here used to be
        # ``orphan_item_ids``, which is written NOWHERE, so the strip was a no-op
        # and the whole orphan list went out over the API verbatim.
        cur.execute(
            "SELECT id, source_type, target_type, status, "
            "(state #- '{dry_run,matches}' #- '{source_path_overrides}' "
            "#- '{post_migration,orphans}') "
            "FROM migration_session WHERE id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    if not row:
        return jsonify({'error': 'session not found'}), 404
    _id, source_type, target_type, status, state = row
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except Exception:
            state = {}
    if isinstance(state, dict):
        _translate_state_source_ids(state)
    return jsonify(
        {
            'id': _id,
            'source_type': source_type,
            'target_type': target_type,
            'status': status,
            'state': state,
        }
    )


# ---------------------------------------------------------------------------
# Routes - probe (delegates to tasks.provider_probe, passes creds explicitly)
# ---------------------------------------------------------------------------


@migration_bp.route('/api/migration/session/<int:session_id>', methods=['DELETE'])
def session_discard(session_id):
    """
    Discard an in-flight migration session.
    ---
    tags:
      - Provider Migration
    summary: Delete a non-terminal session row (used by the wizard's Discard button).
    description: |
      Refuses to touch sessions in `completed` or `failed` status - those are
      pruned automatically on the next `session_start`. A live dry-run or
      source-refresh job is cancelled first (the same global cancel Analysis &
      Clustering's Cancel button uses) rather than blocking the discard,
      since it is a pure fetch with no external side effects. A live execute
      stays a hard block - it writes to the target server and is never
      auto-cancelled here.
    parameters:
      - name: session_id
        in: path
        required: true
        schema: { type: integer }
    responses:
      200:
        description: Session deleted.
      400:
        description: Session is already in a terminal state.
      404:
        description: Session not found.
      409:
        description: A migration is currently executing against this session.
      503:
        description: Could not verify whether a migration planner job is still live.
    """
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT status FROM migration_session WHERE id = %s",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return jsonify({'error': 'session not found'}), 404
        if row[0] in ('completed', 'failed'):
            return jsonify({'error': 'cannot discard a finished session'}), 400
        # A session stays 'dry_run_ready' throughout the execute, so the status
        # check above cannot tell a running migration from an idle wizard. Deleting
        # it mid-execute strands a repointed catalogue with no completion marker,
        # so an active execute - either as the main task, or per this backup
        # task_status probe on exec_task_id - stays a hard block and is never
        # auto-cancelled here.
        exec_live = _migration_job_in_flight(cur, keys=('exec_task_id',))
        if not _no_migration_executing(cur) or exec_live:
            return jsonify(
                {
                    'error': 'A migration job is currently running against this '
                             'session. Wait for it to finish.'
                }
            ), 409
        # A live dry-run/source-refresh, unlike execute above, is a pure fetch
        # with no external side effects - so cancel it (same global cancel the
        # Analysis & Clustering Cancel button uses) instead of leaving the
        # wizard stuck behind a job that could otherwise only be stopped by
        # killing the worker outright.
        try:
            planner_job_id = _live_planner_job_id(cur)
        except Exception:
            logger.exception(
                "Could not check for a live migration planner job; refusing to "
                "discard session %s", session_id,
            )
            return jsonify(
                {
                    'error': 'Could not verify the migration planner job. Try '
                             'again when the database is available.',
                }
            ), 503

    if planner_job_id:
        # cancel_job_and_children_recursive commits internally, which ends the
        # transaction that held the advisory lock above and releases it. That
        # reopens exactly the window the lock exists to close: a fresh
        # /api/migration/execute could slip in here, reserve exec_task_id and
        # enqueue, and then get its session deleted out from under it by the
        # DELETE below. The re-check right before the DELETE, in the SAME
        # transaction as the DELETE, is what closes that window again - it
        # re-acquires the lock and holds it continuously through the commit,
        # so anything that raced in during the gap is caught here instead of
        # silently stranding a migration.
        cancel_job_and_children_recursive(
            planner_job_id,
            reason=f'Migration session {session_id} discarded by user.',
        )

    with db.cursor() as cur:
        if not _no_migration_executing(cur) or _migration_job_in_flight(cur):
            return jsonify(
                {
                    'error': 'A migration job started while the previous one was '
                             'being discarded. Try again.'
                }
            ), 409
        # The status was read above, but execute could have committed since. The
        # predicate makes the check and the delete one act, so a migration that
        # finished in that window keeps the completed marker its retry needs.
        cur.execute(
            "DELETE FROM migration_session WHERE id = %s "
            "AND status NOT IN ('completed', 'failed')",
            (session_id,),
        )
    db.commit()
    return jsonify({'ok': True})


@migration_bp.route('/api/migration/probe/test', methods=['POST'])
def probe_test():
    """
    Test a target-provider connection.
    ---
    tags:
      - Provider Migration
    summary: Probe a media-server provider with given credentials and report path quality.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [type, creds]
            properties:
              type:
                type: string
                enum: [jellyfin, emby, navidrome, lyrion, plex, ampache]
              creds:
                type: object
                additionalProperties: true
    responses:
      200:
        description: Probe result (always 200; check `ok` for success).
        content:
          application/json:
            schema:
              type: object
              properties:
                ok:
                  type: boolean
                error:
                  type: string
                path_format:
                  type: string
                  enum: [absolute, relative, virtual, none]
                sample_count:
                  type: integer
                warnings:
                  type: array
                  items:
                    type: string
    """
    payload = request.get_json(silent=True) or {}
    t = (payload.get('type') or '').lower()
    creds = payload.get('creds') or {}
    ok, reason = _validate_probe_url(creds)
    if not ok:
        return jsonify(
            {'ok': False, 'error': reason, 'path_format': 'none', 'sample_count': 0, 'warnings': []}
        ), 200
    try:
        result = provider_probe.test_connection(t, creds)
    except NotImplementedError:
        logger.warning("test_connection not supported for provider type %s", t)
        return jsonify(
            {
                'ok': False,
                'error': 'Connection testing is not supported for this provider.',
                'path_format': 'none',
                'sample_count': 0,
                'warnings': [],
            }
        ), 200
    except Exception:
        logger.warning("test_connection failed for provider type %s", t, exc_info=True)
        return jsonify(
            {
                'ok': False,
                'error': 'Connection test failed. Check the container logs for details.',
                'path_format': 'none',
                'sample_count': 0,
                'warnings': [],
            }
        ), 200
    return jsonify(result)


@migration_bp.route('/api/migration/libraries', methods=['POST'])
def libraries_list():
    """
    List target-provider music libraries.
    ---
    tags:
      - Provider Migration
    summary: Step 2 - return the target provider's libraries plus the user's prior checkbox selection.
    description: Uses session-stored credentials, never `config`, so the live provider keeps working.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id]
            properties:
              session_id:
                type: integer
    responses:
      200:
        description: Library list (always 200; check `error` for failures).
        content:
          application/json:
            schema:
              type: object
              properties:
                libraries:
                  type: array
                  items:
                    type: object
                unsupported:
                  type: boolean
                selected_libraries:
                  type: array
                  items:
                    type: string
                error:
                  type: string
      400:
        description: Missing session_id.
      404:
        description: Session not found.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    if session_id is None:
        return jsonify({'error': 'session_id is required'}), 400

    session = _fetch_session_creds(session_id)
    if session is None:
        return jsonify({'error': 'session not found'}), 404
    target_type, creds = session
    state = _load_state(session_id) or {}
    selected = state.get('selected_libraries')
    try:
        result = provider_probe.list_libraries(target_type, creds)
    except Exception as e:
        logger.warning("libraries_list failed for session %s: %s", session_id, e, exc_info=True)
        return jsonify(
            {
                'libraries': [],
                'unsupported': False,
                'selected_libraries': selected,
                'error': 'Failed to list libraries. Check the container logs for details.',
            }
        ), 200
    return jsonify(
        {
            'libraries': result.get('libraries', []),
            'unsupported': bool(result.get('unsupported', False)),
            'selected_libraries': selected,
        }
    ), 200


@migration_bp.route('/api/migration/libraries/select', methods=['POST'])
def libraries_select():
    """
    Persist library selection into session state.
    ---
    tags:
      - Provider Migration
    summary: Step 2 - save the user's library checkbox selection (null = no filter, [] = normalized to null).
    description: |
      Library names cannot contain commas because `MUSIC_LIBRARIES` is stored
      as a comma-separated string and split at scan time.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id]
            properties:
              session_id:
                type: integer
              libraries:
                type: array
                nullable: true
                items:
                  type: string
    responses:
      200:
        description: Selection saved.
        content:
          application/json:
            schema:
              type: object
              properties:
                ok:
                  type: boolean
                selected_libraries:
                  type: array
                  nullable: true
                  items:
                    type: string
      400:
        description: Missing session_id, libraries not a list, or comma-containing library name.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    if session_id is None:
        return jsonify({'error': 'session_id is required'}), 400

    libraries = payload.get('libraries')
    if libraries is not None and not isinstance(libraries, list):
        return jsonify({'error': 'libraries must be a list of names or null'}), 400

    if isinstance(libraries, list):
        cleaned = [str(name).strip() for name in libraries if str(name).strip()]
        # MUSIC_LIBRARIES is stored as a comma-separated string and split on
        # ',' at scan time, so a name containing a comma would silently
        # corrupt the round-trip into multiple bogus fragments.
        if any(',' in name for name in cleaned):
            return jsonify({'error': 'Library names cannot contain commas.'}), 400
        selected = cleaned or None
    else:
        selected = None

    _update_state(session_id, selected_libraries=selected)
    return jsonify({'ok': True, 'selected_libraries': selected}), 200


@migration_bp.route('/api/migration/search-albums', methods=['POST'])
def search_albums():
    """
    Search target-provider albums.
    ---
    tags:
      - Provider Migration
    summary: Free-text album search against the target provider (used by step 4 manual matching).
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id]
            properties:
              session_id:
                type: integer
              query:
                type: string
    responses:
      200:
        description: Album candidates.
        content:
          application/json:
            schema:
              type: object
              properties:
                albums:
                  type: array
                  items:
                    type: object
      404:
        description: Session not found.
      500:
        description: Provider error during search.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    query = payload.get('query') or ''

    session = _fetch_session_creds(session_id)
    if session is None:
        return jsonify({'error': 'session not found'}), 404
    target_type, creds = session
    try:
        albums = provider_probe.search_albums(target_type, creds, query)
    except Exception:
        logger.warning("search_albums failed for session %s", session_id, exc_info=True)
        return jsonify({'error': 'Album search failed. Check the container logs for details.'}), 500
    return jsonify({'albums': albums})


# ---------------------------------------------------------------------------
# Routes - dry run, manual match, finalize
# ---------------------------------------------------------------------------


@migration_bp.route('/api/migration/source-paths/refresh', methods=['POST'])
def source_paths_refresh():
    """
    Refresh source-provider real paths.
    ---
    tags:
      - Provider Migration
    summary: Re-probe the currently active provider to build a {item_id -> real_path} override map.
    description: |
      Called when `score.file_path` is unusable (e.g. Navidrome analyzed
      without "Report Real Path"). After refresh, the dry-run can use the
      fresh paths for matcher tiers 1 and 2 without rebuilding analysis
      from scratch.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id]
            properties:
              session_id:
                type: integer
    responses:
      200:
        description: Refresh result with override count and any warnings.
        content:
          application/json:
            schema:
              type: object
              properties:
                ok:
                  type: boolean
                source_type:
                  type: string
                path_format:
                  type: string
                overrides_count:
                  type: integer
                warnings:
                  type: array
                  items:
                    type: string
      400:
        description: Missing session_id, or current provider doesn't support path refresh.
      500:
        description: Provider probe failed.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    if session_id is None:
        return jsonify({'error': 'session_id is required'}), 400

    # Cheap support check (reads config) stays in the request; the full
    # source-catalog fetch is offloaded to a queue worker like the dry-run.
    source_type, _ = _current_provider_creds()
    if not source_type:
        return jsonify(
            {
                'ok': False,
                'error': 'The current provider does not support path refresh.',
            }
        ), 400

    try:
        job_id, reused = _claim_and_enqueue_planner(
            session_id,
            'source_refresh_task_id',
            'tasks.provider_migration_tasks.source_refresh_provider_migration',
            (session_id,),
            claim_status=None,
        )
    except _PlanningClaimError as exc:
        try:
            get_db().rollback()
        except Exception:
            pass
        return jsonify({'error': exc.user_message}), exc.status_code
    except Exception:
        logger.exception("Could not reserve or enqueue the source-path refresh")
        try:
            get_db().rollback()
        except Exception:
            pass
        return jsonify({'error': 'Could not enqueue the refresh. Check the logs.'}), 500
    return jsonify({'task_id': job_id, 'async': True, 'reused': reused})


def run_source_refresh_core(session_id):
    # Revalidate in the worker.  The route's claim is authoritative for mutual
    # exclusion, but a queued job must still fail closed if its session was
    # deleted or superseded before this process started.
    if _fetch_session_creds(session_id, require_plannable=True) is None:
        raise RuntimeError(f'migration session {session_id} is no longer plannable')
    source_type, creds = _current_provider_creds()
    if not source_type:
        raise RuntimeError('The current provider does not support path refresh.')

    tracks = provider_probe.fetch_all_tracks(source_type, creds)

    path_format = _detect_path_format(tracks)
    overrides = _overrides_by_catalogue_id(
        {t['id']: t['path'] for t in tracks if t.get('id') and t.get('path')}
    )

    warnings = []
    if path_format != 'absolute':
        warnings.append(
            f'{source_type} is still not returning absolute paths. '
            'Double-check that "Report Real Path" (Navidrome) or the '
            'equivalent setting is enabled, then refresh again. You can '
            'also proceed with metadata-only matching.'
        )

    # Persist WITHOUT flagging in_progress. A refresh only fills in overrides, so
    # demoting a finalized 'dry_run_ready' session threw the finalization away and
    # Execute then refused the migration. Only finalize_dry_run writes that status
    # back, so the user had to redo the whole dry run.
    _patch_state_keys(session_id, source_path_overrides=overrides)
    return {
        'ok': True,
        'source_type': source_type,
        'path_format': path_format,
        'overrides_count': len(overrides),
        'warnings': warnings,
    }


@migration_bp.route('/api/migration/dry-run', methods=['POST'])
def dry_run():
    """
    Run the migration matcher (dry-run).
    ---
    tags:
      - Provider Migration
    summary: Step 3 - match score rows against the target provider's tracks and persist the result.
    description: |
      Source `score.file_path` values are sanity-checked first. If they don't
      look like absolute filesystem paths, the endpoint returns **409** with
      `needs_source_refresh=true` so the UI can prompt the user to enable
      "Report Real Path" and call `/source-paths/refresh`. Pass
      `bypass_source_check=true` to skip the gate and use metadata-only
      matching.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id]
            properties:
              session_id:
                type: integer
              bypass_source_check:
                type: boolean
                default: false
              allow_title_artist_only:
                type: boolean
                default: false
                description: Allow the matcher to fall back to title+artist when album metadata differs.
    responses:
      200:
        description: Dry-run summary.
        content:
          application/json:
            schema:
              type: object
              properties:
                tier_counts:
                  type: object
                matched:
                  type: integer
                unmatched:
                  type: integer
                unmatched_albums_count:
                  type: integer
      404:
        description: Session not found.
      409:
        description: Source paths look unusable; refresh required.
      500:
        description: Target provider error.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    bypass_source_check = bool(payload.get('bypass_source_check'))
    allow_title_artist_only = bool(payload.get('allow_title_artist_only'))

    session = _fetch_session_creds(session_id, require_plannable=True)
    if session is None:
        return jsonify({'error': 'session not found'}), 404

    # Gate on source path quality (cheap - samples 100 rows). Stays in the
    # request so the UI can prompt for a refresh. Skip if the user has already
    # refreshed (overrides present) or opted into metadata-only matching.
    state = _load_state(session_id) or {}
    source_overrides = state.get('source_path_overrides') or {}
    if not source_overrides and not bypass_source_check:
        source_format = _detect_source_path_format()
        if source_format != 'absolute':
            source_type, _ = _current_provider_creds()
            return jsonify(
                {
                    'needs_source_refresh': True,
                    'current_source_type': source_type,
                    'path_format': source_format,
                    'hint': (
                        'Your score.file_path values are not absolute filesystem '
                        'paths. Automatic path-based matching will fall back to '
                        'metadata only. Refresh source paths, or proceed with '
                        'metadata-only matching.'
                    ),
                }
            ), 409

    # The heavy work (fetch the whole target catalog + match every score row +
    # persist) can take minutes on 100k+ libraries - far past the gunicorn
    # request timeout - so it runs in a queue worker; the UI polls the status.
    try:
        job_id, reused = _claim_and_enqueue_planner(
            session_id,
            'dry_run_task_id',
            'tasks.provider_migration_tasks.dry_run_provider_migration',
            (session_id, allow_title_artist_only),
        )
    except _PlanningClaimError as exc:
        try:
            get_db().rollback()
        except Exception:
            pass
        return jsonify({'error': exc.user_message}), exc.status_code
    except Exception:
        logger.exception("Could not reserve or enqueue the dry run")
        try:
            get_db().rollback()
        except Exception:
            pass
        return jsonify({'error': 'Could not enqueue the dry run. Check the logs.'}), 500
    return jsonify({'task_id': job_id, 'async': True, 'reused': reused})


def run_dry_run_core(session_id, allow_title_artist_only=False):
    session = _fetch_session_creds(session_id, require_plannable=True)
    if session is None:
        raise RuntimeError(f'migration session {session_id} not found')
    target_type, creds = session

    new_tracks = provider_probe.fetch_all_tracks(target_type, creds)

    # Safety guard: a target that returns zero tracks (transient outage, wrong
    # creds, empty/mis-scoped library) would make EVERY score row an orphan,
    # and execute would then delete the entire library. Refuse instead of
    # producing an all-orphan plan. Returned (not raised) so the UI shows the
    # reason rather than a generic failure.
    if not new_tracks:
        logger.warning(
            "provider migration dry-run: target '%s' returned 0 tracks; aborting "
            "to avoid orphaning the whole library (session %s)",
            target_type,
            session_id,
        )
        return {
            'error': (
                'The new provider returned 0 tracks. Aborting so your library is '
                'not deleted as orphans. Check the connection / library selection '
                'and run automatic matching again.'
            )
        }

    state = _load_state(session_id) or {}
    source_overrides = state.get('source_path_overrides') or {}
    old_rows = _load_score_rows_as_dicts()
    _apply_source_path_overrides(old_rows, source_overrides)

    import importlib

    matcher = importlib.import_module('tasks.provider_migration_matcher')
    result = matcher.match_tracks(
        old_rows,
        new_tracks,
        allow_title_artist_only=allow_title_artist_only,
    )

    state_dry_run = {
        'matches': result['matches'],
        'tier_counts': result['tier_counts'],
        'unmatched_albums': _albums_payload(result['unmatched_by_album']),
        # Full count so the wizard can warn when the rendered list is a sample.
        'unmatched_albums_total': len(result['unmatched_by_album']),
    }
    new_meta = {
        n['id']: {
            'path': n.get('path'),
            'title': n.get('title'),
            'artist': n.get('artist'),
            'album': n.get('album'),
            'album_artist': n.get('album_artist'),
            'year': n.get('year'),
        }
        for n in new_tracks
        if n.get('id')
    }
    _store_target_meta(session_id, new_meta)
    _update_state(
        session_id,
        dry_run=state_dry_run,
        manual_matches={},
        manual_unmatches=[],
        final_counts=None,
    )

    return {
        'tier_counts': result['tier_counts'],
        'matched': len(result['matches']),
        'unmatched': len(result['unmatched']),
        'unmatched_albums_count': len(result['unmatched_by_album']),
    }


@migration_bp.route('/api/migration/match-album', methods=['POST'])
def match_album():
    """
    Manually match an album.
    ---
    tags:
      - Provider Migration
    summary: Step 4 - user picked a target album; auto-match its tracks by title (or rematch existing auto-matches).
    description: |
      With `rematch=true`, the endpoint reprocesses rows that were already
      auto-matched for this album: any auto-match for the album is discarded
      and replaced by the new target. Rows that don't match in the new target
      become explicit orphans via `manual_unmatches`.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id, old_album_key, new_album_id]
            properties:
              session_id:
                type: integer
              old_album_key:
                type: array
                items:
                  type: string
                description: "[album_artist, album]"
              new_album_id:
                type: string
              rematch:
                type: boolean
                default: false
    responses:
      200:
        description: Match result for the album.
        content:
          application/json:
            schema:
              type: object
              properties:
                matched:
                  type: integer
                unmatched:
                  type: integer
                unmatched_item_ids:
                  type: array
                  items:
                    type: string
      404:
        description: Session not found.
      500:
        description: Target provider error.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    old_album_key = payload.get('old_album_key')  # [album_artist, album]
    new_album_id = payload.get('new_album_id')
    rematch = bool(payload.get('rematch'))

    session = _fetch_session_creds(session_id)
    if session is None:
        return jsonify({'error': 'session not found'}), 404
    target_type, creds = session

    try:
        new_tracks = provider_probe.get_album_tracks(target_type, creds, new_album_id)
    except Exception:
        logger.warning("get_album_tracks failed for session %s", session_id, exc_info=True)
        return jsonify(
            {'error': 'Failed to fetch album tracks. Check the container logs for details.'}
        ), 500

    import importlib

    matcher = importlib.import_module('tasks.provider_migration_matcher')

    old_album_tuple = tuple(old_album_key) if isinstance(old_album_key, list) else old_album_key
    if rematch:
        old_rows = _load_rows_for_album(old_album_tuple)
    else:
        old_rows = _load_unmatched_for_album(session_id, old_album_tuple)

    # Match within the album: exact title, then normalized title
    by_title = {}
    by_norm_title = {}
    for n in new_tracks:
        t = (n.get('title') or '').lower()
        if t and t not in by_title:
            by_title[t] = n['id']
        nt = matcher.normalize_meta(n.get('title'))
        if nt and nt not in by_norm_title:
            by_norm_title[nt] = n['id']

    newly_matched = {}
    still_unmatched = []
    for old in old_rows:
        title_l = (old.get('title') or '').lower()
        nt = matcher.normalize_meta(old.get('title'))
        if title_l in by_title:
            newly_matched[old['item_id']] = by_title[title_l]
        elif nt and nt in by_norm_title:
            newly_matched[old['item_id']] = by_norm_title[nt]
        else:
            still_unmatched.append(old['item_id'])

    if rematch:
        _rematch_album_rows(session_id, newly_matched, still_unmatched)
    else:
        _merge_manual_matches(session_id, newly_matched)
    # Expose the source server's provider ids, never the internal fp_ id. The count
    # tracks the id list exactly (both drop any id with no provider mapping) so the
    # wizard's counter and its rendered rows never disagree.
    unmatched_mapping = _source_provider_id_map(still_unmatched)
    unmatched_item_ids = [
        unmatched_mapping[i] for i in still_unmatched if i in unmatched_mapping
    ]
    return jsonify(
        {
            'matched': len(newly_matched),
            'unmatched': len(unmatched_item_ids),
            'unmatched_item_ids': unmatched_item_ids,
        }
    )


@migration_bp.route('/api/migration/skip-album', methods=['POST'])
def skip_album():
    """
    Skip an album (mark its rows as orphans).
    ---
    tags:
      - Provider Migration
    summary: Step 4 - orphan an album so its score rows will be deleted by execute.
    description: |
      First-time skips (unmatched albums) just need a ledger note. Rematch
      skips (`rematch=true`) push every row in the album into
      `manual_unmatches` so finalize overrides the existing auto-match.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id, old_album_key]
            properties:
              session_id:
                type: integer
              old_album_key:
                type: array
                items:
                  type: string
              rematch:
                type: boolean
                default: false
    responses:
      200:
        description: Album marked as skipped.
        content:
          application/json:
            schema:
              type: object
              properties:
                ok:
                  type: boolean
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    old_album_key = payload.get('old_album_key')
    rematch = bool(payload.get('rematch'))

    if rematch:
        album_tuple = tuple(old_album_key) if isinstance(old_album_key, list) else old_album_key
        old_rows = _load_rows_for_album(album_tuple)
        all_ids = [r['item_id'] for r in old_rows]
        _rematch_album_rows(session_id, newly_matched={}, newly_unmatched=all_ids)

    _mark_album_skipped(session_id, old_album_key)
    return jsonify({'ok': True})


@migration_bp.route('/api/migration/finalize-dry-run', methods=['POST'])
def finalize_dry_run():
    """
    Finalize the dry-run.
    ---
    tags:
      - Provider Migration
    summary: Compute final counts (with collision dedup) and flip status to `dry_run_ready`.
    description: |
      Runs the same one-to-one dedup logic as `execute` so the user sees any
      collisions (multiple source rows fighting for the same target track)
      before typing the confirmation phrase. Without this, execute would trip
      `UNIQUE(new_id)` on the temp rewrite table and roll back with an opaque
      Postgres error.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id]
            properties:
              session_id:
                type: integer
    responses:
      200:
        description: Final counts including collision details.
        content:
          application/json:
            schema:
              type: object
              properties:
                matched:
                  type: integer
                orphans:
                  type: integer
                collisions:
                  type: integer
                collision_details:
                  type: array
                  items:
                    type: object
                tier_counts:
                  type: object
      404:
        description: Session not found.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')

    state = _load_state(session_id)
    if state is None:
        return jsonify({'error': 'session not found'}), 404

    dry = state.get('dry_run') or {}

    import importlib

    mig_tasks = importlib.import_module('tasks.provider_migration_tasks')
    merged, dropped = mig_tasks.build_mapping(state)

    total_score = _count_score_rows()
    matched = len(merged)
    collisions = len(dropped)
    # Rows with no match at all = total - (rows that were matched) - (rows
    # dropped by collision dedup). Both collision losers and no-match rows
    # get deleted on execute; showing them separately lets the user decide
    # whether to go back to step 4 and fix the duplicates.
    orphans = max(0, total_score - matched - collisions)

    import config

    collision_details_total = collisions
    # Build human-readable collision details so the UI can tell the user
    # exactly which albums to rematch. Only the capped subset is rendered, so
    # fetch only those score rows / target metadata (by id) rather than the
    # whole catalog.
    collision_details = []
    if dropped:
        dropped_for_details = dropped[: config.MIGRATION_MAX_COLLISION_DETAILS]
        needed_old_ids = set()
        needed_new_ids = set()
        for loser_old_id, new_id, winner_old_id in dropped_for_details:
            needed_old_ids.add(loser_old_id)
            needed_old_ids.add(winner_old_id)
            needed_new_ids.add(str(new_id))
        old_by_id = _load_score_rows_by_ids(needed_old_ids)
        meta_by_id = _load_target_meta(session_id, needed_new_ids)
        for loser_old_id, new_id, winner_old_id in dropped_for_details:
            loser = old_by_id.get(loser_old_id) or {}
            winner = old_by_id.get(winner_old_id) or {}
            tgt = meta_by_id.get(str(new_id)) or {}
            collision_details.append(
                {
                    'loser_title': loser.get('title') or '',
                    'loser_artist': loser.get('album_artist') or loser.get('author') or '',
                    'loser_album': loser.get('album') or '',
                    'loser_path': loser.get('file_path') or '',
                    'winner_title': winner.get('title') or '',
                    'winner_artist': winner.get('album_artist') or winner.get('author') or '',
                    'winner_album': winner.get('album') or '',
                    'winner_path': winner.get('file_path') or '',
                    'target_title': tgt.get('title') or '',
                    'target_artist': tgt.get('artist') or '',
                    'target_album': tgt.get('album') or '',
                    'target_path': tgt.get('path') or '',
                }
            )

    final_counts = {
        'matched': matched,
        'orphans': orphans,
        'collisions': collisions,
        'collision_details': collision_details,
        'collision_details_total': collision_details_total,
        'tier_counts': dry.get('tier_counts') or {},
    }

    db = get_db()
    with db.cursor() as cur:
        # Never resurrect an applied migration. A finalize landing after execute
        # committed would flip 'completed' back to 'dry_run_ready', and the retry
        # branch keys off that status: the job would re-run against a state whose
        # mapping is gone, and an empty mapping unbinds the whole default server.
        cur.execute(
            "UPDATE migration_session SET "
            "  state = jsonb_set(state, '{final_counts}', %s::jsonb, true), "
            "  status = 'dry_run_ready' "
            "WHERE id = %s AND status IS DISTINCT FROM 'completed'",
            (json.dumps(_sanitize_json_value(final_counts), ensure_ascii=False), session_id),
        )
    db.commit()
    return jsonify(final_counts)


# ---------------------------------------------------------------------------
# Routes - execute gate + status
# ---------------------------------------------------------------------------


def _execute_locked(db, session_id, confirmation_text):
    with db.cursor() as cur:
        # Lock order is main-task -> migration for every admission endpoint.
        # TRY avoids tying up a web worker behind the long migration transaction.
        cur.execute("SELECT pg_try_advisory_xact_lock(%s)", (_ADVISORY_LOCK_KEY,))
        if not cur.fetchone()[0]:
            return jsonify(
                {'error': 'A migration is already running. Wait for it to finish.'}
            ), 409
        if _restart_handshake_pending(cur):
            return jsonify(
                {
                    'error': 'The committed provider swap is still waiting for '
                    'worker restart acknowledgement.'
                }
            ), 409
        cur.execute(
            "SELECT target_type, status, "
            "(id = (SELECT MAX(id) FROM migration_session)) "
            "FROM migration_session WHERE id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    if not row:
        return jsonify({'error': 'session not found'}), 404
    target_type, status, is_current_session = row[0], row[1], row[2]

    with db.cursor() as planning:
        if _migration_job_in_flight(planning, keys=_PLANNER_TASK_KEYS):
            return jsonify(
                {
                    'error': 'A dry run is still building the plan. Wait for it to '
                    'finish, then confirm the numbers again.'
                }
            ), 409
    if not is_current_session:
        return jsonify(
            {
                'error': 'This is not the current migration session. Start again from '
                'the wizard so the plan matches what you reviewed.'
            }
        ), 409

    expected = f"I want to migrate to {target_type} and unbind unmatched tracks"
    if confirmation_text != expected:
        return jsonify(
            {'error': f'Confirmation text does not match. Expected exactly: "{expected}"'}
        ), 400
    if status != 'dry_run_ready':
        return jsonify(
            {
                'error': f'Dry run must be finalized first. Session status is "{status}", '
                f'expected "dry_run_ready".'
            }
        ), 400

    active = get_active_main_task(exclude_task_types=NON_BLOCKING_TASK_TYPES)
    if active:
        return jsonify(
            {
                'error': 'Another task is running. Wait for it to finish before migrating.',
                'task_id': active['task_id'],
                'task_type': active['task_type'],
                'status': active['status'],
            }
        ), 409

    job_id = str(uuid.uuid4())
    save_task_status(
        job_id,
        MIGRATION_TASK_TYPE,
        TASK_STATUS_PENDING,
        details={'message': 'Provider migration enqueued.'},
        raise_on_error=True,
    )
    try:
        _patch_state_keys(session_id, exec_task_id=job_id)
    except Exception:
        logger.exception(
            "Could not persist provider-migration execute reservation %s", job_id
        )
        save_task_status(
            job_id,
            MIGRATION_TASK_TYPE,
            TASK_STATUS_FAILURE,
            details={'error': 'Could not persist the migration reservation.'},
        )
        return jsonify({'error': 'Could not reserve the migration task.'}), 500

    try:
        taskqueue.enqueue(
            'tasks.provider_migration_tasks.execute_provider_migration',
            args=(session_id,),
            task_id=job_id,
            task_type=MIGRATION_TASK_TYPE,
            queue=taskqueue.QUEUE_HIGH,
            details={'message': 'Provider migration queued.'},
        )
    except Exception:
        logger.exception("Could not queue the provider migration %s", job_id)
        _patch_state_keys(session_id, exec_task_id=None)
        save_task_status(
            job_id,
            MIGRATION_TASK_TYPE,
            TASK_STATUS_FAILURE,
            details={'error': 'Could not queue the migration task.'},
        )
        return jsonify({'error': 'Could not queue the migration. Check the logs.'}), 500

    return jsonify({'task_id': job_id})


@migration_bp.route('/api/migration/execute', methods=['POST'])
def execute():
    """
    Execute the migration.
    ---
    tags:
      - Provider Migration
    summary: Step 5 - gate on backup checkbox + confirmation phrase, then enqueue the execute job.
    description: |
      Requires the session to be in `dry_run_ready` status. The confirmation
      phrase must equal exactly:
      `I want to migrate to <target_type> and unbind unmatched tracks`.
      The job only repoints the default server's `track_server_map` rows at the new
      provider. The catalogue is never touched: no song, embedding or canonical id is
      deleted. Unmatched songs are simply unbound from this server.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id, backup_confirmed, confirmation_text]
            properties:
              session_id:
                type: integer
              backup_confirmed:
                type: boolean
                description: Must be true.
              confirmation_text:
                type: string
                description: Must equal the per-target confirmation phrase exactly.
    responses:
      200:
        description: Execute task enqueued.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
      400:
        description: Missing backup confirmation, wrong confirmation phrase, or session not in `dry_run_ready` state.
      404:
        description: Session not found.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    backup_confirmed = bool(payload.get('backup_confirmed'))
    confirmation_text = payload.get('confirmation_text') or ''

    if not backup_confirmed:
        return jsonify({'error': 'You must confirm the backup checkbox'}), 400

    db = get_db()
    cancel_epoch = _global_cancel_epoch(db)
    with main_task_start_lock():
        # A request that was already waiting when a global Cancel held this lock
        # must not publish an invisible post-cancel root.  Requests begun after the
        # cancellation are intentional and snapshot its new tombstone id.
        if _global_cancel_epoch(db) != cancel_epoch:
            return jsonify(
                {
                    'error': 'A global cancellation completed while this migration '
                    'was waiting. Start it again if you still want to run it.'
                }
            ), 409
        return _execute_locked(db, session_id, confirmation_text)


# Track which finished migration jobs have already triggered a Flask restart,
# so repeated status polls don't schedule the restart multiple times.
_restart_scheduled_for_tasks = set()

# The dry-run and source-refresh jobs return their payload to the worker, which
# parks it under `final_summary_details`. The execute job never gets there: it
# writes its own SUCCESS row through `_report_migration`, which merges the
# summary into the TOP LEVEL of `details`, and the worker's finalize then
# early-returns because the row is no longer RUNNING. Reading only the worker's
# key therefore reported `result: null` for every execute job, so the page could
# never tell the user their similarity / map index had been reset.
_EXECUTE_SUMMARY_KEYS = ('ok', 'matched', 'index_rebuild_needed', 'already_applied')


def _execute_summary_from_details(details):
    summary = {key: details[key] for key in _EXECUTE_SUMMARY_KEYS if key in details}
    return summary or None


@migration_bp.route('/api/migration/status/<task_id>', methods=['GET'])
def job_status(task_id):
    """
    Poll the migration execute task.
    ---
    tags:
      - Provider Migration
    summary: Return the task status; on completion, schedule a Flask restart so config reloads.
    description: |
      When the job finishes, this endpoint reloads `config` in this Flask
      process and (once per task_id) schedules a graceful restart so any
      module-level `from config import X` bindings are rebuilt against the
      new provider settings.
    parameters:
      - name: task_id
        in: path
        required: true
        schema: { type: string }
    responses:
      200:
        description: Job status payload.
        content:
          application/json:
            schema:
              type: object
              properties:
                id:
                  type: string
                status:
                  type: string
                  enum: [NEW, RUNNING, SUCCESS, FAIL, REVOKED]
                result:
                  nullable: true
                error:
                  type: string
                  nullable: true
                restart_scheduled:
                  type: boolean
      404:
        description: Job not found.
    """
    try:
        from database import get_task_info_from_db

        row = get_task_info_from_db(task_id)
        if not row:
            return jsonify({'error': 'Job not found.'}), 404
        status = row.get('status')
        details = coerce_db_details(row.get('details')) or {}
        restart_scheduled = False
        # Only the EXECUTE task changes the active provider and needs the
        # config-reload + Flask restart. The dry-run / source-refresh tasks
        # share this status endpoint but must NOT restart Flask.
        # The classification is read ONLY once the job has succeeded: it is a
        # migration_session probe, and during the execute phase this endpoint is
        # polled every couple of seconds - running the probe on every poll
        # detoasted the multi-megabyte session state for a result only the
        # terminal branch consumes.
        if status == config.TASK_STATUS_SUCCESS and _task_is_the_execute_job(task_id):
            try:
                import config as _cfg

                _cfg.refresh_config()
            except Exception as _e:
                logger.warning("post-migration config reload failed: %s", _e)
            if task_id not in _restart_scheduled_for_tasks:
                try:
                    import restart_manager

                    if restart_manager.schedule_flask_restart():
                        restart_scheduled = True
                        _restart_scheduled_for_tasks.add(task_id)
                except Exception as _e:
                    logger.warning("post-migration Flask restart scheduling failed: %s", _e)
            else:
                restart_scheduled = True
        return jsonify(
            {
                'id': task_id,
                'status': status,
                'result': (
                    details.get('final_summary_details')
                    or details.get('result')
                    or _execute_summary_from_details(details)
                ),
                'error': 'Job failed. Check the container logs for details.'
                if status == config.TASK_STATUS_FAIL
                else None,
                'restart_scheduled': restart_scheduled,
            }
        )
    except Exception:
        logger.warning("migration job status fetch failed for task %s", task_id, exc_info=True)
        return jsonify({'error': 'Job not found.'}), 404


@migration_bp.route('/api/migration/dry-run-report/<int:session_id>', methods=['GET'])
def dry_run_report(session_id):
    """
    Download the dry-run report as CSV.
    ---
    tags:
      - Provider Migration
    summary: CSV showing the planned old->new mapping for every score row (orphans have blank new-side cells).
    description: |
      Columns: old_id, old_artist, old_album, old_album_artist, old_track, old_path, new_id,
      new_artist, new_album, new_album_artist, new_track, new_path, match_source
      (`auto`/`manual`/`orphan`).

      Before the migration runs this is the full planned mapping. Once it has run the
      staging data is gone, so the report narrows to the songs actually left unbound
      (all `orphan`) - the audit that still matters afterwards.
    parameters:
      - name: session_id
        in: path
        required: true
        schema: { type: integer }
    responses:
      200:
        description: CSV attachment.
        content:
          text/csv:
            schema:
              type: string
      404:
        description: Session not found.
      410:
        description: The migration ran before orphan snapshots existed, so nothing is left to report.
    """
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT status, state FROM migration_session WHERE id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    if row is None:
        return jsonify({'error': 'session not found'}), 404

    status, state = row[0], row[1]
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except Exception:
            state = {}
    state = state or {}

    if status in ('completed', 'failed'):
        # Applying a migration discards the per-track staging rows and the mapping
        # blob the full CSV is built from, so the pre-run report cannot be rebuilt.
        # What execute DOES keep is the orphan list, which is the audit that matters
        # afterwards: which songs the target did not have and are now unbound.
        # Rendering the full report from the emptied staging data instead would
        # claim every single song was orphaned - a confident report of total
        # failure for a migration that fully succeeded.
        post_migration = state.get('post_migration') or {}
        if 'orphans' not in post_migration:
            return jsonify(
                {
                    'error': 'This migration ran before orphan snapshots were '
                             'recorded, so its report is no longer available.'
                }
            ), 410
        matches = {}
        manual_matches = {}
        new_meta = {}
        # The provider id and the path are read from the snapshot, not rebuilt: the
        # unbind step deleted exactly these track_server_map rows, so afterwards
        # translate_ids has nothing to map and score.file_path is not populated -
        # the report came out with both identifying columns blank.
        orphans = post_migration.get('orphans') or []
        # PK lookup, NOT _load_score_rows_as_dicts: that one filters to rows still
        # available on the DEFAULT server, which an orphan by definition is not.
        by_item_id = _load_score_rows_by_ids([o.get('item_id') for o in orphans])
        old_rows = []
        snapshot_provider_ids = {}
        for orphan in orphans:
            item_id = orphan.get('item_id')
            row = dict(by_item_id.get(item_id) or {'item_id': item_id})
            row['file_path'] = orphan.get('old_path') or row.get('file_path') or ''
            snapshot_provider_ids[item_id] = orphan.get('old_id') or ''
            old_rows.append(row)
    else:
        dry_run = state.get('dry_run') or {}
        auto_matches = dry_run.get('matches') or {}
        manual_matches = state.get('manual_matches') or {}
        manual_unmatches = set(state.get('manual_unmatches') or [])
        new_meta = _load_target_meta(session_id)

        # Same effective-merge logic as finalize: drop auto rows the user
        # force-orphaned, then manual_matches wins on any remaining conflict.
        matches = {}
        for old_id, new_id in auto_matches.items():
            if old_id not in manual_unmatches:
                matches[old_id] = new_id
        matches.update(manual_matches)
        old_rows = _load_score_rows_as_dicts()
        snapshot_provider_ids = None

    # The old_id column carries the source server's provider id. An internal fp_ id
    # must never reach ANY response - this is an authenticated GET like any other -
    # so a source row with no provider mapping gets a BLANK old_id (fail closed)
    # rather than its raw canonical id. A GENUINE translation error 503s here (so a
    # transient DB hiccup does not silently blank every row with no admin signal);
    # only a truly unmapped row blanks, its other columns (path/artist/album/track,
    # any of which may itself be empty) the best remaining hint.
    # After a migration the snapshot already holds the source provider id, taken
    # from the mapping row at the moment it was deleted. Translating instead would
    # blank every row, because that mapping no longer exists.
    if snapshot_provider_ids is not None:
        old_id_provider_map = snapshot_provider_ids
    else:
        from tasks.mediaserver import registry
        try:
            old_id_provider_map = registry.translate_ids(
                [str(old['item_id']) for old in old_rows if old.get('item_id')], None
            )
        except Exception:
            logger.exception("Dry-run report source id translation failed")
            return jsonify({'error': 'Report generation failed; retry shortly.'}), 503

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            'old_id',
            'old_artist',
            'old_album',
            'old_album_artist',
            'old_track',
            'old_path',
            'new_id',
            'new_artist',
            'new_album',
            'new_album_artist',
            'new_track',
            'new_path',
            'match_source',
        ]
    )
    for old in old_rows:
        old_id = old.get('item_id')
        new_id = matches.get(old_id)
        meta = (new_meta.get(str(new_id)) or new_meta.get(new_id)) if new_id else None
        if new_id and manual_matches.get(old_id):
            source = 'manual'
        elif new_id:
            source = 'auto'
        else:
            source = 'orphan'
        writer.writerow(
            [
                old_id_provider_map.get(old_id, ''),
                old.get('author') or old.get('album_artist') or '',
                old.get('album') or '',
                old.get('album_artist') or '',
                old.get('title') or '',
                old.get('file_path') or '',
                new_id or '',
                (meta or {}).get('artist') or '',
                (meta or {}).get('album') or '',
                (meta or {}).get('album_artist') or '',
                (meta or {}).get('title') or '',
                (meta or {}).get('path') or '',
                source,
            ]
        )

    from flask import Response

    return Response(
        buf.getvalue(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename=migration_session_{session_id}_dry_run.csv',
        },
    )


@migration_bp.route('/api/migration/matched-albums/<int:session_id>', methods=['GET'])
def matched_albums(session_id):
    """
    List currently-matched albums.
    ---
    tags:
      - Provider Migration
    summary: Step 4 review - return albums grouped by old (album_artist, album) with their target album, used for the wizard's correction view.
    description: |
      Auto-matched rows are skipped to keep the review list focused on
      albums the user (or rematch flows) explicitly modified. New-side
      columns use the most common target album across the matched tracks
      in each group.
    parameters:
      - name: session_id
        in: path
        required: true
        schema: { type: integer }
    responses:
      200:
        description: Grouped matched-album list.
        content:
          application/json:
            schema:
              type: object
              properties:
                albums:
                  type: array
                  items:
                    type: object
      404:
        description: Session not found.
    """
    # The review list only ever shows manually re-targeted albums, so we need
    # just ``manual_matches`` (small) - not the full state blob, the auto-match
    # map, or the whole score table. Load only those rows + their target meta.
    found, manual_matches = _read_state_key(session_id, 'manual_matches')
    if not found:
        return jsonify({'error': 'session not found'}), 404
    manual_matches = manual_matches or {}
    if not manual_matches:
        return jsonify({'albums': []})

    old_rows = list(_load_score_rows_by_ids(manual_matches.keys()).values())
    new_meta = _load_target_meta(session_id, list(manual_matches.values()))
    groups = {}  # (old_artist, old_album) -> {'count', 'new_ids', 'tiers'}
    for r in old_rows:
        old_id = r['item_id']
        new_id = manual_matches.get(old_id)
        if new_id is None:
            continue
        key = (r.get('album_artist') or r.get('author') or '', r.get('album') or '')
        g = groups.setdefault(key, {'count': 0, 'new_ids': [], 'tiers': []})
        g['count'] += 1
        g['new_ids'].append(new_id)
        g['tiers'].append('manual')

    albums = []
    for (old_artist, old_album), g in groups.items():
        tally = {}  # (new_artist, new_album) -> count
        for new_id in g['new_ids']:
            meta = new_meta.get(str(new_id)) or new_meta.get(new_id) or {}
            tally_key = (
                meta.get('album_artist') or meta.get('artist') or '',
                meta.get('album') or '',
            )
            tally[tally_key] = tally.get(tally_key, 0) + 1
        if tally:
            (new_artist, new_album), _ = max(tally.items(), key=lambda kv: kv[1])
        else:
            new_artist, new_album = '', ''
        tier_tally = {}
        for t in g['tiers']:
            tier_tally[t] = tier_tally.get(t, 0) + 1
        dominant_tier = (
            max(tier_tally.items(), key=lambda kv: kv[1])[0] if tier_tally else 'unknown'
        )
        albums.append(
            {
                'old_album_artist': old_artist,
                'old_album': old_album,
                'track_count': g['count'],
                'new_album_artist': new_artist,
                'new_album': new_album,
                'tier': dominant_tier,
            }
        )

    albums.sort(
        key=lambda a: (
            (a['old_album_artist'] or '').lower(),
            (a['old_album'] or '').lower(),
        )
    )
    return jsonify({'albums': albums})


# ---------------------------------------------------------------------------
# Small DB helpers (kept near the routes that use them so behavior + SQL live
# together; these are also why the test suite patches ``get_db``).
# ---------------------------------------------------------------------------


def _fetch_session_creds(session_id, *, require_plannable=False):
    db = get_db()
    with db.cursor() as cur:
        query = "SELECT target_type, target_creds FROM migration_session WHERE id = %s"
        if require_plannable:
            query += (
                " AND status NOT IN ('completed', 'failed') "
                "AND id = (SELECT MAX(id) FROM migration_session)"
            )
        cur.execute(query, (session_id,))
        row = cur.fetchone()
    if not row:
        return None
    target_type, creds_raw = row
    try:
        creds = json.loads(creds_raw) if isinstance(creds_raw, str) else (creds_raw or {})
    except Exception:
        creds = {}
    return target_type, creds


def _row_to_score_dict(r):
    return {
        'item_id': r[0],
        'file_path': r[1],
        'title': r[2],
        'author': r[3],
        'album': r[4],
        'album_artist': r[5],
    }


_SCORE_COL_NAMES = ("item_id", "file_path", "title", "author", "album", "album_artist")
_SCORE_COLS = pgsql.SQL(", ").join(pgsql.Identifier(c) for c in _SCORE_COL_NAMES)


def _load_score_rows_as_dicts():
    from tasks.mediaserver import registry

    db = get_db()
    default = registry.get_default_server(db)
    default_id = default['server_id'] if default else None
    if default_id is None:
        with db.cursor() as cur:
            cur.execute(pgsql.SQL("SELECT {} FROM score").format(_SCORE_COLS))
            rows = cur.fetchall() or []
        return [_row_to_score_dict(r) for r in rows]
    with db.cursor() as cur:
        cur.execute(
            "SELECT s.item_id, (SELECT p.file_path FROM track_server_map p "
            "WHERE p.item_id = s.item_id AND p.server_id = %s "
            "AND p.file_path IS NOT NULL LIMIT 1), "
            "s.title, s.author, s.album, s.album_artist "
            "FROM score s WHERE " + registry.availability_sql('s'),
            (default_id, default_id, True),
        )
        rows = cur.fetchall() or []
    return [_row_to_score_dict(r) for r in rows]


def _load_score_rows_by_ids(item_ids):
    ids = [str(i) for i in (item_ids or [])]
    if not ids:
        return {}
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            pgsql.SQL("SELECT {} FROM score WHERE item_id = ANY(%s)").format(_SCORE_COLS),
            (ids,),
        )
        rows = cur.fetchall() or []
    return {r[0]: _row_to_score_dict(r) for r in rows}


def _sanitize_text(value):
    from sanitization import sanitize_string_for_db

    return sanitize_string_for_db(value)


def _store_target_meta(session_id, new_meta):
    db = get_db()
    with db.cursor() as cur:
        # Serialize with the execute transaction's completion UPDATE.  If execute
        # wins the row lock, this worker observes ``completed`` and writes nothing;
        # if the dry run wins, execute waits and then deletes these rows in its own
        # commit.  There is therefore no ordering in which late bulk metadata can
        # repopulate an already-completed session.
        cur.execute(
            "SELECT status, (id = (SELECT MAX(id) FROM migration_session)) "
            "FROM migration_session WHERE id = %s FOR UPDATE",
            (session_id,),
        )
        session = cur.fetchone()
        if not session:
            raise RuntimeError(f'migration session {session_id} no longer exists')
        status, is_current = session
        if status in ('completed', 'failed') or not is_current:
            raise RuntimeError(
                f'migration session {session_id} is no longer mutable; refusing '
                'late target metadata'
            )
        cur.execute("DELETE FROM migration_target_meta WHERE session_id = %s", (session_id,))
        rows = [
            (
                session_id,
                _sanitize_text(new_id),
                _sanitize_text((meta or {}).get('path')),
                _sanitize_text((meta or {}).get('title')),
                _sanitize_text((meta or {}).get('artist')),
                _sanitize_text((meta or {}).get('album')),
                _sanitize_text((meta or {}).get('album_artist')),
                (meta or {}).get('year'),
            )
            for new_id, meta in (new_meta or {}).items()
        ]
        for i in range(0, len(rows), 500):
            chunk = rows[i : i + 500]
            placeholders = ",".join(["(%s,%s,%s,%s,%s,%s,%s,%s)"] * len(chunk))
            flat = [v for row in chunk for v in row]
            cur.execute(
                "INSERT INTO migration_target_meta "
                "(session_id, new_id, path, title, artist, album, album_artist, year) "
                "VALUES " + placeholders,  # nosec B608 - %s-placeholder string only; values are bound params
                flat,
            )
    db.commit()


def _load_target_meta(session_id, new_ids=None):
    if new_ids is not None:
        ids = [str(n) for n in new_ids]
        if not ids:
            return {}
    db = get_db()
    with db.cursor() as cur:
        if new_ids is None:
            cur.execute(
                "SELECT new_id, path, title, artist, album, album_artist, year "
                "FROM migration_target_meta WHERE session_id = %s",
                (session_id,),
            )
        else:
            cur.execute(
                "SELECT new_id, path, title, artist, album, album_artist, year "
                "FROM migration_target_meta WHERE session_id = %s AND new_id = ANY(%s)",
                (session_id, ids),
            )
        rows = cur.fetchall() or []
    return {
        r[0]: {
            'path': r[1],
            'title': r[2],
            'artist': r[3],
            'album': r[4],
            'album_artist': r[5],
            'year': r[6],
        }
        for r in rows
    }


def _load_rows_for_album(album_key):
    target_artist, target_album = (
        album_key[0] if album_key else None,
        album_key[1] if album_key and len(album_key) > 1 else None,
    )
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            pgsql.SQL(
                "SELECT {} FROM score "
                "WHERE COALESCE(NULLIF(album_artist, ''), author) IS NOT DISTINCT FROM %s "
                "AND album IS NOT DISTINCT FROM %s"
            ).format(_SCORE_COLS),
            (target_artist, target_album),
        )
        rows = cur.fetchall() or []
    return [_row_to_score_dict(r) for r in rows]


def _load_unmatched_for_album(session_id, album_key):
    state = _load_state(session_id) or {}
    manual_unmatches = set(state.get('manual_unmatches') or [])
    matched_ids = set((state.get('dry_run') or {}).get('matches', {}).keys()) - manual_unmatches
    matched_ids |= set((state.get('manual_matches') or {}).keys())
    return [r for r in _load_rows_for_album(album_key) if r['item_id'] not in matched_ids]


# Hard cap on the number of unmatched albums returned to the wizard. The
# value is read from ``config.MIGRATION_UNMATCHED_ALBUMS_PAYLOAD_LIMIT`` so
# operators can tune it via env var or the setup wizard's DB-backed
# overrides without touching this module. Callers that need the true
# count should use ``len(unmatched_by_album)`` separately.


def _albums_payload(unmatched_by_album):
    import config

    limit = config.MIGRATION_UNMATCHED_ALBUMS_PAYLOAD_LIMIT
    out = []
    for key, rows in unmatched_by_album.items():
        if len(out) >= limit:
            break
        album_artist, album = key[0], key[1] if len(key) > 1 else None
        out.append(
            {
                'album_artist': album_artist,
                'album': album,
                'track_count': len(rows),
                'sample_titles': [r.get('title') for r in rows[:5]],
            }
        )
    return out


def _load_state(session_id):
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT state FROM migration_session WHERE id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    state = row[0]
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except Exception:
            state = {}
    return state or {}


def _sanitize_json_value(value):
    from sanitization import sanitize_json_for_db

    return sanitize_json_for_db(value)


def _patch_state_keys(session_id, _set_status=None, **patch):
    db = get_db()
    with db.cursor() as cur:
        if _set_status is not None:
            cur.execute(
                "UPDATE migration_session SET status = %s "
                "WHERE id = %s AND status IS DISTINCT FROM 'completed'",
                (_set_status, session_id),
            )
        for k, v in patch.items():
            if v is None:
                cur.execute(
                    "UPDATE migration_session SET state = state - %s "
                    "WHERE id = %s AND status IS DISTINCT FROM 'completed'",
                    (k, session_id),
                )
            else:
                cur.execute(
                    "UPDATE migration_session SET state = jsonb_set("
                    "COALESCE(state, '{}'::jsonb), %s, %s::jsonb, true) "
                    "WHERE id = %s AND status IS DISTINCT FROM 'completed'",
                    ([k], json.dumps(_sanitize_json_value(v), ensure_ascii=False), session_id),
                )
    db.commit()


def _update_state(session_id, **patch):
    _patch_state_keys(session_id, _set_status='in_progress', **patch)


def _read_state_key(session_id, key):
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT state -> %s FROM migration_session WHERE id = %s",
            (key, session_id),
        )
        row = cur.fetchone()
    if row is None:
        return False, None
    return True, row[0]


def _merge_manual_matches(session_id, new_matches):
    _, manual = _read_state_key(session_id, 'manual_matches')
    manual = dict(manual or {})
    manual.update(new_matches)
    # Invalidate final_counts so the user must re-finalize
    _patch_state_keys(session_id, manual_matches=manual, final_counts=None)


def _rematch_album_rows(session_id, newly_matched, newly_unmatched):
    _, manual = _read_state_key(session_id, 'manual_matches')
    _, unmatch_list = _read_state_key(session_id, 'manual_unmatches')
    manual = dict(manual or {})
    unmatches = set(unmatch_list or [])
    for old_id, new_id in newly_matched.items():
        manual[old_id] = new_id
        unmatches.discard(old_id)
    for old_id in newly_unmatched:
        manual.pop(old_id, None)
        unmatches.add(old_id)
    _patch_state_keys(
        session_id,
        manual_matches=manual,
        manual_unmatches=sorted(unmatches),
        final_counts=None,
    )


def _mark_album_skipped(session_id, old_album_key):
    _, skipped = _read_state_key(session_id, 'skipped_albums')
    skipped = list(skipped or [])
    if old_album_key and old_album_key not in skipped:
        skipped.append(old_album_key)
    _patch_state_keys(session_id, skipped_albums=skipped, final_counts=None)


def _count_score_rows():
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM score")
        row = cur.fetchone()
    return int(row[0] or 0) if row else 0
