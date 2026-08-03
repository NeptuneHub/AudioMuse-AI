# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""App-layer helpers composing the data and queue layers for the web/task tiers.

Orchestration and presentation glue on top of ``database`` and ``taskqueue``.
This is NOT the database layer: all SQL lives in ``database.py``. It also
re-exports the most-used ``database`` / ``taskqueue`` handles so the many
modules doing ``from app_helper import get_db, redis_conn, ...`` stay untouched.

Main Features:
* ``cancel_job_and_children_recursive`` recursively cancels an RQ job tree;
  ``revoke_inline_task_row`` handles the tasks that run in the web process with no
  RQ job, revoking only their row instead of wiping the queues to reach nothing.
* ``build_and_store_map_projection`` / ``build_and_store_artist_projection``
  compute a 2D projection and persist it; ``attach_song_features`` /
  ``top_stratified_genre`` enrich API result rows.
* Shared blueprint helpers: ``index_error_body`` builds the structured API
  error body and ``probe_catalogue_canonical_ids`` probes score for canonical
  fp_ ids (None on probe failure so callers pick their own fail-closed policy).
"""

import json
import logging
import time

from psycopg2.extras import DictCursor
import numpy as np

import database
import rq_job_state
from database import (  # noqa: F401
    get_db,
    close_db,
    coerce_db_details,
    INLINE_FLASK_TASK_TYPES,
    save_task_status,
    record_task_history,
    _build_task_note,
    get_score_data_by_ids,
    load_map_projection,
    get_task_info_from_db,
    get_task_statuses,
    main_task_start_lock,
    get_tracks_by_ids,
    save_track_analysis_and_embedding,
    # Used internally by the build_and_store_* projection orchestration below.
    save_map_projection,
    save_artist_projection,
)
from taskqueue import (
    redis_conn,
    rq_queue_high,
    rq_queue_default,
    Job,
    NoSuchJobError,
    send_stop_job_command,
)

from config import (  # noqa: F401
    STRATIFIED_GENRES,
    TASK_STATUS_PENDING,
    TASK_STATUS_STARTED,
    TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
    TASK_STATUS_REVOKED,
)

from error import error_manager
from error.error_dictionary import UNKNOWN_ERROR_CODE

logger = logging.getLogger(__name__)


# The Flask `app` object is intentionally NOT imported here (circular import);
# use the module-level `logger` above. The 2D map/artist projection caches live
# in database.MAP_PROJECTION_CACHE / database.ARTIST_PROJECTION_CACHE, written by
# the build_and_store_* helpers below and read by database.load_*_projection.


def index_error_body(code, message):
    payload = error_manager.build(code)
    payload["error"] = message
    return payload


def probe_catalogue_canonical_ids():
    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM score WHERE item_id LIKE 'fp\\_%%')"
            )
            return bool(cur.fetchone()[0])
    except Exception:
        logger.exception("Canonical-id probe failed; failing closed")
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                logger.exception("Rollback after failed canonical-id probe also failed")
        return None


def sanitize_task_details(details, state, task_type=None):
    """Normalize a persisted task ``details`` dict for any task-status endpoint.

    Applies the same safety pass to every endpoint that surfaces task details:
    drops the internal traceback and the heavyweight analysis-only
    ``checked_album_ids`` key, truncates the log to the last 10 entries, and
    guarantees a well-formed structured ``error`` (plus ``error_message``) on
    failed tasks so the frontend renderer always receives a consistent, safe
    shape whether it hit ``/api/status``, ``/api/last_task`` or ``/api/active_tasks``.
    """
    if not isinstance(details, dict):
        return details

    if task_type and 'analysis' in task_type:
        details.pop('checked_album_ids', None)
    details.pop('traceback', None)

    # Internal canonical (fp_) ids must never reach a task-status response. The
    # clustering-batch child stows raw sampled ids, and the cleaning summary lists
    # orphaned tracks (on no server, so untranslatable) by their catalogue id.
    # Strip them here - no UI reads these, and the parent tasks read the job's
    # return value, not this display copy.
    details.pop('final_subset_track_ids', None)
    details.pop('full_best_result_from_batch', None)
    summary = details.get('final_summary_details')
    if isinstance(summary, dict) and isinstance(summary.get('orphaned_albums'), list):
        from tasks.simhash import is_fingerprint_id
        for album in summary['orphaned_albums']:
            if not isinstance(album, dict) or not isinstance(album.get('tracks'), list):
                continue
            for track in album['tracks']:
                # Hide only the internal canonical (fp_) id; a legacy provider id is
                # not internal, so keep it - matching the is_fingerprint_id gate used
                # everywhere else, instead of over-stripping legacy installs.
                if isinstance(track, dict) and is_fingerprint_id(str(track.get('item_id'))):
                    track.pop('item_id', None)

    log_entries = details.get('log')
    if isinstance(log_entries, list) and len(log_entries) > 10:
        details['log'] = [
            f"... ({len(log_entries) - 10} earlier log entries truncated)",
            *log_entries[-10:],
        ]

    if str(state or '').upper() in ('FAILED', 'FAILURE'):
        existing_error = details.get('error')
        has_full_error = (
            isinstance(existing_error, dict)
            and 'error_code' in existing_error
            and 'error_message' in existing_error
        )
        if not has_full_error:
            if isinstance(existing_error, dict) and 'error_code' in existing_error:
                details['error'] = error_manager.build(existing_error['error_code'])
            else:
                details['error'] = error_manager.build(UNKNOWN_ERROR_CODE)
        details.setdefault('error_message', details['error']['error_message'])

    return details


def top_stratified_genre(mood_vector):
    """Return the highest-scoring genre label present in STRATIFIED_GENRES, or None.

    Mirrors the genre selection used by clustering (tasks/clustering_helper.py): the
    mood_vector also carries non-genre labels (e.g. 'female vocalist') and moods, so
    only labels in STRATIFIED_GENRES qualify as the displayed genre.
    """
    if not mood_vector or not isinstance(mood_vector, str):
        return None
    scores = {}
    for part in mood_vector.split(','):
        label, _, value = part.partition(':')
        label = label.strip()
        if not label:
            continue
        try:
            scores[label] = float(value)
        except ValueError:
            continue
    candidates = [g for g in STRATIFIED_GENRES if g in scores]
    if not candidates:
        return None
    return max(candidates, key=scores.get)


def attach_song_features(rows, id_key='item_id'):
    """Additively add album + mood_vector + other_features + top_genre to each result dict.

    Signature-safe: only fills keys that are missing; never removes or overwrites
    existing data, so callers that already include these fields are unaffected.
    """
    if not rows:
        return rows
    ids = [r.get(id_key) for r in rows if isinstance(r, dict) and r.get(id_key)]
    if not ids:
        return rows
    score = {str(s['item_id']): s for s in get_score_data_by_ids(ids)}
    for r in rows:
        if not isinstance(r, dict):
            continue
        s = score.get(str(r.get(id_key)))
        if s:
            r.setdefault('album', s.get('album'))
            r.setdefault('mood_vector', s.get('mood_vector'))
            r.setdefault('other_features', s.get('other_features'))
            r.setdefault('top_genre', top_stratified_genre(s.get('mood_vector')))
    return rows


def serialize_neighbor_results(
    neighbor_results, missing_album='unknown', include_album_artist=True
):
    """Build the similar-tracks JSON list from neighbor dicts carrying item_id + distance.

    Shared by the IVF similarity endpoints and the sonic-fingerprint endpoint so the
    response shape lives in one place. missing_album / include_album_artist keep each
    caller's existing output shape.
    """
    if not neighbor_results:
        return []
    ids = [n['item_id'] for n in neighbor_results]
    details_map = {d['item_id']: d for d in get_score_data_by_ids(ids)}
    distance_map = {n['item_id']: n['distance'] for n in neighbor_results}
    out = []
    for nid in ids:
        info = details_map.get(nid)
        if not info:
            continue
        # missing_album=None means "no substitution" (sonic fingerprint keeps the
        # raw album, incl. '') -- only fall back when a sentinel is supplied.
        album = info.get('album')
        if missing_album is not None:
            album = album or missing_album
        row = {
            "item_id": info['item_id'],
            "title": info['title'],
            "author": info['author'],
            "album": album,
            "distance": distance_map[nid],
            "mood_vector": info.get('mood_vector'),
            "other_features": info.get('other_features'),
            "top_genre": top_stratified_genre(info.get('mood_vector')),
        }
        if include_album_artist:
            row["album_artist"] = info.get('album_artist') or 'unknown'
        out.append(row)
    return out


def _project_matrix_2d(mat, label):
    from tasks.alchemy_projections import _project_with_umap, _project_to_2d

    projections = None
    try:
        projections = _project_with_umap(mat)
    except Exception:
        logger.exception(f"UMAP projection failed for {label}; falling back to PCA")
        projections = None

    if projections is None:
        try:
            projections = _project_to_2d(mat)
        except Exception as exc:
            logger.exception(f"PCA projection failed for {label}")
            raise RuntimeError(
                f"2D projection failed for {label}: both UMAP and PCA raised; "
                "refusing to store an all-zeros projection."
            ) from exc

    if projections is None:
        raise RuntimeError(
            f"2D projection failed for {label}: no projector produced output; "
            "refusing to store an all-zeros projection."
        )
    return np.array(projections, dtype=np.float32)


def build_and_store_map_projection(index_name='main_map'):
    """Compute 2D projection for all tracks and store it. Uses available projection helpers if present.
    Returns True on success.
    """
    from config import EMBEDDING_DIMENSION
    from tasks.index_build_helpers import stream_embeddings_to_buffer

    try:
        mat, ids = stream_embeddings_to_buffer(
            table="embedding",
            column="embedding",
            dim=EMBEDDING_DIMENSION,
            where_clause="embedding IS NOT NULL",
        )
    except Exception:
        logger.exception("Failed to stream embeddings for map projection")
        return False

    if mat.shape[0] == 0:
        logger.info('No embeddings available to build map projection.')
        return False

    logger.info(f"Starting to build map projection: {mat.shape[0]} embeddings found.")
    projections = _project_matrix_2d(mat, 'map projection')
    logger.info(f"Computed projection shape: {projections.shape}")

    # Save to DB
    try:
        save_map_projection(index_name, ids, projections)
        # Update the canonical in-memory cache (read by database.load_map_projection).
        database.MAP_PROJECTION_CACHE = {
            'index_name': index_name,
            'id_map': ids,
            'projection': projections,
        }
        # Note: Caller (analysis task) is responsible for publishing reload message after all builds complete
        return True
    except Exception:
        logger.exception("Failed to build and store map projection")
        return False


def build_and_store_artist_projection(index_name='artist_map'):
    """Compute 2D projection for all artist GMM components and store it.
    This will be called during analysis to create the artist component map.
    Returns True on success.
    """
    from tasks.artist_gmm_manager import load_artist_index_for_querying

    # Always reload artist GMM params from database (force reload to ensure fresh data)
    load_artist_index_for_querying(force_reload=True)

    # Re-import after loading to get the updated global variable
    from tasks.artist_gmm_manager import artist_gmm_params as loaded_params

    if not loaded_params:
        logger.warning("No artist GMM params available to build artist projection.")
        return False

    from tasks.mediaserver import registry
    artist_ids = registry.artist_ids_for_names(list(loaded_params.keys()))

    # Two-pass build: first pass counts components and infers dim, second
    # pass fills a single pre-allocated ndarray. Avoids the previous
    # ``vectors = []; vectors.append(...); np.vstack(vectors)`` pattern
    # that materialised three copies of the component matrix at once.
    total_components = 0
    component_dim = None
    for gmm in loaded_params.values():
        means = gmm.get('means') or []
        if not len(means):
            continue
        if component_dim is None:
            component_dim = int(np.asarray(means[0], dtype=np.float32).size)
        total_components += len(means)

    if total_components == 0 or component_dim is None:
        logger.info('No artist component vectors available to build projection.')
        return False

    mat = np.empty((total_components, component_dim), dtype=np.float32)
    component_map = []
    row_i = 0
    for artist_name, gmm in loaded_params.items():
        means = gmm.get('means') or []
        weights = gmm.get('weights') or []
        if not len(means):
            continue
        artist_id = artist_ids.get(artist_name) or artist_name
        for comp_idx in range(len(means)):
            mat[row_i] = np.asarray(means[comp_idx], dtype=np.float32)
            component_map.append(
                {
                    'artist_id': artist_id,
                    'artist_name': artist_name,
                    'component_idx': comp_idx,
                    'weight': float(weights[comp_idx]) if comp_idx < len(weights) else 0.0,
                }
            )
            row_i += 1

    logger.info(f"Starting to build artist projection: {mat.shape[0]} component vectors found.")
    projections = _project_matrix_2d(mat, 'artist components')
    logger.info(f"Computed artist projection shape: {projections.shape}")

    try:
        save_artist_projection(index_name, component_map, projections)
        # Update the canonical in-memory cache (read by database.load_artist_projection).
        database.ARTIST_PROJECTION_CACHE = {
            'index_name': index_name,
            'component_map': component_map,
            'projection': projections,
        }
        # Note: Caller (analysis task) is responsible for publishing reload message after all builds complete
        return True
    except Exception:
        logger.exception("Failed to build and store artist projection")
        return False


_INLINE_CANCEL_MESSAGE = (
    "Cancelled. This task runs inside the web process and cannot be interrupted "
    "mid-step, so a step already in flight still finishes."
)


def revoke_inline_task_row(task_id):
    """Revoke ONLY this row when task_id is an in-process task, else return None.

    An inline task (the alchemy radio) is never enqueued, so the global cancel
    below would stop nothing while still emptying both RQ queues and deleting every
    task_status row - destroying unrelated queued work to cancel something it cannot
    reach. Revoking the one row clears it from the UI, which is what Stop is for.
    """
    try:
        task_info = get_task_info_from_db(task_id)
    except Exception:
        logger.exception("Could not read task %s before cancelling", task_id)
        return None
    if not task_info or task_info.get('task_type') not in INLINE_FLASK_TASK_TYPES:
        return None
    save_task_status(
        task_id, task_info['task_type'], TASK_STATUS_REVOKED, progress=100,
        details={
            'message': _INLINE_CANCEL_MESSAGE,
            'status_message': _INLINE_CANCEL_MESSAGE,
        },
    )
    logger.info("Revoked in-process task row %s without touching the queues.", task_id)
    return _INLINE_CANCEL_MESSAGE


def cancel_job_and_children_recursive(
    job_id, reason="Task cancellation processed by API."
):
    """Helper to cancel a job and its children based on DB records.

    NOTE: Minimal global behavior - when invoked from the API cancel endpoint we clear RQ queues,
    attempt to stop all jobs known to RQ, delete all rows in `task_status`, and insert a single
    REVOKED row for the requested `job_id` (so UI sees one canonical cancelled task).
    This is intentionally simple and destructive (as requested).
    """
    # Serialize against the start paths. Cancel scans RQ, then wipes task_status;
    # a starter that had committed its PENDING row but not yet enqueued was invisible
    # to the scan AND had its row deleted, so it enqueued afterwards and ran after the
    # user pressed Cancel.
    with main_task_start_lock():
        return _cancel_job_and_children_locked(job_id, reason)


def _cancel_job_and_children_locked(job_id, reason):
    cancelled_count = 0

    # --- Scan RQ for job ids to cancel ---
    job_ids = set()
    for q in (rq_queue_high, rq_queue_default):
        try:
            ids = getattr(q, 'job_ids', None)
            if ids is None:
                key = f"rq:queue:{getattr(q, 'name', '')}"
                raw = redis_conn.lrange(key, 0, -1)
                ids = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in raw]
            job_ids.update([str(i) for i in ids if i is not None])
        except Exception as e_q:
            logger.warning(f"Could not read queue {getattr(q, 'name', '<unknown>')}: {e_q}")

    # Include job ids from RQ job keys (covers started jobs)
    try:
        for k in redis_conn.scan_iter(match='rq:job:*', count=500):
            kstr = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            parts = kstr.split(':')
            if len(parts) >= 3:
                jid = ':'.join(parts[2:])
                job_ids.add(jid)
    except Exception as e_keys:
        logger.warning(f"Could not list rq job keys: {e_keys}")

    # Attempt to cancel/stop all discovered jobs
    for jid in job_ids:
        try:
            try:
                j = Job.fetch(jid, connection=redis_conn)
                status = j.get_status(refresh=False)
                if not rq_job_state.is_terminal_status(status):
                    # Zero the retry budget FIRST. A stopped job keeps retries_left, and
                    # RQ's StartedJobRegistry.cleanup() requeues any expired execution that
                    # still has one, so a worker restart used to resurrect the very job the
                    # user just cancelled and run it invisibly against a REVOKED row.
                    rq_job_state.forbid_retries(jid, redis_conn)
                    if rq_job_state.is_running_status(status):
                        send_stop_job_command(redis_conn, jid)
                    else:
                        j.cancel()
                    cancelled_count += 1
                    logger.info(f"Sent stop/cancel for job {jid} during global cancel")
            except NoSuchJobError:
                logger.debug(f"Job {jid} not found in RQ during global cancel")
        except Exception:
            logger.exception(f"Error cancelling job {jid} during global cancel")

    # Try to clear the RQ queues using API (preferred) and fallback to key deletion if necessary
    try:
        for q in (rq_queue_high, rq_queue_default):
            try:
                if hasattr(q, 'empty'):
                    q.empty()
                    logger.info(
                        f"Emptied queue {getattr(q, 'name', '<unknown>')} via Queue.empty() as part of global cancel"
                    )
                else:
                    key = f"rq:queue:{getattr(q, 'name', '')}"
                    redis_conn.delete(key)
                    logger.info(
                        f"Deleted Redis key fallback for queue: {key} as part of global cancel"
                    )
            except Exception as e_q:
                logger.warning(
                    f"Failed to empty queue {getattr(q, 'name', '<unknown>')} during global cancel: {e_q}"
                )
    except Exception as e_qdel:
        logger.warning(f'Failed to clear queue lists during global cancel: {e_qdel}')

    # Consolidate DB: wipe task_status and leave ONE REVOKED recap row for the id the
    # user cancelled, so the table cannot grow without bound.
    #
    # The wipe IS the cancellation signal. Every cooperative check therefore treats a
    # MISSING row as revoked, never as "carry on": reading absence as "not cancelled"
    # is what let a cancelled analysis keep enqueuing albums onto the queue the cancel
    # had just emptied. See revoked()/revoked_now() in tasks/analysis.py,
    # make_cancel_check in tasks/multiserver_sync.py, and the guards in
    # tasks/clustering.py.
    db = get_db()
    cur = db.cursor()
    try:
        # Snapshot the in-flight main tasks into the persistent task_history first,
        # so the dashboard's history table keeps showing what was running when the
        # user pressed Cancel.
        try:
            with db.cursor(cursor_factory=DictCursor) as snap_cur:
                snap_cur.execute(
                    "SELECT task_id, task_type, status, details, start_time, end_time "
                    "FROM task_status WHERE parent_task_id IS NULL"
                )
                now_ts = time.time()
                for r in snap_cur.fetchall():
                    duration_s = None
                    if r['start_time'] is not None:
                        end = r['end_time'] if r['end_time'] is not None else now_ts
                        duration_s = max(0.0, float(end) - float(r['start_time']))
                    details_obj = None
                    if r['details']:
                        try:
                            details_obj = json.loads(r['details'])
                        except Exception:
                            details_obj = None
                    final_status = (
                        r['status']
                        if r['status']
                        in (TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)
                        else TASK_STATUS_REVOKED
                    )
                    record_task_history(
                        r['task_id'],
                        r['task_type'],
                        final_status,
                        duration_s,
                        details=details_obj,
                    )
        except Exception as e_snap:
            logger.warning(
                f"Global cancel: failed snapshotting task_status into task_history: {e_snap}"
            )

        cur.execute("DELETE FROM task_status")
        deleted = cur.rowcount
        db.commit()
        logger.info(f"Global cancel DB cleanup: deleted {deleted} task_status rows")
    except Exception:
        db.rollback()
        logger.exception("Error deleting task_status rows during global cancel")
    finally:
        cur.close()

    try:
        # The single surviving row: the id the user actually cancelled, so the UI has
        # one canonical cancelled task to show.
        save_task_status(
            job_id,
            'unknown',
            TASK_STATUS_REVOKED,
            progress=100,
            details={"message": reason, "origin": "global_cancel"},
        )
    except Exception:
        logger.exception(f"Failed to insert REVOKED recap row for {job_id}")

    return cancelled_count
