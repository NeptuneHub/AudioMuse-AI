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
* ``cancel_job_and_children_recursive`` recursively cancels an RQ job tree.
* ``build_and_store_map_projection`` / ``build_and_store_artist_projection``
  compute a 2D projection and persist it; ``attach_song_features`` /
  ``top_stratified_genre`` enrich API result rows.
"""

import json
import logging
import time

from psycopg2.extras import DictCursor
import numpy as np

import database
from database import (  # noqa: F401
    get_db,
    close_db,
    save_task_status,
    record_task_history,
    _build_task_note,
    get_score_data_by_ids,
    load_map_projection,
    get_task_info_from_db,
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


def get_score_data_lite_by_ids(item_ids_list):
    """
    Slim version of get_score_data_by_ids - drops the large text columns
    (mood_vector, other_features) and unused fields (key, scale, energy,
    file_path) so list-style responses (e.g. paginated Smart Search) stay small.
    """
    if not item_ids_list:
        return []
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    query = """
        SELECT s.item_id, s.title, s.author, s.album, s.album_artist,
               s.tempo, s.year, s.rating
        FROM score s
        WHERE s.item_id IN %s
    """
    try:
        cur.execute(query, (tuple(item_ids_list),))
        rows = cur.fetchall()
    except Exception:
        logger.exception("Error fetching lite score data by IDs")
        rows = []
    finally:
        cur.close()
    return [dict(row) for row in rows]


def coerce_db_details(raw_details):
    """Normalize a task_status.details DB value to a dict without double-parsing.

    psycopg2 hands back a TEXT details column as a JSON string (needs json.loads)
    but a JSONB column as an already-parsed dict (must NOT be re-parsed). NULL or
    unparseable values collapse to {}.
    """
    if isinstance(raw_details, dict):
        return raw_details
    if raw_details:
        try:
            return json.loads(raw_details)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


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


def build_and_store_map_projection(index_name='main_map'):
    """Compute 2D projection for all tracks and store it. Uses available projection helpers if present.
    Returns True on success.
    """
    # Import local projection helpers to avoid circular imports
    try:
        from tasks.alchemy_projections import _project_with_umap, _project_to_2d
    except Exception:
        _project_with_umap = None
        _project_to_2d = None

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

    projections = None
    try:
        logger.info(f"Starting to build map projection: {mat.shape[0]} embeddings found.")
        if _project_with_umap is not None:
            projections = _project_with_umap([v for v in mat])
    except Exception as e:
        logger.warning(f"UMAP projection failed during build: {e}")
        projections = None

    if projections is None:
        try:
            if _project_to_2d is not None:
                projections = _project_to_2d([v for v in mat])
        except Exception as e:
            logger.warning(f"PCA projection failed during build: {e}")
            projections = None

    if projections is None:
        projections = np.zeros((mat.shape[0], 2), dtype=np.float32)
    else:
        projections = np.array(projections, dtype=np.float32)
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
    from tasks.alchemy_projections import _project_with_umap, _project_to_2d

    # Always reload artist GMM params from database (force reload to ensure fresh data)
    load_artist_index_for_querying(force_reload=True)

    # Re-import after loading to get the updated global variable
    from tasks.artist_gmm_manager import artist_gmm_params as loaded_params

    if not loaded_params:
        logger.warning("No artist GMM params available to build artist projection.")
        return False

    from app_helper_artist import get_artist_id_by_name

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
        artist_id = get_artist_id_by_name(artist_name) or artist_name
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

    projections = None

    try:
        logger.info(f"Starting to build artist projection: {mat.shape[0]} component vectors found.")
        # Try UMAP first
        if _project_with_umap is not None:
            projections = _project_with_umap([v for v in mat])
    except Exception as e:
        logger.warning(f"UMAP projection failed for artist components: {e}")
        projections = None

    # Fallback to PCA
    if projections is None:
        try:
            if _project_to_2d is not None:
                projections = _project_to_2d([v for v in mat])
        except Exception as e:
            logger.warning(f"PCA projection failed for artist components: {e}")
            projections = None

    if projections is None:
        projections = np.zeros((mat.shape[0], 2), dtype=np.float32)
    else:
        projections = np.array(projections, dtype=np.float32)

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


def _rq_queues():
    """Return the live queue objects so tests and runtime patches remain visible."""
    return rq_queue_high, rq_queue_default


def _queue_job_ids(queue):
    """Read and normalize queued job ids from RQ or its Redis list fallback."""
    ids = getattr(queue, 'job_ids', None)
    if ids is None:
        key = f"rq:queue:{getattr(queue, 'name', '')}"
        ids = redis_conn.lrange(key, 0, -1)
    return {
        item.decode() if isinstance(item, (bytes, bytearray)) else str(item)
        for item in ids
        if item is not None
    }


def _started_rq_job_ids():
    """Return ids represented by RQ job keys, including started jobs."""
    job_ids = set()
    for key in redis_conn.keys('rq:job:*'):
        key_text = key.decode() if isinstance(key, (bytes, bytearray)) else str(key)
        prefix, separator, job_id = key_text.partition('rq:job:')
        if not prefix and separator and job_id:
            job_ids.add(job_id)
    return job_ids


def _discover_rq_job_ids():
    """Discover queued and started RQ jobs without failing the global cleanup."""
    job_ids = set()
    for queue in _rq_queues():
        try:
            job_ids.update(_queue_job_ids(queue))
        except Exception as exc:
            logger.warning(
                f"Could not read queue {getattr(queue, 'name', '<unknown>')}: {exc}"
            )
    try:
        job_ids.update(_started_rq_job_ids())
    except Exception as exc:
        logger.warning(f"Could not list rq job keys: {exc}")
    return job_ids


def _cancel_rq_job(job_id):
    """Stop or cancel one active RQ job and report whether action was sent."""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        logger.debug(f"Job {job_id} not found in RQ during global cancel")
        return False

    if job.is_finished or job.is_failed or job.is_canceled:
        return False
    if job.is_started:
        send_stop_job_command(redis_conn, job_id)
    else:
        job.cancel()
    logger.info(f"Sent stop/cancel for job {job_id} during global cancel")
    return True


def _cancel_discovered_jobs(job_ids):
    """Cancel every discovered job while isolating failures per job."""
    cancelled_count = 0
    for job_id in job_ids:
        try:
            cancelled_count += int(_cancel_rq_job(job_id))
        except Exception:
            logger.exception(f"Error cancelling job {job_id} during global cancel")
    return cancelled_count


def _clear_rq_queue(queue):
    """Empty one RQ queue through its API or its backing Redis key."""
    queue_name = getattr(queue, 'name', '<unknown>')
    if hasattr(queue, 'empty'):
        queue.empty()
        logger.info(f"Emptied queue {queue_name} via Queue.empty() as part of global cancel")
        return
    key = f"rq:queue:{getattr(queue, 'name', '')}"
    redis_conn.delete(key)
    logger.info(f"Deleted Redis key fallback for queue: {key} as part of global cancel")


def _clear_rq_queues():
    """Best-effort cleanup of all application RQ queues."""
    for queue in _rq_queues():
        try:
            _clear_rq_queue(queue)
        except Exception as exc:
            logger.warning(
                f"Failed to empty queue {getattr(queue, 'name', '<unknown>')} "
                f"during global cancel: {exc}"
            )


def _task_history_details(raw_details):
    """Decode optional task details for the persistent history record."""
    if not raw_details:
        return None
    try:
        return json.loads(raw_details)
    except Exception:
        return None


def _task_history_duration(row, now_timestamp):
    """Calculate a non-negative task duration when a start time is available."""
    if row['start_time'] is None:
        return None
    end_timestamp = row['end_time'] if row['end_time'] is not None else now_timestamp
    return max(0.0, float(end_timestamp) - float(row['start_time']))


def _task_history_status(status):
    """Keep terminal states and mark every in-flight task as revoked."""
    terminal_statuses = (TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)
    return status if status in terminal_statuses else TASK_STATUS_REVOKED


def _record_task_snapshot(row, now_timestamp):
    """Persist one task-status row in task history before global deletion."""
    record_task_history(
        row['task_id'],
        row['task_type'],
        _task_history_status(row['status']),
        _task_history_duration(row, now_timestamp),
        details=_task_history_details(row['details']),
    )


def _snapshot_task_status(db):
    """Best-effort snapshot of top-level task status rows into task history."""
    try:
        with db.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(
                "SELECT task_id, task_type, status, details, start_time, end_time "
                "FROM task_status WHERE parent_task_id IS NULL"
            )
            now_timestamp = time.time()
            for row in cursor.fetchall():
                _record_task_snapshot(row, now_timestamp)
    except Exception as exc:
        logger.warning(
            f"Global cancel: failed snapshotting task_status into task_history: {exc}"
        )


def _clear_task_status():
    """Snapshot and delete task status rows as one best-effort database action."""
    db = get_db()
    cursor = db.cursor()
    try:
        _snapshot_task_status(db)
        cursor.execute("DELETE FROM task_status")
        deleted = cursor.rowcount
        db.commit()
        logger.info(f"Global cancel DB cleanup: deleted {deleted} task_status rows")
    except Exception:
        db.rollback()
        logger.exception("Error deleting task_status rows during global cancel")
    finally:
        cursor.close()


def _save_cancel_recap(job_id, reason):
    """Ensure the UI retains one canonical revoked task after global cleanup."""
    try:
        save_task_status(
            job_id,
            'unknown',
            TASK_STATUS_REVOKED,
            progress=100,
            details={"message": reason, "origin": "global_cancel"},
        )
    except Exception:
        logger.exception(f"Failed to insert REVOKED recap row for {job_id}")


def cancel_job_and_children_recursive(job_id, reason="Task cancellation processed by API."):
    """Cancel all active jobs and consolidate their task-status records."""
    job_ids = _discover_rq_job_ids()
    cancelled_count = _cancel_discovered_jobs(job_ids)
    _clear_rq_queues()
    _clear_task_status()
    _save_cancel_recap(job_id, reason)
    return cancelled_count
