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
re-exports the most-used ``database`` handles so the many modules doing
``from app_helper import get_db, save_task_status, ...`` stay untouched.

Main Features:
* ``cancel_job_and_children_recursive`` tombstones every task row and notifies
  every worker; ``revoke_inline_task_row`` handles the tasks that run in the web
  process, which have no worker to signal and only need their own row revoked.
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
import taskqueue
from taskqueue.sql import CONTROL_TASK_TYPE
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
from config import (  # noqa: F401
    STRATIFIED_GENRES,
    OTHER_FEATURE_LABELS,
    TASK_STATUS_NEW,
    TASK_STATUS_RUNNING,
    TASK_STATUS_PENDING,
    TASK_STATUS_STARTED,
    TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAIL,
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

    # The main clustering task's own best_result is kept in full in the database (a
    # worker restart resumes from it, and it is what postprocessing turns into
    # playlists at the end of the run: named_playlists is the per-song composition of
    # every candidate playlist, playlist_centroids and playlist_to_centroid_vector_map
    # are per-playlist embedding vectors, and playlist_primary_genres feeds the
    # diverse-genre-coverage selection - none of it is display data. The top-level
    # best_score and elite_solutions already carry everything the UI shows about the
    # best result found so far, so best_result is dropped here entirely rather than
    # picked apart key by key, matching how this task never persisted it at all before
    # the queue moved onto Postgres. /api/active_tasks is polled every 3 seconds while
    # clustering runs, so leaving it in made every poll re-read, re-decode and
    # re-serialize a payload that grows with the catalogue and the candidate playlist
    # count - large enough to make the polling page feel unresponsive for nothing ever
    # shown on screen.
    details.pop('best_result', None)

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
    if not isinstance(log_entries, list) or not log_entries:
        recap = details.get('status_message') or details.get('message')
        if recap:
            details['log'] = [str(recap)]
    elif len(log_entries) > 10:
        details['log'] = [
            f"... ({len(log_entries) - 10} earlier log entries truncated)",
            *log_entries[-10:],
        ]

    if str(state or '').upper() in (TASK_STATUS_FAIL, 'FAILED', 'FAILURE'):
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


def top_clap_mood(other_features):
    # Dominant CLAP mood label from the stored other_features (danceable /
    # aggressive / happy / party / relaxed / sad), mirroring how top genre is
    # derived from mood_vector. Returns None when no mood is present so callers
    # can skip the mood entirely instead of inventing one.
    if not other_features or not isinstance(other_features, str):
        return None
    scores = {}
    for part in other_features.split(','):
        label, _, value = part.partition(':')
        label = label.strip()
        if not label:
            continue
        try:
            scores[label] = float(value)
        except ValueError:
            continue
    candidates = [m for m in OTHER_FEATURE_LABELS if m in scores]
    if not candidates:
        return None
    return max(candidates, key=scores.get)


def attach_song_features(rows, id_key='item_id'):
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
            r.setdefault('top_mood', top_clap_mood(s.get('other_features')))
    return rows


def serialize_neighbor_results(
    neighbor_results, missing_album='unknown', include_album_artist=True
):
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
            "top_mood": top_clap_mood(info.get('other_features')),
        }
        if include_album_artist:
            row["album_artist"] = info.get('album_artist') or 'unknown'
        out.append(row)
    return out


def _project_matrix_2d(mat, label):
    from tasks.alchemy_projections import _project_with_umap, _project_to_2d, UMAP_MIN_SAMPLES

    projections = None
    if mat.shape[0] < UMAP_MIN_SAMPLES:
        logger.info(
            f"Skipping UMAP for {label}: only {mat.shape[0]} vector(s) found "
            f"(need at least {UMAP_MIN_SAMPLES}); using PCA instead."
        )
    else:
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
    db = get_db()
    snapshots = []
    protected_task_ids = set()
    now_ts = time.time()
    recap_type = 'unknown'
    cur = db.cursor()
    try:
        # Before the snapshot, so a Start whose INSERT has not committed yet
        # cannot slip past. taskqueue.enqueue takes this same transaction lock,
        # so either that INSERT lands first and the DELETE below removes it, or
        # it waits here and is a genuinely later start. Without it the row was
        # invisible to the wipe and survived it, and a full analysis began
        # seconds after the user pressed Cancel.
        taskqueue.take_start_lock(conn=db)
        with db.cursor(cursor_factory=DictCursor) as snap_cur:
            snap_cur.execute(
                "SELECT task_id, task_type, status, details, start_time, end_time "
                "FROM task_status WHERE parent_task_id IS NULL"
            )
            snapshots = list(snap_cur.fetchall())
            # Once provider repointing committed, its restart handshake and staged
            # alignment are recovery obligations, not cancellable work. Removing
            # either row opens the main-task gate before workers acknowledge the
            # new provider and can permanently lose the post-migration alignment.
            snap_cur.execute(
                """
                SELECT ms.state->>'exec_task_id', ms.state->>'alignment_task_id'
                FROM migration_session AS ms
                WHERE ms.status = 'completed'
                  AND lower(COALESCE(ms.state->>'restart_acknowledged', 'false'))
                      NOT IN ('true', '1', 'yes')
                """
            )
            for protected in snap_cur.fetchall():
                protected_task_ids.update(task_id for task_id in protected if task_id)
            # A worker_control request is a restart handshake somebody is waiting
            # on right now, not user work. Deleting it - and the acknowledgement
            # children it is counting - made a concurrent "save settings" report a
            # timeout even though the workers had restarted.
            snap_cur.execute(
                "SELECT task_id FROM task_status "
                "WHERE task_type = %s AND status IN %s",
                (CONTROL_TASK_TYPE, (TASK_STATUS_NEW, TASK_STATUS_RUNNING)),
            )
            protected_task_ids.update(row[0] for row in snap_cur.fetchall())
        cancelled_row_status = None
        for row in snapshots:
            if row['task_id'] == job_id:
                recap_type = row['task_type'] or 'unknown'
                cancelled_row_status = row['status']
                break
        recap_details = json.dumps({
            "message": reason,
            "status_message": reason,
            "origin": "global_cancel",
        })
        if protected_task_ids:
            protected_list = list(protected_task_ids)
            cur.execute(
                # parent_task_id IS NULL on every root row, and NOT (NULL = ANY())
                # is NULL rather than TRUE - so the unguarded form deleted NO roots
                # at all, and the recap INSERT below then hit the task_id UNIQUE
                # constraint and aborted the entire cancellation.
                "DELETE FROM task_status "
                "WHERE NOT (task_id = ANY(%s)) "
                "AND (parent_task_id IS NULL OR NOT (parent_task_id = ANY(%s)))",
                (protected_list, protected_list),
            )
        else:
            cur.execute("DELETE FROM task_status")
        deleted = cur.rowcount
        # The wipe is GLOBAL but a recap used to be written for job_id alone. Any
        # OTHER parentless root deleted here (a concurrent sonic-fingerprint or
        # plugin root, or a cancel aimed at a child/stale id) then had no tombstone:
        # its in-flight parentless RUNNING save re-inserted a phantom row that no
        # worker owns, no reclaim touches, and only the 30-minute stale sweep
        # reaps. Recap every LIVE deleted parentless root so the ON CONFLICT gate
        # on each one discards a late save instead of resurrecting it. A root that
        # was already terminal has no in-flight writer to gate, and a REVOKED
        # tombstone over it reported a completed run as cancelled with another
        # task's reason - so terminal rows are deleted without a recap.
        recap_targets = {
            row['task_id']: (row['task_type'] or 'unknown')
            for row in snapshots
            if row['task_id'] not in protected_task_ids
            and row['status'] in (TASK_STATUS_NEW, TASK_STATUS_RUNNING)
        }
        # job_id may not be in the snapshot (a cancel aimed at a stale or child
        # id); it still gets its recap so the endpoint's own row is terminal -
        # unless the aimed row provably finished already, which no cancel rewrites.
        if job_id not in protected_task_ids and cancelled_row_status in (
            None, TASK_STATUS_NEW, TASK_STATUS_RUNNING
        ):
            recap_targets.setdefault(job_id, recap_type)
        for recap_task_id, recap_task_type in recap_targets.items():
            cur.execute(
                """
                INSERT INTO task_status
                    (task_id, task_type, status, progress, details, timestamp,
                     start_time, end_time)
                VALUES (%s, %s, %s, 100, %s, NOW(), %s, %s)
                -- A second Cancel (a double-click, or two tabs) finds the recap
                -- row this one just wrote and raised UniqueViolation on task_id,
                -- so the endpoint answered 503 AFTER the wipe had already
                -- committed - the cancel had worked and the UI said it failed.
                ON CONFLICT (task_id) DO UPDATE SET
                    task_type = EXCLUDED.task_type,
                    status = EXCLUDED.status,
                    progress = 100,
                    details = EXCLUDED.details,
                    timestamp = NOW(),
                    end_time = EXCLUDED.end_time
                """,
                (
                    recap_task_id,
                    recap_task_type,
                    TASK_STATUS_REVOKED,
                    recap_details,
                    now_ts,
                    now_ts,
                ),
            )
        # The provider-migration endpoints read this counter around their claim
        # so a Cancel landing mid-claim invalidates the reservation instead of
        # queueing a job into a table this transaction just wiped. Bumping it
        # here is what makes that guard real: without a writer it never changed,
        # and the comparison could not fail no matter what the user did.
        database.bump_global_cancel_epoch(conn=db)
        # Inside the transaction on purpose: Postgres delivers a NOTIFY only when
        # its transaction commits, so a worker can never be told to stop for a
        # cancellation that then rolled back.
        taskqueue.request_cancel_all(conn=db)
        db.commit()
        logger.info(
            "Global cancel committed; replaced %d task row(s) and signalled every worker.",
            deleted,
        )
    except Exception:
        db.rollback()
        logger.exception(
            "Global cancel could not commit its tombstone; refusing a false success"
        )
        raise
    finally:
        cur.close()

    _record_cancel_history(snapshots, protected_task_ids, now_ts, reason)
    return len(snapshots)


def _record_one_cancellation(row, now_ts, reason):
    duration = None
    if row['start_time'] is not None:
        end = row['end_time'] if row['end_time'] is not None else now_ts
        duration = max(0.0, float(end) - float(row['start_time']))
    already_terminal = row['status'] in (
        TASK_STATUS_SUCCESS, TASK_STATUS_FAIL, TASK_STATUS_REVOKED
    )
    details = coerce_db_details(row['details']) if already_terminal else None
    database.record_task_history(
        row['task_id'],
        row['task_type'],
        row['status'] if already_terminal else TASK_STATUS_REVOKED,
        duration_seconds=duration,
        note=None if already_terminal else reason,
        details=details,
    )


def _record_cancel_history(snapshots, protected_task_ids, now_ts, reason):
    try:
        for row in snapshots:
            if row['task_id'] in protected_task_ids:
                continue
            if row['task_type'] in (CONTROL_TASK_TYPE, 'provider_migration_planner'):
                continue
            _record_one_cancellation(row, now_ts, reason)
    except Exception:
        logger.exception("Could not record cancellation history; the cancel itself stands")
