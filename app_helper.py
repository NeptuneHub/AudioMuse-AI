"""App-layer helpers that compose the data (``database``) and queue
(``taskqueue``) layers for the web and task tiers.

This is NOT the database layer -- all SQL lives in ``database.py``. What remains
here is orchestration and presentation glue:

- ``cancel_job_and_children_recursive`` -- recursively cancel an RQ job tree.
- ``build_and_store_map_projection`` / ``build_and_store_artist_projection`` --
  compute a 2D projection and persist it via ``database``.
- ``attach_song_features`` / ``top_stratified_genre`` -- enrich API result rows.

It also re-exports the most commonly used ``database`` / ``taskqueue`` handles so
the many modules doing ``from app_helper import get_db, redis_conn, ...`` are
untouched.
"""
import json
import logging
import time

from psycopg2.extras import DictCursor
import numpy as np

import database
from database import (  # noqa: F401
    get_db, close_db, save_task_status, record_task_history, _build_task_note,
    get_score_data_by_ids, load_map_projection, get_task_info_from_db, get_tracks_by_ids,
    save_track_analysis_and_embedding,
    # Used internally by the build_and_store_* projection orchestration below.
    save_map_projection, save_artist_projection,
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
    TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED,
)

logger = logging.getLogger(__name__)


# The Flask `app` object is intentionally NOT imported here (circular import);
# use the module-level `logger` above. The 2D map/artist projection caches live
# in database.MAP_PROJECTION_CACHE / database.ARTIST_PROJECTION_CACHE, written by
# the build_and_store_* helpers below and read by database.load_*_projection.


def get_score_data_lite_by_ids(item_ids_list):
    """
    Slim version of get_score_data_by_ids — drops the large text columns
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
    except Exception as e:
        logger.error(f"Error fetching lite score data by IDs: {e}")
        rows = []
    finally:
        cur.close()
    return [dict(row) for row in rows]


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
    except Exception as e:
        logger.error(f"Failed to stream embeddings for map projection: {e}", exc_info=True)
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
        database.MAP_PROJECTION_CACHE = {'index_name': index_name, 'id_map': ids, 'projection': projections}
        # Note: Caller (analysis task) is responsible for publishing reload message after all builds complete
        return True
    except Exception as e:
        logger.error(f"Failed to build and store map projection: {e}")
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
            component_map.append({
                'artist_id': artist_id,
                'artist_name': artist_name,
                'component_idx': comp_idx,
                'weight': float(weights[comp_idx]) if comp_idx < len(weights) else 0.0,
            })
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
        database.ARTIST_PROJECTION_CACHE = {'index_name': index_name, 'component_map': component_map, 'projection': projections}
        # Note: Caller (analysis task) is responsible for publishing reload message after all builds complete
        return True
    except Exception as e:
        logger.error(f"Failed to build and store artist projection: {e}")
        return False



def cancel_job_and_children_recursive(job_id, task_type_from_db=None, reason="Task cancellation processed by API."):
    """Helper to cancel a job and its children based on DB records.

    NOTE: Minimal global behavior — when invoked from the API cancel endpoint we clear RQ queues,
    attempt to stop all jobs known to RQ, delete all rows in `task_status`, and insert a single
    REVOKED row for the requested `job_id` (so UI sees one canonical cancelled task).
    This keeps the function signature unchanged and is intentionally simple and destructive (as requested).
    """
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
        raw_keys = redis_conn.keys('rq:job:*')
        for k in raw_keys:
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
                if not j.is_finished and not j.is_failed and not j.is_canceled:
                    if j.is_started:
                        send_stop_job_command(redis_conn, jid)
                    else:
                        j.cancel()
                    cancelled_count += 1
                    logger.info(f"Sent stop/cancel for job {jid} during global cancel")
            except NoSuchJobError:
                logger.debug(f"Job {jid} not found in RQ during global cancel")
        except Exception as e_j:
            logger.error(f"Error cancelling job {jid} during global cancel: {e_j}")

    # Try to clear the RQ queues using API (preferred) and fallback to key deletion if necessary
    try:
        for q in (rq_queue_high, rq_queue_default):
            try:
                if hasattr(q, 'empty'):
                    q.empty()
                    logger.info(f"Emptied queue {getattr(q, 'name', '<unknown>')} via Queue.empty() as part of global cancel")
                else:
                    key = f"rq:queue:{getattr(q, 'name', '')}"
                    redis_conn.delete(key)
                    logger.info(f"Deleted Redis key fallback for queue: {key} as part of global cancel")
            except Exception as e_q:
                logger.warning(f"Failed to empty queue {getattr(q, 'name', '<unknown>')} during global cancel: {e_q}")
    except Exception as e_qdel:
        logger.warning(f'Failed to clear queue lists during global cancel: {e_qdel}')

    # Consolidate DB: delete all task_status rows and insert a single REVOKED row for job_id
    db = get_db()
    cur = db.cursor()
    try:
        # Snapshot the in-flight main tasks into the persistent task_history
        # *before* we wipe task_status, so the dashboard's history table keeps
        # showing what was running when the user pressed Cancel.
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
                    # If the task was already in a terminal status, keep that one;
                    # otherwise mark it REVOKED.
                    final_status = r['status'] if r['status'] in (
                        TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED
                    ) else TASK_STATUS_REVOKED
                    record_task_history(
                        r['task_id'], r['task_type'], final_status,
                        duration_s, details=details_obj,
                    )
        except Exception as e_snap:
            logger.warning(f"Global cancel: failed snapshotting task_status into task_history: {e_snap}")

        cur.execute("DELETE FROM task_status")
        deleted = cur.rowcount
        db.commit()
        logger.info(f"Global cancel DB cleanup: deleted {deleted} task_status rows")
    except Exception as e_dbdel:
        db.rollback()
        logger.error(f"Error deleting task_status rows during global cancel: {e_dbdel}")
    finally:
        cur.close()

    try:
        # Ensure a single REVOKED row exists for job_id
        save_task_status(job_id, 'unknown', TASK_STATUS_REVOKED, progress=100, details={"message": reason, "origin": "global_cancel"})
    except Exception as e_save:
        logger.error(f"Failed to insert REVOKED recap row for {job_id}: {e_save}")

    return cancelled_count
