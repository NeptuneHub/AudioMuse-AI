# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Setup wizard preview of clustering playlist naming, run on the default worker.

The web process only admits and enqueues a naming_preview job and reads its
task_status row back. The default-queue worker runs a quick K-Means clustering pass
over at most PREVIEW_MAX_SONGS analyzed songs and names every resulting playlist with
the naming style selected in the wizard, and for the full-title style with the
unsaved prompt. It writes the titles into its own row as it goes, so the wizard can
show them live. Nothing else is written: no media-server playlist, no playlist table
row, no playlist-name history.

Main Features:
* Only one batch runs at a time: the preview is admitted under the same start lock
  and blocking gate as every batch task, so it is refused while another task runs
  and every other batch start is refused while it runs. It never uses the high
  queue, which belongs to task parents and control work.
* It is a side job (task_types.SIDE_JOB_TASK_TYPES): it shows as the running task
  on the dashboard it blocks, but it stays out of the cancel history, never erases
  the last task recap and never records history. It is also refused while a
  server sweep runs, and a sweep is refused while it runs.
* A preview interrupted by a restart fails instead of silently re-running every AI
  call with an unsaved prompt, and progress writes only update a RUNNING row, so a
  late write can never resurrect a cancelled or deleted preview.
* Clustering follows a real run's calibration: the same stratified sample, the same
  K bound halved while too few playlists survive, the minimum-size filter and the
  diverse top-N selection, then each playlist is named through the same naming
  path and duplicate-suffix rule as a real run, starting from an empty used-title
  list, with the provider settings passed as the one ai_config dict that path
  reads. Every calibration attempt reuses the track rows the first one loaded.
* Expected conditions such as an empty library end as TaskFailed with a readable
  message, never a crash traceback.
"""

import json
import logging
import uuid

import config
import task_types
from taskqueue import WORKER_LOST_ERROR, TaskFailed
from taskqueue.sql import SWEEP_TASK_TYPE

logger = logging.getLogger(__name__)

PREVIEW_TASK_TYPE = task_types.NAMING_PREVIEW_TASK_TYPE
PREVIEW_TASK_FUNC = 'tasks.naming_preview.run_naming_preview_task'
PREVIEW_MAX_SONGS = 10000
PREVIEW_LOG_PREFIX = '[NamingPreview]'
PREVIEW_SAMPLE_SONGS = 3
PREVIEW_WAITING_MESSAGE = 'Waiting for a free worker...'
PREVIEW_FAILED_MESSAGE = 'The preview failed. Check the container logs.'
PREVIEW_CANCELLED_MESSAGE = 'The preview was cancelled.'
PREVIEW_RUNNING_MESSAGE = 'A title preview is already running. Wait for it to finish or stop it.'
PREVIEW_BUSY_MESSAGE = (
    'Another task is running ({task_type}). AudioMuse-AI never runs two batch tasks in '
    'parallel, so the preview cannot start until it finishes.'
)
PREVIEW_INTERRUPTED_MESSAGE = (
    'The preview was interrupted by a restart. Click Preview titles again.'
)


class PreviewUnavailable(TaskFailed):
    pass


def _initial_state(mode, message):
    return {
        'mode': mode,
        'message': message,
        'song_count': 0,
        'done': 0,
        'total': 0,
        'titles': [],
    }


def start_preview(mode, instructions):
    import database
    import taskqueue

    db = database.get_db()
    with database.main_task_start_lock(db):
        blocking = database.get_queue_blocking_task(conn=db) or database.get_active_main_task(
            task_type=SWEEP_TASK_TYPE, conn=db
        )
        if blocking:
            db.rollback()
            if blocking.get('task_type') in task_types.SIDE_JOB_TASK_TYPES:
                return None, PREVIEW_RUNNING_MESSAGE
            return None, PREVIEW_BUSY_MESSAGE.format(task_type=blocking.get('task_type'))
        with db.cursor() as cur:
            cur.execute(
                "DELETE FROM task_status WHERE task_type = %s AND status IN %s",
                (PREVIEW_TASK_TYPE, tuple(config.TASK_STATUS_TERMINAL)),
            )
        task_id = str(uuid.uuid4())
        taskqueue.enqueue(
            PREVIEW_TASK_FUNC,
            kwargs={'mode': mode, 'instructions': instructions},
            task_id=task_id,
            task_type=PREVIEW_TASK_TYPE,
            queue=taskqueue.QUEUE_DEFAULT,
            details=_initial_state(mode, PREVIEW_WAITING_MESSAGE),
            conn=db,
        )
        db.commit()
    return task_id, None


def preview_status():
    from psycopg2.extras import DictCursor

    from database import coerce_db_details, get_db

    with get_db().cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            "SELECT task_id, status, details FROM task_status WHERE task_type = %s "
            "ORDER BY timestamp DESC LIMIT 1",
            (PREVIEW_TASK_TYPE,),
        )
        row = cur.fetchone()
    if not row:
        return {'status': 'idle', 'message': '', 'titles': [], 'done': 0, 'total': 0}
    details = coerce_db_details(row['details']) or {}
    ui_status, message = _preview_ui_state(row['status'], details)
    titles = details.get('titles') if isinstance(details.get('titles'), list) else []
    return {
        'task_id': row['task_id'],
        'status': ui_status,
        'message': message,
        'mode': details.get('mode'),
        'song_count': details.get('song_count') or 0,
        'done': len(titles),
        'total': details.get('total') or 0,
        'titles': titles,
    }


def _preview_ui_state(status, details):
    if status == config.TASK_STATUS_NEW:
        return 'running', PREVIEW_WAITING_MESSAGE
    if status in config.TASK_STATUS_LIVE:
        return 'running', details.get('message') or PREVIEW_WAITING_MESSAGE
    if status == config.TASK_STATUS_SUCCESS:
        return 'done', details.get('preview_message') or details.get('message') or ''
    if status == config.TASK_STATUS_REVOKED:
        return 'failed', PREVIEW_CANCELLED_MESSAGE
    if details.get('preview_error'):
        return 'failed', details['preview_error']
    if _worker_was_lost(details.get('error')):
        return 'failed', PREVIEW_INTERRUPTED_MESSAGE
    return 'failed', PREVIEW_FAILED_MESSAGE


def _worker_was_lost(error):
    from error.error_dictionary import ERR_TASK_INTERRUPTED, ERR_WORKER_LOST

    if isinstance(error, dict):
        return error.get('error_code') in (ERR_WORKER_LOST, ERR_TASK_INTERRUPTED)
    return error == WORKER_LOST_ERROR


def _ai_config():
    return {
        'provider': (config.AI_MODEL_PROVIDER or 'NONE').upper(),
        'ollama_url': config.OLLAMA_SERVER_URL,
        'ollama_model': config.OLLAMA_MODEL_NAME,
        'openai_url': config.OPENAI_SERVER_URL,
        'openai_model': config.OPENAI_MODEL_NAME,
        'openai_key': config.OPENAI_API_KEY,
        'gemini_key': config.GEMINI_API_KEY,
        'gemini_model': config.GEMINI_MODEL_NAME,
        'mistral_key': config.MISTRAL_API_KEY,
        'mistral_model': config.MISTRAL_MODEL_NAME,
    }


def run_naming_preview_task(mode='concept', instructions=None):
    import taskqueue
    from flask_app import app
    from tasks.ai.prompts import normalize_naming_mode

    mode = normalize_naming_mode(mode)
    task_id = taskqueue.current_task_id()
    state = _initial_state(mode, 'Selecting songs from your library...')
    with app.app_context():
        if _already_started(task_id):
            _fail(task_id, state, PREVIEW_INTERRUPTED_MESSAGE)
        state['started'] = True
        _report(task_id, state)
        result = _cluster_sample(task_id, state)
        _name_playlists(task_id, state, result, mode, instructions)
    state['preview_message'] = 'Preview complete: %d playlist titles.' % state['total']
    state['message'] = state['preview_message']
    return dict(state)


def _already_started(task_id):
    if not task_id:
        return False
    from database import coerce_db_details, get_db

    db = get_db()
    try:
        with db.cursor() as cur:
            cur.execute("SELECT details FROM task_status WHERE task_id = %s", (task_id,))
            row = cur.fetchone()
        db.commit()
    except Exception:
        logger.exception('%s Could not read the preview row before starting', PREVIEW_LOG_PREFIX)
        _safe_rollback(db)
        return False
    details = coerce_db_details(row[0]) if row else {}
    return bool(isinstance(details, dict) and details.get('started'))


def _safe_rollback(db):
    try:
        db.rollback()
    except Exception:
        logger.exception('%s Rollback failed', PREVIEW_LOG_PREFIX)


def _report(task_id, state, **fields):
    state.update(fields)
    if not task_id:
        return
    from database import get_db

    total = state['total']
    progress = 10 if not total else 10 + int(85 * len(state['titles']) / total)
    db = get_db()
    try:
        with db.cursor() as cur:
            cur.execute(
                "UPDATE task_status SET details = %s, progress = %s, timestamp = NOW() "
                "WHERE task_id = %s AND status = %s",
                (json.dumps(state), progress, task_id, config.TASK_STATUS_RUNNING),
            )
        db.commit()
    except Exception:
        logger.exception('%s Could not record the preview progress', PREVIEW_LOG_PREFIX)
        _safe_rollback(db)


def _fail(task_id, state, message):
    _report(task_id, state, preview_error=message, message=message)
    raise PreviewUnavailable(message)


def preview_limits(song_count):
    playlist_songs = max(1, config.CLUSTERING_MAX_PLAYLIST_SONGS)
    min_size = max(1, config.MIN_PLAYLIST_SIZE_FOR_TOP_N)
    k_floor = max(2, song_count // playlist_songs)
    cap = max(k_floor, song_count // (2 * min_size))
    top_n = config.TOP_N_CLUSTERING_PLAYLIST
    target = top_n if top_n > 0 else cap
    clusters = max(2, config.NUM_CLUSTERS_MAX)
    if cap < clusters:
        clusters = max(k_floor, min(cap, max(2, target)))
    clusters = max(2, min(clusters, song_count - 1))
    needed = max(1, min(target, clusters))
    return clusters, needed


def _sample_item_ids():
    from psycopg2.extras import DictCursor

    from database import get_db
    from tasks.clustering import _calculate_target_songs_per_genre, _prepare_genre_map
    from tasks.clustering_helper import _get_stratified_song_subset

    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    try:
        cur.execute(
            "SELECT item_id, mood_vector FROM score "
            "WHERE mood_vector IS NOT NULL AND mood_vector != ''"
        )
        rows = cur.fetchall()
    finally:
        cur.close()
    db.commit()
    genre_map = _prepare_genre_map(rows)
    del rows
    target = _calculate_target_songs_per_genre(
        genre_map,
        config.STRATIFIED_SAMPLING_TARGET_PERCENTILE,
        config.MIN_SONGS_PER_GENRE_FOR_STRATIFICATION,
    )
    item_ids = [track['item_id'] for track in _get_stratified_song_subset(genre_map, target)]
    if len(item_ids) > PREVIEW_MAX_SONGS:
        item_ids = item_ids[:PREVIEW_MAX_SONGS]
    return item_ids, genre_map


def _cluster_once(item_ids, clusters, tracks_cache=None):
    from tasks.clustering_helper import _perform_single_clustering_iteration

    top_n_moods = config.TOP_N_MOODS
    return _perform_single_clustering_iteration(
        run_idx=1,
        item_ids_for_subset=item_ids,
        clustering_method='kmeans',
        num_clusters_min_max=(clusters, clusters),
        dbscan_params_ranges={
            'eps_min': config.DBSCAN_EPS_MIN,
            'eps_max': config.DBSCAN_EPS_MAX,
            'samples_min': config.DBSCAN_MIN_SAMPLES_MIN,
            'samples_max': config.DBSCAN_MIN_SAMPLES_MAX,
        },
        gmm_params_ranges={'n_components_min': 2, 'n_components_max': 2},
        spectral_params_ranges={'n_clusters_min': 2, 'n_clusters_max': 2},
        pca_params_ranges={
            'components_min': config.PCA_COMPONENTS_MIN,
            'components_max': config.PCA_COMPONENTS_MAX,
        },
        active_mood_labels=(
            config.MOOD_LABELS[:top_n_moods] if top_n_moods > 0 else config.MOOD_LABELS
        ),
        max_songs_per_cluster=config.MAX_SONGS_PER_CLUSTER,
        log_prefix=PREVIEW_LOG_PREFIX,
        elite_solutions_params_list=[],
        exploitation_probability=0.0,
        mutation_config={
            'int_abs_delta': 0, 'float_abs_delta': 0.0, 'coord_mutation_fraction': 0.0,
        },
        score_weights={
            'mood_diversity': 0.0, 'silhouette': 0.0, 'davies_bouldin': 0.0,
            'calinski_harabasz': 0.0, 'mood_purity': 0.0,
            'other_feature_diversity': 0.0, 'other_feature_purity': 0.0,
        },
        enable_clustering_embeddings=config.ENABLE_CLUSTERING_EMBEDDINGS,
        tracks_cache=tracks_cache,
    )


def _keepers(result):
    min_size = max(1, config.MIN_PLAYLIST_SIZE_FOR_TOP_N)
    return sum(
        1 for songs in (result or {}).get('named_playlists', {}).values()
        if len(songs) >= min_size
    )


def _cluster_sample(task_id, state):
    from tasks.clustering_postprocessing import (
        apply_minimum_size_filter_to_clustering_result,
        select_diverse_playlists_with_genre_coverage,
    )

    item_ids, genre_map = _sample_item_ids()
    if len(item_ids) < max(2, config.MIN_PLAYLIST_SIZE_FOR_TOP_N):
        _fail(task_id, state, 'Not enough analyzed songs to preview. Run an analysis first.')

    clusters, needed = preview_limits(len(item_ids))
    best_result, best_keepers = None, -1
    tracks_cache = {}
    for attempt in range(max(1, config.CLUSTERING_CALIBRATION_MAX_TRIES)):
        _report(
            task_id,
            state,
            song_count=len(item_ids),
            message='Clustering %d songs into %d groups with K-Means...' % (len(item_ids), clusters),
        )
        result = _cluster_once(item_ids, clusters, tracks_cache)
        keepers = _keepers(result)
        if keepers > best_keepers:
            best_result, best_keepers = result, keepers
        next_clusters = max(2, needed, clusters // 2)
        if keepers >= needed or next_clusters >= clusters:
            break
        clusters = next_clusters
        logger.info(
            '%s Attempt %d kept %d of %d needed playlists; retrying with K=%d',
            PREVIEW_LOG_PREFIX, attempt + 1, keepers, needed, clusters,
        )

    if not best_result or not best_result.get('named_playlists'):
        _fail(task_id, state, 'The clustering run produced no playlists.')

    result = apply_minimum_size_filter_to_clustering_result(
        best_result, config.MIN_PLAYLIST_SIZE_FOR_TOP_N, log_prefix=PREVIEW_LOG_PREFIX + ' '
    )
    if config.TOP_N_CLUSTERING_PLAYLIST > 0:
        result = select_diverse_playlists_with_genre_coverage(
            result,
            config.TOP_N_CLUSTERING_PLAYLIST,
            primary_genre_counts={
                genre: len(tracks)
                for genre, tracks in genre_map.items()
                if genre != '__other__'
            },
        )
    if not result or not result.get('named_playlists'):
        _fail(task_id, state, 'No playlist was large enough to keep.')
    return result


def _name_playlists(task_id, state, result, mode, instructions):
    from database import get_db
    from tasks.clustering_helper import _name_playlist_with_ai_config

    playlists = [
        (name, songs) for name, songs in result.get('named_playlists', {}).items() if songs
    ]
    _report(
        task_id,
        state,
        total=len(playlists),
        message='Naming %d playlists with your AI provider...' % len(playlists),
    )
    centroids = result.get('playlist_centroids', {})
    primary_genres = result.get('playlist_primary_genres', {})
    ai_config = _ai_config()
    used_names = []
    assigned = set()
    for original_name, songs in playlists:
        try:
            final_name = _name_playlist_with_ai_config(
                original_name,
                songs,
                centroids,
                ai_config,
                used_names,
                primary_genre=primary_genres.get(original_name),
                naming_mode=mode,
                title_prompt=instructions,
            )
        except Exception:
            logger.exception("%s Could not name '%s'", PREVIEW_LOG_PREFIX, original_name)
            _safe_rollback(get_db())
            final_name = original_name
        candidate = final_name
        suffix = 1
        while candidate in assigned:
            suffix += 1
            candidate = '%s (%d)' % (final_name, suffix)
        assigned.add(candidate)
        used_names.append(candidate)
        state['titles'].append({
            'title': candidate,
            'tag_name_kept': final_name == original_name,
            'song_count': len(songs),
            'sample': [
                '%s - %s' % (title or 'Unknown Title', author or 'Unknown Artist')
                for _item_id, title, author in songs[:PREVIEW_SAMPLE_SONGS]
            ],
        })
        state['done'] = len(state['titles'])
        _report(task_id, state)
