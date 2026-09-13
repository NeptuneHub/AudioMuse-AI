# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Setup wizard preview of clustering playlist naming, run on a worker.

The web process only enqueues a naming_preview job and reads its task_status row
back. The worker runs one quick K-Means clustering pass over at most
PREVIEW_MAX_SONGS analyzed songs and names every resulting playlist with the naming
style selected in the wizard, and for the full-title style with the unsaved prompt.
It writes the titles into its own row as it goes, so the wizard can show them live.
Nothing else is written: no media-server playlist, no playlist table row, no
playlist-name history.

Main Features:
* The heavy work never touches the web process: Flask runs one SELECT and one
  enqueue, and the clustering, scikit-learn and AI calls all happen on a worker.
* The job is enqueued with clear_finished=False and its finish is exempt from the
  recap collapse and from task history, so a preview never erases the recap of the
  last real task nor shows up as a run. Its type is non-blocking, so it never stops
  an analysis or clustering from starting, and the dashboard hides it.
* Clustering follows a real run's calibration pass: the same stratified sample, the
  same K bound, the minimum-size filter and the diverse top-N selection, then names
  each playlist through the same _try_ai_name_playlist call and duplicate-suffix rule
  as a real run, with the naming history empty as in a default run.
* Only one preview is live at a time, and the wizard sees a generic message for any
  failure other than the expected "not enough songs" cases.
"""

import logging
import random
import uuid

import config
import task_types

logger = logging.getLogger(__name__)

PREVIEW_TASK_TYPE = task_types.NAMING_PREVIEW_TASK_TYPE
PREVIEW_TASK_FUNC = 'tasks.naming_preview.run_naming_preview_task'
PREVIEW_MAX_SONGS = 10000
PREVIEW_LOG_PREFIX = '[NamingPreview]'
PREVIEW_SAMPLE_SONGS = 3
PREVIEW_WAITING_MESSAGE = 'Waiting for a free worker...'
PREVIEW_FAILED_MESSAGE = 'The preview failed. Check the container logs.'
PREVIEW_CANCELLED_MESSAGE = 'The preview was cancelled.'


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
    import taskqueue
    from database import get_db

    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM task_status WHERE task_type = %s AND status IN %s LIMIT 1",
            (PREVIEW_TASK_TYPE, tuple(config.TASK_STATUS_LIVE)),
        )
        if cur.fetchone():
            db.rollback()
            return None
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
        queue=taskqueue.QUEUE_HIGH,
        max_attempts=0,
        details=_initial_state(mode, PREVIEW_WAITING_MESSAGE),
        conn=db,
        clear_finished=False,
    )
    db.commit()
    return task_id


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
    status = row['status']
    if status in config.TASK_STATUS_LIVE:
        ui_status = 'running'
        message = details.get('message') or PREVIEW_WAITING_MESSAGE
        if status == config.TASK_STATUS_NEW:
            message = PREVIEW_WAITING_MESSAGE
    elif status == config.TASK_STATUS_SUCCESS:
        ui_status = 'done'
        message = details.get('preview_message') or details.get('message') or ''
    else:
        ui_status = 'failed'
        if status == config.TASK_STATUS_REVOKED:
            message = PREVIEW_CANCELLED_MESSAGE
        else:
            message = details.get('preview_error') or PREVIEW_FAILED_MESSAGE
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


def worker_ai_config():
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


class PreviewUnavailable(RuntimeError):
    pass


def run_naming_preview_task(mode='concept', instructions=None):
    import taskqueue
    from tasks.ai.prompts import normalize_naming_mode

    mode = normalize_naming_mode(mode)
    task_id = taskqueue.current_task_id()
    state = _initial_state(mode, 'Selecting songs from your library...')
    _report(task_id, state)
    result = _cluster_sample(task_id, state)
    _name_playlists(task_id, state, result, mode, instructions, worker_ai_config())
    state['preview_message'] = 'Preview complete: %d playlist titles.' % state['total']
    state['message'] = state['preview_message']
    return dict(state)


def _report(task_id, state, **fields):
    state.update(fields)
    if not task_id:
        return
    try:
        from database import save_task_status
        from flask_app import app

        total = state['total']
        progress = 10 if not total else 10 + int(85 * len(state['titles']) / total)
        with app.app_context():
            save_task_status(
                task_id,
                PREVIEW_TASK_TYPE,
                config.TASK_STATUS_RUNNING,
                progress=progress,
                details=dict(state),
            )
    except Exception:
        logger.exception('%s Could not record the preview progress', PREVIEW_LOG_PREFIX)


def _fail(task_id, state, message):
    _report(task_id, state, preview_error=message, message=message)
    raise PreviewUnavailable(message)


def preview_cluster_count(song_count):
    k_floor = max(2, song_count // config.CLUSTERING_MAX_PLAYLIST_SONGS)
    cap = max(k_floor, song_count // (2 * config.MIN_PLAYLIST_SIZE_FOR_TOP_N))
    top_n = config.TOP_N_CLUSTERING_PLAYLIST
    target = top_n if top_n > 0 else cap
    clusters = config.NUM_CLUSTERS_MAX
    if cap < clusters:
        clusters = max(k_floor, min(cap, max(2, target)))
    return max(2, min(clusters, song_count - 1))


def _sample_item_ids():
    from psycopg2.extras import DictCursor

    from database import get_db
    from tasks.clustering import _calculate_target_songs_per_genre, _prepare_genre_map
    from tasks.clustering_helper import _get_stratified_song_subset

    cur = get_db().cursor(cursor_factory=DictCursor)
    try:
        cur.execute(
            "SELECT item_id, mood_vector FROM score "
            "WHERE mood_vector IS NOT NULL AND mood_vector != ''"
        )
        rows = cur.fetchall()
    finally:
        cur.close()
    genre_map = _prepare_genre_map(rows)
    del rows
    target = _calculate_target_songs_per_genre(
        genre_map,
        config.STRATIFIED_SAMPLING_TARGET_PERCENTILE,
        config.MIN_SONGS_PER_GENRE_FOR_STRATIFICATION,
    )
    item_ids = [track['item_id'] for track in _get_stratified_song_subset(genre_map, target)]
    if len(item_ids) > PREVIEW_MAX_SONGS:
        item_ids = random.sample(item_ids, PREVIEW_MAX_SONGS)
    return item_ids, genre_map


def _cluster_sample(task_id, state):
    from flask_app import app
    from tasks.clustering_helper import _perform_single_clustering_iteration
    from tasks.clustering_postprocessing import (
        apply_minimum_size_filter_to_clustering_result,
        select_diverse_playlists_with_genre_coverage,
    )

    with app.app_context():
        item_ids, genre_map = _sample_item_ids()
    if len(item_ids) < max(2, config.MIN_PLAYLIST_SIZE_FOR_TOP_N):
        _fail(task_id, state, 'Not enough analyzed songs to preview. Run an analysis first.')

    clusters = preview_cluster_count(len(item_ids))
    _report(
        task_id,
        state,
        song_count=len(item_ids),
        message='Clustering %d songs into %d groups with K-Means...' % (len(item_ids), clusters),
    )
    top_n_moods = config.TOP_N_MOODS
    result = _perform_single_clustering_iteration(
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
    )
    if not result or not result.get('named_playlists'):
        _fail(task_id, state, 'The clustering run produced no playlists.')

    result = apply_minimum_size_filter_to_clustering_result(
        result, config.MIN_PLAYLIST_SIZE_FOR_TOP_N, log_prefix=PREVIEW_LOG_PREFIX + ' '
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


def _name_playlists(task_id, state, result, mode, instructions, ai_config):
    from flask_app import app
    from tasks.clustering_helper import _try_ai_name_playlist

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
    used_names = []
    assigned = set()
    for original_name, songs in playlists:
        try:
            with app.app_context():
                final_name = _try_ai_name_playlist(
                    original_name,
                    songs,
                    centroids,
                    ai_config['provider'],
                    ai_config['ollama_url'],
                    ai_config['ollama_model'],
                    ai_config['openai_url'],
                    ai_config['openai_model'],
                    ai_config['openai_key'],
                    ai_config['gemini_key'],
                    ai_config['gemini_model'],
                    ai_config['mistral_key'],
                    ai_config['mistral_model'],
                    used_names,
                    primary_genre=primary_genres.get(original_name),
                    naming_mode=mode,
                    title_prompt=instructions,
                )
        except Exception:
            logger.exception("%s Could not name '%s'", PREVIEW_LOG_PREFIX, original_name)
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
