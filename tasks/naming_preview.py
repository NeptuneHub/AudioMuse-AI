# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Setup wizard preview of the full-AI-title playlist naming.

Runs one quick K-Means clustering pass over at most PREVIEW_MAX_SONGS analyzed songs
and names every resulting playlist with the unsaved title prompt, so an admin can read
all the titles a prompt produces before saving it. Nothing is written anywhere: no
playlist, no playlist-name history, no task_status row and no queue job, so the
dashboard recap and the global task start lock are never touched.

Main Features:
* One preview at a time on a daemon thread, polled by the wizard. The state lives in
  memory and dies with the web process, which the wizard restarts on save anyway.
* scikit-learn and the clustering modules are imported only when a preview starts, so
  a web process that never previews carries none of their memory.
* Clustering follows a real run's calibration pass: the same stratified sample, the
  same K bound, the minimum-size filter and the diverse top-N selection, then names
  each playlist through get_ai_playlist_title exactly as a real run in title mode does.
"""

import logging
import random
import threading
import time
import uuid

import config

logger = logging.getLogger(__name__)

PREVIEW_MAX_SONGS = 10000
PREVIEW_LOG_PREFIX = '[NamingPreview]'
PREVIEW_SAMPLE_SONGS = 3

_LOCK = threading.Lock()
_STATE = {
    'id': None,
    'status': 'idle',
    'message': '',
    'song_count': 0,
    'done': 0,
    'total': 0,
    'titles': [],
    'started_at': None,
    'finished_at': None,
}


def preview_status():
    with _LOCK:
        snapshot = dict(_STATE)
        snapshot['titles'] = list(_STATE['titles'])
    return snapshot


def start_preview(instructions, ai_config):
    with _LOCK:
        if _STATE['status'] == 'running':
            return False
        preview_id = uuid.uuid4().hex
        _STATE.update(
            id=preview_id,
            status='running',
            message='Selecting songs from your library...',
            song_count=0,
            done=0,
            total=0,
            titles=[],
            started_at=time.time(),
            finished_at=None,
        )
    worker = threading.Thread(
        target=_run,
        args=(preview_id, instructions, ai_config),
        name='naming-preview',
        daemon=True,
    )
    worker.start()
    return True


def _update(preview_id, **fields):
    with _LOCK:
        if _STATE['id'] == preview_id:
            _STATE.update(fields)


def _append_title(preview_id, entry):
    with _LOCK:
        if _STATE['id'] == preview_id:
            _STATE['titles'].append(entry)
            _STATE['done'] = len(_STATE['titles'])


def _finish(preview_id, status, message):
    _update(preview_id, status=status, message=message, finished_at=time.time())


def _run(preview_id, instructions, ai_config):
    try:
        from flask_app import app

        with app.app_context():
            result = _cluster_sample(preview_id)
        if result is None:
            return
        _name_playlists(preview_id, result, instructions, ai_config)
        with _LOCK:
            total = _STATE['total']
        _finish(preview_id, 'done', 'Preview complete: %d playlist titles.' % total)
    except Exception:
        logger.exception('%s The playlist naming preview failed', PREVIEW_LOG_PREFIX)
        _finish(preview_id, 'failed', 'The preview failed. Check the container logs.')


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


def _cluster_sample(preview_id):
    from tasks.clustering_helper import _perform_single_clustering_iteration
    from tasks.clustering_postprocessing import (
        apply_minimum_size_filter_to_clustering_result,
        select_diverse_playlists_with_genre_coverage,
    )

    item_ids, genre_map = _sample_item_ids()
    if len(item_ids) < max(2, config.MIN_PLAYLIST_SIZE_FOR_TOP_N):
        _finish(
            preview_id,
            'failed',
            'Not enough analyzed songs to preview. Run an analysis first.',
        )
        return None

    clusters = preview_cluster_count(len(item_ids))
    _update(
        preview_id,
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
        _finish(preview_id, 'failed', 'The clustering run produced no playlists.')
        return None

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
        _finish(preview_id, 'failed', 'No playlist was large enough to keep.')
        return None
    return result


def _name_playlists(preview_id, result, instructions, ai_config):
    from tasks.ai.api import get_ai_playlist_title

    playlists = [
        (name, songs) for name, songs in result.get('named_playlists', {}).items() if songs
    ]
    _update(
        preview_id,
        total=len(playlists),
        message='Naming %d playlists with your AI provider...' % len(playlists),
    )
    assigned = set()
    for original_name, songs in playlists:
        try:
            ai_title = get_ai_playlist_title(instructions, songs, ai_config)
        except Exception:
            logger.exception("%s Could not name '%s'", PREVIEW_LOG_PREFIX, original_name)
            ai_title = None
        final_name = ai_title.strip().replace('\n', ' ') if ai_title else original_name
        candidate = final_name
        suffix = 1
        while candidate in assigned:
            suffix += 1
            candidate = '%s (%d)' % (final_name, suffix)
        assigned.add(candidate)
        _append_title(
            preview_id,
            {
                'title': candidate,
                'from_ai': bool(ai_title),
                'song_count': len(songs),
                'sample': [
                    '%s - %s' % (title or 'Unknown Title', author or 'Unknown Artist')
                    for _item_id, title, author in songs[:PREVIEW_SAMPLE_SONGS]
                ],
            },
        )
