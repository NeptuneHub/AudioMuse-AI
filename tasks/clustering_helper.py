# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Per-iteration clustering worker: parameter generation, fitting and scoring.

The inner loop of the clustering search run by tasks.clustering. Given a method
and parameter set it prepares and scales the feature/embedding data, fits a model
(via clustering_gpu), and scores the resulting playlists. Also generates the
random and evolutionary parameter mutations that the elitist search explores.

Main Features:
* _perform_single_clustering_iteration / _apply_clustering_model: run one
  clustering attempt end to end and return a scored result.
* _split_oversized_clusters: DBSCAN components larger than
  CLUSTERING_MAX_PLAYLIST_SONGS are re-split with KMeans into playlist-sized
  chunks - music embeddings form one connected density mass, so raw DBSCAN
  either merges everything into a single giant cluster or marks it all noise.
* _generate_random_parameters / _mutate_parameters / _generate_evolutionary_parameters:
  sample and mutate KMeans/DBSCAN/GMM/spectral/PCA params within configured ranges.
* Playlist shaping helpers (chunking, shuffling, optional AI naming) for each run.
"""

import json
import random
import logging
import time
import numpy as np
from collections import Counter, defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

logger = logging.getLogger(__name__)

try:
    from .clustering_gpu import get_clustering_model, get_pca_model

    GPU_CLUSTERING_AVAILABLE = True
except ImportError:
    GPU_CLUSTERING_AVAILABLE = False
    logger.debug("GPU clustering module not available, using CPU only")

from rq.job import Job, JobStatus
from rq.exceptions import NoSuchJobError

from config import (
    STRATIFIED_GENRES,
    OTHER_FEATURE_LABELS,
    MOOD_LABELS,
    MAX_DISTANCE,
    MAX_SONGS_PER_ARTIST,
    MIN_PLAYLIST_SIZE_FOR_TOP_N,
    CLUSTERING_MAX_PLAYLIST_SONGS,
    CLUSTERING_SUBSET_SONGS,
    GMM_COVARIANCE_TYPE,
    SPECTRAL_N_NEIGHBORS,
    TOP_K_MOODS_FOR_PURITY_CALCULATION,
    LN_MOOD_DIVERSITY_STATS,
    LN_MOOD_PURITY_STATS,
    LN_MOOD_DIVERSITY_EMBEDING_STATS,
    LN_MOOD_PURITY_EMBEDING_STATS,
    LN_OTHER_FEATURES_DIVERSITY_STATS,
    LN_OTHER_FEATURES_PURITY_STATS,
    OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY,
    USE_GPU_CLUSTERING,
    TASK_STATUS_SUCCESS,
    LYRICS_ENABLED,
)
from .commons import score_vector

from tasks.ai.api import get_ai_playlist_name
from tasks.ai.playlist_namer import build_naming_context

from database import (
    get_tracks_by_ids,
    get_score_data_by_ids,
    get_task_info_from_db,
    get_lyrics_axis_vectors,
)
from taskqueue import redis_conn


def _shuffle_playlist_songs(songs, playlist_name):
    final_songs = songs.copy()
    n = len(final_songs)
    if n <= 1:
        logger.info("FINAL: '%s' has only %d songs - no shuffling needed", playlist_name, n)
        return final_songs

    current_time_seed = int(time.time() * 1000000) % 1000000
    for i in range(n - 1, 0, -1):
        j = (random.randint(0, i) + current_time_seed + i * 7) % (i + 1)
        final_songs[i], final_songs[j] = final_songs[j], final_songs[i]
        current_time_seed = (current_time_seed * 1103515245 + 12345) % (2**31)

    logger.info(
        "FINAL FISHER-YATES SHUFFLE applied to '%s': %d songs", playlist_name, len(final_songs)
    )
    logger.info(
        "FINAL ORDER: First song = '%s', Last song = '%s'", final_songs[0][1], final_songs[-1][1]
    )
    return final_songs


def _assign_playlist_chunks(final_songs, max_songs, base_name, final_playlists):
    if max_songs > 0 and len(final_songs) > max_songs:
        chunks = [final_songs[i : i + max_songs] for i in range(0, len(final_songs), max_songs)]
        for idx, chunk in enumerate(chunks, 1):
            final_playlists[f"{base_name} ({idx})"] = chunk
    else:
        final_playlists[base_name] = final_songs


def _try_ai_name_playlist(
    original_name,
    songs,
    centroids,
    ai_provider,
    ollama_url,
    ollama_model,
    openai_url,
    openai_model,
    openai_key,
    gemini_key,
    gemini_model,
    mistral_key,
    mistral_model,
    avoid_names=None,
    primary_genre=None,
):
    if (ai_provider or 'NONE').upper() == 'NONE':
        return original_name
    ai_config = {
        'provider': ai_provider,
        'ollama_url': ollama_url,
        'ollama_model': ollama_model,
        'openai_url': openai_url,
        'openai_model': openai_model,
        'openai_key': openai_key,
        'gemini_key': gemini_key,
        'gemini_model': gemini_model,
        'mistral_key': mistral_key,
        'mistral_model': mistral_model,
    }
    item_ids = [item_id for item_id, _title, _author in songs]
    score_rows = get_score_data_by_ids(item_ids)
    axis_blobs = {}
    columns = []
    if LYRICS_ENABLED:
        try:
            from lyrics.lyrics_transcriber import axis_columns

            columns = list(axis_columns())
            axis_blobs = get_lyrics_axis_vectors(item_ids)
        except Exception:
            logger.exception("Could not load lyric axes for playlist naming")

    context = build_naming_context(
        score_rows,
        centroids.get(original_name, {}),
        axis_blobs.values(),
        len(item_ids),
        columns,
        primary_genre=primary_genre,
        diversify=True,
    )
    logger.info(
        "Playlist naming context for '%s': genre=%s dimension=%s evidence=%s "
        "ideas=%s reliable_axes=%s",
        original_name,
        context['genre'],
        context['naming_dimension'],
        context['naming_evidence'],
        context['ideas'],
        context['axis_labels'],
    )
    ai_name = None
    if context['naming_evidence'] != 'general-purpose listening':
        ai_avoid_names = [
            name
            for name in (avoid_names or [])
            if '_' not in name.partition('_automatic')[0]
        ]
        ai_name = get_ai_playlist_name(
            context['genre'],
            context['naming_dimension'],
            context['naming_evidence'],
            ai_config,
            instrumental=context['instrumental'],
            avoid_names=ai_avoid_names,
        )
    if ai_name:
        return ai_name.strip().replace("\n", " ")
    logger.warning(
        "AI naming failed for '%s'. Keeping the tag-based cluster name.",
        original_name,
    )
    return original_name


def _perform_single_clustering_iteration(
    run_idx,
    item_ids_for_subset,
    clustering_method,
    num_clusters_min_max,
    dbscan_params_ranges,
    gmm_params_ranges,
    spectral_params_ranges,
    pca_params_ranges,
    active_mood_labels,
    max_songs_per_cluster,
    log_prefix,
    elite_solutions_params_list,
    exploitation_probability,
    mutation_config,
    score_weights,
    enable_clustering_embeddings,
):
    try:
        from flask_app import app

        if not item_ids_for_subset:
            logger.warning(
                f"{log_prefix} Iteration {run_idx}: Received empty item ID subset. Skipping."
            )
            return {"fitness_score": -1.0}

        with app.app_context():
            valid_tracks, X_feat_orig, X_embed_raw = _prepare_iteration_data(
                item_ids_for_subset,
                active_mood_labels,
                enable_clustering_embeddings,
                log_prefix,
                run_idx,
            )
        if valid_tracks is None:
            return {"fitness_score": -1.0}

        data_to_cluster, scaler = _prepare_and_scale_data(
            X_feat_orig, X_embed_raw, enable_clustering_embeddings
        )
        if data_to_cluster is None:
            logger.error(
                f"{log_prefix} Iteration {run_idx}: Data for clustering is empty after prep. Cannot proceed."
            )
            return {"fitness_score": -1.0}

        params = _generate_evolutionary_parameters(
            elite_solutions_params_list,
            exploitation_probability,
            mutation_config,
            clustering_method,
            data_to_cluster,
            pca_params_ranges,
            num_clusters_min_max,
            dbscan_params_ranges,
            gmm_params_ranges,
            spectral_params_ranges,
            log_prefix,
            run_idx,
        )

        pca_model, data_after_pca = None, data_to_cluster
        if params['pca_config']['enabled']:
            if USE_GPU_CLUSTERING and GPU_CLUSTERING_AVAILABLE:
                pca_model = get_pca_model(
                    n_components=params['pca_config']['components'], use_gpu=True
                )
            else:
                pca_model = PCA(n_components=params['pca_config']['components'])

            data_after_pca = pca_model.fit_transform(data_to_cluster)
            params['pca_config']['components'] = pca_model.n_components_

        labels, cluster_centers_map, _ = _apply_clustering_model(
            data_after_pca, params['clustering_method_config'], log_prefix, run_idx
        )
        if labels is None:
            return {"fitness_score": -1.0}

        return _format_and_score_iteration_result(
            labels,
            valid_tracks,
            X_feat_orig,
            data_after_pca,
            cluster_centers_map,
            pca_model,
            scaler,
            active_mood_labels,
            params,
            max_songs_per_cluster,
            run_idx,
            enable_clustering_embeddings,
            score_weights,
            log_prefix,
        )

    except Exception:
        logger.exception(f"{log_prefix} Iteration {run_idx} failed critically")
        raise


def _prepare_iteration_data(item_ids, active_mood_labels, use_embeddings, log_prefix, run_idx):
    logger.info(
        f"{log_prefix} Iteration {run_idx}: Fetching data for {len(item_ids)} tracks. Use embeddings: {use_embeddings}"
    )
    rows = get_tracks_by_ids(item_ids) if use_embeddings else get_score_data_by_ids(item_ids)
    valid_tracks, X_feat_orig_list, X_embed_raw_list = [], [], []
    for row_data in (dict(r) for r in rows if r):
        try:
            feature_vec = score_vector(row_data, active_mood_labels, OTHER_FEATURE_LABELS)
            if use_embeddings:
                embedding_vec = row_data.get('embedding_vector')
                if embedding_vec is None or embedding_vec.size == 0:
                    logger.warning(
                        f"Skipping track {row_data.get('item_id')} due to missing embedding."
                    )
                    continue
                X_embed_raw_list.append(embedding_vec)
            X_feat_orig_list.append(feature_vec)
            valid_tracks.append(row_data)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Skipping track {row_data.get('item_id')} due to data parsing error.")
    if not valid_tracks:
        logger.error(f"{log_prefix} Iteration {run_idx}: No valid tracks could be processed.")
        return None, None, None
    return (
        valid_tracks,
        np.array(X_feat_orig_list),
        np.array(X_embed_raw_list) if use_embeddings else None,
    )


def _prepare_and_scale_data(X_feat, X_embed, use_embeddings):
    data_source = X_embed if use_embeddings else X_feat
    if data_source is None or data_source.shape[0] == 0:
        return None, None
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_source)
    return scaled_data, scaler


def _mutate_param(value, min_val, max_val, delta, is_float=False):
    if is_float:
        mutation = random.uniform(-delta, delta)
        new_value = value + mutation
    else:
        int_delta = max(1, int(delta))
        mutation = random.randint(-int_delta, int_delta)
        new_value = value + mutation
    new_value = np.clip(new_value, min_val, max_val)
    return int(new_value) if not is_float else new_value


def _generate_evolutionary_parameters(elites, exploitation_prob, mutation_cfg, method, data, *args):
    if elites and random.random() < exploitation_prob:
        chosen_elite = random.choice(elites)
        return _mutate_parameters(chosen_elite, mutation_cfg, method, data, *args)
    return _generate_random_parameters(method, data, *args)


def _random_pca_config(pca_ranges, data):
    max_pca = min(pca_ranges['components_max'], data.shape[1], data.shape[0] - 1)
    min_pca = pca_ranges['components_min']
    if min_pca > max_pca:
        min_pca = max_pca
    pca_comps = random.randint(min_pca, max_pca) if max_pca > 0 else min_pca
    return {"enabled": pca_comps > 0, "components": pca_comps}


def _clamp_k_bounds(lower_k, upper_k, enforce_min_two=True):
    if enforce_min_two and lower_k < 2 and upper_k >= 2:
        lower_k = 2
    if upper_k < lower_k:
        upper_k = lower_k
    return lower_k, upper_k


def _random_k_in_bounds(lower_k, upper_k):
    return random.randint(lower_k, upper_k) if upper_k >= lower_k and upper_k > 0 else lower_k


def _random_kmeans_params(num_clust_ranges, max_k):
    upper_k = min(num_clust_ranges[1], max_k)
    lower_k = min(num_clust_ranges[0], upper_k)
    lower_k, upper_k = _clamp_k_bounds(lower_k, upper_k)
    return {"n_clusters": _random_k_in_bounds(lower_k, upper_k)}


def _random_dbscan_params(db_ranges):
    eps = round(random.uniform(db_ranges['eps_min'], db_ranges['eps_max']), 2)
    min_samples = random.randint(db_ranges['samples_min'], db_ranges['samples_max'])
    return {"eps": eps, "min_samples": min_samples}


def _random_gmm_params(gmm_ranges, max_k):
    upper_k = min(gmm_ranges['n_components_max'], max_k)
    lower_k = min(gmm_ranges['n_components_min'], upper_k)
    lower_k, upper_k = _clamp_k_bounds(lower_k, upper_k)
    return {"n_components": _random_k_in_bounds(lower_k, upper_k)}


def _random_spectral_params(spec_ranges, data):
    upper_k = min(spec_ranges['n_clusters_max'], data.shape[0] - 1)
    lower_k = spec_ranges['n_clusters_min']
    if lower_k < 2:
        lower_k = 2
    if upper_k < lower_k:
        upper_k = lower_k
    n_clust = random.randint(lower_k, upper_k)
    return {"n_clusters": n_clust, "random_state": random.randint(0, 10000)}


def _generate_random_parameters(
    method, data, pca_ranges, num_clust_ranges, db_ranges, gmm_ranges, spec_ranges, *args
):
    pca_config = _random_pca_config(pca_ranges, data)
    max_k = data.shape[0]

    if method == 'kmeans':
        method_params = _random_kmeans_params(num_clust_ranges, max_k)
    elif method == 'dbscan':
        method_params = _random_dbscan_params(db_ranges)
    elif method == 'gmm':
        method_params = _random_gmm_params(gmm_ranges, max_k)
    elif method == 'spectral':
        method_params = _random_spectral_params(spec_ranges, data)
    else:
        method_params = {}

    return {
        "pca_config": pca_config,
        "clustering_method_config": {"method": method, "params": method_params},
    }


def _mutate_parameters(
    elite_params,
    mutation_cfg,
    method,
    data,
    pca_ranges,
    num_clust_ranges,
    db_ranges,
    gmm_ranges,
    spec_ranges,
    *args,
):
    elite_pca_cfg = elite_params['pca_config']
    elite_method_cfg = elite_params['clustering_method_config']

    max_pca = min(pca_ranges['components_max'], data.shape[1], data.shape[0] - 1)
    min_pca = pca_ranges['components_min']
    if min_pca > max_pca:
        min_pca = max_pca
    mutated_pca_comps = _mutate_param(
        elite_pca_cfg.get('components', 0), min_pca, max_pca, mutation_cfg.get('int_abs_delta', 2)
    )
    pca_config = {"enabled": mutated_pca_comps > 0, "components": mutated_pca_comps}

    max_k = data.shape[0]
    method_params = {}

    if method == 'kmeans':
        upper_k = min(num_clust_ranges[1], max_k)
        lower_k = min(num_clust_ranges[0], upper_k)
        k = _mutate_param(
            elite_method_cfg['params']['n_clusters'],
            lower_k,
            upper_k,
            mutation_cfg.get('int_abs_delta', 2),
        )
        method_params = {"n_clusters": k}
    elif method == 'dbscan':
        mutated_eps = _mutate_param(
            elite_method_cfg['params']['eps'],
            db_ranges['eps_min'],
            db_ranges['eps_max'],
            mutation_cfg.get('float_abs_delta', 0.1),
            is_float=True,
        )
        mutated_min_samples = _mutate_param(
            elite_method_cfg['params']['min_samples'],
            db_ranges['samples_min'],
            db_ranges['samples_max'],
            mutation_cfg.get('int_abs_delta', 2),
        )
        method_params = {"eps": mutated_eps, "min_samples": mutated_min_samples}
    elif method == 'gmm':
        upper_k = min(gmm_ranges['n_components_max'], max_k)
        lower_k = min(gmm_ranges['n_components_min'], upper_k)
        n_comp = _mutate_param(
            elite_method_cfg['params']['n_components'],
            lower_k,
            upper_k,
            mutation_cfg.get('int_abs_delta', 2),
        )
        method_params = {"n_components": n_comp}
    elif method == 'spectral':
        upper_k = min(spec_ranges['n_clusters_max'], max_k - 1)
        lower_k = spec_ranges['n_clusters_min']
        if lower_k < 2:
            lower_k = 2
        if upper_k < lower_k:
            upper_k = lower_k
        n_clust = _mutate_param(
            elite_method_cfg['params']['n_clusters'],
            lower_k,
            upper_k,
            mutation_cfg.get('int_abs_delta', 2),
        )
        elite_random_state = elite_method_cfg['params'].get(
            "random_state", random.randint(0, 10000)
        )
        mutated_random_state = _mutate_param(
            elite_random_state, 0, 10000, mutation_cfg.get("int_abs_delta", 100)
        )
        method_params = {"n_clusters": n_clust, "random_state": mutated_random_state}

    return {
        "pca_config": pca_config,
        "clustering_method_config": {"method": method, "params": method_params},
    }


def _split_oversized_clusters(labels, data):
    labels = np.asarray(labels).copy()
    target = max(2 * MIN_PLAYLIST_SIZE_FOR_TOP_N, CLUSTERING_MAX_PLAYLIST_SONGS // 2)
    next_label = int(labels.max()) + 1
    for cid in [c for c in set(labels.tolist()) if c != -1]:
        idx = np.where(labels == cid)[0]
        if len(idx) <= CLUSTERING_MAX_PLAYLIST_SONGS:
            continue
        n_sub = min(len(idx), max(2, -(-len(idx) // target)))
        sub_labels = KMeans(
            n_clusters=n_sub, init='k-means++', n_init=3
        ).fit_predict(data[idx])
        labels[idx] = next_label + sub_labels
        next_label += n_sub
    return labels


def _apply_clustering_model(data, method_config, log_prefix, run_idx):
    method = method_config['method']
    params = method_config['params']
    model = None
    try:
        if method == 'kmeans':
            if params.get('n_clusters', 0) < 2:
                return None, None, None
        elif method == 'gmm':
            if params.get('n_components', 0) < 2 or params['n_components'] > data.shape[0]:
                return None, None, None
        elif method == 'spectral':
            if params.get('n_clusters', 0) < 2 or params['n_clusters'] >= data.shape[0]:
                return None, None, None

        use_gpu = USE_GPU_CLUSTERING and GPU_CLUSTERING_AVAILABLE

        if use_gpu:
            try:
                model = get_clustering_model(method, params, use_gpu=True)
                labels = model.fit_predict(data)
                logger.debug(f"{log_prefix} Iteration {run_idx}: GPU clustering used for {method}")
            except Exception as e:
                logger.warning(f"{log_prefix} GPU clustering failed, falling back to CPU: {e}")
                use_gpu = False

        if not use_gpu:
            if method == 'kmeans':
                model = KMeans(n_clusters=params['n_clusters'], init='k-means++', n_init=10)
            elif method == 'dbscan':
                model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
            elif method == 'gmm':
                model = GaussianMixture(
                    n_components=params['n_components'],
                    covariance_type=GMM_COVARIANCE_TYPE,
                    init_params='k-means++',
                    n_init=3,
                    random_state=None,
                    reg_covar=1e-4,
                )
            elif method == 'spectral':
                model = SpectralClustering(
                    n_clusters=params['n_clusters'],
                    assign_labels='kmeans',
                    affinity='nearest_neighbors',
                    n_neighbors=SPECTRAL_N_NEIGHBORS,
                    random_state=params.get("random_state"),
                    n_init=10,
                    verbose=False,
                )
            else:
                raise ValueError(f"Unsupported clustering method: {method}")

            labels = model.fit_predict(data)

        if method == 'dbscan' and labels is not None:
            labels = _split_oversized_clusters(labels, data)

        centers = {}
        if hasattr(model, 'cluster_centers_') and model.cluster_centers_ is not None:
            centers = {i: center for i, center in enumerate(model.cluster_centers_)}
        elif hasattr(model, 'means_') and model.means_ is not None:
            centers = {i: mean for i, mean in enumerate(model.means_)}
        else:
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            for label in unique_labels:
                cluster_points = data[labels == label]
                if cluster_points.shape[0] > 0:
                    centers[label] = cluster_points.mean(axis=0)

        return labels, centers, model

    except Exception:
        logger.exception(
            f"{log_prefix} Iteration {run_idx}: Clustering model failed for method {method}"
        )
        return None, None, None


def _get_feature_centroid_for_embedding_cluster(label_id, labels, X_feat_orig):
    cluster_indices = np.where(labels == label_id)[0]
    if len(cluster_indices) == 0:
        return None

    feature_vectors_in_cluster = X_feat_orig[cluster_indices]
    feature_centroid = np.mean(feature_vectors_in_cluster, axis=0)
    return feature_centroid


def _format_and_score_iteration_result(
    labels,
    valid_tracks,
    x_feat_orig,
    data_for_metrics,
    centers,
    pca,
    scaler,
    active_moods,
    params,
    max_songs_per_cluster,
    run_idx,
    use_embeddings,
    score_weights,
    log_prefix,
):
    if labels is None:
        return {"fitness_score": -1.0}

    raw_distances = np.full(len(valid_tracks), np.inf)
    if len(set(labels) - {-1}) > 0:
        for label_id in set(labels):
            if label_id == -1:
                continue
            indices = np.where(labels == label_id)[0]
            if len(indices) > 0 and label_id in centers:
                cluster_center = centers[label_id]
                points = data_for_metrics[indices]
                distances = np.linalg.norm(points - cluster_center, axis=1)
                raw_distances[indices] = distances

    max_dist_val = (
        raw_distances[raw_distances != np.inf].max() if np.any(raw_distances != np.inf) else 1.0
    )
    if max_dist_val == 0:
        max_dist_val = 1.0
    normalized_distances = raw_distances / max_dist_val

    track_info_list = [
        {"row": valid_tracks[i], "label": labels[i], "distance": normalized_distances[i]}
        for i in range(len(valid_tracks))
    ]

    filtered_clusters = defaultdict(list)
    for cid in set(labels):
        if cid == -1:
            continue
        cluster_tracks_info = [
            t_info
            for t_info in track_info_list
            if t_info["label"] == cid and t_info["distance"] <= MAX_DISTANCE
        ]
        if not cluster_tracks_info:
            continue

        cluster_tracks_info.sort(key=lambda x: x["distance"])
        count_per_artist = defaultdict(int)
        selected_tracks_for_playlist = []
        for t_item_info in cluster_tracks_info:
            author = t_item_info["row"].get("author")
            author_norm = (author or "").strip().lower()

            if MAX_SONGS_PER_ARTIST is None or MAX_SONGS_PER_ARTIST <= 0:
                allowed_by_artist = True
            else:
                allowed_by_artist = count_per_artist[author_norm] < MAX_SONGS_PER_ARTIST

            if allowed_by_artist:
                selected_tracks_for_playlist.append(t_item_info)
                count_per_artist[author_norm] += 1

            if (
                max_songs_per_cluster > 0
                and len(selected_tracks_for_playlist) >= max_songs_per_cluster
            ):
                break

        for t_item_info_final in selected_tracks_for_playlist:
            item_id_val, title_val, author_val = (
                t_item_info_final["row"]["item_id"],
                t_item_info_final["row"]["title"],
                t_item_info_final["row"]["author"],
            )
            filtered_clusters[cid].append((item_id_val, title_val, author_val))

    named_playlists, playlist_centroids = {}, {}
    playlist_to_centroid_vector_map = {}
    playlist_primary_genres = {}
    unique_predominant_mood_scores = {}
    unique_predominant_other_feature_scores = {}
    item_id_to_song_index_map = {
        track_data['item_id']: i for i, track_data in enumerate(valid_tracks)
    }
    item_id_to_primary_genre = {
        track_data['item_id']: _get_track_primary_genre(track_data)
        for track_data in valid_tracks
    }

    for label_id, songs_list in filtered_clusters.items():
        if songs_list and label_id in centers:
            center_vec = centers[label_id]
            if use_embeddings:
                feature_centroid_vec = _get_feature_centroid_for_embedding_cluster(
                    label_id, labels, x_feat_orig
                )
                if feature_centroid_vec is None:
                    continue
                name, centroid_details = _name_cluster(
                    feature_centroid_vec, None, False, active_moods, None
                )
            else:
                name, centroid_details = _name_cluster(
                    center_vec, pca, params['pca_config']['enabled'], active_moods, scaler
                )

            temp_name, suffix = name, 1
            while temp_name in named_playlists:
                temp_name = f"{name}_{suffix}"
                suffix += 1

            named_playlists[temp_name] = songs_list
            playlist_centroids[temp_name] = centroid_details
            playlist_to_centroid_vector_map[temp_name] = center_vec
            genre_counts = Counter(
                item_id_to_primary_genre.get(item_id, '__other__')
                for item_id, _, _ in songs_list
            )
            known_genres = [genre for genre in STRATIFIED_GENRES if genre_counts[genre]]
            playlist_primary_genres[temp_name] = (
                max(known_genres, key=genre_counts.get)
                if known_genres
                else '__other__'
            )

            if centroid_details and any(mood in active_moods for mood in centroid_details.keys()):
                predominant_mood_key = max(
                    (k for k in centroid_details if k in MOOD_LABELS),
                    key=centroid_details.get,
                    default=None,
                )
                if predominant_mood_key:
                    current_mood_score = centroid_details.get(predominant_mood_key, 0.0)
                    unique_predominant_mood_scores[predominant_mood_key] = max(
                        unique_predominant_mood_scores.get(predominant_mood_key, 0.0),
                        current_mood_score,
                    )

            centroid_other_features = {
                lk: centroid_details.get(lk, 0.0)
                for lk in OTHER_FEATURE_LABELS
                if lk in centroid_details
            }
            if centroid_other_features:
                predominant_other_key = max(
                    centroid_other_features, key=centroid_other_features.get, default=None
                )
                if (
                    predominant_other_key
                    and centroid_other_features[predominant_other_key]
                    > OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY
                ):
                    unique_predominant_other_feature_scores[predominant_other_key] = max(
                        unique_predominant_other_feature_scores.get(predominant_other_key, 0.0),
                        centroid_other_features[predominant_other_key],
                    )

    metrics = {
        "silhouette": 0.0,
        "davies_bouldin": 0.0,
        "calinski_harabasz": 0.0,
        "mood_diversity": 0.0,
        "mood_purity": 0.0,
        "other_feature_diversity": 0.0,
        "other_feature_purity": 0.0,
    }
    num_clusters = len(named_playlists)

    if num_clusters >= 2 and num_clusters < data_for_metrics.shape[0]:
        if score_weights.get('silhouette', 0) > 0:
            try:
                metrics['silhouette'] = (
                    silhouette_score(data_for_metrics, labels, random_state=42) + 1
                ) / 2.0
            except ValueError:
                pass
        if score_weights.get('davies_bouldin', 0) > 0:
            try:
                metrics['davies_bouldin'] = 1.0 / (
                    1.0 + davies_bouldin_score(data_for_metrics, labels)
                )
            except ValueError:
                pass
        if score_weights.get('calinski_harabasz', 0) > 0:
            try:
                metrics['calinski_harabasz'] = 1.0 - np.exp(
                    -calinski_harabasz_score(data_for_metrics, labels) / 500.0
                )
            except ValueError:
                pass

    raw_mood_diversity_score = sum(unique_predominant_mood_scores.values())
    ln_mood_diversity = np.log1p(raw_mood_diversity_score)
    diversity_stats = (
        LN_MOOD_DIVERSITY_EMBEDING_STATS if use_embeddings else LN_MOOD_DIVERSITY_STATS
    )
    mean_div, sd_div = diversity_stats.get("mean"), diversity_stats.get("sd")
    if mean_div is not None and sd_div is not None and sd_div > 1e-9:
        metrics['mood_diversity'] = (ln_mood_diversity - mean_div) / sd_div

    raw_other_diversity_score = sum(unique_predominant_other_feature_scores.values())
    ln_other_diversity = np.log1p(raw_other_diversity_score)
    other_div_stats = LN_OTHER_FEATURES_DIVERSITY_STATS
    mean_other_div, sd_other_div = other_div_stats.get("mean"), other_div_stats.get("sd")
    if mean_other_div is not None and sd_other_div is not None and sd_other_div > 1e-9:
        metrics['other_feature_diversity'] = (ln_other_diversity - mean_other_div) / sd_other_div

    all_playlist_purities = []
    if named_playlists:
        for name, songs in named_playlists.items():
            centroid_data = playlist_centroids.get(name)
            if not centroid_data or not songs:
                continue

            sorted_moods = sorted(
                [(m, s) for m, s in centroid_data.items() if m in MOOD_LABELS],
                key=lambda item: item[1],
                reverse=True,
            )
            top_moods = [
                m for m, s in sorted_moods[:TOP_K_MOODS_FOR_PURITY_CALCULATION] if s > 0.01
            ]
            if not top_moods:
                continue

            song_purity_scores = []
            for item_id, _, _ in songs:
                song_idx = item_id_to_song_index_map.get(item_id)
                if song_idx is not None and song_idx < x_feat_orig.shape[0]:
                    song_feat_vec = x_feat_orig[song_idx]
                    max_score_for_song = 0.0
                    for mood in top_moods:
                        try:
                            mood_idx = active_moods.index(mood)
                            if 2 + mood_idx < song_feat_vec.shape[0]:
                                song_score = song_feat_vec[2 + mood_idx]
                                if song_score > max_score_for_song:
                                    max_score_for_song = song_score
                        except ValueError:
                            continue
                    if max_score_for_song > 0:
                        song_purity_scores.append(max_score_for_song)
            if song_purity_scores:
                all_playlist_purities.append(sum(song_purity_scores))

    raw_mood_purity = sum(all_playlist_purities)
    ln_mood_purity = np.log1p(raw_mood_purity)
    purity_stats = LN_MOOD_PURITY_EMBEDING_STATS if use_embeddings else LN_MOOD_PURITY_STATS
    mean_pur, sd_pur = purity_stats.get("mean"), purity_stats.get("sd")
    if mean_pur is not None and sd_pur is not None and sd_pur > 1e-9:
        metrics['mood_purity'] = (ln_mood_purity - mean_pur) / sd_pur

    all_other_feature_purities = []
    if named_playlists:
        for name, songs in named_playlists.items():
            centroid_data = playlist_centroids.get(name)
            if not centroid_data or not songs:
                continue

            other_features = {k: v for k, v in centroid_data.items() if k in OTHER_FEATURE_LABELS}
            if not other_features:
                continue

            predominant_other = max(other_features, key=other_features.get, default=None)
            if (
                not predominant_other
                or other_features[predominant_other]
                < OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY
            ):
                continue

            try:
                feature_idx = OTHER_FEATURE_LABELS.index(predominant_other)
                song_purity_scores = []
                for item_id, _, _ in songs:
                    song_idx = item_id_to_song_index_map.get(item_id)
                    if song_idx is not None and song_idx < x_feat_orig.shape[0]:
                        song_feat_vec = x_feat_orig[song_idx]
                        other_features_start_idx = 2 + len(active_moods)
                        if other_features_start_idx + feature_idx < song_feat_vec.shape[0]:
                            song_score = song_feat_vec[other_features_start_idx + feature_idx]
                            song_purity_scores.append(song_score)
                if song_purity_scores:
                    all_other_feature_purities.append(sum(song_purity_scores))
            except ValueError:
                continue

    raw_other_purity = sum(all_other_feature_purities)
    ln_other_purity = np.log1p(raw_other_purity)
    other_purity_stats = LN_OTHER_FEATURES_PURITY_STATS
    mean_other_pur, sd_other_pur = other_purity_stats.get("mean"), other_purity_stats.get("sd")
    if mean_other_pur is not None and sd_other_pur is not None and sd_other_pur > 1e-9:
        metrics['other_feature_purity'] = (ln_other_purity - mean_other_pur) / sd_other_pur

    final_score = sum(score_weights.get(k, 0) * v for k, v in metrics.items())

    log_message = (
        f"{log_prefix} Iteration {run_idx}: Scores -> "
        f"MoodDiv: {metrics['mood_diversity']:.2f} (raw: {raw_mood_diversity_score:.2f}), "
        f"MoodPur: {metrics['mood_purity']:.2f} (raw: {raw_mood_purity:.2f}), "
        f"OtherFeatDiv: {metrics['other_feature_diversity']:.2f} (raw: {raw_other_diversity_score:.2f}), "
        f"OtherFeatPur: {metrics['other_feature_purity']:.2f} (raw: {raw_other_purity:.2f}), "
        f"Sil: {metrics['silhouette']:.2f}, DB: {metrics['davies_bouldin']:.2f}, CH: {metrics['calinski_harabasz']:.2f} | "
        f"FinalScore: {final_score:.2f}"
    )
    logger.info(log_message)

    logger.info(
        f"Run {run_idx} ({params['clustering_method_config']['method']}): "
        f"Created {len(named_playlists)} clusters."
    )
    for name, songs in named_playlists.items():
        song_titles = [f"'{s[1]}'" for s in songs[:5]]
        log_msg = f"  - Cluster '{name}': {', '.join(song_titles)}"
        if len(songs) > 5:
            log_msg += f", ... and {len(songs) - 5} more."
        logger.info(log_msg)

    return {
        "fitness_score": final_score,
        "named_playlists": named_playlists,
        "playlist_centroids": playlist_centroids,
        "playlist_to_centroid_vector_map": playlist_to_centroid_vector_map,
        "playlist_primary_genres": playlist_primary_genres,
        "parameters": {**params, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx},
        "scaler_details": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
        if scaler
        else None,
        "pca_model_details": {
            "components": pca.components_.tolist(),
            "variance": pca.explained_variance_ratio_.tolist(),
        }
        if pca
        else None,
    }


def _name_cluster(centroid_vector, pca_model, pca_enabled, mood_labels, scaler):
    TOP_MOODS_IN_NAME = 3
    OTHER_FEATURE_THRESHOLD_FOR_NAME = 0.5
    MAX_OTHER_FEATURES_IN_NAME = 2

    if scaler:
        vec = centroid_vector.reshape(1, -1)
        if pca_enabled and pca_model:
            vec = pca_model.inverse_transform(vec)
        interpreted_vector = scaler.inverse_transform(vec)[0]
    else:
        interpreted_vector = centroid_vector

    tempo_val = interpreted_vector[0]
    mood_values = interpreted_vector[2 : 2 + len(mood_labels)]

    tempo_label = "Slow" if tempo_val < 0.33 else "Medium" if tempo_val < 0.66 else "Fast"

    if len(mood_values) > 0 and np.sum(mood_values) > 0:
        top_mood_indices = np.argsort(mood_values)[::-1][:TOP_MOODS_IN_NAME]
        mood_names = [
            mood_labels[i].title()
            for i in top_mood_indices
            if i < len(mood_labels) and mood_values[i] > 0.01
        ]
        mood_part = "_".join(mood_names) if mood_names else "Mixed"
    else:
        mood_part = "Mixed"

    base_name = f"{mood_part}_{tempo_label}"

    details = {label: float(val) for label, val in zip(mood_labels, mood_values)}
    other_features_start = 2 + len(mood_labels)
    appended_other_features_str = ""
    other_feature_scores_dict = {}

    if len(interpreted_vector) > other_features_start:
        other_feature_values = interpreted_vector[other_features_start:]
        for i, label in enumerate(OTHER_FEATURE_LABELS):
            if i < len(other_feature_values):
                score = float(other_feature_values[i])
                details[label] = score
                other_feature_scores_dict[label] = score

        if other_feature_scores_dict:
            prominent_features = sorted(
                [
                    (feature, score)
                    for feature, score in other_feature_scores_dict.items()
                    if score >= OTHER_FEATURE_THRESHOLD_FOR_NAME
                ],
                key=lambda item: item[1],
                reverse=True,
            )
            features_to_add = [
                feature.title()
                for feature, score in prominent_features[:MAX_OTHER_FEATURES_IN_NAME]
            ]
            if features_to_add:
                appended_other_features_str = "_" + "_".join(features_to_add)

    final_name = f"{base_name}{appended_other_features_str}"

    return final_name, details


def get_job_result_safely(job_id, parent_task_id, task_type="child task"):
    from flask_app import app

    try:
        job = Job.fetch(job_id, connection=redis_conn)
        if job.is_finished and isinstance(job.result, dict):
            return job.result
    except NoSuchJobError:
        logger.warning(f"[{parent_task_id}] Job {job_id} not in RQ. Checking DB.")
        with app.app_context():
            task_info = get_task_info_from_db(job_id)
            if task_info and task_info.get('status') in [TASK_STATUS_SUCCESS, JobStatus.FINISHED]:
                try:
                    details = json.loads(task_info.get('details'))
                    batch_result = details.get('full_best_result_from_batch') or details.get(
                        'full_result'
                    )
                    if batch_result:
                        return {
                            "status": "SUCCESS",
                            "best_result_from_batch": batch_result,
                            "iterations_completed_in_batch": details.get(
                                "iterations_completed_in_batch", 0
                            ),
                            "final_subset_track_ids": details.get("final_subset_track_ids", []),
                        }
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Could not parse result from DB for job {job_id}")
    return None


def _fill_balanced_quotas(quotas, capacities, remaining):
    remaining = max(0, int(remaining))
    while remaining > 0:
        open_genres = [
            genre for genre, capacity in capacities.items()
            if quotas.get(genre, 0) < capacity
        ]
        if not open_genres:
            break

        lowest_count = min(quotas.get(genre, 0) for genre in open_genres)
        lowest_genres = [
            genre for genre in open_genres
            if quotas.get(genre, 0) == lowest_count
        ]
        random.shuffle(lowest_genres)
        for genre in lowest_genres[:remaining]:
            quotas[genre] = quotas.get(genre, 0) + 1
            remaining -= 1
            if remaining == 0:
                break
    return quotas


def _calculate_stratified_quotas(genre_tracks, sample_size, target_per_genre):
    capacities = {
        genre: len(tracks)
        for genre, tracks in genre_tracks.items()
        if genre in STRATIFIED_GENRES and tracks
    }
    total_known = sum(capacities.values())
    wanted_known = min(max(0, int(sample_size)), total_known)
    target = max(0, int(target_per_genre))

    base_limits = {
        genre: min(capacity, target)
        for genre, capacity in capacities.items()
    }
    if sum(base_limits.values()) >= wanted_known:
        quotas = dict.fromkeys(capacities, 0)
        return _fill_balanced_quotas(quotas, base_limits, wanted_known)

    quotas = dict(base_limits)
    return _fill_balanced_quotas(
        quotas,
        capacities,
        wanted_known - sum(quotas.values()),
    )


def _regroup_tracks_by_primary_genre(genre_map):
    tracks_by_id = {}
    for tracks in genre_map.values():
        for track in tracks:
            track_id = track.get('item_id')
            if track_id is not None and track_id not in tracks_by_id:
                tracks_by_id[track_id] = track

    genre_tracks = defaultdict(list)
    for track in tracks_by_id.values():
        genre_tracks[_get_track_primary_genre(track)].append(track)
    return tracks_by_id, genre_tracks


def _select_tracks_for_genre(
    candidates, quota, previous_ids, change_fraction, selected_ids, rotate
):
    previous_candidates = [
        track for track in candidates if track['item_id'] in previous_ids
    ]
    keep_count = 0
    if rotate:
        keep_count = min(
            len(previous_candidates),
            quota,
            int(quota * (1.0 - change_fraction)),
        )
    kept = (
        random.sample(previous_candidates, keep_count)
        if keep_count < len(previous_candidates)
        else previous_candidates
    )
    chosen = list(kept)
    selected_ids.update(track['item_id'] for track in kept)

    needed = quota - len(kept)
    if needed <= 0:
        return chosen

    fresh = [
        track for track in candidates
        if track['item_id'] not in selected_ids
        and (change_fraction <= 0.0 or track['item_id'] not in previous_ids)
    ]
    added = random.sample(fresh, min(needed, len(fresh)))
    chosen.extend(added)
    selected_ids.update(track['item_id'] for track in added)

    still_needed = quota - len(kept) - len(added)
    if still_needed > 0:
        remaining = [
            track for track in candidates
            if track['item_id'] not in selected_ids
        ]
        reused = random.sample(remaining, still_needed)
        chosen.extend(reused)
        selected_ids.update(track['item_id'] for track in reused)
    return chosen


def _get_stratified_song_subset(
    genre_map,
    target_per_genre,
    prev_ids=None,
    percent_change=0.0,
):
    tracks_by_id, genre_tracks = _regroup_tracks_by_primary_genre(genre_map)

    desired_size = min(max(0, int(CLUSTERING_SUBSET_SONGS)), len(tracks_by_id))
    if desired_size == 0:
        return []

    quotas = _calculate_stratified_quotas(
        genre_tracks,
        desired_size,
        target_per_genre,
    )

    known_quota_total = sum(quotas.values())
    if known_quota_total < desired_size:
        other_capacity = len(genre_tracks.get('__other__', []))
        quotas['__other__'] = min(
            other_capacity,
            desired_size - known_quota_total,
        )

    previous_ids = set(prev_ids or [])
    change_fraction = min(1.0, max(0.0, float(percent_change)))
    rotate = prev_ids is not None
    selected, selected_ids = [], set()

    for genre, quota in quotas.items():
        if quota <= 0:
            continue
        selected.extend(
            _select_tracks_for_genre(
                genre_tracks.get(genre, []),
                quota,
                previous_ids,
                change_fraction,
                selected_ids,
                rotate,
            )
        )

    random.shuffle(selected)
    return selected


def _get_track_primary_genre(track_data):
    if 'mood_vector' in track_data and track_data['mood_vector']:
        mood_scores = {
            p.split(':')[0]: float(p.split(':')[1])
            for p in track_data['mood_vector'].split(',')
            if ':' in p
        }
        return max(
            (g for g in STRATIFIED_GENRES if g in mood_scores),
            key=mood_scores.get,
            default='__other__',
        )
    return '__other__'
