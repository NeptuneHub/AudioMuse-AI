# tasks/clustering_helper.py

import json
import random
import logging
import numpy as np
from collections import defaultdict
# time, re, and cdist imports moved to clustering_postprocessing.py
from psycopg2.extras import DictCursor

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

# RQ imports for safe result fetching
from rq.job import Job
from rq.exceptions import NoSuchJobError

from config import (STRATIFIED_GENRES, OTHER_FEATURE_LABELS, MOOD_LABELS, MAX_DISTANCE,
                    MAX_SONGS_PER_ARTIST, GMM_COVARIANCE_TYPE, SPECTRAL_N_NEIGHBORS,
                    TOP_K_MOODS_FOR_PURITY_CALCULATION, LN_MOOD_DIVERSITY_STATS,
                    LN_MOOD_PURITY_STATS, LN_MOOD_DIVERSITY_EMBEDING_STATS,
                    LN_MOOD_PURITY_EMBEDING_STATS, LN_OTHER_FEATURES_DIVERSITY_STATS,
                    LN_OTHER_FEATURES_PURITY_STATS,
                    OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY)
from .commons import score_vector

logger = logging.getLogger(__name__)

# --- Main Orchestrator for a Single Iteration ---

def _perform_single_clustering_iteration(
    run_idx, item_ids_for_subset,
    clustering_method, num_clusters_min_max, dbscan_params_ranges, gmm_params_ranges,
    spectral_params_ranges, pca_params_ranges, active_mood_labels,
    max_songs_per_cluster, log_prefix,
    elite_solutions_params_list, exploitation_probability, mutation_config,
    score_weights, enable_clustering_embeddings):
    """
    Orchestrates a single evolutionary run of the clustering process.
    This function is now a high-level coordinator.
    """
    try:
        # Local import to prevent circular dependency
        from app import app

        if not item_ids_for_subset:
            logger.warning(f"{log_prefix} Iteration {run_idx}: Received empty item ID subset. Skipping.")
            return {"fitness_score": -1.0}

        # 1. Prepare Data: Fetch full track data and create feature vectors
        with app.app_context():
            valid_tracks, X_feat_orig, X_embed_raw = _prepare_iteration_data(
                item_ids_for_subset, active_mood_labels, enable_clustering_embeddings, log_prefix, run_idx
            )
        if valid_tracks is None:
             return {"fitness_score": -1.0} # Error already logged in helper

        # 2. Prepare data for clustering (embeddings or features) and scale it
        data_to_cluster, scaler = _prepare_and_scale_data(X_feat_orig, X_embed_raw, enable_clustering_embeddings)
        if data_to_cluster is None:
            logger.error(f"{log_prefix} Iteration {run_idx}: Data for clustering is empty after prep. Cannot proceed.")
            return {"fitness_score": -1.0}

        # 3. Generate Parameters: Use evolutionary approach (mutate elite or explore)
        params = _generate_evolutionary_parameters(
            elite_solutions_params_list, exploitation_probability, mutation_config,
            clustering_method, data_to_cluster, pca_params_ranges,
            num_clusters_min_max, dbscan_params_ranges, gmm_params_ranges, spectral_params_ranges,
            log_prefix, run_idx
        )

        # 4. Apply PCA if specified by the generated parameters
        pca_model, data_after_pca = None, data_to_cluster
        if params['pca_config']['enabled']:
            pca_model = PCA(n_components=params['pca_config']['components'])
            data_after_pca = pca_model.fit_transform(data_to_cluster)
            params['pca_config']['components'] = pca_model.n_components_ # Update with actual components

        # 5. Apply the chosen clustering model
        labels, cluster_centers_map, model = _apply_clustering_model(
            data_after_pca, params['clustering_method_config'], log_prefix, run_idx
        )
        if labels is None: # Clustering failed
            return {"fitness_score": -1.0}

        # 6. Format results and calculate fitness score
        return _format_and_score_iteration_result(
            labels, valid_tracks, X_feat_orig, data_after_pca,
            cluster_centers_map, model, pca_model, scaler, active_mood_labels,
            params, max_songs_per_cluster, run_idx, enable_clustering_embeddings, score_weights, log_prefix
        )

    except Exception as e:
        logger.error(f"{log_prefix} Iteration {run_idx} failed critically", exc_info=True)
        raise

# --- Step 1: Data Preparation ---

def _prepare_iteration_data(item_ids, active_mood_labels, use_embeddings, log_prefix, run_idx):
    """Fetches track data, creates feature/embedding vectors, and ensures alignment."""
    # Local imports to prevent circular dependency
    from app_helper import get_tracks_by_ids, get_score_data_by_ids

    logger.info(f"{log_prefix} Iteration {run_idx}: Fetching data for {len(item_ids)} tracks. Use embeddings: {use_embeddings}")
    rows = get_tracks_by_ids(item_ids) if use_embeddings else get_score_data_by_ids(item_ids) # These functions are now imported locally
    valid_tracks, X_feat_orig_list, X_embed_raw_list = [], [], []
    for row_data in (dict(r) for r in rows if r):
        try:
            feature_vec = score_vector(row_data, active_mood_labels, OTHER_FEATURE_LABELS)
            if use_embeddings:
                embedding_vec = row_data.get('embedding_vector')
                if embedding_vec is None or embedding_vec.size == 0:
                    logger.warning(f"Skipping track {row_data.get('item_id')} due to missing embedding.")
                    continue
                X_embed_raw_list.append(embedding_vec)
            X_feat_orig_list.append(feature_vec)
            valid_tracks.append(row_data)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Skipping track {row_data.get('item_id')} due to data parsing error.")
    if not valid_tracks:
        logger.error(f"{log_prefix} Iteration {run_idx}: No valid tracks could be processed.")
        return None, None, None
    return valid_tracks, np.array(X_feat_orig_list), np.array(X_embed_raw_list) if use_embeddings else None

def _prepare_and_scale_data(X_feat, X_embed, use_embeddings):
    """Selects the data source for clustering (features or embeddings) and scales it."""
    data_source = X_embed if use_embeddings else X_feat
    if data_source is None or data_source.shape[0] == 0:
        return None, None
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_source)
    return scaled_data, scaler

# --- Step 2: Evolutionary Parameter Generation ---

def _mutate_param(value, min_val, max_val, delta, is_float=False):
    """Applies a random mutation to a single parameter value."""
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
    """Decides to explore (random params) or exploit (mutate elite params)."""
    if elites and random.random() < exploitation_prob:
        chosen_elite = random.choice(elites)
        return _mutate_parameters(chosen_elite, mutation_cfg, method, data, *args)
    return _generate_random_parameters(method, data, *args)

def _generate_random_parameters(method, data, pca_ranges, num_clust_ranges, db_ranges, gmm_ranges, spec_ranges, *args):
    """Generates a completely new set of random parameters for clustering."""
    max_pca = min(pca_ranges['components_max'], data.shape[1], data.shape[0] - 1)
    min_pca = pca_ranges['components_min']
    if min_pca > max_pca:
        min_pca = max_pca
    
    pca_comps = random.randint(min_pca, max_pca) if max_pca >= min_pca and max_pca > 0 else min_pca
    pca_config = {"enabled": pca_comps > 0, "components": pca_comps}
    
    max_k = data.shape[0]
    method_params = {}

    if method == 'kmeans':
        upper_k = min(num_clust_ranges[1], max_k)
        lower_k = min(num_clust_ranges[0], upper_k)
        if lower_k < 2 and upper_k >= 2: lower_k = 2
        if upper_k < lower_k: upper_k = lower_k
        k = random.randint(lower_k, upper_k) if upper_k >= lower_k and upper_k > 0 else lower_k
        method_params = {"n_clusters": k}

    elif method == 'dbscan':
        eps = round(random.uniform(db_ranges['eps_min'], db_ranges['eps_max']), 2)
        min_samples = random.randint(db_ranges['samples_min'], db_ranges['samples_max'])
        method_params = {"eps": eps, "min_samples": min_samples}

    elif method == 'gmm':
        upper_k = min(gmm_ranges['n_components_max'], max_k)
        lower_k = min(gmm_ranges['n_components_min'], upper_k)
        if lower_k < 2 and upper_k >= 2: lower_k = 2
        if upper_k < lower_k: upper_k = lower_k
        n_comp = random.randint(lower_k, upper_k) if upper_k >= lower_k and upper_k > 0 else lower_k
        method_params = {"n_components": n_comp}

    elif method == 'spectral':
        upper_k = min(spec_ranges['n_clusters_max'], data.shape[0] - 1)
        lower_k = spec_ranges['n_clusters_min']
        if lower_k < 2: lower_k = 2
        if upper_k < lower_k: upper_k = lower_k
        n_clust = random.randint(lower_k, upper_k) if upper_k >= lower_k else lower_k
        method_params = {"n_clusters": n_clust, "random_state": random.randint(0, 10000)}
        
    return {"pca_config": pca_config, "clustering_method_config": {"method": method, "params": method_params}}

def _mutate_parameters(elite_params, mutation_cfg, method, data, pca_ranges, num_clust_ranges, db_ranges, gmm_ranges, spec_ranges, *args):
    """Takes an elite parameter set and applies small random changes."""
    elite_pca_cfg = elite_params['pca_config']
    elite_method_cfg = elite_params['clustering_method_config']
    
    max_pca = min(pca_ranges['components_max'], data.shape[1], data.shape[0] - 1)
    min_pca = pca_ranges['components_min']
    if min_pca > max_pca:
        min_pca = max_pca
    mutated_pca_comps = _mutate_param(elite_pca_cfg.get('components', 0), min_pca, max_pca, mutation_cfg.get('int_abs_delta', 2))
    pca_config = {"enabled": mutated_pca_comps > 0, "components": mutated_pca_comps}
    
    max_k = data.shape[0]
    method_params = {}

    if method == 'kmeans':
        upper_k = min(num_clust_ranges[1], max_k)
        lower_k = min(num_clust_ranges[0], upper_k)
        k = _mutate_param(elite_method_cfg['params']['n_clusters'], lower_k, upper_k, mutation_cfg.get('int_abs_delta', 2))
        method_params = {"n_clusters": k}
    elif method == 'dbscan':
        mutated_eps = _mutate_param(elite_method_cfg['params']['eps'], db_ranges['eps_min'], db_ranges['eps_max'], mutation_cfg.get('float_abs_delta', 0.1), is_float=True)
        mutated_min_samples = _mutate_param(elite_method_cfg['params']['min_samples'], db_ranges['samples_min'], db_ranges['samples_max'], mutation_cfg.get('int_abs_delta', 2))
        method_params = {"eps": mutated_eps, "min_samples": mutated_min_samples}
    elif method == 'gmm':
        upper_k = min(gmm_ranges['n_components_max'], max_k)
        lower_k = min(gmm_ranges['n_components_min'], upper_k)
        n_comp = _mutate_param(elite_method_cfg['params']['n_components'], lower_k, upper_k, mutation_cfg.get('int_abs_delta', 2))
        method_params = {"n_components": n_comp}
    elif method == 'spectral':
        upper_k = min(spec_ranges['n_clusters_max'], max_k - 1)
        lower_k = spec_ranges['n_clusters_min']
        if lower_k < 2: lower_k = 2
        if upper_k < lower_k: upper_k = lower_k
        n_clust = _mutate_param(elite_method_cfg['params']['n_clusters'], lower_k, upper_k, mutation_cfg.get('int_abs_delta', 2))
        elite_random_state = elite_method_cfg['params'].get("random_state", random.randint(0, 10000))
        mutated_random_state = _mutate_param(elite_random_state, 0, 10000, mutation_cfg.get("int_abs_delta", 100))
        method_params = {"n_clusters": n_clust, "random_state": mutated_random_state}
        
    return {"pca_config": pca_config, "clustering_method_config": {"method": method, "params": method_params}}

# --- Step 3 & 4: Apply Models ---

def _apply_clustering_model(data, method_config, log_prefix, run_idx):
    """Initializes and fits the specified clustering model."""
    method = method_config['method']
    params = method_config['params']
    model = None
    try:
        if method == 'kmeans':
            if params.get('n_clusters', 0) < 2:
                return None, None, None
            model = KMeans(n_clusters=params['n_clusters'], init='k-means++', n_init=10)

        elif method == 'dbscan':
            model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])

        elif method == 'gmm':
            if params.get('n_components', 0) < 2 or params['n_components'] > data.shape[0]:
                return None, None, None
            model = GaussianMixture(
                n_components=params['n_components'],
                covariance_type=GMM_COVARIANCE_TYPE,
                init_params='k-means++',
                n_init=10,
                random_state=None,
                reg_covar=1e-4
            )

        elif method == 'spectral':
            if params.get('n_clusters', 0) < 2 or params['n_clusters'] >= data.shape[0]:
                return None, None, None
            model = SpectralClustering(
                n_clusters=params['n_clusters'],
                assign_labels='kmeans',
                affinity='nearest_neighbors',
                n_neighbors=SPECTRAL_N_NEIGHBORS,
                random_state=params.get("random_state"),
                n_init=10,
                verbose=False
            )
        
        if model is None:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        labels = model.fit_predict(data)
        
        centers = {}
        if hasattr(model, 'cluster_centers_') and model.cluster_centers_ is not None:
            centers = {i: center for i, center in enumerate(model.cluster_centers_)}
        elif hasattr(model, 'means_') and model.means_ is not None:
            centers = {i: mean for i, mean in enumerate(model.means_)}
        else: # Fallback for DBSCAN and Spectral
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            for label in unique_labels:
                cluster_points = data[labels == label]
                if cluster_points.shape[0] > 0:
                    centers[label] = cluster_points.mean(axis=0)

        return labels, centers, model

    except Exception as e:
        logger.error(f"{log_prefix} Iteration {run_idx}: Clustering model failed for method {method}", exc_info=True)
        return None, None, None

def _get_feature_centroid_for_embedding_cluster(label_id, labels, X_feat_orig):
    """
    When clustering on embeddings, this calculates a representative centroid
    in the original feature space for naming and analysis.
    """
    cluster_indices = np.where(labels == label_id)[0]
    if len(cluster_indices) == 0:
        return None
    
    feature_vectors_in_cluster = X_feat_orig[cluster_indices]
    feature_centroid = np.mean(feature_vectors_in_cluster, axis=0)
    return feature_centroid

# --- Step 5 & 6: Formatting and Scoring ---

def _format_and_score_iteration_result(
    labels, valid_tracks, X_feat_orig, data_for_metrics,
    centers, model, pca, scaler, active_moods, 
    params, max_songs_per_cluster, run_idx, use_embeddings, score_weights, log_prefix):
    """
    Packages all results from the iteration into a dictionary and calculates the final fitness score.
    This version includes the advanced filtering and scoring logic.
    """
    if labels is None:
        return {"fitness_score": -1.0}

    # --- 1. Filter clusters to create final playlists ---
    raw_distances = np.full(len(valid_tracks), np.inf)
    if len(set(labels) - {-1}) > 0:
        for label_id in set(labels):
            if label_id == -1: continue
            indices = np.where(labels == label_id)[0]
            if len(indices) > 0 and label_id in centers:
                cluster_center = centers[label_id]
                points = data_for_metrics[indices]
                distances = np.linalg.norm(points - cluster_center, axis=1)
                raw_distances[indices] = distances

    max_dist_val = raw_distances[raw_distances != np.inf].max() if np.any(raw_distances != np.inf) else 1.0
    if max_dist_val == 0: max_dist_val = 1.0
    normalized_distances = raw_distances / max_dist_val

    track_info_list = [{"row": valid_tracks[i], "label": labels[i], "distance": normalized_distances[i]} for i in range(len(valid_tracks))]

    filtered_clusters = defaultdict(list)
    for cid in set(labels):
        if cid == -1: continue
        cluster_tracks_info = [t_info for t_info in track_info_list if t_info["label"] == cid and t_info["distance"] <= MAX_DISTANCE]
        if not cluster_tracks_info: continue
        
        cluster_tracks_info.sort(key=lambda x: x["distance"])
        count_per_artist = defaultdict(int)
        selected_tracks_for_playlist = []
        for t_item_info in cluster_tracks_info:
            author = t_item_info["row"]["author"]
            if count_per_artist[author] < MAX_SONGS_PER_ARTIST:
                selected_tracks_for_playlist.append(t_item_info)
                count_per_artist[author] += 1
            if max_songs_per_cluster > 0 and len(selected_tracks_for_playlist) >= max_songs_per_cluster:
                break
        
        for t_item_info_final in selected_tracks_for_playlist:
            item_id_val, title_val, author_val = t_item_info_final["row"]["item_id"], t_item_info_final["row"]["title"], t_item_info_final["row"]["author"]
            filtered_clusters[cid].append((item_id_val, title_val, author_val))

    # --- 2. Format final playlists and centroids ---
    named_playlists, playlist_centroids = {}, {}
    # *** NEW: Map final playlist names to their centroid vectors for Top-N selection ***
    playlist_to_centroid_vector_map = {}
    unique_predominant_mood_scores = {}
    unique_predominant_other_feature_scores = {}
    item_id_to_song_index_map = {track_data['item_id']: i for i, track_data in enumerate(valid_tracks)}

    for label_id, songs_list in filtered_clusters.items():
        if songs_list and label_id in centers:
            center_vec = centers[label_id] # This is the vector in the clustered space
            if use_embeddings:
                feature_centroid_vec = _get_feature_centroid_for_embedding_cluster(label_id, labels, X_feat_orig)
                if feature_centroid_vec is None: continue
                name, centroid_details = _name_cluster(feature_centroid_vec, None, False, active_moods, None)
            else:
                name, centroid_details = _name_cluster(center_vec, pca, params['pca_config']['enabled'], active_moods, scaler)

            temp_name, suffix = name, 1
            while temp_name in named_playlists:
                temp_name = f"{name}_{suffix}"
                suffix += 1
            
            named_playlists[temp_name] = songs_list
            playlist_centroids[temp_name] = centroid_details
            # *** NEW: Store the mapping from the final unique name to the centroid vector ***
            playlist_to_centroid_vector_map[temp_name] = center_vec

            if centroid_details and any(mood in active_moods for mood in centroid_details.keys()):
                predominant_mood_key = max((k for k in centroid_details if k in MOOD_LABELS), key=centroid_details.get, default=None)
                if predominant_mood_key:
                    current_mood_score = centroid_details.get(predominant_mood_key, 0.0)
                    unique_predominant_mood_scores[predominant_mood_key] = max(unique_predominant_mood_scores.get(predominant_mood_key, 0.0), current_mood_score)
            
            centroid_other_features = {lk: centroid_details.get(lk, 0.0) for lk in OTHER_FEATURE_LABELS if lk in centroid_details}
            if centroid_other_features:
                predominant_other_key = max(centroid_other_features, key=centroid_other_features.get, default=None)
                if predominant_other_key and centroid_other_features[predominant_other_key] > OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY:
                     unique_predominant_other_feature_scores[predominant_other_key] = max(unique_predominant_other_feature_scores.get(predominant_other_key, 0.0), centroid_other_features[predominant_other_key])

    # --- 3. Calculate All Metrics ---
    metrics = {"silhouette": 0.0, "davies_bouldin": 0.0, "calinski_harabasz": 0.0, "mood_diversity": 0.0, "mood_purity": 0.0, "other_feature_diversity": 0.0, "other_feature_purity": 0.0}
    num_clusters = len(named_playlists)

    if num_clusters >= 2 and num_clusters < data_for_metrics.shape[0]:
        if score_weights.get('silhouette', 0) > 0:
            try: metrics['silhouette'] = (silhouette_score(data_for_metrics, labels) + 1) / 2.0
            except ValueError: pass
        if score_weights.get('davies_bouldin', 0) > 0:
            try: metrics['davies_bouldin'] = 1.0 / (1.0 + davies_bouldin_score(data_for_metrics, labels))
            except ValueError: pass
        if score_weights.get('calinski_harabasz', 0) > 0:
            try: metrics['calinski_harabasz'] = 1.0 - np.exp(-calinski_harabasz_score(data_for_metrics, labels) / 500.0)
            except ValueError: pass

    raw_mood_diversity_score = sum(unique_predominant_mood_scores.values())
    ln_mood_diversity = np.log1p(raw_mood_diversity_score)
    diversity_stats = LN_MOOD_DIVERSITY_EMBEDING_STATS if use_embeddings else LN_MOOD_DIVERSITY_STATS
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
            if not centroid_data or not songs: continue
            
            sorted_moods = sorted([(m,s) for m,s in centroid_data.items() if m in MOOD_LABELS], key=lambda item: item[1], reverse=True)
            top_moods = [m for m, s in sorted_moods[:TOP_K_MOODS_FOR_PURITY_CALCULATION] if s > 0.01]
            if not top_moods: continue

            song_purity_scores = []
            for item_id, _, _ in songs:
                song_idx = item_id_to_song_index_map.get(item_id)
                if song_idx is not None and song_idx < X_feat_orig.shape[0]:
                    song_feat_vec = X_feat_orig[song_idx]
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
            if not centroid_data or not songs: continue

            other_features = {k: v for k, v in centroid_data.items() if k in OTHER_FEATURE_LABELS}
            if not other_features: continue
            
            predominant_other = max(other_features, key=other_features.get, default=None)
            if not predominant_other or other_features[predominant_other] < OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY:
                continue

            try:
                feature_idx = OTHER_FEATURE_LABELS.index(predominant_other)
                song_purity_scores = []
                for item_id, _, _ in songs:
                    song_idx = item_id_to_song_index_map.get(item_id)
                    if song_idx is not None and song_idx < X_feat_orig.shape[0]:
                        song_feat_vec = X_feat_orig[song_idx]
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

    # --- 4. Calculate Final Score ---
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

    # --- 5. Package Final Result ---
    logger.info(f"Run {run_idx}: Created {len(named_playlists)} clusters.")
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
        "playlist_to_centroid_vector_map": playlist_to_centroid_vector_map, # *** NEW: Return the map ***
        "parameters": {**params, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx},
        "scaler_details": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()} if scaler else None,
        "pca_model_details": {"components": pca.components_.tolist(), "variance": pca.explained_variance_ratio_.tolist()} if pca else None
    }

def _name_cluster(centroid_vector, pca_model, pca_enabled, mood_labels, scaler):
    """Generates a human-readable name for a cluster based on its centroid."""
    # Constants for naming logic
    TOP_MOODS_IN_NAME = 3
    OTHER_FEATURE_THRESHOLD_FOR_NAME = 0.5
    MAX_OTHER_FEATURES_IN_NAME = 2

    # If scaler is None, the vector is already in the original feature space (e.g., from embedding cluster)
    if scaler:
        vec = centroid_vector.reshape(1, -1)
        if pca_enabled and pca_model:
            vec = pca_model.inverse_transform(vec)
        interpreted_vector = scaler.inverse_transform(vec)[0]
    else:
        interpreted_vector = centroid_vector
    
    # --- Extract features from the vector ---
    tempo_val = interpreted_vector[0]
    mood_values = interpreted_vector[2 : 2 + len(mood_labels)]
    
    # --- Build Name Components ---
    tempo_label = "Slow" if tempo_val < 0.33 else "Medium" if tempo_val < 0.66 else "Fast"
    
    if len(mood_values) > 0 and np.sum(mood_values) > 0:
        top_mood_indices = np.argsort(mood_values)[::-1][:TOP_MOODS_IN_NAME]
        mood_names = [mood_labels[i].title() for i in top_mood_indices if i < len(mood_labels) and mood_values[i] > 0.01]
        mood_part = "_".join(mood_names) if mood_names else "Mixed"
    else:
        mood_part = "Mixed"
    
    base_name = f"{mood_part}_{tempo_label}"

    # --- Extract "Other Features" and add them to the name and details dict ---
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
                [(feature, score) for feature, score in other_feature_scores_dict.items() if score >= OTHER_FEATURE_THRESHOLD_FOR_NAME],
                key=lambda item: item[1],
                reverse=True
            )
            features_to_add = [feature.title() for feature, score in prominent_features[:MAX_OTHER_FEATURES_IN_NAME]]
            if features_to_add:
                appended_other_features_str = "_" + "_".join(features_to_add)

    final_name = f"{base_name}{appended_other_features_str}"
    
    return final_name, details

# --- Other Helpers ---

def get_job_result_safely(job_id, parent_task_id, task_type="child task"):
    """Safely retrieves the result of an RQ job, checking both RQ and the database."""
    # Local imports to prevent circular dependency
    from app import app, JobStatus
    from app_helper import redis_conn, get_task_info_from_db, TASK_STATUS_SUCCESS

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
                    return details.get('full_best_result_from_batch') or details.get('full_result')
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Could not parse result from DB for job {job_id}")
    return None

def _get_stratified_song_subset(genre_map, target_per_genre, prev_ids=None, percent_change=0.0):
    """Generates a stratified sample of songs, perturbing a previous subset if provided."""
    new_subset, new_ids = [], set()
    if prev_ids and percent_change > 0:
        sample_size = int(len(prev_ids) * (1.0 - percent_change))
        if len(prev_ids) > sample_size:
            kept_ids = set(random.sample(prev_ids, sample_size))
        else:
            kept_ids = set(prev_ids)
    else:
        kept_ids = set()
    
    id_to_track_map = {t['item_id']: t for g_list in genre_map.values() for t in g_list}
    for track_id in kept_ids:
        if track_id in id_to_track_map:
            new_subset.append(id_to_track_map[track_id])
            new_ids.add(track_id)

    for genre in STRATIFIED_GENRES:
        current_genre_count = sum(1 for t in new_subset if _get_track_primary_genre(t) == genre)
        needed = target_per_genre - current_genre_count
        if needed > 0:
            candidates = [t for t in genre_map.get(genre, []) if t['item_id'] not in new_ids]
            if candidates:
                added_tracks = random.sample(candidates, min(needed, len(candidates)))
                new_subset.extend(added_tracks)
                for t in added_tracks: new_ids.add(t['item_id'])
    random.shuffle(new_subset)
    return new_subset

def _get_track_primary_genre(track_data):
    """Helper to determine the primary stratified genre for a track."""
    if 'mood_vector' in track_data and track_data['mood_vector']:
        mood_scores = {p.split(':')[0]: float(p.split(':')[1]) for p in track_data['mood_vector'].split(',') if ':' in p}
        return max((g for g in STRATIFIED_GENRES if g in mood_scores), key=mood_scores.get, default='__other__')
    return '__other__'

# Post-processing functions have been moved to clustering_postprocessing.py for better organization
