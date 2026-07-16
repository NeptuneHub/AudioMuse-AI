# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Clustering orchestrator: evolutionary search that turns embeddings into playlists.

The main clustering RQ job. run_clustering_task runs the WHOLE pipeline once per
target server, sequentially: each server clusters only its own availability-scoped
catalogue (servers hold different libraries), runs its own evolutionary/elitist
search via run_clustering_batch_task child jobs, and gets its own playlists -
results are never computed once and pushed to other servers. Delegates the
per-iteration work to clustering_helper, the models to clustering_gpu, and
dedup/size/diversity filtering to clustering_postprocessing.

Main Features:
* run_clustering_task: sequential per-server loop over the requested scope
  (a specific server, 'default', or 'all'); _cluster_one_server runs one
  server's full pipeline with per-server batch job ids.
* Per-server persistence: each server's playlists replace ITS OWN rows in the
  playlist table (bare names + server_id) as soon as it succeeds, and every run
  starts by pruning rows of servers no longer configured - the table is always
  the last run per server, never a growing history.
* _monitor_and_process_batches / _launch_batch_job: fan out parameter sets into
  batch jobs, track elites, and adapt sampling each generation. After
  CLUSTERING_EARLY_STOP_BATCHES consecutive batches without a better result no
  new batches are enqueued; in-flight ones drain and the best result stands.
* Genre-stratified sampling (_prepare_genre_map, _calculate_target_songs_per_genre)
  so playlists span the library rather than one dominant genre.
* _calibrate_cluster_params: per-server auto-tuning for EVERY algorithm via up
  to CLUSTERING_CALIBRATION_MAX_TRIES quick single-iteration probes. KMeans,
  GMM and Spectral tune their own cluster/component range plus the sampling
  percentile: small libraries pin the range to top_n_playlists clusters
  directly (never above subset_size / (2 * MIN_PLAYLIST_SIZE_FOR_TOP_N), never
  below subset_size / CLUSTERING_MAX_PLAYLIST_SONGS) and each probe runs at
  the TOP of the range (worst case for emptiness). DBSCAN has no cluster
  count: its eps range is DERIVED from the data instead (k-distance heuristic
  via _derive_dbscan_eps - the configured 0.1-0.5 default is unusable in the
  ~200-dim embedding space where every point would be noise), oversized
  components are re-split by KMeans in clustering_helper, and probes widen
  eps when playlists come out tiny and tighten it when oversized, alongside
  the percentile. A probe only passes with at
  least top_n_playlists playlists of MIN_PLAYLIST_SIZE_FOR_TOP_N+ songs;
  otherwise clusters shrink toward the goal and the percentile rises to
  sample more of the library. Oversized probes (over
  CLUSTERING_MAX_PLAYLIST_SONGS) grow clusters/lower percentile; big beats
  empty. On probe failure the library-size cap still applies. Calibration is
  skipped on crash-recovery resumes (existing batch children).
* _name_and_prepare_playlists: score, name (optionally via AI) and persist results;
  app imports are deferred inside functions to avoid circular imports.
"""

from collections import defaultdict
import numpy as np
import json
import time
import logging
import uuid
import traceback

from rq import get_current_job, Retry
from rq.job import Job
from rq.exceptions import NoSuchJobError

from psycopg2.extras import DictCursor

from config import (
    MAX_SONGS_PER_CLUSTER,
    TOP_N_PLAYLISTS,
    CLUSTERING_AUTO_CALIBRATION,
    MOOD_LABELS,
    STRATIFIED_GENRES,
    MUTATION_KMEANS_COORD_FRACTION,
    MUTATION_INT_ABS_DELTA,
    MUTATION_FLOAT_ABS_DELTA,
    TOP_N_ELITES,
    EXPLOITATION_START_FRACTION,
    EXPLOITATION_PROBABILITY_CONFIG,
    SAMPLING_PERCENTAGE_CHANGE_PER_RUN,
    ITERATIONS_PER_BATCH_JOB,
    MAX_CONCURRENT_BATCH_JOBS,
    MIN_PLAYLIST_SIZE_FOR_TOP_N,
    CLUSTERING_BATCH_TIMEOUT_MINUTES,
    CLUSTERING_MAX_FAILED_BATCHES,
    CLUSTERING_CLEANING,
    CLUSTERING_MAX_PLAYLIST_SONGS,
    CLUSTERING_CALIBRATION_MAX_TRIES,
    CLUSTERING_EARLY_STOP_BATCHES,
    TASK_STATUS_STARTED,
    TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
    TASK_STATUS_REVOKED,
)

from error import error_manager
from error.error_dictionary import ERR_CLUSTERING_FAILED

from app_helper import (
    save_task_status,
    redis_conn,
    get_task_info_from_db,
    get_db,
    rq_queue_default,
)
from database import (
    update_playlist_table,
    prune_playlist_rows_for_missing_servers,
    get_child_tasks_from_db,
)

from sanitization import sanitize_for_json

from .mediaserver import create_playlist, delete_automatic_playlists
from .mediaserver import registry
from sklearn.neighbors import NearestNeighbors

from .clustering_helper import (
    _get_stratified_song_subset,
    get_job_result_safely,
    _perform_single_clustering_iteration,
    _prepare_iteration_data,
    _prepare_and_scale_data,
    _shuffle_playlist_songs,
    _assign_playlist_chunks,
    _try_ai_name_playlist,
)
from .clustering_postprocessing import (
    apply_duplicate_filtering_to_clustering_result,
    apply_minimum_size_filter_to_clustering_result,
    select_top_n_diverse_playlists,
)

logger = logging.getLogger(__name__)


def _derive_dbscan_eps(item_ids, min_samples, active_moods, enable_embeddings):
    valid_tracks, x_feat, x_embed = _prepare_iteration_data(
        item_ids, active_moods, enable_embeddings, '[Calibration]', 0
    )
    if valid_tracks is None:
        return None
    data, _scaler = _prepare_and_scale_data(x_feat, x_embed, enable_embeddings)
    if data is None or data.shape[0] <= min_samples:
        return None
    if len(data) > 1000:
        picks = np.random.default_rng(0).choice(len(data), 1000, replace=False)
        data = data[picks]
    neighbors = NearestNeighbors(n_neighbors=min(min_samples + 1, len(data))).fit(data)
    distances, _indices = neighbors.kneighbors(data)
    kdist = distances[:, -1]
    eps_low = max(0.05, float(np.percentile(kdist, 50)))
    eps_high = max(eps_low * 1.2, float(np.percentile(kdist, 90)))
    return eps_low, eps_high


def _viable_playlists(result, top_n=TOP_N_PLAYLISTS):
    playlists = (result or {}).get('named_playlists') or {}
    keepers = sum(
        1 for songs in playlists.values() if len(songs) >= MIN_PLAYLIST_SIZE_FOR_TOP_N
    )
    return min(keepers, max(1, top_n))


def batch_task_failure_handler(job, connection, type, value, tb):
    from flask_app import app

    with app.app_context():
        task_id = getattr(job, 'id', None) or getattr(job, 'get_id', lambda: None)()
        parent_id = job.kwargs.get('parent_task_id')
        batch_id_str = job.kwargs.get('batch_id_str')

        tb_formatted = ""
        if isinstance(tb, traceback.StackSummary):
            tb_formatted = "".join(tb.format())
        else:
            tb_formatted = "".join(traceback.format_exception(type, value, tb))

        error_details = {
            "message": "Clustering batch sub-task failed permanently after all retries.",
            "error": error_manager.build(ERR_CLUSTERING_FAILED, str(value)),
            "error_type": str(type.__name__),
            "error_value": str(value),
        }

        save_task_status(
            task_id,
            "clustering_batch",
            TASK_STATUS_FAILURE,
            parent_task_id=parent_id,
            sub_type_identifier=batch_id_str,
            progress=100,
            details=error_details,
        )
        app.logger.error(
            f"Clustering batch task {task_id} (parent: {parent_id}) failed permanently. DB status updated.\n{tb_formatted}"
        )


def run_clustering_batch_task(
    batch_id_str,
    start_run_idx,
    num_iterations_in_batch,
    genre_to_lightweight_track_data_map_json,
    target_songs_per_genre,
    sampling_percentage_change_per_run,
    clustering_method,
    active_mood_labels_for_batch,
    num_clusters_min_max_tuple,
    dbscan_params_ranges_dict,
    gmm_params_ranges_dict,
    spectral_params_ranges_dict,
    pca_params_ranges_dict,
    max_songs_per_cluster,
    parent_task_id,
    score_weights_dict,
    elite_solutions_params_list_json,
    exploitation_probability,
    mutation_config_json,
    initial_subset_track_ids_json,
    enable_clustering_embeddings_param,
    top_n_playlists_param=TOP_N_PLAYLISTS,
):
    from flask_app import app

    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())
    logger.info(f"Starting clustering batch task {current_task_id} (Batch: {batch_id_str})")

    with app.app_context():

        def _log_and_update(message, progress, details=None, state=TASK_STATUS_PROGRESS):
            logger.info(f"[ClusteringBatchTask-{current_task_id}] {message}")
            db_details = {
                "batch_id": batch_id_str,
                "start_run_idx": start_run_idx,
                "num_iterations_in_batch": num_iterations_in_batch,
                "status_message": message,
                **(details or {}),
            }
            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.save_meta()
            save_task_status(
                current_task_id,
                "clustering_batch",
                state,
                parent_task_id=parent_task_id,
                sub_type_identifier=batch_id_str,
                progress=progress,
                details=db_details,
            )

        try:
            _log_and_update("Batch started.", 0)
            genre_to_lightweight_track_data_map = json.loads(
                genre_to_lightweight_track_data_map_json
            )
            elite_solutions_params_list = json.loads(elite_solutions_params_list_json)
            mutation_config = json.loads(mutation_config_json)
            current_sampled_track_ids = json.loads(initial_subset_track_ids_json)

            best_result_in_batch = None
            best_score_in_batch = -1.0
            best_rank_in_batch = (-1, -1.0)
            iterations_completed = 0

            for i in range(num_iterations_in_batch):
                current_run_global_idx = start_run_idx + i

                if current_job:
                    # A MISSING row counts as revoked: the cancel wipes task_status,
                    # so a batch that can no longer find its own row was cancelled.
                    task_info = get_task_info_from_db(current_task_id)
                    parent_task_info = get_task_info_from_db(parent_task_id)
                    if (
                        task_info is None
                        or task_info.get('status') == TASK_STATUS_REVOKED
                        or (
                            parent_task_info
                            and parent_task_info.get('status')
                            in [TASK_STATUS_REVOKED, TASK_STATUS_FAILURE]
                        )
                    ):
                        _log_and_update(
                            "Stopping batch due to revocation.", i, state=TASK_STATUS_REVOKED
                        )
                        return {"status": "REVOKED", "message": "Batch task revoked."}

                percentage_change = 0.0 if i == 0 else sampling_percentage_change_per_run
                current_subset_lightweight_data = _get_stratified_song_subset(
                    genre_to_lightweight_track_data_map,
                    target_songs_per_genre,
                    prev_ids=current_sampled_track_ids,
                    percent_change=percentage_change,
                )
                item_ids_for_iteration = [t['item_id'] for t in current_subset_lightweight_data]
                current_sampled_track_ids = list(item_ids_for_iteration)

                if not item_ids_for_iteration:
                    logger.warning(
                        f"No songs in subset for iteration {current_run_global_idx}. Skipping."
                    )
                    continue

                iteration_result = _perform_single_clustering_iteration(
                    run_idx=current_run_global_idx,
                    item_ids_for_subset=item_ids_for_iteration,
                    clustering_method=clustering_method,
                    num_clusters_min_max=num_clusters_min_max_tuple,
                    dbscan_params_ranges=dbscan_params_ranges_dict,
                    gmm_params_ranges=gmm_params_ranges_dict,
                    spectral_params_ranges=spectral_params_ranges_dict,
                    pca_params_ranges=pca_params_ranges_dict,
                    active_mood_labels=active_mood_labels_for_batch,
                    max_songs_per_cluster=max_songs_per_cluster,
                    log_prefix=f"[Batch-{current_task_id}]",
                    elite_solutions_params_list=elite_solutions_params_list,
                    exploitation_probability=exploitation_probability,
                    mutation_config=mutation_config,
                    score_weights=score_weights_dict,
                    enable_clustering_embeddings=enable_clustering_embeddings_param,
                )
                iterations_completed += 1

                iteration_rank = (
                    _viable_playlists(iteration_result, top_n_playlists_param),
                    (iteration_result or {}).get("fitness_score", -1.0),
                )
                if (
                    iteration_result
                    and iteration_result.get("parameters")
                    and iteration_rank > best_rank_in_batch
                ):
                    best_rank_in_batch = iteration_rank
                    best_score_in_batch = iteration_result["fitness_score"]
                    best_result_in_batch = iteration_result

                progress = int(100 * (i + 1) / num_iterations_in_batch)
                _log_and_update(
                    f"Iteration {current_run_global_idx} complete. Batch best score: {best_score_in_batch:.2f}",
                    progress,
                )

            if best_result_in_batch:
                best_result_in_batch = sanitize_for_json(best_result_in_batch)

            final_details = {
                "best_score_in_batch": best_score_in_batch,
                "iterations_completed_in_batch": iterations_completed,
                "full_best_result_from_batch": best_result_in_batch,
                "final_subset_track_ids": current_sampled_track_ids,
            }
            _log_and_update(
                f"Batch complete. Best score: {best_score_in_batch:.2f}",
                100,
                details=final_details,
                state=TASK_STATUS_SUCCESS,
            )
            return {
                "status": "SUCCESS",
                "iterations_completed_in_batch": iterations_completed,
                "best_result_from_batch": best_result_in_batch,
                "final_subset_track_ids": current_sampled_track_ids,
            }

        except Exception as e:
            logger.exception(f"Clustering batch {batch_id_str} failed")
            err = error_manager.record(
                error_manager.classify(e, ERR_CLUSTERING_FAILED), str(e)
            )
            _log_and_update(
                f"Batch failed: {e}", 100, details={"error": err}, state=TASK_STATUS_FAILURE
            )
            return {"status": "FAILURE", "message": str(e)}


def run_clustering_task(
    clustering_method,
    num_clusters_min,
    num_clusters_max,
    dbscan_eps_min,
    dbscan_eps_max,
    dbscan_min_samples_min,
    dbscan_min_samples_max,
    pca_components_min,
    pca_components_max,
    num_clustering_runs,
    max_songs_per_cluster_val,
    gmm_n_components_min,
    gmm_n_components_max,
    spectral_n_clusters_min,
    spectral_n_clusters_max,
    min_songs_per_genre_for_stratification_param,
    stratified_sampling_target_percentile_param,
    score_weight_diversity_param,
    score_weight_silhouette_param,
    score_weight_davies_bouldin_param,
    score_weight_calinski_harabasz_param,
    score_weight_purity_param,
    score_weight_other_feature_diversity_param,
    score_weight_other_feature_purity_param,
    ai_model_provider_param,
    ollama_server_url_param,
    ollama_model_name_param,
    openai_server_url_param,
    openai_model_name_param,
    openai_api_key_param,
    gemini_api_key_param,
    gemini_model_name_param,
    mistral_api_key_param,
    mistral_model_name_param,
    top_n_moods_for_clustering_param,
    top_n_playlists_param,
    enable_clustering_embeddings_param,
    output_server_scope="all",
    auto_calibration_param=None,
):
    from flask_app import app

    if auto_calibration_param is None:
        auto_calibration_param = CLUSTERING_AUTO_CALIBRATION

    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())
    logger.info(f"Starting main clustering task {current_task_id}")

    _ai_naming_summary = {
        "OLLAMA": (ollama_server_url_param, ollama_model_name_param),
        "OPENAI": (openai_server_url_param, openai_model_name_param),
        "GEMINI": ("(gemini-api)", gemini_model_name_param),
        "MISTRAL": ("(mistral-api)", mistral_model_name_param),
    }.get(ai_model_provider_param, ("(none)", "(none)"))
    logger.info(
        "Clustering AI naming -> provider=%s url=%s model=%s",
        ai_model_provider_param,
        _ai_naming_summary[0],
        _ai_naming_summary[1],
    )

    initial_params = {
        "clustering_method": clustering_method,
        "pca_components_min": pca_components_min,
        "pca_components_max": pca_components_max,
        "use_embeddings": enable_clustering_embeddings_param,
        "top_n_playlists": top_n_playlists_param,
        "stratification_percentile": stratified_sampling_target_percentile_param,
        "score_weights": {
            "mood_diversity": score_weight_diversity_param,
            "silhouette": score_weight_silhouette_param,
            "davies_bouldin": score_weight_davies_bouldin_param,
            "calinski_harabasz": score_weight_calinski_harabasz_param,
            "mood_purity": score_weight_purity_param,
            "other_feature_diversity": score_weight_other_feature_diversity_param,
            "other_feature_purity": score_weight_other_feature_purity_param,
        },
    }
    if clustering_method == 'kmeans':
        initial_params["num_clusters_min"] = num_clusters_min
        initial_params["num_clusters_max"] = num_clusters_max
    elif clustering_method == 'gmm':
        initial_params["num_clusters_min"] = gmm_n_components_min
        initial_params["num_clusters_max"] = gmm_n_components_max
    elif clustering_method == 'spectral':
        initial_params["num_clusters_min"] = spectral_n_clusters_min
        initial_params["num_clusters_max"] = spectral_n_clusters_max

    with app.app_context():
        task_info = get_task_info_from_db(current_task_id)
        if task_info and task_info.get('status') in [
            TASK_STATUS_SUCCESS,
            TASK_STATUS_FAILURE,
            TASK_STATUS_REVOKED,
        ]:
            logger.info(
                f"Main clustering task {current_task_id} is already in a terminal state ('{task_info.get('status')}'). Skipping execution."
            )
            return {
                "status": task_info.get('status'),
                "message": f"Task already in terminal state '{task_info.get('status')}'.",
                "details": json.loads(task_info.get('details', '{}')),
            }

        _main_task_accumulated_details = {
            "log": [],
            "total_runs": num_clustering_runs,
            "runs_completed": 0,
            "best_score": -1.0,
            "best_result": None,
            "active_jobs": {},
            "elite_solutions": [],
            "last_subset_ids": [],
            "processed_job_ids": set(),
            "batch_start_times": {},
            "failed_batches": set(),
            "timed_out_batches": set(),
            "stale_batches": 0,
        }

        def _log_and_update(
            message, progress, details_to_add_or_update=None, task_state=TASK_STATUS_PROGRESS
        ):
            logger.info(f"[MainClusteringTask-{current_task_id}] {message}")
            if details_to_add_or_update:
                _main_task_accumulated_details.update(details_to_add_or_update)

            _main_task_accumulated_details["status_message"] = message

            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            _main_task_accumulated_details["log"].append(log_entry)

            details_for_db = _main_task_accumulated_details.copy()
            details_for_db.pop('active_jobs', None)
            details_for_db.pop('best_result', None)
            details_for_db.pop('last_subset_ids', None)
            details_for_db.pop('processed_job_ids', None)
            details_for_db.pop('failed_batches', None)
            details_for_db.pop('timed_out_batches', None)
            details_for_db.pop('batch_start_times', None)

            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.save_meta()

            save_task_status(
                current_task_id,
                "main_clustering",
                task_state,
                progress=progress,
                details=details_for_db,
            )

        try:
            _log_and_update(
                f"Initializing clustering process ({clustering_method})...",
                0,
                task_state=TASK_STATUS_STARTED,
            )

            target_servers = registry.servers_for_scope(output_server_scope)
            if not target_servers:
                raise ValueError(f"No music servers match scope '{output_server_scope}'.")

            if output_server_scope == 'all' and all(target_servers):
                prune_playlist_rows_for_missing_servers(
                    [s['server_id'] for s in target_servers]
                )

            multi_server = len(target_servers) > 1
            server_span = 92.0 / len(target_servers)
            per_server_summary = []
            best_score_overall = -1.0
            best_params_overall = None

            for server_idx, target_server in enumerate(target_servers):
                server_name = target_server['name'] if target_server else 'default server'
                report = _make_server_reporter(
                    _log_and_update,
                    server_name if multi_server else None,
                    3.0 + server_idx * server_span,
                    server_span,
                )

                _main_task_accumulated_details.update({
                    "runs_completed": 0,
                    "best_score": -1.0,
                    "best_result": None,
                    "active_jobs": {},
                    "elite_solutions": [],
                    "last_subset_ids": [],
                    "processed_job_ids": set(),
                    "batch_start_times": {},
                    "failed_batches": set(),
                    "timed_out_batches": set(),
                    "stale_batches": 0,
                    "job_prefix": f"{current_task_id}_s{server_idx}",
                })

                try:
                    status, payload = _cluster_one_server(
                        target_server,
                        _main_task_accumulated_details,
                        report,
                        current_job,
                        current_task_id,
                        clustering_method,
                        num_clusters_min,
                        num_clusters_max,
                        dbscan_eps_min,
                        dbscan_eps_max,
                        dbscan_min_samples_min,
                        dbscan_min_samples_max,
                        pca_components_min,
                        pca_components_max,
                        num_clustering_runs,
                        max_songs_per_cluster_val,
                        gmm_n_components_min,
                        gmm_n_components_max,
                        spectral_n_clusters_min,
                        spectral_n_clusters_max,
                        min_songs_per_genre_for_stratification_param,
                        stratified_sampling_target_percentile_param,
                        score_weight_diversity_param,
                        score_weight_silhouette_param,
                        score_weight_davies_bouldin_param,
                        score_weight_calinski_harabasz_param,
                        score_weight_purity_param,
                        score_weight_other_feature_diversity_param,
                        score_weight_other_feature_purity_param,
                        ai_model_provider_param,
                        ollama_server_url_param,
                        ollama_model_name_param,
                        openai_server_url_param,
                        openai_model_name_param,
                        openai_api_key_param,
                        gemini_api_key_param,
                        gemini_model_name_param,
                        mistral_api_key_param,
                        mistral_model_name_param,
                        top_n_moods_for_clustering_param,
                        top_n_playlists_param,
                        enable_clustering_embeddings_param,
                        auto_calibration_param,
                    )
                except Exception as exc:
                    logger.exception(
                        "Clustering failed on server '%s'; continuing with the "
                        "remaining servers", server_name,
                    )
                    per_server_summary.append(
                        {'server': server_name, 'status': 'failed', 'reason': str(exc)}
                    )
                    continue

                if status == 'revoked':
                    return {"status": "REVOKED", "message": "Main clustering task revoked."}
                if status != 'success':
                    per_server_summary.append(
                        {'server': server_name, 'status': status, 'reason': payload}
                    )
                    continue
                try:
                    update_playlist_table(
                        payload['playlists'],
                        target_server['server_id'] if target_server else None,
                    )
                except Exception:
                    logger.exception(
                        "Persisting playlists failed for server '%s'; the previous "
                        "run's rows were kept", server_name,
                    )
                    per_server_summary.append({
                        'server': server_name,
                        'status': 'failed',
                        'reason': 'playlist persistence failed; previous run kept in database',
                    })
                    continue
                if payload['best_score'] > best_score_overall:
                    best_score_overall = payload['best_score']
                    best_params_overall = payload['best_params']
                per_server_summary.append({
                    'server': server_name,
                    'status': 'success',
                    'best_score': payload['best_score'],
                    'best_params': payload['best_params'],
                    'calibrated_params': payload.get('calibrated_params'),
                    'playlists_created': len(payload['playlists']),
                    'playlist_names': sorted(payload['playlists'].keys()),
                })

            successes = [s for s in per_server_summary if s['status'] == 'success']
            if not successes:
                raise ValueError(
                    "No valid clustering solution found on any server: "
                    + "; ".join(
                        f"{s['server']}: {s.get('reason')}" for s in per_server_summary
                    )
                )

            final_message = (
                f"Clustering task completed successfully on {len(successes)}/"
                f"{len(target_servers)} server(s)!"
            )

            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {final_message}"
            _main_task_accumulated_details["log"].append(log_entry)
            logger.info(f"[MainClusteringTask-{current_task_id}] {final_message}")

            final_log = _main_task_accumulated_details.get('log', [])
            truncated_log = final_log[-10:]

            final_db_summary = {
                "status_message": final_message,
                "running_parameters": initial_params,
                "best_score": best_score_overall,
                "best_params": best_params_overall,
                "num_playlists_created": sum(
                    s.get('playlists_created', 0) for s in successes
                ),
                "per_server": per_server_summary,
                "log": truncated_log,
                "log_storage_info": f"Log truncated to last {len(truncated_log)} entries. Original length: {len(final_log)}."
                if len(final_log) > 10
                else "Full log.",
            }

            if current_job:
                current_job.meta['progress'] = 100
                current_job.meta['status_message'] = final_message
                current_job.save_meta()

            save_task_status(
                current_task_id,
                "main_clustering",
                TASK_STATUS_SUCCESS,
                progress=100,
                details=final_db_summary,
            )

            return {
                "status": "SUCCESS",
                "message": f"Playlists created per server. Best score: {best_score_overall:.2f}",
            }

        except Exception as e:
            logger.critical("FATAL ERROR in main clustering task", exc_info=True)
            err = error_manager.record(
                error_manager.classify(e, ERR_CLUSTERING_FAILED), str(e)
            )
            _log_and_update(
                f"Task failed: {e}",
                100,
                details_to_add_or_update={"error": err},
                task_state=TASK_STATUS_FAILURE,
            )
            raise


def _make_server_reporter(log_and_update, server_label, base_progress, span):
    def report(message, local_pct, task_state=TASK_STATUS_PROGRESS):
        scoped = f"[{server_label}] {message}" if server_label else message
        pct = base_progress + (max(0.0, min(100.0, float(local_pct))) / 100.0) * span
        log_and_update(scoped, pct, task_state=task_state)

    return report


def _calibrate_cluster_params(
    clustering_method,
    genre_map,
    cluster_range_min,
    cluster_range_max,
    percentile,
    min_songs_per_genre,
    dbscan_eps_min,
    dbscan_eps_max,
    dbscan_min_samples_min,
    dbscan_min_samples_max,
    pca_components_min,
    pca_components_max,
    max_songs_per_cluster_val,
    top_n_playlists,
    top_n_moods,
    enable_embeddings,
    report,
):
    count_based = clustering_method in ('kmeans', 'gmm', 'spectral')
    cur_min, cur_max, cur_pct = cluster_range_min, cluster_range_max, percentile
    eps_cap = None
    try:
        best_rank = None
        best = (cluster_range_min, cluster_range_max, percentile)
        for attempt in range(1, CLUSTERING_CALIBRATION_MAX_TRIES + 1):
            target = _calculate_target_songs_per_genre(
                genre_map, cur_pct, min_songs_per_genre
            )
            subset = _get_stratified_song_subset(genre_map, target)
            if count_based:
                k_floor = max(2, len(subset) // CLUSTERING_MAX_PLAYLIST_SONGS)
                cap = max(k_floor, len(subset) // (2 * MIN_PLAYLIST_SIZE_FOR_TOP_N))
                if cap < cur_max:
                    cur_max = max(k_floor, min(cap, max(2, top_n_playlists)))
                cur_min = max(2, min(cur_min, cur_max))
                needed = max(2, min(top_n_playlists, cur_max))
            else:
                needed = max(2, top_n_playlists)
                if attempt == 1:
                    derived = _derive_dbscan_eps(
                        [t['item_id'] for t in subset],
                        max(2, (dbscan_min_samples_min + dbscan_min_samples_max) // 2),
                        MOOD_LABELS[:top_n_moods] if top_n_moods > 0 else MOOD_LABELS,
                        enable_embeddings,
                    )
                    if derived:
                        cur_min, cur_max = derived
                        report(
                            f"DBSCAN eps derived from data: {cur_min:.2f}-{cur_max:.2f} "
                            f"(configured {cluster_range_min}-{cluster_range_max})",
                            2,
                        )
                    eps_cap = cur_max * 1.5
            report(
                f"Calibration {attempt}/{CLUSTERING_CALIBRATION_MAX_TRIES}: "
                + (f"clusters {cur_min}-{cur_max}, " if count_based
                   else f"eps {cur_min:.2f}-{cur_max:.2f}, ")
                + f"subset {len(subset)}, need {needed} playlists of "
                f"{MIN_PLAYLIST_SIZE_FOR_TOP_N}+ songs, percentile {cur_pct}",
                3,
            )
            result = _perform_single_clustering_iteration(
                run_idx=attempt,
                item_ids_for_subset=[t['item_id'] for t in subset],
                clustering_method=clustering_method,
                num_clusters_min_max=(cur_max, cur_max)
                if clustering_method == 'kmeans' else (2, 2),
                dbscan_params_ranges={
                    'eps_min': cur_min if clustering_method == 'dbscan' else dbscan_eps_min,
                    'eps_max': cur_max if clustering_method == 'dbscan' else dbscan_eps_max,
                    'samples_min': dbscan_min_samples_min,
                    'samples_max': dbscan_min_samples_max,
                },
                gmm_params_ranges={'n_components_min': cur_max, 'n_components_max': cur_max}
                if clustering_method == 'gmm'
                else {'n_components_min': 2, 'n_components_max': 2},
                spectral_params_ranges={'n_clusters_min': cur_max, 'n_clusters_max': cur_max}
                if clustering_method == 'spectral'
                else {'n_clusters_min': 2, 'n_clusters_max': 2},
                pca_params_ranges={
                    'components_min': pca_components_min,
                    'components_max': pca_components_max,
                },
                active_mood_labels=MOOD_LABELS[:top_n_moods] if top_n_moods > 0 else MOOD_LABELS,
                max_songs_per_cluster=max_songs_per_cluster_val,
                log_prefix='[Calibration]',
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
                enable_clustering_embeddings=enable_embeddings,
            )
            sizes = [len(songs) for songs in (result or {}).get('named_playlists', {}).values()]
            keepers = sum(1 for s in sizes if s >= MIN_PLAYLIST_SIZE_FOR_TOP_N)
            oversized = sum(1 for s in sizes if s > CLUSTERING_MAX_PLAYLIST_SONGS)
            rank = (
                1 if keepers else 0,
                1 if keepers >= needed else 0,
                -oversized,
                keepers,
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best = (cur_min, cur_max, cur_pct)
            if keepers >= needed and not oversized:
                break
            if count_based:
                if keepers < needed:
                    cur_max = max(needed, cur_max // 2)
                    cur_min = max(2, min(cur_min, cur_max))
                    cur_pct = max(cur_pct, min(90, cur_pct + 15))
                else:
                    cur_min = cur_min + max(1, cur_min // 2)
                    cur_max = cur_max + max(1, cur_max // 2)
                    cur_pct = min(cur_pct, max(20, cur_pct - 15))
            elif oversized:
                cur_min = max(0.05, cur_min * 0.7)
                cur_max = max(cur_min * 1.2, cur_max * 0.7)
                cur_pct = min(cur_pct, max(20, cur_pct - 15))
            else:
                cur_min, cur_max = cur_min * 1.5, cur_max * 1.5
                if eps_cap:
                    cur_max = min(cur_max, eps_cap)
                    cur_min = min(cur_min, cur_max)
                cur_pct = max(cur_pct, min(90, cur_pct + 15))
        report(
            "Calibration chose "
            + (f"clusters {best[0]}-{best[1]}, " if count_based
               else f"eps {best[0]:.2f}-{best[1]:.2f}, ")
            + f"percentile {best[2]}",
            4,
        )
        return best
    except Exception:
        logger.exception("Cluster calibration failed; falling back to library-size caps")
        if not count_based:
            return cur_min, cur_max, cur_pct
        total_tracks = sum(len(tracks) for tracks in genre_map.values())
        cap = max(2, total_tracks // (2 * MIN_PLAYLIST_SIZE_FOR_TOP_N))
        safe_max = min(cluster_range_max, cap)
        safe_min = max(2, min(cluster_range_min, safe_max))
        return safe_min, safe_max, percentile


def _cluster_one_server(
    target_server,
    state,
    report,
    current_job,
    current_task_id,
    clustering_method,
    num_clusters_min,
    num_clusters_max,
    dbscan_eps_min,
    dbscan_eps_max,
    dbscan_min_samples_min,
    dbscan_min_samples_max,
    pca_components_min,
    pca_components_max,
    num_clustering_runs,
    max_songs_per_cluster_val,
    gmm_n_components_min,
    gmm_n_components_max,
    spectral_n_clusters_min,
    spectral_n_clusters_max,
    min_songs_per_genre_for_stratification_param,
    stratified_sampling_target_percentile_param,
    score_weight_diversity_param,
    score_weight_silhouette_param,
    score_weight_davies_bouldin_param,
    score_weight_calinski_harabasz_param,
    score_weight_purity_param,
    score_weight_other_feature_diversity_param,
    score_weight_other_feature_purity_param,
    ai_model_provider_param,
    ollama_server_url_param,
    ollama_model_name_param,
    openai_server_url_param,
    openai_model_name_param,
    openai_api_key_param,
    gemini_api_key_param,
    gemini_model_name_param,
    mistral_api_key_param,
    mistral_model_name_param,
    top_n_moods_for_clustering_param,
    top_n_playlists_param,
    enable_clustering_embeddings_param,
    auto_calibration_param,
):
    server_name = target_server['name'] if target_server else 'default server'
    report("Fetching lightweight track data for stratification...", 1)
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    if target_server is not None:
        cur.execute(
            "SELECT s.item_id, s.author, s.mood_vector FROM score s "
            "WHERE s.mood_vector IS NOT NULL AND s.mood_vector != '' AND "
            + registry.availability_sql('s'),
            (target_server['server_id'], bool(target_server.get('is_default'))),
        )
    else:
        cur.execute(
            "SELECT item_id, author, mood_vector FROM score "
            "WHERE mood_vector IS NOT NULL AND mood_vector != ''"
        )
    lightweight_rows = cur.fetchall()
    cur.close()

    if len(lightweight_rows) < MIN_PLAYLIST_SIZE_FOR_TOP_N:
        reason = f"only {len(lightweight_rows)} clusterable tracks available"
        report(f"Skipping this server: {reason}.", 100)
        return 'skipped', reason

    genre_map = _prepare_genre_map(lightweight_rows)
    genre_map_json = json.dumps(genre_map)

    job_prefix = state.get("job_prefix") or current_task_id
    child_tasks_from_db = [
        t for t in get_child_tasks_from_db(current_task_id)
        if str(t.get('task_id', '')).startswith(job_prefix + "_batch_")
    ]

    state["top_n_playlists"] = top_n_playlists_param
    calibrated_summary = None
    if not auto_calibration_param:
        report("Automatic parameter discovery disabled; using configured defaults.", 2)
    elif child_tasks_from_db and clustering_method == 'dbscan':
        try:
            resume_target = _calculate_target_songs_per_genre(
                genre_map,
                stratified_sampling_target_percentile_param,
                min_songs_per_genre_for_stratification_param,
            )
            resume_subset = _get_stratified_song_subset(genre_map, resume_target)
            derived = _derive_dbscan_eps(
                [t['item_id'] for t in resume_subset],
                max(2, (dbscan_min_samples_min + dbscan_min_samples_max) // 2),
                MOOD_LABELS[:top_n_moods_for_clustering_param]
                if top_n_moods_for_clustering_param > 0 else MOOD_LABELS,
                enable_clustering_embeddings_param,
            )
            if derived:
                dbscan_eps_min, dbscan_eps_max = derived
                report(
                    f"Resume: DBSCAN eps derived from data: "
                    f"{dbscan_eps_min:.2f}-{dbscan_eps_max:.2f}",
                    2,
                )
        except Exception:
            logger.exception("Resume eps derivation failed; keeping configured eps")
    if auto_calibration_param and not child_tasks_from_db:
        if clustering_method == 'gmm':
            range_min, range_max = gmm_n_components_min, gmm_n_components_max
        elif clustering_method == 'spectral':
            range_min, range_max = spectral_n_clusters_min, spectral_n_clusters_max
        elif clustering_method == 'dbscan':
            range_min, range_max = dbscan_eps_min, dbscan_eps_max
        else:
            range_min, range_max = num_clusters_min, num_clusters_max
        range_min, range_max, stratified_sampling_target_percentile_param = (
            _calibrate_cluster_params(
                clustering_method,
                genre_map,
                range_min,
                range_max,
                stratified_sampling_target_percentile_param,
                min_songs_per_genre_for_stratification_param,
                dbscan_eps_min,
                dbscan_eps_max,
                dbscan_min_samples_min,
                dbscan_min_samples_max,
                pca_components_min,
                pca_components_max,
                max_songs_per_cluster_val,
                top_n_playlists_param,
                top_n_moods_for_clustering_param,
                enable_clustering_embeddings_param,
                report,
            )
        )
        if clustering_method == 'gmm':
            gmm_n_components_min, gmm_n_components_max = range_min, range_max
        elif clustering_method == 'spectral':
            spectral_n_clusters_min, spectral_n_clusters_max = range_min, range_max
        elif clustering_method == 'kmeans':
            num_clusters_min, num_clusters_max = range_min, range_max
        elif clustering_method == 'dbscan':
            dbscan_eps_min, dbscan_eps_max = range_min, range_max
        calibrated_summary = {
            'stratification_percentile': stratified_sampling_target_percentile_param,
        }
        if clustering_method == 'dbscan':
            calibrated_summary['dbscan_eps_min'] = range_min
            calibrated_summary['dbscan_eps_max'] = range_max
        else:
            calibrated_summary['num_clusters_min'] = range_min
            calibrated_summary['num_clusters_max'] = range_max
    target_songs_per_genre = _calculate_target_songs_per_genre(
        genre_map,
        stratified_sampling_target_percentile_param,
        min_songs_per_genre_for_stratification_param,
    )
    report(f"Target songs per genre for stratification: {target_songs_per_genre}", 3)

    num_total_batches = (
        (num_clustering_runs + ITERATIONS_PER_BATCH_JOB - 1) // ITERATIONS_PER_BATCH_JOB
        if ITERATIONS_PER_BATCH_JOB > 0
        else 0
    )
    next_batch_to_launch = 0

    if child_tasks_from_db:
        logger.info(
            f"Found {len(child_tasks_from_db)} existing child tasks for '{server_name}'. Attempting state recovery."
        )
        _monitor_and_process_batches(state, current_task_id, initial_check=True)

        runs_accounted_for = state["runs_completed"]
        next_batch_to_launch = runs_accounted_for // ITERATIONS_PER_BATCH_JOB

        logger.info(
            f"Recovery complete. Resuming. Runs accounted for: {runs_accounted_for}/{num_clustering_runs}. Next batch index to launch: {next_batch_to_launch}"
        )

    if not state["last_subset_ids"]:
        initial_subset_data = _get_stratified_song_subset(genre_map, target_songs_per_genre)
        state["last_subset_ids"] = [t['item_id'] for t in initial_subset_data]

    last_progress_time = time.time()
    last_known_runs = state["runs_completed"]
    local_pct = 5

    while state["runs_completed"] < num_clustering_runs:
        task_info = get_task_info_from_db(current_task_id)
        if current_job and (
            current_job.is_stopped
            or task_info is None
            or task_info.get('status') == TASK_STATUS_REVOKED
        ):
            report("Task revoked, stopping.", local_pct, task_state=TASK_STATUS_REVOKED)
            return 'revoked', None

        _monitor_and_process_batches(state, current_task_id)

        if state["runs_completed"] > last_known_runs:
            last_known_runs = state["runs_completed"]
            last_progress_time = time.time()
        elif time.time() - last_progress_time > CLUSTERING_BATCH_TIMEOUT_MINUTES * 60:
            stale_minutes = (time.time() - last_progress_time) / 60
            report(
                f"STALENESS WATCHDOG: No progress for {stale_minutes:.1f} min (limit: {CLUSTERING_BATCH_TIMEOUT_MINUTES} min). "
                f"Forcing completion at {last_known_runs}/{num_clustering_runs} runs.",
                local_pct,
            )
            logger.warning(
                f"STALENESS WATCHDOG triggered. runs_completed stuck at {last_known_runs}/{num_clustering_runs} for {stale_minutes:.1f} min."
            )
            state["runs_completed"] = num_clustering_runs

        failed_batch_count = len(state.get("failed_batches", set()))
        if failed_batch_count >= CLUSTERING_MAX_FAILED_BATCHES:
            logger.warning(
                f"Stopping new batch launches: {failed_batch_count} batches have failed (max: {CLUSTERING_MAX_FAILED_BATCHES})"
            )
            remaining_runs = num_clustering_runs - state["runs_completed"]
            if remaining_runs > 0:
                state["runs_completed"] = num_clustering_runs
                logger.warning(
                    f"Forced completion of {remaining_runs} remaining runs due to batch failures"
                )

        stale_batches = state.get("stale_batches", 0)
        if (
            stale_batches >= CLUSTERING_EARLY_STOP_BATCHES
            and not state["active_jobs"]
            and state["runs_completed"] < num_clustering_runs
        ):
            report(
                f"Early stop: {stale_batches} consecutive batches without a better "
                f"result. Finishing at {state['runs_completed']}/{num_clustering_runs} runs.",
                local_pct,
            )
            state["runs_completed"] = num_clustering_runs

        while (
            len(state["active_jobs"]) < MAX_CONCURRENT_BATCH_JOBS
            and next_batch_to_launch < num_total_batches
            and failed_batch_count < CLUSTERING_MAX_FAILED_BATCHES
            and stale_batches < CLUSTERING_EARLY_STOP_BATCHES
        ):
            _launch_batch_job(
                state,
                current_task_id,
                next_batch_to_launch,
                num_clustering_runs,
                genre_map_json,
                target_songs_per_genre,
                clustering_method,
                num_clusters_min,
                num_clusters_max,
                dbscan_eps_min,
                dbscan_eps_max,
                dbscan_min_samples_min,
                dbscan_min_samples_max,
                gmm_n_components_min,
                gmm_n_components_max,
                spectral_n_clusters_min,
                spectral_n_clusters_max,
                pca_components_min,
                pca_components_max,
                max_songs_per_cluster_val,
                score_weight_diversity_param,
                score_weight_silhouette_param,
                score_weight_davies_bouldin_param,
                score_weight_calinski_harabasz_param,
                score_weight_purity_param,
                score_weight_other_feature_diversity_param,
                score_weight_other_feature_purity_param,
                top_n_moods_for_clustering_param,
                enable_clustering_embeddings_param,
            )
            next_batch_to_launch += 1

        local_pct = (
            5 + int(75 * state["runs_completed"] / num_clustering_runs)
            if num_clustering_runs > 0
            else 5
        )
        report(
            f"Progress: {state['runs_completed']}/{num_clustering_runs} runs. Active batches: {len(state['active_jobs'])}. Best score: {state['best_score']:.2f}",
            local_pct,
        )

        if state["runs_completed"] >= num_clustering_runs and len(state["active_jobs"]) == 0:
            report(
                f"All runs ({state['runs_completed']}) are processed or accounted for. Forcing loop exit to prevent starvation.",
                local_pct,
            )
            break

        time.sleep(3)

    _monitor_and_process_batches(state, current_task_id)

    report("All batches completed. Finalizing...", 82)

    if not state["best_result"]:
        report("No valid clustering solution found after all runs.", 100)
        return 'failed', 'no valid clustering solution found after all runs'

    best_result = state["best_result"]

    initial_playlist_count = len(best_result.get("named_playlists", {}))
    report(f"Starting post-processing with {initial_playlist_count} playlists", 83)

    report("Applying duplicate filtering to remove similar songs...", 84)
    best_result = apply_duplicate_filtering_to_clustering_result(
        best_result, log_prefix="[DuplicateFilter] "
    )
    report(
        f"After duplicate filtering: {len(best_result.get('named_playlists', {}))} playlists",
        85,
    )

    min_size_threshold = MIN_PLAYLIST_SIZE_FOR_TOP_N
    report(f"Applying minimum size filter (>= {min_size_threshold} songs)...", 86)
    best_result = apply_minimum_size_filter_to_clustering_result(
        best_result, min_size_threshold, log_prefix="[MinSizeFilter] "
    )
    report(
        f"After minimum size filtering: {len(best_result.get('named_playlists', {}))} playlists",
        87,
    )

    if (
        top_n_playlists_param > 0
        and len(best_result.get("named_playlists", {})) > top_n_playlists_param
    ):
        report(f"Filtering for Top {top_n_playlists_param} most diverse playlists...", 88)
        best_result = select_top_n_diverse_playlists(best_result, top_n_playlists_param)
        state["best_result"] = best_result

    final_playlist_count = len(best_result.get("named_playlists", {}))
    report(
        f"Post-processing complete: {initial_playlist_count} -> {final_playlist_count} playlists",
        89,
    )

    report(
        f"Best clustering found with score: {state['best_score']:.2f}. Creating playlists...",
        90,
    )

    final_playlists_with_details = _name_and_prepare_playlists(
        best_result,
        ai_model_provider_param,
        ollama_server_url_param,
        ollama_model_name_param,
        openai_server_url_param,
        openai_model_name_param,
        openai_api_key_param,
        gemini_api_key_param,
        gemini_model_name_param,
        mistral_api_key_param,
        mistral_model_name_param,
    )

    report(f"Creating {len(final_playlists_with_details)} playlists on this server...", 96)
    with registry.bind(target_server):
        if CLUSTERING_CLEANING:
            delete_automatic_playlists()
        for name, songs_with_details in final_playlists_with_details.items():
            item_ids = [item_id for item_id, _, _ in songs_with_details]
            try:
                create_playlist(name, item_ids)
            except ValueError:
                logger.warning(
                    "Playlist '%s' skipped on server '%s': none of its "
                    "tracks are available there.",
                    name,
                    server_name,
                )

    return 'success', {
        'playlists': final_playlists_with_details,
        'best_score': state['best_score'],
        'best_params': (state['best_result'] or {}).get('parameters'),
        'calibrated_params': calibrated_summary,
    }


def _prepare_genre_map(lightweight_rows):
    genre_map = defaultdict(list)
    for row in lightweight_rows:
        if row.get('mood_vector'):
            mood_scores = {
                p.split(':')[0]: float(p.split(':')[1])
                for p in row['mood_vector'].split(',')
                if ':' in p
            }
            top_genre = max(
                (g for g in STRATIFIED_GENRES if g in mood_scores),
                key=mood_scores.get,
                default='__other__',
            )
            genre_map[top_genre].append(
                {'item_id': row['item_id'], 'mood_vector': row['mood_vector']}
            )
    return genre_map


def _calculate_target_songs_per_genre(genre_map, percentile, min_songs):
    counts = [len(tracks) for g, tracks in genre_map.items() if g in STRATIFIED_GENRES]
    if not counts:
        return min_songs
    target = np.percentile(counts, np.clip(percentile, 0, 100))
    return max(min_songs, int(np.floor(target)))


def _monitor_and_process_batches(state_dict, parent_task_id, initial_check=False):
    current_time = time.time()
    timeout_seconds = CLUSTERING_BATCH_TIMEOUT_MINUTES * 60
    processed_jobs = state_dict.get("processed_job_ids", set())

    timed_out_jobs = []
    for job_id, start_time in list(state_dict.get("batch_start_times", {}).items()):
        if job_id not in processed_jobs:
            elapsed_time = current_time - start_time
            if elapsed_time > timeout_seconds:
                logger.warning(
                    f"TIMEOUT: Batch {job_id} has timed out after {elapsed_time / 60:.1f} minutes (limit: {CLUSTERING_BATCH_TIMEOUT_MINUTES} min)"
                )
                timed_out_jobs.append(job_id)
                state_dict.setdefault("timed_out_batches", set()).add(job_id)
                state_dict.setdefault("failed_batches", set()).add(job_id)
    for job_id in timed_out_jobs:
        try:
            batch_idx = None
            if "_batch_" in job_id:
                batch_idx = int(job_id.rsplit("_batch_", 1)[1])
            if batch_idx is not None:
                total_runs = state_dict.get("total_runs", 0)
                start_run = batch_idx * ITERATIONS_PER_BATCH_JOB
                num_iterations = min(ITERATIONS_PER_BATCH_JOB, total_runs - start_run)
                if num_iterations > 0 and state_dict["runs_completed"] < total_runs:
                    runs_to_add = min(num_iterations, total_runs - state_dict["runs_completed"])
                    state_dict["runs_completed"] += runs_to_add
                    logger.warning(
                        f"Job {job_id} timed out. Forced runs_completed count to increase by {runs_to_add} to prevent starvation."
                    )
        except Exception:
            logger.exception(f"Could not compute runs for timed out job {job_id}.")
        state_dict.setdefault("processed_job_ids", set()).add(job_id)
        if job_id in state_dict.get("active_jobs", {}):
            del state_dict["active_jobs"][job_id]

    all_child_tasks = get_child_tasks_from_db(parent_task_id)
    job_prefix = state_dict.get("job_prefix")
    if job_prefix:
        all_child_tasks = [
            t for t in all_child_tasks
            if str(t.get('task_id', '')).startswith(job_prefix + "_batch_")
        ]

    jobs_for_status_check = []
    for task_info in all_child_tasks:
        if task_info['task_id'] not in processed_jobs:
            jobs_for_status_check.append(task_info)

    for job_id in state_dict["active_jobs"].keys():
        if job_id not in processed_jobs and not any(
            t['task_id'] == job_id for t in jobs_for_status_check
        ):
            jobs_for_status_check.append(
                {
                    'task_id': job_id,
                    'status': TASK_STATUS_STARTED,
                    'sub_type_identifier': None,
                    'details': None,
                }
            )

    jobs_ready_for_result_extraction = []

    for task_info in jobs_for_status_check:
        job_id = task_info['task_id']
        db_status = task_info['status']

        is_terminal_in_db = db_status in [
            TASK_STATUS_SUCCESS,
            TASK_STATUS_FAILURE,
            TASK_STATUS_REVOKED,
        ]

        if is_terminal_in_db:
            jobs_ready_for_result_extraction.append(job_id)
            continue

        try:
            job = Job.fetch(job_id, connection=redis_conn)
            if job.is_finished or job.is_failed or job.get_status() == 'canceled':
                jobs_ready_for_result_extraction.append(job_id)
            elif job_id not in state_dict["active_jobs"]:
                state_dict["active_jobs"][job_id] = job
        except NoSuchJobError:
            logger.warning(
                f"Job {job_id} (status: {db_status}) not found in RQ (likely cleared). Treating as finished to prevent main task starvation."
            )
            jobs_ready_for_result_extraction.append(job_id)
        except Exception:
            logger.exception(
                f"Error checking RQ status for job {job_id}. Assuming terminal state to prevent starvation."
            )
            jobs_ready_for_result_extraction.append(job_id)

    for job_id in jobs_ready_for_result_extraction:
        if job_id in processed_jobs:
            continue

        result = get_job_result_safely(job_id, parent_task_id, "clustering_batch")

        if result and result.get("status") == TASK_STATUS_SUCCESS:
            state_dict["runs_completed"] += result.get("iterations_completed_in_batch", 0)
            state_dict["last_subset_ids"] = result.get(
                "final_subset_track_ids", state_dict["last_subset_ids"]
            )
            best_from_batch = result.get("best_result_from_batch")
            improved = False
            if best_from_batch and best_from_batch.get("parameters"):
                current_best_score = best_from_batch.get("fitness_score", -1.0)
                state_dict["elite_solutions"].append(
                    {"score": current_best_score, "params": best_from_batch.get("parameters")}
                )
                top_n = state_dict.get("top_n_playlists") or TOP_N_PLAYLISTS
                current_rank = (
                    _viable_playlists(best_from_batch, top_n),
                    current_best_score,
                )
                best_rank = (
                    _viable_playlists(state_dict["best_result"], top_n),
                    state_dict["best_score"],
                )
                if current_rank > best_rank:
                    state_dict["best_score"] = current_best_score
                    state_dict["best_result"] = best_from_batch
                    improved = True
            if not initial_check:
                state_dict["stale_batches"] = (
                    0 if improved else state_dict.get("stale_batches", 0) + 1
                )
        else:
            state_dict.setdefault("failed_batches", set()).add(job_id)
            if not initial_check:
                state_dict["stale_batches"] = state_dict.get("stale_batches", 0) + 1

            task_info_for_runs = next((t for t in all_child_tasks if t['task_id'] == job_id), None)

            if task_info_for_runs and task_info_for_runs.get('sub_type_identifier'):
                if task_info_for_runs['sub_type_identifier'].startswith('Batch_'):
                    try:
                        batch_idx = int(task_info_for_runs['sub_type_identifier'].split('_')[-1])
                        total_runs = state_dict['total_runs']

                        start_run = batch_idx * ITERATIONS_PER_BATCH_JOB
                        num_iterations = min(ITERATIONS_PER_BATCH_JOB, total_runs - start_run)

                        if num_iterations > 0 and state_dict["runs_completed"] < total_runs:
                            runs_to_add = min(
                                num_iterations, total_runs - state_dict["runs_completed"]
                            )
                            state_dict["runs_completed"] += runs_to_add
                            logger.warning(
                                f"Job {job_id} failed/missing result. Forced runs_completed count to increase by {runs_to_add} to prevent main task starvation."
                            )

                    except Exception:
                        logger.exception(
                            f"Could not calculate runs for failed/missing job {job_id} using sub_type_identifier."
                        )
            else:
                try:
                    if "_batch_" in job_id:
                        batch_idx = int(job_id.rsplit("_batch_", 1)[1])
                        total_runs = state_dict.get('total_runs', 0)
                        start_run = batch_idx * ITERATIONS_PER_BATCH_JOB
                        num_iterations = min(ITERATIONS_PER_BATCH_JOB, total_runs - start_run)
                        if num_iterations > 0 and state_dict["runs_completed"] < total_runs:
                            runs_to_add = min(
                                num_iterations, total_runs - state_dict["runs_completed"]
                            )
                            state_dict["runs_completed"] += runs_to_add
                            logger.warning(
                                f"Job {job_id} failed/missing result (no DB info). Inferred batch index and adjusted runs_completed by {runs_to_add}."
                            )
                except Exception:
                    logger.exception(
                        f"Could not infer runs for failed/missing job {job_id} from job_id."
                    )

        state_dict.setdefault("processed_job_ids", set()).add(job_id)
        if job_id in state_dict["active_jobs"]:
            del state_dict["active_jobs"][job_id]

    failed_batch_count = len(state_dict.get("failed_batches", set()))
    if failed_batch_count >= CLUSTERING_MAX_FAILED_BATCHES:
        logger.warning(
            f"Reached maximum failed batches ({failed_batch_count}/{CLUSTERING_MAX_FAILED_BATCHES}). Some jobs may be unstable."
        )

    state_dict["elite_solutions"].sort(key=lambda x: x["score"], reverse=True)
    state_dict["elite_solutions"] = state_dict["elite_solutions"][:TOP_N_ELITES]


def _launch_batch_job(
    state_dict, parent_task_id, batch_idx, total_runs, genre_map_json, target_per_genre, *args
):
    (
        clustering_method,
        num_clusters_min,
        num_clusters_max,
        dbscan_eps_min,
        dbscan_eps_max,
        dbscan_min_samples_min,
        dbscan_min_samples_max,
        gmm_n_components_min,
        gmm_n_components_max,
        spectral_n_clusters_min,
        spectral_n_clusters_max,
        pca_components_min,
        pca_components_max,
        max_songs_per_cluster,
        score_weight_diversity,
        score_weight_silhouette,
        score_weight_davies_bouldin,
        score_weight_calinski_harabasz,
        score_weight_purity,
        score_weight_other_feature_diversity,
        score_weight_other_feature_purity,
        top_n_moods,
        enable_embeddings,
    ) = args

    batch_job_id = f"{state_dict.get('job_prefix') or parent_task_id}_batch_{batch_idx}"
    start_run = batch_idx * ITERATIONS_PER_BATCH_JOB
    num_iterations = min(ITERATIONS_PER_BATCH_JOB, total_runs - start_run)

    exploitation_prob = (
        EXPLOITATION_PROBABILITY_CONFIG
        if start_run >= (total_runs * EXPLOITATION_START_FRACTION)
        else 0.0
    )

    job_args = {
        "batch_id_str": f"Batch_{batch_idx}",
        "start_run_idx": start_run,
        "num_iterations_in_batch": num_iterations,
        "genre_to_lightweight_track_data_map_json": genre_map_json,
        "target_songs_per_genre": target_per_genre,
        "sampling_percentage_change_per_run": SAMPLING_PERCENTAGE_CHANGE_PER_RUN,
        "clustering_method": clustering_method,
        "active_mood_labels_for_batch": MOOD_LABELS[:top_n_moods]
        if top_n_moods > 0
        else MOOD_LABELS,
        "num_clusters_min_max_tuple": (num_clusters_min, num_clusters_max),
        "dbscan_params_ranges_dict": {
            "eps_min": dbscan_eps_min,
            "eps_max": dbscan_eps_max,
            "samples_min": dbscan_min_samples_min,
            "samples_max": dbscan_min_samples_max,
        },
        "gmm_params_ranges_dict": {
            "n_components_min": gmm_n_components_min,
            "n_components_max": gmm_n_components_max,
        },
        "spectral_params_ranges_dict": {
            "n_clusters_min": spectral_n_clusters_min,
            "n_clusters_max": spectral_n_clusters_max,
        },
        "pca_params_ranges_dict": {
            "components_min": pca_components_min,
            "components_max": pca_components_max,
        },
        "max_songs_per_cluster": max_songs_per_cluster,
        "parent_task_id": parent_task_id,
        "score_weights_dict": {
            "mood_diversity": score_weight_diversity,
            "silhouette": score_weight_silhouette,
            "davies_bouldin": score_weight_davies_bouldin,
            "calinski_harabasz": score_weight_calinski_harabasz,
            "mood_purity": score_weight_purity,
            "other_feature_diversity": score_weight_other_feature_diversity,
            "other_feature_purity": score_weight_other_feature_purity,
        },
        "elite_solutions_params_list_json": json.dumps(
            [e["params"] for e in state_dict["elite_solutions"]]
        ),
        "exploitation_probability": exploitation_prob,
        "mutation_config_json": json.dumps(
            {
                "int_abs_delta": MUTATION_INT_ABS_DELTA,
                "float_abs_delta": MUTATION_FLOAT_ABS_DELTA,
                "coord_mutation_fraction": MUTATION_KMEANS_COORD_FRACTION,
            }
        ),
        "initial_subset_track_ids_json": json.dumps(state_dict["last_subset_ids"]),
        "enable_clustering_embeddings_param": enable_embeddings,
        "top_n_playlists_param": state_dict.get("top_n_playlists") or TOP_N_PLAYLISTS,
    }

    new_job = rq_queue_default.enqueue(
        'tasks.clustering.run_clustering_batch_task',
        kwargs=job_args,
        job_id=batch_job_id,
        job_timeout=CLUSTERING_BATCH_TIMEOUT_MINUTES * 60,
        retry=Retry(max=3),
        on_failure=batch_task_failure_handler,
    )
    state_dict["active_jobs"][new_job.id] = new_job

    state_dict.setdefault("batch_start_times", {})[new_job.id] = time.time()

    logger.info(
        f"Enqueued batch job {new_job.id} for runs {start_run}-{start_run + num_iterations - 1}."
    )


def _name_and_prepare_playlists(
    best_result,
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
):
    final_playlists = {}
    named_playlists = best_result.get("named_playlists", {})
    max_songs = best_result.get("parameters", {}).get(
        "max_songs_per_cluster", MAX_SONGS_PER_CLUSTER
    )

    for original_name, songs in named_playlists.items():
        if not songs:
            continue

        try:
            final_name = _try_ai_name_playlist(
                original_name,
                songs,
                best_result.get("playlist_centroids", {}),
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
                list(final_playlists),
            )
        except Exception as e:
            logger.warning(f"AI naming failed for '{original_name}': {e}. Using original name.")
            final_name = original_name

        temp_name = final_name
        suffix = 1
        while temp_name in final_playlists:
            suffix += 1
            temp_name = f"{final_name} ({suffix})"
        final_name = temp_name

        base_name = f"{final_name}_automatic"
        shuffled = _shuffle_playlist_songs(songs, base_name)
        _assign_playlist_chunks(shuffled, max_songs, base_name, final_playlists)

    return final_playlists
