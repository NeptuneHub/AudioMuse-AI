# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Clustering orchestrator: evolutionary search that turns embeddings into playlists.

run_clustering_task runs the whole pipeline once per target server,
sequentially: each server clusters only its own availability-scoped catalogue,
runs its own evolutionary/elitist search via run_clustering_batch_task child
jobs, and gets its own playlists. Delegates per-iteration work to
clustering_helper, models to clustering_gpu, and
dedup/size/diversity filtering to clustering_postprocessing.

Main Features:
* Per-server persistence: playlists replace ITS OWN rows, so the table is always
  the last run per server, never a growing history.
* Fan-out of parameter sets into batch jobs with elite tracking and adaptive
  sampling; early-stop after CLUSTERING_EARLY_STOP_BATCHES that brought back
  nothing better. A CRASHED batch counts as one of those, deliberately: after
  that many failures the run ends with the best result it holds rather than
  feeding more batches to workers that keep dying.
* The drain loop REAPS finished children (row deleted as the result is read) so
  a batch is never counted twice; no per-batch timeout, and
  CLUSTERING_STALL_TIMEOUT_MINUTES bounds the one wedge case (native code
  that never returns) by revoking and finishing with the best result held.
  That window SLIDES on any sign of life - a batch finishing, failing, launching,
  or a live batch merely advancing its own iteration counter - so a slow batch and
  a batch a fresh worker picked up after the old one died both hold it open.
  The window, the victim rule and the give-up bound are ChildDrainSupervisor in
  tasks.recovery, shared with the analysis twin so the two cannot drift again.
* ONE iteration is a single opaque fit (spectral/GMM over CLUSTERING_SUBSET_SONGS
  songs) and the batch writes its row only AFTER it returns, so with one worker
  container - the default - an iteration slower than the stall window looked
  exactly like a wedge and a healthy batch was revoked. run_clustering_batch_task
  now holds a row_heartbeat across each iteration and the parent reads the child's
  beat_at, so being alive is visible without waiting for the iteration to end. The
  heartbeat is bounded, so a fit that never returns is still caught.
* The parent persists its own progress on its row (_resumable_progress), so a
  crashed main task resumes with the winning result instead of redoing the search.
* Reap and launch ride the SAME status write (never a separate commit), so a
  parent dying mid-pass cannot lose a finished batch or double-count a launch.
* Genre-stratified sampling and per-server calibration
  (_calibrate_cluster_params) auto-tune every algorithm via quick probes.
"""

from collections import defaultdict
import numpy as np
import json
import time
import logging
import uuid

import taskqueue


from psycopg2 import OperationalError
from psycopg2.extras import DictCursor

from config import (
    MAX_SONGS_PER_CLUSTER,
    TOP_N_CLUSTERING_PLAYLIST,
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
    CLUSTERING_MAX_FAILED_BATCHES,
    CLUSTERING_CLEANING,
    CLUSTER_NAMING_AI_HISTORY,
    CLUSTERING_MAX_PLAYLIST_SONGS,
    CLUSTERING_CALIBRATION_MAX_TRIES,
    CLUSTERING_EARLY_STOP_BATCHES,
    CLUSTERING_STALL_TIMEOUT_MINUTES,
    TASK_STATUS_STARTED,
    TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
    TASK_STATUS_REVOKED,
)

from error import error_manager
from error.error_dictionary import ERR_CLUSTERING_FAILED, ERR_DB_CONNECTION

from .recovery import ChildDrainSupervisor, row_heartbeat, slow_step_budget_minutes

from database import (
    save_task_status,
    get_task_info_from_db,
    get_db,
    coerce_db_details,
    update_playlist_table,
    prune_playlist_rows_for_missing_servers,
    get_child_tasks_from_db,
    get_recent_playlist_names,
    main_task_start_lock,
    MAX_LOG_ENTRIES_STORED,
)

from sanitization import sanitize_for_json

from .mediaserver import (
    PlaylistIdTranslationError,
    create_playlist,
    delete_automatic_playlists,
)
from .mediaserver import registry
from sklearn.neighbors import NearestNeighbors

from .clustering_helper import (
    _get_stratified_song_subset,
    _get_track_primary_genre,
    _perform_single_clustering_iteration,
    _prepare_iteration_data,
    _prepare_and_normalize_data,
    _shuffle_playlist_songs,
    _assign_playlist_chunks,
    _try_ai_name_playlist,
)
from .clustering_postprocessing import (
    apply_duplicate_filtering_to_clustering_result,
    apply_minimum_size_filter_to_clustering_result,
    select_diverse_playlists_with_genre_coverage,
)

logger = logging.getLogger(__name__)

_PARENT_CANCELLED_MESSAGE = "Parent task was cancelled."


def _derive_dbscan_eps(item_ids, min_samples, active_moods, enable_embeddings):
    valid_tracks, x_feat, x_embed = _prepare_iteration_data(
        item_ids, active_moods, enable_embeddings, '[Calibration]', 0
    )
    if valid_tracks is None:
        return None
    data = _prepare_and_normalize_data(x_feat, x_embed, enable_embeddings)
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


def _viable_playlists(result, target=TOP_N_CLUSTERING_PLAYLIST):
    playlists = (result or {}).get('named_playlists') or {}
    keepers = sum(
        1 for songs in playlists.values() if len(songs) >= MIN_PLAYLIST_SIZE_FOR_TOP_N
    )
    return min(keepers, max(1, target))


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
    top_n_playlists_param=None,
):
    from flask_app import app

    claimed_task_id = taskqueue.current_task_id()
    current_task_id = claimed_task_id or str(uuid.uuid4())
    top_n_clustering_playlist_param = (
        top_n_playlists_param
        if top_n_playlists_param is not None
        else TOP_N_CLUSTERING_PLAYLIST
    )
    logger.info(f"Starting clustering batch task {current_task_id} (Batch: {batch_id_str})")

    with app.app_context():
        def _log_and_update(message, progress, details=None, state=TASK_STATUS_PROGRESS):
            logger.info(f"[ClusteringBatchTask-{current_task_id}] {message}")
            db_details = {
                "batch_id": batch_id_str,
                "start_run_idx": start_run_idx,
                "num_iterations_in_batch": num_iterations_in_batch,
                "message": message,
                "status_message": message,
                **(details or {}),
            }

            def write_if_parent_live():
                if claimed_task_id:
                    parent = get_task_info_from_db(parent_task_id)
                    if parent is None or parent.get('status') in (
                        TASK_STATUS_SUCCESS,
                        TASK_STATUS_FAILURE,
                        TASK_STATUS_REVOKED,
                    ):
                        logger.info(
                            "Suppressing batch status write for %s because parent %s "
                            "is missing or terminal.",
                            current_task_id, parent_task_id,
                        )
                        return False
                return save_task_status(
                    current_task_id,
                    "clustering_batch",
                    state,
                    parent_task_id=parent_task_id,
                    sub_type_identifier=batch_id_str,
                    progress=progress,
                    details=db_details,
                )

            return write_if_parent_live()

        try:
            parent_task_info = get_task_info_from_db(parent_task_id)
            if claimed_task_id and (
                parent_task_info is None
                or parent_task_info.get('status')
                in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]
            ):
                logger.info(
                    "Clustering batch %s will not start because parent %s is "
                    "missing or terminal.",
                    current_task_id, parent_task_id,
                )
                return {"status": "REVOKED", "message": _PARENT_CANCELLED_MESSAGE}
            if not _log_and_update("Batch started.", 0):
                return {"status": "REVOKED", "message": _PARENT_CANCELLED_MESSAGE}
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
            tracks_cache = {}

            for i in range(num_iterations_in_batch):
                current_run_global_idx = start_run_idx + i

                if claimed_task_id:
                    task_info = get_task_info_from_db(current_task_id)
                    parent_task_info = get_task_info_from_db(parent_task_id)
                    if (
                        task_info is None
                        or task_info.get('status') == TASK_STATUS_REVOKED
                        or parent_task_info is None
                        or parent_task_info.get('status')
                        in [TASK_STATUS_SUCCESS, TASK_STATUS_REVOKED, TASK_STATUS_FAILURE]
                    ):
                        logger.info(
                            "Stopping batch %s due to missing/terminal cancellation "
                            "state; no task row will be recreated.",
                            current_task_id,
                        )
                        return {"status": "REVOKED", "message": "Batch task revoked."}

                previous_subset_ids = set(current_sampled_track_ids)
                percentage_change = sampling_percentage_change_per_run
                current_subset_lightweight_data = _get_stratified_song_subset(
                    genre_to_lightweight_track_data_map,
                    target_songs_per_genre,
                    prev_ids=current_sampled_track_ids,
                    percent_change=percentage_change,
                )
                item_ids_for_iteration = [t['item_id'] for t in current_subset_lightweight_data]
                current_sampled_track_ids = list(item_ids_for_iteration)
                retained_count = len(previous_subset_ids & set(current_sampled_track_ids))
                logger.info(
                    "[Batch-%s] Sampling run %d: %d/%d tracks retained; %d changed.",
                    current_task_id,
                    current_run_global_idx,
                    retained_count,
                    len(current_sampled_track_ids),
                    len(current_sampled_track_ids) - retained_count,
                )

                if not item_ids_for_iteration:
                    logger.warning(
                        f"No songs in subset for iteration {current_run_global_idx}. Skipping."
                    )
                    continue

                iteration_step = (
                    f"iteration {current_run_global_idx} "
                    f"({clustering_method} over {len(item_ids_for_iteration)} songs)"
                )
                with row_heartbeat(
                    claimed_task_id,
                    iteration_step,
                    every_minutes=CLUSTERING_STALL_TIMEOUT_MINUTES,
                    stop_after_minutes=slow_step_budget_minutes(
                        CLUSTERING_STALL_TIMEOUT_MINUTES
                    ),
                ):
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
                        tracks_cache=tracks_cache,
                    )
                iterations_completed += 1

                iteration_rank = (
                    _viable_playlists(iteration_result, top_n_clustering_playlist_param),
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
            if not _log_and_update(
                f"Batch complete. Best score: {best_score_in_batch:.2f}",
                100,
                details=final_details,
                state=TASK_STATUS_SUCCESS,
            ):
                return {"status": "REVOKED", "message": _PARENT_CANCELLED_MESSAGE}
            return {
                "status": "SUCCESS",
                "iterations_completed_in_batch": iterations_completed,
                "best_result_from_batch": best_result_in_batch,
                "final_subset_track_ids": current_sampled_track_ids,
            }

        except OperationalError as e:
            logger.exception(
                "Database connection error during clustering batch %s; leaving the "
                "row for the queue to requeue rather than failing it here.",
                batch_id_str,
            )
            error_manager.record(ERR_DB_CONNECTION, str(e))
            raise
        except Exception as e:
            logger.exception(f"Clustering batch {batch_id_str} failed")
            err = error_manager.record(
                error_manager.classify(e, ERR_CLUSTERING_FAILED), str(e)
            )
            if not _log_and_update(
                f"Batch failed: {e}", 100, details={"error": err}, state=TASK_STATUS_FAILURE
            ):
                return {"status": "REVOKED", "message": _PARENT_CANCELLED_MESSAGE}
            return {"status": TASK_STATUS_FAILURE, "message": str(e)}


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
    top_n_playlists_param=None,
    enable_clustering_embeddings_param=True,
    output_server_scope="all",
    auto_calibration_param=None,
):
    from flask_app import app

    if auto_calibration_param is None:
        auto_calibration_param = CLUSTERING_AUTO_CALIBRATION
    top_n_clustering_playlist_param = (
        top_n_playlists_param
        if top_n_playlists_param is not None
        else TOP_N_CLUSTERING_PLAYLIST
    )

    claimed_task_id = taskqueue.current_task_id()
    current_task_id = claimed_task_id or str(uuid.uuid4())
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
        "top_n_clustering_playlist": top_n_clustering_playlist_param,
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
        from .task_run import terminal_skip

        task_info = get_task_info_from_db(current_task_id)
        skip = terminal_skip(
            current_task_id, claimed_task_id, task_info,
            revoked_message="Task was cancelled before execution.",
            terminal_message="Task already in terminal state.",
            terminal_details=lambda info: json.loads(info.get('details', '{}')),
        )
        if skip is not None:
            return skip

        _main_task_accumulated_details = {
            "total_runs": num_clustering_runs,
            "runs_completed": 0,
            "best_score": -1.0,
            "best_result": None,
            "elite_solutions": [],
            "last_subset_ids": [],
            "failed_batches": 0,
            "stale_batches": 0,
            "batches_launched": 0,
            "server_idx": 0,
            "log": [],
        }
        resume_from = _resumable_progress(task_info, num_clustering_runs)
        if resume_from:
            _main_task_accumulated_details.update(resume_from)
            logger.info(
                "Main clustering task %s is resuming a previous attempt at server "
                "index %d with %d/%d runs already completed.",
                current_task_id, resume_from.get("server_idx", 0),
                resume_from.get("runs_completed", 0), num_clustering_runs,
            )

        def _log_and_update(
            message, progress, details_to_add_or_update=None, task_state=TASK_STATUS_PROGRESS
        ):
            logger.info(f"[MainClusteringTask-{current_task_id}] {message}")
            if details_to_add_or_update:
                _main_task_accumulated_details.update(details_to_add_or_update)

            _main_task_accumulated_details["status_message"] = message
            _main_task_accumulated_details["message"] = message
            run_log = _main_task_accumulated_details["log"]
            run_log.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")
            if len(run_log) > MAX_LOG_ENTRIES_STORED:
                del run_log[:-MAX_LOG_ENTRIES_STORED]

            details_for_db = _main_task_accumulated_details.copy()
            details_for_db.pop('last_subset_ids', None)
            if details_for_db.get('best_result') is not None:
                details_for_db['best_result'] = _persistable_best_result(
                    details_for_db['best_result']
                )

            def write_if_still_claimed():
                if claimed_task_id:
                    own = get_task_info_from_db(current_task_id)
                    if own is None or own.get('status') == TASK_STATUS_REVOKED:
                        logger.info(
                            "Suppressing main clustering status write for %s: its "
                            "claim is gone, so the run was cancelled.",
                            current_task_id,
                        )
                        return False
                return save_task_status(
                    current_task_id,
                    "main_clustering",
                    task_state,
                    progress=progress,
                    details=details_for_db,
                )

            return write_if_still_claimed()

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

                if server_idx < _main_task_accumulated_details.get("server_idx", 0):
                    logger.info(
                        "Skipping server index %d: a previous attempt of this task "
                        "already finished it.", server_idx,
                    )
                    continue
                if server_idx > _main_task_accumulated_details.get("server_idx", 0):
                    _main_task_accumulated_details.update({
                        "runs_completed": 0,
                        "best_score": -1.0,
                        "best_result": None,
                        "elite_solutions": [],
                        "last_subset_ids": [],
                        "failed_batches": 0,
                        "stale_batches": 0,
                        "batches_launched": 0,
                    })
                _main_task_accumulated_details.update({
                    "server_idx": server_idx,
                    "job_prefix": f"{current_task_id}_s{server_idx}",
                })

                try:
                    status, payload = _cluster_one_server(
                        target_server,
                        _main_task_accumulated_details,
                        report,
                        claimed_task_id,
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
                        top_n_clustering_playlist_param,
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

            skipped = [s for s in per_server_summary if s['status'] != 'success']
            skipped_detail = (
                ' Skipped: '
                + '; '.join(
                    f"{s['server']}: {s.get('reason') or 'no reason reported'}"
                    for s in skipped
                )
                if skipped
                else ''
            )
            final_message = (
                f"Clustering task completed successfully on {len(successes)}/"
                f"{len(target_servers)} server(s)!{skipped_detail}"
            )

            logger.info(f"[MainClusteringTask-{current_task_id}] {final_message}")

            final_db_summary = {
                "status_message": final_message,
                "running_parameters": initial_params,
                "best_score": best_score_overall,
                "best_params": best_params_overall,
                "num_playlists_created": sum(
                    s.get('playlists_created', 0) for s in successes
                ),
                "per_server": per_server_summary,
                "log": _main_task_accumulated_details.get("log", [])[-MAX_LOG_ENTRIES_STORED:],
            }


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
        return log_and_update(scoped, pct, task_state=task_state)

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
    top_n_clustering_playlist,
    top_n_moods,
    enable_embeddings,
    report,
):
    count_based = clustering_method in ('kmeans', 'gmm', 'spectral')
    cur_min, cur_max = cluster_range_min, cluster_range_max
    eps_cap = None
    try:
        best_rank = None
        best = (cluster_range_min, cluster_range_max, percentile)
        target = _calculate_target_songs_per_genre(
            genre_map, percentile, min_songs_per_genre
        )
        subset = _get_stratified_song_subset(genre_map, target)
        tracks_cache = {}
        for attempt in range(1, CLUSTERING_CALIBRATION_MAX_TRIES + 1):
            if count_based:
                k_floor = max(2, len(subset) // CLUSTERING_MAX_PLAYLIST_SONGS)
                cap = max(k_floor, len(subset) // (2 * MIN_PLAYLIST_SIZE_FOR_TOP_N))
                target_playlists = (
                    top_n_clustering_playlist if top_n_clustering_playlist > 0 else cap
                )
                if cap < cur_max:
                    cur_max = max(k_floor, min(cap, max(2, target_playlists)))
                cur_min = max(2, min(cur_min, cur_max))
                needed = max(2, min(target_playlists, cur_max))
            else:
                needed = (
                    max(2, top_n_clustering_playlist)
                    if top_n_clustering_playlist > 0
                    else 2
                )
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
                f"{MIN_PLAYLIST_SIZE_FOR_TOP_N}+ songs, fixed percentile {percentile}",
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
                tracks_cache=tracks_cache,
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
                best = (cur_min, cur_max, percentile)
            if keepers >= needed and not oversized:
                break
            if count_based:
                if keepers < needed:
                    cur_max = max(needed, cur_max // 2)
                    cur_min = max(2, min(cur_min, cur_max))
                else:
                    cur_min = cur_min + max(1, cur_min // 2)
                    cur_max = cur_max + max(1, cur_max // 2)
            elif oversized:
                cur_min = max(0.05, cur_min * 0.7)
                cur_max = max(cur_min * 1.2, cur_max * 0.7)
            else:
                cur_min, cur_max = cur_min * 1.5, cur_max * 1.5
                if eps_cap:
                    cur_max = min(cur_max, eps_cap)
                    cur_min = min(cur_min, cur_max)
        report(
            "Calibration chose "
            + (f"clusters {best[0]}-{best[1]}, " if count_based
               else f"eps {best[0]:.2f}-{best[1]:.2f}, ")
            + f"fixed percentile {percentile}",
            4,
        )
        return best
    except Exception:
        logger.exception("Cluster calibration failed; falling back to library-size caps")
        if not count_based:
            return cur_min, cur_max, percentile
        total_tracks = sum(len(tracks) for tracks in genre_map.values())
        cap = max(2, total_tracks // (2 * MIN_PLAYLIST_SIZE_FOR_TOP_N))
        safe_max = min(cluster_range_max, cap)
        safe_min = max(2, min(cluster_range_min, safe_max))
        return safe_min, safe_max, percentile


def _run_claim_is_gone(claimed_task_id, task_id):
    if not claimed_task_id:
        return False
    info = get_task_info_from_db(task_id)
    return info is None or info.get('status') == TASK_STATUS_REVOKED


def _cluster_one_server(
    target_server,
    state,
    report,
    claimed_task_id,
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
    top_n_clustering_playlist_param,
    enable_clustering_embeddings_param,
    auto_calibration_param,
):
    server_name = target_server['name'] if target_server else 'default server'
    report("Fetching lightweight track data for stratification...", 1)
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    if target_server is not None:
        cur.execute(
            "SELECT s.item_id, s.mood_vector FROM score s "
            "WHERE s.mood_vector IS NOT NULL AND s.mood_vector != '' AND "
            + registry.availability_sql('s'),
            (target_server['server_id'], bool(target_server.get('is_default'))),
        )
    else:
        cur.execute(
            "SELECT item_id, mood_vector FROM score "
            "WHERE mood_vector IS NOT NULL AND mood_vector != ''"
        )
    lightweight_rows = cur.fetchall()
    cur.close()

    if len(lightweight_rows) < MIN_PLAYLIST_SIZE_FOR_TOP_N:
        reason = f"only {len(lightweight_rows)} clusterable tracks available"
        report(f"Skipping this server: {reason}.", 100)
        return 'skipped', reason

    genre_map = _prepare_genre_map(lightweight_rows)
    del lightweight_rows
    genre_map_json = json.dumps(genre_map)
    shared_genre_map_token = taskqueue.put_shared_payload(current_task_id, genre_map_json)

    job_prefix = state.get("job_prefix") or current_task_id
    _revoke_foreign_batches(current_task_id, job_prefix)
    child_tasks_from_db = [
        t for t in get_child_tasks_from_db(current_task_id)
        if str(t.get('task_id', '')).startswith(job_prefix + "_batch_")
    ]

    state["top_n_clustering_playlist"] = top_n_clustering_playlist_param
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
                top_n_clustering_playlist_param,
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
    next_batch_to_launch = int(state.get("batches_launched") or 0)

    local_pct = 5

    def persist_progress(message):
        return report(message, local_pct)

    if child_tasks_from_db or next_batch_to_launch:
        _absorb_finished_batches(state, current_task_id, persist_progress)
        logger.info(
            "Resuming '%s' from its own persisted progress: %d/%d runs, %d batches "
            "already launched.",
            server_name, state["runs_completed"], num_clustering_runs, next_batch_to_launch,
        )

    if not state["last_subset_ids"]:
        initial_subset_data = _get_stratified_song_subset(genre_map, target_songs_per_genre)
        state["last_subset_ids"] = [t['item_id'] for t in initial_subset_data]

    stop_launching = False
    supervisor = ChildDrainSupervisor(
        current_task_id,
        lambda job_id, message: _revoke_batch(job_id, current_task_id, message),
        CLUSTERING_STALL_TIMEOUT_MINUTES,
        1,
        lambda: time.monotonic(),
        label='batch',
    )

    while True:
        task_info = get_task_info_from_db(current_task_id)
        if claimed_task_id and (
            task_info is None or task_info.get('status') == TASK_STATUS_REVOKED
        ):
            report("Task revoked, stopping.", local_pct, task_state=TASK_STATUS_REVOKED)
            return 'revoked', None

        _absorb_finished_batches(state, current_task_id, persist_progress)

        try:
            live_marks = _live_batches(state, current_task_id)
        except Exception:
            logger.exception("Could not list the live clustering batches; retrying")
            supervisor.restart()
            time.sleep(3)
            continue
        live = [job_id for job_id, _progress, _beat, _status in live_marks]

        failed_batch_count = state.get("failed_batches", 0)
        if failed_batch_count >= CLUSTERING_MAX_FAILED_BATCHES and not stop_launching:
            stop_launching = True
            logger.warning(
                f"Stopping new batch launches: {failed_batch_count} batches have failed (max: {CLUSTERING_MAX_FAILED_BATCHES})"
            )

        stale_batches = state.get("stale_batches", 0)
        if stale_batches >= CLUSTERING_EARLY_STOP_BATCHES and not stop_launching:
            stop_launching = True
            report(
                f"Early stop: {stale_batches} consecutive batches without a better "
                f"result. Finishing at {state['runs_completed']}/{num_clustering_runs} runs.",
                local_pct,
            )

        while (
            not stop_launching
            and len(live) < MAX_CONCURRENT_BATCH_JOBS
            and next_batch_to_launch < num_total_batches
        ):
            launch_conn = get_db()
            launched = _launch_batch_job(
                state,
                current_task_id,
                next_batch_to_launch,
                num_clustering_runs,
                shared_genre_map_token,
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
                conn=launch_conn,
            )
            if not launched:
                _rollback_quietly(launch_conn)
                report("Task revoked before the next batch could start.", local_pct,
                       task_state=TASK_STATUS_REVOKED)
                return 'revoked', None
            state["batches_launched"] = next_batch_to_launch + 1
            if not persist_progress(
                f"Started batch {next_batch_to_launch + 1}/{num_total_batches}."
            ):
                state["batches_launched"] = next_batch_to_launch
                _rollback_quietly(launch_conn)
                logger.warning(
                    "Could not record batch %d of %s as launched; its enqueue was "
                    "rolled back with it so the next pass launches it exactly once.",
                    next_batch_to_launch, current_task_id,
                )
                break
            live.append(f"{state.get('job_prefix') or current_task_id}_batch_{next_batch_to_launch}")
            next_batch_to_launch += 1

        local_pct = (
            5 + int(75 * min(state["runs_completed"], num_clustering_runs) / num_clustering_runs)
            if num_clustering_runs > 0
            else 5
        )
        progress_signature = (
            state["runs_completed"], state["best_score"], tuple(sorted(live_marks)),
            next_batch_to_launch, stop_launching,
        )
        if supervisor.moved(progress_signature):
            report(
                f"Progress: {state['runs_completed']}/{num_clustering_runs} runs. Active batches: {len(live)}. Best score: {state['best_score']:.2f}",
                local_pct,
            )
        elif live and supervisor.expired():
            abandoned, stalled_minutes = supervisor.give_up(
                [(job_id, status) for job_id, _progress, _beat, status in live_marks],
                [job_id for job_id, _progress, _beat, _status in live_marks],
            )
            if supervisor.exhausted():
                stop_launching = True
            report(
                f"No progress of any kind for {stalled_minutes:.0f} min (limit: "
                f"{CLUSTERING_STALL_TIMEOUT_MINUTES} min). Gave up on {abandoned} of "
                f"{len(live)} unfinished batch(es) and finishing with the best result "
                f"from {state['runs_completed']}/{num_clustering_runs} runs.",
                local_pct,
            )

        if not live and (stop_launching or next_batch_to_launch >= num_total_batches):
            report(
                f"All batches are done: {state['runs_completed']}/{num_clustering_runs} runs completed.",
                local_pct,
            )
            break

        time.sleep(3)

    _absorb_finished_batches(state, current_task_id, persist_progress)

    try:
        taskqueue.clear_shared_payload(current_task_id, shared_genre_map_token)
    except Exception:
        logger.exception(
            "Could not clear the shared genre map for %s; maintenance will sweep it",
            current_task_id,
        )

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

    if top_n_clustering_playlist_param > 0:
        report(
            "Selecting up to "
            f"{top_n_clustering_playlist_param} playlists with the 6+4 diversity strategy...",
            88,
        )
        best_result = select_diverse_playlists_with_genre_coverage(
            best_result,
            top_n_clustering_playlist_param,
            primary_genre_counts={
                genre: len(tracks)
                for genre, tracks in genre_map.items()
                if genre != '__other__'
            },
        )
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

    if _run_claim_is_gone(claimed_task_id, current_task_id):
        logger.info(
            "Clustering %s was cancelled before naming; no playlist is created.",
            current_task_id,
        )
        return 'revoked', None

    previous_playlist_names = _previous_names_for_naming(
        target_server['server_id'] if target_server else None
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
        previous_playlist_names=previous_playlist_names,
    )

    if _run_claim_is_gone(claimed_task_id, current_task_id):
        logger.info(
            "Clustering %s was cancelled before playlist creation; the server is "
            "left untouched.",
            current_task_id,
        )
        return 'revoked', None

    report(f"Creating {len(final_playlists_with_details)} playlists on this server...", 96)
    with registry.bind(target_server):
        if CLUSTERING_CLEANING:
            delete_automatic_playlists()
        for name, songs_with_details in final_playlists_with_details.items():
            item_ids = [item_id for item_id, _, _ in songs_with_details]
            try:
                create_playlist(name, item_ids)
            except PlaylistIdTranslationError:
                logger.exception(
                    "PLAYLIST CREATION ABORTED on server '%s': the id "
                    "translation infrastructure failed while creating '%s'. "
                    "This is NOT an availability skip; failing the task.",
                    server_name,
                    name,
                )
                raise
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
            genre_map[_get_track_primary_genre(row)].append(
                {'item_id': row['item_id'], 'mood_vector': row['mood_vector']}
            )
    return genre_map


def _calculate_target_songs_per_genre(genre_map, percentile, min_songs):
    counts = [len(tracks) for g, tracks in genre_map.items() if g in STRATIFIED_GENRES]
    if not counts:
        return min_songs
    target = np.percentile(counts, np.clip(percentile, 0, 100))
    return max(min_songs, int(np.floor(target)))


def _revoke_batch(job_id, parent_task_id, message):
    try:
        save_task_status(
            job_id, 'clustering_batch', TASK_STATUS_REVOKED, progress=100,
            parent_task_id=parent_task_id, details={'message': message},
        )
        taskqueue.request_cancel(job_id)
        return True
    except Exception:
        logger.exception("Could not cancel the clustering batch %s", job_id)
        return False


def _revoke_foreign_batches(parent_task_id, job_prefix):
    try:
        live = taskqueue.live_children(parent_task_id)
    except Exception:
        logger.exception(
            "Could not list the live children of %s; leaving any stale batch alone",
            parent_task_id,
        )
        return 0
    mine = job_prefix + "_batch_"
    revoked = 0
    for child in live:
        job_id = str(child.get('task_id') or '')
        if not job_id or "_batch_" not in job_id or job_id.startswith(mine):
            continue
        if _revoke_batch(
            job_id, parent_task_id,
            'This batch belonged to an earlier phase of a run that restarted, so it '
            'was cancelled rather than left to fail on a genre map that is gone.',
        ):
            revoked += 1
    if revoked:
        logger.warning(
            "Revoked %d clustering batch(es) left over from an earlier phase of this task.",
            revoked,
        )
    return revoked


_RESUMABLE_KEYS = (
    "runs_completed", "best_score", "best_result", "elite_solutions",
    "failed_batches", "stale_batches", "batches_launched", "server_idx",
)

# The PCA component matrix (n_components x EMBEDDING_DIMENSION floats) is
# produced by every iteration but read by NOBODY once the run is over. Dropping
# it is what makes the winning result cheap enough to keep on the parent row.
# The centroid maps below it ARE read, by clustering_postprocessing.
_UNUSED_BEST_RESULT_KEYS = ("pca_model_details",)


def _persistable_best_result(best_result):
    if not isinstance(best_result, dict):
        return best_result
    return {
        key: value for key, value in best_result.items()
        if key not in _UNUSED_BEST_RESULT_KEYS
    }


def _resumable_progress(task_info, num_clustering_runs):
    if not task_info:
        return None
    details = coerce_db_details(task_info.get('details'))
    if not isinstance(details, dict):
        return None
    if details.get("total_runs") != num_clustering_runs:
        return None
    resumed = {key: details[key] for key in _RESUMABLE_KEYS if key in details}
    if not resumed.get("batches_launched") and not resumed.get("runs_completed"):
        return None
    if not isinstance(resumed.get("elite_solutions"), list):
        resumed["elite_solutions"] = []
    return resumed


def _absorb_batch_result(state_dict, job_id, status, details):
    if status != TASK_STATUS_SUCCESS:
        state_dict["failed_batches"] = state_dict.get("failed_batches", 0) + 1
        state_dict["stale_batches"] = state_dict.get("stale_batches", 0) + 1
        logger.warning("Clustering batch %s ended as %s.", job_id, status)
        return

    state_dict["runs_completed"] += int(details.get("iterations_completed_in_batch") or 0)
    subset = details.get("final_subset_track_ids")
    if subset:
        state_dict["last_subset_ids"] = subset

    best_from_batch = (
        details.get("full_best_result_from_batch") or details.get("full_result")
    )
    improved = False
    if best_from_batch and best_from_batch.get("parameters"):
        score = best_from_batch.get("fitness_score", -1.0)
        state_dict["elite_solutions"].append(
            {"score": score, "params": best_from_batch.get("parameters")}
        )
        target = state_dict.get("top_n_clustering_playlist")
        if target is None:
            target = TOP_N_CLUSTERING_PLAYLIST
        current_rank = (_viable_playlists(best_from_batch, target), score)
        best_rank = (
            _viable_playlists(state_dict["best_result"], target),
            state_dict["best_score"],
        )
        if current_rank > best_rank:
            state_dict["best_score"] = score
            state_dict["best_result"] = best_from_batch
            improved = True

    state_dict["stale_batches"] = 0 if improved else state_dict.get("stale_batches", 0) + 1


_ABSORBED_KEYS = (
    "runs_completed", "best_score", "best_result", "elite_solutions",
    "failed_batches", "stale_batches", "last_subset_ids",
)


def _absorbed_snapshot(state_dict):
    snapshot = {key: state_dict.get(key) for key in _ABSORBED_KEYS}
    snapshot["elite_solutions"] = list(snapshot["elite_solutions"] or [])
    return snapshot


def _rollback_quietly(db):
    if db is None:
        return
    try:
        db.rollback()
    except Exception:
        logger.exception("Could not roll back the clustering parent transaction")


def _discard_reap(db, state_dict, snapshot):
    _rollback_quietly(db)
    state_dict.update(snapshot)


def _absorb_finished_batches(state_dict, parent_task_id, persist):
    snapshot = _absorbed_snapshot(state_dict)
    mine = (state_dict.get("job_prefix") or parent_task_id) + "_batch_"
    absorbed = 0
    db = None
    try:
        db = get_db()
        reaped = taskqueue.reap_finished_children(parent_task_id, conn=db)
        for child in reaped:
            job_id = str(child.get('task_id') or '')
            if not job_id.startswith(mine):
                continue
            details = child.get('details')
            _absorb_batch_result(
                state_dict, job_id, child.get('status'),
                details if isinstance(details, dict) else {},
            )
            absorbed += 1

        if not absorbed:
            db.commit()
            return 0

        state_dict["elite_solutions"].sort(key=lambda x: x["score"], reverse=True)
        state_dict["elite_solutions"] = state_dict["elite_solutions"][:TOP_N_ELITES]
        persisted = persist(
            f"Absorbed {absorbed} finished batch(es): "
            f"{state_dict['runs_completed']} runs completed."
        )
    except Exception:
        logger.exception(
            "Could not reap the finished clustering batches of %s; retrying next pass",
            parent_task_id,
        )
        persisted = False

    if not persisted:
        _discard_reap(db, state_dict, snapshot)
        return 0
    return absorbed


def _live_batches(state_dict, parent_task_id):
    mine = (state_dict.get("job_prefix") or parent_task_id) + "_batch_"
    return [
        (
            str(child.get('task_id')),
            child.get('progress'),
            str(child.get('beat_at') or ''),
            str(child.get('status') or ''),
        )
        for child in taskqueue.live_children(parent_task_id)
        if str(child.get('task_id') or '').startswith(mine)
    ]


def _launch_batch_job(
    state_dict, parent_task_id, batch_idx, total_runs, shared_genre_map_token,
    target_per_genre, *args, conn=None
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

    batch_top_n = state_dict.get("top_n_clustering_playlist")
    if batch_top_n is None:
        batch_top_n = TOP_N_CLUSTERING_PLAYLIST

    job_args = {
        "batch_id_str": f"Batch_{batch_idx}",
        "start_run_idx": start_run,
        "num_iterations_in_batch": num_iterations,
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
        "top_n_playlists_param": batch_top_n,
    }

    with main_task_start_lock():
        parent = get_task_info_from_db(parent_task_id)
        if parent is None or parent.get('status') in (
            TASK_STATUS_SUCCESS,
            TASK_STATUS_FAILURE,
            TASK_STATUS_REVOKED,
        ):
            logger.info(
                "Not enqueueing batch %s because parent %s is missing or terminal.",
                batch_job_id, parent_task_id,
            )
            return False
        try:
            taskqueue.enqueue(
                'tasks.clustering.run_clustering_batch_task',
                kwargs=job_args,
                task_id=batch_job_id,
                task_type='clustering_batch',
                queue=taskqueue.QUEUE_DEFAULT,
                parent_task_id=parent_task_id,
                shared={
                    'genre_to_lightweight_track_data_map_json': shared_genre_map_token,
                },
                conn=conn,
            )
        except (taskqueue.TaskNotQueued, taskqueue.TaskAlreadyRunning):
            logger.info(
                "Batch %s already has a queue row from a previous attempt; leaving it "
                "to finish instead of re-enqueueing.", batch_job_id,
            )

    logger.info(
        f"Enqueued batch job {batch_job_id} for runs {start_run}-{start_run + num_iterations - 1}."
    )
    return True


def _previous_names_for_naming(server_id):
    if not CLUSTER_NAMING_AI_HISTORY:
        return []
    return get_recent_playlist_names(server_id, limit=60)


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
    previous_playlist_names=None,
):
    final_playlists = {}
    used_playlist_names = list(reversed(previous_playlist_names or []))
    assigned_names = set()
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
                used_playlist_names,
                primary_genre=best_result.get("playlist_primary_genres", {}).get(
                    original_name
                ),
            )
        except Exception as e:
            logger.warning(f"AI naming failed for '{original_name}': {e}. Using original name.")
            final_name = original_name

        temp_name = final_name
        suffix = 1
        while temp_name in assigned_names:
            suffix += 1
            temp_name = f"{final_name} ({suffix})"
        final_name = temp_name
        assigned_names.add(final_name)
        used_playlist_names.append(final_name)

        base_name = f"{final_name}_automatic"
        shuffled = _shuffle_playlist_songs(songs, base_name)
        _assign_playlist_chunks(shuffled, max_songs, base_name, final_playlists)

    return final_playlists
