# tasks/clustering.py

import os
import shutil
from collections import defaultdict
import numpy as np
import json
import time
import random
import logging
import uuid
import traceback
# Regex and distance calculations now handled in clustering_postprocessing.py

# RQ import
from rq import get_current_job, Retry
from rq.job import Job
from rq.exceptions import NoSuchJobError

# NOTE: Imports from 'app' are moved inside functions to prevent circular imports.
from psycopg2.extras import DictCursor

# Import configuration
from config import (MAX_SONGS_PER_CLUSTER, MOOD_LABELS, STRATIFIED_GENRES,
                    MUTATION_KMEANS_COORD_FRACTION, MUTATION_INT_ABS_DELTA, MUTATION_FLOAT_ABS_DELTA,
                    TOP_N_ELITES, EXPLOITATION_START_FRACTION, EXPLOITATION_PROBABILITY_CONFIG,
                    SAMPLING_PERCENTAGE_CHANGE_PER_RUN, ITERATIONS_PER_BATCH_JOB, MAX_CONCURRENT_BATCH_JOBS,
                    MIN_PLAYLIST_SIZE_FOR_TOP_N, CLUSTERING_BATCH_TIMEOUT_MINUTES, CLUSTERING_MAX_FAILED_BATCHES,
                    CLUSTERING_BATCH_CHECK_INTERVAL_SECONDS)

# Import AI naming function and prompt template
from ai import get_ai_playlist_name, creative_prompt_template
# Import media server functions
from .mediaserver import create_playlist, delete_automatic_playlists
# Import refactored clustering helpers
from .clustering_helper import (
    _get_stratified_song_subset,
    get_job_result_safely,
    _perform_single_clustering_iteration
)
# Import post-processing functions from dedicated module
from .clustering_postprocessing import (
    apply_duplicate_filtering_to_clustering_result,
    apply_minimum_size_filter_to_clustering_result,
    select_top_n_diverse_playlists
)

# we want to maintain np.float_ for backwards compatibility but it was removed in numpy 2.0
# the check below in sanitize_for_json causes an AttributeError that crashes the clustering algo
# since it tries to access np.float_ so we monkeypatch np.float_ to point to np.float64
if not np.__dict__.get('float_'):
    np.float_ = np.float64

logger = logging.getLogger(__name__)

def batch_task_failure_handler(job, connection, type, value, tb):
    """A failure handler for the clustering batch sub-task, executed by the worker."""
    from app import app
    from app_helper import save_task_status, TASK_STATUS_FAILURE
    with app.app_context():
        task_id = job.get_id()
        parent_id = job.kwargs.get('parent_task_id')
        batch_id_str = job.kwargs.get('batch_id_str')
        
        # --- FIX: Handle different traceback types, especially from rq-janitor ---
        tb_formatted = ""
        if isinstance(tb, traceback.StackSummary):
            tb_formatted = "".join(tb.format())
        else:
            tb_formatted = "".join(traceback.format_exception(type, value, tb))

        error_details = {
            "message": "Clustering batch sub-task failed permanently after all retries.",
            "error_type": str(type.__name__),
            "error_value": str(value),
            "traceback": tb_formatted
        }
        
        save_task_status(
            task_id,
            "clustering_batch",
            TASK_STATUS_FAILURE,
            parent_task_id=parent_id,
            sub_type_identifier=batch_id_str,
            progress=100,
            details=error_details
        )
        app.logger.error(f"Clustering batch task {task_id} (parent: {parent_id}) failed permanently. DB status updated.")

def _sanitize_for_json(obj):
    """
    Recursively converts numpy arrays and numpy numeric types to native Python types
    to ensure the object is JSON serializable.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle numpy numeric types which are not JSON serializable by default
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def run_clustering_batch_task(
    batch_id_str, start_run_idx, num_iterations_in_batch,
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
    enable_clustering_embeddings_param
):
    """
    Executes a batch of clustering iterations. This task is enqueued by the main clustering task.
    """
    # --- Local imports to prevent circular dependency ---
    from app import app
    from app_helper import (redis_conn, save_task_status, get_task_info_from_db,
                     TASK_STATUS_PROGRESS, TASK_STATUS_REVOKED, TASK_STATUS_FAILURE,
                     TASK_STATUS_SUCCESS)

    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())
    logger.info(f"Starting clustering batch task {current_task_id} (Batch: {batch_id_str})")

    with app.app_context():
        # Helper for logging and updating task status
        def _log_and_update(message, progress, details=None, state=TASK_STATUS_PROGRESS):
            logger.info(f"[ClusteringBatchTask-{current_task_id}] {message}")
            db_details = {
                "batch_id": batch_id_str,
                "start_run_idx": start_run_idx,
                "num_iterations_in_batch": num_iterations_in_batch,
                "status_message": message,
                **(details or {})
            }
            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.save_meta()
            save_task_status(current_task_id, "clustering_batch", state, parent_task_id=parent_task_id,
                             sub_type_identifier=batch_id_str, progress=progress, details=db_details)

        try:
            _log_and_update("Batch started.", 0)
            genre_to_lightweight_track_data_map = json.loads(genre_to_lightweight_track_data_map_json)
            elite_solutions_params_list = json.loads(elite_solutions_params_list_json)
            mutation_config = json.loads(mutation_config_json)
            current_sampled_track_ids = json.loads(initial_subset_track_ids_json)

            best_result_in_batch = None
            best_score_in_batch = -1.0 # Use -1.0 as a safe initial value
            iterations_completed = 0

            for i in range(num_iterations_in_batch):
                current_run_global_idx = start_run_idx + i

                # Revocation Check
                if current_job:
                    task_info = get_task_info_from_db(current_task_id)
                    parent_task_info = get_task_info_from_db(parent_task_id)
                    if (task_info and task_info.get('status') == TASK_STATUS_REVOKED) or \
                       (parent_task_info and parent_task_info.get('status') in [TASK_STATUS_REVOKED, TASK_STATUS_FAILURE]):
                        _log_and_update("Stopping batch due to revocation.", i, state=TASK_STATUS_REVOKED)
                        return {"status": "REVOKED", "message": "Batch task revoked."}

                # Get a new subset of songs for this iteration, perturbing the previous one
                percentage_change = 0.0 if i == 0 else sampling_percentage_change_per_run
                current_subset_lightweight_data = _get_stratified_song_subset(
                    genre_to_lightweight_track_data_map,
                    target_songs_per_genre,
                    prev_ids=current_sampled_track_ids,
                    percent_change=percentage_change
                )
                item_ids_for_iteration = [t['item_id'] for t in current_subset_lightweight_data]
                current_sampled_track_ids = list(item_ids_for_iteration)

                if not item_ids_for_iteration:
                    logger.warning(f"No songs in subset for iteration {current_run_global_idx}. Skipping.")
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
                    enable_clustering_embeddings=enable_clustering_embeddings_param
                )
                iterations_completed += 1

                if iteration_result and iteration_result.get("fitness_score", -1.0) > best_score_in_batch:
                    best_score_in_batch = iteration_result["fitness_score"]
                    best_result_in_batch = iteration_result

                progress = int(100 * (i + 1) / num_iterations_in_batch)
                _log_and_update(f"Iteration {current_run_global_idx} complete. Batch best score: {best_score_in_batch:.2f}", progress)

            # *** FIX: Sanitize the result to make it JSON-serializable before logging/returning ***
            if best_result_in_batch:
                best_result_in_batch = _sanitize_for_json(best_result_in_batch)

            final_details = {
                "best_score_in_batch": best_score_in_batch,
                "iterations_completed_in_batch": iterations_completed,
                "full_best_result_from_batch": best_result_in_batch,
                "final_subset_track_ids": current_sampled_track_ids
            }
            _log_and_update(f"Batch complete. Best score: {best_score_in_batch or -1:.2f}", 100, details=final_details, state=TASK_STATUS_SUCCESS)
            return {
                "status": "SUCCESS",
                "iterations_completed_in_batch": iterations_completed,
                "best_result_from_batch": best_result_in_batch,
                "final_subset_track_ids": current_sampled_track_ids
            }

        except Exception as e:
            logger.error(f"Clustering batch {batch_id_str} failed", exc_info=True)
            _log_and_update(f"Batch failed: {e}", 100, details={"error": str(e)}, state=TASK_STATUS_FAILURE)
            return {"status": "FAILURE", "message": str(e)}


def run_clustering_task(
    clustering_method, num_clusters_min, num_clusters_max,
    dbscan_eps_min, dbscan_eps_max, dbscan_min_samples_min, dbscan_min_samples_max,
    pca_components_min, pca_components_max, num_clustering_runs, max_songs_per_cluster_val,
    gmm_n_components_min, gmm_n_components_max,
    spectral_n_clusters_min, spectral_n_clusters_max,
    min_songs_per_genre_for_stratification_param,
    stratified_sampling_target_percentile_param,
    score_weight_diversity_param, score_weight_silhouette_param,
    score_weight_davies_bouldin_param, score_weight_calinski_harabasz_param,
    score_weight_purity_param,
    score_weight_other_feature_diversity_param,
    score_weight_other_feature_purity_param,
    ai_model_provider_param, ollama_server_url_param, ollama_model_name_param,
    gemini_api_key_param, gemini_model_name_param,
    mistral_api_key_param, mistral_model_name_param,
    top_n_moods_for_clustering_param,
    top_n_playlists_param, # *** NEW: Accept Top N parameter ***
    enable_clustering_embeddings_param):
    """
    Main entry point for the clustering process.
    Orchestrates data preparation, batch job creation, result aggregation, and playlist creation.
    """
    # --- Local imports to prevent circular dependency ---
    from app import app
    from app_helper import (redis_conn, get_db, save_task_status, get_task_info_from_db,
                     update_playlist_table, get_child_tasks_from_db,
                     TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
                     TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)

    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())
    logger.info(f"Starting main clustering task {current_task_id}")

    # Capture initial parameters for the final report
    initial_params = {
        "clustering_method": clustering_method,
        "pca_components_min": pca_components_min,
        "pca_components_max": pca_components_max,
        "use_embeddings": enable_clustering_embeddings_param,
        "top_n_playlists": top_n_playlists_param, # *** NEW: Log Top N parameter ***
        "stratification_percentile": stratified_sampling_target_percentile_param,
        "score_weights": {
            "mood_diversity": score_weight_diversity_param,
            "silhouette": score_weight_silhouette_param,
            "davies_bouldin": score_weight_davies_bouldin_param,
            "calinski_harabasz": score_weight_calinski_harabasz_param,
            "mood_purity": score_weight_purity_param,
            "other_feature_diversity": score_weight_other_feature_diversity_param,
            "other_feature_purity": score_weight_other_feature_purity_param
        }
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
        # IDEMPOTENCY CHECK
        task_info = get_task_info_from_db(current_task_id)
        if task_info and task_info.get('status') in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]:
            logger.info(f"Main clustering task {current_task_id} is already in a terminal state ('{task_info.get('status')}'). Skipping execution.")
            return {"status": task_info.get('status'), "message": f"Task already in terminal state '{task_info.get('status')}'.", "details": json.loads(task_info.get('details', '{}'))}

        # This dictionary will hold the state and be passed to the logging function.
        _main_task_accumulated_details = {
            "log": [],
            "total_runs": num_clustering_runs,
            "runs_completed": 0,
            "best_score": -1.0, # Use -1.0 as a safe initial value
            "best_result": None,
            "active_jobs": {},
            "elite_solutions": [],
            "last_subset_ids": [],
            "processed_job_ids": set(), # *** FIX 1: Add set to track processed jobs ***
            # NEW: Batch tracking for timeout and failure recovery
            "batch_start_times": {},  # job_id -> start_timestamp
            "failed_batches": set(),  # Set of failed/timed out batch job_ids
            "timed_out_batches": set() # Set of job_ids that have timed out
        }

        # Helper for logging and updating main task status, using a shared dictionary.
        def _log_and_update(message, progress, details_to_add_or_update=None, task_state=TASK_STATUS_PROGRESS):
            nonlocal _main_task_accumulated_details
            
            logger.info(f"[MainClusteringTask-{current_task_id}] {message}")
            if details_to_add_or_update:
                _main_task_accumulated_details.update(details_to_add_or_update)
            
            _main_task_accumulated_details["status_message"] = message
            
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            _main_task_accumulated_details["log"].append(log_entry)

            # Prepare details for saving (a copy to avoid modifying the original during iteration)
            details_for_db = _main_task_accumulated_details.copy()
            details_for_db.pop('active_jobs', None) # Don't save job objects to DB
            details_for_db.pop('best_result', None) # Don't save the full result object in every progress update
            details_for_db.pop('last_subset_ids', None) # Remove the large list of IDs
            details_for_db.pop('processed_job_ids', None) # Don't save the set of job IDs to DB
            details_for_db.pop('failed_batches', None) # Don't save the set of failed batches to DB
            details_for_db.pop('timed_out_batches', None) # Don't save the set of timed out batches to DB
            details_for_db.pop('batch_start_times', None) # Don't save batch start times to DB

            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.save_meta()
            
            save_task_status(current_task_id, "main_clustering", task_state, progress=progress, details=details_for_db)

        try:
            _log_and_update("Initializing clustering process...", 0, task_state=TASK_STATUS_STARTED)

            # --- 1. Data Preparation and Stratification Setup ---
            _log_and_update("Fetching lightweight track data for stratification...", 1)
            db = get_db()
            cur = db.cursor(cursor_factory=DictCursor)
            cur.execute("SELECT item_id, author, mood_vector FROM score WHERE mood_vector IS NOT NULL AND mood_vector != ''")
            lightweight_rows = cur.fetchall()
            cur.close()

            if len(lightweight_rows) < (num_clusters_min or 2):
                raise ValueError(f"Not enough tracks in DB ({len(lightweight_rows)}) for clustering.")

            genre_map = _prepare_genre_map(lightweight_rows)
            target_songs_per_genre = _calculate_target_songs_per_genre(
                genre_map, stratified_sampling_target_percentile_param, min_songs_per_genre_for_stratification_param
            )
            _log_and_update(f"Target songs per genre for stratification: {target_songs_per_genre}", 5)

            # --- 2. Batch Job Orchestration ---
            num_total_batches = (num_clustering_runs + ITERATIONS_PER_BATCH_JOB - 1) // ITERATIONS_PER_BATCH_JOB if ITERATIONS_PER_BATCH_JOB > 0 else 0
            next_batch_to_launch = 0
            batches_completed_count = 0

            # STATE RECOVERY
            child_tasks_from_db = get_child_tasks_from_db(current_task_id)
            if child_tasks_from_db:
                logger.info(f"Found {len(child_tasks_from_db)} existing child tasks. Attempting state recovery.")
                _monitor_and_process_batches(_main_task_accumulated_details, current_task_id, initial_check=True)
                # Count batches processed during recovery (these are now in processed_job_ids)
                batches_completed_count = len(_main_task_accumulated_details.get('processed_job_ids', set()))
                
                # Determine next batch to launch based on total runs accounted for
                runs_accounted_for = _main_task_accumulated_details["runs_completed"]
                next_batch_to_launch = runs_accounted_for // ITERATIONS_PER_BATCH_JOB
                
                logger.info(f"Recovery complete. Resuming. Runs accounted for: {runs_accounted_for}/{num_clustering_runs}. Next batch index to launch: {next_batch_to_launch}")

            if not _main_task_accumulated_details["last_subset_ids"]:
                initial_subset_data = _get_stratified_song_subset(genre_map, target_songs_per_genre)
                _main_task_accumulated_details["last_subset_ids"] = [t['item_id'] for t in initial_subset_data]

            while _main_task_accumulated_details["runs_completed"] < num_clustering_runs:
                if current_job and (current_job.is_stopped or get_task_info_from_db(current_task_id).get('status') == TASK_STATUS_REVOKED):
                    _log_and_update("Task revoked, stopping.", _main_task_accumulated_details['runs_completed'], task_state=TASK_STATUS_REVOKED)
                    return {"status": "REVOKED", "message": "Main clustering task revoked."}

                _monitor_and_process_batches(_main_task_accumulated_details, current_task_id)

                # Check if we should stop launching new batches due to too many failures
                failed_batch_count = len(_main_task_accumulated_details.get("failed_batches", set()))
                if failed_batch_count >= CLUSTERING_MAX_FAILED_BATCHES:
                    logger.warning(f"Stopping new batch launches: {failed_batch_count} batches have failed (max: {CLUSTERING_MAX_FAILED_BATCHES})")
                    # Force completion of remaining runs to prevent hanging
                    remaining_runs = num_clustering_runs - _main_task_accumulated_details["runs_completed"]
                    if remaining_runs > 0:
                        _main_task_accumulated_details["runs_completed"] = num_clustering_runs
                        logger.warning(f"Forced completion of {remaining_runs} remaining runs due to batch failures")
                
                while (len(_main_task_accumulated_details["active_jobs"]) < MAX_CONCURRENT_BATCH_JOBS 
                       and next_batch_to_launch < num_total_batches 
                       and failed_batch_count < CLUSTERING_MAX_FAILED_BATCHES):
                    _launch_batch_job(
                        _main_task_accumulated_details, current_task_id, next_batch_to_launch, num_clustering_runs,
                        genre_map, target_songs_per_genre, clustering_method,
                        num_clusters_min, num_clusters_max, dbscan_eps_min, dbscan_eps_max,
                        dbscan_min_samples_min, dbscan_min_samples_max, gmm_n_components_min,
                        gmm_n_components_max, spectral_n_clusters_min, spectral_n_clusters_max,
                        pca_components_min, pca_components_max, max_songs_per_cluster_val,
                        score_weight_diversity_param, score_weight_silhouette_param,
                        score_weight_davies_bouldin_param, score_weight_calinski_harabasz_param,
                        score_weight_purity_param, score_weight_other_feature_diversity_param,
                        score_weight_other_feature_purity_param, top_n_moods_for_clustering_param,
                        enable_clustering_embeddings_param
                    )
                    next_batch_to_launch += 1

                progress = 5 + int(85 * _main_task_accumulated_details["runs_completed"] / num_clustering_runs) if num_clustering_runs > 0 else 5
                _log_and_update(
                    f"Progress: {_main_task_accumulated_details['runs_completed']}/{num_clustering_runs} runs. Active batches: {len(_main_task_accumulated_details['active_jobs'])}. Best score: {_main_task_accumulated_details['best_score']:.2f}",
                    progress
                )
                
                # If all runs are accounted for (or runs_completed is now >= total_runs due to recovery/failure counting)
                # AND no jobs are active, we can break the loop safely.
                if _main_task_accumulated_details["runs_completed"] >= num_clustering_runs and len(_main_task_accumulated_details["active_jobs"]) == 0:
                    _log_and_update(f"All runs ({_main_task_accumulated_details['runs_completed']}) are processed or accounted for. Forcing loop exit to prevent starvation.", progress)
                    break
                
                time.sleep(3)
            
            # Ensure final state update accounts for any remaining jobs that finished right after the loop broke
            _monitor_and_process_batches(_main_task_accumulated_details, current_task_id)


            _log_and_update("All batches completed. Finalizing...", 90)

            # --- 3. Finalization and Playlist Creation ---
            if not _main_task_accumulated_details["best_result"]:
                raise ValueError("No valid clustering solution found after all runs.")

            best_result = _main_task_accumulated_details["best_result"]

            # --- POST-PROCESSING PIPELINE: Apply filtering steps after clustering is complete ---
            
            # Log initial state for debugging
            initial_playlist_count = len(best_result.get("named_playlists", {}))
            _log_and_update(f"Starting post-processing with {initial_playlist_count} playlists", 90.2)
            
            # *** STEP 1: Apply duplicate filtering to remove similar songs within playlists ***
            # Uses the same distance-based filtering logic as voyager_manager to avoid duplicate tracks
            _log_and_update("Applying duplicate filtering to remove similar songs...", 90.5)
            _log_and_update(f"Before duplicate filtering: {len(best_result.get('named_playlists', {}))} playlists", 90.5)
            best_result = apply_duplicate_filtering_to_clustering_result(best_result, log_prefix="[DuplicateFilter] ")
            _log_and_update(f"After duplicate filtering: {len(best_result.get('named_playlists', {}))} playlists", 90.5)
            
            # *** STEP 2: Apply minimum size filter to remove small playlists ***
            # Removes playlists with fewer than the configured minimum number of songs
            # Use the configured minimum playlist size from config.py
            min_size_threshold = MIN_PLAYLIST_SIZE_FOR_TOP_N
            _log_and_update(f"Applying minimum size filter (>= {min_size_threshold} songs)...", 91)
            _log_and_update(f"Before minimum size filtering: {len(best_result.get('named_playlists', {}))} playlists", 91)
            best_result = apply_minimum_size_filter_to_clustering_result(best_result, min_size_threshold, log_prefix="[MinSizeFilter] ")
            _log_and_update(f"After minimum size filtering: {len(best_result.get('named_playlists', {}))} playlists", 91)

            # *** STEP 3: Filter for Top N Most Diverse Playlists (only if still needed) ***
            # Selects the N most diverse playlists if there are more than requested
            if top_n_playlists_param > 0 and len(best_result.get("named_playlists", {})) > top_n_playlists_param:
                _log_and_update(f"Filtering for Top {top_n_playlists_param} most diverse playlists...", 91.5)
                best_result = select_top_n_diverse_playlists(best_result, top_n_playlists_param)
                _main_task_accumulated_details["best_result"] = best_result # Update main dict with filtered result
                
            final_playlist_count = len(best_result.get("named_playlists", {}))
            _log_and_update(f"Post-processing complete: {initial_playlist_count} -> {final_playlist_count} playlists", 91.8)

            _log_and_update(f"Best clustering found with score: {_main_task_accumulated_details['best_score']:.2f}. Creating playlists...", 92)

            final_playlists_with_details = _name_and_prepare_playlists(
                best_result, # Use the potentially filtered result
                ai_model_provider_param, ollama_server_url_param,
                ollama_model_name_param, gemini_api_key_param, gemini_model_name_param,
                mistral_api_key_param, mistral_model_name_param,
                enable_clustering_embeddings_param
            )

            _log_and_update("Deleting existing automatic playlists...", 97)
            delete_automatic_playlists()

            # *** ABSOLUTE FINAL SHUFFLE: Guarantee random order right before database storage ***
            logger.info("=== ABSOLUTE FINAL SHUFFLE: Randomizing all playlists before database storage ===")
            final_shuffled_playlists = {}
            for playlist_name, songs_list in final_playlists_with_details.items():
                if len(songs_list) > 1:
                    # ULTIMATE FISHER-YATES SHUFFLE - Last chance to randomize before database
                    shuffled_list = songs_list.copy()
                    n = len(shuffled_list)
                    # Triple-source randomization: system time + random + position-based seed
                    ultra_seed = int(time.time() * 1000000) % 1000000
                    
                    # Apply Fisher-Yates with enhanced randomization
                    for i in range(n - 1, 0, -1):
                        # Multi-source random index generation
                        base_random = random.randint(0, i)
                        time_component = ultra_seed % (i + 1)
                        position_component = (i * 13 + 7) % (i + 1)
                        j = (base_random + time_component + position_component) % (i + 1)
                        
                        # Perform swap
                        shuffled_list[i], shuffled_list[j] = shuffled_list[j], shuffled_list[i]
                        ultra_seed = (ultra_seed * 1664525 + 1013904223) % (2**32)
                    
                    # Verify shuffle effectiveness
                    original_first_5 = [song[1] for song in songs_list[:5]]
                    shuffled_first_5 = [song[1] for song in shuffled_list[:5]]
                    
                    logger.info(f"ULTIMATE SHUFFLE '{playlist_name}': {len(shuffled_list)} songs")
                    logger.info(f"  ORIGINAL: {original_first_5}")
                    logger.info(f"  SHUFFLED: {shuffled_first_5}")
                    
                    # Emergency fallback if shuffle didn't work (shouldn't happen)
                    if original_first_5 == shuffled_first_5:
                        logger.warning(f"Emergency fallback: manually reversing '{playlist_name}'")
                        shuffled_list = list(reversed(shuffled_list))
                    
                    final_shuffled_playlists[playlist_name] = shuffled_list
                else:
                    final_shuffled_playlists[playlist_name] = songs_list
                    logger.info(f"ULTIMATE SHUFFLE '{playlist_name}': only {len(songs_list)} songs, no shuffle needed")

            _log_and_update(f"Creating {len(final_shuffled_playlists)} new playlists...", 98)
            for name, songs_with_details in final_shuffled_playlists.items():
                item_ids = [item_id for item_id, _, _ in songs_with_details]
                create_playlist(name, item_ids)

            update_playlist_table(final_shuffled_playlists)

            # --- Final Success Reporting ---
            final_message = "Clustering task completed successfully!"
            
            # Add final message to the log before preparing the summary
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {final_message}"
            _main_task_accumulated_details["log"].append(log_entry)
            logger.info(f"[MainClusteringTask-{current_task_id}] {final_message}")

            final_log = _main_task_accumulated_details.get('log', [])
            truncated_log = final_log[-10:]

            # This dictionary is the final, clean state for the DB.
            # It includes running parameters, excludes elite solutions, and has a truncated log.
            final_db_summary = {
                "status_message": final_message,
                "running_parameters": initial_params,
                "best_score": _main_task_accumulated_details["best_score"],
                "best_params": _main_task_accumulated_details["best_result"].get("parameters"),
                "num_playlists_created": len(final_playlists_with_details),
                "log": truncated_log,
                "log_storage_info": f"Log truncated to last {len(truncated_log)} entries. Original length: {len(final_log)}." if len(final_log) > 10 else "Full log."
            }
            
            if current_job:
                current_job.meta['progress'] = 100
                current_job.meta['status_message'] = final_message
                current_job.save_meta()

            # Direct call to save_task_status with the clean details object
            save_task_status(current_task_id, "main_clustering", TASK_STATUS_SUCCESS, progress=100, details=final_db_summary)

            return {"status": "SUCCESS", "message": f"Playlists created. Best score: {_main_task_accumulated_details['best_score']:.2f}"}

        except Exception as e:
            logger.critical("FATAL ERROR in main clustering task", exc_info=True)
            _log_and_update(f"Task failed: {e}", 100, details_to_add_or_update={"error": str(e)}, task_state=TASK_STATUS_FAILURE)
            raise

# --- Internal Helper Functions for run_clustering_task ---

def _prepare_genre_map(lightweight_rows):
    """Creates a map of genre -> list of tracks from raw DB rows."""
    genre_map = defaultdict(list)
    for row in lightweight_rows:
        if row.get('mood_vector'):
            mood_scores = {p.split(':')[0]: float(p.split(':')[1]) for p in row['mood_vector'].split(',') if ':' in p}
            top_genre = max((g for g in STRATIFIED_GENRES if g in mood_scores), key=mood_scores.get, default='__other__')
            genre_map[top_genre].append({'item_id': row['item_id'], 'mood_vector': row['mood_vector']})
    return genre_map

def _calculate_target_songs_per_genre(genre_map, percentile, min_songs):
    """Calculates the target number of songs per genre for stratification."""
    counts = [len(tracks) for g, tracks in genre_map.items() if g in STRATIFIED_GENRES]
    if not counts:
        return min_songs
    target = np.percentile(counts, np.clip(percentile, 0, 100))
    return max(min_songs, int(np.floor(target)))

def _monitor_and_process_batches(state_dict, parent_task_id, initial_check=False):
    """
    Robust batch monitoring with timeout and failure recovery.
    
    This function ensures clustering never hangs forever by:
    1. Tracking batch start times and detecting timeouts
    2. Processing timed-out batches as failed but continuing
    3. Limiting the number of failed batches before stopping
    4. Always making progress even if some batches fail
    
    CRITICAL: This prevents the main task from hanging at 4980/5000 runs
    by implementing timeouts and forced progress tracking.
    """
    import time
    from app_helper import (redis_conn, get_child_tasks_from_db, get_task_info_from_db,
                    TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED, 
                    TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS)
    
    current_time = time.time()
    timeout_seconds = CLUSTERING_BATCH_TIMEOUT_MINUTES * 60
    processed_jobs = state_dict.get("processed_job_ids", set())
    
    # 1. Check for timed-out batches first - CRITICAL for preventing hangs
    timed_out_jobs = []
    for job_id, start_time in list(state_dict.get("batch_start_times", {}).items()):
        if job_id not in processed_jobs:
            elapsed_time = current_time - start_time
            if elapsed_time > timeout_seconds:
                logger.warning(f"TIMEOUT: Batch {job_id} has timed out after {elapsed_time/60:.1f} minutes (limit: {CLUSTERING_BATCH_TIMEOUT_MINUTES} min)")
                timed_out_jobs.append(job_id)
                state_dict.setdefault("timed_out_batches", set()).add(job_id)
                state_dict.setdefault("failed_batches", set()).add(job_id)  # Timeouts count as failures
    
    # 2. Get all child tasks from database
    all_child_tasks = get_child_tasks_from_db(parent_task_id)

    # 2. Identify all jobs that need a status check or result processing (i.e., not processed yet).
    jobs_for_status_check = []
    for task_info in all_child_tasks:
        if task_info['task_id'] not in processed_jobs:
            jobs_for_status_check.append(task_info)
    
    # Add jobs known to be active in memory but might not be in the DB yet (for safety right after launch).
    for job_id in state_dict["active_jobs"].keys():
        if job_id not in processed_jobs and not any(t['task_id'] == job_id for t in jobs_for_status_check):
            jobs_for_status_check.append({'task_id': job_id, 'status': TASK_STATUS_STARTED, 'sub_type_identifier': None, 'details': None}) # Mock DB info if not found

    jobs_ready_for_result_extraction = []

    for task_info in jobs_for_status_check:
        job_id = task_info['task_id']
        db_status = task_info['status']
        
        is_terminal_in_db = db_status in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]

        if is_terminal_in_db:
            # If DB is terminal, we must process the result now to count the runs.
            jobs_ready_for_result_extraction.append(job_id)
            continue
            
        # If DB status is non-terminal, check RQ status
        try:
            job = Job.fetch(job_id, connection=redis_conn)
            if job.is_finished or job.is_failed or job.get_status() == 'canceled':
                jobs_ready_for_result_extraction.append(job_id)
            elif job_id not in state_dict["active_jobs"]:
                 # If it's active in RQ but not in memory, add it to active_jobs.
                 state_dict["active_jobs"][job_id] = job
        except NoSuchJobError:
            # Job not in RQ (cleared) and not marked terminal in DB. 
            # This is the original stuck case. We assume it's done/cleared and process it.
            logger.warning(f"Job {job_id} (status: {db_status}) not found in RQ (likely cleared). Treating as finished to prevent main task starvation.")
            jobs_ready_for_result_extraction.append(job_id)
        except Exception as e:
            # Generic error during RQ fetch (e.g., connection issue). Assume terminal to prevent starvation.
            logger.error(f"Error checking RQ status for job {job_id}: {e}. Assuming terminal state to prevent starvation.")
            jobs_ready_for_result_extraction.append(job_id)


    # 3. Process all identified finished/ready jobs
    for job_id in jobs_ready_for_result_extraction:
        # Re-check processed set (shouldn't happen here, but safe)
        if job_id in processed_jobs:
            continue
            
        # Try to get the result from RQ or DB.
        result = get_job_result_safely(job_id, parent_task_id, "clustering_batch")
        
        # If successful, process result and count runs
        if result and result.get("status") == TASK_STATUS_SUCCESS:
            state_dict["runs_completed"] += result.get("iterations_completed_in_batch", 0)
            state_dict["last_subset_ids"] = result.get("final_subset_track_ids", state_dict["last_subset_ids"])
            best_from_batch = result.get("best_result_from_batch")
            if best_from_batch:
                current_best_score = best_from_batch.get("fitness_score", -1.0)
                state_dict["elite_solutions"].append({
                    "score": current_best_score,
                    "params": best_from_batch.get("parameters")
                })
                if current_best_score > state_dict["best_score"]:
                    state_dict["best_score"] = current_best_score
                    state_dict["best_result"] = best_from_batch
        else:
            # Track this as a failed batch
            state_dict.setdefault("failed_batches", set()).add(job_id)
            
            # --- FIX: Account for runs from jobs that failed or were force-processed with no usable result ---
            # This is critical for the starvation case where 4940/5000 is stuck.
            task_info_for_runs = next((t for t in all_child_tasks if t['task_id'] == job_id), None)
            
            # We must rely on the sub_type_identifier stored in the database by the batch task.
            if task_info_for_runs and task_info_for_runs.get('sub_type_identifier'):
                if task_info_for_runs['sub_type_identifier'].startswith('Batch_'):
                    try:
                        batch_idx = int(task_info_for_runs['sub_type_identifier'].split('_')[-1])
                        total_runs = state_dict['total_runs']
                        
                        start_run = batch_idx * ITERATIONS_PER_BATCH_JOB
                        num_iterations = min(ITERATIONS_PER_BATCH_JOB, total_runs - start_run)
                        
                        if num_iterations > 0 and state_dict["runs_completed"] < total_runs:
                             runs_to_add = min(num_iterations, total_runs - state_dict["runs_completed"])
                             state_dict["runs_completed"] += runs_to_add
                             logger.warning(f"Job {job_id} failed/missing result. Forced runs_completed count to increase by {runs_to_add} to prevent main task starvation.")
                             
                    except Exception:
                        logger.error(f"Could not calculate runs for failed/missing job {job_id} using sub_type_identifier.")
        
        # Mark as processed and remove from active jobs list
        state_dict.setdefault("processed_job_ids", set()).add(job_id)
        if job_id in state_dict["active_jobs"]:
            del state_dict["active_jobs"][job_id]

    # Check if we have too many failed batches
    failed_batch_count = len(state_dict.get("failed_batches", set()))
    if failed_batch_count >= CLUSTERING_MAX_FAILED_BATCHES:
        logger.warning(f"Reached maximum failed batches ({failed_batch_count}/{CLUSTERING_MAX_FAILED_BATCHES}). Some jobs may be unstable.")

    # Prune elite solutions to keep only the best
    state_dict["elite_solutions"].sort(key=lambda x: x["score"], reverse=True)
    state_dict["elite_solutions"] = state_dict["elite_solutions"][:TOP_N_ELITES]


def _launch_batch_job(state_dict, parent_task_id, batch_idx, total_runs, genre_map, target_per_genre, *args):
    """Constructs and enqueues a single batch job."""
    from app_helper import rq_queue_default # Local import to avoid circular dependency issues at top-level

    # Unpack all the parameters passed via *args
    (
        clustering_method,
        num_clusters_min, num_clusters_max, dbscan_eps_min, dbscan_eps_max,
        dbscan_min_samples_min, dbscan_min_samples_max, gmm_n_components_min,
        gmm_n_components_max, spectral_n_clusters_min, spectral_n_clusters_max,
        pca_components_min, pca_components_max, max_songs_per_cluster,
        score_weight_diversity, score_weight_silhouette, score_weight_davies_bouldin,
        score_weight_calinski_harabasz, score_weight_purity,
        score_weight_other_feature_diversity, score_weight_other_feature_purity,
        top_n_moods, enable_embeddings
    ) = args

    batch_job_id = f"{parent_task_id}_batch_{batch_idx}"
    start_run = batch_idx * ITERATIONS_PER_BATCH_JOB
    num_iterations = min(ITERATIONS_PER_BATCH_JOB, total_runs - start_run)

    exploitation_prob = EXPLOITATION_PROBABILITY_CONFIG if start_run >= (total_runs * EXPLOITATION_START_FRACTION) else 0.0

    # Package parameters for the batch task
    job_args = {
        "batch_id_str": f"Batch_{batch_idx}",
        "start_run_idx": start_run,
        "num_iterations_in_batch": num_iterations,
        "genre_to_lightweight_track_data_map_json": json.dumps(genre_map),
        "target_songs_per_genre": target_per_genre,
        "sampling_percentage_change_per_run": SAMPLING_PERCENTAGE_CHANGE_PER_RUN,
        "clustering_method": clustering_method,
        "active_mood_labels_for_batch": MOOD_LABELS[:top_n_moods] if top_n_moods > 0 else MOOD_LABELS,
        "num_clusters_min_max_tuple": (num_clusters_min, num_clusters_max),
        "dbscan_params_ranges_dict": {"eps_min": dbscan_eps_min, "eps_max": dbscan_eps_max, "samples_min": dbscan_min_samples_min, "samples_max": dbscan_min_samples_max},
        "gmm_params_ranges_dict": {"n_components_min": gmm_n_components_min, "n_components_max": gmm_n_components_max},
        "spectral_params_ranges_dict": {"n_clusters_min": spectral_n_clusters_min, "n_clusters_max": spectral_n_clusters_max},
        "pca_params_ranges_dict": {"components_min": pca_components_min, "components_max": pca_components_max},
        "max_songs_per_cluster": max_songs_per_cluster,
        "parent_task_id": parent_task_id,
        "score_weights_dict": {
            "mood_diversity": score_weight_diversity, 
            "silhouette": score_weight_silhouette,
            "davies_bouldin": score_weight_davies_bouldin, 
            "calinski_harabasz": score_weight_calinski_harabasz,
            "mood_purity": score_weight_purity, 
            "other_feature_diversity": score_weight_other_feature_diversity,
            "other_feature_purity": score_weight_other_feature_purity
        },
        "elite_solutions_params_list_json": json.dumps([e["params"] for e in state_dict["elite_solutions"]]),
        "exploitation_probability": exploitation_prob,
        "mutation_config_json": json.dumps({
            "int_abs_delta": MUTATION_INT_ABS_DELTA, "float_abs_delta": MUTATION_FLOAT_ABS_DELTA,
            "coord_mutation_fraction": MUTATION_KMEANS_COORD_FRACTION
        }),
        "initial_subset_track_ids_json": json.dumps(state_dict["last_subset_ids"]),
        "enable_clustering_embeddings_param": enable_embeddings
    }

    new_job = rq_queue_default.enqueue(
        'tasks.clustering.run_clustering_batch_task',
        kwargs=job_args,
        job_id=batch_job_id,
        job_timeout=CLUSTERING_BATCH_TIMEOUT_MINUTES * 60,  # Convert minutes to seconds
        retry=Retry(max=3),
        on_failure=batch_task_failure_handler
    )
    state_dict["active_jobs"][new_job.id] = new_job
    
    # Record batch start time for timeout detection
    import time
    state_dict.setdefault("batch_start_times", {})[new_job.id] = time.time()
    
    logger.info(f"Enqueued batch job {new_job.id} for runs {start_run}-{start_run + num_iterations - 1}.")


def _name_and_prepare_playlists(best_result, ai_provider, ollama_url, ollama_model, gemini_key, gemini_model, mistral_key, mistral_model, embeddings_used):
    """
    Uses AI to name playlists and formats them for creation.
    Returns a dictionary mapping final playlist names to lists of song tuples (id, title, author).
    """
    final_playlists = {}
    centroids = best_result.get("playlist_centroids", {})
    named_playlists = best_result.get("named_playlists", {})
    max_songs = best_result.get("parameters", {}).get("max_songs_per_cluster", MAX_SONGS_PER_CLUSTER)

    for original_name, songs in named_playlists.items():
        if not songs:
            continue

        final_name = original_name
        if ai_provider in ["OLLAMA", "GEMINI", "MISTRAL"]:
            try:
                # Simplified feature extraction for AI prompt
                name_parts = original_name.split('_')
                feature1 = name_parts[0] if len(name_parts) > 0 else "Music"
                feature2 = name_parts[1] if len(name_parts) > 1 else "Vibes"
                feature3 = name_parts[2] if len(name_parts) > 2 else "Collection"
                if embeddings_used:
                    feature1, feature2, feature3 = "Vibe", "Focused", "Collection"

                ai_name = get_ai_playlist_name(
                    ai_provider, ollama_url, ollama_model, gemini_key, gemini_model,
                    mistral_key, mistral_model,
                    creative_prompt_template, feature1, feature2, feature3,
                    [{'title': s_title, 'author': s_author} for _, s_title, s_author in songs],
                    centroids.get(original_name, {})
                )
                if ai_name and "Error" not in ai_name:
                    final_name = ai_name.strip().replace("\n", " ")
            except Exception as e:
                logger.warning(f"AI naming failed for '{original_name}': {e}. Using original name.")

        # Ensure unique names
        temp_name = final_name
        suffix = 1
        while temp_name in final_playlists:
            suffix += 1
            temp_name = f"{final_name} ({suffix})"
        final_name = temp_name

        # Add suffix and handle chunking
        base_name_with_suffix = f"{final_name}_automatic"
        
        # The 'songs' variable is already the list of tuples: [(item_id, title, author), ...]
        # *** FINAL SAFETY SHUFFLE: Ensure songs are randomized in final playlists ***
        final_songs = songs.copy()
        n = len(final_songs)
        
        if n > 1:
            # FISHER-YATES MANUAL SHUFFLE - GUARANTEED TO RANDOMIZE
            current_time_seed = int(time.time() * 1000000) % 1000000
            
            for i in range(n - 1, 0, -1):
                j = (random.randint(0, i) + current_time_seed + i * 7) % (i + 1)
                final_songs[i], final_songs[j] = final_songs[j], final_songs[i]
                current_time_seed = (current_time_seed * 1103515245 + 12345) % (2**31)
            
            logger.info(f"FINAL FISHER-YATES SHUFFLE applied to '{base_name_with_suffix}': {len(final_songs)} songs")
            logger.info(f"FINAL ORDER: First song = '{final_songs[0][1]}', Last song = '{final_songs[-1][1]}'")
        else:
            logger.info(f"FINAL: '{base_name_with_suffix}' has only {n} songs - no shuffling needed")
        
        if max_songs > 0 and len(final_songs) > max_songs:
             chunks = [final_songs[i:i+max_songs] for i in range(0, len(final_songs), max_songs)]
             for idx, chunk in enumerate(chunks, 1):
                 final_playlists[f"{base_name_with_suffix} ({idx})"] = chunk # Store the chunk of tuples
        else:
            final_playlists[base_name_with_suffix] = final_songs # Store the list of tuples

    return final_playlists



