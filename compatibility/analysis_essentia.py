# tasks/analysis.py

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


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score # type: ignore
from sklearn.preprocessing import StandardScaler

# Essentia imports
from essentia.standard import MonoLoader, RhythmExtractor2013, KeyExtractor, TensorflowPredictMusiCNN, TensorflowPredict2D, Energy # type: ignore

# RQ import
from rq import get_current_job, Retry
from rq.job import Job # Import Job class
from rq.exceptions import NoSuchJobError, InvalidJobOperation

# NOTE: Top-level 'from app import ...' has been removed to prevent circular dependencies
# and moved into the functions that require them.

# Import project modules
from . import mediaserver
from .commons import score_vector
from .voyager_manager import build_and_store_voyager_index
from ai import get_ai_playlist_name, creative_prompt_template

# Import configuration (ensure config.py is in PYTHONPATH or same directory)
from config import (TEMP_DIR, MAX_DISTANCE, MAX_SONGS_PER_CLUSTER, MAX_SONGS_PER_ARTIST,
    GMM_COVARIANCE_TYPE, MOOD_LABELS, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, ENERGY_MIN, ENERGY_MAX,
    TEMPO_MIN_BPM, TEMPO_MAX_BPM, OTHER_FEATURE_LABELS, REDIS_URL, DATABASE_URL,
    OLLAMA_SERVER_URL, OLLAMA_MODEL_NAME, AI_MODEL_PROVIDER, GEMINI_API_KEY, GEMINI_MODEL_NAME,
    DANCEABILITY_MODEL_PATH, AGGRESSIVE_MODEL_PATH, HAPPY_MODEL_PATH, PARTY_MODEL_PATH, RELAXED_MODEL_PATH, SAD_MODEL_PATH,
    SCORE_WEIGHT_SILHOUETTE, SCORE_WEIGHT_DAVIES_BOULDIN, SCORE_WEIGHT_CALINSKI_HARABASZ, # type: ignore
    SCORE_WEIGHT_DIVERSITY, SCORE_WEIGHT_PURITY, SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY, SCORE_WEIGHT_OTHER_FEATURE_PURITY,
    MUTATION_KMEANS_COORD_FRACTION, MUTATION_INT_ABS_DELTA, MUTATION_FLOAT_ABS_DELTA,
    TOP_N_ELITES, EXPLOITATION_START_FRACTION, EXPLOITATION_PROBABILITY_CONFIG, TOP_N_MOODS, TOP_N_OTHER_FEATURES,
    STRATIFIED_GENRES, MIN_SONGS_PER_GENRE_FOR_STRATIFICATION, SAMPLING_PERCENTAGE_CHANGE_PER_RUN, ITERATIONS_PER_BATCH_JOB, MAX_CONCURRENT_BATCH_JOBS, REBUILD_INDEX_BATCH_SIZE, # type: ignore
    MAX_QUEUED_ANALYSIS_JOBS, # New config for limiting queued analysis jobs
    TOP_K_MOODS_FOR_PURITY_CALCULATION, LN_MOOD_DIVERSITY_STATS, LN_MOOD_PURITY_STATS,
    LN_OTHER_FEATURES_DIVERSITY_STATS, LN_OTHER_FEATURES_PURITY_STATS, # Import new stats for other features
    STRATIFIED_SAMPLING_TARGET_PERCENTILE, # Import new config
    OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY as CONFIG_OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY, OPENAI_DMR_MODEL_NAME, OPENAI_API_KEY, DMR_BASE_URL) # Import the new config


logger = logging.getLogger(__name__)

# Task specific function
def clean_temp(temp_dir):
    """Cleans up the temporary directory."""
    os.makedirs(temp_dir, exist_ok=True)
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.warning("Could not remove %s from %s: %s", file_path, temp_dir, e)

def predict_moods(embeddings_input, prediction_model_path, mood_labels_list):
    """
    Predicts moods using a 2D matrix of segment-level embeddings.
    It returns a dictionary of track-level mood scores (averaged from segment predictions).
    """
    # TensorflowPredict2D expects a 2D matrix (segments, features), which is what raw_musicnn_embeddings is.
    model = TensorflowPredict2D(
        graphFilename=prediction_model_path,
        input="serving_default_model_Placeholder", # Ensure this matches your mood model's input tensor name
        output="PartitionedCall" # Ensure this matches your mood model's output tensor name
    )
    # The model will output a 2D matrix of predictions (segments, num_moods).
    segment_predictions = model(embeddings_input)

    # Average the predictions across all segments to get a single track-level prediction vector.
    track_level_predictions = np.mean(segment_predictions, axis=0)

    mood_results = dict(zip(mood_labels_list, track_level_predictions))
    return {label: float(score) for label, score in mood_results.items()}

def predict_other_models(embeddings): # Now accepts embeddings
    """
    Predicts other features using a 2D matrix of segment-level embeddings.
    Returns a dictionary of track-level scores.
    """
    model_paths = {
        "danceable": DANCEABILITY_MODEL_PATH,
        "aggressive": AGGRESSIVE_MODEL_PATH,
        "happy": HAPPY_MODEL_PATH,
        "party": PARTY_MODEL_PATH,
        "relaxed": RELAXED_MODEL_PATH,
        "sad": SAD_MODEL_PATH,
    }
    predictions = {}
    for mood, path in model_paths.items():
        try:
            model = TensorflowPredict2D(graphFilename=path, output="model/Softmax") # Using "model/Softmax" as per your example
            # The model will output a 2D matrix of predictions (segments, 2 for binary classification).
            segment_predictions = model(embeddings)

            # Average the predictions across all segments.
            track_level_prediction = np.mean(segment_predictions, axis=0)

            if track_level_prediction is not None and track_level_prediction.size == 2:
                predictions[mood] = float(track_level_prediction[0])  # Probability of the positive class
            else:
                predictions[mood] = 0.0  # Default value if prediction is invalid
        except Exception as e:
            logger.error("Error predicting %s: %s", mood, e, exc_info=True)
            predictions[mood] = 0.0  # Default value in case of error
    return predictions


def analyze_track(file_path, embedding_model_path, prediction_model_path, mood_labels_list):
    """Analyzes a single track for tempo, key, scale, moods, and other models."""
    try:
        # Load audio once at 16000 Hz for all analyses.
        audio = MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()
    except RuntimeError as e:
        # Essentia often wraps loading errors in a generic RuntimeError.
        # Catch any audio loading error, log it, and skip the track.
        logger.warning("Skipping track %s due to an audio loading error in Essentia: %s", os.path.basename(file_path), e)
        return None, None # Return a specific failure signal

    # Generate MusiCNN embeddings once
    musicnn_embedding_model = TensorflowPredictMusiCNN(
        graphFilename=embedding_model_path, output="model/dense/BiasAdd" # Main embedding model
    )
    # This is the 2D matrix of segment embeddings, which will be used for predictions.
    raw_musicnn_embeddings = musicnn_embedding_model(audio)

    processed_musicnn_embeddings = np.array([]) # Initialize as empty 1D array

    if isinstance(raw_musicnn_embeddings, np.ndarray):
        if raw_musicnn_embeddings.ndim == 1:
            processed_musicnn_embeddings = raw_musicnn_embeddings
        elif raw_musicnn_embeddings.ndim == 2:
            if raw_musicnn_embeddings.shape[0] > 0: # If there are rows (segments or batch_size=1)
                # Average across the first dimension
                processed_musicnn_embeddings = np.mean(raw_musicnn_embeddings, axis=0)
            else: # Shape is (0, D) - no rows
                logger.warning("Raw MusicNN embeddings are 2D with no rows (shape: %s). Resulting in empty 1D embedding.", raw_musicnn_embeddings.shape) # type: ignore
                # processed_musicnn_embeddings remains np.array([])
        else: # ndim > 2 or ndim == 0 (scalar, unlikely)
            logger.warning("Raw MusicNN embeddings have unexpected ndim: %s (shape: %s). Resulting in empty 1D embedding.", raw_musicnn_embeddings.ndim, raw_musicnn_embeddings.shape) # type: ignore
            # processed_musicnn_embeddings remains np.array([])
    else:
        logger.warning("Raw MusicNN output is not a NumPy array. Type: %s. Resulting in empty 1D embedding.", type(raw_musicnn_embeddings))
        # processed_musicnn_embeddings remains np.array([])

    tempo, _, _, _, _ = RhythmExtractor2013()(audio)
    key, scale, _ = KeyExtractor()(audio)

    # Calculate raw total energy
    raw_total_energy = Energy()(audio)
    num_samples = len(audio)

    # Calculate average energy per sample
    if num_samples > 0:
        average_energy_per_sample = raw_total_energy / float(num_samples)
    else:
        average_energy_per_sample = 0.0 # Should not happen for valid audio

    # Initialize moods and other_predictions with defaults
    moods = {label: 0.0 for label in mood_labels_list}
    other_predictions = {
        "danceable": 0.0, "aggressive": 0.0, "happy": 0.0,
        "party": 0.0, "relaxed": 0.0, "sad": 0.0
    }

    # --- Perform predictions using the RAW (2D) segment-level embeddings ---
    # This is more accurate as it averages predictions, not features.
    if isinstance(raw_musicnn_embeddings, np.ndarray) and raw_musicnn_embeddings.ndim == 2 and raw_musicnn_embeddings.shape[0] > 0:
        try:
            moods = predict_moods(raw_musicnn_embeddings, PREDICTION_MODEL_PATH, MOOD_LABELS)
        except Exception as e_mood:
            logger.error("Error during predict_moods: %s. Using default moods.", e_mood, exc_info=True)
        try:
            other_predictions = predict_other_models(raw_musicnn_embeddings)
        except Exception as e_other:
            logger.error("Error during predict_other_models: %s. Using default other_predictions.", e_other, exc_info=True)
    else:
        logger.warning("Raw MusicNN embeddings are not a valid 2D matrix. Skipping mood/other predictions. Shape: %s", raw_musicnn_embeddings.shape if isinstance(raw_musicnn_embeddings, np.ndarray) else 'N/A')

    # Combine all predictions into a single dictionary
    all_predictions = {
        "tempo": tempo,
        "key": key,
        "scale": scale,
        "moods": moods,
        "energy": float(average_energy_per_sample), # Store average energy per sample
        **other_predictions  # Include danceable, aggressive, etc. directly
    }
    # Return the predictions dictionary and the PROCESSED (1D) embeddings numpy array
    return all_predictions, processed_musicnn_embeddings

# --- RQ Task Definitions ---


def analyze_album_task(album_id, album_name, top_n_moods, parent_task_id):
    """RQ task to analyze a single album."""
    # --- LOCAL IMPORTS TO PREVENT CIRCULAR DEPENDENCIES ---
    from app import (app, redis_conn, get_db, save_task_status, get_task_info_from_db,
                     save_track_analysis, save_track_embedding, JobStatus,
                     TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
                     TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)
    # --- END LOCAL IMPORTS ---

    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        initial_details = {"album_name": album_name, "log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Album analysis task started."]}
        save_task_status(current_task_id, "album_analysis", TASK_STATUS_STARTED, parent_task_id=parent_task_id, sub_type_identifier=album_id, progress=0, details=initial_details)
        tracks_analyzed_count = 0
        tracks_skipped_count = 0
        current_progress_val = 0
        # Accumulate logs here for the current task
        current_task_logs = initial_details["log"]

        def log_and_update_album_task(message, progress, current_track_name=None, task_state=TASK_STATUS_PROGRESS, current_track_analysis_details=None, final_summary_details=None):
            nonlocal current_progress_val
            nonlocal current_task_logs
            current_progress_val = progress
            logger.info("[AlbumTask-%s-%s] %s", current_task_id, album_name, message)

            # Base details for DB
            db_details = {"album_name": album_name}
            if current_track_name: db_details["current_track"] = current_track_name
            if current_track_analysis_details: db_details["current_track_analysis"] = current_track_analysis_details
            if final_summary_details: # Merge final summary if provided
                db_details.update(final_summary_details)

            # Manage logs
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            if task_state == TASK_STATUS_SUCCESS:
                db_details["log"] = [f"Task completed successfully. Final status: {message}"]
                # Ensure other summary fields from final_summary_details are in db_details
            elif task_state == TASK_STATUS_FAILURE or task_state == TASK_STATUS_REVOKED:
                current_task_logs.append(log_entry) # Add final error/revoked message
                db_details["log"] = current_task_logs # Keep all logs for failure/revocation
                if "error" not in db_details and task_state == TASK_STATUS_FAILURE : db_details["error"] = message # Ensure error message is captured
            else: # PROGRESS or STARTED
                current_task_logs.append(log_entry)
                db_details["log"] = current_task_logs

            # Details for RQ job.meta (leaner)
            meta_details = {
                "album_name": album_name,
                "current_track": current_track_name,
                "status_message": message # Main status message for meta
            }
            if current_track_analysis_details: meta_details["current_track_analysis_short"] = {k:v for k,v in current_track_analysis_details.items() if k in ['tempo', 'key']}

            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.meta['details'] = meta_details
                current_job.save_meta()

            save_task_status(current_task_id, "album_analysis", task_state, parent_task_id=parent_task_id, sub_type_identifier=album_id, progress=progress, details=db_details)

        try:
            log_and_update_album_task(f"Fetching tracks for album: {album_name}", 5)
            tracks = mediaserver.get_tracks_from_album(album_id)
            if not tracks:
                log_and_update_album_task(f"No tracks found for album: {album_name}", 100, task_state='SUCCESS')
                return {"status": "SUCCESS", "message": f"No tracks in album {album_name}", "tracks_analyzed": 0}

            # --- OPTIMIZATION: Batch check for existing tracks ---
            all_track_ids_in_album = [t['Id'] for t in tracks]
            
            def get_existing_track_ids_in_batch(track_ids):
                """Queries the DB for a batch of track IDs to see which ones are fully analyzed."""
                if not track_ids:
                    return set()
                conn = get_db()
                cur = conn.cursor()
                try:
                    query = """
                        SELECT s.item_id
                        FROM score s
                        JOIN embedding e ON s.item_id = e.item_id
                        WHERE s.item_id IN %s
                          AND s.other_features IS NOT NULL AND s.other_features != ''
                          AND s.energy IS NOT NULL
                          AND s.mood_vector IS NOT NULL AND s.mood_vector != ''
                          AND s.tempo IS NOT NULL
                    """
                    cur.execute(query, (tuple(track_ids),))
                    existing_ids = {row[0] for row in cur.fetchall()}
                    return existing_ids
                finally:
                    cur.close()

            existing_track_ids_set = get_existing_track_ids_in_batch(all_track_ids_in_album)
            logger.info(f"Album '{album_name}': Found {len(existing_track_ids_set)} of {len(all_track_ids_in_album)} tracks already analyzed.")
            # --- END OPTIMIZATION ---

            total_tracks_in_album = len(tracks)
            for idx, item in enumerate(tracks, 1):
                # === Cooperative Cancellation Check for Album Task ===
                if current_job: # Check if running as an RQ job
                    with app.app_context(): # Ensure DB access for status check
                        album_task_db_info = get_task_info_from_db(current_task_id)
                        parent_task_db_info = get_task_info_from_db(parent_task_id) if parent_task_id else None

                        is_self_revoked = album_task_db_info and album_task_db_info.get('status') == 'REVOKED'
                        # Check if parent is in a terminal failure or revoked state
                        is_parent_failed_or_revoked = parent_task_db_info and parent_task_db_info.get('status') in [TASK_STATUS_REVOKED, TASK_STATUS_FAILURE]

                        if is_self_revoked or is_parent_failed_or_revoked:
                            parent_status_for_reason = parent_task_db_info.get('status') if parent_task_db_info else "N/A"
                            revocation_reason = "self was REVOKED" if is_self_revoked else f"parent task {parent_task_id} status is {parent_status_for_reason}"
                            # Ensure path is cleaned up if it exists at this point
                            temp_file_to_clean = locals().get('path') # Get 'path' if defined in this scope
                            if temp_file_to_clean and os.path.exists(temp_file_to_clean):
                                try: os.remove(temp_file_to_clean)
                                except Exception as e_cleanup: logger.warning("Failed to clean up %s during revocation: %s", temp_file_to_clean, e_cleanup)
                            log_and_update_album_task(f"🛑 Album analysis task {current_task_id} for '{album_name}' stopping because {revocation_reason}.", current_progress_val, task_state=TASK_STATUS_REVOKED)
                            return {"status": "REVOKED", "message": f"Album analysis for '{album_name}' stopped because {revocation_reason}."}
                # === End Cooperative Cancellation Check ===

                track_name_full = f"{item['Name']} by {item.get('AlbumArtist', 'Unknown')}"
                current_progress_val = 10 + int(85 * (idx / float(total_tracks_in_album))) if total_tracks_in_album > 0 else 10
                # Clear previous track's analysis details when starting a new track
                log_and_update_album_task(
                    f"Analyzing track: {track_name_full} ({idx}/{total_tracks_in_album})",
                    current_progress_val,
                    current_track_name=track_name_full,
                    current_track_analysis_details=None)

                if item['Id'] in existing_track_ids_set:
                    log_and_update_album_task(f"Skipping '{track_name_full}' (already fully analyzed)", current_progress_val, current_track_name=track_name_full)
                    tracks_skipped_count +=1
                    continue

                path = mediaserver.download_track(TEMP_DIR, item)
                log_and_update_album_task(f"Download attempt for '{track_name_full}': {'Success' if path else 'Failed'}", current_progress_val, current_track_name=track_name_full)
                if not path:
                    log_and_update_album_task(f"Failed to download '{track_name_full}'. Skipping.", current_progress_val, current_track_name=track_name_full)
                    continue

                try:
                    # analyze_track now returns two values: results dict and the embedding
                    analysis_results, track_embedding = analyze_track(path, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, MOOD_LABELS)
                    
                    # NEW: Check for the failure case from analyze_track
                    if analysis_results is None:
                        # The error has already been logged in analyze_track.
                        # We can add another log here for the album task's context.
                        log_and_update_album_task(f"Skipping '{track_name_full}' due to an audio loading error (e.g., multi-channel).", current_progress_val, current_track_name=track_name_full)
                        tracks_skipped_count += 1
                        continue # Move to the next track

                    tempo = analysis_results.get("tempo", 0.0)
                    key = analysis_results.get("key", "")
                    scale = analysis_results.get("scale", "")
                    all_moods_predicted = analysis_results.get("moods", {}) # All moods initially
                    # Include new predictions directly
                    danceable = analysis_results.get("danceable", 0.0)
                    energy = analysis_results.get("energy", 0.0) # Get energy
                    aggressive = analysis_results.get("aggressive", 0.0)
                    happy = analysis_results.get("happy", 0.0)
                    party = analysis_results.get("party", 0.0)
                    relaxed = analysis_results.get("relaxed", 0.0)
                    sad = analysis_results.get("sad", 0.0)

                    # Select only the top_n_moods for saving and detailed logging
                    sorted_moods_for_saving = sorted(all_moods_predicted.items(), key=lambda item: item[1], reverse=True)
                    top_moods_to_save_dict = dict(sorted_moods_for_saving[:top_n_moods])

                    # Prepare for database storage
                    other_features_str = f"danceable:{danceable:.2f},aggressive:{aggressive:.2f},happy:{happy:.2f},party:{party:.2f},relaxed:{relaxed:.2f},sad:{sad:.2f}" # Ensure no spaces for easier parsing
                    current_track_details_for_api = {
                        "name": track_name_full,
                        "tempo": round(tempo, 2),
                        "key": key,
                        "scale": scale,
                        "moods": {k: round(v, 2) for k, v in all_moods_predicted.items()}
                    }
                    # Save only the top_n_moods
                    # IMPORTANT: Ensure save_track_analysis and save_track_embedding are defined
                    # and accessible here, likely from your app module.
                    save_track_analysis(item['Id'], item['Name'], item.get('AlbumArtist', 'Unknown'), tempo, key, scale, top_moods_to_save_dict, energy=energy, other_features=other_features_str)
                    # IMPORTANT: Ensure save_track_embedding now saves the embedding as a binary blob
                    # as shown in the new annoy_manager.py example.
                    save_track_embedding(item['Id'], track_embedding)
                    tracks_analyzed_count += 1

                    # For logging, we already have the top_moods_to_save_dict
                    # If top_n_moods passed to task is different from a global TOP_N_MOODS for logging, adjust here.
                    # Assuming top_n_moods arg is the definitive count for both saving and logging summary.
                    top_n_moods_for_log = list(top_moods_to_save_dict.items()) # Already top N
                    mood_details_str_log = ', '.join(f'{k}:{v:.2f}' for k,v in top_n_moods_for_log)

                    # Prepare for logging: New features
                    other_features_log_str = f"Energy: {energy:.2f}, Danceable: {danceable:.2f}, Aggressive: {aggressive:.2f}, Happy: {happy:.2f}, Party: {party:.2f}, Relaxed: {relaxed:.2f}, Sad: {sad:.2f}" # Include energy in log

                    analysis_summary_msg = f"Tempo: {tempo:.2f}, Key: {key} {scale}."
                    analysis_summary_msg += f" Top {top_n_moods} Moods: {mood_details_str_log}. Other Features: {other_features_log_str}"
                    log_and_update_album_task(
                        f"Analyzed '{track_name_full}'. {analysis_summary_msg}",
                        current_progress_val,
                        current_track_name=track_name_full,
                        current_track_analysis_details=current_track_details_for_api)
                except Exception as e_analyze:
                    logger.error("Error analyzing '%s': %s", track_name_full, e_analyze, exc_info=True)
                    log_and_update_album_task(f"Error analyzing '{track_name_full}': {e_analyze}", current_progress_val, current_track_name=track_name_full, task_state=TASK_STATUS_FAILURE)
                finally:
                    if path and os.path.exists(path):
                        try: os.remove(path)
                        except Exception as cleanup_e: logger.warning("Failed to clean up temp file %s: %s", path, cleanup_e)
            success_summary = {
                "tracks_analyzed": tracks_analyzed_count,
                "tracks_skipped": tracks_skipped_count,
                "total_tracks_in_album": total_tracks_in_album,
                "message": f"Album '{album_name}' analysis complete."
            }
            log_and_update_album_task(f"Album '{album_name}' analysis complete. Analyzed {tracks_analyzed_count}, Skipped {tracks_skipped_count} (out of {total_tracks_in_album} total).",
                                      100, task_state=TASK_STATUS_SUCCESS, final_summary_details=success_summary)
            return {"status": "SUCCESS", "message": f"Album '{album_name}' analysis complete.", "tracks_analyzed": tracks_analyzed_count, "tracks_skipped": tracks_skipped_count, "total_tracks": total_tracks_in_album}
        except Exception as e:
            # Log the full traceback on the server for debugging.
            logger.critical("Album analysis %s failed: %s", album_id, e, exc_info=True)
            # Save a generic error to the DB, without the stack trace.
            failure_details = {"error": "An unexpected error occurred during album analysis.", "album_name": album_name}
            log_and_update_album_task(
                f"Failed to analyze album '{album_name}': An unexpected error occurred.",
                current_progress_val,
                task_state=TASK_STATUS_FAILURE,
                final_summary_details=failure_details
            )
            raise

def run_analysis_task(num_recent_albums, top_n_moods):
    """Main RQ task to orchestrate the analysis of multiple albums."""
    # --- LOCAL IMPORTS TO PREVENT CIRCULAR DEPENDENCIES ---
    from app import (app, redis_conn, get_db, save_task_status, get_task_info_from_db,
                     TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED,
                     TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
                     get_child_tasks_from_db, rq_queue_default, JobStatus)
    # --- END LOCAL IMPORTS ---

    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        # --- STATE RECOVERY ---
        task_info = get_task_info_from_db(current_task_id)
        if task_info and task_info.get('status') in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]:
            logger.info(f"Main analysis task {current_task_id} is already in a terminal state ('{task_info.get('status')}'). Skipping execution.")
            # ... (rest of idempotency check)
            return {"status": task_info.get('status'), "message": f"Task already in terminal state '{task_info.get('status')}'."}
        
        try:
            task_details_json = task_info.get('details') if task_info else "{}"
            parsed_details = json.loads(task_details_json)
            checked_album_ids = set(parsed_details.get('checked_album_ids', []))
        except (json.JSONDecodeError, TypeError):
            checked_album_ids = set()
        # --- END STATE RECOVERY ---

        initial_details = {"message": "Fetching albums...", "log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Main analysis task started."]}
        save_task_status(current_task_id, "main_analysis", TASK_STATUS_STARTED, progress=0, details=initial_details)
        current_progress = 0
        current_task_logs = initial_details["log"] # Initialize with the first log

        def log_and_update_main_analysis(message, progress, details_extra=None, task_state=TASK_STATUS_PROGRESS):
            nonlocal current_progress
            nonlocal current_task_logs
            current_progress = progress
            logger.info("[MainAnalysisTask-%s] %s", current_task_id, message)

            current_details_for_db = dict(details_extra) if details_extra else {}
            current_details_for_db["status_message"] = message

            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            if task_state == TASK_STATUS_SUCCESS:
                current_details_for_db["log"] = [f"Task completed successfully. Final status: {message}"]
                current_details_for_db.pop('log_storage_info', None)
            elif task_state == TASK_STATUS_FAILURE or task_state == TASK_STATUS_REVOKED:
                current_task_logs.append(log_entry)
                current_details_for_db["log"] = current_task_logs
                if "error" not in current_details_for_db and task_state == TASK_STATUS_FAILURE: current_details_for_db["error"] = message
            else: # PROGRESS or STARTED
                current_task_logs.append(log_entry)
                current_details_for_db["log"] = current_task_logs

            meta_details = {"status_message": message}
            if details_extra:
                for key in ["albums_completed", "total_albums", "successful_albums", "failed_albums", "total_tracks_analyzed", "albums_found", "album_task_ids", "albums_skipped", "albums_to_process", "checked_album_ids"]:
                    if key in details_extra:
                        meta_details[key] = details_extra[key]

            if current_job:
                current_job.meta['progress'] = progress
                current_job.meta['status_message'] = message
                current_job.meta['details'] = meta_details
                current_job.save_meta()
            save_task_status(current_task_id, "main_analysis", task_state, progress=progress, details=current_details_for_db)

        try:
            log_and_update_main_analysis("🚀 Starting main analysis process...", 0)
            clean_temp(TEMP_DIR)
            all_albums = mediaserver.get_recent_albums(num_recent_albums)
            if not all_albums:
                log_and_update_main_analysis("⚠️ No new albums to analyze.", 100, {"albums_found": 0}, task_state=TASK_STATUS_SUCCESS)
                return {"status": "SUCCESS", "message": "No new albums to analyze.", "albums_processed": 0}

            total_albums_to_check = len(all_albums)
            log_and_update_main_analysis(f"Found {total_albums_to_check} albums to check.", 2)
            
            active_jobs_map = {}
            launched_jobs = []
            albums_skipped_count = 0
            albums_launched_count = 0
            albums_completed_count = 0
            
            # --- Incremental Index Rebuild Configuration ---
            last_rebuild_at_album_count = 0


            def get_existing_track_ids(track_ids):
                if not track_ids: return set()
                conn = get_db()
                cur = conn.cursor()
                try:
                    query = """
                        SELECT s.item_id FROM score s JOIN embedding e ON s.item_id = e.item_id
                        WHERE s.item_id IN %s AND s.other_features IS NOT NULL AND s.other_features != ''
                          AND s.energy IS NOT NULL AND s.mood_vector IS NOT NULL AND s.mood_vector != ''
                          AND s.tempo IS NOT NULL
                    """
                    cur.execute(query, (tuple(track_ids),))
                    return {row[0] for row in cur.fetchall()}
                finally:
                    cur.close()

            def monitor_and_clear_jobs():
                nonlocal albums_completed_count
                nonlocal last_rebuild_at_album_count

                completed_in_cycle = []
                # Iterate over a copy of the keys to allow modification during the loop
                for job_id in list(active_jobs_map.keys()):
                    try:
                        # Fetch the job fresh from Redis to get the most up-to-date status
                        job = Job.fetch(job_id, connection=redis_conn)
                        if job.is_finished or job.is_failed or job.is_canceled:
                            completed_in_cycle.append(job_id)
                    except NoSuchJobError: # This is the critical case to handle robustly
                        logger.warning(f"Sub-task job {job_id} not found in Redis. Checking database for final status...")
                        # The main task runs in an app_context, so we can safely call DB functions
                        db_status_info = get_task_info_from_db(job_id)
                        if db_status_info and db_status_info.get('status') in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]:
                            logger.info(f"  DB confirms task {job_id} is in a terminal state: {db_status_info.get('status')}. Marking as complete.")
                            completed_in_cycle.append(job_id)
                        else:
                            # This indicates a lost job. It's gone from Redis but not marked as complete in the DB.
                            # This can happen if a worker dies without moving the job to the failed queue.
                            # We will mark it as FAILED to be safe and prevent the main task from waiting forever.
                            db_status = db_status_info.get('status') if db_status_info else 'NOT_IN_DB'
                            error_msg = f"Job lost from Redis queue while in a non-terminal state (DB status: {db_status})."
                            logger.error(f"  CRITICAL: Job {job_id} is missing from Redis but its DB status is '{db_status}'. Marking as failed.")
                            save_task_status(job_id, "album_analysis", TASK_STATUS_FAILURE, parent_task_id=current_task_id, progress=100, details={"error": error_msg})
                            completed_in_cycle.append(job_id) # Treat as "completed" to unblock the main task.
                
                for job_id in completed_in_cycle:
                    if job_id in active_jobs_map:
                        del active_jobs_map[job_id]
                        albums_completed_count += 1
                
                # --- Incremental Index Rebuild Logic ---
                if albums_completed_count > last_rebuild_at_album_count and \
                   (albums_completed_count - last_rebuild_at_album_count) >= REBUILD_INDEX_BATCH_SIZE:
                    
                    progress_for_rebuild = 5 + int(85 * ((albums_skipped_count + albums_completed_count) / float(total_albums_to_check)))
                    log_and_update_main_analysis(
                        f"Batch of {albums_completed_count - last_rebuild_at_album_count} albums complete. Rebuilding index...",
                        progress_for_rebuild
                    )
                    
                    try:
                        db_conn_rebuild = get_db()
                        build_and_store_voyager_index(db_conn_rebuild)
                        redis_conn.publish('index-updates', 'reload')
                        logger.info(f"Incremental index rebuild complete after {albums_completed_count} albums.")
                        # IMPORTANT: Update the counter to prevent immediate re-rebuilds
                        last_rebuild_at_album_count = albums_completed_count
                    except Exception as e_rebuild:
                        logger.error(f"Failed to perform incremental index rebuild: {e_rebuild}", exc_info=True)
                        # We don't stop the main task, just log the error and continue.

            # Main Dispatch and Monitoring Loop
            for idx, album in enumerate(all_albums):
                if album['Id'] in checked_album_ids:
                    albums_skipped_count += 1
                    logger.info(f"Skipping album '{album['Name']}' (already checked in a previous run).")
                    continue

                # --- Concurrency Management ---
                while len(active_jobs_map) >= MAX_QUEUED_ANALYSIS_JOBS:
                    monitor_and_clear_jobs()
                    progress = 5 + int(85 * ((albums_skipped_count + albums_completed_count) / float(total_albums_to_check)))
                    status_message = f"Launched: {albums_launched_count}. Completed: {albums_completed_count}/{albums_launched_count}. Active: {len(active_jobs_map)}. Skipped: {albums_skipped_count}/{total_albums_to_check}. (Throttling)"
                    log_and_update_main_analysis(status_message, progress, details_extra={"checked_album_ids": list(checked_album_ids)})
                    time.sleep(5)

                # --- Check a single album ---
                tracks = mediaserver.get_tracks_from_album(album['Id'])
                if not tracks:
                    albums_skipped_count += 1
                    logger.info(f"Skipping album '{album['Name']}' (no tracks found).")
                    checked_album_ids.add(album['Id'])
                    continue

                track_ids = [t['Id'] for t in tracks]
                existing_ids_set = get_existing_track_ids(track_ids)

                if len(existing_ids_set) >= len(track_ids):
                    albums_skipped_count += 1
                    logger.info(f"Skipping album '{album['Name']}' (all tracks already analyzed).")
                    checked_album_ids.add(album['Id'])
                    continue
                
                # --- Enqueue album for processing ---
                sub_task_id = str(uuid.uuid4())
                job = rq_queue_default.enqueue(
                    'tasks.analysis.analyze_album_task',
                    args=(album['Id'], album['Name'], top_n_moods, current_task_id),
                    job_id=sub_task_id,
                    description=f"Analyzing album: {album['Name']}",
                    job_timeout=-1,
                    retry=Retry(max=3),
                    meta={'parent_task_id': current_task_id}
                )
                active_jobs_map[job.id] = job
                launched_jobs.append(job)
                albums_launched_count += 1
                checked_album_ids.add(album['Id'])
                
                progress = 5 + int(idx / float(total_albums_to_check))
                status_message = f"Launched: {albums_launched_count}. Completed: {albums_completed_count}/{albums_launched_count}. Active: {len(active_jobs_map)}. Skipped: {albums_skipped_count}/{total_albums_to_check}."
                log_and_update_main_analysis(
                    status_message,
                    progress,
                    details_extra={"albums_to_process": albums_launched_count, "albums_skipped": albums_skipped_count, "checked_album_ids": list(checked_album_ids)}
                )

            # --- Final Monitoring Loop ---
            while active_jobs_map:
                monitor_and_clear_jobs()
                progress = 5 + int(85 * ((albums_skipped_count + albums_completed_count) / float(total_albums_to_check)))
                status_message = f"Launched: {albums_launched_count}. Completed: {albums_completed_count}/{albums_launched_count}. Active: {len(active_jobs_map)}. Skipped: {albums_skipped_count}/{total_albums_to_check}. (Finalizing)"
                log_and_update_main_analysis(status_message, progress, details_extra={"checked_album_ids": list(checked_album_ids)})
                time.sleep(5)

            # Final Summary
            successful_albums = 0
            failed_albums = 0
            total_tracks_analyzed_all_albums = 0
            total_tracks_skipped_in_albums = 0
            for job_instance in launched_jobs:
                try:
                    job_instance.refresh()
                    if job_instance.is_finished and isinstance(job_instance.result, dict):
                        successful_albums += 1
                        total_tracks_analyzed_all_albums += job_instance.result.get("tracks_analyzed", 0)
                        total_tracks_skipped_in_albums += job_instance.result.get("tracks_skipped", 0)
                    else:
                        failed_albums += 1
                except Exception:
                    failed_albums += 1
            
            # --- Voyager Index Creation ---
            # Perform a final rebuild to include any remaining albums that didn't form a full batch.
            log_and_update_main_analysis("Performing final index rebuild to include all tracks...", 95)
            db_conn = get_db()
            build_and_store_voyager_index(db_conn)
            
            final_message = f"Main analysis complete. Found {total_albums_to_check} albums, skipped {albums_skipped_count}. Of the {albums_launched_count} launched, {successful_albums} succeeded, {failed_albums} failed. Total tracks analyzed: {total_tracks_analyzed_all_albums}. Voyager index created."

            log_and_update_main_analysis("Publishing index-reload notification to Redis...", 98)
            try:
                # Use Redis Pub/Sub to notify the web server to reload the index.
                # This is more robust than an HTTP call in a containerized environment.
                redis_conn.publish('index-updates', 'reload')
                logger.info("Successfully published 'reload' message to 'index-updates' channel.")
            except Exception as e:
                logger.warning(f"Could not publish index-reload notification to Redis: {e}")
                final_message += " (Warning: Could not notify web server for live reload.)"

            if total_tracks_skipped_in_albums > 0:
                final_message += f" An additional {total_tracks_skipped_in_albums} tracks were skipped within these albums due to errors (e.g., multi-channel audio)."

            final_details = {"successful_albums": successful_albums, "failed_albums": failed_albums, "total_tracks_analyzed": total_tracks_analyzed_all_albums, "total_tracks_skipped_in_albums": total_tracks_skipped_in_albums}
            log_and_update_main_analysis(final_message, 100, details_extra=final_details, task_state=TASK_STATUS_SUCCESS)
            clean_temp(TEMP_DIR)
            return {"status": "SUCCESS", "message": final_message}

        except Exception as e:
            # Log the full traceback on the server for debugging.
            logger.critical("FATAL ERROR: Analysis failed: %s", e, exc_info=True)
            with app.app_context():
                # Save a generic error to the DB, without the stack trace.
                log_and_update_main_analysis(
                    "❌ Main analysis failed due to an unexpected error.",
                    current_progress,
                    details_extra={"error_message": "An unexpected error occurred. Check server logs for details."},
                    task_state=TASK_STATUS_FAILURE
                )
            raise
