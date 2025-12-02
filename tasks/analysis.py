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
from pydub import AudioSegment
from tempfile import NamedTemporaryFile

import librosa
import onnx
import onnxruntime as ort

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# RQ import
from rq import get_current_job, Retry
from rq.job import Job
from rq.exceptions import NoSuchJobError

# Import configuration from the user's provided config file
from config import (
    TEMP_DIR, MAX_DISTANCE, MAX_SONGS_PER_CLUSTER, MAX_SONGS_PER_ARTIST,
    GMM_COVARIANCE_TYPE, MOOD_LABELS, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH, ENERGY_MIN, ENERGY_MAX,
    TEMPO_MIN_BPM, TEMPO_MAX_BPM, JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, EMBY_URL, EMBY_USER_ID, EMBY_TOKEN, OTHER_FEATURE_LABELS, REDIS_URL, DATABASE_URL,
    OLLAMA_SERVER_URL, OLLAMA_MODEL_NAME, AI_MODEL_PROVIDER, GEMINI_API_KEY, GEMINI_MODEL_NAME,
    DANCEABILITY_MODEL_PATH, AGGRESSIVE_MODEL_PATH, HAPPY_MODEL_PATH, PARTY_MODEL_PATH, RELAXED_MODEL_PATH, SAD_MODEL_PATH,
    SCORE_WEIGHT_SILHOUETTE, SCORE_WEIGHT_DAVIES_BOULDIN, SCORE_WEIGHT_CALINSKI_HARABASZ,
    SCORE_WEIGHT_DIVERSITY, SCORE_WEIGHT_PURITY, SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY, SCORE_WEIGHT_OTHER_FEATURE_PURITY,
    MUTATION_KMEANS_COORD_FRACTION, MUTATION_INT_ABS_DELTA, MUTATION_FLOAT_ABS_DELTA,
    TOP_N_ELITES, EXPLOITATION_START_FRACTION, EXPLOITATION_PROBABILITY_CONFIG, TOP_N_MOODS, TOP_N_OTHER_FEATURES,
    STRATIFIED_GENRES, MIN_SONGS_PER_GENRE_FOR_STRATIFICATION, SAMPLING_PERCENTAGE_CHANGE_PER_RUN, ITERATIONS_PER_BATCH_JOB, MAX_CONCURRENT_BATCH_JOBS, REBUILD_INDEX_BATCH_SIZE,
    MAX_QUEUED_ANALYSIS_JOBS,
    TOP_K_MOODS_FOR_PURITY_CALCULATION, LN_MOOD_DIVERSITY_STATS, LN_MOOD_PURITY_STATS,
    LN_OTHER_FEATURES_DIVERSITY_STATS, LN_OTHER_FEATURES_PURITY_STATS,
    STRATIFIED_SAMPLING_TARGET_PERCENTILE,
    OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY as CONFIG_OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY,
    AUDIO_LOAD_TIMEOUT # Add this to your config.py, e.g., AUDIO_LOAD_TIMEOUT = 600 (for a 10-minute timeout)
)


# Import other project modules
from ai import get_ai_playlist_name, creative_prompt_template
from .commons import score_vector
# MODIFIED: Import from voyager_manager instead of annoy_manager
from .voyager_manager import build_and_store_voyager_index
# Import artist GMM manager for artist similarity index
from .artist_gmm_manager import build_and_store_artist_index
# MODIFIED: The functions from mediaserver no longer need server-specific parameters.
from .mediaserver import get_recent_albums, get_tracks_from_album, download_track


from psycopg2 import OperationalError
from redis.exceptions import TimeoutError as RedisTimeoutError # Import with an alias
logger = logging.getLogger(__name__)

# --- Tensor Name Definitions ---
# Based on a full review of all error logs and the Essentia examples,
# this is the definitive mapping.
DEFINED_TENSOR_NAMES = {
    # Takes spectrograms, outputs embeddings
    'embedding': {
        'input': 'model/Placeholder:0',
        'output': 'model/dense/BiasAdd:0'
    },
    # Takes embeddings, outputs mood predictions
    'prediction': {
        'input': 'serving_default_model_Placeholder:0',
        'output': 'PartitionedCall:0'
    },
    # Takes a single aggregated embedding, outputs a binary classification
    'danceable': {
        'input': 'model/Placeholder:0',
        'output': 'model/Softmax:0'
    },
    'aggressive': {
        'input': 'model/Placeholder:0',
        'output': 'model/Softmax:0'
    },
    'happy': {
        'input': 'model/Placeholder:0',
        'output': 'model/Softmax:0'
    },
    'party': {
        'input': 'model/Placeholder:0',
        'output': 'model/Softmax:0'
    },
    'relaxed': {
        'input': 'model/Placeholder:0',
        'output': 'model/Softmax:0'
    },
    'sad': {
        'input': 'model/Placeholder:0',
        'output': 'model/Softmax:0'
    }
}

# --- Class Index Mapping ---
# Based on confirmed metadata from the user.
CLASS_INDEX_MAP = {
    "aggressive": 0,
    "happy": 0,
    "relaxed": 1,
    "sad": 1,
    "danceable": 0,
    "party": 1,
}


# --- Utility Functions ---
def clean_temp(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.warning(f"Could not remove {file_path} from {temp_dir}: {e}")

# --- Core Analysis Functions ---

def _find_onnx_name(candidate_name, names):
    """Try several heuristics to match a TF-style tensor name to an ONNX input/output name."""
    if candidate_name in names:
        return candidate_name
    # strip trailing :0
    stripped = candidate_name.split(':')[0]
    if stripped in names:
        return stripped
    # try last part after '/'
    last = stripped.split('/')[-1]
    if last in names:
        return last
    # try replacing '/' with '_'
    alt = stripped.replace('/', '_')
    if alt in names:
        return alt
    # fallback: return first name
    return names[0] if names else None

def run_inference(onnx_session, feed_dict, output_tensor_name=None):
    """Run inference on an ONNX Runtime session.

    onnx_session: ort.InferenceSession
    feed_dict: dict mapping possible tensor names to numpy arrays
    output_tensor_name: optional expected output name (TF-style). If None, use first output.
    """
    # Build input name -> value map for ONNX
    input_meta = onnx_session.get_inputs()
    input_names = [i.name for i in input_meta]
    mapped = {}
    logger.debug(f"ONNX session inputs: {input_names}")
    for key, val in feed_dict.items():
        onnx_name = _find_onnx_name(key, input_names)
        if onnx_name is None:
            logger.error(f"Could not map input name '{key}' to any ONNX input names: {input_names}")
            return None
        mapped[onnx_name] = val

    # Determine outputs
    output_meta = onnx_session.get_outputs()
    output_names = [o.name for o in output_meta]
    logger.debug(f"ONNX session outputs: {output_names}")
    if output_tensor_name:
        onnx_output_name = _find_onnx_name(output_tensor_name, output_names)
    else:
        onnx_output_name = output_names[0] if output_names else None

    if onnx_output_name is None:
        logger.error("No ONNX output name available to run inference.")
        return None

    # Run and return numpy array
    result = onnx_session.run([onnx_output_name], mapped)
    # onnxruntime returns a list of outputs in the same order
    return result[0] if isinstance(result, list) and len(result) > 0 else result

def sigmoid(x):
    """Numerically stable sigmoid function."""
    return 1 / (1 + np.exp(-x))

def robust_load_audio_with_fallback(file_path, target_sr=16000):
    """
    Attempts to load an audio file directly with Librosa. If it fails or
    results in an empty audio signal, it falls back to a more robust method
    using pydub (and ffmpeg) to convert the file to a temporary WAV before loading.
    """
    audio = None
    sr = None
    
    # --- Primary Method: Direct Librosa Load ---
    try:
        # Use kaiser_fast for speed
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True, duration=AUDIO_LOAD_TIMEOUT, res_type='kaiser_fast')
        
        # An empty audio signal is a failure condition, so we raise an error to trigger the fallback.
        if audio is None or audio.size == 0:
            raise ValueError("Librosa returned an empty audio signal.")
            
        logger.debug(f"Successfully loaded {os.path.basename(file_path)} directly with Librosa.")
        return audio, sr

    except Exception as e_direct_load:
        logger.warning(f"Direct librosa load failed for {os.path.basename(file_path)}: {e_direct_load}. Attempting fallback conversion.")

    # --- Fallback Method: Convert to WAV with pydub ---
    temp_wav_path = None
    try:
        # Check the audio content with pydub before converting
        # Use more robust parameters for problematic codecs
        audio_segment = AudioSegment.from_file(
            file_path,
            # Add parameters to help with codec detection issues
            parameters=[
                "-analyzeduration", "10M",  # Increase analysis duration
                "-probesize", "10M",        # Increase probe size  
                "-ignore_unknown",          # Ignore unknown streams
                "-err_detect", "ignore_err", # Ignore decode errors
                "-ac", "2"                  # Force downmix to stereo to handle multichannel files
            ]
        )
        if len(audio_segment) == 0:
            logger.error(f"Pydub loaded a zero-duration audio segment from {os.path.basename(file_path)}. The file is likely corrupt or empty.")
            return None, None

        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
            temp_wav_path = temp_wav_file.name
        
        # --- MEMORY OPTIMIZATION FOR LARGE FILES ---
        # Resample and convert to mono during export to create a much smaller temp file.
        # This is critical for handling very large source files without running out of memory.
        logger.info(f"Fallback: Pre-processing {os.path.basename(file_path)} to a smaller WAV for safe loading...")
        processed_segment = audio_segment.set_frame_rate(target_sr).set_channels(1)
        # Use more robust export parameters
        processed_segment.export(
            temp_wav_path, 
            format="wav",
            parameters=[
                "-codec:a", "pcm_s16le",  # Fix the typo: was pcm_s0le, should be pcm_s16le
                "-ar", str(target_sr),    # Set sample rate explicitly
                "-ac", "1"                # Set mono explicitly
            ]
        )
        
        logger.info(f"Fallback: Converted {os.path.basename(file_path)} to temporary WAV for robust loading.")
        
        # Load the safe, downsampled WAV file
        # Load the safe, downsampled WAV file
        # Use kaiser_fast for speed
        audio, sr = librosa.load(temp_wav_path, sr=target_sr, mono=True, duration=AUDIO_LOAD_TIMEOUT, res_type='kaiser_fast')
        
        # Final check on the fallback's output for silence or emptiness
        if audio is None or audio.size == 0 or not np.any(audio):
            logger.error(f"Fallback method also resulted in an empty or silent audio signal for {os.path.basename(file_path)}.")
            return None, None
            
        return audio, sr

    except Exception as e_fallback:
        logger.error(f"Fallback loading method also failed for {os.path.basename(file_path)}: {e_fallback}")
        return None, None
    finally:
        # Clean up the temporary WAV file if it was created
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

# --- Refactored Analysis Functions for Batch Processing ---

def process_audio_and_embedding(file_path, embedding_session):
    """
    Loads audio, computes CPU features (tempo, key, energy), creates spectrogram,
    and generates the embedding using the provided session.
    Returns: (cpu_features_dict, embedding_vector)
    """
    # --- 1. Load Audio and Compute Basic Features ---
    audio, sr = robust_load_audio_with_fallback(file_path, target_sr=16000)
    
    if audio is None or not np.any(audio) or audio.size == 0:
        logger.warning(f"Could not load a valid audio signal for {os.path.basename(file_path)}. Skipping.")
        return None, None

    # CPU-bound features
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    average_energy = np.mean(librosa.feature.rms(y=audio))
    
    # Key/Scale detection
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key_vals = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    
    major_correlations = np.array([np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1] for i in range(12)])
    minor_correlations = np.array([np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1] for i in range(12)])

    major_key_idx = np.argmax(major_correlations)
    minor_key_idx = np.argmax(minor_correlations)

    if major_correlations[major_key_idx] > minor_correlations[minor_key_idx]:
        musical_key = key_vals[major_key_idx]
        scale = 'major'
    else:
        musical_key = key_vals[minor_key_idx]
        scale = 'minor'

    cpu_features = {
        "tempo": float(tempo),
        "energy": float(average_energy),
        "key": musical_key,
        "scale": scale
    }

    # --- 2. Prepare Spectrograms --- 
    try:
        n_mels, hop_length, n_fft, frame_size = 96, 256, 512, 187
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window='hann', center=False, power=2.0, norm='slaney', htk=False)
        log_mel_spec = np.log10(1 + 10000 * mel_spec)
        spec_patches = [log_mel_spec[:, i:i+frame_size] for i in range(0, log_mel_spec.shape[1] - frame_size + 1, frame_size)]
        
        if not spec_patches:
            logger.warning(f"Track too short for spectrogram: {os.path.basename(file_path)}")
            return None, None
        
        transposed_patches = np.array(spec_patches).transpose(0, 2, 1)
        final_patches = transposed_patches.astype(np.float32)

    except Exception as e:
        logger.error(f"Spectrogram creation failed for {os.path.basename(file_path)}: {e}", exc_info=True)
        return None, None

    # --- 3. Run Embedding Model ---
    try:
        embedding_feed_dict = {DEFINED_TENSOR_NAMES['embedding']['input']: final_patches}
        embeddings_per_patch = run_inference(embedding_session, embedding_feed_dict, DEFINED_TENSOR_NAMES['embedding']['output'])
        
        # Average patches to get single embedding vector
        processed_embedding = np.mean(embeddings_per_patch, axis=0)
        
        # Return the raw patch embeddings too if needed for other models? 
        # Actually, other models take 'embeddings_per_patch' as input.
        # Wait, the original code passed `embeddings_per_patch` to prediction models.
        # So we must return `embeddings_per_patch` to be used by subsequent phases.
        # But `processed_embedding` is what gets saved to DB.
        
        return cpu_features, embeddings_per_patch

    except Exception as e:
        logger.error(f"Embedding inference failed for {os.path.basename(file_path)}: {e}", exc_info=True)
        return None, None

def predict_moods(prediction_session, embeddings_per_patch, mood_labels_list):
    try:
        prediction_feed_dict = {DEFINED_TENSOR_NAMES['prediction']['input']: embeddings_per_patch}
        mood_logits = run_inference(prediction_session, prediction_feed_dict, DEFINED_TENSOR_NAMES['prediction']['output'])
        averaged_logits = np.mean(mood_logits, axis=0)
        final_mood_predictions = sigmoid(averaged_logits)
        return {label: float(score) for label, score in zip(mood_labels_list, final_mood_predictions)}
    except Exception as e:
        logger.error(f"Mood prediction failed: {e}")
        return {}

def predict_other_feature(session, embeddings_per_patch, feature_key):
    try:
        feed_dict = {DEFINED_TENSOR_NAMES[feature_key]['input']: embeddings_per_patch}
        probabilities_per_patch = run_inference(session, feed_dict, DEFINED_TENSOR_NAMES[feature_key]['output'])

        if probabilities_per_patch is None:
            return 0.0
        
        if isinstance(probabilities_per_patch, np.ndarray) and probabilities_per_patch.ndim == 2 and probabilities_per_patch.shape[1] == 2:
            positive_class_index = CLASS_INDEX_MAP.get(feature_key, 0)
            class_probs = probabilities_per_patch[:, positive_class_index]
            return float(np.mean(class_probs))
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Feature '{feature_key}' prediction failed: {e}")
        return 0.0


# --- RQ Task Definitions ---
# MODIFIED: Removed jellyfin_url, jellyfin_user_id, jellyfin_token as they are no longer needed for the function calls.
def analyze_album_task(album_id, album_name, top_n_moods, parent_task_id):
    from app import (app, JobStatus)
    from app_helper import (redis_conn, get_db, save_task_status, get_task_info_from_db,
                     save_track_analysis_and_embedding,
                     TASK_STATUS_STARTED, TASK_STATUS_PROGRESS, TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)
    
    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        initial_details = {"album_name": album_name, "log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Album analysis task started."]}
        save_task_status(current_task_id, "album_analysis", TASK_STATUS_STARTED, parent_task_id=parent_task_id, sub_type_identifier=album_id, progress=0, details=initial_details)
        tracks_analyzed_count, tracks_skipped_count, current_progress_val = 0, 0, 0
        current_task_logs = initial_details["log"]
        
        model_paths = {
            'embedding': EMBEDDING_MODEL_PATH,
            'prediction': PREDICTION_MODEL_PATH,
            'danceable': DANCEABILITY_MODEL_PATH,
            'aggressive': AGGRESSIVE_MODEL_PATH,
            'happy': HAPPY_MODEL_PATH,
            'party': PARTY_MODEL_PATH,
            'relaxed': RELAXED_MODEL_PATH,
            'sad': SAD_MODEL_PATH
        }

        def log_and_update_album_task(message, progress, **kwargs):
            nonlocal current_progress_val, current_task_logs
            current_progress_val = progress
            logger.info(f"[AlbumTask-{current_task_id}-{album_name}] {message}")
            db_details = {"album_name": album_name, **kwargs}
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            task_state = kwargs.get('task_state', TASK_STATUS_PROGRESS)

            if task_state in [TASK_STATUS_FAILURE, TASK_STATUS_REVOKED] or task_state != TASK_STATUS_SUCCESS:
                current_task_logs.append(log_entry)
                db_details["log"] = current_task_logs
            else:
                db_details["log"] = [f"Task completed successfully. Final status: {message}"]
            
            if current_job:
                current_job.meta.update({'progress': progress, 'status_message': message})
                current_job.save_meta()
            save_task_status(current_task_id, "album_analysis", task_state, parent_task_id=parent_task_id, sub_type_identifier=album_id, progress=progress, details=db_details)

        try:
            log_and_update_album_task(f"Fetching tracks for album: {album_name}", 5)
            # MODIFIED: Call to get_tracks_from_album no longer needs server parameters.
            tracks = get_tracks_from_album(album_id)
            if not tracks:
                log_and_update_album_task(f"No tracks found for album: {album_name}", 100, task_state=TASK_STATUS_SUCCESS)
                return {"status": "SUCCESS", "message": f"No tracks in album {album_name}", "tracks_analyzed": 0}

            def get_existing_track_ids(track_ids):
                if not track_ids: return set()
                with get_db() as conn, conn.cursor() as cur:
                    # MODIFIED: Cast the integer track IDs to TEXT for the database query.
                    track_ids_as_strings = [str(id) for id in track_ids]
                    cur.execute("SELECT s.item_id FROM score s JOIN embedding e ON s.item_id = e.item_id WHERE s.item_id IN %s AND s.other_features IS NOT NULL AND s.energy IS NOT NULL AND s.mood_vector IS NOT NULL AND s.tempo IS NOT NULL AND s.album IS NOT NULL AND s.album != '' AND s.song_artist IS NOT NULL AND s.song_artist != ''", (tuple(track_ids_as_strings),))
                    return {row[0] for row in cur.fetchall()}

            existing_track_ids_set = get_existing_track_ids( [str(t['Id']) for t in tracks])
            total_tracks_in_album = len(tracks)

            # --- PHASE 1: Audio Loading, CPU Features, and Embeddings ---
            log_and_update_album_task(f"Phase 1/4: Processing audio and generating embeddings...", 10)
            
            track_results = {} # Store results by track ID
            
            # Initialize Embedding Session ONLY
            try:
                embedding_sess = ort.InferenceSession(model_paths['embedding'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            except Exception as e:
                logger.critical(f"Failed to initialize embedding model: {e}")
                return {"status": "FAILURE", "message": "Embedding model init failed"}

            for idx, item in enumerate(tracks, 1):
                if current_job:
                    task_info = get_task_info_from_db(current_task_id)
                    parent_info = get_task_info_from_db(parent_task_id) if parent_task_id else None
                    if (task_info and task_info.get('status') == 'REVOKED') or (parent_info and parent_info.get('status') in ['REVOKED', 'FAILURE']):
                        log_and_update_album_task(f"Stopping album analysis for '{album_name}' due to parent/self revocation.", current_progress_val, task_state=TASK_STATUS_REVOKED)
                        return {"status": "REVOKED"}

                if str(item['Id']) in existing_track_ids_set:
                    tracks_skipped_count += 1
                    continue

                track_name_full = f"{item['Name']} by {item.get('AlbumArtist', 'Unknown')}"
                progress = 10 + int(30 * (idx / float(total_tracks_in_album))) # Phase 1 is 10-40%
                log_and_update_album_task(f"Processing audio: {track_name_full} ({idx}/{total_tracks_in_album})", progress, current_track_name=track_name_full)

                # Store artist mapping
                try:
                    from app_helper_artist import upsert_artist_mapping
                    if item.get('AlbumArtist') and item.get('ArtistId'):
                        upsert_artist_mapping(item.get('AlbumArtist'), item.get('ArtistId'))
                except Exception:
                    pass

                path = download_track(TEMP_DIR, item)
                if not path:
                    continue

                try:
                    cpu_features, embeddings_per_patch = process_audio_and_embedding(path, embedding_sess)
                    
                    if cpu_features and embeddings_per_patch is not None:
                        # Store everything needed for next phases
                        track_results[item['Id']] = {
                            'item': item,
                            'cpu_features': cpu_features,
                            'embeddings_per_patch': embeddings_per_patch
                        }
                    else:
                        logger.warning(f"Failed to process audio for {track_name_full}")
                        tracks_skipped_count += 1
                finally:
                    if path and os.path.exists(path):
                        os.remove(path)

            # Unload Embedding Session
            del embedding_sess
            import gc
            gc.collect()

            if not track_results:
                log_and_update_album_task(f"No tracks successfully processed in Phase 1.", 100, task_state=TASK_STATUS_SUCCESS)
                return {"status": "SUCCESS", "tracks_analyzed": 0}

            # --- PHASE 2: Mood Prediction ---
            log_and_update_album_task(f"Phase 2/4: Predicting moods...", 40)
            try:
                prediction_sess = ort.InferenceSession(model_paths['prediction'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                
                for i, (tid, data) in enumerate(track_results.items()):
                    moods = predict_moods(prediction_sess, data['embeddings_per_patch'], MOOD_LABELS)
                    track_results[tid]['moods'] = moods
                    # Update progress slightly
                    if i % 5 == 0:
                        log_and_update_album_task(f"Predicting moods... ({i+1}/{len(track_results)})", 40 + int(20 * (i / len(track_results))))

                del prediction_sess
                gc.collect()
            except Exception as e:
                logger.error(f"Phase 2 failed: {e}")
                raise

            # --- PHASE 3: Other Features ---
            log_and_update_album_task(f"Phase 3/4: Predicting other features...", 60)
            other_feature_keys = ["danceable", "aggressive", "happy", "party", "relaxed", "sad"]
            
            for f_idx, key in enumerate(other_feature_keys):
                log_and_update_album_task(f"Predicting feature: {key}...", 60 + int(30 * (f_idx / len(other_feature_keys))))
                try:
                    sess = ort.InferenceSession(model_paths[key], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                    for tid, data in track_results.items():
                        val = predict_other_feature(sess, data['embeddings_per_patch'], key)
                        track_results[tid][key] = val
                    del sess
                    gc.collect()
                except Exception as e:
                    logger.error(f"Failed to predict feature {key}: {e}")
                    # Continue with other features, assume 0.0 for this one
                    for tid in track_results:
                        track_results[tid][key] = 0.0

            # --- PHASE 4: Saving Results ---
            log_and_update_album_task(f"Phase 4/4: Saving results...", 90)
            
            for tid, data in track_results.items():
                item = data['item']
                cpu = data['cpu_features']
                moods = data['moods']
                
                # Calculate average embedding for storage
                avg_embedding = np.mean(data['embeddings_per_patch'], axis=0)
                
                top_moods = dict(sorted(moods.items(), key=lambda i: i[1], reverse=True)[:top_n_moods])
                other_features_str = ",".join([f"{k}:{data.get(k, 0.0):.2f}" for k in OTHER_FEATURE_LABELS])

                save_track_analysis_and_embedding(
                    item['Id'],
                    item['Name'],
                    item.get('AlbumArtist', 'Unknown'),
                    cpu['tempo'],
                    cpu['key'],
                    cpu['scale'],
                    top_moods,
                    avg_embedding,
                    energy=cpu['energy'],
                    other_features=other_features_str,
                    album=album_name,
                    song_artist=item.get('SongArtist'),
                    album_artist=item.get('OriginalAlbumArtist')
                )
                tracks_analyzed_count += 1

            summary = {"tracks_analyzed": tracks_analyzed_count, "tracks_skipped": tracks_skipped_count, "total_tracks_in_album": total_tracks_in_album}
            log_and_update_album_task(f"Album '{album_name}' analysis complete.", 100, task_state=TASK_STATUS_SUCCESS, final_summary_details=summary)
            return {"status": "SUCCESS", **summary}

        except OperationalError as e:
            logger.error(f"Database connection error during album analysis {album_id}: {e}. This job will be retried.", exc_info=True)
            log_and_update_album_task(f"Database connection failed for album '{album_name}'. Retrying...", current_progress_val, task_state=TASK_STATUS_FAILURE, final_summary_details={"error": str(e), "traceback": traceback.format_exc()})
            raise
        except Exception as e:
            logger.critical(f"Album analysis {album_id} failed: {e}", exc_info=True)
            log_and_update_album_task(f"Failed to analyze album '{album_name}': {e}", current_progress_val, task_state=TASK_STATUS_FAILURE, final_summary_details={"error": str(e), "traceback": traceback.format_exc()})
            raise

# MODIFIED: Removed jellyfin_url, jellyfin_user_id, jellyfin_token from signature.
def run_analysis_task(num_recent_albums, top_n_moods):
    from app import app
    from app_helper import (redis_conn, get_db, rq_queue_default, save_task_status, get_task_info_from_db, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS, TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)

    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())    

    with app.app_context():
        if num_recent_albums < 0:
             logger.warning("num_recent_albums is negative, treating as 0 (all albums).")
             num_recent_albums = 0

        task_info = get_task_info_from_db(current_task_id)
        if task_info and task_info.get('status') in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]:
            return {"status": task_info.get('status'), "message": "Task already in terminal state."}
        
        checked_album_ids = set(json.loads(task_info.get('details', '{}')).get('checked_album_ids', [])) if task_info else set()
        
        initial_details = {"message": "Fetching albums...", "log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Main analysis task started."]}

        save_task_status(current_task_id, "main_analysis", TASK_STATUS_STARTED, progress=0, details=initial_details)
        current_progress = 0
        current_task_logs = initial_details["log"]

        def log_and_update_main(message, progress, **kwargs):
            nonlocal current_progress, current_task_logs
            current_progress = progress
            logger.info(f"[MainAnalysisTask-{current_task_id}] {message}")
            details = {**kwargs, "status_message": message}
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            task_state = kwargs.get('task_state', TASK_STATUS_PROGRESS)
            
            if task_state != TASK_STATUS_SUCCESS:
                current_task_logs.append(log_entry)
                details["log"] = current_task_logs
            else:
                details["log"] = [f"Task completed successfully. Final status: {message}"]

            if current_job:
                current_job.meta.update({'progress': progress, 'status_message': message, 'details':details})
                current_job.save_meta()
            save_task_status(current_task_id, "main_analysis", task_state, progress=progress, details=details)

        try:
            log_and_update_main("🚀 Starting main analysis process...", 0)
            clean_temp(TEMP_DIR)
            # MODIFIED: Call to get_recent_albums no longer needs server parameters.
            all_albums = get_recent_albums(num_recent_albums)
            if not all_albums:
                log_and_update_main("⚠️ No new albums to analyze.", 100, albums_found=0, task_state=TASK_STATUS_SUCCESS)
                return {"status": "SUCCESS", "message": "No new albums to analyze."}

            total_albums_to_check = len(all_albums)
            active_jobs, launched_jobs = {}, []
            albums_skipped, albums_launched, albums_completed, last_rebuild_count = 0, 0, 0, 0

            def get_existing_track_ids(track_ids):
                if not track_ids: return set()
                with get_db() as conn, conn.cursor() as cur:
                    # Convert integer track IDs to strings for database comparison
                    track_ids_as_strings = [str(track_id) for track_id in track_ids]
                    cur.execute("SELECT s.item_id FROM score s JOIN embedding e ON s.item_id = e.item_id WHERE s.item_id IN %s AND s.other_features IS NOT NULL AND s.energy IS NOT NULL AND s.mood_vector IS NOT NULL AND s.tempo IS NOT NULL AND s.album IS NOT NULL AND s.album != '' AND s.song_artist IS NOT NULL AND s.song_artist != ''", (tuple(track_ids_as_strings),))
                    return {row[0] for row in cur.fetchall()}

            def monitor_and_clear_jobs():
                """Monitor active RQ jobs and keep `albums_completed` in sync.

                This function first tries to use RQ's Job.fetch to detect terminal jobs
                (finished/failed/canceled). As a more reliable fallback it also queries
                the database for child task records (which are updated by the child
                job when it finishes) and uses that as the source of truth. This
                helps in cases where RQ job state is not available or the worker
                uses a different Redis namespace.
                """
                nonlocal albums_completed, last_rebuild_count
                removed = 0

                # First: try to detect terminal jobs via RQ
                for job_id in list(active_jobs.keys()):
                    try:
                        job = Job.fetch(job_id, connection=redis_conn)
                        if job.is_finished or job.is_failed or job.is_canceled:
                            del active_jobs[job_id]
                            removed += 1
                    except NoSuchJobError:
                        logger.debug(f"Job {job_id} not found in RQ. Will reconcile with DB status.")
                        # Do not increment removed here; we'll reconcile via DB below.
                    except RedisTimeoutError:
                        logger.warning(f"Redis timeout while fetching job {job_id}. Will retry on next loop.")
                        continue
                    except Exception as e:
                        logger.warning(f"Unexpected error while fetching job {job_id}: {e}. Will retry on next loop.", exc_info=True)
                        continue

                if removed:
                    albums_completed += removed

                # Second: reconcile with DB child task statuses (authoritative)
                try:
                    from app_helper import get_child_tasks_from_db
                    child_tasks = get_child_tasks_from_db(current_task_id)
                    terminal_statuses = {TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED}
                    db_completed = sum(1 for t in child_tasks if t.get('status') in terminal_statuses)

                    if db_completed != albums_completed:
                        logger.info(f"Reconciling albums_completed: RQ_count={albums_completed} DB_count={db_completed}")
                        albums_completed = db_completed
                        # Remove any active_jobs whose IDs are in DB terminal list
                        terminal_ids = {t['task_id'] for t in child_tasks if t.get('status') in terminal_statuses}
                        for job_id in list(active_jobs.keys()):
                            if job_id in terminal_ids:
                                try:
                                    del active_jobs[job_id]
                                except KeyError:
                                    pass
                except Exception as e:
                    logger.error(f"Failed to reconcile child tasks from DB: {e}", exc_info=True)

                # Rebuild index in batches as before
                if albums_completed > last_rebuild_count and (albums_completed - last_rebuild_count) >= REBUILD_INDEX_BATCH_SIZE:
                    log_and_update_main(f"Batch of {albums_completed - last_rebuild_count} albums complete. Rebuilding index and map...", current_progress)
                    
                    # Build Voyager index
                    build_and_store_voyager_index(get_db())
                    
                    # Build artist similarity index
                    try:
                        build_and_store_artist_index(get_db())
                        logger.info('Artist similarity index rebuilt during batch.')
                    except Exception as e:
                        logger.warning(f"Failed to build/store artist similarity index during batch rebuild: {e}")
                    
                    # Build song map projection
                    try:
                        from app_helper import build_and_store_map_projection
                        build_and_store_map_projection('main_map')
                        logger.info('Song map projection rebuilt during batch.')
                    except Exception as e:
                        logger.warning(f"Failed to build/store map projection during batch rebuild: {e}")
                    
                    # Build artist component projection
                    try:
                        from app_helper import build_and_store_artist_projection
                        build_and_store_artist_projection('artist_map')
                        logger.info('Artist component projection rebuilt during batch.')
                    except Exception as e:
                        logger.warning(f"Failed to build/store artist projection during batch rebuild: {e}")
                    
                    # Publish single reload message to trigger Flask container to reload ALL indexes and maps
                    try:
                        redis_conn.publish('index-updates', 'reload')
                        logger.info('Published reload message to Flask container after batch rebuild.')
                    except Exception as e:
                        logger.warning(f'Could not publish reload message to redis during batch rebuild: {e}')
                    
                    last_rebuild_count = albums_completed

            for idx, album in enumerate(all_albums):
                # Periodically check for completed jobs to update progress
                monitor_and_clear_jobs()

                if album['Id'] in checked_album_ids:
                    albums_skipped += 1
                    continue
                
                while len(active_jobs) >= MAX_QUEUED_ANALYSIS_JOBS:
                    monitor_and_clear_jobs()
                    time.sleep(5)
                
                # MODIFIED: Call to get_tracks_from_album no longer needs server parameters.
                tracks = get_tracks_from_album(album['Id'])
                # If no tracks returned, skip and log reason.
                if not tracks:
                    albums_skipped += 1
                    checked_album_ids.add(album['Id'])
                    logger.info(f"Skipping album '{album.get('Name')}' (ID: {album.get('Id')}) - no tracks returned by media server.")
                    continue

                # Store artist ID mappings for all tracks in this album (even if already analyzed)
                try:
                    from app_helper_artist import upsert_artist_mapping
                    for track in tracks:
                        artist_name = track.get('AlbumArtist')
                        artist_id = track.get('ArtistId')
                        if artist_name and artist_id:
                            upsert_artist_mapping(artist_name, artist_id)
                            logger.info(f"✓ Mapped artist: '{artist_name}' → '{artist_id}'")
                        elif artist_name and not artist_id:
                            logger.warning(f"✗ No artist_id for '{artist_name}' in album '{album.get('Name')}'")
                except Exception as e:
                    logger.error(f"Failed to store artist mappings for album '{album.get('Name')}': {e}", exc_info=True)

                # If all tracks already exist in DB, skip and log how many.
                try:
                    existing_count = len(get_existing_track_ids([t['Id'] for t in tracks]))
                except Exception as e:
                    # Defensive: if DB check fails, log and continue to next album to avoid blocking the main loop.
                    logger.warning(f"Failed to verify existing tracks for album '{album.get('Name')}' (ID: {album.get('Id')}): {e}")
                    checked_album_ids.add(album['Id'])
                    albums_skipped += 1
                    continue

                if existing_count >= len(tracks):
                    albums_skipped += 1
                    checked_album_ids.add(album['Id'])
                    logger.info(f"Skipping album '{album.get('Name')}' (ID: {album.get('Id')}) - all {existing_count}/{len(tracks)} tracks already analyzed.")
                    continue
                
                # MODIFIED: Enqueue call for analyze_album_task now passes fewer arguments.
                job = rq_queue_default.enqueue('tasks.analysis.analyze_album_task', args=(album['Id'], album['Name'], top_n_moods, current_task_id), job_id=str(uuid.uuid4()), job_timeout=-1, retry=Retry(max=3))
                active_jobs[job.id] = job
                launched_jobs.append(job)
                albums_launched += 1
                checked_album_ids.add(album['Id'])
                
                progress = 5 + int(85 * (idx / float(total_albums_to_check)))
                status_message = f"Launched: {albums_launched}. Completed: {albums_completed}/{albums_launched}. Active: {len(active_jobs)}. Skipped: {albums_skipped}/{total_albums_to_check}."
                log_and_update_main(
                    status_message,
                    progress,
                    albums_to_process=albums_launched,
                    albums_skipped=albums_skipped,
                    checked_album_ids=list(checked_album_ids)
                )
                
            # If we never enqueued any album jobs for the batch, warn operator so they can investigate.
            if albums_launched == 0 and albums_skipped == total_albums_to_check:
                logger.warning(f"No albums were enqueued: all {total_albums_to_check} albums were skipped (no tracks or already analyzed). If unexpected, try running with num_recent_albums=0 to fetch more or inspect the media server responses and Spotify filtering.")

            while active_jobs:
                monitor_and_clear_jobs()
                progress = 5 + int(85 * ((albums_skipped + albums_completed) / float(total_albums_to_check)))
                status_message = f"Launched: {albums_launched}. Completed: {albums_completed}/{albums_launched}. Active: {len(active_jobs)}. Skipped: {albums_skipped}/{total_albums_to_check}. (Finalizing)"
                log_and_update_main(status_message, progress, checked_album_ids=list(checked_album_ids))
                time.sleep(5)

            log_and_update_main("Performing final index rebuild...", 95)
            # Build Voyager index (song embeddings)
            build_and_store_voyager_index(get_db())
            
            # Build artist similarity index
            log_and_update_main("Building artist similarity index...", 96)
            try:
                build_and_store_artist_index(get_db())
                logger.info('Artist similarity index built and stored.')
            except Exception as e:
                logger.warning(f"Failed to build/store artist similarity index: {e}")

            # Build and store the 2D map projection for the web map (best-effort)
            try:
                from app_helper import build_and_store_map_projection
                built = build_and_store_map_projection('main_map')
                if built:
                    logger.info('Precomputed map projection built and stored.')
                else:
                    logger.info('Precomputed map projection build returned no data (no embeddings?).')
            except Exception as e:
                logger.warning(f"Failed to build/store precomputed map projection: {e}")
            
            # Build and store the 2D artist component projection
            try:
                from app_helper import build_and_store_artist_projection
                built = build_and_store_artist_projection('artist_map')
                if built:
                    logger.info('Precomputed artist component projection built and stored.')
                else:
                    logger.info('Artist component projection build returned no data.')
            except Exception as e:
                logger.warning(f"Failed to build/store artist component projection: {e}")

            # Publish reload message to trigger Flask container to reload all indexes and maps
            try:
                redis_conn.publish('index-updates', 'reload')
                logger.info('Published reload message to Flask container after final analysis builds.')
            except Exception as e:
                logger.warning(f'Could not publish reload message to redis: {e}')

            final_message = f"Main analysis complete. Launched {albums_launched}, Skipped {albums_skipped}."
            log_and_update_main(final_message, 100, task_state=TASK_STATUS_SUCCESS)
            clean_temp(TEMP_DIR)
            return {"status": "SUCCESS", "message": final_message}

        except OperationalError as e:
            logger.critical(f"FATAL ERROR: Main analysis task failed due to DB connection issue: {e}", exc_info=True)
            log_and_update_main(f"❌ Main analysis failed due to a database connection error. The task may be retried.", current_progress, task_state=TASK_STATUS_FAILURE, error_message=str(e), traceback=traceback.format_exc())
            # Re-raise to allow RQ to handle retries if configured on the task itself
            raise
        except Exception as e:
            logger.critical(f"FATAL ERROR: Analysis failed: {e}", exc_info=True)
            log_and_update_main(f"❌ Main analysis failed: {e}", current_progress, task_state=TASK_STATUS_FAILURE, error_message=str(e), traceback=traceback.format_exc())
            raise


