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
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True, duration=AUDIO_LOAD_TIMEOUT)
        
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
        audio, sr = librosa.load(temp_wav_path, sr=target_sr, mono=True, duration=AUDIO_LOAD_TIMEOUT)
        
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

def rebuild_all_indexes_task():
    """
    Rebuild all indexes (Voyager, Artist GMM, Map, Artist projection) as a standalone RQ task.
    This is enqueued on the default queue to run serially with album analysis tasks,
    preventing CPU overlap between song analysis and index rebuilds.
    """
    from app import app
    from app_helper import get_db, redis_conn
    
    logger.info("🔨 Starting index rebuild task (enqueued as subtask)...")
    
    with app.app_context():
        try:
            # Build Voyager index
            build_and_store_voyager_index(get_db())
            logger.info('✓ Voyager index rebuilt')
            
            # Build artist similarity index
            try:
                build_and_store_artist_index(get_db())
                logger.info('✓ Artist similarity index rebuilt')
            except Exception as e:
                logger.warning(f"Failed to build/store artist similarity index: {e}")
            
            # Build song map projection
            try:
                from app_helper import build_and_store_map_projection
                build_and_store_map_projection('main_map')
                logger.info('✓ Song map projection rebuilt')
            except Exception as e:
                logger.warning(f"Failed to build/store map projection: {e}")
            
            # Build artist component projection
            try:
                from app_helper import build_and_store_artist_projection
                build_and_store_artist_projection('artist_map')
                logger.info('✓ Artist component projection rebuilt')
            except Exception as e:
                logger.warning(f"Failed to build/store artist projection: {e}")
            
            # Publish reload message to Flask container
            try:
                redis_conn.publish('index-updates', 'reload')
                logger.info('✓ Published reload message to Flask container')
            except Exception as e:
                logger.warning(f'Could not publish reload message: {e}')
            
            logger.info("✅ Index rebuild task completed successfully")
            return {"status": "SUCCESS", "message": "All indexes rebuilt"}
            
        except Exception as e:
            logger.error(f"❌ Index rebuild task failed: {e}", exc_info=True)
            return {"status": "FAILURE", "message": str(e)}

def analyze_track(file_path, mood_labels_list, model_paths, onnx_sessions=None):
    """
    Analyzes a single track using ONNX Runtime for inference.
    
    Args:
        file_path: Path to audio file
        mood_labels_list: List of mood labels
        model_paths: Dict of model paths
        onnx_sessions: Optional dict of pre-loaded ONNX sessions (for album-level reuse)
    """
    logger.info(f"Starting analysis for: {os.path.basename(file_path)}")

    # --- 1. Load Audio and Compute Basic Features ---
    audio, sr = robust_load_audio_with_fallback(file_path, target_sr=16000)
    
    if audio is None or not np.any(audio) or audio.size == 0:
        logger.warning(f"Could not load a valid audio signal for {os.path.basename(file_path)} after all attempts. Skipping track.")
        return None, None

    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    average_energy = np.mean(librosa.feature.rms(y=audio))
    
    # Improved key/scale detection
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


    # --- 2. Prepare Spectrograms --- 
    try:
        # Using the spectrogram settings confirmed to work for the main model
        n_mels, hop_length, n_fft, frame_size = 96, 256, 512, 187
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window='hann', center=False, power=2.0, norm='slaney', htk=False)


        log_mel_spec = np.log10(1 + 10000 * mel_spec)

        spec_patches = [log_mel_spec[:, i:i+frame_size] for i in range(0, log_mel_spec.shape[1] - frame_size + 1, frame_size)]
        if not spec_patches:
            logger.warning(f"Track too short to create spectrogram patches: {os.path.basename(file_path)}")
            return None, None
        
        transposed_patches = np.array(spec_patches).transpose(0, 2, 1)

        # =================================================================
        # === START: CORRECT FIX FOR DATA TYPE PRECISION ===
        # The crash on specific CPUs is due to a float precision mismatch. The model
        # expects float32, but the array can sometimes be float64. Explicitly casting
        # to float32 is the correct, minimal fix that preserves all data and
        # ensures compatibility.
        final_patches = transposed_patches.astype(np.float32)
        # === END: CORRECT FIX FOR DATA TYPE PRECISION ===
        # =================================================================

    except Exception as e:
        logger.error(f"Spectrogram creation failed for {os.path.basename(file_path)}: {e}", exc_info=True)
        return None, None

# --- 3. Run Main Models (Embedding and Prediction) ---
    try:
        # Use pre-loaded sessions if provided, otherwise load per-song
        if onnx_sessions is not None:
            embedding_sess = onnx_sessions['embedding']
            prediction_sess = onnx_sessions['prediction']
            should_cleanup_sessions = False
        else:
            # Load and run embedding model (ONNX) with GPU memory management
            # ONNX Runtime handles CUDA availability and fallback internally
            available_providers = ort.get_available_providers()
            
            # Configure provider options for GPU memory management
            if 'CUDAExecutionProvider' in available_providers:
                # Get GPU device ID from environment or default to 0
                # Docker sets NVIDIA_VISIBLE_DEVICES, CUDA runtime uses CUDA_VISIBLE_DEVICES
                gpu_device_id = 0
                cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                if cuda_visible and cuda_visible != '-1':
                    # If CUDA_VISIBLE_DEVICES is set, use first device (already mapped to 0)
                    gpu_device_id = 0
                
                # GPU mode with memory management to prevent fragmentation
                cuda_options = {
                    'device_id': gpu_device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',  # Better memory allocation
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',      # Find memory-efficient algorithms
                    'do_copy_in_default_stream': True,           # Better memory sync
                }
                provider_options = [('CUDAExecutionProvider', cuda_options), ('CPUExecutionProvider', {})]
                logger.info(f"CUDA provider available - attempting to use GPU for analysis (device_id={gpu_device_id})")
            else:
                provider_options = [('CPUExecutionProvider', {})]
                logger.info("CUDA provider not available - using CPU only")
            
            try:
                embedding_sess = ort.InferenceSession(
                    model_paths['embedding'],
                    providers=[p[0] for p in provider_options],
                    provider_options=[p[1] for p in provider_options]
                )
            except Exception:
                # Fallback to CPU if preferred providers fail
                logger.warning(f"Failed to load embedding model with GPU - falling back to CPU")
                embedding_sess = ort.InferenceSession(
                    model_paths['embedding'],
                    providers=['CPUExecutionProvider']
                )
            
            try:
                prediction_sess = ort.InferenceSession(
                    model_paths['prediction'],
                    providers=[p[0] for p in provider_options],
                    provider_options=[p[1] for p in provider_options]
                )
            except Exception:
                # Fallback to CPU if preferred providers fail
                logger.warning(f"Failed to load prediction model with GPU - falling back to CPU")
                prediction_sess = ort.InferenceSession(
                    model_paths['prediction'],
                    providers=['CPUExecutionProvider']
                )
            should_cleanup_sessions = True
        
        embedding_feed_dict = {DEFINED_TENSOR_NAMES['embedding']['input']: final_patches}
        try:
            embeddings_per_patch = run_inference(embedding_sess, embedding_feed_dict, DEFINED_TENSOR_NAMES['embedding']['output'])
        except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
            if "Failed to allocate memory" in str(e):
                logger.warning(f"GPU OOM detected for {os.path.basename(file_path)} during embedding inference, retrying on CPU...")
                # Create temporary CPU session
                embedding_sess_cpu = ort.InferenceSession(
                    model_paths['embedding'],
                    providers=['CPUExecutionProvider']
                )
                embeddings_per_patch = run_inference(embedding_sess_cpu, embedding_feed_dict, DEFINED_TENSOR_NAMES['embedding']['output'])
                del embedding_sess_cpu
                import gc
                gc.collect()
                logger.info(f"Successfully completed embedding inference on CPU after GPU OOM")
            else:
                raise
        
        prediction_feed_dict = {DEFINED_TENSOR_NAMES['prediction']['input']: embeddings_per_patch}
        try:
            mood_logits = run_inference(prediction_sess, prediction_feed_dict, DEFINED_TENSOR_NAMES['prediction']['output'])
        except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
            if "Failed to allocate memory" in str(e):
                logger.warning(f"GPU OOM detected for {os.path.basename(file_path)} during prediction inference, retrying on CPU...")
                # Create temporary CPU session
                prediction_sess_cpu = ort.InferenceSession(
                    model_paths['prediction'],
                    providers=['CPUExecutionProvider']
                )
                mood_logits = run_inference(prediction_sess_cpu, prediction_feed_dict, DEFINED_TENSOR_NAMES['prediction']['output'])
                del prediction_sess_cpu
                import gc
                gc.collect()
                logger.info(f"Successfully completed prediction inference on CPU after GPU OOM")
            else:
                raise
        
        # Only cleanup if we loaded models in this function (not reusing album-level sessions)
        if should_cleanup_sessions:
            del embedding_sess, prediction_sess
            import gc
            gc.collect()

        averaged_logits = np.mean(mood_logits, axis=0)
        # Apply sigmoid to convert raw model outputs (logits) into probabilities
        final_mood_predictions = sigmoid(averaged_logits)

        moods = {label: float(score) for label, score in zip(mood_labels_list, final_mood_predictions)}

    except Exception as e:
        logger.error(f"Main model inference failed for {os.path.basename(file_path)}: {e}", exc_info=True)
        return None, None

        
    # --- 4. Run Secondary Models ---
    other_predictions = {}

    for key in ["danceable", "aggressive", "happy", "party", "relaxed", "sad"]:
        try:
            # Use pre-loaded sessions if provided, otherwise load per-song
            if onnx_sessions is not None:
                other_sess = onnx_sessions[key]
                should_cleanup_other = False
            else:
                model_path = model_paths[key]
                # Load model with same provider configuration as main models
                try:
                    other_sess = ort.InferenceSession(
                        model_path,
                        providers=[p[0] for p in provider_options],
                        provider_options=[p[1] for p in provider_options]
                    )
                except Exception:
                    # Fallback to CPU if preferred providers fail
                    logger.warning(f"Failed to load {key} model with GPU - falling back to CPU")
                    other_sess = ort.InferenceSession(
                        model_path,
                        providers=['CPUExecutionProvider']
                    )
                should_cleanup_other = True
            
            feed_dict = {DEFINED_TENSOR_NAMES[key]['input']: embeddings_per_patch}
            try:
                probabilities_per_patch = run_inference(other_sess, feed_dict, DEFINED_TENSOR_NAMES[key]['output'])
            except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
                if "Failed to allocate memory" in str(e):
                    logger.warning(f"GPU OOM detected for {os.path.basename(file_path)} during {key} inference, retrying on CPU...")
                    # Create temporary CPU session
                    other_sess_cpu = ort.InferenceSession(
                        model_paths[key],
                        providers=['CPUExecutionProvider']
                    )
                    probabilities_per_patch = run_inference(other_sess_cpu, feed_dict, DEFINED_TENSOR_NAMES[key]['output'])
                    del other_sess_cpu
                    import gc
                    gc.collect()
                    logger.info(f"Successfully completed {key} inference on CPU after GPU OOM")
                else:
                    raise
            
            # Only cleanup if we loaded model in this function
            if should_cleanup_other:
                del other_sess
                import gc
                gc.collect()

            if probabilities_per_patch is None:
                other_predictions[key] = 0.0
            else:
                if isinstance(probabilities_per_patch, np.ndarray) and probabilities_per_patch.ndim == 2 and probabilities_per_patch.shape[1] == 2:
                    # Using the CLASS_INDEX_MAP to select the correct probability
                    positive_class_index = CLASS_INDEX_MAP.get(key, 0)
                    class_probs = probabilities_per_patch[:, positive_class_index]
                    other_predictions[key] = float(np.mean(class_probs))
                else:
                    other_predictions[key] = 0.0

        except Exception as e:
            logger.error(f"Error predicting '{key}' for {os.path.basename(file_path)}: {e}", exc_info=True)
            other_predictions[key] = 0.0

    # --- 5. Final Aggregation for Storage ---
    processed_embeddings = np.mean(embeddings_per_patch, axis=0)

    return {
        "tempo": float(tempo), "key": musical_key, "scale": scale,
        "moods": moods, "energy": float(average_energy), **other_predictions
    }, processed_embeddings


# --- RQ Task Definitions ---
# MODIFIED: Removed jellyfin_url, jellyfin_user_id, jellyfin_token as they are no longer needed for the function calls.
def analyze_album_task(album_id, album_name, top_n_moods, parent_task_id):
    from app import (app, JobStatus)
    from app_helper import (redis_conn, get_db, save_task_status, get_task_info_from_db,
                     save_track_analysis_and_embedding, save_clap_embedding,
                     TASK_STATUS_STARTED, TASK_STATUS_PROGRESS, TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)
    from .clap_analyzer import analyze_audio_file as clap_analyze, is_clap_available
    from .mulan_analyzer import analyze_audio_file as mulan_analyze
    from config import MULAN_ENABLED
    
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
        
        # Essentia models will be lazy-loaded on first song that needs MusiCNN analysis
        onnx_sessions = None

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
                    cur.execute("SELECT s.item_id FROM score s JOIN embedding e ON s.item_id = e.item_id WHERE s.item_id IN %s AND s.other_features IS NOT NULL AND s.energy IS NOT NULL AND s.mood_vector IS NOT NULL AND s.tempo IS NOT NULL", (tuple(track_ids_as_strings),))
                    return {row[0] for row in cur.fetchall()}

            def get_missing_clap_track_ids(track_ids):
                """Returns set of track IDs that need CLAP analysis (missing from clap_embedding table)."""
                if not track_ids: return set()
                with get_db() as conn, conn.cursor() as cur:
                    track_ids_as_strings = [str(id) for id in track_ids]
                    cur.execute("SELECT item_id FROM clap_embedding WHERE item_id IN %s", (tuple(track_ids_as_strings),))
                    existing_clap_ids = {row[0] for row in cur.fetchall()}
                    # Return tracks that DON'T have CLAP embeddings yet
                    return set(track_ids_as_strings) - existing_clap_ids

            def get_missing_mulan_track_ids(track_ids):
                """Returns set of track IDs that need MuLan analysis (missing from mulan_embedding table)."""
                if not track_ids: return set()
                with get_db() as conn, conn.cursor() as cur:
                    track_ids_as_strings = [str(id) for id in track_ids]
                    cur.execute("SELECT item_id FROM mulan_embedding WHERE item_id IN %s", (tuple(track_ids_as_strings),))
                    existing_mulan_ids = {row[0] for row in cur.fetchall()}
                    # Return tracks that DON'T have MuLan embeddings yet
                    return set(track_ids_as_strings) - existing_mulan_ids

            existing_track_ids_set = get_existing_track_ids( [str(t['Id']) for t in tracks])
            missing_clap_ids_set = get_missing_clap_track_ids([str(t['Id']) for t in tracks]) if is_clap_available() else set()
            missing_mulan_ids_set = get_missing_mulan_track_ids([str(t['Id']) for t in tracks]) if MULAN_ENABLED else set()
            total_tracks_in_album = len(tracks)

            for idx, item in enumerate(tracks, 1):
                if current_job:
                    task_info = get_task_info_from_db(current_task_id)
                    parent_info = get_task_info_from_db(parent_task_id) if parent_task_id else None
                    if (task_info and task_info.get('status') == 'REVOKED') or (parent_info and parent_info.get('status') in ['REVOKED', 'FAILURE']):
                        log_and_update_album_task(f"Stopping album analysis for '{album_name}' due to parent/self revocation.", current_progress_val, task_state=TASK_STATUS_REVOKED)
                        return {"status": "REVOKED"}

                track_name_full = f"{item['Name']} by {item.get('AlbumArtist', 'Unknown')}"
                progress = 10 + int(85 * (idx / float(total_tracks_in_album)))
                log_and_update_album_task(f"Analyzing track: {track_name_full} ({idx}/{total_tracks_in_album})", progress, current_track_name=track_name_full)

                # Store artist ID mapping for all tracks (even if already analyzed)
                try:
                    from app_helper_artist import upsert_artist_mapping
                    artist_name = item.get('AlbumArtist')
                    artist_id = item.get('ArtistId')
                    logger.info(f"Track '{item.get('Name')}': artist_name='{artist_name}', artist_id='{artist_id}'")
                    if artist_name and artist_id:
                        upsert_artist_mapping(artist_name, artist_id)
                        logger.info(f"✓ Stored artist mapping: '{artist_name}' → '{artist_id}'")
                    else:
                        if not artist_id:
                            logger.warning(f"✗ No artist_id for track '{item.get('Name')}' by '{artist_name}'")
                except Exception as mapping_error:
                    logger.error(f"Failed to store artist mapping for '{artist_name}': {mapping_error}", exc_info=True)

                track_id_str = str(item['Id'])
                needs_musicnn = track_id_str not in existing_track_ids_set
                needs_clap = track_id_str in missing_clap_ids_set
                needs_mulan = track_id_str in missing_mulan_ids_set
                
                # Skip if ALL enabled analyses are already done
                if not needs_musicnn and not needs_clap and not needs_mulan:
                    tracks_skipped_count += 1
                    # Build dynamic status message based on enabled features
                    status_parts = ["MusiCNN: ✓"]
                    if is_clap_available():
                        status_parts.append("CLAP: ✓")
                    if MULAN_ENABLED:
                        status_parts.append("MuLan: ✓")
                    logger.info(f"Skipping '{track_name_full}' - all analyses complete ({', '.join(status_parts)})")
                    continue
                
                # MODIFIED: Call to download_track simplified. Assumes it gets server details from config.
                path = download_track(TEMP_DIR, item)
                if not path:
                    continue

                try:
                    # Track if we processed anything (MusiCNN or CLAP)
                    track_processed = False
                    
                    # MusiCNN analysis (only if needed)
                    if needs_musicnn:
                        # Lazy-load Essentia models on first song that needs analysis
                        if onnx_sessions is None:
                            logger.info(f"Lazy-loading Essentia models for album: {album_name}")
                            onnx_sessions = {}
                            available_providers = ort.get_available_providers()
                            
                            if 'CUDAExecutionProvider' in available_providers:
                                gpu_device_id = 0
                                cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                                if cuda_visible and cuda_visible != '-1':
                                    gpu_device_id = 0
                                cuda_options = {
                                    'device_id': gpu_device_id,
                                    'arena_extend_strategy': 'kNextPowerOfTwo',  # Better memory allocation
                                    'cudnn_conv_algo_search': 'EXHAUSTIVE',      # Find memory-efficient algorithms
                                    'do_copy_in_default_stream': True,           # Better memory sync
                                }
                                provider_options = [('CUDAExecutionProvider', cuda_options), ('CPUExecutionProvider', {})]
                            else:
                                provider_options = [('CPUExecutionProvider', {})]
                            
                            try:
                                for model_name, model_path in model_paths.items():
                                    try:
                                        onnx_sessions[model_name] = ort.InferenceSession(
                                            model_path,
                                            providers=[p[0] for p in provider_options],
                                            provider_options=[p[1] for p in provider_options]
                                        )
                                    except Exception:
                                        onnx_sessions[model_name] = ort.InferenceSession(
                                            model_path,
                                            providers=['CPUExecutionProvider']
                                        )
                                logger.info(f"✓ Loaded {len(onnx_sessions)} Essentia models for album reuse")
                            except Exception as e:
                                logger.error(f"Failed to load Essentia models: {e}")
                                onnx_sessions = None
                        
                        analysis, embedding = analyze_track(path, MOOD_LABELS, model_paths, onnx_sessions=onnx_sessions)
                        if analysis is None:
                            logger.warning(f"Skipping track {track_name_full} as analysis returned None.")
                            tracks_skipped_count += 1
                            continue
                        
                        top_moods = dict(sorted(analysis['moods'].items(), key=lambda i: i[1], reverse=True)[:top_n_moods])
                        other_features = ",".join([f"{k}:{analysis.get(k, 0.0):.2f}" for k in OTHER_FEATURE_LABELS])
                        
                        logger.info(f"SUCCESSFULLY ANALYZED '{track_name_full}' (ID: {item['Id']}):")
                        logger.info(f"  - Tempo: {analysis['tempo']:.2f}, Energy: {analysis['energy']:.4f}, Key: {analysis['key']} {analysis['scale']}")
                        logger.info(f"  - Top Moods: {top_moods}")
                        logger.info(f"  - Other Features: {other_features}")
                        
                        save_track_analysis_and_embedding(item['Id'], item['Name'], item.get('AlbumArtist', 'Unknown'), analysis['tempo'], analysis['key'], analysis['scale'], top_moods, embedding, energy=analysis['energy'], other_features=other_features)
                        track_processed = True
                    else:
                        logger.info(f"SKIPPED MusiCNN for '{track_name_full}' (already analyzed)")
                    
                    # CLAP analysis (only if enabled AND needed)
                    if needs_clap and is_clap_available():
                        logger.info(f"  - Starting CLAP analysis for {track_name_full}...")
                        try:
                            clap_embedding, _, _ = clap_analyze(path)
                            if clap_embedding is not None:
                                save_clap_embedding(item['Id'], clap_embedding)
                                logger.info(f"  - CLAP embedding saved (512-dim)")
                                track_processed = True
                        except Exception as e:
                            logger.warning(f"  - CLAP analysis failed: {e}")
                    elif not needs_clap and is_clap_available():
                        logger.info(f"  - CLAP embedding already exists, skipping")
                    else:
                        logger.info(f"  - CLAP skipped: needs_clap={needs_clap}, available={is_clap_available()}")
                    
                    # MuLan analysis (only if enabled AND needed)
                    if needs_mulan and MULAN_ENABLED:
                        logger.info(f"  - Starting MuLan analysis for {track_name_full}...")
                        try:
                            mulan_embedding, duration, num_segments = mulan_analyze(path)
                            if mulan_embedding is not None:
                                from app_helper import save_mulan_embedding
                                save_mulan_embedding(item['Id'], mulan_embedding)
                                logger.info(f"  - MuLan embedding saved (512-dim, duration: {duration:.1f}s)")
                                track_processed = True
                        except Exception as e:
                            logger.warning(f"  - MuLan analysis failed: {e}")
                    elif not needs_mulan and MULAN_ENABLED:
                        logger.info(f"  - MuLan embedding already exists, skipping")
                    
                    # Count track as analyzed if we processed MusiCNN, CLAP, or MuLan
                    if track_processed:
                        tracks_analyzed_count += 1
                    
                finally:
                    if path and os.path.exists(path):
                        os.remove(path)
            
            # Cleanup all models after album analysis to free memory
            if onnx_sessions:
                logger.info(f"Cleaning up {len(onnx_sessions)} Essentia model sessions")
                del onnx_sessions
                import gc
                gc.collect()
            
            # Cleanup CLAP model if it was loaded during this album
            from .clap_analyzer import unload_clap_model, is_clap_model_loaded
            if is_clap_model_loaded():
                logger.info("Cleaning up CLAP model after album analysis")
                unload_clap_model()
            
            # Cleanup MuLan model if it was loaded during this album
            from .mulan_analyzer import unload_mulan_model, is_mulan_model_loaded
            if is_mulan_model_loaded():
                logger.info("Cleaning up MuLan model after album analysis")
                unload_mulan_model()

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
    from .clap_analyzer import is_clap_available
    import config  # Import config to access MULAN_ENABLED

    MULAN_ENABLED = getattr(config, 'MULAN_ENABLED', False)  # Get MULAN_ENABLED from config

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
                    cur.execute("SELECT s.item_id FROM score s JOIN embedding e ON s.item_id = e.item_id WHERE s.item_id IN %s AND s.other_features IS NOT NULL AND s.energy IS NOT NULL AND s.mood_vector IS NOT NULL AND s.tempo IS NOT NULL", (tuple(track_ids_as_strings),))
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

                # Enqueue index rebuild as subtask on default queue (runs serially with song analysis)
                if albums_completed > last_rebuild_count and (albums_completed - last_rebuild_count) >= REBUILD_INDEX_BATCH_SIZE:
                    log_and_update_main(f"Batch of {albums_completed - last_rebuild_count} albums complete. Enqueueing index rebuild...", current_progress)
                    
                    rebuild_job = rq_queue_default.enqueue(
                        'tasks.analysis.rebuild_all_indexes_task',
                        job_id=str(uuid.uuid4()),
                        job_timeout=-1,
                        retry=Retry(max=3)
                    )
                    logger.info(f"⏰ Enqueued index rebuild job {rebuild_job.id} on default queue (will run serially with album tasks)")
                    
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

                # Check if album needs any analysis (MusiCNN OR CLAP OR MuLan)
                try:
                    track_ids = [t['Id'] for t in tracks]
                    existing_count = len(get_existing_track_ids(track_ids))
                    
                    # Check CLAP if enabled
                    needs_clap_analysis = False
                    if is_clap_available():
                        with get_db() as conn, conn.cursor() as cur:
                            track_ids_as_strings = [str(id) for id in track_ids]
                            cur.execute("SELECT item_id FROM clap_embedding WHERE item_id IN %s", (tuple(track_ids_as_strings),))
                            existing_clap_ids = {row[0] for row in cur.fetchall()}
                            needs_clap_analysis = len(existing_clap_ids) < len(tracks)
                    
                    # Check MuLan only if enabled
                    needs_mulan_analysis = False
                    if MULAN_ENABLED:
                        with get_db() as conn, conn.cursor() as cur:
                            track_ids_as_strings = [str(id) for id in track_ids]
                            cur.execute("SELECT item_id FROM mulan_embedding WHERE item_id IN %s", (tuple(track_ids_as_strings),))
                            existing_mulan_ids = {row[0] for row in cur.fetchall()}
                            needs_mulan_analysis = len(existing_mulan_ids) < len(tracks)
                    
                except Exception as e:
                    # Defensive: if DB check fails, log and continue to next album to avoid blocking the main loop.
                    logger.warning(f"Failed to verify existing tracks for album '{album.get('Name')}' (ID: {album.get('Id')}): {e}")
                    checked_album_ids.add(album['Id'])
                    albums_skipped += 1
                    continue

                # Skip ONLY if all tracks have MusiCNN AND CLAP (if enabled) AND MuLan (if enabled)
                if existing_count >= len(tracks) and not needs_clap_analysis and not needs_mulan_analysis:
                    albums_skipped += 1
                    checked_album_ids.add(album['Id'])
                    # Build dynamic status message based on enabled features
                    status_parts = ["MusiCNN"]
                    if is_clap_available():
                        status_parts.append("CLAP")
                    if MULAN_ENABLED:
                        status_parts.append("MuLan")
                    logger.info(f"Skipping album '{album.get('Name')}' (ID: {album.get('Id')}) - all {existing_count}/{len(tracks)} tracks already analyzed ({' + '.join(status_parts)}).")
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

            # Top query computation disabled - using default queries from database only
            logger.info('Analysis complete. CLAP text search uses default queries (no auto-regeneration).')

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
