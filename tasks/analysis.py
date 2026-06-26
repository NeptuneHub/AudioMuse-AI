# tasks/analysis.py
#
# Public RQ entry-points and orchestration for the AudioMuse analysis pipeline.
# Reusable building blocks (ONNX helpers, audio feature extraction, DB lookups,
# metadata refresh, model cleanup) live in tasks/analysis_helper.py.

import os
import shutil
import numpy as np
import time
import logging
import uuid
import gc
import platform

import librosa
import onnxruntime as ort  # noqa: F401  re-exported: tests patch `tasks.analysis.ort.InferenceSession`

# RQ import
from rq import get_current_job, Retry
from rq.job import Job
from rq.exceptions import NoSuchJobError

# Import configuration from the user's provided config file
from config import (
    TEMP_DIR, MOOD_LABELS, EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH,
    OTHER_FEATURE_LABELS,
    REBUILD_INDEX_BATCH_SIZE, MAX_QUEUED_ANALYSIS_JOBS, PER_SONG_MODEL_RELOAD,
    AUDIO_LOAD_TIMEOUT, LYRICS_ENABLED,
    ANALYSIS_MONITOR_DB_INTERVAL,
)


# Import other project modules
from .mediaserver import (
    get_recent_albums, get_tracks_from_album, download_track,
    test_connection as mediaserver_test_connection,
)
from .memory_utils import (
    cleanup_cuda_memory,
    cleanup_onnx_session,
    SessionRecycler,
    comprehensive_memory_cleanup,
)

# `app_helper` is safe to import here (no module-level cycle back into
# tasks.analysis). The Flask `app` instance lives in `flask_app` (a tiny
# shared module) precisely so we can import it at module top without
# creating a cycle with app.py.
from flask_app import app
from app_helper import (
    redis_conn, rq_queue_default, get_db, save_task_status,
    get_task_info_from_db,
    build_and_store_map_projection, build_and_store_artist_projection,
    TASK_STATUS_STARTED, TASK_STATUS_PROGRESS, TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE, TASK_STATUS_REVOKED,
)
from database import get_child_tasks_from_db

from error import error_manager
from error.error_dictionary import (
    ERR_ANALYSIS_FAILED,
    ERR_ALBUM_ANALYSIS_FAILED,
    ERR_DB_CONNECTION,
    ERR_MEDIASERVER_LIBRARY,
    ERR_MEDIASERVER_AUTH,
    ERR_MEDIASERVER_UNREACHABLE,
    ERR_INDEX_BUILD,
    ERR_INDEX_EMPTY,
)

# Helper module — exposes refactored utilities. The explicit re-exports
# below keep the legacy ``tasks.analysis.<symbol>`` attribute surface that
# tests depend on (``run_inference``, ``_find_onnx_name``, ``sigmoid``).
# Helpers consumed only inside this file go through ``_ah.<name>`` instead.
from . import analysis_helper as _ah
from .analysis_helper import (  # noqa: F401
    DEFINED_TENSOR_NAMES,
    _find_onnx_name,           # re-export: tests do `from tasks.analysis import _find_onnx_name`
    run_inference,             # re-export: tests do `from tasks.analysis import run_inference`
    sigmoid,
    extract_basic_features,
    prepare_spectrogram_patches,
    get_provider_options,
    create_onnx_session,
    load_musicnn_sessions,
    cleanup_musicnn_sessions,
    cleanup_optional_models,
    run_inference_with_oom_fallback,
)


from psycopg2 import OperationalError
from redis.exceptions import TimeoutError as RedisTimeoutError  # alias
logger = logging.getLogger(__name__)


# --- Utility Functions ---
def clean_temp(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    for name in os.listdir(temp_dir):
        path = os.path.join(temp_dir, name)
        try:
            (shutil.rmtree if os.path.isdir(path) and not os.path.islink(path) else os.unlink)(path)
        except Exception as e:
            logger.warning(f"Could not remove {path} from {temp_dir}: {e}")


def _release_freed_ram_to_os():
    gc.collect()
    
    #malloc_trim is Linux/glibc specific
    if platform.system() != "Linux":
        return
        
    try:
        import ctypes
        import ctypes.util
        libc_name = ctypes.util.find_library("c")
        if not libc_name:
            return
        libc = ctypes.CDLL(libc_name)
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _run_all_index_builds(log_fn=None):
    """Run every index-rebuild step. log_fn(stage, progress) is optional.

    Each step announces itself via ``log_fn`` before running so the dashboard
    shows which builder is currently active (otherwise users see "Building
    CLAP text search index..." for the entire 95–97 % window even while the
    lyrics or SemGrove builds are running).

    The index-builder modules are imported here rather than at module top so
    that importing ``tasks.analysis`` does not pull in the ivf / CLAP /
    lyrics / SemGrove / artist-GMM subsystems; they are only needed when a
    rebuild actually runs.
    """
    from .ivf_manager import build_and_store_ivf_index
    from .clap_text_search import build_and_store_clap_index
    from .lyrics_manager import build_and_store_lyrics_index, build_and_store_lyrics_axes_index
    from .sem_grove_manager import build_and_store_sem_grove_index
    from .artist_gmm_manager import build_and_store_artist_index

    def _step(label, fn, progress=None, banner=None, fatal=False):
        if log_fn and progress is not None and banner is not None:
            try:
                log_fn(banner, progress)
            except Exception:
                pass
        try:
            fn()
            logger.info(f"✓ {label}")
        except Exception as e:
            logger.warning(f"Failed to build/store {label}: {e}")
            if fatal:
                raise
        finally:
            gc.collect()

    if log_fn:
        log_fn("Performing final index rebuild...", 95)
    _step("IVF index rebuilt",
          lambda: build_and_store_ivf_index(get_db()),
          progress=95, banner="Building IVF audio index...",
          fatal=True)
    _step("CLAP text search index",
          lambda: build_and_store_clap_index(get_db()),
          progress=96, banner="Building CLAP text search index...")
    _step("Lyrics search index",
          lambda: build_and_store_lyrics_index(get_db()),
          progress=96, banner="Building lyrics search index...")
    _step("Lyrics axes index",
          lambda: build_and_store_lyrics_axes_index(get_db()),
          progress=96, banner="Building lyrics axes index...")
    _step("SemGrove merged index rebuilt",
          lambda: build_and_store_sem_grove_index(get_db()),
          progress=96, banner="Building SemGrove merged index...")
    _step("Artist similarity index rebuilt",
          lambda: build_and_store_artist_index(get_db()),
          progress=97, banner="Building artist similarity index...")
    _step("Song map projection rebuilt",
          lambda: build_and_store_map_projection('main_map'),
          progress=97, banner="Building song map projection...")
    _step("Artist component projection rebuilt",
          lambda: build_and_store_artist_projection('artist_map'),
          progress=97, banner="Building artist component projection...")
    try:
        redis_conn.publish('index-updates', 'reload')
        logger.info('✓ Published reload message to Flask container')
    except Exception as e:
        logger.warning(f'Could not publish reload message: {e}')

    _release_freed_ram_to_os()
    logger.info('✓ Released freed RAM back to OS after index rebuild')


# --- Core Analysis Functions ---

def _decode_audio_with_pyav(file_path, target_sr):
    import av

    resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=target_sr)
    max_samples = int(AUDIO_LOAD_TIMEOUT * target_sr) if AUDIO_LOAD_TIMEOUT else None
    chunks = []
    total = 0
    with av.open(file_path) as container:
        if not container.streams.audio:
            return np.array([], dtype=np.float32)
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for rframe in resampler.resample(frame):
                arr = rframe.to_ndarray().reshape(-1)
                if arr.size:
                    chunks.append(arr)
                    total += arr.size
            if max_samples and total >= max_samples:
                break
        for rframe in resampler.resample(None):
            arr = rframe.to_ndarray().reshape(-1)
            if arr.size:
                chunks.append(arr)
    if not chunks:
        return np.array([], dtype=np.float32)
    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    if max_samples:
        audio = audio[:max_samples]
    return audio


def robust_load_audio_with_fallback(file_path, target_sr=16000):
    """
    Try librosa.load directly; on failure or empty signal, fall back to an
    in-process PyAV (ffmpeg) decode to mono float32 at target_sr.
    """
    name = os.path.basename(file_path)
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True, duration=AUDIO_LOAD_TIMEOUT)
        if audio is None or audio.size == 0:
            raise ValueError("Librosa returned an empty audio signal.")
        return audio, sr
    except Exception as e:
        logger.warning(f"Direct librosa load failed for {name}: {e}. Attempting PyAV fallback.")

    try:
        audio = _decode_audio_with_pyav(file_path, target_sr)
        if audio is None or audio.size == 0 or not np.any(audio):
            logger.error(f"PyAV fallback resulted in empty/silent audio for {name}.")
            return None, None
        return audio, target_sr
    except Exception as e:
        logger.error(f"PyAV fallback loading also failed for {name}: {e}")
        return None, None

def rebuild_all_indexes_task():
    """Rebuild all indexes as a standalone RQ task (enqueued on default queue)."""
    logger.info("Starting index rebuild task (enqueued as subtask)...")
    with app.app_context():
        try:
            _run_all_index_builds()
            logger.info("✅ Index rebuild task completed successfully")
            return {"status": "SUCCESS", "message": "All indexes rebuilt"}
        except Exception as e:
            logger.error(f"❌ Index rebuild task failed: {e}", exc_info=True)
            return {"status": "FAILURE", "message": str(e)}

def analyze_track(file_path, mood_labels_list, model_paths, onnx_sessions=None, return_audio=False):
    """
    Analyzes a single track using ONNX Runtime for inference.
    
    Args:
        file_path: Path to audio file
        mood_labels_list: List of mood labels
        model_paths: Dict of model paths
        onnx_sessions: Optional dict of pre-loaded ONNX sessions (for album-level reuse)
        return_audio: If True, return the loaded audio array and sample rate as part of the result.
    """
    logger.info(f"Starting analysis for: {os.path.basename(file_path)}")

    # --- 1. Load Audio and Compute Basic Features ---
    audio, sr = robust_load_audio_with_fallback(file_path, target_sr=16000)

    if audio is None or not np.any(audio) or audio.size == 0:
        logger.warning(f"Could not load a valid audio signal for {os.path.basename(file_path)} after all attempts. Skipping track.")
        return (None, None, None, None) if return_audio else (None, None)

    tempo, average_energy, musical_key, scale = extract_basic_features(audio, sr)

    # --- 2. Prepare Spectrograms ---
    try:
        final_patches = prepare_spectrogram_patches(audio, sr)
        if final_patches is None:
            logger.warning(f"Track too short to create spectrogram patches: {os.path.basename(file_path)}")
            return (None, None, None, None) if return_audio else (None, None)
    except Exception as e:
        logger.error(f"Spectrogram creation failed for {os.path.basename(file_path)}: {e}", exc_info=True)
        return (None, None, None, None) if return_audio else (None, None)

    # --- 3. Run Main Models (Embedding and Prediction) ---
    embedding_sess = None
    prediction_sess = None
    should_cleanup_sessions = False
    embeddings_per_patch = None
    mood_logits = None
    mood_probs_per_patch = None
    # Initialized here so the finally block can always reference them safely, even
    # if create_onnx_session raises before the in-try assignment is reached.
    original_embedding_sess = None
    original_prediction_sess = None

    try:
        if onnx_sessions is not None:
            embedding_sess = onnx_sessions['embedding']
            prediction_sess = onnx_sessions['prediction']
            should_cleanup_sessions = False
        else:
            provider_options = get_provider_options()
            embedding_sess = create_onnx_session(model_paths['embedding'], provider_options, label='embedding')
            prediction_sess = create_onnx_session(model_paths['prediction'], provider_options, label='prediction')
            should_cleanup_sessions = True

        # Capture originals so we can detect OOM-fallback replacements below.
        original_embedding_sess = embedding_sess
        original_prediction_sess = prediction_sess
        embedding_feed_dict = {DEFINED_TENSOR_NAMES['embedding']['input']: final_patches}
        embeddings_per_patch, embedding_sess = run_inference_with_oom_fallback(
            embedding_sess, embedding_feed_dict,
            DEFINED_TENSOR_NAMES['embedding']['output'],
            model_paths['embedding'], 'embedding',
            should_cleanup_sessions, os.path.basename(file_path),
        )
        # If GPU OOM happened and we're working with a shared (album-level)
        # session dict, rewire the dict to the new CPU session AND drop the
        # captured-original reference. Without this, the dict keeps pinning
        # the OOM'd GPU session: it leaks for the rest of the album, every
        # subsequent track re-pulls it and re-OOMs, and the new CPU session
        # we just built gets thrown away in the finally. We also drop the
        # local `original_*` ref so GC can reclaim the GPU buffers right now
        # rather than at album end.
        if embedding_sess is not original_embedding_sess:
            if onnx_sessions is not None:
                onnx_sessions['embedding'] = embedding_sess
            original_embedding_sess = None

        prediction_feed_dict = {DEFINED_TENSOR_NAMES['prediction']['input']: embeddings_per_patch}
        mood_logits, prediction_sess = run_inference_with_oom_fallback(
            prediction_sess, prediction_feed_dict,
            DEFINED_TENSOR_NAMES['prediction']['output'],
            model_paths['prediction'], 'prediction',
            should_cleanup_sessions, os.path.basename(file_path),
        )
        if prediction_sess is not original_prediction_sess:
            if onnx_sessions is not None:
                onnx_sessions['prediction'] = prediction_sess
            original_prediction_sess = None

        # Double-sigmoid to replicate old production behaviour:
        # The old Essentia-exported model (msd-msd-musicnn-1.onnx) had sigmoid built
        # into its ONNX graph, so each patch output was already a probability [0-1].
        # The old code then applied sigmoid(mean(those probs)) on top — a
        # "double sigmoid" that pushed values into the ~0.50-0.56 range.
        # The new musicnn_prediction.onnx outputs raw logits, so we replicate
        # the full old pipeline: sigmoid(logits) → mean → sigmoid.
        mood_probs_per_patch = sigmoid(mood_logits)
        final_mood_predictions = sigmoid(np.mean(mood_probs_per_patch, axis=0))

        moods = {label: float(score) for label, score in zip(mood_labels_list, final_mood_predictions)}

    except Exception as e:
        logger.error(f"Main model inference failed for {os.path.basename(file_path)}: {e}", exc_info=True)
        return (None, None, None, None) if return_audio else (None, None)
    finally:
        # Clean up sessions we own outright. When shared sessions were
        # provided and an OOM fallback occurred, the new CPU session has
        # already been written back into ``onnx_sessions`` above so the
        # album-level dict owns it — DO NOT release it here, or the next
        # track will SEGV trying to run on a destroyed session.
        if should_cleanup_sessions:
            try:
                cleanup_onnx_session(embedding_sess, "embedding")
                cleanup_onnx_session(prediction_sess, "prediction")
                cleanup_cuda_memory(force=True)
                logger.debug(f"Cleaned up sessions for {os.path.basename(file_path)}")
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")
        # Belt-and-suspenders: drop the captured originals unconditionally so
        # an OOM'd GPU session pinned only by this frame can be GC'd as the
        # function unwinds (the dict-rewire above already nulled them on the
        # happy fallback path, but if an exception interrupted between the
        # first inference and the rewire, this catches that case too).
        original_embedding_sess = None
        original_prediction_sess = None

    # --- 4. Final Aggregation for Storage ---
    processed_embeddings = np.mean(embeddings_per_patch, axis=0)
    analysis_result = {
        "tempo": tempo,
        "key": musical_key,
        "scale": scale,
        "moods": moods,
        "energy": average_energy,
    }

    return_values = (analysis_result, processed_embeddings, audio, sr) if return_audio else (analysis_result, processed_embeddings)
    try:
        if not return_audio:
            del audio, sr
        del embeddings_per_patch, final_patches, embedding_feed_dict, prediction_feed_dict, mood_logits, mood_probs_per_patch
        gc.collect()
        comprehensive_memory_cleanup(force_cuda=False, reset_onnx_pool=False)
    except Exception as cleanup_error:
        logger.warning(f"Error during final tensor cleanup: {cleanup_error}")

    return return_values


# --- RQ Task Definitions ---
def analyze_album_task(album_id, album_name, top_n_moods, parent_task_id):
    from .clap_analyzer import is_clap_available, get_or_cache_other_feature_text_embeddings

    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        initial_details = {"album_name": album_name, "log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Album analysis task started."]}
        save_task_status(current_task_id, "album_analysis", TASK_STATUS_STARTED, parent_task_id=parent_task_id, sub_type_identifier=album_id, progress=0, details=initial_details)
        tracks_analyzed_count, tracks_skipped_count, current_progress_val = 0, 0, 0
        current_task_logs = initial_details["log"]
        
        model_paths = {'embedding': EMBEDDING_MODEL_PATH, 'prediction': PREDICTION_MODEL_PATH}

        clap_label_embeddings = None

        onnx_sessions = None  # Lazy-loaded on first song that needs MusiCNN.
        # Recycle interval: 1 song if PER_SONG_MODEL_RELOAD else 20.
        recycle_interval = 1 if PER_SONG_MODEL_RELOAD else 20
        session_recycler = SessionRecycler(recycle_interval=recycle_interval)
        logger.info(f"MusiCNN session recycling: every {recycle_interval} song(s) (PER_SONG_MODEL_RELOAD={PER_SONG_MODEL_RELOAD})")

        def log_and_update_album_task(message, progress, **kwargs):
            nonlocal current_progress_val
            current_progress_val = progress
            logger.info(f"[AlbumTask-{current_task_id}-{album_name}] {message}")
            db_details = {"album_name": album_name, **kwargs}
            task_state = kwargs.get('task_state', TASK_STATUS_PROGRESS)
            if task_state == TASK_STATUS_SUCCESS:
                db_details["log"] = [f"Task completed successfully. Final status: {message}"]
            else:
                current_task_logs.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")
                db_details["log"] = current_task_logs
            if current_job:
                current_job.meta.update({'progress': progress, 'status_message': message})
                current_job.save_meta()
            save_task_status(current_task_id, "album_analysis", task_state, parent_task_id=parent_task_id, sub_type_identifier=album_id, progress=progress, details=db_details)

        try:
            log_and_update_album_task(f"Fetching tracks for album: {album_name}", 5)
            tracks = get_tracks_from_album(album_id)
            if not tracks:
                log_and_update_album_task(f"No tracks found for album: {album_name}", 100, task_state=TASK_STATUS_SUCCESS)
                return {"status": "SUCCESS", "message": f"No tracks in album {album_name}", "tracks_analyzed": 0}

            track_ids_all = [str(t['Id']) for t in tracks]
            existing_track_ids_set = _ah.get_existing_track_ids(track_ids_all)
            missing_clap_ids_set = _ah.get_missing_ids_in_table('clap_embedding', track_ids_all) if is_clap_available() else set()
            missing_lyrics_ids_set = _ah.get_missing_ids_in_table('lyrics_embedding', track_ids_all) if LYRICS_ENABLED else set()
            total_tracks_in_album = len(tracks)

            any_track_needs_musicnn = len(existing_track_ids_set) < total_tracks_in_album
            if any_track_needs_musicnn and is_clap_available():
                try:
                    clap_label_embeddings = get_or_cache_other_feature_text_embeddings(redis_conn)
                    if clap_label_embeddings:
                        logger.info(f"✓ CLAP other feature text embeddings ready ({len(clap_label_embeddings)} labels)")
                    else:
                        logger.warning("Could not load CLAP text embeddings - other_features will be zeros")
                except Exception as e:
                    logger.warning(f"Failed to load CLAP text embeddings: {e}")
            elif not any_track_needs_musicnn:
                logger.info("No track in this album needs MusiCNN - skipping CLAP text embedding load")
            else:
                logger.info("CLAP not available - other_features will be zeros")

            existing_top_moods_by_id = {}
            if LYRICS_ENABLED and existing_track_ids_set and missing_lyrics_ids_set:
                already_analyzed_needing_lyrics = [
                    tid for tid in track_ids_all
                    if tid in existing_track_ids_set and tid in missing_lyrics_ids_set
                ]
                if already_analyzed_needing_lyrics:
                    existing_top_moods_by_id = _ah.fetch_existing_top_moods(
                        already_analyzed_needing_lyrics, top_n_moods,
                    )
                    logger.info(
                        f"Prefetched prior moods for {len(existing_top_moods_by_id)}/"
                        f"{len(already_analyzed_needing_lyrics)} already-analyzed tracks "
                        f"in '{album_name}' (used as lyrics-pipeline prior)"
                    )

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
                _ah.upsert_artist_mappings_for_tracks([item], album_name=album_name)

                track_id_str = str(item['Id'])
                needs_musicnn, needs_clap, needs_lyrics = _ah.decide_track_needs(
                    track_id_str, existing_track_ids_set, missing_clap_ids_set,
                    missing_lyrics_ids_set, LYRICS_ENABLED,
                )
                track_audio, track_sr = None, None

                if not (needs_musicnn or needs_clap or needs_lyrics):
                    tracks_skipped_count += 1
                    status_parts = _ah.build_feature_status_parts(
                        is_clap_available(), LYRICS_ENABLED, include_check_marks=True,
                    )
                    logger.info(f"Skipping '{track_name_full}' - all analyses complete ({', '.join(status_parts)})")
                    continue

                needs_audio_upfront = needs_musicnn or needs_clap
                if needs_audio_upfront:
                    path = download_track(TEMP_DIR, item)
                    if not path:
                        continue
                else:
                    path = None

                def _ensure_track_download():
                    nonlocal path
                    if path is None:
                        path = download_track(TEMP_DIR, item)
                    return path

                try:
                    track_processed = False  # MusiCNN | CLAP | Lyrics produced data?

                    if needs_musicnn:
                        if onnx_sessions is None:
                            logger.info(f"Lazy-loading MusiCNN models for album: {album_name}")
                            onnx_sessions = load_musicnn_sessions(model_paths)
                        elif session_recycler.should_recycle():
                            logger.info(f"Recycling ONNX sessions after {session_recycler.get_use_count()} tracks")
                            cleanup_musicnn_sessions(onnx_sessions, context="recycle")
                            comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)
                            onnx_sessions = load_musicnn_sessions(model_paths)
                            if onnx_sessions:
                                logger.info(f"✓ Recycled {len(onnx_sessions)} MusiCNN model sessions")
                            session_recycler.mark_recycled()

                        if needs_lyrics and LYRICS_ENABLED:
                            analysis, embedding, track_audio, track_sr = analyze_track(path, MOOD_LABELS, model_paths, onnx_sessions=onnx_sessions, return_audio=True)
                        else:
                            analysis, embedding = analyze_track(path, MOOD_LABELS, model_paths, onnx_sessions=onnx_sessions)
                        if analysis is None:
                            logger.warning(f"Skipping track {track_name_full} as analysis returned None.")
                            tracks_skipped_count += 1
                            continue

                        top_moods = dict(sorted(analysis['moods'].items(), key=lambda i: i[1], reverse=True)[:top_n_moods])
                        musicnn_analysis, musicnn_embedding = analysis, embedding
                        track_processed = True
                        session_recycler.increment()
                        cleanup_cuda_memory(force=False)  # Prevent gradual VRAM accumulation.
                    else:
                        musicnn_analysis = musicnn_embedding = None
                        top_moods = existing_top_moods_by_id.get(track_id_str) or None
                        if top_moods:
                            logger.info(
                                f"SKIPPED MusiCNN for '{track_name_full}' (already analyzed); "
                                f"using {len(top_moods)} prior top moods from DB as lyrics prior: "
                                f"{list(top_moods.keys())}"
                            )
                        else:
                            logger.info(f"SKIPPED MusiCNN for '{track_name_full}' (already analyzed)")

                    clap_embedding_for_track = _ah.run_clap_for_track(
                        path, track_name_full, needs_clap, is_clap_available(), PER_SONG_MODEL_RELOAD,
                    )
                    if clap_embedding_for_track is not None:
                        track_processed = True
                    elif not needs_clap and is_clap_available():
                        logger.info("  - CLAP embedding already exists, skipping")

                    if needs_musicnn and musicnn_analysis is not None:
                        other_features = _ah.compute_other_features_str(
                            clap_embedding_for_track, needs_clap, clap_label_embeddings, item['Id'], OTHER_FEATURE_LABELS,
                        )
                        logger.info(f"SUCCESSFULLY ANALYZED '{track_name_full}' (ID: {item['Id']}):")
                        logger.info(f"  - Tempo: {musicnn_analysis['tempo']:.2f}, Energy: {musicnn_analysis['energy']:.4f}, Key: {musicnn_analysis['key']} {musicnn_analysis['scale']}")
                        logger.info(f"  - Top Moods: {top_moods}")
                        logger.info(f"  - Other Features: {other_features}")
                        _ah.persist_musicnn_results(item, musicnn_analysis, top_moods, musicnn_embedding, other_features)

                    # CLAP must be saved AFTER score (FK: clap_embedding.item_id → score.item_id).
                    _ah.persist_clap_embedding(item['Id'], clap_embedding_for_track, needs_clap)

                    if _ah.run_lyrics_for_track(item, path, track_audio, track_sr, track_name_full,
                                                needs_lyrics, LYRICS_ENABLED, robust_load_audio_with_fallback,
                                                top_moods=top_moods, download_fn=_ensure_track_download):
                        track_processed = True

                    if track_processed:
                        tracks_analyzed_count += 1
                finally:
                    if path and os.path.exists(path):
                        os.remove(path)

            cleanup_musicnn_sessions(onnx_sessions, context="album end")
            onnx_sessions = None
            cleanup_optional_models(context="album end")
            logger.info("Performing final comprehensive cleanup after album analysis")
            comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)

            summary = {"tracks_analyzed": tracks_analyzed_count, "tracks_skipped": tracks_skipped_count, "total_tracks_in_album": total_tracks_in_album}
            log_and_update_album_task(f"Album '{album_name}' analysis complete.", 100, task_state=TASK_STATUS_SUCCESS, final_summary_details=summary)
            return {"status": "SUCCESS", **summary}

        except OperationalError as e:
            logger.error(f"Database connection error during album analysis {album_id}: {e}. This job will be retried.", exc_info=True)
            err = error_manager.record(ERR_DB_CONNECTION, str(e), exc=e)
            log_and_update_album_task(f"Database connection failed for album '{album_name}'. Retrying...", current_progress_val, task_state=TASK_STATUS_FAILURE, error=err, final_summary_details={"error": str(e)})
            raise
        except Exception as e:
            logger.critical(f"Album analysis {album_id} failed: {e}", exc_info=True)
            err = error_manager.record(error_manager.classify(e, ERR_ALBUM_ANALYSIS_FAILED), str(e), exc=e)
            log_and_update_album_task(f"Failed to analyze album '{album_name}': {e}", current_progress_val, task_state=TASK_STATUS_FAILURE, error=err, final_summary_details={"error": str(e)})
            raise
        finally:
            cleanup_musicnn_sessions(onnx_sessions, context="finally")
            onnx_sessions = None
            try:
                comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)
            except Exception as e:
                logger.warning(f"Error during final comprehensive cleanup: {e}")
            cleanup_optional_models(context="finally")
            _release_freed_ram_to_os()

_AUTH_FAILURE_HINTS = (
    'wrong username', 'wrong password', 'unauthorized', 'unauthorised',
    'invalid login', 'invalid credentials', 'permission denied',
    'not authorized', 'authentication failed', '401', '403',
)


def _probe_looks_like_auth_failure(probe):
    """True when a failed test_connection result points to a credentials problem."""
    if not probe:
        return False
    if probe.get('auth_failed'):
        return True
    message = str(probe.get('error') or '').lower()
    return any(hint in message for hint in _AUTH_FAILURE_HINTS)


def _verify_media_server_reachable():
    """Probe the configured media server and raise AudioMuseError if it is failing.

    A scan that returns zero albums is ambiguous: the library may genuinely have
    nothing new, or the server rejected our credentials / was unreachable (the
    provider clients swallow those and return an empty list). Probing here turns a
    real failure into a coded error instead of a misleading "0 albums" success.
    """
    try:
        probe = mediaserver_test_connection()
    except error_manager.AudioMuseError:
        raise
    except Exception as e:
        raise error_manager.AudioMuseError(
            error_manager.classify(e, ERR_MEDIASERVER_UNREACHABLE), str(e), cause=e
        ) from e

    if probe and probe.get('ok'):
        return

    message = (probe or {}).get('error') or None
    if _probe_looks_like_auth_failure(probe):
        raise error_manager.AudioMuseError(ERR_MEDIASERVER_AUTH, message)
    raise error_manager.AudioMuseError(ERR_MEDIASERVER_UNREACHABLE, message)


def run_analysis_task(num_recent_albums, top_n_moods):
    from .clap_analyzer import is_clap_available

    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        if num_recent_albums < 0:
             logger.warning("num_recent_albums is negative, treating as 0 (all albums).")
             num_recent_albums = 0

        task_info = get_task_info_from_db(current_task_id)
        # Only truly-terminal states stop a re-entry; a prior FAILURE resumes via RQ retry.
        if task_info and task_info.get('status') in [TASK_STATUS_SUCCESS, TASK_STATUS_REVOKED]:
            return {"status": task_info.get('status'), "message": "Task already in terminal state."}

        # RAM-only dedup for this run; resume correctness comes from the DB, so never persisted.
        checked_album_ids = set()
        
        initial_details = {"message": "Fetching albums...", "log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Main analysis task started."]}

        save_task_status(current_task_id, "main_analysis", TASK_STATUS_STARTED, progress=0, details=initial_details)
        current_progress = 0
        current_task_logs = initial_details["log"]

        def log_and_update_main(message, progress, **kwargs):
            nonlocal current_progress
            current_progress = progress
            logger.info(f"[MainAnalysisTask-{current_task_id}] {message}")
            details = {**kwargs, "status_message": message}
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            task_state = kwargs.get('task_state', TASK_STATUS_PROGRESS)

            if task_state != TASK_STATUS_SUCCESS:
                current_task_logs.append(log_entry)
                if len(current_task_logs) > 200:
                    del current_task_logs[:-200]
                details["log"] = current_task_logs
            else:
                details["log"] = [f"Task completed successfully. Final status: {message}"]

            if current_job:
                current_job.meta.update({'progress': progress, 'status_message': message, 'details':details})
                current_job.save_meta()
            save_task_status(current_task_id, "main_analysis", task_state, progress=progress, details=details)

        try:
            log_and_update_main("Starting main analysis process...", 0)
            clean_temp(TEMP_DIR)
            all_albums = get_recent_albums(num_recent_albums)
            if not all_albums:
                # Distinguish an empty library from a silent auth/connection failure.
                _verify_media_server_reachable()
                log_and_update_main("No new albums to analyze.", 100, albums_found=0, task_state=TASK_STATUS_SUCCESS)
                return {"status": "SUCCESS", "message": "No new albums to analyze."}

            total_albums_to_check = len(all_albums)
            active_jobs = {}
            launched_job_ids = set()  # Track job IDs launched in THIS run only
            albums_skipped, albums_launched, albums_completed, last_rebuild_count = 0, 0, 0, 0
            albums_no_tracks = 0
            last_monitor_db_check = float('-inf')  # -inf so the first reconcile always fires (monotonic epoch is boot-relative)

            def monitor_and_clear_jobs():
                """Sync `albums_completed` with terminal RQ jobs and DB child-task statuses.

                Tries `Job.fetch` first; reconciles against DB child rows (the
                authoritative source — RQ state may be missing if the worker
                uses a different Redis namespace). Also drops stale `active_jobs`
                entries that were never launched in this run (zombies).
                Enqueues an index-rebuild subtask each time `REBUILD_INDEX_BATCH_SIZE`
                fresh albums have completed.
                """
                nonlocal albums_completed, last_rebuild_count, last_monitor_db_check
                removed = 0
                for job_id in list(active_jobs.keys()):
                    if job_id not in launched_job_ids:
                        logger.warning(f"Removing zombie job {job_id} from active_jobs")
                        del active_jobs[job_id]
                        continue
                    try:
                        job = Job.fetch(job_id, connection=redis_conn)
                        if job.is_finished or job.is_failed or job.is_canceled:
                            del active_jobs[job_id]
                            removed += 1
                    except NoSuchJobError:
                        logger.debug(f"Job {job_id} not in RQ; will reconcile via DB.")
                    except RedisTimeoutError:
                        logger.warning(f"Redis timeout fetching {job_id}; retry next loop.")
                    except Exception as e:
                        logger.warning(f"Error fetching job {job_id}: {e}; retry next loop.", exc_info=True)
                if removed:
                    albums_completed += removed

                # Throttled status-only reconcile (no details); RQ Job.fetch above drains active_jobs every poll.
                now = time.monotonic()
                if now - last_monitor_db_check >= ANALYSIS_MONITOR_DB_INTERVAL:
                    last_monitor_db_check = now
                    try:
                        terminal = {TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED}
                        child_tasks = get_child_tasks_from_db(current_task_id)
                        db_done = sum(1 for t in child_tasks
                                      if t.get('status') in terminal and t.get('task_id') in launched_job_ids)
                        if db_done != albums_completed:
                            logger.info(f"Reconciling albums_completed: RQ={albums_completed} DB={db_done} (of {len(launched_job_ids)} launched)")
                            albums_completed = db_done
                            terminal_ids = {t['task_id'] for t in child_tasks
                                            if t.get('status') in terminal and t.get('task_id') in launched_job_ids}
                            for j in list(active_jobs.keys()):
                                if j in terminal_ids:
                                    active_jobs.pop(j, None)
                    except Exception as e:
                        logger.error(f"Failed to reconcile child tasks from DB: {e}", exc_info=True)

                if albums_completed - last_rebuild_count >= REBUILD_INDEX_BATCH_SIZE:
                    log_and_update_main(
                        f"Batch of {albums_completed - last_rebuild_count} albums complete. Enqueueing index rebuild...",
                        current_progress,
                    )
                    rebuild_job = rq_queue_default.enqueue(
                        'tasks.analysis.rebuild_all_indexes_task',
                        job_id=str(uuid.uuid4()), job_timeout=-1, retry=Retry(max=3),
                    )
                    logger.info(f"⏰ Enqueued index rebuild job {rebuild_job.id} on default queue")
                    last_rebuild_count = albums_completed

            for idx, album in enumerate(all_albums):
                # Skip before polling, so a large re-scan doesn't poll per already-done album.
                if album['Id'] in checked_album_ids:
                    albums_skipped += 1
                    continue
                monitor_and_clear_jobs()
                while len(active_jobs) >= MAX_QUEUED_ANALYSIS_JOBS:
                    monitor_and_clear_jobs()
                    time.sleep(5)

                tracks = get_tracks_from_album(album['Id'])
                if not tracks:
                    albums_skipped += 1
                    albums_no_tracks += 1
                    checked_album_ids.add(album['Id'])
                    logger.info(f"Skipping album '{album.get('Name')}' (ID: {album.get('Id')}) - no tracks returned by media server.")
                    continue

                _ah.upsert_artist_mappings_for_tracks(tracks, album_name=album.get('Name'))

                try:
                    existing_count, needs_clap_analysis, needs_lyrics_analysis = _ah.compute_album_needs(
                        tracks, is_clap_available(), LYRICS_ENABLED,
                    )
                except Exception as e:
                    logger.warning(f"Failed to verify existing tracks for album '{album.get('Name')}' (ID: {album.get('Id')}): {e}")
                    checked_album_ids.add(album['Id'])
                    albums_skipped += 1
                    continue

                # Skip only when MusiCNN + every enabled feature is already complete.
                if existing_count >= len(tracks) and not (needs_clap_analysis or needs_lyrics_analysis):
                    for item in tracks:
                        _ah.refresh_track_metadata(item, album.get('Name'))
                    albums_skipped += 1
                    checked_album_ids.add(album['Id'])
                    status_parts = _ah.build_feature_status_parts(is_clap_available(), LYRICS_ENABLED)
                    logger.info(f"Skipping album '{album.get('Name')}' (ID: {album.get('Id')}) - all {existing_count}/{len(tracks)} tracks already analyzed ({' + '.join(status_parts)}).")
                    continue

                job = rq_queue_default.enqueue(
                    'tasks.analysis.analyze_album_task',
                    args=(album['Id'], album['Name'], top_n_moods, current_task_id),
                    job_id=str(uuid.uuid4()), job_timeout=-1, retry=Retry(max=3),
                )
                active_jobs[job.id] = job
                launched_job_ids.add(job.id)
                albums_launched += 1
                checked_album_ids.add(album['Id'])

                progress = 5 + int(85 * (idx / float(total_albums_to_check)))
                status_message = f"Launched: {albums_launched}. Completed: {albums_completed}/{albums_launched}. Active: {len(active_jobs)}. Skipped: {albums_skipped}/{total_albums_to_check}."
                log_and_update_main(status_message, progress,
                                    albums_to_process=albums_launched,
                                    albums_skipped=albums_skipped)

            if albums_launched == 0 and total_albums_to_check > 0 and albums_no_tracks == total_albums_to_check:
                logger.error(f"No tracks were returned for any of the {total_albums_to_check} albums; the media server library may be unreachable or empty.")
                raise error_manager.AudioMuseError(ERR_MEDIASERVER_LIBRARY, f"The media server returned no tracks for any of the {total_albums_to_check} album(s).")

            if albums_launched == 0 and albums_skipped == total_albums_to_check:
                logger.warning(f"No albums were enqueued: all {total_albums_to_check} albums were skipped (no tracks or already analyzed). Try num_recent_albums=0 or inspect media server responses.")

            while active_jobs:
                monitor_and_clear_jobs()
                progress = 5 + int(85 * ((albums_skipped + albums_completed) / float(total_albums_to_check)))
                status_message = f"Launched: {albums_launched}. Completed: {albums_completed}/{albums_launched}. Active: {len(active_jobs)}. Skipped: {albums_skipped}/{total_albums_to_check}. (Finalizing)"
                log_and_update_main(status_message, progress)
                time.sleep(5)

            log_and_update_main("Performing final index rebuild...", 95)
            try:
                _run_all_index_builds(log_fn=log_and_update_main)
            except error_manager.AudioMuseError:
                raise
            except Exception as e:
                code = ERR_INDEX_EMPTY if type(e).__name__ == "EmptyIndexError" else ERR_INDEX_BUILD
                raise error_manager.AudioMuseError(code, str(e), cause=e) from e
            logger.info('Analysis complete. CLAP text search uses default queries (no auto-regeneration).')

            final_message = f"Main analysis complete. Launched {albums_launched}, Skipped {albums_skipped}."
            log_and_update_main(final_message, 100, task_state=TASK_STATUS_SUCCESS)
            clean_temp(TEMP_DIR)
            return {"status": "SUCCESS", "message": final_message}

        except OperationalError as e:
            logger.critical(f"FATAL ERROR: Main analysis task failed due to DB connection issue: {e}", exc_info=True)
            err = error_manager.record(ERR_DB_CONNECTION, str(e), exc=e)
            log_and_update_main(f"❌ Main analysis failed due to a database connection error. The task may be retried.", current_progress, task_state=TASK_STATUS_FAILURE, error_message=str(e), error=err)
            # Re-raise to allow RQ to handle retries if configured on the task itself
            raise
        except Exception as e:
            logger.critical(f"FATAL ERROR: Analysis failed: {e}", exc_info=True)
            err = error_manager.record(error_manager.classify(e, ERR_ANALYSIS_FAILED), str(e), exc=e)
            log_and_update_main(f"❌ Main analysis failed: {e}", current_progress, task_state=TASK_STATUS_FAILURE, error_message=str(e), error=err)
            raise
