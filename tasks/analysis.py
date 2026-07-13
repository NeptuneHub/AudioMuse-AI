# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Audio analysis pipeline stage: feature extraction, embeddings and index rebuilds.

Runs as RQ jobs. run_analysis_task fans work out into per-album analyze_album_task
jobs, each of which downloads/decodes tracks, runs the MusiCNN mood/embedding models
(via analysis_helper), the optional CLAP and lyrics embedders, and persists results.
Once analysis completes it rebuilds the six similarity indexes; the clustering,
similarity and radio features consume the vectors this stage produces.

Main Features:
* analyze_track / analyze_album_task / run_analysis_task: decode audio and extract
  mood tags, MusiCNN embeddings, CLAP and lyrics embeddings, then upsert to the DB
  under the canonical embedding-hash id; a Hamming-tolerant catalogue check makes a
  song already analyzed from another server just gain a server mapping instead of a
  duplicate row.
* Per-track failures are logged and skipped so one bad track cannot abort a whole
  album; the album is still marked FAILURE afterwards so RQ retries the remainder.
* run_analysis_task skips (instead of falling back to the config default) when no
  enabled server matches the requested scope.
* Media-server reachability and auth probing before enqueuing, so a bad server aborts
  early instead of failing every child job.
* rebuild_all_indexes_task and _run_all_index_builds rebuild every similarity index
  after new embeddings land, plus periodically every REBUILD_INDEX_BATCH_SIZE
  completed albums during large runs; freed audio RAM is returned to the OS
  between albums.
"""

import os
import shutil
import numpy as np
import time
import logging
import uuid
import gc
import platform

import librosa

from rq import get_current_job, Retry
from rq.job import Job
from rq.exceptions import NoSuchJobError

from config import (
    TEMP_DIR,
    MOOD_LABELS,
    EMBEDDING_MODEL_PATH,
    PREDICTION_MODEL_PATH,
    OTHER_FEATURE_LABELS,
    MAX_QUEUED_ANALYSIS_JOBS,
    PER_SONG_MODEL_RELOAD,
    AUDIO_LOAD_TIMEOUT,
    LYRICS_ENABLED,
    ANALYSIS_MONITOR_DB_INTERVAL,
    REBUILD_INDEX_BATCH_SIZE,
)


from .mediaserver import (
    get_recent_albums,
    get_tracks_from_album,
    download_track,
    registry,
    test_connection as mediaserver_test_connection,
)
from .memory_utils import (
    cleanup_cuda_memory,
    cleanup_onnx_session,
    SessionRecycler,
    comprehensive_memory_cleanup,
)

from flask_app import app
from app_helper import (
    redis_conn,
    rq_queue_default,
    get_db,
    save_task_status,
    get_task_info_from_db,
    get_task_statuses,
    build_and_store_map_projection,
    build_and_store_artist_projection,
    TASK_STATUS_STARTED,
    TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
    TASK_STATUS_REVOKED,
)
from database import get_child_tasks_from_db, get_failed_child_summary

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

from . import analysis_helper as _ah
from .analysis_helper import (  # noqa: F401
    DEFINED_TENSOR_NAMES,
    sigmoid,
    extract_basic_features,
    prepare_spectrogram_patches,
    resolve_providers,
    create_onnx_session,
    load_musicnn_sessions,
    cleanup_musicnn_sessions,
    cleanup_optional_models,
    run_inference_with_oom_fallback,
)


from psycopg2 import OperationalError
from redis.exceptions import TimeoutError as RedisTimeoutError

logger = logging.getLogger(__name__)


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


def _run_all_index_builds(log_fn=None, progress_start=95, progress_end=98):
    """Rebuild every similarity index, reporting each step through ``log_fn``.

    The progress range is the caller's: an analysis run has these builds as its
    tail (95-98%), while the standalone rebuild task owns the whole bar.
    """
    from .ivf_manager import build_and_store_ivf_index
    from .clap_text_search import build_and_store_clap_index
    from .lyrics_manager import build_and_store_lyrics_index, build_and_store_lyrics_axes_index
    from .sem_grove_manager import build_and_store_sem_grove_index
    from .artist_gmm_manager import build_and_store_artist_index

    steps = (
        ("IVF index rebuilt", "Building IVF audio index...",
         lambda: build_and_store_ivf_index(get_db()), True),
        ("CLAP text search index", "Building CLAP text search index...",
         lambda: build_and_store_clap_index(get_db()), False),
        ("Lyrics search index", "Building lyrics search index...",
         lambda: build_and_store_lyrics_index(get_db()), False),
        ("Lyrics axes index", "Building lyrics axes index...",
         lambda: build_and_store_lyrics_axes_index(get_db()), False),
        ("SemGrove merged index rebuilt", "Building SemGrove merged index...",
         lambda: build_and_store_sem_grove_index(get_db()), False),
        ("Artist similarity index rebuilt", "Building artist similarity index...",
         lambda: build_and_store_artist_index(get_db()), False),
        ("Song map projection rebuilt", "Building song map projection...",
         lambda: build_and_store_map_projection('main_map'), False),
        ("Artist component projection rebuilt", "Building artist component projection...",
         lambda: build_and_store_artist_projection('artist_map'), False),
    )
    span = max(0, progress_end - progress_start)

    if log_fn:
        try:
            log_fn("Rebuilding similarity indexes...", progress_start)
        except Exception:
            pass
    for index, (label, banner, build, fatal) in enumerate(steps):
        if log_fn:
            try:
                log_fn(
                    f"{banner} ({index + 1}/{len(steps)})",
                    progress_start + (span * index) // len(steps),
                )
            except Exception:
                pass
        try:
            build()
            logger.info(f"OK {label}")
        except Exception as e:
            logger.warning(f"Failed to build/store {label}: {e}")
            if fatal:
                raise
        finally:
            gc.collect()
    try:
        redis_conn.publish('index-updates', 'reload')
        logger.info('OK Published reload message to Flask container')
    except Exception as e:
        logger.warning(f'Could not publish reload message: {e}')

    _release_freed_ram_to_os()
    logger.info('OK Released freed RAM back to OS after index rebuild')


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
    except Exception:
        logger.exception(f"PyAV fallback loading also failed for {name}")
        return None, None


def rebuild_all_indexes_task():
    """Rebuild every similarity index, reporting progress like any other task.

    Minutes of work that the user otherwise has no way to see, so it reports into
    task_status exactly as analysis/clustering/cleaning do rather than running
    invisibly and looking like it never started. The startup migration does NOT
    enqueue this: a relabel renames tracks without moving a vector, so it
    repoints the indexes in place instead.
    """
    logger.info("Starting index rebuild task...")
    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        initial_details = {
            "message": "Rebuilding similarity indexes...",
            "log": [
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Index rebuild started."
            ],
        }
        save_task_status(
            current_task_id, "index_rebuild", TASK_STATUS_STARTED,
            progress=0, details=initial_details,
        )
        task_logs = initial_details["log"]

        def log_and_update(message, progress, **kwargs):
            logger.info(f"[IndexRebuild-{current_task_id}] {message}")
            task_state = kwargs.get('task_state', TASK_STATUS_PROGRESS)
            details = {**kwargs, "status_message": message, "message": message}
            task_logs.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")
            details["log"] = task_logs
            if current_job:
                current_job.meta.update(
                    {'progress': progress, 'status_message': message, 'details': details}
                )
                current_job.save_meta()
            save_task_status(
                current_task_id, "index_rebuild", task_state,
                progress=progress, details=details,
            )

        try:
            _run_all_index_builds(
                log_fn=log_and_update, progress_start=0, progress_end=99
            )
        except Exception:
            logger.exception("X Index rebuild task failed")
            log_and_update(
                "Index rebuild failed. Check the container logs for details.",
                100, task_state=TASK_STATUS_FAILURE,
            )
            return {"status": "FAILURE", "message": "Index rebuild failed"}
        log_and_update(
            "All similarity indexes rebuilt.", 100, task_state=TASK_STATUS_SUCCESS
        )
        logger.info("OK Index rebuild task completed successfully")
        return {"status": "SUCCESS", "message": "All indexes rebuilt"}


def analyze_track(file_path, mood_labels_list, model_paths, onnx_sessions=None, return_audio=False):
    logger.info(f"Starting analysis for: {os.path.basename(file_path)}")

    audio, sr = robust_load_audio_with_fallback(file_path, target_sr=16000)

    if audio is None or not np.any(audio) or audio.size == 0:
        logger.warning(
            f"Could not load a valid audio signal for {os.path.basename(file_path)} after all attempts. Skipping track."
        )
        return (None, None, None, None) if return_audio else (None, None)

    tempo, average_energy, musical_key, scale = extract_basic_features(audio, sr)

    try:
        final_patches = prepare_spectrogram_patches(audio, sr)
        if final_patches is None:
            logger.warning(
                f"Track too short to create spectrogram patches: {os.path.basename(file_path)}"
            )
            return (None, None, None, None) if return_audio else (None, None)
    except Exception:
        logger.exception(
            f"Spectrogram creation failed for {os.path.basename(file_path)}"
        )
        return (None, None, None, None) if return_audio else (None, None)

    embedding_sess = None
    prediction_sess = None
    should_cleanup_sessions = False
    embeddings_per_patch = None
    mood_logits = None
    mood_probs_per_patch = None
    original_embedding_sess = None
    original_prediction_sess = None

    try:
        if onnx_sessions is not None:
            embedding_sess = onnx_sessions['embedding']
            prediction_sess = onnx_sessions['prediction']
            should_cleanup_sessions = False
        else:
            provider_options = resolve_providers()
            embedding_sess = create_onnx_session(
                model_paths['embedding'], provider_options, label='embedding'
            )
            prediction_sess = create_onnx_session(
                model_paths['prediction'], provider_options, label='prediction'
            )
            should_cleanup_sessions = True

        original_embedding_sess = embedding_sess
        original_prediction_sess = prediction_sess
        embedding_feed_dict = {DEFINED_TENSOR_NAMES['embedding']['input']: final_patches}
        embeddings_per_patch, embedding_sess = run_inference_with_oom_fallback(
            embedding_sess,
            embedding_feed_dict,
            DEFINED_TENSOR_NAMES['embedding']['output'],
            model_paths['embedding'],
            'embedding',
            os.path.basename(file_path),
        )
        if embedding_sess is not original_embedding_sess:
            if onnx_sessions is not None:
                onnx_sessions['embedding'] = embedding_sess
            original_embedding_sess = None

        prediction_feed_dict = {DEFINED_TENSOR_NAMES['prediction']['input']: embeddings_per_patch}
        mood_logits, prediction_sess = run_inference_with_oom_fallback(
            prediction_sess,
            prediction_feed_dict,
            DEFINED_TENSOR_NAMES['prediction']['output'],
            model_paths['prediction'],
            'prediction',
            os.path.basename(file_path),
        )
        if prediction_sess is not original_prediction_sess:
            if onnx_sessions is not None:
                onnx_sessions['prediction'] = prediction_sess
            original_prediction_sess = None

        mood_probs_per_patch = sigmoid(mood_logits)
        final_mood_predictions = sigmoid(np.mean(mood_probs_per_patch, axis=0))

        moods = {
            label: float(score) for label, score in zip(mood_labels_list, final_mood_predictions)
        }

    except Exception:
        logger.exception(
            f"Main model inference failed for {os.path.basename(file_path)}"
        )
        return (None, None, None, None) if return_audio else (None, None)
    finally:
        if should_cleanup_sessions:
            try:
                cleanup_onnx_session(embedding_sess, "embedding")
                cleanup_onnx_session(prediction_sess, "prediction")
                cleanup_cuda_memory(force=True)
                logger.debug(f"Cleaned up sessions for {os.path.basename(file_path)}")
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")
        original_embedding_sess = None
        original_prediction_sess = None

    processed_embeddings = np.mean(embeddings_per_patch, axis=0)
    analysis_result = {
        "tempo": tempo,
        "key": musical_key,
        "scale": scale,
        "moods": moods,
        "energy": average_energy,
    }

    return_values = (
        (analysis_result, processed_embeddings, audio, sr)
        if return_audio
        else (analysis_result, processed_embeddings)
    )
    try:
        if not return_audio:
            del audio, sr
        del (
            embeddings_per_patch,
            final_patches,
            embedding_feed_dict,
            prediction_feed_dict,
            mood_logits,
            mood_probs_per_patch,
        )
        gc.collect()
        comprehensive_memory_cleanup(force_cuda=False, reset_onnx_pool=False)
    except Exception as cleanup_error:
        logger.warning(f"Error during final tensor cleanup: {cleanup_error}")

    return return_values


def _bind_server_context(server_id):
    """Resolve a registry server context inside an app context (workers have none)."""
    if not server_id:
        return None
    with app.app_context():
        return registry.context_for(server_id)


def analyze_album_task(album_id, album_name, top_n_moods, parent_task_id, server_id=None):
    """Run one album under an explicitly bound media-server context."""
    from tasks.mediaserver import context as server_context

    with server_context.use_server(_bind_server_context(server_id)):
        return _analyze_album_task_impl(album_id, album_name, top_n_moods, parent_task_id)


def _analyze_album_task_impl(album_id, album_name, top_n_moods, parent_task_id):
    from .clap_analyzer import is_clap_available, get_or_cache_other_feature_text_embeddings
    from .mediaserver import context as server_context

    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        initial_details = {
            "album_name": album_name,
            "log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Album analysis task started."],
        }
        save_task_status(
            current_task_id,
            "album_analysis",
            TASK_STATUS_STARTED,
            parent_task_id=parent_task_id,
            sub_type_identifier=album_id,
            progress=0,
            details=initial_details,
        )
        tracks_analyzed_count, tracks_skipped_count, current_progress_val = 0, 0, 0
        current_task_logs = initial_details["log"]

        model_paths = {'embedding': EMBEDDING_MODEL_PATH, 'prediction': PREDICTION_MODEL_PATH}

        clap_label_embeddings = None

        onnx_sessions = None
        recycle_interval = 1 if PER_SONG_MODEL_RELOAD else 20
        session_recycler = SessionRecycler(recycle_interval=recycle_interval)
        logger.info(
            f"MusiCNN session recycling: every {recycle_interval} song(s) (PER_SONG_MODEL_RELOAD={PER_SONG_MODEL_RELOAD})"
        )

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
                if len(current_task_logs) > 50:
                    del current_task_logs[:-50]
                db_details["log"] = current_task_logs
            if current_job:
                current_job.meta.update({'progress': progress, 'status_message': message})
                current_job.save_meta()
            save_task_status(
                current_task_id,
                "album_analysis",
                task_state,
                parent_task_id=parent_task_id,
                sub_type_identifier=album_id,
                progress=progress,
                details=db_details,
            )

        try:
            log_and_update_album_task(f"Fetching tracks for album: {album_name}", 5)
            tracks = get_tracks_from_album(album_id)
            if not tracks:
                log_and_update_album_task(
                    f"No tracks found for album: {album_name}", 100, task_state=TASK_STATUS_SUCCESS
                )
                return {
                    "status": "SUCCESS",
                    "message": f"No tracks in album {album_name}",
                    "tracks_analyzed": 0,
                }

            is_default_server = (
                server_context.active_server_id() or registry.get_default_server_id()
            ) == registry.get_default_server_id()

            _ah.attach_catalog_item_ids(tracks)
            track_ids_all = [_ah.catalog_item_id(t) for t in tracks]
            existing_track_ids_set = _ah.get_existing_track_ids(track_ids_all)
            missing_clap_ids_set = (
                _ah.get_missing_ids_in_table('clap_embedding', track_ids_all)
                if is_clap_available()
                else set()
            )
            missing_lyrics_ids_set = (
                _ah.get_missing_ids_in_table('lyrics_embedding', track_ids_all)
                if LYRICS_ENABLED
                else set()
            )
            total_tracks_in_album = len(tracks)

            logger.info(
                "Feature plan for album '%s': MusiCNN=%d, DCLAP=%d, Lyrics=%d of %d tracks.",
                album_name,
                total_tracks_in_album - len(existing_track_ids_set),
                len(missing_clap_ids_set),
                len(missing_lyrics_ids_set),
                total_tracks_in_album,
            )
            fingerprint_index = None

            any_track_needs_musicnn = len(existing_track_ids_set) < total_tracks_in_album
            if any_track_needs_musicnn and is_clap_available():
                try:
                    clap_label_embeddings = get_or_cache_other_feature_text_embeddings(redis_conn)
                    if clap_label_embeddings:
                        logger.info(
                            f"OK CLAP other feature text embeddings ready ({len(clap_label_embeddings)} labels)"
                        )
                    else:
                        logger.warning(
                            "Could not load CLAP text embeddings - other_features will be zeros"
                        )
                except Exception as e:
                    logger.warning(f"Failed to load CLAP text embeddings: {e}")
            elif not any_track_needs_musicnn:
                logger.info(
                    "No track in this album needs MusiCNN - skipping CLAP text embedding load"
                )
            else:
                logger.info("CLAP not available - other_features will be zeros")

            existing_top_moods_by_id = {}
            if LYRICS_ENABLED and existing_track_ids_set and missing_lyrics_ids_set:
                already_analyzed_needing_lyrics = [
                    tid
                    for tid in track_ids_all
                    if tid in existing_track_ids_set and tid in missing_lyrics_ids_set
                ]
                if already_analyzed_needing_lyrics:
                    existing_top_moods_by_id = _ah.fetch_existing_top_moods(
                        already_analyzed_needing_lyrics,
                        top_n_moods,
                    )
                    logger.info(
                        f"Prefetched prior moods for {len(existing_top_moods_by_id)}/"
                        f"{len(already_analyzed_needing_lyrics)} already-analyzed tracks "
                        f"in '{album_name}' (used as lyrics-pipeline prior)"
                    )

            _ah.upsert_artist_mappings_for_tracks(tracks, album_name=album_name)

            pending_track_maps = {}
            failed_tracks = []
            last_revocation_check = float('-inf')

            def revoked():
                """Has this album job (or its parent) been cancelled?

                One round-trip for both ids, throttled: a track takes seconds of
                MusiCNN, so re-reading the status per track bought nothing and
                cost two queries each, including for tracks that were skipped.
                """
                nonlocal last_revocation_check
                if not current_job:
                    return False
                now = time.monotonic()
                if now - last_revocation_check < ANALYSIS_MONITOR_DB_INTERVAL:
                    return False
                last_revocation_check = now
                statuses = get_task_statuses([current_task_id, parent_task_id])
                if statuses.get(current_task_id) == TASK_STATUS_REVOKED:
                    return True
                parent_status = statuses.get(parent_task_id) if parent_task_id else None
                return parent_status in (TASK_STATUS_REVOKED, TASK_STATUS_FAILURE)

            for idx, item in enumerate(tracks, 1):
                if revoked():
                    log_and_update_album_task(
                        f"Stopping album analysis for '{album_name}' due to parent/self revocation.",
                        current_progress_val,
                        task_state=TASK_STATUS_REVOKED,
                    )
                    return {"status": "REVOKED"}

                track_name_full = f"{item['Name']} by {item.get('AlbumArtist', 'Unknown')}"
                track_id_str = _ah.catalog_item_id(item)
                needs_musicnn, needs_clap, needs_lyrics = _ah.decide_track_needs(
                    track_id_str,
                    existing_track_ids_set,
                    missing_clap_ids_set,
                    missing_lyrics_ids_set,
                    LYRICS_ENABLED,
                )
                track_audio, track_sr = None, None

                if not (needs_musicnn or needs_clap or needs_lyrics):
                    tracks_skipped_count += 1
                    status_parts = _ah.build_feature_status_parts(
                        is_clap_available(),
                        LYRICS_ENABLED,
                        include_check_marks=True,
                    )
                    logger.info(
                        f"Skipping '{track_name_full}' - all analyses complete ({', '.join(status_parts)})"
                    )
                    continue

                progress = 10 + int(85 * (idx / float(total_tracks_in_album)))
                log_and_update_album_task(
                    f"Analyzing track: {track_name_full} ({idx}/{total_tracks_in_album})",
                    progress,
                    current_track_name=track_name_full,
                )

                path = None
                try:
                    needs_audio_upfront = needs_musicnn or needs_clap
                    if needs_audio_upfront:
                        path = download_track(TEMP_DIR, item)
                        if not path:
                            raise RuntimeError(
                                f"Failed to download required audio for {track_name_full}"
                            )

                    def _ensure_track_download(item=item):
                        nonlocal path
                        if path is None:
                            path = download_track(TEMP_DIR, item)
                        return path

                    track_processed = False

                    if needs_musicnn:
                        if onnx_sessions is None:
                            logger.info(f"Lazy-loading MusiCNN models for album: {album_name}")
                            onnx_sessions = load_musicnn_sessions(model_paths)
                        elif session_recycler.should_recycle():
                            logger.info(
                                f"Recycling ONNX sessions after {session_recycler.get_use_count()} tracks"
                            )
                            cleanup_musicnn_sessions(onnx_sessions, context="recycle")
                            comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)
                            onnx_sessions = load_musicnn_sessions(model_paths)
                            if onnx_sessions:
                                logger.info(
                                    f"OK Recycled {len(onnx_sessions)} MusiCNN model sessions"
                                )
                            session_recycler.mark_recycled()

                        if needs_lyrics and LYRICS_ENABLED:
                            analysis, embedding, track_audio, track_sr = analyze_track(
                                path,
                                MOOD_LABELS,
                                model_paths,
                                onnx_sessions=onnx_sessions,
                                return_audio=True,
                            )
                        else:
                            analysis, embedding = analyze_track(
                                path, MOOD_LABELS, model_paths, onnx_sessions=onnx_sessions
                            )
                        if analysis is None:
                            raise RuntimeError(
                                f"MusiCNN analysis returned no result for {track_name_full}"
                            )

                        top_moods = dict(
                            sorted(analysis['moods'].items(), key=lambda i: i[1], reverse=True)[
                                :top_n_moods
                            ]
                        )
                        musicnn_analysis, musicnn_embedding = analysis, embedding
                        track_processed = True
                        session_recycler.increment()
                        cleanup_cuda_memory(force=False)

                        source_server_id = (
                            server_context.active_server_id()
                            or registry.get_default_server_id()
                        )
                        provider_id = str(item.get('Id') or item.get('id'))
                        if fingerprint_index is None:
                            fingerprint_index = _ah.load_fingerprint_index()
                        kind, resolved_id = fingerprint_index.resolve(musicnn_embedding)
                        if resolved_id is None:
                            if source_server_id:
                                pending_track_maps.setdefault(source_server_id, {})[
                                    track_id_str
                                ] = (provider_id, 'analysis')
                            logger.warning(
                                "Embedding signature unavailable for '%s'; keeping "
                                "provider id %s and mapping it, so it is not "
                                "re-analyzed on every future run.",
                                track_name_full,
                                track_id_str,
                            )
                        elif kind == 'existing':
                            if source_server_id:
                                pending_track_maps.setdefault(source_server_id, {})[
                                    resolved_id
                                ] = (provider_id, 'fingerprint')
                            logger.info(
                                "Embedding signature + cosine matched '%s' to existing "
                                "catalogue id %s; marked it for this server and "
                                "skipped the duplicate persist.",
                                track_name_full,
                                resolved_id,
                            )
                            item['_catalog_item_id'] = resolved_id
                            track_id_str = resolved_id
                            musicnn_analysis = musicnn_embedding = None
                            needs_clap = needs_clap and bool(
                                _ah.get_missing_ids_in_table('clap_embedding', [resolved_id])
                            )
                            needs_lyrics = needs_lyrics and bool(
                                _ah.get_missing_ids_in_table('lyrics_embedding', [resolved_id])
                            )
                            if not (needs_clap or needs_lyrics):
                                tracks_analyzed_count += 1
                                continue
                            logger.info(
                                "Existing catalogue row %s still needs%s%s; running the "
                                "missing stages for it.",
                                resolved_id,
                                " CLAP" if needs_clap else "",
                                " lyrics" if needs_lyrics else "",
                            )
                        else:
                            item['_catalog_item_id'] = resolved_id
                            track_id_str = resolved_id
                            if source_server_id:
                                pending_track_maps.setdefault(source_server_id, {})[
                                    resolved_id
                                ] = (provider_id, 'fingerprint')
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
                            logger.info(
                                f"SKIPPED MusiCNN for '{track_name_full}' (already analyzed)"
                            )

                    clap_embedding_for_track = _ah.run_clap_for_track(
                        path,
                        track_name_full,
                        needs_clap,
                        is_clap_available(),
                        PER_SONG_MODEL_RELOAD,
                    )
                    if clap_embedding_for_track is not None:
                        track_processed = True
                    elif needs_clap:
                        raise RuntimeError(
                            f"DCLAP analysis returned no embedding for {track_name_full}"
                        )
                    elif not needs_clap and is_clap_available():
                        logger.info("  - CLAP embedding already exists, skipping")

                    if needs_musicnn and musicnn_analysis is not None:
                        other_features = _ah.compute_other_features_str(
                            clap_embedding_for_track,
                            needs_clap,
                            clap_label_embeddings,
                            track_id_str,
                            OTHER_FEATURE_LABELS,
                        )
                        logger.info(
                            f"SUCCESSFULLY ANALYZED '{track_name_full}' (ID: {item['Id']}):"
                        )
                        logger.info(
                            f"  - Tempo: {musicnn_analysis['tempo']:.2f}, Energy: {musicnn_analysis['energy']:.4f}, Key: {musicnn_analysis['key']} {musicnn_analysis['scale']}"
                        )
                        logger.info(f"  - Top Moods: {top_moods}")
                        logger.info(f"  - Other Features: {other_features}")
                        _ah.persist_musicnn_results(
                            item,
                            musicnn_analysis,
                            top_moods,
                            musicnn_embedding,
                            other_features,
                            is_default_server=is_default_server,
                        )

                    _ah.persist_clap_embedding(
                        track_id_str, clap_embedding_for_track, needs_clap
                    )

                    lyrics_saved = _ah.run_lyrics_for_track(
                        item,
                        path,
                        track_audio,
                        track_sr,
                        track_name_full,
                        needs_lyrics,
                        LYRICS_ENABLED,
                        robust_load_audio_with_fallback,
                        top_moods=top_moods,
                        download_fn=_ensure_track_download,
                    )
                    if lyrics_saved:
                        track_processed = True
                    elif needs_lyrics:
                        raise RuntimeError(
                            f"Lyrics analysis returned no embedding for {track_name_full}"
                        )

                    if track_processed:
                        _ah.run_song_analyzed_hook(
                            item, path, musicnn_analysis, musicnn_embedding,
                            clap_embedding_for_track, top_moods, album_id, album_name,
                            parent_task_id,
                        )
                        tracks_analyzed_count += 1
                except OperationalError:
                    raise
                except Exception as e:
                    logger.exception(
                        f"Track analysis failed for '{track_name_full}'; continuing with the next track."
                    )
                    failed_tracks.append(f"{track_name_full}: {e}")
                finally:
                    if path and os.path.exists(path):
                        os.remove(path)

            map_flush_errors = []
            for map_server_id, pending in pending_track_maps.items():
                try:
                    ready_ids = _ah.get_existing_track_ids(list(pending))
                    filtered = {cid: pending[cid] for cid in pending if cid in ready_ids}
                    if filtered:
                        registry.upsert_track_maps(map_server_id, filtered)
                except Exception:
                    logger.exception(
                        "Failed to persist %d pending track map(s) for server %s in album '%s'",
                        len(pending),
                        map_server_id,
                        album_name,
                    )
                    map_flush_errors.append(str(map_server_id))

            cleanup_musicnn_sessions(onnx_sessions, context="album end")
            onnx_sessions = None
            cleanup_optional_models(context="album end")
            logger.info("Performing final comprehensive cleanup after album analysis")
            comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)

            failure_reasons = []
            if failed_tracks:
                preview = "; ".join(failed_tracks[:3])
                failure_reasons.append(
                    f"{len(failed_tracks)}/{total_tracks_in_album} tracks failed analysis; "
                    f"first failures: {preview}"
                )
            if map_flush_errors:
                failure_reasons.append(
                    f"track-server map flush failed for server(s): {', '.join(map_flush_errors)}"
                )
            if failure_reasons:
                raise RuntimeError(" | ".join(failure_reasons))

            summary = {
                "tracks_analyzed": tracks_analyzed_count,
                "tracks_skipped": tracks_skipped_count,
                "total_tracks_in_album": total_tracks_in_album,
            }
            log_and_update_album_task(
                f"Album '{album_name}' analysis complete.",
                100,
                task_state=TASK_STATUS_SUCCESS,
                final_summary_details=summary,
            )
            return {"status": "SUCCESS", **summary}

        except OperationalError as e:
            logger.exception(
                f"Database connection error during album analysis {album_id}. This job will be retried."
            )
            err = error_manager.record(ERR_DB_CONNECTION, str(e))
            log_and_update_album_task(
                f"Database connection failed for album '{album_name}'. Retrying...",
                current_progress_val,
                task_state=TASK_STATUS_FAILURE,
                error=err,
            )
            raise
        except Exception as e:
            logger.critical(f"Album analysis {album_id} failed: {e}", exc_info=True)
            err = error_manager.record(
                error_manager.classify(e, ERR_ALBUM_ANALYSIS_FAILED), str(e)
            )
            log_and_update_album_task(
                f"Failed to analyze album '{album_name}': {e}",
                current_progress_val,
                task_state=TASK_STATUS_FAILURE,
                error=err,
            )
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
    'wrong username',
    'wrong password',
    'unauthorized',
    'unauthorised',
    'invalid login',
    'invalid credentials',
    'permission denied',
    'not authorized',
    'authentication failed',
    '401',
    '403',
)


def _probe_looks_like_auth_failure(probe):
    if not probe:
        return False
    if probe.get('auth_failed'):
        return True
    message = str(probe.get('error') or '').lower()
    return any(hint in message for hint in _AUTH_FAILURE_HINTS)


def _verify_media_server_reachable():
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


def run_analysis_server_task(
    num_recent_albums,
    top_n_moods,
    server_id=None,
    finalize_indexes=True,
    task_id=None,
    progress_base=0.0,
    progress_span=100.0,
    final_phase=True,
    albums=None,
    albums_offset=0,
    albums_total=None,
):
    """Analyze one server while persisting everything by canonical catalogue id."""
    from tasks.mediaserver import context as server_context

    with server_context.use_server(_bind_server_context(server_id)):
        return _run_analysis_server_task_impl(
            num_recent_albums,
            top_n_moods,
            server_id=server_id,
            finalize_indexes=finalize_indexes,
            task_id=task_id,
            progress_base=progress_base,
            progress_span=progress_span,
            final_phase=final_phase,
            albums=albums,
            albums_offset=albums_offset,
            albums_total=albums_total,
        )


def _run_analysis_server_task_impl(
    num_recent_albums,
    top_n_moods,
    server_id=None,
    finalize_indexes=True,
    task_id=None,
    progress_base=0.0,
    progress_span=100.0,
    final_phase=True,
    albums=None,
    albums_offset=0,
    albums_total=None,
):
    from .clap_analyzer import is_clap_available

    current_job = get_current_job(redis_conn)
    current_task_id = task_id or (current_job.id if current_job else str(uuid.uuid4()))

    with app.app_context():
        if num_recent_albums < 0:
            logger.warning("num_recent_albums is negative, treating as 0 (all albums).")
            num_recent_albums = 0

        task_info = get_task_info_from_db(current_task_id)
        if task_info and task_info.get('status') in [TASK_STATUS_SUCCESS, TASK_STATUS_REVOKED]:
            return {"status": task_info.get('status'), "message": "Task already in terminal state."}

        checked_album_ids = set()

        initial_details = {
            "message": "Fetching albums...",
            "log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Main analysis task started."],
        }

        save_task_status(
            current_task_id,
            "main_analysis",
            TASK_STATUS_STARTED,
            progress=int(progress_base),
            details=initial_details,
        )
        current_progress = 0
        current_task_logs = initial_details["log"]

        def log_and_update_main(message, progress, **kwargs):
            nonlocal current_progress
            current_progress = progress
            logger.info(f"[MainAnalysisTask-{current_task_id}] {message}")
            details = {**kwargs, "status_message": message}
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            task_state = kwargs.get('task_state', TASK_STATUS_PROGRESS)
            if not final_phase and task_state in (TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE):
                task_state = TASK_STATUS_PROGRESS
                details['phase_task_state'] = kwargs.get('task_state')
            progress = int(progress_base + (progress or 0) * progress_span / 100.0)

            if task_state != TASK_STATUS_SUCCESS:
                current_task_logs.append(log_entry)
                if len(current_task_logs) > 200:
                    del current_task_logs[:-200]
                details["log"] = current_task_logs
            else:
                details["log"] = [f"Task completed successfully. Final status: {message}"]

            if current_job:
                current_job.meta.update(
                    {'progress': progress, 'status_message': message, 'details': details}
                )
                current_job.save_meta()
            save_task_status(
                current_task_id, "main_analysis", task_state, progress=progress, details=details
            )

        try:
            log_and_update_main("Starting main analysis process...", 0)
            clean_temp(TEMP_DIR)
            all_albums = albums if albums is not None else get_recent_albums(num_recent_albums)
            if not all_albums:
                _verify_media_server_reachable()
                log_and_update_main(
                    "No new albums to analyze.", 100, albums_found=0, task_state=TASK_STATUS_SUCCESS
                )
                return {"status": "SUCCESS", "message": "No new albums to analyze."}

            total_albums_to_check = len(all_albums)
            reported_total = albums_total or total_albums_to_check
            clap_available = is_clap_available()
            work_map = _ah.load_server_work_map(
                server_id or registry.get_default_server_id(),
                server_id is None or server_id == registry.get_default_server_id(),
                clap_available,
                LYRICS_ENABLED,
            )
            done_bits = _ah.work_done_bits(clap_available, LYRICS_ENABLED)
            logger.info(
                "Work map for this server: %d provider tracks already known.",
                len(work_map),
            )
            # Every server phase of a union run files its album children under the
            # SAME parent task id, so the failure count is cumulative. Take a
            # baseline now (earlier phases are finished) and report only the
            # failures THIS server produced - otherwise one bad server marks every
            # healthy one that follows it as failed too.
            baseline_failed_count, _baseline_errors = get_failed_child_summary(current_task_id)
            active_jobs = {}
            launched_job_ids = set()
            albums_skipped, albums_launched, albums_completed = 0, 0, 0
            last_rebuild_count = 0
            albums_no_tracks = 0
            albums_needing_musicnn = 0
            albums_needing_clap = 0
            albums_needing_lyrics = 0
            songs_seen = 0
            songs_done = 0
            songs_to_analyze = 0
            last_monitor_db_check = float('-inf')
            last_status_report = float('-inf')
            last_status_snapshot = None

            def monitor_and_clear_jobs():
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
                        logger.warning(
                            f"Error fetching job {job_id}: {e}; retry next loop.", exc_info=True
                        )
                if removed:
                    albums_completed += removed

                now = time.monotonic()
                if now - last_monitor_db_check >= ANALYSIS_MONITOR_DB_INTERVAL:
                    last_monitor_db_check = now
                    try:
                        terminal = {TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED}
                        child_tasks = get_child_tasks_from_db(current_task_id)
                        db_done = sum(
                            1
                            for t in child_tasks
                            if t.get('status') in terminal and t.get('task_id') in launched_job_ids
                        )
                        if db_done != albums_completed:
                            logger.info(
                                f"Reconciling albums_completed: RQ={albums_completed} DB={db_done} (of {len(launched_job_ids)} launched)"
                            )
                            albums_completed = db_done
                            terminal_ids = {
                                t['task_id']
                                for t in child_tasks
                                if t.get('status') in terminal
                                and t.get('task_id') in launched_job_ids
                            }
                            for j in list(active_jobs.keys()):
                                if j in terminal_ids:
                                    active_jobs.pop(j, None)
                    except Exception:
                        logger.exception("Failed to reconcile child tasks from DB")

                # Gated exactly like the final rebuild below: a union run sets
                # finalize_indexes=False on every phase and builds the indexes ONCE
                # at the end, so a per-phase mid-run rebuild would be thrown away
                # by the next phase and by the consolidated build.
                if (
                    finalize_indexes
                    and albums_completed - last_rebuild_count >= REBUILD_INDEX_BATCH_SIZE
                ):
                    log_and_update_main(
                        f"Batch of {albums_completed - last_rebuild_count} albums complete. Enqueueing index rebuild...",
                        current_progress,
                    )
                    rebuild_job = rq_queue_default.enqueue(
                        'tasks.analysis.rebuild_all_indexes_task',
                        job_id=str(uuid.uuid4()),
                        job_timeout=-1,
                        retry=Retry(max=3),
                    )
                    logger.info(f"Enqueued index rebuild job {rebuild_job.id} on default queue")
                    last_rebuild_count = albums_completed

            def report_progress(force=False):
                """Write the phase's live status, throttled to one DB write per 5s.

                One line: albums DONE (analyzed + skipped) out of the total across
                every configured server, so the number keeps climbing across the
                phases of a union run instead of restarting per server. The album
                and song breakdowns stay in ``details`` for anything that wants
                them, and in the container log.
                """
                nonlocal last_status_report, last_status_snapshot
                snapshot = (
                    albums_launched,
                    albums_completed,
                    len(active_jobs),
                    albums_skipped,
                    songs_seen,
                )
                now = time.monotonic()
                if not force and (
                    snapshot == last_status_snapshot or now - last_status_report < 5
                ):
                    return
                last_status_report = now
                last_status_snapshot = snapshot
                done = albums_skipped + albums_completed
                progress = 5 + int(85 * (done / float(total_albums_to_check)))
                log_and_update_main(
                    f"Albums {albums_offset + done}/{reported_total}",
                    progress,
                    albums_to_process=albums_launched,
                    albums_skipped=albums_skipped,
                    albums_completed=albums_completed,
                    albums_active=len(active_jobs),
                    songs_to_analyze=songs_to_analyze,
                    songs_already_analyzed=songs_done,
                    songs_seen=songs_seen,
                    feature_albums={
                        'musicnn': albums_needing_musicnn,
                        'dclap': albums_needing_clap,
                        'lyrics': albums_needing_lyrics,
                    },
                )

            for album in all_albums:
                if album['Id'] in checked_album_ids:
                    albums_skipped += 1
                    report_progress()
                    continue
                monitor_and_clear_jobs()
                while len(active_jobs) >= MAX_QUEUED_ANALYSIS_JOBS:
                    monitor_and_clear_jobs()
                    report_progress()
                    time.sleep(5)

                tracks = get_tracks_from_album(album['Id'])
                if not tracks:
                    albums_skipped += 1
                    albums_no_tracks += 1
                    checked_album_ids.add(album['Id'])
                    logger.info(
                        f"Skipping album '{album.get('Name')}' (ID: {album.get('Id')}) - no tracks returned by media server."
                    )
                    report_progress()
                    continue

                masks = [
                    work_map.get(str(t.get('Id') or t.get('id')), 0) for t in tracks
                ]
                album_done = sum(1 for m in masks if m & done_bits == done_bits)
                songs_seen += len(tracks)
                songs_done += album_done
                songs_to_analyze += len(tracks) - album_done

                existing_count = sum(1 for m in masks if m & _ah.WORK_MUSICNN)
                needs_musicnn_analysis = existing_count < len(tracks)
                needs_clap_analysis = clap_available and any(
                    not m & _ah.WORK_CLAP for m in masks
                )
                needs_lyrics_analysis = LYRICS_ENABLED and any(
                    not m & _ah.WORK_LYRICS for m in masks
                )

                if album_done == len(tracks):
                    albums_skipped += 1
                    checked_album_ids.add(album['Id'])
                    status_parts = _ah.build_feature_status_parts(
                        clap_available, LYRICS_ENABLED
                    )
                    logger.info(
                        f"Skipping album '{album.get('Name')}' (ID: {album.get('Id')}) - all {len(tracks)} tracks already analyzed ({' + '.join(status_parts)})."
                    )
                    report_progress()
                    continue

                job = rq_queue_default.enqueue(
                    'tasks.analysis.analyze_album_task',
                    args=(album['Id'], album['Name'], top_n_moods, current_task_id, server_id),
                    job_id=str(uuid.uuid4()),
                    job_timeout=-1,
                    retry=Retry(max=3),
                )
                active_jobs[job.id] = job
                launched_job_ids.add(job.id)
                albums_launched += 1
                albums_needing_musicnn += int(needs_musicnn_analysis)
                albums_needing_clap += int(needs_clap_analysis)
                albums_needing_lyrics += int(needs_lyrics_analysis)
                checked_album_ids.add(album['Id'])

                logger.info(
                    "Queued album '%s' for feature work: MusiCNN=%s, DCLAP=%s, Lyrics=%s.",
                    album.get('Name'),
                    needs_musicnn_analysis,
                    needs_clap_analysis,
                    needs_lyrics_analysis,
                )

                report_progress(force=True)

            if (
                albums_launched == 0
                and total_albums_to_check > 0
                and albums_no_tracks == total_albums_to_check
            ):
                logger.error(
                    f"No tracks were returned for any of the {total_albums_to_check} albums; the media server library may be unreachable or empty."
                )
                raise error_manager.AudioMuseError(
                    ERR_MEDIASERVER_LIBRARY,
                    f"The media server returned no tracks for any of the {total_albums_to_check} album(s).",
                )

            if albums_launched == 0 and albums_skipped == total_albums_to_check:
                logger.warning(
                    f"No albums were enqueued: all {total_albums_to_check} albums were skipped (no tracks or already analyzed). Try num_recent_albums=0 or inspect media server responses."
                )

            while active_jobs:
                monitor_and_clear_jobs()
                report_progress(force=True)
                time.sleep(5)

            if finalize_indexes:
                log_and_update_main("Performing final index rebuild...", 95)
                try:
                    _run_all_index_builds(log_fn=log_and_update_main)
                except error_manager.AudioMuseError:
                    raise
                except Exception as e:
                    code = (
                        ERR_INDEX_EMPTY
                        if type(e).__name__ == "EmptyIndexError"
                        else ERR_INDEX_BUILD
                    )
                    raise error_manager.AudioMuseError(code, str(e), cause=e) from e
            logger.info(
                'Analysis complete. CLAP text search uses default queries (no auto-regeneration).'
            )

            total_failed_count, failed_errors = get_failed_child_summary(current_task_id)
            # Only this phase's own failures (see the baseline above).
            failed_count = max(0, total_failed_count - baseline_failed_count)
            if not failed_count:
                failed_errors = []
            logger.info(
                "Phase complete. Albums: %d launched, %d skipped of %d, %d failed. "
                "Songs: %d sent for analysis, %d already analyzed of %d. "
                "Feature albums: MusiCNN %d, DCLAP %d, Lyrics %d.",
                albums_launched, albums_skipped, total_albums_to_check, failed_count,
                songs_to_analyze, songs_done, songs_seen,
                albums_needing_musicnn, albums_needing_clap, albums_needing_lyrics,
            )
            final_done = albums_offset + albums_skipped + albums_completed
            final_message = f"Albums {final_done}/{reported_total}"
            if failed_count:
                final_message += f" ({failed_count} failed)"
            phase_status = TASK_STATUS_FAILURE if failed_count else TASK_STATUS_SUCCESS
            final_kwargs = {"task_state": phase_status}
            if failed_count:
                final_kwargs["failed_albums"] = failed_count
                final_kwargs["failed_album_errors"] = failed_errors
            log_and_update_main(final_message, 100, **final_kwargs)
            clean_temp(TEMP_DIR)
            return {
                "status": phase_status,
                "message": final_message,
                "failed_albums": failed_count,
            }

        except OperationalError as e:
            logger.critical(
                f"FATAL ERROR: Main analysis task failed due to DB connection issue: {e}",
                exc_info=True,
            )
            err = error_manager.record(ERR_DB_CONNECTION, str(e))
            log_and_update_main(
                "X Main analysis failed due to a database connection error. The task may be retried.",
                current_progress,
                task_state=TASK_STATUS_FAILURE,
                error=err,
            )
            raise
        except Exception as e:
            logger.critical(f"FATAL ERROR: Analysis failed: {e}", exc_info=True)
            err = error_manager.record(
                error_manager.classify(e, ERR_ANALYSIS_FAILED), str(e)
            )
            log_and_update_main(
                f"X Main analysis failed: {e}",
                current_progress,
                task_state=TASK_STATUS_FAILURE,
                error=err,
            )
            raise


def _albums_per_server(servers, num_recent_albums):
    """Each server's album list, fetched ONCE, before any phase starts.

    The status line counts albums across EVERY server, so the total has to be
    known up front instead of restarting at each phase's own denominator. The
    lists are then handed to the phases, so this costs no extra provider call:
    it is the same album fetch the phase would have made itself. A server whose
    listing FAILS here yields None, not an empty list, so its phase re-fetches and
    runs its own unreachable/empty-library error path instead of quietly
    reporting "no albums to analyze".
    """
    from tasks.mediaserver import context as server_context

    albums = []
    for server in servers:
        server_id = server['server_id'] if server else None
        try:
            with server_context.use_server(_bind_server_context(server_id)):
                albums.append(get_recent_albums(num_recent_albums) or [])
        except Exception:
            logger.exception(
                "Could not list albums for '%s'; its phase will retry the fetch",
                server['name'] if server else 'default server',
            )
            albums.append(None)
    return albums


def _enabled_analysis_servers(server_scope):
    with app.app_context():
        try:
            return registry.servers_for_scope(server_scope)
        except Exception:
            logger.exception("Server registry unavailable; analyzing the config default only")
            return [None]


def run_analysis_task(num_recent_albums, top_n_moods, server_scope="all"):
    """Analyze the union catalogue server-by-server, default server first.

    Every phase runs inline in THIS job. The album jobs a phase enqueues are the
    only children, so they keep the historical one-level topology: this task holds
    a 'high' worker while the album jobs run on the 'default' workers. Nesting a
    per-server job on the 'default' queue instead would let a phase occupy the only
    default worker while waiting for album jobs on that same queue, which deadlocks.
    Analysis never sweeps: each track resolves to the shared catalogue at analyze
    time (mapped provider id -> skip; new hash -> persist; known hash -> just map),
    so the servers stay aligned by construction and one index build ends the run.
    """
    current_job = get_current_job(redis_conn)
    parent_id = current_job.id if current_job else str(uuid.uuid4())

    servers = _enabled_analysis_servers(server_scope)
    if not servers:
        message = f"No enabled server matches scope '{server_scope}'; analysis skipped."
        logger.warning(message)
        with app.app_context():
            save_task_status(
                parent_id,
                "main_analysis",
                TASK_STATUS_SUCCESS,
                progress=100,
                details={"message": message},
            )
        return {'status': 'SKIPPED', 'message': message}
    if len(servers) == 1:
        server = servers[0]
        server_id = server['server_id'] if server else None
        return run_analysis_server_task(num_recent_albums, top_n_moods, server_id=server_id)

    albums_by_server = _albums_per_server(servers, num_recent_albums)
    grand_total = sum(len(a or []) for a in albums_by_server)
    logger.info(
        "Union analysis: %d albums to check across %d servers.", grand_total, len(servers)
    )

    summaries = []
    failed = []
    span = 90.0 / len(servers)
    albums_offset = 0
    for index, server in enumerate(servers):
        with app.app_context():
            info = get_task_info_from_db(parent_id)
            if info and info.get('status') == TASK_STATUS_REVOKED:
                return {'status': 'REVOKED', 'servers_completed': len(summaries)}
        logger.info(
            "Union analysis phase %d/%d: %s", index + 1, len(servers), server['name']
        )
        try:
            phase_summary = run_analysis_server_task(
                num_recent_albums,
                top_n_moods,
                server_id=server['server_id'],
                finalize_indexes=False,
                task_id=parent_id,
                progress_base=index * span,
                progress_span=span,
                final_phase=False,
                albums=albums_by_server[index],
                albums_offset=albums_offset,
                albums_total=grand_total,
            )
            summaries.append(phase_summary)
            if phase_summary.get('status') != TASK_STATUS_SUCCESS:
                failed.append(server['name'])
        except Exception:
            failed.append(server['name'])
            logger.exception("Union analysis phase failed for %s", server['name'])
        albums_offset += len(albums_by_server[index] or [])

    with app.app_context():
        save_task_status(
            parent_id,
            "main_analysis",
            TASK_STATUS_PROGRESS,
            progress=92,
            details={"message": "Building union catalogue indexes once..."},
        )
        try:
            _run_all_index_builds()
        except Exception:
            logger.exception("Final union index build failed")
            save_task_status(
                parent_id,
                "main_analysis",
                TASK_STATUS_FAILURE,
                progress=100,
                details={
                    "message": (
                        "Union analysis failed during the final index rebuild. "
                        "Check container logs."
                    ),
                    "failed_servers": failed,
                },
            )
            raise
        message = (
            f"Union analysis complete across {len(servers)} servers; "
            f"{len(failed)} source phase(s) failed."
        )
        save_task_status(
            parent_id,
            "main_analysis",
            TASK_STATUS_SUCCESS if not failed else TASK_STATUS_FAILURE,
            progress=100,
            details={"message": message, "failed_servers": failed},
        )
    return {
        'status': 'SUCCESS' if not failed else 'FAILURE',
        'message': message,
        'servers': summaries,
        'failed_servers': failed,
    }
