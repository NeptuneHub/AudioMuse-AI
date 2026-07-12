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
  mood tags, MusiCNN embeddings, CLAP and lyrics embeddings, then upsert to the DB.
* Media-server reachability and auth probing before enqueuing, so a bad server aborts
  early instead of failing every child job.
* rebuild_all_indexes_task and _run_all_index_builds rebuild every similarity index
  after new embeddings land; freed audio RAM is returned to the OS between albums.
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
import onnxruntime as ort  # noqa: F401  re-exported: tests patch `tasks.analysis.ort.InferenceSession`

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
)


from .mediaserver import (
    get_recent_albums,
    get_tracks_from_album,
    download_track,
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
    _find_onnx_name,
    run_inference,
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


def _run_all_index_builds(log_fn=None):
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
            logger.info(f"OK {label}")
        except Exception as e:
            logger.warning(f"Failed to build/store {label}: {e}")
            if fatal:
                raise
        finally:
            gc.collect()

    if log_fn:
        log_fn("Performing final index rebuild...", 95)
    _step(
        "IVF index rebuilt",
        lambda: build_and_store_ivf_index(get_db()),
        progress=95,
        banner="Building IVF audio index...",
        fatal=True,
    )
    _step(
        "CLAP text search index",
        lambda: build_and_store_clap_index(get_db()),
        progress=96,
        banner="Building CLAP text search index...",
    )
    _step(
        "Lyrics search index",
        lambda: build_and_store_lyrics_index(get_db()),
        progress=96,
        banner="Building lyrics search index...",
    )
    _step(
        "Lyrics axes index",
        lambda: build_and_store_lyrics_axes_index(get_db()),
        progress=96,
        banner="Building lyrics axes index...",
    )
    _step(
        "SemGrove merged index rebuilt",
        lambda: build_and_store_sem_grove_index(get_db()),
        progress=96,
        banner="Building SemGrove merged index...",
    )
    _step(
        "Artist similarity index rebuilt",
        lambda: build_and_store_artist_index(get_db()),
        progress=97,
        banner="Building artist similarity index...",
    )
    _step(
        "Song map projection rebuilt",
        lambda: build_and_store_map_projection('main_map'),
        progress=97,
        banner="Building song map projection...",
    )
    _step(
        "Artist component projection rebuilt",
        lambda: build_and_store_artist_projection('artist_map'),
        progress=97,
        banner="Building artist component projection...",
    )
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


def rebuild_all_indexes_task(log_fn=None):
    logger.info("Starting index rebuild task (enqueued as subtask)...")
    with app.app_context():
        try:
            _run_all_index_builds(log_fn=log_fn)
            logger.info("OK Index rebuild task completed successfully")
            return {"status": "SUCCESS", "message": "All indexes rebuilt"}
        except Exception as e:
            logger.exception("X Index rebuild task failed")
            return {"status": "FAILURE", "message": str(e)}


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
            provider_options = get_provider_options()
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
    from tasks.mediaserver import registry

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
            missing_chromaprint_ids_set = _ah.get_missing_chromaprint_ids(track_ids_all)
            total_tracks_in_album = len(tracks)

            logger.info(
                "Feature plan for album '%s': MusiCNN=%d, DCLAP=%d, Lyrics=%d, "
                "Chromaprint=%d of %d tracks.",
                album_name,
                total_tracks_in_album - len(existing_track_ids_set),
                len(missing_clap_ids_set),
                len(missing_lyrics_ids_set),
                len(missing_chromaprint_ids_set),
                total_tracks_in_album,
            )

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

            for idx, item in enumerate(tracks, 1):
                if current_job:
                    task_info = get_task_info_from_db(current_task_id)
                    parent_info = get_task_info_from_db(parent_task_id) if parent_task_id else None
                    if (task_info and task_info.get('status') == 'REVOKED') or (
                        parent_info and parent_info.get('status') in ['REVOKED', 'FAILURE']
                    ):
                        log_and_update_album_task(
                            f"Stopping album analysis for '{album_name}' due to parent/self revocation.",
                            current_progress_val,
                            task_state=TASK_STATUS_REVOKED,
                        )
                        return {"status": "REVOKED"}

                track_name_full = f"{item['Name']} by {item.get('AlbumArtist', 'Unknown')}"
                progress = 10 + int(85 * (idx / float(total_tracks_in_album)))
                log_and_update_album_task(
                    f"Analyzing track: {track_name_full} ({idx}/{total_tracks_in_album})",
                    progress,
                    current_track_name=track_name_full,
                )

                _ah.upsert_artist_mappings_for_tracks([item], album_name=album_name)

                track_id_str = _ah.catalog_item_id(item)
                needs_musicnn, needs_clap, needs_lyrics, needs_chromaprint = (
                    _ah.decide_track_needs(
                        track_id_str,
                        existing_track_ids_set,
                        missing_clap_ids_set,
                        missing_lyrics_ids_set,
                        LYRICS_ENABLED,
                        missing_chromaprint_ids_set,
                    )
                )
                track_audio, track_sr = None, None

                if not (needs_musicnn or needs_clap or needs_lyrics or needs_chromaprint):
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

                needs_audio_upfront = needs_musicnn or needs_clap or needs_chromaprint
                if needs_audio_upfront:
                    path = download_track(TEMP_DIR, item)
                    if not path:
                        continue
                else:
                    path = None

                def _ensure_track_download(item=item):
                    nonlocal path
                    if path is None:
                        path = download_track(TEMP_DIR, item)
                    return path

                try:
                    track_processed = False
                    chromaprint = None
                    if needs_chromaprint:
                        from .audio_fingerprint import (
                            chromaprint_canonical_id,
                            compute_chromaprint,
                        )

                        chromaprint = compute_chromaprint(path)
                        candidate_id = chromaprint_canonical_id(chromaprint)
                        if (
                            candidate_id
                            and candidate_id != track_id_str
                            and _ah.get_existing_track_ids([candidate_id])
                        ):
                            from .mediaserver import context as server_context, registry

                            source_server_id = (
                                server_context.active_server_id()
                                or registry.get_default_server_id()
                            )
                            if source_server_id:
                                registry.upsert_track_maps(
                                    source_server_id,
                                    {
                                        candidate_id: (
                                            str(item.get('Id') or item.get('id')),
                                            'chromaprint',
                                        )
                                    },
                                )
                            item['_catalog_item_id'] = candidate_id
                            track_id_str = candidate_id
                            needs_musicnn = False
                            needs_clap = bool(
                                is_clap_available()
                                and _ah.get_missing_ids_in_table(
                                    'clap_embedding', [candidate_id]
                                )
                            )
                            needs_lyrics = bool(
                                LYRICS_ENABLED
                                and _ah.get_missing_ids_in_table(
                                    'lyrics_embedding', [candidate_id]
                                )
                            )
                            needs_chromaprint = False
                            chromaprint = None
                            logger.info(
                                "Chromaprint matched '%s' to existing catalogue id %s; "
                                "skipping duplicate MusiCNN analysis.",
                                track_name_full,
                                candidate_id,
                            )

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
                            logger.warning(
                                f"Skipping track {track_name_full} as analysis returned None."
                            )
                            tracks_skipped_count += 1
                            continue

                        top_moods = dict(
                            sorted(analysis['moods'].items(), key=lambda i: i[1], reverse=True)[
                                :top_n_moods
                            ]
                        )
                        musicnn_analysis, musicnn_embedding = analysis, embedding
                        track_processed = True
                        session_recycler.increment()
                        cleanup_cuda_memory(force=False)
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
                            item, musicnn_analysis, top_moods, musicnn_embedding, other_features
                        )

                    if chromaprint and _ah.persist_chromaprint(track_id_str, chromaprint):
                        track_processed = True

                    _ah.persist_clap_embedding(
                        track_id_str, clap_embedding_for_track, needs_clap
                    )

                    if _ah.run_lyrics_for_track(
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
                    ):
                        track_processed = True

                    if track_processed:
                        _ah.run_song_analyzed_hook(
                            item, path, musicnn_analysis, musicnn_embedding,
                            clap_embedding_for_track, top_moods, album_id, album_name,
                            parent_task_id,
                        )
                        tracks_analyzed_count += 1
                finally:
                    if path and os.path.exists(path):
                        os.remove(path)

            cleanup_musicnn_sessions(onnx_sessions, context="album end")
            onnx_sessions = None
            cleanup_optional_models(context="album end")
            logger.info("Performing final comprehensive cleanup after album analysis")
            comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)

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
    enqueue_sweep=True,
    task_id=None,
    progress_base=0.0,
    progress_span=100.0,
    final_phase=True,
):
    """Analyze one server while persisting everything by canonical catalogue id."""
    from tasks.mediaserver import context as server_context

    with server_context.use_server(_bind_server_context(server_id)):
        return _run_analysis_server_task_impl(
            num_recent_albums,
            top_n_moods,
            server_id=server_id,
            finalize_indexes=finalize_indexes,
            enqueue_sweep=enqueue_sweep,
            task_id=task_id,
            progress_base=progress_base,
            progress_span=progress_span,
            final_phase=final_phase,
        )


def _run_analysis_server_task_impl(
    num_recent_albums,
    top_n_moods,
    server_id=None,
    finalize_indexes=True,
    enqueue_sweep=True,
    task_id=None,
    progress_base=0.0,
    progress_span=100.0,
    final_phase=True,
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
            from .audio_fingerprint import fpcalc_available

            if not fpcalc_available():
                raise RuntimeError(
                    "Required Chromaprint executable 'fpcalc' is not available in the "
                    "analysis worker. Rebuild/redeploy the updated container before analysis."
                )
            clean_temp(TEMP_DIR)
            all_albums = get_recent_albums(num_recent_albums)
            if not all_albums:
                _verify_media_server_reachable()
                log_and_update_main(
                    "No new albums to analyze.", 100, albums_found=0, task_state=TASK_STATUS_SUCCESS
                )
                return {"status": "SUCCESS", "message": "No new albums to analyze."}

            total_albums_to_check = len(all_albums)
            active_jobs = {}
            launched_job_ids = set()
            albums_skipped, albums_launched, albums_completed = 0, 0, 0
            albums_no_tracks = 0
            albums_needing_musicnn = 0
            albums_needing_clap = 0
            albums_needing_lyrics = 0
            albums_needing_chromaprint = 0
            last_monitor_db_check = float('-inf')

            def monitor_and_clear_jobs():
                nonlocal albums_completed, last_monitor_db_check
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

            for idx, album in enumerate(all_albums):
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
                    logger.info(
                        f"Skipping album '{album.get('Name')}' (ID: {album.get('Id')}) - no tracks returned by media server."
                    )
                    continue

                _ah.upsert_artist_mappings_for_tracks(tracks, album_name=album.get('Name'))

                try:
                    (
                        existing_count,
                        needs_clap_analysis,
                        needs_lyrics_analysis,
                        needs_chromaprint_analysis,
                    ) = (
                        _ah.compute_album_needs(
                            tracks,
                            is_clap_available(),
                            LYRICS_ENABLED,
                            server_id=server_id,
                        )
                    )
                except Exception as e:
                    logger.exception(
                        "Cannot determine feature needs for album '%s' (ID: %s). "
                        "Stopping instead of silently skipping real analysis.",
                        album.get('Name'),
                        album.get('Id'),
                    )
                    raise RuntimeError(
                        f"Cannot determine analysis needs for album '{album.get('Name')}': {e}"
                    ) from e

                needs_musicnn_analysis = existing_count < len(tracks)

                if existing_count >= len(tracks) and not (
                    needs_clap_analysis
                    or needs_lyrics_analysis
                    or needs_chromaprint_analysis
                ):
                    for item in tracks:
                        _ah.refresh_track_metadata(item, album.get('Name'))
                    albums_skipped += 1
                    checked_album_ids.add(album['Id'])
                    status_parts = _ah.build_feature_status_parts(
                        is_clap_available(), LYRICS_ENABLED
                    )
                    logger.info(
                        f"Skipping album '{album.get('Name')}' (ID: {album.get('Id')}) - all {existing_count}/{len(tracks)} tracks already analyzed ({' + '.join(status_parts)})."
                    )
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
                albums_needing_chromaprint += int(needs_chromaprint_analysis)
                checked_album_ids.add(album['Id'])

                logger.info(
                    "Queued album '%s' for feature work: MusiCNN=%s, DCLAP=%s, "
                    "Lyrics=%s, Chromaprint=%s.",
                    album.get('Name'),
                    needs_musicnn_analysis,
                    needs_clap_analysis,
                    needs_lyrics_analysis,
                    needs_chromaprint_analysis,
                )

                progress = 5 + int(85 * (idx / float(total_albums_to_check)))
                status_message = f"Launched: {albums_launched}. Completed: {albums_completed}/{albums_launched}. Active: {len(active_jobs)}. Skipped: {albums_skipped}/{total_albums_to_check}."
                log_and_update_main(
                    status_message,
                    progress,
                    albums_to_process=albums_launched,
                    albums_skipped=albums_skipped,
                    feature_albums={
                        'musicnn': albums_needing_musicnn,
                        'dclap': albums_needing_clap,
                        'lyrics': albums_needing_lyrics,
                        'chromaprint': albums_needing_chromaprint,
                    },
                )

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
                progress = 5 + int(
                    85 * ((albums_skipped + albums_completed) / float(total_albums_to_check))
                )
                status_message = f"Launched: {albums_launched}. Completed: {albums_completed}/{albums_launched}. Active: {len(active_jobs)}. Skipped: {albums_skipped}/{total_albums_to_check}. (Finalizing)"
                log_and_update_main(status_message, progress)
                time.sleep(5)

            log_and_update_main("Relabelling catalogue to content fingerprint ids...", 93)
            try:
                from tasks.fingerprint_canonicalize import canonicalize_fingerprinted_ids
                canonicalize_fingerprinted_ids(
                    rebuild=False, source_server_id=server_id
                )
            except Exception as exc:
                logger.exception(
                    "Fingerprint canonicalization failed; analysis cannot safely continue"
                )
                raise RuntimeError(
                    "Feature analysis completed, but Chromaprint canonicalization failed"
                ) from exc

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

            if enqueue_sweep:
                try:
                    rq_queue_default.enqueue(
                        'tasks.multiserver_sync.sweep_all_secondary_servers',
                        job_id=str(uuid.uuid4()),
                        job_timeout=-1,
                    )
                    logger.info("Enqueued multi-server matching sweep for configured servers.")
                except Exception:
                    logger.exception("Failed to enqueue multi-server matching sweep.")

            failed_count, failed_errors = get_failed_child_summary(current_task_id)
            final_message = (
                f"Main analysis complete. Launched {albums_launched}, "
                f"Skipped {albums_skipped}, Failed {failed_count}. "
                f"Feature albums: MusiCNN {albums_needing_musicnn}, "
                f"DCLAP {albums_needing_clap}, Lyrics {albums_needing_lyrics}, "
                f"Chromaprint {albums_needing_chromaprint}."
            )
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


def _enabled_analysis_servers(server_scope):
    from tasks.mediaserver import registry

    with app.app_context():
        try:
            servers = [s for s in registry.list_servers() if s['enabled']]
        except Exception:
            logger.exception("Server registry unavailable; analyzing the config default only")
            return []
    if server_scope == 'default':
        default_only = [s for s in servers if s['is_default']]
        return default_only or servers[:1]
    return servers


def run_analysis_task(num_recent_albums, top_n_moods, server_scope="all"):
    """Analyze the union catalogue server-by-server, default server first.

    Every phase runs inline in THIS job. The album jobs a phase enqueues are the
    only children, so they keep the historical one-level topology: this task holds
    a 'high' worker while the album jobs run on the 'default' workers. Nesting a
    per-server job on the 'default' queue instead would let a phase occupy the only
    default worker while waiting for album jobs on that same queue, which deadlocks.
    """
    current_job = get_current_job(redis_conn)
    parent_id = current_job.id if current_job else str(uuid.uuid4())

    servers = _enabled_analysis_servers(server_scope)
    if len(servers) <= 1:
        server_id = servers[0]['server_id'] if servers else None
        return run_analysis_server_task(num_recent_albums, top_n_moods, server_id=server_id)

    summaries = []
    failed = []
    span = 90.0 / len(servers)
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
                enqueue_sweep=False,
                task_id=parent_id,
                progress_base=index * span,
                progress_span=span,
                final_phase=False,
            )
            summaries.append(phase_summary)
            if phase_summary.get('status') != TASK_STATUS_SUCCESS:
                failed.append(server['name'])
        except Exception:
            failed.append(server['name'])
            logger.exception("Union analysis phase failed for %s", server['name'])

        remaining_server_ids = [
            remaining['server_id'] for remaining in servers[index + 1:]
        ]
        if remaining_server_ids:
            try:
                from tasks.multiserver_sync import sweep_all_secondary_servers

                sweep_all_secondary_servers(
                    task_id=str(uuid.uuid4()), server_ids=remaining_server_ids
                )
            except Exception:
                logger.exception("Post-phase server alignment failed after %s", server['name'])

    with app.app_context():
        save_task_status(
            parent_id,
            "main_analysis",
            TASK_STATUS_PROGRESS,
            progress=92,
            details={"message": "Building union catalogue indexes once..."},
        )
        _run_all_index_builds()
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
