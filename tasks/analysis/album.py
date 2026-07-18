# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The per-album analysis job: FOR EACH SONG, run the stages that are missing.

One RQ job per album. Each track gets a TrackPlan, then _analyze_single_track
runs the ordered stages: download -> MusiCNN -> identity resolve -> persist ->
CLAP -> lyrics -> plugin hook. A track that yields nothing is SKIPPED and
counted (error 2007), never failed; the album fails only on real errors, and a
job that dies before writing anything still leaves a FAILURE row so the parent
phase cannot count it as completed.

Main Features:
* analyze_album_task: the RQ entry point, binding the server context and
  guaranteeing a failure row on any pre-analysis crash.
* _analyze_single_track: the linear stage sequence, one `if plan.<stage>:` per
  stage; the fingerprint identity decision maps this server's file onto the
  union catalogue (same audio on N servers = ONE catalogue row).
* Per-track map flush and temp-file cleanup in `finally`, so a Stop or a killed
  worker cannot strand a committed track without its mapping.
"""

import logging
import os
import time
import uuid

from rq import get_current_job

from config import (
    TEMP_DIR,
    MOOD_LABELS,
    EMBEDDING_MODEL_PATH,
    PREDICTION_MODEL_PATH,
    OTHER_FEATURE_LABELS,
    PER_SONG_MODEL_RELOAD,
    LYRICS_ENABLED,
    ANALYSIS_MONITOR_DB_INTERVAL,
)

from flask_app import app
from app_helper import (
    redis_conn,
    save_task_status,
    get_task_statuses,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
    TASK_STATUS_REVOKED,
)
from psycopg2 import OperationalError

from error import error_manager
from error.error_dictionary import (
    ERR_ALBUM_ANALYSIS_FAILED,
    ERR_DB_CONNECTION,
    ERR_TRACK_NOT_ANALYZABLE,
)

from ..mediaserver import get_tracks_from_album, download_track, registry
from ..memory_utils import (
    cleanup_cuda_memory,
    SessionRecycler,
    comprehensive_memory_cleanup,
)
from . import helper as _ah
from .helper import make_task_reporter, _bind_server_context
from .song import (
    analyze_track,
    cleanup_musicnn_sessions,
    cleanup_optional_models,
    robust_load_audio_with_fallback,
)


logger = logging.getLogger(__name__)


class TrackNotAnalyzable(Exception):
    pass


def _stage_download(item, track_name_full):
    path = download_track(TEMP_DIR, item)
    if not path:
        raise RuntimeError(f"Failed to download required audio for {track_name_full}")
    return path


def _stage_musicnn(path, track_name_full, plan, model_paths, session_recycler,
                   onnx_sessions, album_name):
    onnx_sessions = _ah.ensure_musicnn_sessions(
        onnx_sessions, model_paths, session_recycler, album_name
    )
    if plan.lyrics:
        analysis, embedding, track_audio, track_sr = analyze_track(
            path, MOOD_LABELS, model_paths, onnx_sessions=onnx_sessions, return_audio=True
        )
    else:
        analysis, embedding = analyze_track(
            path, MOOD_LABELS, model_paths, onnx_sessions=onnx_sessions
        )
        track_audio = track_sr = None
    if analysis is None:
        raise TrackNotAnalyzable(f"no decodable audio for {track_name_full}")
    session_recycler.increment()
    cleanup_cuda_memory(force=False)
    return onnx_sessions, analysis, embedding, track_audio, track_sr


def _stage_identity(item, plan, track_name_full, musicnn_embedding, fingerprint_index,
                    pending_track_maps, track_duration=None):
    from ..mediaserver import context as server_context

    source_server_id = (
        server_context.active_server_id() or registry.get_default_server_id()
    )
    if fingerprint_index is None:
        fingerprint_index = _ah.load_fingerprint_index()
    kind, track_id_str, provider_id = _ah.resolve_track_identity(
        fingerprint_index, musicnn_embedding, item, source_server_id,
        duration=track_duration,
    )
    if source_server_id:
        tier = 'analysis' if kind == 'unsignable' else 'fingerprint'
        pending_track_maps.setdefault(source_server_id, {})[provider_id] = (
            track_id_str, tier, item.get('FilePath')
        )

    if kind == 'existing':
        plan = _ah.replan_for_catalogue_row(plan, track_id_str)
        logger.info(
            "'%s' is already catalogued as %s; running only its missing stages (%s).",
            track_name_full, track_id_str, plan.describe(),
        )
        return fingerprint_index, plan, track_id_str, False
    if kind == 'unsignable':
        logger.warning(
            "No embedding signature for '%s'; catalogued under the "
            "server-scoped id %s so it is not re-analyzed forever.",
            track_name_full, track_id_str,
        )
    return fingerprint_index, plan, track_id_str, True


def _stage_persist_musicnn(item, track_name_full, track_id_str, musicnn_analysis,
                           top_moods, musicnn_embedding):
    logger.info(
        "SUCCESSFULLY ANALYZED '%s' as %s: tempo %.2f, energy %.4f, key %s %s, moods %s",
        track_name_full, track_id_str,
        musicnn_analysis['tempo'], musicnn_analysis['energy'],
        musicnn_analysis['key'], musicnn_analysis['scale'], top_moods,
    )
    _ah.persist_musicnn_results(
        item, musicnn_analysis, top_moods, musicnn_embedding,
        _ah.ZERO_OTHER_FEATURES,
    )


def _stage_clap(path, track_id_str, track_name_full, clap_label_embeddings):
    embedding = _ah.run_clap_for_track(path, track_name_full)
    if embedding is None:
        logger.warning(
            "  - CLAP produced no embedding for '%s'; its other stages still run "
            "and CLAP is retried on the next run", track_name_full,
        )
        return None, False
    if not _ah.persist_clap_embedding(track_id_str, embedding):
        return embedding, False
    other_features = _ah.compute_other_features_str(
        embedding, clap_label_embeddings, OTHER_FEATURE_LABELS
    )
    logger.info(f"  - Other Features: {other_features}")
    _ah.refresh_other_features(track_id_str, other_features)
    return embedding, True


def _stage_lyrics(item, path, track_audio, track_sr, track_name_full, top_moods,
                  ensure_download):
    saved = _ah.run_lyrics_for_track(
        item, path, track_audio, track_sr, track_name_full,
        robust_load_audio_with_fallback, top_moods=top_moods, download_fn=ensure_download,
    )
    if not saved:
        logger.info(
            "  - No lyrics for '%s' (instrumental or ungradable transcript); "
            "its other stages still run", track_name_full,
        )
    return bool(saved)


def _analyze_single_track(
    item, plan, track_name_full, album_id, album_name, parent_task_id, top_n_moods,
    model_paths, session_recycler, onnx_sessions, fingerprint_index,
    clap_label_embeddings, existing_top_moods_by_id, pending_track_maps,
):
    track_id_str = _ah.catalog_item_id(item)
    path = None
    track_audio = track_sr = None
    musicnn_analysis = musicnn_embedding = None
    clap_embedding = None
    top_moods = None
    produced = False
    try:
        if plan.needs_audio:
            path = _stage_download(item, track_name_full)

        def ensure_download():
            nonlocal path
            if path is None:
                path = download_track(TEMP_DIR, item)
            return path

        if plan.musicnn:
            onnx_sessions, musicnn_analysis, musicnn_embedding, track_audio, track_sr = (
                _stage_musicnn(
                    path, track_name_full, plan, model_paths, session_recycler,
                    onnx_sessions, album_name,
                )
            )
            top_moods = _ah.top_moods_from(musicnn_analysis, top_n_moods)
            produced = True

            fingerprint_index, plan, track_id_str, keep_analysis = _stage_identity(
                item, plan, track_name_full, musicnn_embedding, fingerprint_index,
                pending_track_maps,
                track_duration=musicnn_analysis.get('duration_seconds'),
            )
            if not keep_analysis:
                musicnn_analysis = musicnn_embedding = None
            if not plan.any_stage:
                return onnx_sessions, fingerprint_index
        else:
            top_moods = existing_top_moods_by_id.get(track_id_str) or None
            logger.info(
                "SKIPPED MusiCNN for '%s' (already analyzed); running %s",
                track_name_full, plan.describe(),
            )

        if musicnn_analysis is not None:
            _stage_persist_musicnn(
                item, track_name_full, track_id_str, musicnn_analysis,
                top_moods, musicnn_embedding,
            )

        if plan.clap:
            clap_embedding, clap_saved = _stage_clap(
                path, track_id_str, track_name_full, clap_label_embeddings
            )
            produced = produced or clap_saved

        if plan.lyrics:
            produced = _stage_lyrics(
                item, path, track_audio, track_sr, track_name_full, top_moods,
                ensure_download,
            ) or produced

        if not produced:
            raise TrackNotAnalyzable(f"no stage produced anything for {track_name_full}")

        _ah.run_song_analyzed_hook(
            item, path, musicnn_analysis, musicnn_embedding, clap_embedding,
            top_moods, album_id, album_name, parent_task_id,
        )
        return onnx_sessions, fingerprint_index
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError as e:
                logger.warning(
                    "Could not remove temp audio %s (%s); clean_temp sweeps it next run.",
                    path, e,
                )


def analyze_album_task(album_id, album_name, top_n_moods, parent_task_id, server_id=None):
    from tasks.mediaserver import context as server_context

    try:
        with server_context.use_server(_bind_server_context(server_id)):
            return _analyze_album_task_impl(album_id, album_name, top_n_moods, parent_task_id)
    except Exception as e:
        _record_album_failure_row(album_id, album_name, parent_task_id, e)
        raise


def _record_album_failure_row(album_id, album_name, parent_task_id, exc):
    current_job = get_current_job(redis_conn)
    if current_job is None:
        return
    try:
        with app.app_context():
            statuses = get_task_statuses([current_job.id])
            if statuses.get(current_job.id) == TASK_STATUS_FAILURE:
                return
            err = error_manager.from_exception(
                exc, code=error_manager.classify(exc, ERR_ALBUM_ANALYSIS_FAILED),
                logger=logger,
            )
            save_task_status(
                current_job.id,
                "album_analysis",
                TASK_STATUS_FAILURE,
                parent_task_id=parent_task_id,
                sub_type_identifier=album_id,
                progress=0,
                details={
                    "album_name": album_name,
                    "message": f"Album '{album_name}' failed before analysis could start.",
                    "error": err,
                },
            )
    except Exception:
        logger.exception(
            "Could not record the failure row for album '%s'; the phase may "
            "count this job as completed.", album_name,
        )


def _analyze_album_task_impl(album_id, album_name, top_n_moods, parent_task_id):
    from ..clap_analyzer import is_clap_available

    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        tracks_analyzed_count, tracks_skipped_count = 0, 0
        tracks_not_analyzable_count = 0
        model_paths = {'embedding': EMBEDDING_MODEL_PATH, 'prediction': PREDICTION_MODEL_PATH}
        onnx_sessions = None
        recycle_interval = 1 if PER_SONG_MODEL_RELOAD else 20
        session_recycler = SessionRecycler(recycle_interval=recycle_interval)

        log_and_update_album_task = make_task_reporter(
            current_task_id, "album_analysis", current_job,
            "Album analysis task started.",
            parent_task_id=parent_task_id, sub_type_identifier=album_id,
            base_details={"album_name": album_name}, log_cap=50,
            prefix=f"AlbumTask-{current_task_id}-{album_name}",
            min_db_interval=ANALYSIS_MONITOR_DB_INTERVAL,
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

            total_tracks_in_album = len(tracks)
            (
                existing_track_ids_set,
                missing_clap_ids_set,
                missing_lyrics_ids_set,
                clap_label_embeddings,
                existing_top_moods_by_id,
            ) = _ah.build_album_plan(album_name, tracks, top_n_moods, redis_conn, LYRICS_ENABLED)

            _ah.upsert_artist_mappings_for_tracks(tracks, album_name=album_name)

            fingerprint_index = None
            pending_track_maps = {}
            failed_tracks = []
            map_flush_errors = []
            last_revocation_check = float('-inf')

            def revoked():
                nonlocal last_revocation_check
                if not current_job:
                    return False
                now = time.monotonic()
                if now - last_revocation_check < ANALYSIS_MONITOR_DB_INTERVAL:
                    return False
                last_revocation_check = now
                statuses = get_task_statuses([current_task_id, parent_task_id])
                if statuses.get(current_task_id, TASK_STATUS_REVOKED) == TASK_STATUS_REVOKED:
                    return True
                parent_status = statuses.get(parent_task_id) if parent_task_id else None
                return parent_status in (TASK_STATUS_REVOKED, TASK_STATUS_FAILURE)

            for idx, item in enumerate(tracks, 1):
                if revoked():
                    _ah.flush_pending_track_maps(
                        pending_track_maps, map_flush_errors, album_name
                    )
                    log_and_update_album_task(
                        f"Stopping album analysis for '{album_name}' due to parent/self revocation.",
                        log_and_update_album_task.state['progress'],
                        task_state=TASK_STATUS_REVOKED,
                    )
                    return {"status": "REVOKED"}

                track_name_full = f"{item['Name']} by {item.get('AlbumArtist', 'Unknown')}"
                plan = _ah.plan_track_stages(
                    _ah.catalog_item_id(item),
                    existing_track_ids_set,
                    missing_clap_ids_set,
                    missing_lyrics_ids_set,
                    LYRICS_ENABLED,
                )

                if not plan.any_stage:
                    tracks_skipped_count += 1
                    status_parts = _ah.build_feature_status_parts(
                        is_clap_available(), LYRICS_ENABLED, include_check_marks=True
                    )
                    logger.info(
                        f"Skipping '{track_name_full}' - all analyses complete ({', '.join(status_parts)})"
                    )
                    continue

                log_and_update_album_task(
                    f"Analyzing track: {track_name_full} ({idx}/{total_tracks_in_album})",
                    10 + int(85 * (idx / float(total_tracks_in_album))),
                    current_track_name=track_name_full,
                )

                try:
                    onnx_sessions, fingerprint_index = _analyze_single_track(
                        item, plan, track_name_full, album_id, album_name,
                        parent_task_id, top_n_moods, model_paths, session_recycler,
                        onnx_sessions, fingerprint_index, clap_label_embeddings,
                        existing_top_moods_by_id, pending_track_maps,
                    )
                    tracks_analyzed_count += 1
                except OperationalError:
                    raise
                except TrackNotAnalyzable as e:
                    error_manager.record(
                        ERR_TRACK_NOT_ANALYZABLE, str(e), logger=logger, level=logging.WARNING
                    )
                    tracks_not_analyzable_count += 1
                except Exception as e:
                    logger.exception(
                        f"Track analysis failed for '{track_name_full}'; continuing with the next track."
                    )
                    failed_tracks.append(f"{track_name_full}: {e}")
                finally:
                    _ah.flush_pending_track_maps(
                        pending_track_maps, map_flush_errors, album_name
                    )

            _ah.flush_pending_track_maps(pending_track_maps, map_flush_errors, album_name)

            cleanup_musicnn_sessions(onnx_sessions, context="album end")
            onnx_sessions = None
            cleanup_optional_models(context="album end")
            comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)

            _ah.raise_album_failures(failed_tracks, map_flush_errors, total_tracks_in_album)

            summary = {
                "tracks_analyzed": tracks_analyzed_count,
                "tracks_skipped": tracks_skipped_count,
                "tracks_not_analyzable": tracks_not_analyzable_count,
                "total_tracks_in_album": total_tracks_in_album,
            }
            completion_message = f"Album '{album_name}' analysis complete."
            if tracks_not_analyzable_count:
                completion_message += (
                    f" {tracks_not_analyzable_count}/{total_tracks_in_album} track(s) carried no "
                    "analyzable audio and were skipped."
                )
            log_and_update_album_task(
                completion_message,
                100,
                task_state=TASK_STATUS_SUCCESS,
                final_summary_details=summary,
            )
            return {"status": "SUCCESS", **summary}

        except OperationalError as e:
            err = error_manager.from_exception(e, code=ERR_DB_CONNECTION, logger=logger)
            log_and_update_album_task(
                f"Database connection failed for album '{album_name}'. Retrying...",
                log_and_update_album_task.state['progress'],
                task_state=TASK_STATUS_FAILURE,
                error=err,
            )
            raise
        except Exception as e:
            err = error_manager.from_exception(
                e, code=error_manager.classify(e, ERR_ALBUM_ANALYSIS_FAILED), logger=logger
            )
            log_and_update_album_task(
                f"Failed to analyze album '{album_name}': {e}",
                log_and_update_album_task.state['progress'],
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
