# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The per-album analysis job: FOR EACH SONG, run the stages that are missing.

One queue job per album. Each track gets a TrackPlan, then _analyze_single_track
runs the ordered stages: download -> MusiCNN -> identity resolve -> persist ->
CLAP -> lyrics -> plugin hook. A track that yields nothing is SKIPPED and
counted (error 2007), never failed; the album fails only on real errors, and a
job that dies before writing anything still leaves a FAILURE row so the parent
phase cannot count it as completed.

A database connectivity error is the one exception and is re-raised untouched at
every level, entry point included. Claiming only looks at NEW rows and reclaiming
only at RUNNING ones, so a FAILURE row written while the database is away is
never touched again: a two-second blip would permanently fail the album and count
it into the parent's failed tally. Left alone, the row stays RUNNING and the
queue requeues it.

Main Features:
* analyze_album_task: the queue entry point, binding the server context and
  guaranteeing a failure row on any pre-analysis crash other than a connectivity
  one.
* _analyze_single_track: the linear stage sequence, one `if plan.<stage>:` per
  stage; the fingerprint identity decision maps this server's file onto the
  union catalogue (same audio on N servers = ONE catalogue row).
* Per-track map flush and temp-file cleanup in `finally`, so a Stop or a killed
  worker cannot strand a committed track without its mapping.
"""

import logging
import os
import uuid

import taskqueue

from config import (
    TEMP_DIR,
    MOOD_LABELS,
    EMBEDDING_MODEL_PATH,
    PREDICTION_MODEL_PATH,
    OTHER_FEATURE_LABELS,
    PER_SONG_MODEL_RELOAD,
    LYRICS_ENABLED,
    ANALYSIS_MONITOR_DB_INTERVAL,
    CHROMAPRINT_COLLECTION_ENABLED,
    TASK_STATUS_RUNNING,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
    TASK_STATUS_REVOKED,
)

from flask_app import app
from database import (
    save_task_status,
    get_task_statuses,
)
from psycopg2 import InterfaceError, OperationalError

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
from .. import chromaprint
from database import (
    persist_chromaprint,
    get_chromaprint,
    record_analysis_exclusion,
)
from . import helper as _ah
from .helper import make_task_reporter, _bind_server_context
from ..task_run import TaskCancelled, make_cancel_check
from .song import (
    analyze_track_for_album,
    AudioNotDecodableError,
    cleanup_musicnn_sessions,
    cleanup_optional_models,
    decode_audio_once,
    extract_basic_features,
    resample_audio,
    robust_load_audio_with_fallback,
)


logger = logging.getLogger(__name__)

analyze_track = analyze_track_for_album


class TrackNotAnalyzable(Exception):
    def __init__(self, message, cacheable=False):
        super().__init__(message)
        self.cacheable = bool(cacheable)


class TrackSourceUnavailable(Exception):
    pass


def _stage_download(item, track_name_full):
    path = download_track(TEMP_DIR, item)
    if not path:
        raise TrackSourceUnavailable(f"source audio unavailable for {track_name_full}")
    return path


def _stage_collect_chromaprint(item, path, track_name_full):
    if not (path and CHROMAPRINT_COLLECTION_ENABLED and chromaprint.is_available()):
        return
    from ..mediaserver import context as server_context

    server_id = server_context.active_server_id() or registry.get_default_server_id()
    provider_id = _ah.provider_item_id(item)
    existing = get_chromaprint(server_id, provider_id) if server_id else None
    if existing is not None:
        item['_chromaprint'] = existing
        return
    blob = chromaprint.compute(path)
    item['_chromaprint'] = blob
    if blob:
        logger.info("Calculated Chromaprint for '%s'", track_name_full)
    else:
        logger.warning("Could not calculate Chromaprint for '%s'", track_name_full)
    if server_id:
        persist_chromaprint(server_id, provider_id, blob)


def _stage_musicnn(path, track_name_full, plan, model_paths, session_recycler,
                   onnx_sessions, album_name, native_audio=None, native_sr=None):
    onnx_sessions = _ah.ensure_musicnn_sessions(
        onnx_sessions, model_paths, session_recycler, album_name
    )
    try:
        if plan.lyrics:
            analysis, embedding, track_audio, track_sr = analyze_track(
                path, MOOD_LABELS, model_paths,
                onnx_sessions=onnx_sessions, return_audio=True,
                native_audio=native_audio, native_sr=native_sr,
            )
        else:
            analysis, embedding = analyze_track(
                path, MOOD_LABELS, model_paths, onnx_sessions=onnx_sessions,
                native_audio=native_audio, native_sr=native_sr,
            )
            track_audio = track_sr = None
    except AudioNotDecodableError as exc:
        raise TrackNotAnalyzable(str(exc), cacheable=True) from exc
    if analysis is None:
        raise TrackNotAnalyzable(
            f"analysis produced no result for {track_name_full}", cacheable=False
        )
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
        "SUCCESSFULLY ANALYZED '%s' as %s: tempo %.2f, energy %.4f, key %s %s",
        track_name_full, track_id_str,
        musicnn_analysis['tempo'], musicnn_analysis['energy'],
        musicnn_analysis['key'], musicnn_analysis['scale'],
    )
    logger.info("-- moods %s", top_moods)
    _ah.persist_musicnn_results(
        item, musicnn_analysis, top_moods, musicnn_embedding,
        _ah.ZERO_OTHER_FEATURES,
    )


def _base_features_from(analysis):
    if not analysis:
        return None
    values = (
        analysis.get('tempo'), analysis.get('energy'),
        analysis.get('key'), analysis.get('scale'),
    )
    return None if any(v is None for v in values) else values


def _stage_base(path, track_id_str, track_name_full, native_audio=None, native_sr=None,
                precomputed=None):
    if precomputed is not None:
        tempo, energy, musical_key, scale = precomputed
    else:
        if native_audio is not None and native_sr is not None:
            audio, sr = resample_audio(native_audio, native_sr, 16000), 16000
        else:
            audio, sr = robust_load_audio_with_fallback(path, target_sr=16000)
        if audio is None or audio.size == 0:
            logger.warning(
                "  - Could not decode audio to recompute base features for '%s'",
                track_name_full,
            )
            return False
        tempo, energy, musical_key, scale = extract_basic_features(audio, sr)
    if _ah.refresh_base_features(track_id_str, tempo, energy, musical_key, scale):
        logger.info(
            "  - Base features stored for '%s': tempo %.2f, energy %.4f, key %s %s",
            track_name_full, tempo, energy, musical_key, scale,
        )
        return True
    return False


def _stage_clap(path, track_id_str, track_name_full, clap_label_embeddings,
                native_audio=None, native_sr=None):
    embedding = _ah.run_clap_for_track(
        path, track_name_full, native_audio=native_audio, native_sr=native_sr
    )
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
    try:
        saved = _ah.run_lyrics_for_track(
            item, path, track_audio, track_sr, track_name_full,
            robust_load_audio_with_fallback, top_moods=top_moods, download_fn=ensure_download,
        )
    except AudioNotDecodableError as exc:
        raise TrackNotAnalyzable(str(exc), cacheable=True) from exc
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
    native_audio = native_sr = None
    musicnn_analysis = musicnn_embedding = None
    base_features = None
    clap_embedding = None
    top_moods = None
    produced = False
    audio_undecodable = False
    try:
        if plan.needs_audio:
            path = _stage_download(item, track_name_full)

        if plan.musicnn:
            _stage_collect_chromaprint(item, path, track_name_full)

        if path and plan.needs_audio:
            native_audio, native_sr = decode_audio_once(path)
            audio_undecodable = native_audio is None
            if audio_undecodable:
                logger.warning(
                    "  - Audio for '%s' could not be decoded; every stage that needs "
                    "it is skipped instead of re-reading the file.", track_name_full
                )
                raise TrackNotAnalyzable(
                    f"no decodable audio for {track_name_full}", cacheable=True
                )

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
                    native_audio=native_audio, native_sr=native_sr,
                )
            )
            top_moods = _ah.top_moods_from(musicnn_analysis, top_n_moods)
            produced = True

            fingerprint_index, plan, track_id_str, keep_analysis = _stage_identity(
                item, plan, track_name_full, musicnn_embedding, fingerprint_index,
                pending_track_maps,
                track_duration=musicnn_analysis.get('duration_seconds'),
            )
            base_features = _base_features_from(musicnn_analysis)
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

        if plan.base:
            produced = _stage_base(
                path, track_id_str, track_name_full,
                native_audio=native_audio, native_sr=native_sr,
                precomputed=base_features,
            ) or produced

        if plan.clap:
            clap_embedding, clap_saved = _stage_clap(
                path, track_id_str, track_name_full, clap_label_embeddings,
                native_audio=native_audio, native_sr=native_sr,
            )
            produced = produced or clap_saved

        if plan.lyrics:
            if track_audio is None and native_audio is not None and native_sr:
                track_audio = resample_audio(native_audio, native_sr, 16000)
                track_sr = 16000
            native_audio = None
            produced = _stage_lyrics(
                item, path, track_audio, track_sr, track_name_full, top_moods,
                ensure_download,
            ) or produced

        if not produced:
            raise TrackNotAnalyzable(
                f"no stage produced anything for {track_name_full}",
                cacheable=audio_undecodable,
            )

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
    except (OperationalError, InterfaceError, TaskCancelled):
        raise
    except Exception as e:
        _record_album_error_on_row(album_id, album_name, parent_task_id, e)
        raise


def _record_album_error_on_row(album_id, album_name, parent_task_id, exc):
    current_task_id = taskqueue.current_task_id()
    if current_task_id is None:
        return
    try:
        with app.app_context():
            ids = [current_task_id]
            if parent_task_id:
                ids.append(parent_task_id)
            statuses = get_task_statuses(ids)
            if parent_task_id and (
                parent_task_id not in statuses
                or statuses.get(parent_task_id)
                in (TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)
            ):
                return
            err = error_manager.from_exception(
                exc, code=error_manager.classify(exc, ERR_ALBUM_ANALYSIS_FAILED),
                logger=logger,
            )
            save_task_status(
                current_task_id,
                "album_analysis",
                TASK_STATUS_RUNNING,
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
            "Could not record the error for album '%s' on its row; the queue still "
            "fails the row, with only the exception text", album_name,
        )


def _analyze_album_task_impl(album_id, album_name, top_n_moods, parent_task_id):
    from ..clap_analyzer import is_clap_available

    claimed_task_id = taskqueue.current_task_id()
    current_task_id = claimed_task_id or str(uuid.uuid4())

    with app.app_context():
        if claimed_task_id and parent_task_id:
            parent_statuses = get_task_statuses([parent_task_id])
            if (
                parent_task_id not in parent_statuses
                or parent_statuses.get(parent_task_id)
                in (TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)
            ):
                logger.info(
                    "Album task %s will not start because parent %s is missing or terminal.",
                    current_task_id,
                    parent_task_id,
                )
                return {
                    "status": TASK_STATUS_REVOKED,
                    "message": "Parent analysis was cancelled.",
                }
        tracks_analyzed_count, tracks_skipped_count = 0, 0
        tracks_not_analyzable_count = 0
        model_paths = {'embedding': EMBEDDING_MODEL_PATH, 'prediction': PREDICTION_MODEL_PATH}
        onnx_sessions = None
        recycle_interval = 1 if PER_SONG_MODEL_RELOAD else 20
        session_recycler = SessionRecycler(recycle_interval=recycle_interval)

        log_and_update_album_task = make_task_reporter(
            current_task_id, "album_analysis",
            "Album analysis task started.",
            parent_task_id=parent_task_id, sub_type_identifier=album_id,
            base_details={"album_name": album_name},
            prefix=f"AlbumTask-{current_task_id}-{album_name}",
            min_db_interval=ANALYSIS_MONITOR_DB_INTERVAL,
        )
        pending_track_maps = {}
        map_flush_errors = []
        cancel, close_cancel = make_cancel_check(
            claimed_task_id, parent_task_id, every_seconds=ANALYSIS_MONITOR_DB_INTERVAL,
        )
        try:
            log_and_update_album_task(f"Fetching tracks for album: {album_name}", 5)
            tracks = get_tracks_from_album(album_id)
            if not tracks:
                return {
                    "status": TASK_STATUS_SUCCESS,
                    "message": f"No tracks found for album: {album_name}",
                    "tracks_analyzed": 0,
                }

            total_tracks_in_album = len(tracks)
            (
                existing_track_ids_set,
                missing_clap_ids_set,
                missing_lyrics_ids_set,
                missing_base_ids_set,
                clap_label_embeddings,
                existing_top_moods_by_id,
            ) = _ah.build_album_plan(album_name, tracks, top_n_moods, LYRICS_ENABLED)
            analysis_exclusions = _ah.load_album_analysis_exclusions(tracks)

            _ah.upsert_artist_mappings_for_tracks(tracks, album_name=album_name)

            fingerprint_index = None
            failed_tracks = []
            unavailable_tracks = []

            for idx, item in enumerate(tracks, 1):
                cancel()

                track_name_full = f"{item['Name']} by {item.get('AlbumArtist', 'Unknown')}"
                provider_id = _ah.provider_item_id(item)
                exclusion = analysis_exclusions.get(provider_id)
                if exclusion and _ah.analysis_exclusion_matches(item, exclusion):
                    tracks_not_analyzable_count += 1
                    logger.info(
                        "Skipping '%s' - previously marked as not analyzable "
                        "for this music server.", track_name_full
                    )
                    continue
                if exclusion:
                    logger.info(
                        "Retrying '%s' because its provider metadata changed "
                        "since the not-analyzable marker was written.", track_name_full
                    )
                plan = _ah.plan_track_stages(
                    _ah.catalog_item_id(item),
                    existing_track_ids_set,
                    missing_clap_ids_set,
                    missing_lyrics_ids_set,
                    missing_base_ids_set,
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
                except (OperationalError, InterfaceError):
                    raise
                except TrackNotAnalyzable as e:
                    error_manager.record(
                        ERR_TRACK_NOT_ANALYZABLE, str(e), logger=logger, level=logging.WARNING
                    )
                    if e.cacheable:
                        from ..mediaserver import context as server_context

                        server_id = (
                            server_context.active_server_id()
                            or registry.get_default_server_id()
                        )
                        metadata = _ah.analysis_exclusion_metadata(item)
                        record_analysis_exclusion(
                            server_id,
                            provider_id,
                            duration_seconds=metadata['duration_seconds'],
                            title=metadata['title'],
                            artist=metadata['artist'],
                            reason_code=ERR_TRACK_NOT_ANALYZABLE,
                        )
                    tracks_not_analyzable_count += 1
                except TrackSourceUnavailable:
                    logger.warning(
                        "Source audio unavailable for '%s' (deleted or moved on the "
                        "server); skipping it and continuing with the album.",
                        track_name_full,
                    )
                    unavailable_tracks.append(track_name_full)
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

            album_had_reachable_tracks = bool(
                tracks_analyzed_count or tracks_skipped_count or tracks_not_analyzable_count
            )
            if unavailable_tracks and not album_had_reachable_tracks:
                raise RuntimeError(
                    f"Every track in '{album_name}' was unavailable for download "
                    f"({len(unavailable_tracks)}/{total_tracks_in_album}); the album "
                    "source appears entirely missing from the server."
                )

            summary = {
                "tracks_analyzed": tracks_analyzed_count,
                "tracks_skipped": tracks_skipped_count,
                "tracks_not_analyzable": tracks_not_analyzable_count,
                "tracks_unavailable": len(unavailable_tracks),
                "total_tracks_in_album": total_tracks_in_album,
            }
            completion_message = f"Album '{album_name}' analysis complete."
            if tracks_not_analyzable_count:
                completion_message += (
                    f" {tracks_not_analyzable_count}/{total_tracks_in_album} track(s) carried no "
                    "analyzable audio and were skipped."
                )
            if unavailable_tracks:
                completion_message += (
                    f" {len(unavailable_tracks)}/{total_tracks_in_album} track(s) were "
                    "unavailable on the server and were skipped."
                )
            return {"status": TASK_STATUS_SUCCESS, "message": completion_message, **summary}

        except (OperationalError, InterfaceError) as e:
            error_manager.from_exception(e, code=ERR_DB_CONNECTION, logger=logger)
            raise
        except TaskCancelled:
            _ah.flush_pending_track_maps(pending_track_maps, map_flush_errors, album_name)
            raise
        except Exception as e:
            err = error_manager.from_exception(
                e, code=error_manager.classify(e, ERR_ALBUM_ANALYSIS_FAILED), logger=logger
            )
            log_and_update_album_task(
                f"Failed to analyze album '{album_name}': {e}",
                log_and_update_album_task.state['progress'],
                error=err,
            )
            raise
        finally:
            close_cancel()
            cleanup_musicnn_sessions(onnx_sessions, context="finally")
            onnx_sessions = None
            try:
                comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)
            except Exception as e:
                logger.warning(f"Error during final comprehensive cleanup: {e}")
            cleanup_optional_models(context="finally")
