# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Analysis orchestration: FOR EACH SERVER, dispatch FOR EACH ALBUM and drain.

run_analysis_task runs one phase per enabled server (union catalogue, default
first): loads the work map ONCE, walks albums, enqueues
tasks.analysis.album.analyze_album_task children, drains them, and rebuilds the
indexes. A run fails only if it crashed or analyzed not one song (2005/2006/2007).

Main Features:
* run_analysis_task / run_analysis_server_task queue entry points.
* _run_analysis_server_task_impl: work map -> skip-or-enqueue -> drain -> rebuild.
* _verify_media_server_reachable: pre-flight probe aborting early (1101/1104).
* _carried_over_tracks: a reclaim requeues the parent (row back to NEW), carrying
  an earlier attempt's analysed songs into this attempt's total.
* BOTH album waits - the dispatch throttle that holds at
  MAX_QUEUED_ANALYSIS_JOBS and the tail drain - watch the shared StallValve on
  ANALYSIS_STALL_TIMEOUT_MINUTES. The throttle is where a wedge on a real library
  actually lands, and its own progress writes keep the parent row fresh, so the
  wedged-main nudge cannot see it either: guarding only the tail left the hang.
  The give-up then fails only the albums a worker is actually HOLDING, never the
  ones still queued behind them: a queued album is not wedged, it is waiting, and
  ending the album that holds the worker is what lets it run. Failing the queue
  too would burn a whole MAX_QUEUED_ANALYSIS_JOBS window of untouched albums. An album whose worker is alive but whose native
  code never returns holds its advisory lock, so reclaim cannot take it and the
  parent would wait on it forever. The window slides on any sign of life, a live
  album advancing one track included, so only a wedged album runs it out; it is
  then FAILED (not revoked) so the ordinary reap counts it into the album failure
  tally the run reports, instead of vanishing from the totals.
* A swept track keeps its Chromaprint: upsert_track_maps hands each new mapping
  the Chromaprint already stored for its canonical track in the SAME transaction
  it is written, so a sweep never leaves a mapping fingerprintless.

TEMP_DIR is SHARED by every worker, so the start-of-run wipe is gated on this
task having no live children; if they cannot be read the wipe is skipped.
"""

import os
import shutil
import time
import logging
import uuid

import taskqueue


from config import (
    TEMP_DIR,
    MAX_QUEUED_ANALYSIS_JOBS,
    LYRICS_ENABLED,
    ANALYSIS_MONITOR_DB_INTERVAL,
    ANALYSIS_STALL_TIMEOUT_MINUTES,
    QUEUE_MAX_ERRORS_KEPT,
    REBUILD_INDEX_BATCH_SIZE,
    TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
    TASK_STATUS_REVOKED,
    TASK_STATUS_RUNNING,
)

from ..mediaserver import (
    get_recent_albums,
    get_tracks_from_album,
    registry,
    test_connection as mediaserver_test_connection,
)

from flask_app import app
from database import (
    save_task_status,
    get_task_statuses,
)
from psycopg2 import InterfaceError, OperationalError

from error import error_manager
from error.error_dictionary import (
    ERR_ANALYSIS_FAILED,
    ERR_ANALYSIS_NO_TRACKS_ANALYZED,
    ERR_ANALYSIS_SERVER_FAILED,
    ERR_DB_CONNECTION,
    ERR_MEDIASERVER_LIBRARY,
    ERR_MEDIASERVER_AUTH,
    ERR_MEDIASERVER_UNREACHABLE,
    ERR_INDEX_BUILD,
)

from . import helper as _ah
from .helper import make_task_reporter, _bind_server_context


def _run_all_index_builds(*args, **kwargs):
    from .index import _run_all_index_builds as impl

    return impl(*args, **kwargs)


logger = logging.getLogger(__name__)


def _carried_over_tracks(parent_task_id):
    try:
        finished = taskqueue.reap_finished_children(parent_task_id)
    except Exception:
        logger.exception(
            "Could not clear the finished album jobs of a previous attempt; this "
            "run's failure tally may include theirs"
        )
        return 0
    if not finished:
        return 0
    carried = 0
    failed = 0
    for child in finished:
        if not child.get('sub_type_identifier'):
            continue
        if child.get('status') != TASK_STATUS_SUCCESS:
            failed += 1
            continue
        details = child.get('details')
        if not isinstance(details, dict):
            continue
        summary = details.get('final_summary_details')
        counted = summary.get('tracks_analyzed') if isinstance(summary, dict) else None
        if counted is None:
            counted = details.get('tracks_analyzed')
        if isinstance(counted, (int, float)):
            carried += int(counted)
    logger.info(
        "A previous attempt of this task had finished %d album job(s): carrying %d "
        "analyzed song(s) into this attempt's total and dropping %d failure(s).",
        len(finished), carried, failed,
    )
    return carried


def _inflight_children(parent_task_id):
    try:
        return taskqueue.live_children(parent_task_id)
    except Exception:
        logger.exception(
            "Could not check for a previous attempt's in-flight album jobs; "
            "any still running will be re-enqueued and deduplicated per track"
        )
        return None


def clean_temp(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    for name in os.listdir(temp_dir):
        path = os.path.join(temp_dir, name)
        try:
            (shutil.rmtree if os.path.isdir(path) and not os.path.islink(path) else os.unlink)(path)
        except Exception as e:
            logger.warning(f"Could not remove {path} from {temp_dir}: {e}")


def _task_revoked_in_db(task_id):
    try:
        statuses = get_task_statuses([task_id])
    except Exception:
        logger.exception("Revocation poll failed; assuming the run is live")
        return False
    return statuses.get(task_id, TASK_STATUS_REVOKED) == TASK_STATUS_REVOKED


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


def _phase_outcome(final_done, reported_total, albums_launched, failed_count,
                   failed_errors, albums_work_check_failed):
    final_message = f"Albums {min(final_done, reported_total)}/{reported_total}"
    if failed_count:
        final_message += f" ({failed_count} could not be analyzed)"
    if albums_work_check_failed:
        final_message += f" ({albums_work_check_failed} could not be checked)"

    nothing_analyzed = (
        (albums_launched > 0 and failed_count >= albums_launched)
        or (albums_launched == 0 and albums_work_check_failed > 0)
    )
    phase_status = TASK_STATUS_FAILURE if nothing_analyzed else TASK_STATUS_SUCCESS

    final_kwargs = {"task_state": phase_status}
    if failed_count:
        final_kwargs["failed_albums"] = failed_count
        final_kwargs["failed_album_errors"] = failed_errors
    if albums_work_check_failed:
        final_kwargs["albums_work_check_failed"] = albums_work_check_failed
    if nothing_analyzed:
        reason = (
            f"All {albums_launched} album(s) queued for analysis failed."
            if albums_launched
            else f"{albums_work_check_failed} album(s) could not be checked and "
                 "none was analyzed."
        )
        final_kwargs["error"] = error_manager.record(
            ERR_ANALYSIS_NO_TRACKS_ANALYZED, reason, logger=logger,
        )
    return final_message, phase_status, final_kwargs


def run_analysis_server_task(num_recent_albums, top_n_moods, server_id=None, **kwargs):
    from tasks.mediaserver import context as server_context

    with server_context.use_server(_bind_server_context(server_id)):
        return _run_analysis_server_task_impl(
            num_recent_albums, top_n_moods, server_id=server_id, **kwargs
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
    from ..clap_analyzer import is_clap_available
    from ..task_run import task_run_prologue, terminal_skip

    with app.app_context():
        if num_recent_albums < 0:
            logger.warning("num_recent_albums is negative, treating as 0 (all albums).")
            num_recent_albums = 0

        claimed_task_id, current_task_id, task_info = task_run_prologue(task_id)
        skip = terminal_skip(
            current_task_id, claimed_task_id, task_info,
            revoked_message="Task was cancelled before execution.",
            terminal_message="Task already in terminal state.",
        )
        if skip is not None:
            return skip

        log_and_update_main = make_task_reporter(
            current_task_id, "main_analysis",
            "Starting main analysis process...",
            prefix=f"MainAnalysisTask-{current_task_id}",
            progress_base=progress_base, progress_span=progress_span,
            downgrade_terminal=not final_phase,
        )
        try:
            carried_over_tracks = _carried_over_tracks(current_task_id)
            inflight_children = _inflight_children(current_task_id)
            if inflight_children is not None and not inflight_children:
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
            wm_server_id = server_id or registry.get_default_server_id()
            try:
                work_map = _ah.load_server_work_map(
                    wm_server_id, clap_available, LYRICS_ENABLED
                )
                work_map_bulk_ok = True
            except (OperationalError, InterfaceError):
                raise
            except Exception:
                logger.warning(
                    "Bulk work-map scan failed for server %s; falling back to "
                    "per-album checks so one scan error does not abort the phase.",
                    wm_server_id, exc_info=True,
                )
                work_map = {}
                work_map_bulk_ok = False
            done_bits = _ah.work_done_bits(clap_available, LYRICS_ENABLED)
            logger.info(
                "Work map for this server: %d provider tracks already known%s.",
                len(work_map),
                "" if work_map_bulk_ok else " (bulk scan FAILED; per-album fallback)",
            )
            failed_count = 0
            failed_errors = []

            def _remember_album_error(child):
                nonlocal failed_count
                failed_count += 1
                if len(failed_errors) >= QUEUE_MAX_ERRORS_KEPT:
                    return
                album = child.get('sub_type_identifier') or child.get('task_id')
                detail = child.get('details') or {}
                reason = (
                    detail.get('error', {}).get('error_message')
                    if isinstance(detail.get('error'), dict) else detail.get('error')
                ) or detail.get('message') or 'analysis failed'
                failed_errors.append(f"Album {album}: {reason}")

            active_jobs = set()
            albums_skipped, albums_launched, albums_completed = 0, 0, 0
            tracks_analyzed_total = [carried_over_tracks]
            last_rebuild_count = 0
            albums_no_tracks = 0
            albums_work_check_failed = 0
            albums_needing_musicnn = 0
            albums_needing_clap = 0
            albums_needing_lyrics = 0
            albums_needing_base = 0
            songs_seen = 0
            songs_done = 0
            last_monitor_db_check = float('-inf')
            last_status_report = float('-inf')
            last_revocation_poll = float('-inf')
            adopted_albums = set()
            for child in (inflight_children or ()):
                if not child['sub_type_identifier']:
                    continue
                active_jobs.add(child['task_id'])
                adopted_albums.add(str(child['sub_type_identifier']))
            if active_jobs:
                albums_launched += len(active_jobs)
                logger.info(
                    "Adopted %d still-running album job(s) from a previous "
                    "attempt of this task; their albums will not be enqueued again.",
                    len(active_jobs),
                )

            def revoked_now():
                nonlocal last_revocation_poll
                now = time.monotonic()
                if now - last_revocation_poll < ANALYSIS_MONITOR_DB_INTERVAL:
                    return False
                last_revocation_poll = now
                return _task_revoked_in_db(current_task_id)

            def monitor_and_clear_jobs():
                nonlocal albums_completed, last_rebuild_count, last_monitor_db_check
                now = time.monotonic()
                if now - last_monitor_db_check >= ANALYSIS_MONITOR_DB_INTERVAL:
                    last_monitor_db_check = now
                    try:
                        for child in taskqueue.reap_finished_children(current_task_id):
                            active_jobs.discard(child['task_id'])
                            if not child.get('sub_type_identifier'):
                                continue
                            albums_completed += 1
                            child_details = child.get('details') or {}
                            if isinstance(child_details, dict):
                                child_summary = child_details.get('final_summary_details')
                                counted = (
                                    child_summary.get('tracks_analyzed')
                                    if isinstance(child_summary, dict) else None
                                )
                                if counted is None:
                                    counted = child_details.get('tracks_analyzed')
                                if isinstance(counted, (int, float)):
                                    tracks_analyzed_total[0] += int(counted)
                            if child['status'] == TASK_STATUS_FAILURE:
                                _remember_album_error(child)
                    except Exception:
                        logger.exception("Failed to reap finished album tasks")

                if (
                    finalize_indexes
                    and albums_completed - last_rebuild_count >= REBUILD_INDEX_BATCH_SIZE
                ):
                    rebuild_task_id = str(uuid.uuid4())
                    taskqueue.enqueue(
                        'tasks.analysis.rebuild_all_indexes_task',
                        args=(current_task_id,),
                        task_id=rebuild_task_id,
                        task_type='index_rebuild',
                        queue=taskqueue.QUEUE_DEFAULT,
                        parent_task_id=current_task_id,
                    )
                    log_and_update_main(
                        f"Batch of {albums_completed - last_rebuild_count} albums complete; "
                        f"index rebuild {rebuild_task_id} enqueued.",
                        log_and_update_main.state['progress'],
                    )
                    last_rebuild_count = albums_completed

            def _live_album_marks():
                return tuple(sorted(
                    (
                        str(child.get('task_id')),
                        str(child.get('status') or ''),
                        child.get('progress'),
                    )
                    for child in taskqueue.live_children(current_task_id)
                    if child.get('task_id') in active_jobs
                ))

            def _give_up_on_stalled_albums(stalled_minutes, marks):
                message = (
                    'The analysis gave up on this album: nothing anywhere in the run '
                    f'changed for {stalled_minutes:.0f} minutes, so it stopped waiting '
                    'rather than hang the whole analysis on it.'
                )
                held = [
                    task_id for task_id, status, _progress in marks
                    if status == TASK_STATUS_RUNNING
                ]
                victims = held or sorted(active_jobs)
                spared = len(active_jobs) - len(victims)
                abandoned = 0
                for job_id in victims:
                    try:
                        save_task_status(
                            job_id, 'album_analysis', TASK_STATUS_FAILURE, progress=100,
                            parent_task_id=current_task_id,
                            details={'message': message},
                        )
                        taskqueue.request_cancel(job_id)
                        abandoned += 1
                    except Exception:
                        logger.exception("Could not cancel the album job %s", job_id)
                logger.warning(
                    "Analysis %s made no progress of any kind for %.0f minutes (limit: "
                    "%d minutes); gave up on %d of %d unfinished album job(s) and left "
                    "%d that no worker had started yet to run once one is free again.",
                    current_task_id, stalled_minutes, ANALYSIS_STALL_TIMEOUT_MINUTES,
                    abandoned, len(active_jobs), spared,
                )
                return abandoned

            from ..task_run import StallValve

            drain_valve = StallValve(
                ANALYSIS_STALL_TIMEOUT_MINUTES, lambda: time.monotonic()
            )
            last_marks_check = float('-inf')

            def watch_for_a_wedged_album():
                nonlocal last_marks_check
                now = time.monotonic()
                if now - last_marks_check < ANALYSIS_MONITOR_DB_INTERVAL:
                    return
                last_marks_check = now
                try:
                    marks = _live_album_marks()
                except Exception:
                    logger.exception("Could not list the live album jobs; retrying")
                    drain_valve.restart()
                    return
                if drain_valve.moved(marks) or not drain_valve.expired():
                    return
                stalled_minutes = drain_valve.stalled_minutes()
                drain_valve.restart()
                _give_up_on_stalled_albums(stalled_minutes, marks)

            def report_progress(force=False):
                nonlocal last_status_report
                now = time.monotonic()
                if not force and now - last_status_report < 5:
                    return
                last_status_report = now
                done = min(
                    albums_skipped + albums_completed + albums_work_check_failed,
                    total_albums_to_check,
                )
                progress = 5 + int(85 * (done / float(total_albums_to_check)))
                log_and_update_main(
                    f"Albums {min(albums_offset + done, reported_total)}/{reported_total}",
                    progress,
                    albums_completed=albums_completed,
                    tracks_analyzed=tracks_analyzed_total[0],
                )

            all_albums = list({a['Id']: a for a in all_albums}.values())
            for album in all_albums:
                if revoked_now():
                    logger.info("Analysis revoked; stopping album dispatch.")
                    return {'status': TASK_STATUS_REVOKED}
                monitor_and_clear_jobs()
                if str(album['Id']) in adopted_albums:
                    report_progress()
                    continue
                while len(active_jobs) >= MAX_QUEUED_ANALYSIS_JOBS:
                    if revoked_now():
                        logger.info("Analysis revoked; stopping album dispatch.")
                        return {'status': TASK_STATUS_REVOKED}
                    monitor_and_clear_jobs()
                    report_progress()
                    watch_for_a_wedged_album()
                    time.sleep(5)

                tracks = get_tracks_from_album(album['Id'])
                if not tracks:
                    albums_skipped += 1
                    albums_no_tracks += 1
                    logger.info(
                        f"Skipping album '{album.get('Name')}' (ID: {album.get('Id')}) - no tracks returned by media server."
                    )
                    report_progress()
                    continue

                ids = [_ah.provider_item_id(t) for t in tracks]
                if work_map_bulk_ok:
                    masks = [work_map.get(i, 0) for i in ids]
                else:
                    try:
                        am = _ah.album_work_masks(
                            ids, wm_server_id, clap_available, LYRICS_ENABLED
                        )
                    except (OperationalError, InterfaceError):
                        raise
                    except Exception:
                        logger.warning(
                            "Per-album work check failed for album '%s'; skipping it this run.",
                            album.get('Name'), exc_info=True,
                        )
                        albums_work_check_failed += 1
                        report_progress()
                        continue
                    masks = [am.get(i, 0) for i in ids]
                (
                    album_done,
                    needs_musicnn_analysis,
                    needs_clap_analysis,
                    needs_lyrics_analysis,
                    needs_base_analysis,
                ) = _ah.album_feature_needs(masks, done_bits, clap_available, LYRICS_ENABLED)
                songs_seen += len(tracks)
                songs_done += album_done

                if album_done == len(tracks):
                    albums_skipped += 1
                    status_parts = _ah.build_feature_status_parts(
                        clap_available, LYRICS_ENABLED
                    )
                    logger.info(
                        f"Skipping album '{album.get('Name')}' (ID: {album.get('Id')}) - all {len(tracks)} tracks already analyzed ({' + '.join(status_parts)})."
                    )
                    report_progress()
                    continue

                album_task_id = str(uuid.uuid4())
                taskqueue.enqueue(
                    'tasks.analysis.analyze_album_task',
                    args=(album['Id'], album['Name'], top_n_moods, current_task_id, server_id),
                    task_id=album_task_id,
                    task_type='album_analysis',
                    queue=taskqueue.QUEUE_DEFAULT,
                    parent_task_id=current_task_id,
                    sub_type_identifier=album['Id'],
                )
                active_jobs.add(album_task_id)
                albums_launched += 1
                albums_needing_musicnn += int(needs_musicnn_analysis)
                albums_needing_clap += int(needs_clap_analysis)
                albums_needing_lyrics += int(needs_lyrics_analysis)
                albums_needing_base += int(needs_base_analysis)
                report_progress()

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

            work_map = None
            all_albums = None

            while active_jobs:
                if revoked_now():
                    logger.info("Analysis revoked; abandoning the drain loop.")
                    return {'status': TASK_STATUS_REVOKED}
                monitor_and_clear_jobs()
                report_progress(force=True)
                if not active_jobs:
                    break
                watch_for_a_wedged_album()
                time.sleep(5)

            if finalize_indexes:
                log_and_update_main("Performing final index rebuild...", 95)
                try:
                    _run_all_index_builds(log_fn=log_and_update_main)
                except (OperationalError, InterfaceError):
                    raise
                except error_manager.AudioMuseError:
                    raise
                except Exception as e:
                    raise error_manager.AudioMuseError(
                        error_manager.classify(e, ERR_INDEX_BUILD), str(e), cause=e
                    ) from e
            logger.info(
                "Phase complete. Albums: %d launched, %d skipped of %d, %d failed. "
                "Songs: %d sent for analysis, %d already analyzed of %d. "
                "Feature albums: Base %d, MusiCNN %d, DCLAP %d, Lyrics %d.",
                albums_launched, albums_skipped, total_albums_to_check, failed_count,
                songs_seen - songs_done, songs_done, songs_seen,
                albums_needing_base, albums_needing_musicnn,
                albums_needing_clap, albums_needing_lyrics,
            )
            final_message, phase_status, final_kwargs = _phase_outcome(
                albums_offset + albums_skipped + albums_completed + albums_work_check_failed,
                reported_total, albums_launched, failed_count, failed_errors,
                albums_work_check_failed,
            )
            log_and_update_main(
                final_message, 100,
                albums_completed=albums_completed,
                tracks_analyzed=tracks_analyzed_total[0],
                **final_kwargs,
            )
            clean_temp(TEMP_DIR)
            return {
                "status": phase_status,
                "message": final_message,
                "failed_albums": failed_count,
                "albums_completed": albums_completed,
                "tracks_analyzed": tracks_analyzed_total[0],
            }

        except (OperationalError, InterfaceError) as e:
            error_manager.from_exception(e, code=ERR_DB_CONNECTION, logger=logger)
            raise
        except Exception as e:
            err = error_manager.from_exception(
                e, code=error_manager.classify(e, ERR_ANALYSIS_FAILED), logger=logger
            )
            log_and_update_main(
                f"X Main analysis failed: {e}",
                log_and_update_main.state['progress'],
                task_state=TASK_STATUS_FAILURE,
                error=err,
            )
            raise


def _albums_per_server(servers, num_recent_albums):
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
        except (OperationalError, InterfaceError):
            raise
        except Exception:
            logger.exception("Server registry unavailable; analyzing the config default only")
            return [None]


def _run_already_finished(task_id, *, require_claim=False):
    with app.app_context():
        try:
            statuses = get_task_statuses([task_id])
        except Exception:
            logger.exception("Could not read the run's own status; assuming it is live")
            return None
    status = statuses.get(task_id)
    if require_claim and task_id not in statuses:
        logger.info(
            "Analysis %s has no live DB claim; treating the dequeued queue job as revoked.",
            task_id,
        )
        return TASK_STATUS_REVOKED
    if status in (TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED):
        logger.info(
            "Analysis %s is already %s; refusing to run. A cancelled, failed or "
            "completed task must never restart, even if something requeued its job.",
            task_id, status,
        )
        return status
    return None


def run_analysis_task(num_recent_albums, top_n_moods, server_scope="all"):
    claimed_task_id = taskqueue.current_task_id()
    parent_id = claimed_task_id or str(uuid.uuid4())

    already = _run_already_finished(parent_id, require_claim=claimed_task_id is not None)
    if already:
        return {'status': already, 'message': 'Task already in terminal state.'}

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
            if _task_revoked_in_db(parent_id):
                logger.info("Union analysis revoked; stopping before phase %d.", index + 1)
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
            phase_status = phase_summary.get('status')
            if phase_status == TASK_STATUS_REVOKED:
                return {'status': 'REVOKED', 'servers_completed': len(summaries)}
            if phase_status != TASK_STATUS_SUCCESS:
                failed.append(server['name'])
        except (OperationalError, InterfaceError) as e:
            error_manager.from_exception(e, code=ERR_DB_CONNECTION, logger=logger)
            raise
        except Exception as e:
            failed.append(server['name'])
            error_manager.record(
                error_manager.classify(e, ERR_ANALYSIS_SERVER_FAILED),
                f"{server['name']}: {e}", exc=e, logger=logger, level=logging.WARNING,
            )
        albums_offset += len(albums_by_server[index] or [])

    already = _run_already_finished(parent_id)
    if already:
        return {'status': already, 'servers_completed': len(summaries)}

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
        except (OperationalError, InterfaceError) as e:
            error_manager.from_exception(e, code=ERR_DB_CONNECTION, logger=logger)
            raise
        except Exception as e:
            err = error_manager.record(
                error_manager.classify(e, ERR_INDEX_BUILD), str(e), exc=e, logger=logger
            )
            save_task_status(
                parent_id,
                "main_analysis",
                TASK_STATUS_FAILURE,
                progress=100,
                details={
                    "message": (
                        "The analysis finished, but the final similarity index rebuild "
                        "failed. Check the container logs."
                    ),
                    "failed_servers": failed,
                    "error": err,
                },
            )
            raise

        analyzed_servers = len(servers) - len(failed)
        run_failed = analyzed_servers == 0
        details = {
            "failed_servers": failed,
            "tracks_analyzed": sum(
                int(s.get('tracks_analyzed') or 0) for s in summaries
            ),
            "albums_completed": sum(
                int(s.get('albums_completed') or 0) for s in summaries
            ),
        }
        if not failed:
            message = f"Analysis complete across all {len(servers)} music servers."
        elif run_failed:
            message = (
                f"Analysis could not be completed: all {len(servers)} music servers failed "
                f"({', '.join(failed)})."
            )
            details["error"] = error_manager.record(
                ERR_ANALYSIS_SERVER_FAILED,
                f"Every music server failed: {', '.join(failed)}.",
                logger=logger,
            )
        else:
            message = (
                f"Analysis complete for {analyzed_servers} of {len(servers)} music servers. "
                f"Could not analyze: {', '.join(failed)}."
            )
        details["message"] = message
        save_task_status(
            parent_id,
            "main_analysis",
            TASK_STATUS_FAILURE if run_failed else TASK_STATUS_SUCCESS,
            progress=100,
            details=details,
        )
    return {
        'status': TASK_STATUS_FAILURE if run_failed else TASK_STATUS_SUCCESS,
        'message': message,
        'servers': summaries,
        'failed_servers': failed,
    }
