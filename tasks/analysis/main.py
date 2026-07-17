# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Analysis orchestration: FOR EACH SERVER, dispatch FOR EACH ALBUM and drain.

run_analysis_task runs one phase per enabled server (union catalogue, default
first). Each phase loads the server's work map ONCE, walks the albums, enqueues
tasks.analysis.album.analyze_album_task children for the ones with work, drains
them, and rebuilds the indexes at the end. A run fails only if it crashed or
analyzed not one song (error codes 2005/2006/2007); a wiped task_status row IS
the cancellation signal at every level.

Main Features:
* run_analysis_task / run_analysis_server_task: the RQ entry points.
* _run_analysis_server_task_impl: work map -> skip-or-enqueue -> drain -> final
  index rebuild, with revocation polls and DB reconciliation throttled to
  ANALYSIS_MONITOR_DB_INTERVAL.
* _verify_media_server_reachable: pre-flight probe so an unreachable or
  unauthenticated server aborts early with 1101/1104 instead of failing every
  child job.
"""

import os
import shutil
import time
import logging
import uuid

from rq import get_current_job, Retry
from rq.job import Job
from rq.exceptions import NoSuchJobError

from config import (
    TEMP_DIR,
    MAX_QUEUED_ANALYSIS_JOBS,
    LYRICS_ENABLED,
    ANALYSIS_MONITOR_DB_INTERVAL,
    REBUILD_INDEX_BATCH_SIZE,
)

from ..mediaserver import (
    get_recent_albums,
    get_tracks_from_album,
    registry,
    test_connection as mediaserver_test_connection,
)

from flask_app import app
from app_helper import (
    redis_conn,
    rq_queue_default,
    save_task_status,
    get_task_info_from_db,
    get_task_statuses,
    TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
    TASK_STATUS_REVOKED,
)
from database import count_terminal_children, get_failed_child_summary
from redis.exceptions import TimeoutError as RedisTimeoutError
from psycopg2 import OperationalError

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


def clean_temp(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    for name in os.listdir(temp_dir):
        path = os.path.join(temp_dir, name)
        try:
            (shutil.rmtree if os.path.isdir(path) and not os.path.islink(path) else os.unlink)(path)
        except Exception as e:
            logger.warning(f"Could not remove {path} from {temp_dir}: {e}")


def _rq_job_still_pending(job_id):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        return False
    except Exception:
        logger.debug("Could not fetch job %s while reconciling; assuming done.", job_id)
        return False
    return job.is_queued or job.is_scheduled or job.is_started


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
    final_message = f"Albums {final_done}/{reported_total}"
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

    current_job = get_current_job(redis_conn)
    current_task_id = task_id or (current_job.id if current_job else str(uuid.uuid4()))

    with app.app_context():
        if num_recent_albums < 0:
            logger.warning("num_recent_albums is negative, treating as 0 (all albums).")
            num_recent_albums = 0

        task_info = get_task_info_from_db(current_task_id)
        if task_info and task_info.get('status') in [TASK_STATUS_SUCCESS, TASK_STATUS_REVOKED]:
            return {"status": task_info.get('status'), "message": "Task already in terminal state."}

        log_and_update_main = make_task_reporter(
            current_task_id, "main_analysis", current_job,
            "Starting main analysis process...",
            prefix=f"MainAnalysisTask-{current_task_id}",
            progress_base=progress_base, progress_span=progress_span,
            downgrade_terminal=not final_phase,
        )
        try:
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
            except OperationalError:
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
            baseline_failed_count, _baseline_errors = get_failed_child_summary(current_task_id)
            active_jobs = set()
            albums_skipped, albums_launched, albums_completed = 0, 0, 0
            last_rebuild_count = 0
            albums_no_tracks = 0
            albums_work_check_failed = 0
            albums_needing_musicnn = 0
            albums_needing_clap = 0
            albums_needing_lyrics = 0
            songs_seen = 0
            songs_done = 0
            last_monitor_db_check = float('-inf')
            last_status_report = float('-inf')
            last_revocation_poll = float('-inf')
            try:
                completed_baseline = count_terminal_children(current_task_id)
            except Exception:
                logger.exception("Could not read the completed-children baseline")
                completed_baseline = 0

            def revoked_now():
                nonlocal last_revocation_poll
                now = time.monotonic()
                if now - last_revocation_poll < ANALYSIS_MONITOR_DB_INTERVAL:
                    return False
                last_revocation_poll = now
                try:
                    statuses = get_task_statuses([current_task_id])
                except Exception:
                    logger.exception("Revocation poll failed; assuming the run is live")
                    return False
                return statuses.get(current_task_id, TASK_STATUS_REVOKED) == TASK_STATUS_REVOKED

            def monitor_and_clear_jobs():
                nonlocal albums_completed, last_rebuild_count, last_monitor_db_check
                ids = list(active_jobs)
                if ids:
                    try:
                        fetched = Job.fetch_many(ids, connection=redis_conn)
                    except RedisTimeoutError:
                        logger.warning("Redis timeout fetching jobs; retry next loop.")
                        fetched = []
                    except Exception as e:
                        logger.warning(
                            f"Error fetching jobs: {e}; retry next loop.", exc_info=True
                        )
                        fetched = []
                    removed = 0
                    for job_id, job in zip(ids, fetched):
                        if job is None:
                            logger.debug(f"Job {job_id} not in RQ; will reconcile via DB.")
                        elif job.is_finished or job.is_failed or job.is_canceled:
                            active_jobs.discard(job_id)
                            removed += 1
                    if removed:
                        albums_completed += removed

                now = time.monotonic()
                if now - last_monitor_db_check >= ANALYSIS_MONITOR_DB_INTERVAL:
                    last_monitor_db_check = now
                    try:
                        terminal = {TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED}
                        db_done = count_terminal_children(current_task_id) - completed_baseline
                        in_flight = list(active_jobs)
                        if in_flight:
                            statuses = get_task_statuses(in_flight)
                            for job_id in in_flight:
                                if (
                                    statuses.get(job_id) in terminal
                                    and not _rq_job_still_pending(job_id)
                                ):
                                    active_jobs.discard(job_id)
                        if db_done != albums_completed:
                            logger.info(
                                f"Reconciling albums_completed: RQ={albums_completed} DB={db_done} (of {albums_launched} launched)"
                            )
                            albums_completed = max(0, db_done)
                    except Exception:
                        logger.exception("Failed to reconcile child tasks from DB")

                if (
                    finalize_indexes
                    and albums_completed - last_rebuild_count >= REBUILD_INDEX_BATCH_SIZE
                ):
                    rebuild_job = rq_queue_default.enqueue(
                        'tasks.analysis.rebuild_all_indexes_task',
                        job_id=str(uuid.uuid4()),
                        job_timeout=-1,
                        retry=Retry(max=3),
                    )
                    log_and_update_main(
                        f"Batch of {albums_completed - last_rebuild_count} albums complete; "
                        f"index rebuild {rebuild_job.id} enqueued.",
                        log_and_update_main.state['progress'],
                    )
                    last_rebuild_count = albums_completed

            def report_progress(force=False):
                nonlocal last_status_report
                now = time.monotonic()
                if not force and now - last_status_report < 5:
                    return
                last_status_report = now
                done = albums_skipped + albums_completed + albums_work_check_failed
                progress = 5 + int(85 * (done / float(total_albums_to_check)))
                log_and_update_main(
                    f"Albums {albums_offset + done}/{reported_total}",
                    progress,
                    albums_completed=albums_completed,
                )

            all_albums = list({a['Id']: a for a in all_albums}.values())
            for album in all_albums:
                if revoked_now():
                    logger.info("Analysis revoked; stopping album dispatch.")
                    return {'status': TASK_STATUS_REVOKED}
                monitor_and_clear_jobs()
                while len(active_jobs) >= MAX_QUEUED_ANALYSIS_JOBS:
                    if revoked_now():
                        logger.info("Analysis revoked; stopping album dispatch.")
                        return {'status': TASK_STATUS_REVOKED}
                    monitor_and_clear_jobs()
                    report_progress()
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
                    except OperationalError:
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

                job = rq_queue_default.enqueue(
                    'tasks.analysis.analyze_album_task',
                    args=(album['Id'], album['Name'], top_n_moods, current_task_id, server_id),
                    job_id=str(uuid.uuid4()),
                    job_timeout=-1,
                    retry=Retry(max=3),
                )
                active_jobs.add(job.id)
                albums_launched += 1
                albums_needing_musicnn += int(needs_musicnn_analysis)
                albums_needing_clap += int(needs_clap_analysis)
                albums_needing_lyrics += int(needs_lyrics_analysis)
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
                time.sleep(5)

            if finalize_indexes:
                log_and_update_main("Performing final index rebuild...", 95)
                try:
                    _run_all_index_builds(log_fn=log_and_update_main)
                except error_manager.AudioMuseError:
                    raise
                except Exception as e:
                    raise error_manager.AudioMuseError(
                        error_manager.classify(e, ERR_INDEX_BUILD), str(e), cause=e
                    ) from e
            total_failed_count, failed_errors = get_failed_child_summary(current_task_id)
            failed_count = max(0, total_failed_count - baseline_failed_count)
            if not failed_count:
                failed_errors = []
            logger.info(
                "Phase complete. Albums: %d launched, %d skipped of %d, %d failed. "
                "Songs: %d sent for analysis, %d already analyzed of %d. "
                "Feature albums: MusiCNN %d, DCLAP %d, Lyrics %d.",
                albums_launched, albums_skipped, total_albums_to_check, failed_count,
                songs_seen - songs_done, songs_done, songs_seen,
                albums_needing_musicnn, albums_needing_clap, albums_needing_lyrics,
            )
            final_message, phase_status, final_kwargs = _phase_outcome(
                albums_offset + albums_skipped + albums_completed + albums_work_check_failed,
                reported_total, albums_launched, failed_count, failed_errors,
                albums_work_check_failed,
            )
            log_and_update_main(final_message, 100, **final_kwargs)
            clean_temp(TEMP_DIR)
            return {
                "status": phase_status,
                "message": final_message,
                "failed_albums": failed_count,
            }

        except OperationalError as e:
            err = error_manager.from_exception(e, code=ERR_DB_CONNECTION, logger=logger)
            log_and_update_main(
                "X Main analysis failed due to a database connection error. The task may be retried.",
                log_and_update_main.state['progress'],
                task_state=TASK_STATUS_FAILURE,
                error=err,
            )
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
        except Exception:
            logger.exception("Server registry unavailable; analyzing the config default only")
            return [None]


def run_analysis_task(num_recent_albums, top_n_moods, server_scope="all"):
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
            try:
                statuses = get_task_statuses([parent_id])
            except Exception:
                logger.exception("Union revocation poll failed; assuming the run is live")
                statuses = {parent_id: TASK_STATUS_PROGRESS}
            if statuses.get(parent_id, TASK_STATUS_REVOKED) == TASK_STATUS_REVOKED:
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
        except Exception as e:
            failed.append(server['name'])
            error_manager.record(
                error_manager.classify(e, ERR_ANALYSIS_SERVER_FAILED),
                f"{server['name']}: {e}", exc=e, logger=logger, level=logging.WARNING,
            )
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
        details = {"failed_servers": failed}
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
        'status': 'FAILURE' if run_failed else 'SUCCESS',
        'message': message,
        'servers': summaries,
        'failed_servers': failed,
    }
