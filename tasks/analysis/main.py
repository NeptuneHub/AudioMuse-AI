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
    QUEUE_MAX_ERRORS_KEPT,
    REBUILD_INDEX_BATCH_SIZE,
    CHROMAPRINT_COLLECTION_ENABLED,
    CHROMAPRINT_BACKFILL_ALBUMS_PER_RUN,
    CHROMAPRINT_BACKFILL_REPORT_SECONDS,
    CHROMAPRINT_BACKFILL_DISTRIBUTED,
    CHROMAPRINT_BACKFILL_BATCH_SIZE,
    TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
    TASK_STATUS_REVOKED,
)

from ..mediaserver import (
    get_recent_albums,
    get_tracks_from_album,
    download_track,
    registry,
    test_connection as mediaserver_test_connection,
)
from .. import chromaprint

from flask_app import app
from database import (
    persist_chromaprint,
    get_db,
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


def _chromaprint_backfill_targets(server_id, album_limit):
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            "WITH missing AS ("
            "  SELECT m.provider_track_id, m.file_path, s.album "
            "  FROM track_server_map m "
            "  JOIN score s ON s.item_id = m.item_id "
            "  LEFT JOIN chromaprint c "
            "    ON c.server_id = m.server_id AND c.provider_track_id = m.provider_track_id "
            "  WHERE m.server_id = %s AND c.provider_track_id IS NULL "
            "    AND s.album IS NOT NULL AND s.album <> ''"
            "), picked AS ("
            "  SELECT album FROM missing GROUP BY album ORDER BY album LIMIT %s"
            ") "
            "SELECT missing.provider_track_id, missing.file_path "
            "FROM missing JOIN picked ON picked.album = missing.album",
            (str(server_id), album_limit),
        )
        return cur.fetchall()


def _backfill_one_track(server_id, provider_track_id, file_path):
    item = {'Id': provider_track_id, 'id': provider_track_id, 'FilePath': file_path}
    name = os.path.basename(file_path) if file_path else provider_track_id
    path = None
    try:
        path = download_track(TEMP_DIR, item)
        if not path:
            return False
        blob = chromaprint.compute(path)
        persist_chromaprint(server_id, provider_track_id, blob)
        if blob:
            logger.info("Calculated Chromaprint for '%s' (backfill)", name)
            return True
        logger.warning("Could not calculate Chromaprint for '%s' (backfill)", name)
        return False
    except Exception:
        logger.exception(
            "Chromaprint backfill failed for %s/%s", server_id, provider_track_id
        )
        return False
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _noop_progress(message, progress):
    return None


def _backfill_server_chromaprints(server_id, log_fn=None, should_stop=None):
    from ..mediaserver import context as server_context

    targets = _chromaprint_backfill_targets(server_id, CHROMAPRINT_BACKFILL_ALBUMS_PER_RUN)
    if not targets:
        return False
    log_fn = log_fn or _noop_progress
    total = len(targets)
    log_fn(
        f"Calculating Chromaprint fingerprints for {total} track(s) "
        f"on server {server_id}...", 99,
    )
    # Distributing needs a claimed parent task to hang the child jobs on and to reap
    # them from. Without one (a non-queued, in-process run) there is no worker pool to
    # help anyway, so fall through to the serial loop.
    if CHROMAPRINT_BACKFILL_DISTRIBUTED and taskqueue.current_task_id() is not None:
        return _backfill_server_chromaprints_distributed(
            server_id, targets, total, log_fn=log_fn, should_stop=should_stop
        )
    filled = 0
    stopped = False
    last_tick = time.monotonic()
    with server_context.use_server(_bind_server_context(server_id)):
        for done, (provider_track_id, file_path) in enumerate(targets, 1):
            if _backfill_one_track(server_id, provider_track_id, file_path):
                filled += 1
            now = time.monotonic()
            if now - last_tick < CHROMAPRINT_BACKFILL_REPORT_SECONDS:
                continue
            last_tick = now
            if should_stop and should_stop():
                stopped = True
                break
            log_fn(
                f"Calculating Chromaprint fingerprints on server {server_id}: "
                f"{done}/{total} track(s)...", 99,
            )
    logger.info(
        "Chromaprint backfill filled %d of %d track(s) on server %s%s",
        filled, total, server_id, " (cancelled early)" if stopped else "",
    )
    return stopped


_CHROMAPRINT_BACKFILL_DRAIN_POLL_SECONDS = 5


def _enqueue_backfill_batches(
    server_id, targets, total, batch_size, parent_task_id, should_stop, active_jobs
):
    """Chunk the targets and enqueue one chromaprint_backfill batch job per chunk.

    Adds each enqueued job's id to active_jobs. Stops early (without enqueuing the
    remaining chunks) if should_stop fires. Returns (batches_enqueued, stopped).
    """
    batches_enqueued = 0
    for start in range(0, total, batch_size):
        if should_stop and should_stop():
            return batches_enqueued, True
        batch = targets[start:start + batch_size]
        batch_task_id = str(uuid.uuid4())
        taskqueue.enqueue(
            'tasks.analysis.chromaprint_backfill_batch_task',
            args=(server_id, batch, parent_task_id),
            task_id=batch_task_id,
            task_type='chromaprint_backfill',
            queue=taskqueue.QUEUE_DEFAULT,
            parent_task_id=parent_task_id,
            sub_type_identifier=batch_task_id,
        )
        active_jobs.add(batch_task_id)
        batches_enqueued += 1
    return batches_enqueued, False


def _reap_backfill_batches(parent_task_id, active_jobs):
    """Reap finished backfill batches, returning the (filled, done) deltas.

    Drops each completed job from active_jobs and sums the filled/total it published
    in final_summary_details. A reap failure is logged and treated as no progress this
    tick so the caller simply retries on the next poll.
    """
    try:
        reaped = taskqueue.reap_finished_children(parent_task_id)
    except Exception:
        logger.exception("Failed to reap finished Chromaprint backfill batches")
        return 0, 0
    filled = 0
    done = 0
    for child in reaped:
        if child['task_id'] not in active_jobs:
            continue
        active_jobs.discard(child['task_id'])
        details = child.get('details') or {}
        summary = (
            details.get('final_summary_details') if isinstance(details, dict) else None
        )
        if isinstance(summary, dict):
            filled += int(summary.get('filled') or 0)
            done += int(summary.get('total') or 0)
    return filled, done


def _backfill_server_chromaprints_distributed(
    server_id, targets, total, log_fn=None, should_stop=None
):
    """Fan the backfill targets out as chromaprint_backfill child jobs and drain them.

    Mirrors the album phase's enqueue/reap loop: each batch of targets becomes a
    queued chromaprint_backfill_batch_task that any worker can pick up, and the
    parent reaps the finished children to aggregate how many fingerprints were
    filled. Progress is reported in the same format the serial loop uses so the UI
    is unchanged. Revocation stops enqueuing and stops waiting, returning True.
    """
    log_fn = log_fn or _noop_progress
    parent_task_id = taskqueue.current_task_id()
    batch_size = max(1, CHROMAPRINT_BACKFILL_BATCH_SIZE)
    active_jobs = set()

    batches_enqueued, stopped = _enqueue_backfill_batches(
        server_id, targets, total, batch_size, parent_task_id, should_stop, active_jobs
    )

    filled = 0
    done = 0
    while active_jobs:
        if should_stop and should_stop():
            stopped = True
            break
        batch_filled, batch_done = _reap_backfill_batches(parent_task_id, active_jobs)
        filled += batch_filled
        done += batch_done
        log_fn(
            f"Calculating Chromaprint fingerprints on server {server_id}: "
            f"{min(done, total)}/{total} track(s)...", 99,
        )
        if active_jobs:
            time.sleep(_CHROMAPRINT_BACKFILL_DRAIN_POLL_SECONDS)

    logger.info(
        "Chromaprint backfill distributed %d track(s) on server %s across %d batch(es); "
        "filled %d%s",
        total, server_id, batches_enqueued, filled,
        " (cancelled early)" if stopped else "",
    )
    return stopped


def chromaprint_backfill_batch_task(server_id, targets, parent_task_id):
    """Fingerprint one batch of backfill targets, one queue job the workers share.

    A distributed-backfill child job: it binds the same media-server context the
    serial loop uses and runs the unchanged per-track work (download then fpcalc via
    _backfill_one_track) for each (provider_track_id, file_path) in its batch. A track
    that cannot be fingerprinted is counted as not filled but never aborts the batch,
    exactly as the serial loop already tolerates a single failure. The filled/total
    tally is published in final_summary_details so the parent's drain can aggregate it.
    """
    from tasks.mediaserver import context as server_context

    with server_context.use_server(_bind_server_context(server_id)):
        return _chromaprint_backfill_batch_task_impl(server_id, targets, parent_task_id)


def _backfill_batch_parent_unavailable(claimed_task_id, parent_task_id):
    """True when a claimed batch's parent analysis is already gone or terminal.

    Only a queue-claimed batch with a parent can be orphaned; an unclaimed
    (in-process) run has no parent to check, so it is never unavailable.
    """
    if not (claimed_task_id and parent_task_id):
        return False
    parent_statuses = get_task_statuses([parent_task_id])
    return (
        parent_task_id not in parent_statuses
        or parent_statuses.get(parent_task_id)
        in (TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)
    )


def _make_backfill_batch_revocation_check(claimed_task_id, current_task_id, parent_task_id):
    """Build the throttled should-stop predicate for a distributed backfill batch.

    Unclaimed (in-process) runs are never revoked. Claimed runs poll task status at
    most once per ANALYSIS_MONITOR_DB_INTERVAL and stop when their own row or the
    parent's row goes terminal, mirroring the album-analysis revocation contract.
    """
    last_revocation_check = [float('-inf')]

    def revoked():
        if not claimed_task_id:
            return False
        now = time.monotonic()
        if now - last_revocation_check[0] < ANALYSIS_MONITOR_DB_INTERVAL:
            return False
        last_revocation_check[0] = now
        statuses = get_task_statuses([current_task_id, parent_task_id])
        if statuses.get(current_task_id, TASK_STATUS_REVOKED) == TASK_STATUS_REVOKED:
            return True
        parent_status = statuses.get(parent_task_id) if parent_task_id else None
        return parent_status in (TASK_STATUS_REVOKED, TASK_STATUS_FAILURE)

    return revoked


def _fingerprint_backfill_batch(server_id, targets, log_and_update, revoked):
    """Fingerprint each target in one batch, tolerating per-track failures.

    Runs the unchanged per-track work (download then fpcalc via _backfill_one_track);
    a track that cannot be fingerprinted is counted as not filled but never aborts the
    batch. Returns the batch result dict, stopping early with TASK_STATUS_REVOKED the
    moment `revoked()` reports the run was cancelled.
    """
    total = len(targets)
    filled = 0
    for done, target in enumerate(targets, 1):
        if revoked():
            log_and_update(
                f"Chromaprint backfill batch stopped after {done - 1}/{total} "
                f"track(s) on server {server_id}.",
                log_and_update.state['progress'],
                task_state=TASK_STATUS_REVOKED,
            )
            return {"status": TASK_STATUS_REVOKED, "filled": filled, "total": total}
        provider_track_id, file_path = target
        if _backfill_one_track(server_id, provider_track_id, file_path):
            filled += 1
        log_and_update(
            f"Calculating Chromaprint fingerprints on server {server_id}: "
            f"{done}/{total} track(s)...",
            int(100 * (done / float(total))) if total else 100,
        )
    summary = {"filled": filled, "total": total}
    log_and_update(
        f"Chromaprint backfill batch complete: {filled}/{total} filled on "
        f"server {server_id}.",
        100, task_state=TASK_STATUS_SUCCESS,
        final_summary_details=summary,
    )
    return {"status": TASK_STATUS_SUCCESS, **summary}


def _chromaprint_backfill_batch_task_impl(server_id, targets, parent_task_id):
    claimed_task_id = taskqueue.current_task_id()
    current_task_id = claimed_task_id or str(uuid.uuid4())

    with app.app_context():
        if _backfill_batch_parent_unavailable(claimed_task_id, parent_task_id):
            logger.info(
                "Chromaprint backfill batch %s will not start because parent %s "
                "is missing or terminal.", current_task_id, parent_task_id,
            )
            return {
                "status": TASK_STATUS_REVOKED,
                "message": "Parent analysis was cancelled.",
            }

        log_and_update = make_task_reporter(
            current_task_id, "chromaprint_backfill",
            "Chromaprint backfill batch started.",
            parent_task_id=parent_task_id,
            base_details={"batch_size": len(targets)},
            prefix=f"ChromaprintBatch-{current_task_id}",
            min_db_interval=ANALYSIS_MONITOR_DB_INTERVAL,
        )
        revoked = _make_backfill_batch_revocation_check(
            claimed_task_id, current_task_id, parent_task_id
        )

        try:
            return _fingerprint_backfill_batch(
                server_id, targets, log_and_update, revoked
            )
        except (OperationalError, InterfaceError) as e:
            error_manager.from_exception(e, code=ERR_DB_CONNECTION, logger=logger)
            raise
        except Exception as e:
            err = error_manager.from_exception(
                e, code=error_manager.classify(e, ERR_ANALYSIS_FAILED), logger=logger
            )
            log_and_update(
                f"Chromaprint backfill batch failed on server {server_id}: {e}",
                log_and_update.state['progress'],
                task_state=TASK_STATUS_FAILURE,
                error=err,
            )
            raise


def _run_chromaprint_backfill(server_ids, log_fn=None, should_stop=None):
    if not CHROMAPRINT_COLLECTION_ENABLED or not chromaprint.is_available():
        return False
    for server_id in server_ids:
        if not server_id:
            continue
        if should_stop and should_stop():
            logger.info("Chromaprint backfill cancelled before server %s.", server_id)
            return True
        try:
            if _backfill_server_chromaprints(
                server_id, log_fn=log_fn, should_stop=should_stop
            ):
                return True
        except Exception:
            logger.exception("Chromaprint backfill failed for server %s", server_id)
    return False


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
                if _run_chromaprint_backfill(
                    [server_id], log_fn=log_and_update_main, should_stop=revoked_now
                ):
                    logger.info("Analysis revoked during the Chromaprint backfill.")
                    return {'status': TASK_STATUS_REVOKED}
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

        def _chromaprint_progress(message, _progress=99):
            save_task_status(
                parent_id, "main_analysis", TASK_STATUS_PROGRESS,
                progress=99, details={"message": message},
            )

        backfill_cancelled = _run_chromaprint_backfill(
            [server['server_id'] for server in servers if server['name'] not in failed],
            log_fn=_chromaprint_progress,
            should_stop=lambda: _task_revoked_in_db(parent_id),
        )
        if backfill_cancelled:
            logger.info("Union analysis revoked during the Chromaprint backfill.")
            return {'status': 'REVOKED', 'servers_completed': len(summaries)}

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
