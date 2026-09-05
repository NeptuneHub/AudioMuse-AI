# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Library cleanup task: unbind server mappings for tracks a server no longer has.

Runs as a queue job. Fetches the current track set of every configured media
server through the sweep's OWN enumeration and pruning
(multiserver_sync.fetch_server_catalogue / prune_stale_mappings, library filter
applied), so the prune baseline can never disagree with the enumeration that
created the mappings, and removes ONLY that server's rows from track_server_map
for tracks it no longer has. A song bound to NO server (an orphan) is deleted
from the catalogue, and that delete happens ONLY when every server was read
completely (none failed, empty or partial). Every run then executes the same
full similarity-index rebuild analysis runs, INLINE, and is not reported
complete until Flask reloads the indexes.

Main Features:
* identify_and_clean_orphaned_albums_task: the queue entry point.
* Reuses the sweep's public helpers so cleaning and the sweep never drift apart.
* Refreshes each server's stored library size (music_servers.track_count).
* Runs the Chromaprint dedup (Path B) each time: splits merged groups whose
  stored fingerprints prove they are different recordings (skip-if-missing).
* Cleaning is a MAIN_TASK_TYPE, so it holds the one-live-main index and the
  wedged-main nudge watches its row. Both of its opaque phases therefore hold a
  row_heartbeat: the one whole-catalogue fetch PER SERVER, and the final index
  rebuild. Each is a single call that writes no row while it runs, which the
  nudge cannot tell from a wedge; both are bounded, so a fetch that really never
  returns is still handed back to it.
* A run that could not read every server ends in CleaningIncomplete, which is a
  TaskFailed: the run itself COMPLETED (every reachable server was cleaned and
  the summary is on the row), the orphan delete needs every server read, and a
  queue retry would repeat one whole-catalogue fetch per server plus the full
  index rebuild while holding the one-live-main slot, to reach the same result.
  The next cron run is the retry.
"""

import logging
from collections import defaultdict

from taskqueue import TaskFailed
from config import (
    CLEANING_SAFETY_LIMIT,
    CLEANING_CATALOGUE,
    CHROMAPRINT_GATE_ENABLED,
    QUEUE_WEDGED_MAIN_TASK_MINUTES,
    TASK_STATUS_SUCCESS,
)

from error import error_manager
from error.error_dictionary import ERR_CLEANING_FAILED, ERR_INDEX_BUILD

from .mediaserver import registry
from .recovery import row_heartbeat, slow_step_budget_minutes

from psycopg2 import OperationalError

logger = logging.getLogger(__name__)

STARTING_MESSAGE = "Starting per-server library cleanup..."


class CleaningIncomplete(TaskFailed):
    pass


def identify_and_clean_orphaned_albums_task(clean_catalogue=None):
    clean_catalogue = CLEANING_CATALOGUE if clean_catalogue is None else bool(clean_catalogue)

    from flask_app import app
    from database import (
        get_db,
        delete_stale_analysis_exclusions,
    )
    from .multiserver_sync import (
        fetch_server_catalogue,
        prune_stale_mappings,
        _store_server_track_count,
    )

    from .task_run import (
        TaskCancelled, task_run_prologue, terminal_skip, cancel_guard, make_task_reporter,
    )

    with app.app_context():
        claimed_task_id, current_task_id, task_info = task_run_prologue()
        skip = terminal_skip(
            current_task_id, claimed_task_id, task_info,
            revoked_message="Library cleanup was cancelled before execution.",
            terminal_message="Library cleanup is already terminal.",
        )
        if skip is not None:
            return skip
        log_and_update_main = make_task_reporter(
            current_task_id, "cleaning", STARTING_MESSAGE,
            prefix=f"CleaningTask-{current_task_id}",
        )

        with cancel_guard(claimed_task_id) as cancel:
            try:
                cancel(force=True)
                log_and_update_main(STARTING_MESSAGE, 5)

                servers = registry.servers_for_scope('all')
                present_canonical_ids = set()
                failed_servers = []
                refused_servers = []
                unbound_total = 0
                unbound_by_server = {}
                total_tracks_on_servers = 0
                deleted_analysis_exclusions = 0

                for server_idx, server in enumerate(servers):
                    cancel()
                    server_name = server['name'] if server else 'default server'
                    server_id = server['server_id'] if server else None
                    window_start = 10 + int(70 * server_idx / len(servers))
                    log_and_update_main(
                        f"Fetching the track list from {server_name}...", window_start
                    )
                    try:
                        with row_heartbeat(
                            current_task_id,
                            f"fetching the whole track list of {server_name}, one call "
                            "that writes no row until it returns",
                            stop_after_minutes=slow_step_budget_minutes(
                                QUEUE_WEDGED_MAIN_TASK_MINUTES
                            ),
                        ):
                            tracks = fetch_server_catalogue(server)
                    except Exception:
                        logger.exception(f"Failed to fetch the library from {server_name}")
                        failed_servers.append(server_name)
                        continue
                    if not tracks:
                        logger.warning(
                            f"No tracks found on {server_name}; skipping its cleanup "
                            "so a fetch problem cannot unbind everything."
                        )
                        failed_servers.append(server_name)
                        continue
                    provider_ids = {str(t['id']) for t in tracks if t.get('id')}
                    tracks = None
                    total_tracks_on_servers += len(provider_ids)
                    log_and_update_main(
                        f"Found {len(provider_ids)} tracks on {server_name}",
                        window_start + int(35 / len(servers)),
                    )

                    refused = []
                    if server_id:
                        _store_server_track_count(get_db(), server_id, len(provider_ids))
                        unbound = prune_stale_mappings(
                            get_db(), server_id, sorted(provider_ids), refused=refused
                        )
                        if refused:
                            refused_servers.append(server_name)
                        unbound_by_server[server_name] = unbound
                        unbound_total += unbound
                        if unbound:
                            log_and_update_main(
                                f"Unbound {unbound} tracks no longer on {server_name} "
                                "(kept in the shared catalogue).",
                                window_start + int(70 / len(servers)),
                            )
                    marker_server_id = server_id or registry.get_default_server_id()
                    if marker_server_id and not refused:
                        deleted_analysis_exclusions += delete_stale_analysis_exclusions(
                            marker_server_id, provider_ids, conn=get_db()
                        )
                    provider_list = sorted(provider_ids)
                    for start in range(0, len(provider_list), 5000):
                        cancel()
                        chunk = provider_list[start:start + 5000]
                        mapping = registry.reverse_translate_ids(chunk, server_id)
                        present_canonical_ids.update(str(v) for v in mapping.values())

                log_and_update_main("Checking for catalogue tracks bound to no server...", 85)
                with get_db() as conn, conn.cursor() as cur:
                    cur.execute(
                        "SELECT s.item_id FROM score s "
                        "JOIN embedding e ON s.item_id = e.item_id"
                    )
                    database_track_ids = {row[0] for row in cur.fetchall()}

                fully_unbound = (
                    database_track_ids - present_canonical_ids if not failed_servers else set()
                )
                orphaned_albums_info = defaultdict(lambda: {"tracks": [], "track_count": 0})
                report_ids = list(fully_unbound)[:CLEANING_SAFETY_LIMIT * 50]
                if report_ids:
                    with get_db() as conn, conn.cursor() as cur:
                        for start in range(0, len(report_ids), 5000):
                            chunk = report_ids[start:start + 5000]
                            cur.execute(
                                "SELECT item_id, title, author FROM score WHERE item_id = ANY(%s)",
                                (chunk,),
                            )
                            for track_id, title, author in cur.fetchall():
                                album_key = f"{author}" if author else "Unknown Artist"
                                orphaned_albums_info[album_key]["tracks"].append(
                                    {"item_id": track_id, "title": title, "author": author}
                                )
                                orphaned_albums_info[album_key]["track_count"] += 1

                orphaned_albums_list = [
                    {"artist": artist, "track_count": info["track_count"], "tracks": info["tracks"]}
                    for artist, info in orphaned_albums_info.items()
                ]
                orphaned_albums_list.sort(key=lambda x: x["track_count"], reverse=True)
                orphaned_albums_list = orphaned_albums_list[:CLEANING_SAFETY_LIMIT]

                deleted_count = 0
                deletable = (
                    clean_catalogue and bool(fully_unbound)
                    and not failed_servers and not refused_servers
                )
                if deletable and len(fully_unbound) > len(database_track_ids) // 2:
                    logger.warning(
                        "Cleaning: %d of %d catalogue tracks look orphaned - too large a "
                        "share for a healthy library; deleting nothing this run.",
                        len(fully_unbound), len(database_track_ids),
                    )
                    deletable = False
                if deletable:
                    orphan_ids = list(fully_unbound)
                    with get_db() as conn, conn.cursor() as cur:
                        for start in range(0, len(orphan_ids), 5000):
                            cancel()
                            chunk = orphan_ids[start:start + 5000]
                            cur.execute(
                                "DELETE FROM score WHERE item_id = ANY(%s)", (chunk,)
                            )
                            deleted_count += len(chunk)
                    log_and_update_main(
                        f"Deleted {deleted_count} orphaned catalogue tracks (on no "
                        "server); their analysis is re-created if the files return.",
                        90,
                    )

                chromaprint_splits = 0
                if CHROMAPRINT_GATE_ENABLED:
                    log_and_update_main("Re-checking merged duplicates against Chromaprint...", 91)
                    from .duplicate_repair import split_chromaprint_false_merges
                    cp_result = split_chromaprint_false_merges() or {}
                    chromaprint_splits = cp_result.get('split', 0)
                    if chromaprint_splits:
                        log_and_update_main(
                            f"Thanks to Chromaprint, {chromaprint_splits} false merge(s) were "
                            "split into separate songs; each re-analyzes under its own id.",
                            91,
                        )

                from .analysis.index import _run_all_index_builds
                log_and_update_main("Performing final index rebuild...", 92)
                try:
                    _run_all_index_builds(
                        log_fn=log_and_update_main, progress_start=92, progress_end=99,
                        task_id=current_task_id,
                    )
                except error_manager.AudioMuseError:
                    raise
                except Exception as e:
                    raise error_manager.AudioMuseError(
                        error_manager.classify(e, ERR_INDEX_BUILD), str(e), cause=e
                    ) from e

                summary = {
                    "total_media_server_tracks": total_tracks_on_servers,
                    "total_catalogue_tracks_present": len(present_canonical_ids),
                    "total_database_tracks": len(database_track_ids),
                    "orphaned_tracks_count": len(fully_unbound),
                    "orphaned_albums_count": len(orphaned_albums_list),
                    "orphaned_albums": orphaned_albums_list,
                    "unbound_mappings": unbound_total,
                    "unbound_by_server": unbound_by_server,
                    "failed_servers": failed_servers,
                    "prune_refused_servers": refused_servers,
                    "deleted_count": deleted_count,
                    "deleted_analysis_exclusions": deleted_analysis_exclusions,
                    "catalogue_deletion": clean_catalogue,
                    "chromaprint_splits": chromaprint_splits,
                }

                if failed_servers:
                    message = (
                        f"Cleanup finished with problems: server(s) {', '.join(failed_servers)} "
                        f"could not be fully read and were skipped; {unbound_total} stale "
                        "mappings unbound elsewhere. Stale not-analyzable markers were "
                        "removed only for complete server reads. The catalogue was not modified."
                    )
                elif refused_servers:
                    message = (
                        f"Cleanup finished: {unbound_total} stale server mappings unbound, but "
                        f"server(s) {', '.join(refused_servers)} returned fewer than half the "
                        "tracks they still have mapped, so their stale mappings were NOT pruned. "
                        "Re-run the cleanup if the library really did shrink that much."
                    )
                elif clean_catalogue:
                    message = (
                        f"Cleanup complete: {unbound_total} stale server mappings unbound; "
                        f"{deleted_count} of {len(fully_unbound)} orphaned catalogue tracks "
                        f"(on no server) deleted; {deleted_analysis_exclusions} stale "
                        "not-analyzable marker(s) removed."
                    )
                else:
                    message = (
                        f"Cleanup complete: {unbound_total} stale server mappings unbound; "
                        f"{len(fully_unbound)} catalogue tracks are on no server and were "
                        f"kept (catalogue cleaning is off - enable it to delete them); "
                        f"{deleted_analysis_exclusions} stale not-analyzable marker(s) removed."
                    )
                log_and_update_main(message, 100, final_summary_details=summary)
                if failed_servers:
                    raise CleaningIncomplete(message)
                return {"status": TASK_STATUS_SUCCESS, "message": message, **summary}

            except (TaskCancelled, OperationalError, CleaningIncomplete):
                raise
            except Exception as e:
                logger.critical(f"Library cleanup failed: {e}", exc_info=True)
                err = error_manager.record(
                    error_manager.classify(e, ERR_CLEANING_FAILED), str(e)
                )
                log_and_update_main(
                    f"X Library cleanup failed: {e}",
                    log_and_update_main.state['progress'],
                    error=err,
                )
                raise
