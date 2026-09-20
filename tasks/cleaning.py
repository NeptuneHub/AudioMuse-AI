# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Library cleanup task: unbind server mappings for tracks a server no longer has.

Runs as a queue job. Fetches every configured server's track set through the
sweep's OWN enumeration and pruning (multiserver_sync.fetch_server_catalogue /
prune_stale_mappings, library filter applied), so the prune baseline can never
disagree with the enumeration that created the mappings, and removes ONLY that
server's track_server_map rows for tracks it no longer has. A song found on NO server (an orphan) is deleted
from the catalogue when catalogue cleaning is on, at most CLEANING_SAFETY_LIMIT
albums per run; the next run deletes the next albums. Every run then executes
the same full similarity-index rebuild analysis runs, INLINE, and is not
reported complete until Flask reloads the indexes.

Main Features:
* identify_and_clean_orphaned_albums_task: the queue entry point.
* Reuses the sweep's helpers so cleaning and the sweep never drift apart.
* What a server returns is what it has: no ratio or share guard ever blocks the
  unbind or the orphan delete. The ONE limit is CLEANING_SAFETY_LIMIT albums
  deleted per run.
* A server whose fetch raises, or returns an empty track list while AudioMuse
  still has songs on it (providers answer a failed library lookup with an empty
  list), was not read, so only the songs it may still hold (its own mappings,
  plus unmapped legacy rows when it is the default server) are kept; every
  other orphan is still cleaned. A partial but non-empty list is trusted.
* Refreshes each server's library size (music_servers.track_count).
* Runs the Chromaprint dedup (Path B) each time: splits merged groups whose
  stored fingerprints prove they are different recordings (skip-if-missing).
* Cleaning is a MAIN_TASK_TYPE, so it holds the one-live-main index and the
  wedged-main nudge watches its row. Both of its opaque phases therefore hold a
  row_heartbeat: the one whole-catalogue fetch PER SERVER, and the final index
  rebuild. Each is a single call that writes no row while it runs, which the
  nudge cannot tell from a wedge; both are bounded, so a fetch that really never
  returns is still handed back to it.
* A server that could not be read ends the run in CleaningIncomplete, a
  TaskFailed: the run COMPLETED (every readable server was cleaned and the
  summary is on the row), and a retry would repeat one whole-catalogue fetch per
  server plus the index rebuild while holding the one-live-main slot. The next
  cron run is the retry.
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


def _songs_on_unread_servers(cur, unread_servers):
    if not unread_servers:
        return set()
    cur.execute(
        "SELECT item_id FROM track_server_map WHERE server_id = ANY(%s)",
        ([s['server_id'] for s in unread_servers],),
    )
    kept = {row[0] for row in cur.fetchall()}
    if any(s.get('is_default') for s in unread_servers):
        cur.execute(
            "SELECT s.item_id FROM score s WHERE NOT EXISTS "
            "(SELECT 1 FROM track_server_map m WHERE m.item_id = s.item_id)"
        )
        kept.update(row[0] for row in cur.fetchall())
    return kept


def _server_still_holds_songs(server):
    from database import get_db

    with get_db() as conn, conn.cursor() as cur:
        if server:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM track_server_map WHERE server_id = %s)",
                (server['server_id'],),
            )
            if cur.fetchone()[0]:
                return True
        if server and not server.get('is_default'):
            return False
        cur.execute("SELECT EXISTS (SELECT 1 FROM score)")
        return bool(cur.fetchone()[0])


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
        TaskCancelled, task_run_prologue, cancel_guard, make_task_reporter,
    )

    with app.app_context():
        claimed_task_id, current_task_id = task_run_prologue()

        with cancel_guard(claimed_task_id) as cancel:
            cancel(force=True)
            log_and_update_main = make_task_reporter(
                current_task_id, "cleaning", STARTING_MESSAGE,
                prefix=f"CleaningTask-{current_task_id}",
            )
            try:
                log_and_update_main(STARTING_MESSAGE, 5)

                servers = registry.servers_for_scope('all')
                present_canonical_ids = set()
                failed_servers = []
                unread_servers = []
                legacy_server_unread = False
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
                        tracks = None
                    provider_ids = {str(t['id']) for t in (tracks or []) if t.get('id')}
                    if not provider_ids and (tracks is None or _server_still_holds_songs(server)):
                        if tracks is not None:
                            logger.error(
                                "%s returned an empty track list while AudioMuse still has songs "
                                "on it; treating it as unreadable so nothing of it is unbound or "
                                "deleted this run.",
                                server_name,
                            )
                        failed_servers.append(server_name)
                        if server_id:
                            unread_servers.append(server)
                        else:
                            legacy_server_unread = True
                        continue
                    tracks = None
                    total_tracks_on_servers += len(provider_ids)
                    log_and_update_main(
                        f"Found {len(provider_ids)} tracks on {server_name}",
                        window_start + int(35 / len(servers)),
                    )

                    if server_id:
                        _store_server_track_count(get_db(), server_id, len(provider_ids))
                        unbound = prune_stale_mappings(
                            get_db(), server_id, sorted(provider_ids)
                        )
                        unbound_by_server[server_name] = unbound
                        unbound_total += unbound
                        if unbound:
                            log_and_update_main(
                                f"Unbound {unbound} tracks no longer on {server_name} "
                                "(kept in the shared catalogue).",
                                window_start + int(70 / len(servers)),
                            )
                    marker_server_id = server_id or registry.get_default_server_id()
                    if marker_server_id:
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
                    kept_on_unread_servers = _songs_on_unread_servers(cur, unread_servers)

                if legacy_server_unread:
                    fully_unbound = set()
                else:
                    fully_unbound = (
                        database_track_ids - present_canonical_ids - kept_on_unread_servers
                    )

                orphan_albums = defaultdict(list)
                orphan_list = sorted(fully_unbound)
                if orphan_list:
                    with get_db() as conn, conn.cursor() as cur:
                        for start in range(0, len(orphan_list), 5000):
                            cancel()
                            chunk = orphan_list[start:start + 5000]
                            cur.execute(
                                "SELECT item_id, title, author, album, album_artist "
                                "FROM score WHERE item_id = ANY(%s)",
                                (chunk,),
                            )
                            for track_id, title, author, album, album_artist in cur.fetchall():
                                album_key = (
                                    album_artist or author or "Unknown Artist",
                                    album or "Unknown Album",
                                )
                                orphan_albums[album_key].append(
                                    {"item_id": track_id, "title": title, "author": author}
                                )

                ordered_albums = sorted(
                    orphan_albums.items(), key=lambda kv: (-len(kv[1]), kv[0])
                )
                orphaned_albums_list = [
                    {
                        "artist": artist,
                        "album": album,
                        "track_count": len(tracks_of_album),
                        "tracks": tracks_of_album,
                    }
                    for (artist, album), tracks_of_album
                    in ordered_albums[:max(CLEANING_SAFETY_LIMIT, 0)]
                ]

                deleted_count = 0
                deleted_albums_count = 0
                if clean_catalogue and orphaned_albums_list:
                    orphan_ids = [
                        track["item_id"]
                        for album in orphaned_albums_list
                        for track in album["tracks"]
                    ]
                    with get_db() as conn, conn.cursor() as cur:
                        for start in range(0, len(orphan_ids), 5000):
                            cancel()
                            chunk = orphan_ids[start:start + 5000]
                            cur.execute(
                                "DELETE FROM score WHERE item_id = ANY(%s)", (chunk,)
                            )
                            deleted_count += len(chunk)
                    deleted_albums_count = len(orphaned_albums_list)
                    log_and_update_main(
                        f"Deleted {deleted_count} orphaned catalogue tracks from "
                        f"{deleted_albums_count} album(s) (on no server); their analysis "
                        "is re-created if the files return.",
                        90,
                    )
                remaining_orphans = len(fully_unbound) - deleted_count

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
                    "orphaned_albums_count": len(orphan_albums),
                    "orphaned_albums": orphaned_albums_list,
                    "unbound_mappings": unbound_total,
                    "unbound_by_server": unbound_by_server,
                    "failed_servers": failed_servers,
                    "deleted_count": deleted_count,
                    "deleted_albums_count": deleted_albums_count,
                    "remaining_orphans_count": remaining_orphans,
                    "cleaning_safety_limit": CLEANING_SAFETY_LIMIT,
                    "deleted_analysis_exclusions": deleted_analysis_exclusions,
                    "catalogue_deletion": clean_catalogue,
                    "chromaprint_splits": chromaprint_splits,
                }

                if clean_catalogue:
                    message = (
                        f"Cleanup complete: {unbound_total} stale server mappings unbound; "
                        f"{deleted_count} orphaned catalogue tracks from "
                        f"{deleted_albums_count} album(s) (on no server) deleted, "
                        f"{remaining_orphans} left for the next run (at most "
                        f"{CLEANING_SAFETY_LIMIT} albums per run); "
                        f"{deleted_analysis_exclusions} stale not-analyzable marker(s) removed."
                    )
                else:
                    message = (
                        f"Cleanup complete: {unbound_total} stale server mappings unbound; "
                        f"{len(fully_unbound)} catalogue tracks are on no server and were "
                        f"kept (catalogue cleaning is off - enable it to delete them); "
                        f"{deleted_analysis_exclusions} stale not-analyzable marker(s) removed."
                    )
                if failed_servers:
                    message = (
                        f"Server(s) {', '.join(failed_servers)} could not be read, so only "
                        "the songs they may still hold were kept. " + message
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
