# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Library cleanup task: unbind server mappings for tracks a server no longer has.

Runs as an RQ job. Fetches the current track set of every configured media
server through the sweep's OWN enumeration and pruning
(``multiserver_sync.fetch_server_catalogue`` / ``prune_stale_mappings``, library
filter applied), so the prune baseline can never disagree with the enumeration
that created the mappings, and removes ONLY that server's rows from
track_server_map for tracks it no longer has. The
centralized catalogue is NEVER touched: a song that disappeared from one
server keeps its analysis, embeddings and mappings on every other server, and
a song present on no server at all stays in the catalogue as unbound (it
simply stops appearing in server-scoped results via the availability mask).
Because no catalogue row changes, no index rebuild is needed. A server whose
fetch fails, returns nothing, or looks partial is skipped, so an incomplete
library view can never unbind valid mappings.

Main Features:
* identify_and_clean_orphaned_albums_task: the RQ entry point that fetches each
  server's tracks and prunes that server's stale mappings only.
* Reuses the sweep's public helpers rather than re-implementing the fetch and
  the prune, so cleaning and the sweep can never drift apart.
* Refreshes each server's stored library size (``music_servers.track_count``)
  from the fetch it already performs, keeping the dashboard's coverage
  denominator current on every cleaning run.
* Reports (never deletes) the catalogue tracks currently bound to no server.
"""

import time
import logging
import uuid
from collections import defaultdict

from rq import get_current_job

from config import CLEANING_SAFETY_LIMIT

from error import error_manager
from error.error_dictionary import ERR_CLEANING_FAILED, ERR_DB_CONNECTION

from .mediaserver import registry

from psycopg2 import OperationalError

logger = logging.getLogger(__name__)


def identify_and_clean_orphaned_albums_task():
    from flask_app import app
    from app_helper import redis_conn, get_db, save_task_status
    from config import (
        TASK_STATUS_STARTED,
        TASK_STATUS_PROGRESS,
        TASK_STATUS_SUCCESS,
        TASK_STATUS_FAILURE,
        TASK_STATUS_REVOKED,
    )
    from .multiserver_sync import (
        fetch_server_catalogue,
        prune_stale_mappings,
        make_cancel_check,
        SweepCancelled,
        _store_server_track_count,
    )

    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        initial_details = {
            "message": "Starting per-server library cleanup...",
            "log": [
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Library cleanup task started."
            ],
        }
        save_task_status(
            current_task_id, "cleaning", TASK_STATUS_STARTED, progress=0, details=initial_details
        )
        current_progress = 0
        current_task_logs = initial_details["log"]

        def log_and_update_main(message, progress, **kwargs):
            nonlocal current_progress
            current_progress = progress
            logger.info(f"[CleaningTask-{current_task_id}] {message}")
            details = {**kwargs, "status_message": message}
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            task_state = kwargs.get('task_state', TASK_STATUS_PROGRESS)

            if task_state != TASK_STATUS_SUCCESS:
                current_task_logs.append(log_entry)
                details["log"] = current_task_logs
            else:
                details["log"] = [f"Task completed successfully. Final status: {message}"]

            if current_job:
                current_job.meta.update(
                    {'progress': progress, 'status_message': message, 'details': details}
                )
                current_job.save_meta()
            save_task_status(
                current_task_id, "cleaning", task_state, progress=progress, details=details
            )

        cancel, close_cancel = make_cancel_check(current_task_id)
        try:
            log_and_update_main("Starting per-server library cleanup...", 5)

            servers = registry.servers_for_scope('all')
            present_canonical_ids = set()
            failed_servers = []
            refused_servers = []
            unbound_total = 0
            unbound_by_server = {}
            total_tracks_on_servers = 0

            for server_idx, server in enumerate(servers):
                cancel()
                server_name = server['name'] if server else 'default server'
                server_id = server['server_id'] if server else None
                window_start = 10 + int(70 * server_idx / len(servers))
                log_and_update_main(
                    f"Fetching the track list from {server_name}...", window_start
                )
                try:
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

                if server_id:
                    _store_server_track_count(get_db(), server_id, len(provider_ids))
                    refused = []
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
                "deleted_count": 0,
            }

            state = TASK_STATUS_FAILURE if failed_servers else TASK_STATUS_SUCCESS
            if failed_servers:
                message = (
                    f"Cleanup finished with problems: server(s) {', '.join(failed_servers)} "
                    f"could not be fully read and were skipped; {unbound_total} stale "
                    "mappings unbound elsewhere. The catalogue was not modified."
                )
            elif refused_servers:
                message = (
                    f"Cleanup finished: {unbound_total} stale server mappings unbound, but "
                    f"server(s) {', '.join(refused_servers)} returned fewer than half the "
                    "tracks they still have mapped, so their stale mappings were NOT pruned. "
                    "Re-run the cleanup if the library really did shrink that much."
                )
            else:
                message = (
                    f"Cleanup complete: {unbound_total} stale server mappings unbound; "
                    f"{len(fully_unbound)} catalogue tracks are now on no server and are "
                    "hidden from results by the per-server availability filter."
                )
            log_and_update_main(message, 100, task_state=state, final_summary_details=summary)
            return {"status": "SUCCESS" if not failed_servers else "FAILURE",
                    "message": message, **summary}

        except SweepCancelled:
            # Must precede the generic handler below, or a user pressing Stop is
            # recorded as ERR_CLEANING_FAILED and re-raised into an RQ retry.
            logger.info("Library cleanup revoked by the user; stopping.")
            log_and_update_main(
                "Library cleanup cancelled.",
                current_progress,
                task_state=TASK_STATUS_REVOKED,
            )
            return {"status": TASK_STATUS_REVOKED, "message": "Library cleanup cancelled."}
        except OperationalError as e:
            logger.exception(
                "Database connection error during cleaning. This job will be retried."
            )
            err = error_manager.record(ERR_DB_CONNECTION, str(e))
            log_and_update_main(
                "Database connection failed. Retrying...",
                current_progress,
                task_state=TASK_STATUS_FAILURE,
                error=err,
            )
            raise
        except Exception as e:
            logger.critical(f"Library cleanup failed: {e}", exc_info=True)
            err = error_manager.record(
                error_manager.classify(e, ERR_CLEANING_FAILED), str(e)
            )
            log_and_update_main(
                f"X Library cleanup failed: {e}",
                current_progress,
                task_state=TASK_STATUS_FAILURE,
                error=err,
            )
            raise
        finally:
            close_cancel()
