# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Library cleanup task: prune analysis rows for tracks no longer on any server.

Runs as an RQ job. Fetches the current track set of every enabled media server,
translates each server's provider track ids to the canonical catalogue ids via
the server registry, and removes database entries (and their embeddings) present
on no server so stale data does not leak into clustering, similarity or radio
results. If any server's library cannot be fully fetched the run aborts without
deleting anything, so a partial view never causes deletions.

Main Features:
* identify_and_clean_orphaned_albums_task: the RQ entry point that collects
  track ids across all enabled servers, maps them to canonical ids and deletes
  the DB tracks found on no server, aborting on any partial fetch.
* delete_orphaned_albums_sync: synchronous helper that removes analysis and
  embedding rows for a set of orphaned track ids across the related tables
  using batched deletes.
"""

import time
import logging
import uuid
from collections import defaultdict

from rq import get_current_job

from config import CLEANING_SAFETY_LIMIT

from error import error_manager
from error.error_dictionary import ERR_CLEANING_FAILED, ERR_DB_CONNECTION

from .mediaserver import context as ms_context, registry
from .mediaserver import get_recent_albums, get_tracks_from_album

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
    )

    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        initial_details = {
            "message": "Starting orphaned album identification...",
            "log": [
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Orphaned album identification task started."
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

        try:
            log_and_update_main("Starting orphaned album identification...", 5)

            log_and_update_main("Fetching all albums from configured media server(s)...", 10)
            servers = registry.servers_for_scope('all')
            if not servers:
                abort_message = (
                    "Aborting cleaning: no enabled media servers found. "
                    "No tracks were deleted."
                )
                summary = {
                    "status": "ABORTED",
                    "message": abort_message,
                    "failed_servers": [],
                    "orphaned_albums": [],
                    "deleted_count": 0,
                }
                log_and_update_main(
                    abort_message,
                    100,
                    task_state=TASK_STATUS_FAILURE,
                    final_summary_details=summary,
                )
                return summary

            per_server_provider_ids = []
            failed_servers = []
            zero_album_servers = []
            total_albums = 0

            for server_idx, server in enumerate(servers):
                server_name = server['name'] if server else 'default server'
                ctx = registry.context_for(server['server_id']) if server else None
                window_start = 20 + int(55 * server_idx / len(servers))
                with ms_context.use_server(ctx):
                    try:
                        server_albums = get_recent_albums(0)
                    except Exception:
                        logger.exception(f"Failed to fetch albums from {server_name}")
                        failed_servers.append(server_name)
                        continue
                    if not server_albums:
                        logger.warning(f"No albums found on {server_name}.")
                        zero_album_servers.append(server_name)
                        continue
                    total_albums += len(server_albums)
                    log_and_update_main(
                        f"Found {len(server_albums)} albums on {server_name}", window_start
                    )
                    provider_ids = set()
                    albums_processed = 0
                    for idx, album in enumerate(server_albums):
                        try:
                            album_tracks = get_tracks_from_album(album['Id'])
                            if album_tracks:
                                for track in album_tracks:
                                    provider_ids.add(str(track['Id']))
                            albums_processed += 1

                            if idx % 10 == 0:
                                progress = window_start + int(
                                    (55.0 / len(servers)) * (idx / float(len(server_albums)))
                                )
                                log_and_update_main(
                                    f"Processed {albums_processed}/{len(server_albums)} albums on {server_name}...",
                                    progress,
                                )

                        except Exception:
                            logger.warning(
                                f"Failed to get tracks for album {album.get('Name', 'Unknown')} on {server_name}",
                                exc_info=True,
                            )
                            continue
                    per_server_provider_ids.append((server, provider_ids))

            if failed_servers or (zero_album_servers and len(servers) > 1):
                problem_servers = failed_servers + zero_album_servers
                abort_message = (
                    "Aborting cleaning: incomplete library view from server(s): "
                    f"{', '.join(problem_servers)}. No tracks were deleted."
                )
                summary = {
                    "status": "ABORTED",
                    "message": abort_message,
                    "failed_servers": problem_servers,
                    "orphaned_albums": [],
                    "deleted_count": 0,
                }
                log_and_update_main(
                    abort_message,
                    100,
                    task_state=TASK_STATUS_FAILURE,
                    final_summary_details=summary,
                )
                return summary

            if zero_album_servers:
                log_and_update_main(
                    "No albums found on media server.", 95, task_state=TASK_STATUS_PROGRESS
                )
                log_and_update_main("Rebuilding all indexes and maps...", 96)
                try:
                    from .analysis import _run_all_index_builds

                    _run_all_index_builds(log_fn=None)
                    log_and_update_main("OK All indexes and maps rebuilt successfully.", 99)
                except Exception as e:
                    logger.warning("Failed to rebuild indexes and maps", exc_info=True)
                    log_and_update_main(
                        f"Warning: Failed to rebuild indexes and maps: {str(e)}", 99
                    )

                summary = {
                    "status": "SUCCESS",
                    "message": "No albums found on media server.",
                    "orphaned_albums": [],
                    "deleted_count": 0,
                }
                log_and_update_main(
                    "OK Database cleaning completed - no albums on media server!",
                    100,
                    task_state=TASK_STATUS_SUCCESS,
                    final_summary_details=summary,
                )
                return summary

            log_and_update_main("Translating server track ids to canonical catalogue ids...", 76)
            present_canonical_ids = set()
            for server, provider_ids in per_server_provider_ids:
                server_id = server['server_id'] if server else None
                provider_list = list(provider_ids)
                for start in range(0, len(provider_list), 5000):
                    chunk = provider_list[start:start + 5000]
                    mapping = registry.reverse_translate_ids(chunk, server_id)
                    present_canonical_ids.update(str(v) for v in mapping.values())

            log_and_update_main(
                f"Found {len(present_canonical_ids)} total tracks across {len(servers)} server(s)",
                78,
            )

            log_and_update_main("Fetching all track IDs from database...", 80)
            with get_db() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT s.item_id, s.title, s.author
                    FROM score s
                    JOIN embedding e ON s.item_id = e.item_id
                """)
                database_tracks = cur.fetchall()

            database_track_ids = {row[0] for row in database_tracks}
            log_and_update_main(f"Found {len(database_track_ids)} tracks in database", 85)

            orphaned_track_ids = database_track_ids - present_canonical_ids
            log_and_update_main(f"Identified {len(orphaned_track_ids)} orphaned tracks", 90)

            orphaned_albums_info = defaultdict(lambda: {"tracks": [], "track_count": 0})

            for track_data in database_tracks:
                track_id, title, author = track_data
                if track_id in orphaned_track_ids:
                    album_key = f"{author}" if author else "Unknown Artist"
                    orphaned_albums_info[album_key]["tracks"].append(
                        {"item_id": track_id, "title": title, "author": author}
                    )
                    orphaned_albums_info[album_key]["track_count"] += 1

            orphaned_albums_list = []
            for artist, info in orphaned_albums_info.items():
                orphaned_albums_list.append(
                    {"artist": artist, "track_count": info["track_count"], "tracks": info["tracks"]}
                )

            orphaned_albums_list.sort(key=lambda x: x["track_count"], reverse=True)

            total_orphaned_albums = len(orphaned_albums_list)
            safety_limit_applied = False
            if total_orphaned_albums > CLEANING_SAFETY_LIMIT:
                safety_limit_applied = True
                log_and_update_main(
                    f"Safety limit: Found {total_orphaned_albums} orphaned albums, limiting to first {CLEANING_SAFETY_LIMIT} for safety",
                    92,
                )
                orphaned_albums_list = orphaned_albums_list[:CLEANING_SAFETY_LIMIT]
                limited_track_ids = set()
                for album in orphaned_albums_list:
                    for track in album["tracks"]:
                        limited_track_ids.add(track["item_id"])
                orphaned_track_ids = limited_track_ids

            if len(orphaned_track_ids) == 0:
                log_and_update_main(
                    "OK No orphaned tracks found. Database is clean!",
                    95,
                    task_state=TASK_STATUS_PROGRESS,
                )
                log_and_update_main("Rebuilding all indexes and maps...", 96)
                try:
                    from .analysis import _run_all_index_builds

                    _run_all_index_builds(log_fn=None)
                    log_and_update_main("OK All indexes and maps rebuilt successfully.", 99)
                except Exception as e:
                    logger.warning("Failed to rebuild indexes and maps", exc_info=True)
                    log_and_update_main(
                        f"Warning: Failed to rebuild indexes and maps: {str(e)}", 99
                    )

                summary = {
                    "total_media_server_albums": total_albums,
                    "total_media_server_tracks": len(present_canonical_ids),
                    "total_database_tracks": len(database_track_ids),
                    "orphaned_tracks_count": 0,
                    "orphaned_albums_count": 0,
                    "deleted_count": 0,
                }

                log_and_update_main(
                    "OK Database cleaning completed - no orphaned tracks found!",
                    100,
                    task_state=TASK_STATUS_SUCCESS,
                    final_summary_details=summary,
                )
                return {
                    "status": "SUCCESS",
                    "message": "No orphaned tracks found. Database is clean!",
                    **summary,
                }

            log_and_update_main(
                f"Starting automatic deletion of {len(orphaned_track_ids)} orphaned tracks...", 93
            )

            deletion_result = delete_orphaned_albums_sync(list(orphaned_track_ids))

            summary = {
                "total_media_server_albums": total_albums,
                "total_media_server_tracks": len(present_canonical_ids),
                "total_database_tracks": len(database_track_ids),
                "orphaned_tracks_count": len(orphaned_track_ids),
                "orphaned_albums_count": len(orphaned_albums_list),
                "orphaned_albums": orphaned_albums_list,
                "deletion_result": deletion_result,
                "deleted_count": deletion_result.get("deleted_count", 0),
                "failed_deletions": deletion_result.get("failed_deletions", []),
            }

            if deletion_result["status"] == "SUCCESS":
                log_and_update_main(
                    f"OK Successfully deleted {deletion_result['deleted_count']} orphaned tracks.",
                    96,
                )

                log_and_update_main("Rebuilding all indexes and maps after cleaning...", 97)
                try:
                    from .analysis import _run_all_index_builds

                    _run_all_index_builds(log_fn=None)
                    log_and_update_main(
                        "OK All indexes and maps rebuilt successfully after cleaning.", 99
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to rebuild indexes and maps after cleaning", exc_info=True
                    )
                    log_and_update_main(
                        f"Warning: Failed to rebuild indexes and maps: {str(e)}", 99
                    )

                safety_message = (
                    f" (Safety limit: deleted {len(orphaned_albums_list)} out of {total_orphaned_albums} albums)"
                    if safety_limit_applied
                    else ""
                )

                log_and_update_main(
                    f"OK Cleaning complete! Identified and deleted {len(orphaned_albums_list)} orphaned albums ({deletion_result['deleted_count']} tracks).{safety_message}",
                    100,
                    task_state=TASK_STATUS_SUCCESS,
                    final_summary_details=summary,
                )

                if safety_limit_applied:
                    remaining_count = total_orphaned_albums - len(orphaned_albums_list)
                    if remaining_count > 0:
                        log_and_update_main(
                            f"Safety note: {remaining_count} additional orphaned albums remain. Run cleaning again to process more.",
                            100,
                            task_state=TASK_STATUS_SUCCESS,
                        )

                return {
                    "status": "SUCCESS",
                    "message": f"Successfully cleaned {deletion_result['deleted_count']} orphaned tracks from {len(orphaned_albums_list)} albums",
                    **summary,
                }
            else:
                log_and_update_main(
                    f"Cleaning partially failed. Deletion error: {deletion_result.get('message', 'Unknown error')}",
                    100,
                    task_state=TASK_STATUS_FAILURE,
                    final_summary_details=summary,
                )
                raise RuntimeError(
                    f"Deletion failed: {deletion_result.get('message', 'Unknown error')}"
                )

        except OperationalError as e:
            logger.exception(
                "Database connection error during cleaning identification. This job will be retried."
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
            logger.critical(f"Orphaned album identification failed: {e}", exc_info=True)
            err = error_manager.record(
                error_manager.classify(e, ERR_CLEANING_FAILED), str(e)
            )
            log_and_update_main(
                f"X Orphaned album identification failed: {e}",
                current_progress,
                task_state=TASK_STATUS_FAILURE,
                error=err,
            )
            raise


def delete_orphaned_albums_sync(orphaned_track_ids):
    from app_helper import get_db

    if not orphaned_track_ids:
        return {"status": "SUCCESS", "message": "No tracks to delete", "deleted_count": 0}

    track_ids = [str(track_id) for track_id in orphaned_track_ids]
    chunk_size = 1000

    try:
        deleted_count = 0
        failed_deletions = []

        with get_db() as conn:
            with conn.cursor() as cur:

                def _table_exists(table_name):
                    try:
                        cur.execute("SELECT to_regclass(%s)", (table_name,))
                        row = cur.fetchone()
                        return bool(row and row[0] is not None)
                    except Exception:
                        logger.warning(
                            f"Could not check existence of table {table_name}", exc_info=True
                        )
                        return False

                def _delete_from_child_table(table_name):
                    if not _table_exists(table_name):
                        logger.info(f"Skipping {table_name}: table does not exist.")
                        return
                    logger.info(
                        f"Deleting {len(track_ids)} tracks from {table_name} table..."
                    )
                    for start in range(0, len(track_ids), chunk_size):
                        chunk = track_ids[start:start + chunk_size]
                        try:
                            cur.execute(
                                f"DELETE FROM {table_name} WHERE item_id = ANY(%s)",
                                (chunk,),
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to delete a batch of {len(chunk)} tracks from {table_name}",
                                exc_info=True,
                            )
                            failed_deletions.append(
                                {"track_ids": chunk, "table": table_name, "error": str(e)}
                            )

                _delete_from_child_table("embedding")
                _delete_from_child_table("lyrics_embedding")
                _delete_from_child_table("clap_embedding")

                logger.info(f"Deleting {len(track_ids)} tracks from score table...")
                for start in range(0, len(track_ids), chunk_size):
                    chunk = track_ids[start:start + chunk_size]
                    try:
                        cur.execute("DELETE FROM score WHERE item_id = ANY(%s)", (chunk,))
                        deleted_count += cur.rowcount
                    except Exception as e:
                        logger.warning(
                            f"Failed to delete a batch of {len(chunk)} tracks from score",
                            exc_info=True,
                        )
                        failed_deletions.append(
                            {"track_ids": chunk, "table": "score", "error": str(e)}
                        )

                conn.commit()
                logger.info(f"Successfully deleted {deleted_count} orphaned tracks from database")

        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    for start in range(0, len(track_ids), chunk_size):
                        chunk = track_ids[start:start + chunk_size]
                        cur.execute("DELETE FROM playlist WHERE item_id = ANY(%s)", (chunk,))
                    conn.commit()
                    logger.info("Cleaned up playlist references for deleted tracks")
        except Exception:
            logger.warning("Failed to clean up playlist references", exc_info=True)

        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM artist_mapping
                        WHERE artist_name NOT IN (
                            SELECT DISTINCT author
                            FROM score
                            WHERE author IS NOT NULL AND author != ''
                        )
                    """)
                    orphaned_artists_count = cur.rowcount
                    conn.commit()
                    if orphaned_artists_count > 0:
                        logger.info(
                            f"Cleaned up {orphaned_artists_count} orphaned artists from artist_mapping table"
                        )
        except Exception:
            logger.warning(
                "Failed to clean up orphaned artists from artist_mapping", exc_info=True
            )

        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT to_regclass('artist_server_map')")
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        cur.execute("""
                            DELETE FROM artist_server_map
                            WHERE artist_name NOT IN (
                                SELECT DISTINCT author
                                FROM score
                                WHERE author IS NOT NULL AND author != ''
                            )
                        """)
                        orphaned_server_artists_count = cur.rowcount
                        conn.commit()
                        if orphaned_server_artists_count > 0:
                            logger.info(
                                f"Cleaned up {orphaned_server_artists_count} orphaned artists from artist_server_map table"
                            )
        except Exception:
            logger.warning(
                "Failed to clean up orphaned artists from artist_server_map", exc_info=True
            )

        return {
            "status": "SUCCESS",
            "message": f"Successfully deleted {deleted_count} orphaned tracks",
            "deleted_count": deleted_count,
            "failed_deletions": failed_deletions,
            "total_requested": len(orphaned_track_ids),
        }

    except Exception as e:
        logger.exception("Failed to delete orphaned albums")
        return {
            "status": "FAILURE",
            "message": f"Failed to delete orphaned albums: {str(e)}",
            "deleted_count": 0,
            "error": str(e),
        }
