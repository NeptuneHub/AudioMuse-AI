# tasks/cleaning.py

import time
import logging
import uuid
import traceback
from collections import defaultdict

# RQ import
from rq import get_current_job

# Import configuration
from config import CLEANING_SAFETY_LIMIT, EMBEDDING_DIMENSION, INDEX_NAME, SONIC_BACKEND

from error import error_manager
from error.error_dictionary import ERR_CLEANING_FAILED, ERR_DB_CONNECTION

# Import other project modules
from .mediaserver import get_recent_albums, get_tracks_from_album

from psycopg2 import OperationalError

logger = logging.getLogger(__name__)


# --- Sonic-backend storage inspection / cleanup -----------------------------
#
# Per-backend storage means every backend's embedding rows and Voyager
# index live alongside each other (composite (item_id, backend) PK on
# ``embedding``; namespaced ``voyager_index_data.index_name``). The
# admin "Cleaning" panel uses these helpers to:
#   * report per-backend row counts so users can see what's on disk;
#   * drop a specific backend's audio data. The active backend is
#     protected — switching away first is required to clear it.


def _voyager_rows_for(cur, backend: str):
    """Return (row_count, embedding_dim) for ``backend``'s Voyager rows.

    Matches both the single-row form (``music_library_<backend>``) and
    the segmented form (``music_library_<backend>_<part>_<total>``).
    """
    from config import INDEX_NAME
    main = f"{INDEX_NAME}_{backend}"
    seg_pattern = main.replace("_", r"\_") + r"\_%\_%"
    cur.execute(
        "SELECT COUNT(*), MIN(embedding_dimension) FROM voyager_index_data "
        "WHERE index_name = %s OR index_name LIKE %s ESCAPE '\\'",
        (main, seg_pattern),
    )
    row = cur.fetchone() or (0, None)
    return int(row[0] or 0), (int(row[1]) if row[1] is not None else None)


def _embedding_rows_for(cur, backend: str):
    """Return (row_count, sampled_dim) for ``backend``'s embedding rows."""
    cur.execute("SELECT COUNT(*) FROM embedding WHERE backend = %s", (backend,))
    count = int((cur.fetchone() or (0,))[0] or 0)
    sample_dim = None
    if count > 0:
        cur.execute(
            "SELECT embedding FROM embedding "
            "WHERE backend = %s AND embedding IS NOT NULL LIMIT 1",
            (backend,),
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            raw = bytes(row[0])
            if len(raw) % 4 == 0:
                sample_dim = len(raw) // 4
    return count, sample_dim


def inspect_sonic_state():
    """Return per-backend embedding + Voyager state for the Cleaning UI.

    Shape (JSON-safe; returned by ``GET /api/cleaning/sonic_state``):
      * ``active_backend``: the SONIC_BACKEND currently writing data.
      * ``active_dim``: the embedding dim that backend produces.
      * ``backends``: list of ``{backend, embedding_row_count,
        sample_stored_dim, voyager_row_count, stored_voyager_dim,
        is_active}`` rows — one per backend with any stored audio data,
        plus the active backend even if it currently has none.

    Stored backends are detected by SELECT DISTINCT on ``embedding`` +
    parsing ``voyager_index_data.index_name`` so the panel surfaces
    legacy backends the user no longer has registered, not just the
    backends shipped in this image.
    """
    from app_helper import get_db
    from config import INDEX_NAME

    snapshot = {
        "active_backend": SONIC_BACKEND,
        "active_dim": int(EMBEDDING_DIMENSION),
        "backends": [],
    }

    try:
        with get_db() as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT backend FROM embedding")
            backends_with_emb = {r[0] for r in cur.fetchall() if r[0]}

            cur.execute(
                "SELECT DISTINCT regexp_replace("
                "  regexp_replace(index_name, '_[0-9]+_[0-9]+$', ''), "
                "  %s, '') AS backend "
                "FROM voyager_index_data "
                "WHERE index_name LIKE %s ESCAPE '\\'",
                (f"^{INDEX_NAME}_", INDEX_NAME.replace("_", r"\_") + r"\_%"),
            )
            backends_with_voy = {r[0] for r in cur.fetchall() if r[0]}

            backend_set = backends_with_emb | backends_with_voy | {SONIC_BACKEND}
            for backend in sorted(backend_set):
                emb_count, emb_dim = _embedding_rows_for(cur, backend)
                voy_count, voy_dim = _voyager_rows_for(cur, backend)
                snapshot["backends"].append({
                    "backend": backend,
                    "embedding_row_count": emb_count,
                    "sample_stored_dim": emb_dim,
                    "voyager_row_count": voy_count,
                    "stored_voyager_dim": voy_dim,
                    "is_active": backend == SONIC_BACKEND,
                })
    except Exception as e:
        logger.warning(f"Failed to inspect sonic embedding state: {e}", exc_info=True)
        snapshot["error"] = str(e)
    return snapshot


def clear_inactive_backend_data(backend: str):
    """Drop one inactive backend's embedding rows + Voyager index.

    Refuses to operate on ``SONIC_BACKEND`` (a configuration swap must
    happen first; otherwise the next analysis pass would simply repopulate
    the rows). Only touches ``embedding`` rows where ``backend = X`` and
    ``voyager_index_data`` rows under that backend's namespace. The
    CLAP / lyrics / artist / score / playlist / config tables are
    untouched.

    Returns a small summary dict for the API response.
    """
    if not backend or not isinstance(backend, str):
        raise ValueError("backend must be a non-empty string")
    if backend == SONIC_BACKEND:
        raise ValueError(
            f"Refusing to clear the active backend ({backend!r}). Set "
            "SONIC_BACKEND to a different backend first, then come back "
            "to this panel."
        )

    from app_helper import get_db
    from config import INDEX_NAME
    main = f"{INDEX_NAME}_{backend}"
    seg_pattern = main.replace("_", r"\_") + r"\_%\_%"

    summary = {
        "backend": backend,
        "deleted_embeddings": 0,
        "deleted_voyager_rows": 0,
        "deleted_mood_centroids": 0,
    }
    with get_db() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM embedding WHERE backend = %s", (backend,))
        summary["deleted_embeddings"] = cur.rowcount or 0

        cur.execute(
            "DELETE FROM voyager_index_data "
            "WHERE index_name = %s OR index_name LIKE %s ESCAPE '\\'",
            (main, seg_pattern),
        )
        summary["deleted_voyager_rows"] = cur.rowcount or 0

        # mood_centroids_data is per-(backend, mood) — drop every mood
        # row for the cleared backend so stale centroids don't linger.
        # Table may not exist on a partial install; the SAVEPOINT lets us
        # swallow the miss without aborting the surrounding transaction.
        cur.execute("SAVEPOINT mood_centroids_delete")
        try:
            cur.execute("DELETE FROM mood_centroids_data WHERE backend = %s", (backend,))
            summary["deleted_mood_centroids"] = cur.rowcount or 0
            cur.execute("RELEASE SAVEPOINT mood_centroids_delete")
        except Exception as e:
            cur.execute("ROLLBACK TO SAVEPOINT mood_centroids_delete")
            logger.info("mood_centroids_data not cleared (%s); skipping.", e)

        conn.commit()

    logger.info(
        "Cleared inactive backend %r: %d embeddings, %d voyager rows. "
        "(active backend remains %s/%d-dim.)",
        backend, summary["deleted_embeddings"], summary["deleted_voyager_rows"],
        SONIC_BACKEND, EMBEDDING_DIMENSION,
    )
    return summary


def identify_and_clean_orphaned_albums_task():
    """
    Main RQ task to identify and automatically clean orphaned albums from the database.
    This combines identification and deletion into a single automated process.
    """
    from flask_app import app
    from app_helper import redis_conn, get_db, save_task_status, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS, TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE

    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        initial_details = {
            "message": "Starting orphaned album identification...", 
            "log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Orphaned album identification task started."]
        }
        save_task_status(current_task_id, "cleaning", TASK_STATUS_STARTED, progress=0, details=initial_details)
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
                current_job.meta.update({'progress': progress, 'status_message': message, 'details': details})
                current_job.save_meta()
            save_task_status(current_task_id, "cleaning", task_state, progress=progress, details=details)

        try:
            log_and_update_main("🔍 Starting orphaned album identification...", 5)
            
            # Step 1: Get all albums from media server (fetch all albums with limit=0)
            log_and_update_main("📡 Fetching all albums from media server...", 10)
            all_media_server_albums = get_recent_albums(0)  # 0 means fetch all albums
            
            if not all_media_server_albums:
                log_and_update_main("⚠️ No albums found on media server.", 95, task_state=TASK_STATUS_PROGRESS)
                log_and_update_main(f"🔄 Rebuilding all indexes and maps...", 96)
                try:
                    from .analysis import _run_all_index_builds
                    _run_all_index_builds(log_fn=None)
                    log_and_update_main(f"✅ All indexes and maps rebuilt successfully.", 99)
                except Exception as e:
                    logger.warning(f"Failed to rebuild indexes and maps: {e}")
                    log_and_update_main(f"⚠️ Warning: Failed to rebuild indexes and maps: {str(e)}", 99)
                
                summary = {"status": "SUCCESS", "message": "No albums found on media server.", "orphaned_albums": [], "deleted_count": 0}
                log_and_update_main("✅ Database cleaning completed - no albums on media server!", 100, task_state=TASK_STATUS_SUCCESS, final_summary_details=summary)
                return summary
            
            log_and_update_main(f"📊 Found {len(all_media_server_albums)} albums on media server", 20)
            
            # Step 2: Get all track IDs that exist on the media server
            log_and_update_main("🎵 Collecting all track IDs from media server...", 25)
            media_server_track_ids = set()
            albums_processed = 0
            
            for idx, album in enumerate(all_media_server_albums):
                try:
                    album_tracks = get_tracks_from_album(album['Id'])
                    if album_tracks:
                        for track in album_tracks:
                            media_server_track_ids.add(str(track['Id']))
                    albums_processed += 1
                    
                    # Update progress every 10 albums
                    if idx % 10 == 0:
                        progress = 25 + int(50 * (idx / float(len(all_media_server_albums))))
                        log_and_update_main(f"📝 Processed {albums_processed}/{len(all_media_server_albums)} albums...", progress)
                        
                except Exception as e:
                    logger.warning(f"Failed to get tracks for album {album.get('Name', 'Unknown')}: {e}")
                    continue
            
            log_and_update_main(f"🎯 Found {len(media_server_track_ids)} total tracks on media server", 75)
            
            # Step 3: Get all track IDs from database
            log_and_update_main("🗄️ Fetching all track IDs from database...", 80)
            with get_db() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT s.item_id, s.title, s.author 
                    FROM score s 
                    JOIN embedding e ON s.item_id = e.item_id
                """)
                database_tracks = cur.fetchall()
            
            database_track_ids = {row[0] for row in database_tracks}
            log_and_update_main(f"📚 Found {len(database_track_ids)} tracks in database", 85)
            
            # Step 4: Identify orphaned tracks (in database but not on media server)
            orphaned_track_ids = database_track_ids - media_server_track_ids
            log_and_update_main(f"🧹 Identified {len(orphaned_track_ids)} orphaned tracks", 90)
            
            # Step 5: Group orphaned tracks by artist/album for better presentation
            orphaned_albums_info = defaultdict(lambda: {"tracks": [], "track_count": 0})
            
            for track_data in database_tracks:
                track_id, title, author = track_data
                if track_id in orphaned_track_ids:
                    album_key = f"{author}" if author else "Unknown Artist"
                    orphaned_albums_info[album_key]["tracks"].append({
                        "item_id": track_id,
                        "title": title,
                        "author": author
                    })
                    orphaned_albums_info[album_key]["track_count"] += 1
            
            # Convert to list for JSON serialization
            orphaned_albums_list = []
            for artist, info in orphaned_albums_info.items():
                orphaned_albums_list.append({
                    "artist": artist,
                    "track_count": info["track_count"],
                    "tracks": info["tracks"]
                })
            
            # Sort by track count (albums with more tracks first)
            orphaned_albums_list.sort(key=lambda x: x["track_count"], reverse=True)
            
            # Safety check: limit deletion to prevent accidents
            total_orphaned_albums = len(orphaned_albums_list)
            safety_limit_applied = False
            if total_orphaned_albums > CLEANING_SAFETY_LIMIT:
                safety_limit_applied = True
                log_and_update_main(f"⚠️ Safety limit: Found {total_orphaned_albums} orphaned albums, limiting to first {CLEANING_SAFETY_LIMIT} for safety", 92)
                # Keep only first CLEANING_SAFETY_LIMIT albums
                orphaned_albums_list = orphaned_albums_list[:CLEANING_SAFETY_LIMIT]
                # Recalculate track IDs for limited albums
                limited_track_ids = set()
                for album in orphaned_albums_list:
                    for track in album["tracks"]:
                        limited_track_ids.add(track["item_id"])
                orphaned_track_ids = limited_track_ids
            
            if len(orphaned_track_ids) == 0:
                log_and_update_main("✅ No orphaned tracks found. Database is clean!", 95, task_state=TASK_STATUS_PROGRESS)
                log_and_update_main(f"🔄 Rebuilding all indexes and maps...", 96)
                try:
                    from .analysis import _run_all_index_builds
                    _run_all_index_builds(log_fn=None)
                    log_and_update_main(f"✅ All indexes and maps rebuilt successfully.", 99)
                except Exception as e:
                    logger.warning(f"Failed to rebuild indexes and maps: {e}")
                    log_and_update_main(f"⚠️ Warning: Failed to rebuild indexes and maps: {str(e)}", 99)
                
                summary = {
                    "total_media_server_albums": len(all_media_server_albums),
                    "total_media_server_tracks": len(media_server_track_ids),
                    "total_database_tracks": len(database_track_ids),
                    "orphaned_tracks_count": 0,
                    "orphaned_albums_count": 0,
                    "deleted_count": 0
                }
                
                log_and_update_main("✅ Database cleaning completed - no orphaned tracks found!", 100, task_state=TASK_STATUS_SUCCESS, final_summary_details=summary)
                return {
                    "status": "SUCCESS", 
                    "message": "No orphaned tracks found. Database is clean!",
                    **summary
                }
            
            log_and_update_main(f"🧹 Starting automatic deletion of {len(orphaned_track_ids)} orphaned tracks...", 93)
            
            # Step 6: Automatically delete all orphaned tracks
            deletion_result = delete_orphaned_albums_sync(list(orphaned_track_ids))
            
            summary = {
                "total_media_server_albums": len(all_media_server_albums),
                "total_media_server_tracks": len(media_server_track_ids),
                "total_database_tracks": len(database_track_ids),
                "orphaned_tracks_count": len(orphaned_track_ids),
                "orphaned_albums_count": len(orphaned_albums_list),
                "orphaned_albums": orphaned_albums_list,
                "deletion_result": deletion_result,
                "deleted_count": deletion_result.get("deleted_count", 0),
                "failed_deletions": deletion_result.get("failed_deletions", [])
            }
            
            if deletion_result["status"] == "SUCCESS":
                log_and_update_main(f"✅ Successfully deleted {deletion_result['deleted_count']} orphaned tracks.", 96)

                log_and_update_main(f"🔄 Rebuilding all indexes and maps after cleaning...", 97)
                try:
                    from .analysis import _run_all_index_builds
                    _run_all_index_builds(log_fn=None)
                    log_and_update_main(f"✅ All indexes and maps rebuilt successfully after cleaning.", 99)
                except Exception as e:
                    logger.warning(f"Failed to rebuild indexes and maps after cleaning: {e}")
                    log_and_update_main(f"⚠️ Warning: Failed to rebuild indexes and maps: {str(e)}", 99)
                
                safety_message = f" (Safety limit: deleted {len(orphaned_albums_list)} out of {total_orphaned_albums} albums)" if safety_limit_applied else ""
                
                log_and_update_main(
                    f"✅ Cleaning complete! Identified and deleted {len(orphaned_albums_list)} orphaned albums ({deletion_result['deleted_count']} tracks).{safety_message}", 
                    100, 
                    task_state=TASK_STATUS_SUCCESS,
                    final_summary_details=summary
                )
                
                # Only show additional cleanup message if we actually hit the safety limit
                if safety_limit_applied:
                    remaining_count = total_orphaned_albums - len(orphaned_albums_list) 
                    if remaining_count > 0:
                        log_and_update_main(f"ℹ️ Safety note: {remaining_count} additional orphaned albums remain. Run cleaning again to process more.", 100, task_state=TASK_STATUS_SUCCESS)
                
                return {
                    "status": "SUCCESS", 
                    "message": f"Successfully cleaned {deletion_result['deleted_count']} orphaned tracks from {len(orphaned_albums_list)} albums",
                    **summary
                }
            else:
                log_and_update_main(
                    f"⚠️ Cleaning partially failed. Deletion error: {deletion_result.get('message', 'Unknown error')}", 
                    100, 
                    task_state=TASK_STATUS_FAILURE,
                    final_summary_details=summary
                )
                raise Exception(f"Deletion failed: {deletion_result.get('message', 'Unknown error')}")

        except OperationalError as e:
            logger.error(f"Database connection error during cleaning identification: {e}. This job will be retried.", exc_info=True)
            err = error_manager.record(ERR_DB_CONNECTION, str(e), exc=e)
            log_and_update_main(f"Database connection failed. Retrying...", current_progress, task_state=TASK_STATUS_FAILURE, error=err, final_summary_details={"error": str(e)})
            raise
        except Exception as e:
            logger.critical(f"Orphaned album identification failed: {e}", exc_info=True)
            err = error_manager.record(error_manager.classify(e, ERR_CLEANING_FAILED), str(e), exc=e)
            log_and_update_main(f"❌ Orphaned album identification failed: {e}", current_progress, task_state=TASK_STATUS_FAILURE, error=err, final_summary_details={"error": str(e)})
            raise


def delete_orphaned_albums_sync(orphaned_track_ids):
    """
    Synchronous function to delete orphaned albums from the database.
    This function is called after user confirmation.
    
    Args:
        orphaned_track_ids (list): List of track IDs to delete from database
        
    Returns:
        dict: Result summary with deletion statistics
    """
    from app_helper import get_db
    
    if not orphaned_track_ids:
        return {"status": "SUCCESS", "message": "No tracks to delete", "deleted_count": 0}
    
    try:
        deleted_count = 0
        failed_deletions = []
        
        with get_db() as conn:
            with conn.cursor() as cur:
                def _table_exists(table_name):
                    """Return True if a regular table with this (unqualified) name exists
                    in the current search_path. Uses ``to_regclass`` so it never raises
                    even if the table is missing."""
                    try:
                        cur.execute("SELECT to_regclass(%s)", (table_name,))
                        row = cur.fetchone()
                        return bool(row and row[0] is not None)
                    except Exception as e:
                        logger.warning(f"Could not check existence of table {table_name}: {e}")
                        return False

                def _delete_from_child_table(table_name):
                    """Delete the orphaned track rows from a child table of `score`.
                    Skips silently if the table doesn't exist (older deployments)."""
                    if not _table_exists(table_name):
                        logger.info(f"Skipping {table_name}: table does not exist.")
                        return
                    logger.info(f"Deleting {len(orphaned_track_ids)} tracks from {table_name} table...")
                    for track_id in orphaned_track_ids:
                        try:
                            cur.execute(
                                f"DELETE FROM {table_name} WHERE item_id = %s",
                                (track_id,),
                            )
                            logger.debug(f"Deleted {table_name} for track ID: {track_id}")
                        except Exception as e:
                            logger.warning(f"Failed to delete {table_name} for track {track_id}: {e}")
                            failed_deletions.append({"track_id": track_id, "table": table_name, "error": str(e)})

                # Delete from child tables first (foreign key constraint).
                # All are declared with ON DELETE CASCADE on score(item_id), so this
                # is technically redundant — but explicit cleanup gives us per-row error
                # tracking via failed_deletions. Tables that don't exist on this
                # deployment are skipped silently.
                _delete_from_child_table("embedding")
                _delete_from_child_table("lyrics_embedding")
                _delete_from_child_table("clap_embedding")

                # Delete from score table
                logger.info(f"Deleting {len(orphaned_track_ids)} tracks from score table...")
                for track_id in orphaned_track_ids:
                    try:
                        cur.execute("DELETE FROM score WHERE item_id = %s", (track_id,))
                        if cur.rowcount > 0:
                            deleted_count += 1
                            logger.debug(f"Deleted score for track ID: {track_id}")
                        else:
                            logger.warning(f"No score record found for track ID: {track_id}")
                    except Exception as e:
                        logger.warning(f"Failed to delete score for track {track_id}: {e}")
                        failed_deletions.append({"track_id": track_id, "table": "score", "error": str(e)})
                
                # Commit the transaction
                conn.commit()
                logger.info(f"Successfully deleted {deleted_count} orphaned tracks from database")
        
        # Also clean up any related data that might reference these tracks
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    # Clean up playlist entries for deleted tracks
                    for track_id in orphaned_track_ids:
                        cur.execute("DELETE FROM playlist WHERE item_id = %s", (track_id,))
                    conn.commit()
                    logger.info("Cleaned up playlist references for deleted tracks")
        except Exception as e:
            logger.warning(f"Failed to clean up playlist references: {e}")
        
        # Clean up orphaned artists from artist_mapping table
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    # Find artists that no longer have any tracks in the score table
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
                        logger.info(f"Cleaned up {orphaned_artists_count} orphaned artists from artist_mapping table")
        except Exception as e:
            logger.warning(f"Failed to clean up orphaned artists from artist_mapping: {e}")
        
        return {
            "status": "SUCCESS",
            "message": f"Successfully deleted {deleted_count} orphaned tracks",
            "deleted_count": deleted_count,
            "failed_deletions": failed_deletions,
            "total_requested": len(orphaned_track_ids)
        }
        
    except Exception as e:
        logger.error(f"Failed to delete orphaned albums: {e}", exc_info=True)
        return {
            "status": "FAILURE",
            "message": f"Failed to delete orphaned albums: {str(e)}",
            "deleted_count": 0,
            "error": str(e)
        }