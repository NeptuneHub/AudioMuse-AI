# tasks/cleaning.py

import os
import time
import logging
import uuid
import traceback
import json
from collections import defaultdict

# RQ import
from rq import get_current_job
from rq.exceptions import NoSuchJobError

# Import configuration
from config import (
    REDIS_URL, DATABASE_URL, MAX_QUEUED_ANALYSIS_JOBS, CLEANING_SAFETY_LIMIT
)

# Import other project modules
from .mediaserver import get_recent_albums, get_tracks_from_album
from .voyager_manager import build_and_store_voyager_index

from psycopg2 import OperationalError
from redis.exceptions import TimeoutError as RedisTimeoutError

logger = logging.getLogger(__name__)


def identify_and_clean_orphaned_albums_task():
    """
    Main RQ task to identify and automatically clean orphaned albums from the database.
    This combines identification and deletion into a single automated process.
    """
    from app import app
    from app_helper import (redis_conn, get_db, save_task_status, get_task_info_from_db, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS, TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)

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
            nonlocal current_progress, current_task_logs
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
                # Still rebuild voyager index and map even when no albums found
                log_and_update_main(f"🔄 Rebuilding voyager index and map...", 98)
                try:
                    build_and_store_voyager_index(get_db())
                    from app_helper import build_and_store_map_projection
                    build_and_store_map_projection('main_map')
                    try:
                        redis_conn.publish('index-updates', 'reload')
                    except Exception:
                        logger.debug('Could not publish index-updates to redis after rebuild.')
                    log_and_update_main(f"✅ Voyager index and map rebuilt successfully.", 99)
                except Exception as e:
                    logger.warning(f"Failed to rebuild voyager index and map: {e}")
                    log_and_update_main(f"⚠️ Warning: Failed to rebuild voyager index and map: {str(e)}", 99)
                
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
                # Still rebuild voyager index and map even when no cleaning needed
                log_and_update_main(f"🔄 Rebuilding voyager index and map...", 98)
                try:
                    build_and_store_voyager_index(get_db())
                    from app_helper import build_and_store_map_projection
                    build_and_store_map_projection('main_map')
                    try:
                        redis_conn.publish('index-updates', 'reload')
                    except Exception:
                        logger.debug('Could not publish index-updates to redis after rebuild.')
                    log_and_update_main(f"✅ Voyager index and map rebuilt successfully.", 99)
                except Exception as e:
                    logger.warning(f"Failed to rebuild voyager index and map: {e}")
                    log_and_update_main(f"⚠️ Warning: Failed to rebuild voyager index and map: {str(e)}", 99)
                
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
                
                # Rebuild voyager index and map after cleaning like analysis does
                log_and_update_main(f"🔄 Rebuilding voyager index and map after cleaning...", 98)
                try:
                    build_and_store_voyager_index(get_db())
                    from app_helper import build_and_store_map_projection
                    build_and_store_map_projection('main_map')
                    try:
                        redis_conn.publish('index-updates', 'reload')
                    except Exception:
                        logger.debug('Could not publish index-updates to redis after rebuild.')
                    log_and_update_main(f"✅ Voyager index and map rebuilt successfully after cleaning.", 99)
                except Exception as e:
                    logger.warning(f"Failed to rebuild voyager index and map after cleaning: {e}")
                    log_and_update_main(f"⚠️ Warning: Failed to rebuild voyager index and map: {str(e)}", 99)
                
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
            log_and_update_main(f"Database connection failed. Retrying...", current_progress, task_state=TASK_STATUS_FAILURE, final_summary_details={"error": str(e), "traceback": traceback.format_exc()})
            raise
        except Exception as e:
            logger.critical(f"Orphaned album identification failed: {e}", exc_info=True)
            log_and_update_main(f"❌ Orphaned album identification failed: {e}", current_progress, task_state=TASK_STATUS_FAILURE, final_summary_details={"error": str(e), "traceback": traceback.format_exc()})
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
    from app import get_db
    
    if not orphaned_track_ids:
        return {"status": "SUCCESS", "message": "No tracks to delete", "deleted_count": 0}
    
    try:
        deleted_count = 0
        failed_deletions = []
        
        with get_db() as conn:
            with conn.cursor() as cur:
                # Delete from embedding table first (foreign key constraint)
                logger.info(f"Deleting {len(orphaned_track_ids)} tracks from embedding table...")
                for track_id in orphaned_track_ids:
                    try:
                        cur.execute("DELETE FROM embedding WHERE item_id = %s", (track_id,))
                        logger.debug(f"Deleted embedding for track ID: {track_id}")
                    except Exception as e:
                        logger.warning(f"Failed to delete embedding for track {track_id}: {e}")
                        failed_deletions.append({"track_id": track_id, "table": "embedding", "error": str(e)})
                
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