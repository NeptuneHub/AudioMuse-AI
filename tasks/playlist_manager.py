import logging
import time
import uuid
import traceback
from rq import get_current_job
from tasks.mediaserver import get_all_playlists, get_tracks_from_album

logger = logging.getLogger(__name__)

def fetch_playlists_from_mediaserver_task(parent_task_id=None):
    from app import app
    from app_helper import (redis_conn, get_db, save_task_status, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS, TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE)
    
    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        initial_details = {"log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Playlist fetch task started."]}
        save_task_status(current_task_id, "fetch_playlists", TASK_STATUS_STARTED, parent_task_id=parent_task_id, progress=0, details=initial_details)
        
        current_task_logs = initial_details["log"]
        current_progress = 0

        def log_and_update(message, progress, **kwargs):
            nonlocal current_progress, current_task_logs
            current_progress = progress
            logger.info(f"[FetchPlaylistsTask-{current_task_id}] {message}")
            details = {**kwargs, "status_message": message}
            log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            
            task_state = kwargs.get('task_state', TASK_STATUS_PROGRESS)
            if task_state != TASK_STATUS_SUCCESS:
                current_task_logs.append(log_entry)
                details["log"] = current_task_logs
            else:
                details["log"] = [f"Task completed successfully. Final status: {message}"]

            if current_job:
                current_job.meta.update({'progress': progress, 'status_message': message})
                current_job.save_meta()
            save_task_status(current_task_id, "fetch_playlists", task_state, parent_task_id=parent_task_id, progress=progress, details=details)

        try:
            log_and_update("Fetching playlists from media server...", 5)
            playlists = get_all_playlists()
            
            if not playlists:
                log_and_update("No playlists found on media server.", 100, task_state=TASK_STATUS_SUCCESS)
                return {"status": "SUCCESS", "message": "No playlists found."}

            total_playlists = len(playlists)
            log_and_update(f"Found {total_playlists} playlists. Processing...", 10)

            conn = get_db()
            cur = conn.cursor()
            
            for idx, playlist in enumerate(playlists):
                name = playlist.get('Name')
                p_id = playlist.get('Id') or playlist.get('id') # Handle different casing
                
                if not name or not p_id:
                    continue

                log_and_update(f"Processing playlist: {name} ({idx+1}/{total_playlists})", 10 + int(80 * (idx / total_playlists)))
                
                # Fetch tracks for this playlist
                # We use get_tracks_from_album because it uses ParentId which works for playlists in Jellyfin
                tracks = get_tracks_from_album(p_id)
                
                if not tracks:
                    logger.info(f"Playlist '{name}' is empty.")
                    continue

                # Update DB
                try:
                    # Delete existing entries for this playlist to ensure sync
                    cur.execute("DELETE FROM playlist WHERE playlist_name = %s", (name,))
                    
                    for track in tracks:
                        t_id = track.get('Id') or track.get('id')
                        t_name = track.get('Name')
                        t_artist = track.get('AlbumArtist') or track.get('Artist') or 'Unknown'
                        
                        if t_id and t_name:
                            cur.execute("INSERT INTO playlist (playlist_name, item_id, title, author) VALUES (%s, %s, %s, %s) ON CONFLICT (playlist_name, item_id) DO NOTHING", (name, t_id, t_name, t_artist))
                    
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to update playlist '{name}' in DB: {e}")
            
            cur.close()
            log_and_update("Successfully fetched and synced playlists.", 100, task_state=TASK_STATUS_SUCCESS)
            return {"status": "SUCCESS", "message": f"Synced {total_playlists} playlists."}

        except Exception as e:
            logger.error(f"Fetch playlists task failed: {e}", exc_info=True)
            log_and_update(f"Task failed: {e}", current_progress, task_state=TASK_STATUS_FAILURE, details={"error": str(e), "traceback": traceback.format_exc()})
            raise
