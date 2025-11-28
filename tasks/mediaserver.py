# tasks/mediaserver.py

import logging
import config  # Import the config module to access server type and settings

# Import the specific implementations
from tasks.mediaserver_jellyfin import (
    resolve_user as jellyfin_resolve_user,
    get_all_playlists as jellyfin_get_all_playlists,
    delete_playlist as jellyfin_delete_playlist,
    get_recent_albums as jellyfin_get_recent_albums,
    get_tracks_from_album as jellyfin_get_tracks_from_album,
    download_track as jellyfin_download_track,
    get_all_songs as jellyfin_get_all_songs,
    get_playlist_by_name as jellyfin_get_playlist_by_name,
    create_playlist as jellyfin_create_playlist,
    create_instant_playlist as jellyfin_create_instant_playlist,
    get_top_played_songs as jellyfin_get_top_played_songs,
    get_last_played_time as jellyfin_get_last_played_time,
)
from tasks.mediaserver_navidrome import (
    get_all_playlists as navidrome_get_all_playlists,
    delete_playlist as navidrome_delete_playlist,
    get_recent_albums as navidrome_get_recent_albums,
    get_tracks_from_album as navidrome_get_tracks_from_album,
    download_track as navidrome_download_track,
    get_all_songs as navidrome_get_all_songs,
    get_playlist_by_name as navidrome_get_playlist_by_name,
    create_playlist as navidrome_create_playlist,
    create_instant_playlist as navidrome_create_instant_playlist,
    get_top_played_songs as navidrome_get_top_played_songs,
    get_last_played_time as navidrome_get_last_played_time,
)
from tasks.mediaserver_lyrion import (
    get_all_playlists as lyrion_get_all_playlists,
    delete_playlist as lyrion_delete_playlist,
    get_recent_albums as lyrion_get_recent_albums,
    get_tracks_from_album as lyrion_get_tracks_from_album,
    download_track as lyrion_download_track,
    get_all_songs as lyrion_get_all_songs,
    get_playlist_by_name as lyrion_get_playlist_by_name,
    create_playlist as lyrion_create_playlist,
    create_instant_playlist as lyrion_create_instant_playlist,
    get_top_played_songs as lyrion_get_top_played_songs,
    get_last_played_time as lyrion_get_last_played_time,
)
from tasks.mediaserver_mpd import (
    get_all_playlists as mpd_get_all_playlists,
    delete_playlist as mpd_delete_playlist,
    get_recent_albums as mpd_get_recent_albums,
    get_tracks_from_album as mpd_get_tracks_from_album,
    download_track as mpd_download_track,
    get_all_songs as mpd_get_all_songs,
    get_playlist_by_name as mpd_get_playlist_by_name,
    create_playlist as mpd_create_playlist,
    create_instant_playlist as mpd_create_instant_playlist,
    get_top_played_songs as mpd_get_top_played_songs,
    get_last_played_time as mpd_get_last_played_time,
)
from tasks.mediaserver_emby import (
    resolve_user as emby_resolve_user,
    get_all_playlists as emby_get_all_playlists,
    delete_playlist as emby_delete_playlist,
    get_recent_albums as emby_get_recent_albums,
    get_recent_music_items as emby_get_recent_music_items,
    get_tracks_from_album as emby_get_tracks_from_album,
    download_track as emby_download_track,
    get_all_songs as emby_get_all_songs,
    get_playlist_by_name as emby_get_playlist_by_name,
    create_playlist as emby_create_playlist,
    create_instant_playlist as emby_create_instant_playlist,
    get_top_played_songs as emby_get_top_played_songs,
    get_last_played_time as emby_get_last_played_time,
)

logger = logging.getLogger(__name__)


# ##############################################################################
# PUBLIC API (Dispatcher functions)
# ##############################################################################

def resolve_emby_jellyfin_user(identifier, token):
    """Public dispatcher for resolving a Jellyfin or Emby user identifier."""
    # This is specific to Jellyfin, so we call it directly.
    if config.MEDIASERVER_TYPE == 'jellyfin': return jellyfin_resolve_user(identifier, token)
    if config.MEDIASERVER_TYPE == 'emby': return emby_resolve_user(identifier, token)
    return []

def delete_automatic_playlists():
    """Deletes all playlists ending with '_automatic' using admin credentials."""
    logger.info("Starting deletion of all '_automatic' playlists.")
    deleted_count = 0
    
    playlists_to_check = []
    delete_function = None

    if config.MEDIASERVER_TYPE == 'jellyfin':
        playlists_to_check = jellyfin_get_all_playlists()
        delete_function = jellyfin_delete_playlist
    elif config.MEDIASERVER_TYPE == 'navidrome':
        playlists_to_check = navidrome_get_all_playlists()
        delete_function = navidrome_delete_playlist
    elif config.MEDIASERVER_TYPE == 'lyrion':
        playlists_to_check = lyrion_get_all_playlists()
        delete_function = lyrion_delete_playlist
    elif config.MEDIASERVER_TYPE == 'mpd':
        playlists_to_check = mpd_get_all_playlists()
        delete_function = mpd_delete_playlist
    elif config.MEDIASERVER_TYPE == 'emby':
        playlists_to_check = emby_get_all_playlists()
        delete_function = emby_delete_playlist

    if delete_function:
        for p in playlists_to_check:
            # Navidrome uses 'id', others use 'Id'. Check for both.
            playlist_id = p.get('Id') or p.get('id')
            if p.get('Name', '').endswith('_automatic') and delete_function(playlist_id):
                deleted_count += 1
                
    logger.info(f"Finished deletion. Deleted {deleted_count} playlists.")

def get_recent_albums(limit):
    """Fetches recently added albums using admin credentials."""
    if config.MEDIASERVER_TYPE == 'jellyfin': return jellyfin_get_recent_albums(limit)
    if config.MEDIASERVER_TYPE == 'navidrome': return navidrome_get_recent_albums(limit)
    if config.MEDIASERVER_TYPE == 'lyrion': return lyrion_get_recent_albums(limit)
    if config.MEDIASERVER_TYPE == 'mpd': return mpd_get_recent_albums(limit)
    if config.MEDIASERVER_TYPE == 'emby': return emby_get_recent_albums(limit)
    return []

def get_recent_music_items(limit):
    """
    Fetches both recent albums AND standalone tracks for comprehensive music discovery.
    This ensures no music is missed during analysis, even with incomplete metadata.
    Now implemented for Jellyfin, Navidrome, and Lyrion - all provide comprehensive discovery.
    """
    if config.MEDIASERVER_TYPE == 'jellyfin': 
        return jellyfin_get_recent_music_items(limit)
    elif config.MEDIASERVER_TYPE == 'navidrome': 
        return navidrome_get_recent_music_items(limit)
    elif config.MEDIASERVER_TYPE == 'lyrion': 
        return lyrion_get_recent_music_items(limit)
    elif config.MEDIASERVER_TYPE == 'emby': 
        return emby_get_recent_music_items(limit)
    else:
        # Fallback to regular album fetching for servers without comprehensive discovery
        logger.info(f"get_recent_music_items not yet implemented for {config.MEDIASERVER_TYPE}, falling back to get_recent_albums")
        return get_recent_albums(limit)

def get_tracks_from_album(album_id):
    """Fetches tracks for an album using admin credentials."""
    if config.MEDIASERVER_TYPE == 'jellyfin': return jellyfin_get_tracks_from_album(album_id)
    if config.MEDIASERVER_TYPE == 'navidrome': return navidrome_get_tracks_from_album(album_id)
    if config.MEDIASERVER_TYPE == 'lyrion': return lyrion_get_tracks_from_album(album_id)
    if config.MEDIASERVER_TYPE == 'mpd': return mpd_get_tracks_from_album(album_id)
    if config.MEDIASERVER_TYPE == 'emby': return emby_get_tracks_from_album(album_id)
    return []

def download_track(temp_dir, item):
    """Downloads a track using admin credentials."""
    if config.MEDIASERVER_TYPE == 'jellyfin': return jellyfin_download_track(temp_dir, item)
    if config.MEDIASERVER_TYPE == 'navidrome': return navidrome_download_track(temp_dir, item)
    if config.MEDIASERVER_TYPE == 'lyrion': return lyrion_download_track(temp_dir, item)
    if config.MEDIASERVER_TYPE == 'mpd': return mpd_download_track(temp_dir, item)
    if config.MEDIASERVER_TYPE == 'emby': return emby_download_track(temp_dir, item)
    return None

def get_all_songs():
    """Fetches all songs using admin credentials."""
    if config.MEDIASERVER_TYPE == 'jellyfin': return jellyfin_get_all_songs()
    if config.MEDIASERVER_TYPE == 'navidrome': return navidrome_get_all_songs()
    if config.MEDIASERVER_TYPE == 'lyrion': return lyrion_get_all_songs()
    if config.MEDIASERVER_TYPE == 'mpd': return mpd_get_all_songs()
    if config.MEDIASERVER_TYPE == 'emby': return emby_get_all_songs()
    return []

def get_all_playlists():
    """Fetches all playlists using admin credentials."""
    if config.MEDIASERVER_TYPE == 'jellyfin': return jellyfin_get_all_playlists()
    if config.MEDIASERVER_TYPE == 'navidrome': return navidrome_get_all_playlists()
    if config.MEDIASERVER_TYPE == 'lyrion': return lyrion_get_all_playlists()
    if config.MEDIASERVER_TYPE == 'mpd': return mpd_get_all_playlists()
    if config.MEDIASERVER_TYPE == 'emby': return emby_get_all_playlists()
    return []

def get_playlist_by_name(playlist_name):
    """Finds a playlist by name using admin credentials."""
    if not playlist_name: raise ValueError("Playlist name is required.")
    if config.MEDIASERVER_TYPE == 'jellyfin': return jellyfin_get_playlist_by_name(playlist_name)
    if config.MEDIASERVER_TYPE == 'navidrome': return navidrome_get_playlist_by_name(playlist_name)
    if config.MEDIASERVER_TYPE == 'lyrion': return lyrion_get_playlist_by_name(playlist_name)
    if config.MEDIASERVER_TYPE == 'mpd': return mpd_get_playlist_by_name(playlist_name)
    if config.MEDIASERVER_TYPE == 'emby': return emby_get_playlist_by_name(playlist_name)
    return None

def create_playlist(base_name, item_ids):
    """Creates a playlist using admin credentials."""
    if not base_name: raise ValueError("Playlist name is required.")
    if not item_ids: raise ValueError("Track IDs are required.")
    if config.MEDIASERVER_TYPE == 'jellyfin': jellyfin_create_playlist(base_name, item_ids)
    elif config.MEDIASERVER_TYPE == 'navidrome': navidrome_create_playlist(base_name, item_ids)
    elif config.MEDIASERVER_TYPE == 'lyrion': lyrion_create_playlist(base_name, item_ids)
    elif config.MEDIASERVER_TYPE == 'mpd': mpd_create_playlist(base_name, item_ids)
    elif config.MEDIASERVER_TYPE == 'emby': emby_create_playlist(base_name, item_ids)

def create_instant_playlist(playlist_name, item_ids, user_creds=None, add_instant_suffix=True):
    """Creates an instant playlist. Uses user_creds if provided, otherwise admin."""
    if not playlist_name: raise ValueError("Playlist name is required.")
    if not item_ids: raise ValueError("Track IDs are required.")

    if config.MEDIASERVER_TYPE == 'jellyfin':
        return jellyfin_create_instant_playlist(playlist_name, item_ids, user_creds, add_instant_suffix)
    if config.MEDIASERVER_TYPE == 'navidrome':
        return navidrome_create_instant_playlist(playlist_name, item_ids, user_creds, add_instant_suffix)
    if config.MEDIASERVER_TYPE == 'lyrion':
        return lyrion_create_instant_playlist(playlist_name, item_ids, add_instant_suffix)
    if config.MEDIASERVER_TYPE == 'mpd':
        return mpd_create_instant_playlist(playlist_name, item_ids, user_creds, add_instant_suffix)
    if config.MEDIASERVER_TYPE == 'emby':
        return emby_create_instant_playlist(playlist_name, item_ids, user_creds, add_instant_suffix)
    return None

def get_top_played_songs(limit, user_creds=None):
    """Fetches top played songs. Uses user_creds if provided, otherwise admin."""
    if config.MEDIASERVER_TYPE == 'jellyfin':
        return jellyfin_get_top_played_songs(limit, user_creds)
    if config.MEDIASERVER_TYPE == 'navidrome':
        return navidrome_get_top_played_songs(limit, user_creds)
    if config.MEDIASERVER_TYPE == 'lyrion':
        return lyrion_get_top_played_songs(limit)
    if config.MEDIASERVER_TYPE == 'mpd':
        return mpd_get_top_played_songs(limit, user_creds)
    if config.MEDIASERVER_TYPE == 'emby':
        return emby_get_top_played_songs(limit, user_creds)
    return []

def get_last_played_time(item_id, user_creds=None):
    """Fetches last played time for a track. Uses user_creds if provided, otherwise admin."""
    if config.MEDIASERVER_TYPE == 'jellyfin':
        return jellyfin_get_last_played_time(item_id, user_creds)
    if config.MEDIASERVER_TYPE == 'navidrome':
        return navidrome_get_last_played_time(item_id, user_creds)
    if config.MEDIASERVER_TYPE == 'lyrion':
        return lyrion_get_last_played_time(item_id)
    if config.MEDIASERVER_TYPE == 'mpd':
        return mpd_get_last_played_time(item_id, user_creds)
    if config.MEDIASERVER_TYPE == 'emby':
        return emby_get_last_played_time(item_id, user_creds)
    return None

