# tasks/mediaserver.py
"""
Media Server Dispatcher for AudioMuse-AI

This module provides a unified interface to multiple media server providers.
It dispatches function calls to the appropriate provider implementation based
on the configured MEDIASERVER_TYPE.

Supported providers:
- jellyfin: Jellyfin Media Server
- navidrome: Navidrome (Subsonic API)
- lyrion: Lyrion Music Server (formerly LMS)
- emby: Emby Media Server
- localfiles: Local file system scanner

Multi-provider support:
When multi_provider_enabled is true in app_settings, multiple providers can be
configured and used simultaneously. Tracks are linked via file paths, allowing
analysis data to be shared across providers.
"""

import logging
import os
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
from tasks.mediaserver_localfiles import (
    get_all_playlists as localfiles_get_all_playlists,
    delete_playlist as localfiles_delete_playlist,
    get_recent_albums as localfiles_get_recent_albums,
    get_tracks_from_album as localfiles_get_tracks_from_album,
    download_track as localfiles_download_track,
    get_all_songs as localfiles_get_all_songs,
    get_playlist_by_name as localfiles_get_playlist_by_name,
    create_playlist as localfiles_create_playlist,
    create_instant_playlist as localfiles_create_instant_playlist,
    get_top_played_songs as localfiles_get_top_played_songs,
    get_last_played_time as localfiles_get_last_played_time,
    test_connection as localfiles_test_connection,
    get_provider_info as localfiles_get_provider_info,
)

logger = logging.getLogger(__name__)


# ##############################################################################
# PROVIDER REGISTRY
# ##############################################################################

PROVIDER_TYPES = {
    'jellyfin': {
        'name': 'Jellyfin',
        'description': 'Jellyfin Media Server - Open source media solution',
        'supports_user_auth': True,
        'supports_play_history': True,
    },
    'navidrome': {
        'name': 'Navidrome',
        'description': 'Navidrome - Modern music server (Subsonic API)',
        'supports_user_auth': True,
        'supports_play_history': True,
    },
    'lyrion': {
        'name': 'Lyrion',
        'description': 'Lyrion Music Server (formerly Logitech Media Server)',
        'supports_user_auth': False,
        'supports_play_history': True,
    },
    'emby': {
        'name': 'Emby',
        'description': 'Emby Media Server - Personal media server',
        'supports_user_auth': True,
        'supports_play_history': True,
    },
    'localfiles': {
        'name': 'Local Files',
        'description': 'Scan local directories for audio files',
        'supports_user_auth': False,
        'supports_play_history': False,
    },
}


def get_available_provider_types():
    """Return information about all available provider types."""
    return PROVIDER_TYPES.copy()


# ##############################################################################
# PROVIDER FACTORY
# ##############################################################################

# Provider module mapping for dynamic dispatch
PROVIDER_MODULES = {
    'jellyfin': 'tasks.mediaserver_jellyfin',
    'navidrome': 'tasks.mediaserver_navidrome',
    'lyrion': 'tasks.mediaserver_lyrion',
    'emby': 'tasks.mediaserver_emby',
    'localfiles': 'tasks.mediaserver_localfiles',
}

# Pre-imported function maps for performance (avoid repeated imports)
_PROVIDER_FUNCTIONS = {
    'jellyfin': {
        'get_all_playlists': jellyfin_get_all_playlists,
        'delete_playlist': jellyfin_delete_playlist,
        'get_recent_albums': jellyfin_get_recent_albums,
        'get_tracks_from_album': jellyfin_get_tracks_from_album,
        'download_track': jellyfin_download_track,
        'get_all_songs': jellyfin_get_all_songs,
        'get_playlist_by_name': jellyfin_get_playlist_by_name,
        'create_playlist': jellyfin_create_playlist,
        'create_instant_playlist': jellyfin_create_instant_playlist,
        'get_top_played_songs': jellyfin_get_top_played_songs,
        'get_last_played_time': jellyfin_get_last_played_time,
    },
    'navidrome': {
        'get_all_playlists': navidrome_get_all_playlists,
        'delete_playlist': navidrome_delete_playlist,
        'get_recent_albums': navidrome_get_recent_albums,
        'get_tracks_from_album': navidrome_get_tracks_from_album,
        'download_track': navidrome_download_track,
        'get_all_songs': navidrome_get_all_songs,
        'get_playlist_by_name': navidrome_get_playlist_by_name,
        'create_playlist': navidrome_create_playlist,
        'create_instant_playlist': navidrome_create_instant_playlist,
        'get_top_played_songs': navidrome_get_top_played_songs,
        'get_last_played_time': navidrome_get_last_played_time,
    },
    'lyrion': {
        'get_all_playlists': lyrion_get_all_playlists,
        'delete_playlist': lyrion_delete_playlist,
        'get_recent_albums': lyrion_get_recent_albums,
        'get_tracks_from_album': lyrion_get_tracks_from_album,
        'download_track': lyrion_download_track,
        'get_all_songs': lyrion_get_all_songs,
        'get_playlist_by_name': lyrion_get_playlist_by_name,
        'create_playlist': lyrion_create_playlist,
        'create_instant_playlist': lyrion_create_instant_playlist,
        'get_top_played_songs': lyrion_get_top_played_songs,
        'get_last_played_time': lyrion_get_last_played_time,
    },
    'emby': {
        'get_all_playlists': emby_get_all_playlists,
        'delete_playlist': emby_delete_playlist,
        'get_recent_albums': emby_get_recent_albums,
        'get_tracks_from_album': emby_get_tracks_from_album,
        'download_track': emby_download_track,
        'get_all_songs': emby_get_all_songs,
        'get_playlist_by_name': emby_get_playlist_by_name,
        'create_playlist': emby_create_playlist,
        'create_instant_playlist': emby_create_instant_playlist,
        'get_top_played_songs': emby_get_top_played_songs,
        'get_last_played_time': emby_get_last_played_time,
    },
    'localfiles': {
        'get_all_playlists': localfiles_get_all_playlists,
        'delete_playlist': localfiles_delete_playlist,
        'get_recent_albums': localfiles_get_recent_albums,
        'get_tracks_from_album': localfiles_get_tracks_from_album,
        'download_track': localfiles_download_track,
        'get_all_songs': localfiles_get_all_songs,
        'get_playlist_by_name': localfiles_get_playlist_by_name,
        'create_playlist': localfiles_create_playlist,
        'create_instant_playlist': localfiles_create_instant_playlist,
        'get_top_played_songs': localfiles_get_top_played_songs,
        'get_last_played_time': localfiles_get_last_played_time,
    },
}


def get_provider_function(provider_type: str, function_name: str):
    """
    Get a specific function for a provider type.

    Args:
        provider_type: Type of provider (jellyfin, navidrome, etc.)
        function_name: Name of the function to retrieve

    Returns:
        The provider function, or None if not found

    Usage:
        get_all_songs = get_provider_function('jellyfin', 'get_all_songs')
        songs = get_all_songs()
    """
    provider_funcs = _PROVIDER_FUNCTIONS.get(provider_type)
    if not provider_funcs:
        logger.warning(f"Unknown provider type: {provider_type}")
        return None
    return provider_funcs.get(function_name)


def dispatch_to_provider(function_name: str, provider_type: str = None, *args, **kwargs):
    """
    Dispatch a function call to the appropriate provider.

    Args:
        function_name: Name of the function to call
        provider_type: Optional provider type override (defaults to config.MEDIASERVER_TYPE)
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Result of the provider function call

    Usage:
        songs = dispatch_to_provider('get_all_songs')
        songs = dispatch_to_provider('get_all_songs', provider_type='navidrome')
    """
    if provider_type is None:
        provider_type = config.MEDIASERVER_TYPE

    func = get_provider_function(provider_type, function_name)
    if func is None:
        raise ValueError(f"Function '{function_name}' not found for provider '{provider_type}'")

    return func(*args, **kwargs)


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
    if config.MEDIASERVER_TYPE == 'emby': return emby_get_recent_albums(limit)
    if config.MEDIASERVER_TYPE == 'localfiles': return localfiles_get_recent_albums(limit)
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
    if config.MEDIASERVER_TYPE == 'emby': return emby_get_tracks_from_album(album_id)
    if config.MEDIASERVER_TYPE == 'localfiles': return localfiles_get_tracks_from_album(album_id)
    return []

def download_track(temp_dir, item):
    """Downloads a track using admin credentials. Detects format from file if .tmp extension is used."""
    downloaded_path = None
    
    if config.MEDIASERVER_TYPE == 'jellyfin': downloaded_path = jellyfin_download_track(temp_dir, item)
    elif config.MEDIASERVER_TYPE == 'navidrome': downloaded_path = navidrome_download_track(temp_dir, item)
    elif config.MEDIASERVER_TYPE == 'lyrion': downloaded_path = lyrion_download_track(temp_dir, item)
    elif config.MEDIASERVER_TYPE == 'emby': downloaded_path = emby_download_track(temp_dir, item)
    elif config.MEDIASERVER_TYPE == 'localfiles': downloaded_path = localfiles_download_track(temp_dir, item)
    
    # If download failed or returned None, return as is
    if not downloaded_path:
        return None
    
    # If file has .tmp extension, try to detect real format from file content
    if downloaded_path.endswith('.tmp'):
        try:
            # Check if file exists before trying to detect format
            if not os.path.exists(downloaded_path):
                logger.warning(f"Downloaded file does not exist: {downloaded_path}")
                return downloaded_path
                
            detected_ext = _detect_audio_format(downloaded_path)
            if detected_ext and detected_ext != '.tmp':
                new_path = downloaded_path.replace('.tmp', detected_ext)
                # Check if target file already exists (avoid overwriting)
                if os.path.exists(new_path):
                    logger.warning(f"Target file already exists, keeping .tmp: {new_path}")
                    return downloaded_path
                os.rename(downloaded_path, new_path)
                logger.info(f"Detected format and renamed: {os.path.basename(downloaded_path)} -> {os.path.basename(new_path)}")
                return new_path
        except Exception as e:
            logger.debug(f"Format detection failed for {os.path.basename(downloaded_path)}, keeping .tmp: {e}")
    
    return downloaded_path


def _detect_audio_format(filepath):
    """Detects audio format from file magic numbers. Returns extension like '.mp3' or '.flac'."""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(12)
            
            # Check magic numbers for common audio formats
            if len(header) < 4:
                return '.tmp'
            
            # FLAC: fLaC
            if header[:4] == b'fLaC':
                return '.flac'
            
            # MP3: ID3 tag or MP3 sync bits
            if header[:3] == b'ID3' or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
                return '.mp3'
            
            # OGG: OggS
            if header[:4] == b'OggS':
                return '.ogg'
            
            # WAV/RIFF: RIFF....WAVE
            if header[:4] == b'RIFF' and len(header) >= 12 and header[8:12] == b'WAVE':
                return '.wav'
            
            # M4A/AAC: ftyp
            if len(header) >= 8 and header[4:8] == b'ftyp':
                return '.m4a'
            
            # WMA: ASF header
            if header[:4] == b'\x30\x26\xb2\x75':
                return '.wma'
            
            logger.debug(f"Unknown audio format, header: {header[:4].hex()}")
            return '.tmp'
            
    except Exception as e:
        logger.debug(f"Error detecting audio format: {e}")
        return '.tmp'

def get_all_songs():
    """Fetches all songs using admin credentials."""
    if config.MEDIASERVER_TYPE == 'jellyfin': return jellyfin_get_all_songs()
    if config.MEDIASERVER_TYPE == 'navidrome': return navidrome_get_all_songs()
    if config.MEDIASERVER_TYPE == 'lyrion': return lyrion_get_all_songs()
    if config.MEDIASERVER_TYPE == 'emby': return emby_get_all_songs()
    if config.MEDIASERVER_TYPE == 'localfiles': return localfiles_get_all_songs()
    return []

def get_playlist_by_name(playlist_name):
    """Finds a playlist by name using admin credentials."""
    if not playlist_name: raise ValueError("Playlist name is required.")
    if config.MEDIASERVER_TYPE == 'jellyfin': return jellyfin_get_playlist_by_name(playlist_name)
    if config.MEDIASERVER_TYPE == 'navidrome': return navidrome_get_playlist_by_name(playlist_name)
    if config.MEDIASERVER_TYPE == 'lyrion': return lyrion_get_playlist_by_name(playlist_name)
    if config.MEDIASERVER_TYPE == 'emby': return emby_get_playlist_by_name(playlist_name)
    if config.MEDIASERVER_TYPE == 'localfiles': return localfiles_get_playlist_by_name(playlist_name)
    return None

def create_playlist(base_name, item_ids):
    """Creates a playlist using admin credentials."""
    if not base_name: raise ValueError("Playlist name is required.")
    if not item_ids: raise ValueError("Track IDs are required.")
    if config.MEDIASERVER_TYPE == 'jellyfin': jellyfin_create_playlist(base_name, item_ids)
    elif config.MEDIASERVER_TYPE == 'navidrome': navidrome_create_playlist(base_name, item_ids)
    elif config.MEDIASERVER_TYPE == 'lyrion': lyrion_create_playlist(base_name, item_ids)
    elif config.MEDIASERVER_TYPE == 'emby': emby_create_playlist(base_name, item_ids)
    elif config.MEDIASERVER_TYPE == 'localfiles': localfiles_create_playlist(base_name, item_ids)

def create_instant_playlist(playlist_name, item_ids, user_creds=None):
    """Creates an instant playlist. Uses user_creds if provided, otherwise admin."""
    if not playlist_name: raise ValueError("Playlist name is required.")
    if not item_ids: raise ValueError("Track IDs are required.")

    if config.MEDIASERVER_TYPE == 'jellyfin':
        return jellyfin_create_instant_playlist(playlist_name, item_ids, user_creds)
    if config.MEDIASERVER_TYPE == 'navidrome':
        return navidrome_create_instant_playlist(playlist_name, item_ids, user_creds)
    if config.MEDIASERVER_TYPE == 'lyrion':
        return lyrion_create_instant_playlist(playlist_name, item_ids)
    if config.MEDIASERVER_TYPE == 'emby':
        return emby_create_instant_playlist(playlist_name, item_ids, user_creds)
    if config.MEDIASERVER_TYPE == 'localfiles':
        return localfiles_create_instant_playlist(playlist_name, item_ids, user_creds)
    return None

def get_top_played_songs(limit, user_creds=None):
    """Fetches top played songs. Uses user_creds if provided, otherwise admin."""
    if config.MEDIASERVER_TYPE == 'jellyfin':
        return jellyfin_get_top_played_songs(limit, user_creds)
    if config.MEDIASERVER_TYPE == 'navidrome':
        return navidrome_get_top_played_songs(limit, user_creds)
    if config.MEDIASERVER_TYPE == 'lyrion':
        return lyrion_get_top_played_songs(limit)
    if config.MEDIASERVER_TYPE == 'emby':
        return emby_get_top_played_songs(limit, user_creds)
    if config.MEDIASERVER_TYPE == 'localfiles':
        return localfiles_get_top_played_songs(limit, user_creds)
    return []

def get_last_played_time(item_id, user_creds=None):
    """Fetches last played time for a track. Uses user_creds if provided, otherwise admin."""
    if config.MEDIASERVER_TYPE == 'jellyfin':
        return jellyfin_get_last_played_time(item_id, user_creds)
    if config.MEDIASERVER_TYPE == 'navidrome':
        return navidrome_get_last_played_time(item_id, user_creds)
    if config.MEDIASERVER_TYPE == 'lyrion':
        return lyrion_get_last_played_time(item_id)
    if config.MEDIASERVER_TYPE == 'emby':
        return emby_get_last_played_time(item_id, user_creds)
    if config.MEDIASERVER_TYPE == 'localfiles':
        return localfiles_get_last_played_time(item_id, user_creds)
    return None


# ##############################################################################
# MULTI-PROVIDER SUPPORT FUNCTIONS
# ##############################################################################

def test_provider_connection(provider_type: str, config_dict: dict = None):
    """
    Test connection to a specific provider.

    Args:
        provider_type: Type of provider (jellyfin, navidrome, localfiles, etc.)
        config_dict: Optional configuration dictionary for the provider

    Returns:
        Tuple of (success: bool, message: str)
    """
    import requests

    try:
        if provider_type == 'localfiles':
            return localfiles_test_connection(config_dict)

        elif provider_type == 'jellyfin':
            url = config_dict.get('url') if config_dict else config.JELLYFIN_URL
            token = config_dict.get('token') if config_dict else config.JELLYFIN_TOKEN
            if not url or not token:
                return False, "Jellyfin URL and token are required"
            resp = requests.get(f"{url.rstrip('/')}/System/Info",
                              headers={"X-Emby-Token": token}, timeout=10)
            if resp.status_code == 200:
                return True, f"Connected to Jellyfin at {url}"
            return False, f"Jellyfin returned status {resp.status_code}"

        elif provider_type == 'navidrome':
            import hashlib
            import secrets
            url = config_dict.get('url') if config_dict else config.NAVIDROME_URL
            user = config_dict.get('user') if config_dict else config.NAVIDROME_USER
            password = config_dict.get('password') if config_dict else config.NAVIDROME_PASSWORD
            if not url or not user or not password:
                return False, "Navidrome URL, user, and password are required"
            salt = secrets.token_hex(8)
            token = hashlib.md5((password + salt).encode()).hexdigest()
            params = {'u': user, 't': token, 's': salt, 'v': '1.16.1', 'c': 'audiomuse', 'f': 'json'}
            resp = requests.get(f"{url.rstrip('/')}/rest/ping", params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('subsonic-response', {}).get('status') == 'ok':
                    return True, f"Connected to Navidrome at {url}"
                err = data.get('subsonic-response', {}).get('error', {}).get('message', 'Unknown error')
                return False, f"Navidrome error: {err}"
            return False, f"Navidrome returned status {resp.status_code}"

        elif provider_type == 'lyrion':
            url = config_dict.get('url') if config_dict else config.LYRION_URL
            if not url:
                return False, "Lyrion URL is required"
            resp = requests.get(f"{url.rstrip('/')}/status.html", timeout=10)
            if resp.status_code == 200:
                return True, f"Connected to Lyrion at {url}"
            return False, f"Lyrion returned status {resp.status_code}"

        elif provider_type == 'emby':
            url = config_dict.get('url') if config_dict else config.EMBY_URL
            token = config_dict.get('token') if config_dict else config.EMBY_TOKEN
            if not url or not token:
                return False, "Emby URL and token are required"
            resp = requests.get(f"{url.rstrip('/')}/System/Info",
                              headers={"X-Emby-Token": token}, timeout=10)
            if resp.status_code == 200:
                return True, f"Connected to Emby at {url}"
            return False, f"Emby returned status {resp.status_code}"

        else:
            return False, f"Unknown provider type: {provider_type}"

    except requests.RequestException as e:
        return False, f"Network error: {str(e)}"
    except Exception as e:
        return False, f"Connection test failed: {str(e)}"


def get_sample_tracks_from_provider(provider_type: str, config_dict: dict, limit: int = 50):
    """
    Fetch sample tracks from a provider using provided configuration.

    This is used during provider setup to detect the music_path_prefix
    by comparing track paths with existing data.

    Args:
        provider_type: Type of provider (jellyfin, navidrome, etc.)
        config_dict: Configuration dictionary for the provider
        limit: Maximum number of tracks to fetch

    Returns:
        List of track dicts with keys: title, artist, file_path
    """
    import requests

    try:
        if provider_type == 'jellyfin':
            url = config_dict.get('url')
            user_id = config_dict.get('user_id')
            token = config_dict.get('token')
            if not url or not user_id or not token:
                return []

            api_url = f"{url.rstrip('/')}/Users/{user_id}/Items"
            headers = {"X-Emby-Token": token}
            params = {
                "IncludeItemTypes": "Audio",
                "Recursive": True,
                "Fields": "Path",
                "Limit": limit
            }
            r = requests.get(api_url, headers=headers, params=params, timeout=15)
            if r.status_code != 200:
                return []

            items = r.json().get("Items", [])
            tracks = []
            for item in items:
                tracks.append({
                    'title': item.get('Name'),
                    'artist': item.get('AlbumArtist') or (item.get('Artists', [None])[0] if item.get('Artists') else None),
                    'file_path': item.get('Path')
                })
            return tracks

        elif provider_type == 'navidrome':
            import hashlib
            import secrets

            url = config_dict.get('url')
            user = config_dict.get('user')
            password = config_dict.get('password')
            if not url or not user or not password:
                return []

            salt = secrets.token_hex(8)
            token = hashlib.md5((password + salt).encode()).hexdigest()
            params = {
                'u': user, 't': token, 's': salt,
                'v': '1.16.1', 'c': 'audiomuse', 'f': 'json',
                'query': '', 'songCount': limit, 'songOffset': 0
            }
            r = requests.get(f"{url.rstrip('/')}/rest/search3", params=params, timeout=15)
            if r.status_code != 200:
                return []

            data = r.json()
            songs = data.get('subsonic-response', {}).get('searchResult3', {}).get('song', [])
            tracks = []
            for s in songs:
                tracks.append({
                    'title': s.get('title'),
                    'artist': s.get('artist'),
                    'file_path': s.get('path')
                })
            return tracks

        elif provider_type == 'emby':
            url = config_dict.get('url')
            user_id = config_dict.get('user_id')
            token = config_dict.get('token')
            if not url or not user_id or not token:
                return []

            api_url = f"{url.rstrip('/')}/Users/{user_id}/Items"
            headers = {"X-Emby-Token": token}
            params = {
                "IncludeItemTypes": "Audio",
                "Recursive": True,
                "Fields": "Path",
                "Limit": limit
            }
            r = requests.get(api_url, headers=headers, params=params, timeout=15)
            if r.status_code != 200:
                return []

            items = r.json().get("Items", [])
            tracks = []
            for item in items:
                tracks.append({
                    'title': item.get('Name'),
                    'artist': item.get('AlbumArtist') or (item.get('Artists', [None])[0] if item.get('Artists') else None),
                    'file_path': item.get('Path')
                })
            return tracks

        elif provider_type == 'lyrion':
            url = config_dict.get('url')
            if not url:
                return []

            # Lyrion uses JSON-RPC for queries
            api_url = f"{url.rstrip('/')}/jsonrpc.js"
            payload = {
                "id": 1,
                "method": "slim.request",
                "params": ["", ["titles", "0", str(limit), "tags:aspu"]]
            }
            r = requests.post(api_url, json=payload, timeout=15)
            if r.status_code != 200:
                return []

            data = r.json()
            titles_loop = data.get('result', {}).get('titles_loop', [])
            tracks = []
            for t in titles_loop:
                tracks.append({
                    'title': t.get('title'),
                    'artist': t.get('artist'),
                    'file_path': t.get('url')  # Lyrion uses 'url' for file path
                })
            return tracks

        elif provider_type == 'localfiles':
            import os
            music_dir = config_dict.get('music_directory')
            if not music_dir or not os.path.isdir(music_dir):
                return []

            formats = config_dict.get('supported_formats', '.mp3,.flac,.ogg,.m4a,.wav,.wma,.aac')
            if isinstance(formats, str):
                formats = [f.strip().lower() for f in formats.split(',')]

            tracks = []
            count = 0
            for root, dirs, files in os.walk(music_dir):
                for f in files:
                    if count >= limit:
                        break
                    ext = os.path.splitext(f)[1].lower()
                    if ext in formats or ext.lstrip('.') in [fmt.lstrip('.') for fmt in formats]:
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, music_dir)
                        # Extract artist/title from path structure (Artist/Album/Track.ext)
                        parts = rel_path.split(os.sep)
                        artist = parts[0] if len(parts) > 1 else 'Unknown'
                        title = os.path.splitext(parts[-1])[0]
                        tracks.append({
                            'title': title,
                            'artist': artist,
                            'file_path': rel_path
                        })
                        count += 1
                if count >= limit:
                    break
            return tracks

        else:
            return []

    except Exception as e:
        logger.error(f"Error fetching sample tracks from {provider_type}: {e}")
        return []


def get_provider_info(provider_type: str):
    """Get detailed information about a provider type including config fields."""
    if provider_type == 'localfiles':
        return localfiles_get_provider_info()

    # Return basic info for other providers
    if provider_type in PROVIDER_TYPES:
        info = PROVIDER_TYPES[provider_type].copy()
        info['type'] = provider_type
        info['config_fields'] = _get_provider_config_fields(provider_type)
        return info

    return None


def _get_provider_config_fields(provider_type: str):
    """Get configuration fields for a provider type."""
    # Common field for path normalization in multi-provider setups
    music_path_prefix_field = {
        'name': 'music_path_prefix',
        'label': 'Music Path Prefix',
        'type': 'text',
        'required': False,
        'description': 'Folder prefix to strip for cross-provider matching (e.g., "MyLibrary" if paths include library name). Leave empty if provider paths start directly with artist folders.',
    }

    fields = {
        'jellyfin': [
            {'name': 'url', 'label': 'Server URL', 'type': 'url', 'required': True,
             'description': 'Jellyfin server URL (e.g., http://192.168.1.100:8096)'},
            {'name': 'user_id', 'label': 'User ID', 'type': 'text', 'required': True,
             'description': 'Jellyfin user ID (found in dashboard)'},
            {'name': 'token', 'label': 'API Token', 'type': 'password', 'required': True,
             'description': 'API key from Jellyfin settings'},
            music_path_prefix_field,
        ],
        'navidrome': [
            {'name': 'url', 'label': 'Server URL', 'type': 'url', 'required': True,
             'description': 'Navidrome server URL (e.g., http://192.168.1.100:4533)'},
            {'name': 'user', 'label': 'Username', 'type': 'text', 'required': True,
             'description': 'Navidrome username'},
            {'name': 'password', 'label': 'Password', 'type': 'password', 'required': True,
             'description': 'Navidrome password'},
            music_path_prefix_field,
        ],
        'lyrion': [
            {'name': 'url', 'label': 'Server URL', 'type': 'url', 'required': True,
             'description': 'Lyrion server URL (e.g., http://192.168.1.100:9000)'},
            music_path_prefix_field,
        ],
        'emby': [
            {'name': 'url', 'label': 'Server URL', 'type': 'url', 'required': True,
             'description': 'Emby server URL (e.g., http://192.168.1.100:8096)'},
            {'name': 'user_id', 'label': 'User ID', 'type': 'text', 'required': True,
             'description': 'Emby user ID'},
            {'name': 'token', 'label': 'API Token', 'type': 'password', 'required': True,
             'description': 'API key from Emby settings'},
            music_path_prefix_field,
        ],
        'localfiles': [
            {'name': 'music_directory', 'label': 'Music Directory', 'type': 'path', 'required': True,
             'description': 'Path to your music library folder (e.g., /music)', 'default': '/music'},
            {'name': 'playlist_directory', 'label': 'Playlist Directory', 'type': 'path', 'required': False,
             'description': 'Where to save generated playlists', 'default': '/music/playlists'},
            {'name': 'formats', 'label': 'Audio Formats', 'type': 'text', 'required': False,
             'description': 'Comma-separated list of formats to scan', 'default': '.mp3,.flac,.ogg,.m4a,.wav'},
            {'name': 'scan_subdirs', 'label': 'Scan Subdirectories', 'type': 'boolean', 'required': False,
             'description': 'Recursively scan subdirectories', 'default': True},
        ],
    }
    return fields.get(provider_type, [])


# ##############################################################################
# MULTI-PROVIDER PLAYLIST FUNCTIONS
# ##############################################################################

def get_all_playlists_multi_provider(provider_ids=None):
    """
    Get playlists from multiple providers with deduplication.

    Args:
        provider_ids: List of provider IDs to query, or None for all enabled providers

    Returns:
        List of playlists with provider info, deduplicated by name
    """
    from app_helper import get_providers, get_provider_by_id

    all_playlists = []
    seen_names = {}  # Track playlist names to detect duplicates

    # Get providers to query
    if provider_ids is None:
        providers = get_providers(enabled_only=True)
    else:
        providers = [get_provider_by_id(pid) for pid in provider_ids if get_provider_by_id(pid)]

    for provider in providers:
        try:
            provider_type = provider['provider_type']
            playlists = _get_playlists_for_provider_type(provider_type)

            for playlist in playlists:
                playlist_name = playlist.get('Name') or playlist.get('name', '')
                playlist_id = playlist.get('Id') or playlist.get('id', '')

                # Add provider info to playlist
                playlist['provider_id'] = provider['id']
                playlist['provider_type'] = provider_type
                playlist['provider_name'] = provider.get('name', provider_type)

                # Check for duplicates by name
                if playlist_name in seen_names:
                    # Mark as duplicate
                    playlist['is_duplicate'] = True
                    playlist['duplicate_of_provider'] = seen_names[playlist_name]
                else:
                    playlist['is_duplicate'] = False
                    seen_names[playlist_name] = provider['id']

                all_playlists.append(playlist)

        except Exception as e:
            logger.warning(f"Failed to get playlists from provider {provider.get('name', 'unknown')}: {e}")
            continue

    return all_playlists


def _get_playlists_for_provider_type(provider_type):
    """Get playlists for a specific provider type using current config."""
    func = get_provider_function(provider_type, 'get_all_playlists')
    if func:
        return func()
    return []


def remap_item_ids_for_provider(item_ids: list, source_provider_id: int, target_provider_id: int) -> list:
    """
    Remap item IDs from one provider's namespace to another's using file_path as the common key.

    When creating playlists across providers, item_ids from the source provider need to be
    translated to the target provider's item_ids. This is done by:
    1. Looking up file_path for each source item_id in the score table
    2. Finding the matching item_id in the target provider by file_path

    Args:
        item_ids: List of item IDs from the source provider
        source_provider_id: ID of the source provider
        target_provider_id: ID of the target provider

    Returns:
        List of remapped item IDs for the target provider (preserving order, skipping unmatchable)
    """
    if not item_ids:
        return []

    # If same provider, no remapping needed
    if source_provider_id == target_provider_id:
        return item_ids

    from app_helper import get_db

    db = get_db()
    remapped_ids = []

    try:
        with db.cursor() as cur:
            # Get file_paths for source item_ids
            cur.execute("""
                SELECT item_id, file_path
                FROM score
                WHERE item_id = ANY(%s) AND file_path IS NOT NULL
            """, (item_ids,))
            source_id_to_path = {row[0]: row[1] for row in cur.fetchall()}

            if not source_id_to_path:
                logger.warning(f"No file_paths found for source item_ids (provider {source_provider_id})")
                return item_ids  # Fall back to original IDs

            # Get the file paths we need to look up
            paths_to_find = list(source_id_to_path.values())

            # Find matching item_ids in target provider by file_path
            # The score table has file_path but item_ids are provider-specific
            # We need to normalize paths and match
            cur.execute("""
                SELECT item_id, file_path
                FROM score
                WHERE file_path = ANY(%s)
            """, (paths_to_find,))
            path_to_ids = {}
            for row in cur.fetchall():
                path = row[1]
                if path not in path_to_ids:
                    path_to_ids[path] = []
                path_to_ids[path].append(row[0])

            # Remap in order, preserving the original playlist order
            for orig_id in item_ids:
                path = source_id_to_path.get(orig_id)
                if path and path in path_to_ids:
                    # Use the first matching ID (there might be multiple if same file analyzed multiple times)
                    target_ids = path_to_ids[path]
                    if target_ids:
                        # Prefer an ID that's different from source if available (for cross-provider)
                        for tid in target_ids:
                            if tid != orig_id:
                                remapped_ids.append(tid)
                                break
                        else:
                            # All IDs are the same, use the first one
                            remapped_ids.append(target_ids[0])
                else:
                    # No match found, keep original ID (might work if providers share IDs)
                    logger.debug(f"No cross-provider match for item_id {orig_id}, keeping original")
                    remapped_ids.append(orig_id)

            logger.info(f"Remapped {len(remapped_ids)} of {len(item_ids)} item IDs for cross-provider playlist")

    except Exception as e:
        logger.error(f"Error remapping item IDs: {e}")
        return item_ids  # Fall back to original IDs

    return remapped_ids


def create_playlist_multi_provider(playlist_name, item_ids, provider_ids=None, user_creds=None):
    """
    Create a playlist on one or more providers.

    Args:
        playlist_name: Name of the playlist to create
        item_ids: List of track IDs to add
        provider_ids: List of provider IDs to create playlist on,
                     'all' for all enabled providers,
                     or None for the primary/default provider
        user_creds: Optional user credentials for providers that support them

    Returns:
        Dict with results for each provider: {provider_id: {'success': bool, 'playlist_id': str, 'error': str}}
    """
    from app_setup import get_providers, get_provider_by_id
    from app_helper import get_primary_provider_id

    if not playlist_name:
        raise ValueError("Playlist name is required")
    if not item_ids:
        raise ValueError("Track IDs are required")

    results = {}

    # Determine which providers to use
    if provider_ids == 'all':
        providers = get_providers(enabled_only=True)
    elif provider_ids is None:
        # Use primary provider or fall back to current config
        primary_id = get_primary_provider_id()
        if primary_id:
            provider = get_provider_by_id(primary_id)
            providers = [provider] if provider else []
        else:
            # Fall back to creating on current configured provider
            try:
                created = create_instant_playlist(playlist_name, item_ids, user_creds=user_creds)
                return {'default': {'success': True, 'playlist_id': created.get('Id') if created else None}}
            except Exception as e:
                return {'default': {'success': False, 'error': str(e)}}
    else:
        # Specific provider IDs
        if isinstance(provider_ids, (list, tuple)):
            providers = [get_provider_by_id(pid) for pid in provider_ids if get_provider_by_id(pid)]
        else:
            provider = get_provider_by_id(provider_ids)
            providers = [provider] if provider else []

    # Determine source provider for ID remapping
    # If we have a primary provider, use its IDs as the source
    source_provider_id = get_primary_provider_id()
    if not source_provider_id and providers:
        source_provider_id = providers[0]['id']

    # Create playlist on each provider
    for provider in providers:
        provider_id = provider['id']
        provider_type = provider['provider_type']

        try:
            # Remap item IDs if creating on a different provider
            if source_provider_id and provider_id != source_provider_id:
                remapped_ids = remap_item_ids_for_provider(item_ids, source_provider_id, provider_id)
                logger.info(f"Cross-provider playlist: remapped {len(item_ids)} IDs for provider {provider.get('name')}")
            else:
                remapped_ids = item_ids

            if not remapped_ids:
                logger.warning(f"No valid track IDs after remapping for provider {provider.get('name')}")
                results[provider_id] = {
                    'success': False,
                    'error': 'No valid track IDs after cross-provider remapping',
                    'provider_name': provider.get('name', provider_type)
                }
                continue

            created = _create_playlist_for_provider_type(provider_type, playlist_name, remapped_ids, user_creds, provider.get('config'))

            results[provider_id] = {
                'success': True,
                'playlist_id': created.get('Id') or created.get('id') if created else None,
                'provider_name': provider.get('name', provider_type),
                'tracks_added': len(remapped_ids)
            }
        except Exception as e:
            logger.error(f"Failed to create playlist on provider {provider.get('name')}: {e}")
            results[provider_id] = {
                'success': False,
                'error': str(e),
                'provider_name': provider.get('name', provider_type)
            }

    return results


def _create_playlist_for_provider_type(provider_type, playlist_name, item_ids, user_creds=None, provider_config=None):
    """Create playlist on a specific provider type.

    When called from multi-provider mode, provider_config contains the DB-stored
    config (url, token, etc.).  We temporarily apply these values to the config
    module so that downstream provider functions (which read config.JELLYFIN_URL
    etc.) pick them up, and we build user_creds from the stored config when the
    caller hasn't supplied explicit credentials.
    """
    # Build a mapping of config module attributes to override from the provider's
    # stored DB config so that provider functions that read config.* work correctly
    # even when environment variables are empty.
    _config_overrides = {}
    if provider_config:
        if provider_type == 'jellyfin':
            _config_overrides = {
                'JELLYFIN_URL': provider_config.get('url', ''),
                'JELLYFIN_TOKEN': provider_config.get('token', ''),
                'JELLYFIN_USER_ID': provider_config.get('user_id', ''),
            }
            if not user_creds:
                user_creds = {
                    'token': provider_config.get('token', ''),
                    'user_identifier': provider_config.get('user_id', ''),
                }
        elif provider_type == 'navidrome':
            _config_overrides = {
                'NAVIDROME_URL': provider_config.get('url', ''),
                'NAVIDROME_USER': provider_config.get('user', ''),
                'NAVIDROME_PASSWORD': provider_config.get('password', ''),
            }
        elif provider_type == 'emby':
            _config_overrides = {
                'EMBY_URL': provider_config.get('url', ''),
                'EMBY_TOKEN': provider_config.get('token', ''),
                'EMBY_USER_ID': provider_config.get('user_id', ''),
            }
            if not user_creds:
                user_creds = {
                    'token': provider_config.get('token', ''),
                    'user_id': provider_config.get('user_id', ''),
                }
        elif provider_type == 'lyrion':
            _config_overrides = {
                'LYRION_URL': provider_config.get('url', ''),
            }
        elif provider_type == 'localfiles':
            _config_overrides = {
                'LOCALFILES_MUSIC_DIR': provider_config.get('music_directory', ''),
                'LOCALFILES_PLAYLIST_DIR': provider_config.get('playlist_directory', ''),
            }

    # Temporarily apply config overrides
    _saved = {}
    for attr, val in _config_overrides.items():
        if val:  # Only override if the provider config has a value
            _saved[attr] = getattr(config, attr, '')
            setattr(config, attr, val)

    try:
        if provider_type == 'jellyfin':
            return jellyfin_create_instant_playlist(playlist_name, item_ids, user_creds)
        elif provider_type == 'navidrome':
            return navidrome_create_instant_playlist(playlist_name, item_ids, user_creds)
        elif provider_type == 'lyrion':
            return lyrion_create_instant_playlist(playlist_name, item_ids)
        elif provider_type == 'emby':
            return emby_create_instant_playlist(playlist_name, item_ids, user_creds)
        elif provider_type == 'localfiles':
            return localfiles_create_instant_playlist(playlist_name, item_ids, user_creds)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    finally:
        # Restore original config values
        for attr, val in _saved.items():
            setattr(config, attr, val)


def get_enabled_providers_for_playlists():
    """
    Get list of enabled providers for use in playlist dropdowns.

    Returns:
        List of dicts with 'id', 'name', 'type' for each enabled provider
    """
    from app_setup import get_providers

    providers = get_providers(enabled_only=True)
    return [
        {
            'id': p['id'],
            'name': p.get('name') or p['provider_type'],
            'type': p['provider_type']
        }
        for p in providers
    ]
