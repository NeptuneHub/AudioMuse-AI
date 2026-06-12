# tasks/mediaserver/__init__.py

import logging
import os
from importlib import import_module

import config

logger = logging.getLogger(__name__)

_PROVIDER_NAMES = ('jellyfin', 'navidrome', 'lyrion', 'emby')
_warned_unsupported = set()

_PLAYLIST_NAME_REQUIRED = "Playlist name is required."
_TRACK_IDS_REQUIRED = "Track IDs are required."


def _provider(provider_type=None):
    """Return the backend module for the given (or configured) provider type.

    Provider modules are imported lazily on first use so that importing
    ``tasks.mediaserver`` does not load the four inactive backends or
    initialize their HTTP sessions. Returns None for unsupported types,
    matching the old dispatcher fall-through behavior.
    """
    name = provider_type or config.MEDIASERVER_TYPE
    if name not in _PROVIDER_NAMES:
        if name not in _warned_unsupported:
            _warned_unsupported.add(name)
            logger.warning(
                "Unsupported MEDIASERVER_TYPE %r (supported: %s); media-server operations are no-ops.",
                name, ', '.join(_PROVIDER_NAMES))
        return None
    return import_module('.' + name, __name__)


# ##############################################################################
# PUBLIC API (Dispatcher functions)
# ##############################################################################

def resolve_emby_jellyfin_user(identifier, token):
    """Public dispatcher for resolving a Jellyfin or Emby user identifier."""
    if config.MEDIASERVER_TYPE in ('jellyfin', 'emby'):
        return _provider().resolve_user(identifier, token)
    return []

def _delete_matching_playlists(playlists_to_check, delete_function, suffix):
    """Deletes every playlist whose name ends with the suffix; keeps going if one deletion fails."""
    deleted_count = 0
    for p in playlists_to_check:
        # Navidrome uses 'id', others use 'Id'. Check for both.
        playlist_id = p.get('Id') or p.get('id')
        try:
            if p.get('Name', '').endswith(suffix) and delete_function(playlist_id):
                deleted_count += 1
        except Exception:
            logger.exception(f"Failed to delete playlist {playlist_id}; continuing with the remaining playlists.")
    return deleted_count

def delete_playlists_by_suffix(suffix):
    """Deletes all playlists whose name ends with the given suffix using admin credentials."""
    logger.info(f"Starting deletion of all '{suffix}' playlists.")
    deleted_count = 0

    provider = _provider()
    if provider is not None:
        deleted_count = _delete_matching_playlists(provider.get_all_playlists(), provider.delete_playlist, suffix)

    logger.info(f"Finished deletion. Deleted {deleted_count} playlists.")

def delete_automatic_playlists():
    """Deletes all playlists ending with '_automatic' using admin credentials."""
    delete_playlists_by_suffix('_automatic')

def get_recent_albums(limit):
    """Fetches recently added albums using admin credentials."""
    provider = _provider()
    if provider is None:
        return []
    return provider.get_recent_albums(limit)

def get_tracks_from_album(album_id, user_creds=None, provider_type=None):
    """Fetches tracks for an album, optionally using explicit creds."""
    provider = _provider(provider_type)
    if provider is None:
        return []
    return provider.get_tracks_from_album(album_id, user_creds=user_creds)

def download_track(temp_dir, item):
    """Downloads a track using admin credentials. Detects format from file if .tmp extension is used."""
    provider = _provider()
    downloaded_path = provider.download_track(temp_dir, item) if provider is not None else None

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

def get_all_songs(user_creds=None, provider_type=None, apply_filter=True):
    """Fetches all songs using admin credentials or explicit creds.

    ``apply_filter`` is forwarded to providers that honor
    ``config.MUSIC_LIBRARIES`` (currently Navidrome). Migration probes pass
    ``apply_filter=False`` so the source provider's library filter does not
    falsely exclude tracks from the target server during dry-run.
    """
    provider_type = provider_type or config.MEDIASERVER_TYPE
    provider = _provider(provider_type)
    if provider is None:
        return []
    if provider_type == 'navidrome':
        return provider.get_all_songs(user_creds=user_creds, apply_filter=apply_filter)
    return provider.get_all_songs(user_creds=user_creds)

def list_libraries(user_creds=None, provider_type=None):
    """List all music libraries/folders a provider exposes.

    Returns {'libraries': [{'id': str, 'name': str}, ...], 'unsupported': bool}.
    The setup wizard and migration assistant use this to render a checkbox list
    after a successful test-connection. Uses admin credentials when
    ``user_creds`` is None, or the supplied creds when probing a target.
    """
    provider = _provider(provider_type)
    if provider is None:
        return {'libraries': [], 'unsupported': True}
    return {'libraries': provider.list_libraries(user_creds=user_creds), 'unsupported': False}

def search_albums(query, user_creds=None, provider_type=None):
    """Searches for albums using admin credentials or explicit creds."""
    provider = _provider(provider_type)
    if provider is None:
        return []
    return provider.search_albums(query, user_creds=user_creds)

def test_connection(user_creds=None, provider_type=None):
    """Tests provider connection using admin credentials or explicit creds."""
    provider_type = provider_type or config.MEDIASERVER_TYPE
    provider = _provider(provider_type)
    if provider is None:
        return {'ok': False, 'error': f"Provider '{provider_type}' not supported", 'sample_count': 0, 'path_format': 'none', 'warnings': []}
    return provider.test_connection(user_creds=user_creds)

def get_playlist_by_name(playlist_name):
    """Finds a playlist by name using admin credentials."""
    if not playlist_name: raise ValueError(_PLAYLIST_NAME_REQUIRED)
    provider = _provider()
    if provider is None:
        return None
    return provider.get_playlist_by_name(playlist_name)

def create_playlist(base_name, item_ids):
    """Creates a playlist using admin credentials."""
    if not base_name: raise ValueError(_PLAYLIST_NAME_REQUIRED)
    if not item_ids: raise ValueError(_TRACK_IDS_REQUIRED)
    provider = _provider()
    if provider is not None:
        provider.create_playlist(base_name, item_ids)

def create_instant_playlist(playlist_name, item_ids, user_creds=None):
    """Creates an instant playlist. Uses user_creds if provided, otherwise admin."""
    if not playlist_name: raise ValueError(_PLAYLIST_NAME_REQUIRED)
    if not item_ids: raise ValueError(_TRACK_IDS_REQUIRED)

    provider = _provider()
    if provider is None:
        return None
    if config.MEDIASERVER_TYPE == 'lyrion':
        return provider.create_instant_playlist(playlist_name, item_ids)
    return provider.create_instant_playlist(playlist_name, item_ids, user_creds)


def create_or_replace_playlist(playlist_name, item_ids, user_creds=None):
    """Cron-only upsert: create the playlist if missing, or replace its contents in place.

    Used by the scheduled sonic_fingerprint task so the same server-side playlist (and ID,
    where the backend allows) gets reused across runs. Raises NotImplementedError for any
    unsupported backend — the cron handler catches that and falls back to legacy
    date-suffixed playlist creation.
    """
    if not playlist_name:
        raise ValueError(_PLAYLIST_NAME_REQUIRED)
    if not item_ids:
        raise ValueError(_TRACK_IDS_REQUIRED)

    provider = _provider()
    if provider is None:
        raise NotImplementedError(
            f"create_or_replace_playlist not supported for MEDIASERVER_TYPE={config.MEDIASERVER_TYPE!r}"
        )
    return provider.create_or_replace_playlist(playlist_name, item_ids, user_creds)

def get_top_played_songs(limit, user_creds=None):
    """Fetches top played songs. Uses user_creds if provided, otherwise admin."""
    provider = _provider()
    if provider is None:
        return []
    if config.MEDIASERVER_TYPE == 'lyrion':
        return provider.get_top_played_songs(limit)
    return provider.get_top_played_songs(limit, user_creds)

def get_last_played_time(item_id, user_creds=None):
    """Fetches last played time for a track. Uses user_creds if provided, otherwise admin."""
    provider = _provider()
    if provider is None:
        return None
    if config.MEDIASERVER_TYPE == 'lyrion':
        return provider.get_last_played_time(item_id)
    return provider.get_last_played_time(item_id, user_creds)

def get_lyrics(track_id: str, timeout: float = 2.5):
    """Fetch lyrics embedded in the media server for a given track ID.

    Supported servers: Jellyfin, Emby, Navidrome, Lyrion.
    Returns plain text or None.
    """
    provider = _provider()
    if provider is None:
        return None
    return provider.get_lyrics(track_id, timeout=timeout)
