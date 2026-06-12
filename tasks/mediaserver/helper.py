"""Shared media server helper utilities."""

import logging
import os
import re

logger = logging.getLogger(__name__)


def select_best_artist(item, title="Unknown"):
    """
    Selects the best artist field from a Jellyfin/Emby item, prioritizing track
    artists over album artists. This helps avoid "Various Artists" issues in
    compilation albums.
    Returns tuple: (artist_name, artist_id)
    """
    # Priority: Artists array (track artists) > AlbumArtist > fallback
    # Jellyfin/Emby provides ArtistItems array with Id and Name
    if item.get('ArtistItems') and len(item['ArtistItems']) > 0:
        track_artist = item['ArtistItems'][0].get('Name', 'Unknown Artist')
        artist_id = item['ArtistItems'][0].get('Id')
    elif item.get('Artists') and len(item['Artists']) > 0:
        track_artist = item['Artists'][0]  # Take first artist if multiple
        artist_id = None
    elif item.get('AlbumArtist'):
        track_artist = item['AlbumArtist']
        artist_id = None
    else:
        track_artist = 'Unknown Artist'
        artist_id = None

    return track_artist, artist_id


def detect_download_extension(item):
    """Derive a file extension for a Jellyfin/Emby download.

    Prefers the item's Container field (most reliable), falls back to the
    Path extension, and defaults to '.tmp' so the dispatcher's magic-number
    sniffing can rename the file after download.
    """
    file_extension = '.tmp'
    try:
        container = item.get('Container')
        if container and isinstance(container, str) and container.strip():
            # Ensure container value is safe (no path separators, etc.)
            safe_container = container.strip().replace('/', '').replace('\\', '')
            if safe_container:
                file_extension = f".{safe_container}"
                logger.debug(f"Using Container field for format: {file_extension}")
        elif item.get('Path'):
            file_extension = os.path.splitext(item['Path'])[1] or '.tmp'
    except Exception as e:
        logger.debug(f"Error getting format from Container/Path, using .tmp: {e}")
    return file_extension


def detect_path_format(tracks):
    """Classify track path samples as absolute, relative, none, or mixed."""
    def _is_absolute_path(path):
        if not path:
            return False
        path_str = str(path)
        lower = path_str.lower()
        return (
            path_str.startswith('/')
            or path_str.startswith('\\')
            or lower.startswith('file://')
            or re.match(r'^[A-Za-z]:[\\/]', path_str)
        )

    paths = []
    for track in tracks or []:
        if not isinstance(track, dict):
            continue
        # Support lowercase/uppercase path keys and legacy URL fields.
        path = (
            track.get('path')
            or track.get('Path')
            or track.get('url')
            or track.get('Url')
        )
        if path:
            paths.append(path)

    if not paths:
        return 'none'

    ratio = sum(1 for p in paths if _is_absolute_path(p)) / len(paths)
    if ratio >= 0.8:
        return 'absolute'
    if ratio <= 0.2:
        return 'relative'
    return 'mixed'
