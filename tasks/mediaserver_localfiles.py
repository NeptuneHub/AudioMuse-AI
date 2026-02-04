# tasks/mediaserver_localfiles.py
"""
Local File Media Provider for AudioMuse-AI

This provider scans local directories for audio files and extracts metadata
from embedded tags (ID3 for MP3, Vorbis comments for FLAC/OGG, etc.).

The item_id for each track is a SHA-256 hash of the normalized relative file path,
ensuring stable, predictable identifiers that won't change unless files move.
"""

import logging
import os
import hashlib
import shutil
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import List, Dict, Optional, Tuple
import json

try:
    from mutagen import File as MutagenFile
    from mutagen.mp3 import MP3
    from mutagen.flac import FLAC
    from mutagen.oggvorbis import OggVorbis
    from mutagen.mp4 import MP4
    from mutagen.id3 import ID3
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

import config

logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_FORMATS = {'.mp3', '.flac', '.ogg', '.m4a', '.mp4', '.wav', '.wma', '.aac', '.opus'}

# ##############################################################################
# CONFIGURATION
# ##############################################################################

def get_config() -> Dict:
    """Get local file provider configuration from environment or defaults."""
    return {
        'music_directory': os.environ.get('LOCALFILES_MUSIC_DIRECTORY', '/music'),
        'supported_formats': os.environ.get('LOCALFILES_FORMATS', ','.join(SUPPORTED_FORMATS)).split(','),
        'scan_subdirectories': os.environ.get('LOCALFILES_SCAN_SUBDIRS', 'true').lower() == 'true',
        'use_embedded_metadata': os.environ.get('LOCALFILES_USE_METADATA', 'true').lower() == 'true',
        'playlist_directory': os.environ.get('LOCALFILES_PLAYLIST_DIR', '/music/playlists'),
    }


# ##############################################################################
# UTILITY FUNCTIONS
# ##############################################################################

def normalize_file_path(path: str, base_path: str = "") -> str:
    """
    Normalize a file path for cross-provider matching.

    - Convert to POSIX style (forward slashes)
    - Make relative to music library root
    - Strip leading/trailing whitespace
    """
    p = Path(path)

    # Make relative if absolute and base_path provided
    if base_path and p.is_absolute():
        try:
            base = Path(base_path)
            p = p.relative_to(base)
        except ValueError:
            pass  # Not relative to base, keep as-is

    # Convert to POSIX style
    normalized = PurePosixPath(p).as_posix()

    return normalized.strip()


def file_path_hash(normalized_path: str) -> str:
    """Generate SHA-256 hash of normalized file path for use as item_id."""
    return hashlib.sha256(normalized_path.encode('utf-8')).hexdigest()


def extract_metadata(file_path: str) -> Dict:
    """
    Extract metadata from an audio file using mutagen.

    Returns a dict with keys: title, artist, album, album_artist, track_number, year, genre
    """
    metadata = {
        'title': os.path.splitext(os.path.basename(file_path))[0],  # Default to filename
        'artist': 'Unknown Artist',
        'album': 'Unknown Album',
        'album_artist': None,
        'track_number': None,
        'year': None,
        'genre': None,
        'duration': None,
    }

    if not MUTAGEN_AVAILABLE:
        logger.warning("Mutagen not available, using filename as title")
        return metadata

    try:
        audio = MutagenFile(file_path, easy=True)
        if audio is None:
            logger.debug(f"Mutagen couldn't read: {file_path}")
            return metadata

        # Extract common tags (easy=True gives us simplified tag access)
        if hasattr(audio, 'info') and audio.info:
            metadata['duration'] = getattr(audio.info, 'length', None)

        # Handle different tag formats
        if isinstance(audio.tags, dict) or hasattr(audio, 'tags'):
            tags = audio.tags if isinstance(audio.tags, dict) else dict(audio)

            # Title
            if 'title' in tags:
                val = tags['title']
                metadata['title'] = val[0] if isinstance(val, list) else str(val)

            # Artist
            if 'artist' in tags:
                val = tags['artist']
                metadata['artist'] = val[0] if isinstance(val, list) else str(val)
            elif 'performer' in tags:
                val = tags['performer']
                metadata['artist'] = val[0] if isinstance(val, list) else str(val)

            # Album
            if 'album' in tags:
                val = tags['album']
                metadata['album'] = val[0] if isinstance(val, list) else str(val)

            # Album Artist
            if 'albumartist' in tags:
                val = tags['albumartist']
                metadata['album_artist'] = val[0] if isinstance(val, list) else str(val)
            elif 'album artist' in tags:
                val = tags['album artist']
                metadata['album_artist'] = val[0] if isinstance(val, list) else str(val)

            # Track number
            if 'tracknumber' in tags:
                val = tags['tracknumber']
                track_str = val[0] if isinstance(val, list) else str(val)
                try:
                    # Handle "1/12" format
                    metadata['track_number'] = int(track_str.split('/')[0])
                except (ValueError, IndexError):
                    pass

            # Year/Date
            if 'date' in tags:
                val = tags['date']
                date_str = val[0] if isinstance(val, list) else str(val)
                try:
                    metadata['year'] = int(date_str[:4])
                except (ValueError, IndexError):
                    pass
            elif 'year' in tags:
                val = tags['year']
                year_str = val[0] if isinstance(val, list) else str(val)
                try:
                    metadata['year'] = int(year_str)
                except ValueError:
                    pass

            # Genre
            if 'genre' in tags:
                val = tags['genre']
                metadata['genre'] = val[0] if isinstance(val, list) else str(val)

    except Exception as e:
        logger.warning(f"Error extracting metadata from {file_path}: {e}")

    return metadata


def _format_song(file_path: str, base_path: str) -> Dict:
    """Format a local file into the standard song format used by AudioMuse-AI."""
    normalized_path = normalize_file_path(file_path, base_path)
    item_id = file_path_hash(normalized_path)

    metadata = extract_metadata(file_path)

    # Get file stats
    try:
        stat = os.stat(file_path)
        file_size = stat.st_size
        file_modified = datetime.fromtimestamp(stat.st_mtime)
    except OSError:
        file_size = None
        file_modified = None

    return {
        'Id': item_id,
        'Name': metadata['title'],
        'Artist': metadata['artist'],
        'AlbumArtist': metadata['album_artist'] or metadata['artist'],
        'Album': metadata['album'],
        'Path': file_path,
        'RelativePath': normalized_path,
        'TrackNumber': metadata['track_number'],
        'Year': metadata['year'],
        'Genre': metadata['genre'],
        'Duration': metadata['duration'],
        'FileSize': file_size,
        'last-modified': file_modified.isoformat() if file_modified else None,
        # For compatibility with other providers
        'ArtistId': None,  # Local files don't have artist IDs
    }


# ##############################################################################
# PUBLIC API
# ##############################################################################

def test_connection(config_override: Dict = None) -> Tuple[bool, str]:
    """Test if the local file provider can access the music directory.

    Args:
        config_override: Optional dict with configuration to test instead of default
    """
    if config_override:
        cfg = {
            'music_directory': config_override.get('music_directory', '/music'),
            'supported_formats': config_override.get('supported_formats', SUPPORTED_FORMATS),
            'scan_subdirectories': config_override.get('scan_subdirectories', True),
            'playlist_directory': config_override.get('playlist_directory', '/music/playlists'),
        }
    else:
        cfg = get_config()
    music_dir = cfg['music_directory']

    if not os.path.exists(music_dir):
        return False, f"Music directory does not exist: {music_dir}"

    if not os.path.isdir(music_dir):
        return False, f"Music path is not a directory: {music_dir}"

    if not os.access(music_dir, os.R_OK):
        return False, f"Music directory is not readable: {music_dir}"

    # Count files to verify
    try:
        audio_count = 0
        for root, _, files in os.walk(music_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in SUPPORTED_FORMATS:
                    audio_count += 1
                    if audio_count >= 10:  # Quick check, don't scan everything
                        break
            if audio_count >= 10:
                break

        if audio_count == 0:
            return False, f"No audio files found in: {music_dir}"

        return True, f"Found audio files in: {music_dir}"
    except Exception as e:
        return False, f"Error scanning music directory: {e}"


def get_all_songs() -> List[Dict]:
    """Fetch all audio files from the music directory."""
    cfg = get_config()
    music_dir = cfg['music_directory']
    supported = set(fmt.lower() if fmt.startswith('.') else f'.{fmt.lower()}'
                    for fmt in cfg['supported_formats'])
    scan_subdirs = cfg['scan_subdirectories']

    all_songs = []

    if not os.path.isdir(music_dir):
        logger.error(f"Music directory not found: {music_dir}")
        return []

    logger.info(f"Scanning local music directory: {music_dir}")

    try:
        if scan_subdirs:
            for root, _, files in os.walk(music_dir):
                for filename in files:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in supported:
                        full_path = os.path.join(root, filename)
                        try:
                            song = _format_song(full_path, music_dir)
                            all_songs.append(song)
                        except Exception as e:
                            logger.warning(f"Error processing {full_path}: {e}")
        else:
            for filename in os.listdir(music_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported:
                    full_path = os.path.join(music_dir, filename)
                    if os.path.isfile(full_path):
                        try:
                            song = _format_song(full_path, music_dir)
                            all_songs.append(song)
                        except Exception as e:
                            logger.warning(f"Error processing {full_path}: {e}")

        logger.info(f"Found {len(all_songs)} audio files in local library")

    except Exception as e:
        logger.error(f"Error scanning music directory: {e}", exc_info=True)

    return all_songs


def get_recent_albums(limit: int) -> List[Dict]:
    """
    Get recently modified albums from the local music directory.

    For local files, we group songs by album and return the most recently
    modified albums based on the newest file in each album.
    """
    cfg = get_config()
    music_dir = cfg['music_directory']

    all_songs = get_all_songs()
    if not all_songs:
        return []

    # Group by album
    albums = {}
    for song in all_songs:
        album_name = song.get('Album', 'Unknown Album')
        album_artist = song.get('AlbumArtist', 'Unknown Artist')
        album_key = f"{album_artist} - {album_name}"

        if album_key not in albums:
            albums[album_key] = {
                'Id': album_key,  # Use album name as ID
                'Name': album_name,
                'Artist': album_artist,
                'tracks': [],
                'last_modified': None
            }

        albums[album_key]['tracks'].append(song)

        # Track the most recent modification time
        mod_time = song.get('last-modified')
        if mod_time:
            if albums[album_key]['last_modified'] is None or mod_time > albums[album_key]['last_modified']:
                albums[album_key]['last_modified'] = mod_time

    # Sort by modification time (most recent first)
    sorted_albums = sorted(
        albums.values(),
        key=lambda a: a.get('last_modified') or '',
        reverse=True
    )

    # Return requested limit (0 = all)
    if limit == 0:
        return sorted_albums
    return sorted_albums[:limit]


def get_tracks_from_album(album_id: str) -> List[Dict]:
    """
    Get all tracks from an album.

    For local files, album_id is "Artist - Album Name" format.
    """
    all_songs = get_all_songs()

    # Filter songs matching this album
    tracks = []
    for song in all_songs:
        album_name = song.get('Album', 'Unknown Album')
        album_artist = song.get('AlbumArtist', 'Unknown Artist')
        song_album_key = f"{album_artist} - {album_name}"

        if song_album_key == album_id or album_name == album_id:
            tracks.append(song)

    # Sort by track number if available
    tracks.sort(key=lambda t: (t.get('TrackNumber') or 999, t.get('Name', '')))

    logger.info(f"Found {len(tracks)} tracks for album '{album_id}'")
    return tracks


def download_track(temp_dir: str, item: Dict) -> Optional[str]:
    """
    'Download' a track - for local files, we simply copy to temp directory.

    Returns the path to the temporary file.
    """
    source_path = item.get('Path')
    if not source_path or not os.path.exists(source_path):
        logger.error(f"Source file not found: {source_path}")
        return None

    try:
        # Create a unique filename in temp directory
        filename = os.path.basename(source_path)
        dest_path = os.path.join(temp_dir, filename)

        # Handle filename collisions
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            item_id = item.get('Id', '')[:8]
            dest_path = os.path.join(temp_dir, f"{name}_{item_id}{ext}")

        # Copy file to temp directory
        shutil.copy2(source_path, dest_path)
        logger.info(f"Copied '{item.get('Name', filename)}' to temp directory")

        return dest_path

    except Exception as e:
        logger.error(f"Error copying file {source_path}: {e}", exc_info=True)
        return None


def get_all_playlists() -> List[Dict]:
    """Get all M3U playlists from the playlist directory."""
    cfg = get_config()
    playlist_dir = cfg['playlist_directory']

    playlists = []

    if not os.path.isdir(playlist_dir):
        logger.info(f"Playlist directory not found: {playlist_dir}")
        return playlists

    try:
        for filename in os.listdir(playlist_dir):
            if filename.lower().endswith(('.m3u', '.m3u8')):
                name = os.path.splitext(filename)[0]
                playlists.append({
                    'Id': filename,
                    'Name': name,
                    'Path': os.path.join(playlist_dir, filename)
                })
    except Exception as e:
        logger.error(f"Error listing playlists: {e}")

    return playlists


def get_playlist_by_name(playlist_name: str) -> Optional[Dict]:
    """Find a playlist by name."""
    playlists = get_all_playlists()
    for p in playlists:
        if p['Name'] == playlist_name:
            return p
    return None


def create_playlist(base_name: str, item_ids: List[str]) -> Optional[str]:
    """
    Create an M3U playlist file.

    item_ids are the file path hashes - we need to look up the actual paths.
    """
    cfg = get_config()
    playlist_dir = cfg['playlist_directory']
    music_dir = cfg['music_directory']

    # Ensure playlist directory exists
    os.makedirs(playlist_dir, exist_ok=True)

    # Build a lookup from item_id to file path
    all_songs = get_all_songs()
    id_to_path = {song['Id']: song['Path'] for song in all_songs}

    # Resolve paths
    paths = []
    for item_id in item_ids:
        if item_id in id_to_path:
            # Use relative path for portability
            full_path = id_to_path[item_id]
            try:
                rel_path = os.path.relpath(full_path, playlist_dir)
            except ValueError:
                rel_path = full_path  # Different drive on Windows
            paths.append(rel_path)
        else:
            logger.warning(f"Track not found for item_id: {item_id}")

    if not paths:
        logger.error("No valid tracks found for playlist")
        return None

    # Write M3U file
    playlist_name = f"{base_name}_automatic.m3u"
    playlist_path = os.path.join(playlist_dir, playlist_name)

    try:
        with open(playlist_path, 'w', encoding='utf-8') as f:
            f.write("#EXTM3U\n")
            for path in paths:
                f.write(f"{path}\n")

        logger.info(f"Created playlist '{playlist_name}' with {len(paths)} tracks")
        return playlist_name

    except Exception as e:
        logger.error(f"Error creating playlist: {e}", exc_info=True)
        return None


def delete_playlist(playlist_id: str) -> bool:
    """Delete an M3U playlist file."""
    cfg = get_config()
    playlist_dir = cfg['playlist_directory']

    playlist_path = os.path.join(playlist_dir, playlist_id)

    if not os.path.exists(playlist_path):
        logger.warning(f"Playlist file not found: {playlist_path}")
        return False

    try:
        os.remove(playlist_path)
        logger.info(f"Deleted playlist: {playlist_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting playlist: {e}")
        return False


def create_instant_playlist(playlist_name: str, item_ids: List[str], user_creds=None) -> Optional[Dict]:
    """Create an instant playlist (same as regular playlist for local files)."""
    final_name = f"{playlist_name.strip()}_instant"
    result = create_playlist(final_name, item_ids)
    if result:
        return {'Id': result, 'Name': final_name}
    return None


def get_top_played_songs(limit: int, user_creds=None) -> List[Dict]:
    """Not supported for local files - no play history tracking."""
    logger.warning("get_top_played_songs is not supported for local files provider")
    return []


def get_last_played_time(item_id: str, user_creds=None):
    """Not supported for local files - no play history tracking."""
    logger.warning("get_last_played_time is not supported for local files provider")
    return None


# ##############################################################################
# PROVIDER INFO
# ##############################################################################

def get_provider_info() -> Dict:
    """Return information about this provider."""
    cfg = get_config()
    return {
        'type': 'localfiles',
        'name': 'Local Files',
        'description': 'Scan local directories for audio files',
        'supports_playlists': True,
        'supports_play_history': False,
        'supports_user_auth': False,
        'config_fields': [
            {
                'name': 'music_directory',
                'label': 'Music Directory',
                'type': 'path',
                'required': True,
                'description': 'Path to your music library folder',
                'default': '/music'
            },
            {
                'name': 'supported_formats',
                'label': 'Supported Formats',
                'type': 'text',
                'required': False,
                'description': 'Comma-separated list of audio file extensions',
                'default': ','.join(SUPPORTED_FORMATS)
            },
            {
                'name': 'scan_subdirectories',
                'label': 'Scan Subdirectories',
                'type': 'boolean',
                'required': False,
                'description': 'Include files in subdirectories',
                'default': True
            },
            {
                'name': 'playlist_directory',
                'label': 'Playlist Directory',
                'type': 'path',
                'required': False,
                'description': 'Where to save generated M3U playlists',
                'default': '/music/playlists'
            }
        ]
    }
