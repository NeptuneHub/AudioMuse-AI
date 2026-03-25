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

def get_config(overrides: Dict = None) -> Dict:
    """Get local file provider configuration from config module, then environment, then defaults.

    The config module attributes are checked first so that _get_all_songs_with_config()
    in app_setup.py can temporarily patch them for multi-provider sync.
    """
    cfg = {
        'music_directory': getattr(config, 'LOCALFILES_MUSIC_DIRECTORY', None)
                           or os.environ.get('LOCALFILES_MUSIC_DIRECTORY', '/music'),
        'supported_formats': getattr(config, 'LOCALFILES_FORMATS', None)
                             or os.environ.get('LOCALFILES_FORMATS', ','.join(SUPPORTED_FORMATS)),
        'scan_subdirectories': getattr(config, 'LOCALFILES_SCAN_SUBDIRS', None),
        'use_embedded_metadata': os.environ.get('LOCALFILES_USE_METADATA', 'true').lower() == 'true',
        'playlist_directory': getattr(config, 'LOCALFILES_PLAYLIST_DIR', None)
                              or os.environ.get('LOCALFILES_PLAYLIST_DIR', '/music/playlists'),
    }
    # Handle supported_formats: ensure it's a list
    if isinstance(cfg['supported_formats'], str):
        cfg['supported_formats'] = cfg['supported_formats'].split(',')
    # Handle scan_subdirectories: could be bool (from config) or None (fallback to env)
    if cfg['scan_subdirectories'] is None:
        cfg['scan_subdirectories'] = os.environ.get('LOCALFILES_SCAN_SUBDIRS', 'true').lower() == 'true'
    if overrides:
        cfg.update(overrides)
    return cfg


# ##############################################################################
# DB CACHE HELPERS
# ##############################################################################

def _get_songs_from_db() -> List[Dict]:
    """
    Query the score table for songs with file_path set (previously analyzed).
    Returns a list of dicts matching the format returned by get_all_songs().
    Falls back to empty list if DB is unavailable.

    Uses provider_track join to return the localfiles provider's native item_id
    when available, with fallback to score.track_id for legacy data.
    """
    try:
        from app_helper import get_db
        db = get_db()
        if not db:
            return []
        with db.cursor() as cur:
            # Try provider-filtered query first (returns localfiles provider item_ids)
            # Includes last_synced timestamp for mtime-based cache invalidation
            cur.execute("""
                SELECT pt.item_id, s.title, s.author, s.album, s.album_artist,
                       s.file_path, s.year, s.rating, pt.last_synced, t.normalized_path
                FROM score s
                JOIN provider_track pt ON pt.track_id = s.track_id
                JOIN provider p ON p.id = pt.provider_id AND p.provider_type = 'localfiles'
                JOIN track t ON t.id = pt.track_id
                WHERE s.file_path IS NOT NULL
                LIMIT 100000
            """)
            rows = cur.fetchall()
            if not rows:
                # Fallback for legacy data without provider_track mappings
                cur.execute("""
                    SELECT track_id, title, author, album, album_artist, file_path,
                           year, rating, NULL
                    FROM score
                    WHERE file_path IS NOT NULL
                    LIMIT 100000
                """)
                rows = cur.fetchall()
            songs = []
            for row in rows:
                songs.append({
                    'Id': row[0],
                    'Name': row[1] or 'Unknown',
                    'AlbumArtist': row[4] or row[2] or 'Unknown Artist',
                    'Album': row[3] or 'Unknown Album',
                    'OriginalAlbumArtist': row[4],
                    'Path': row[5],
                    'FilePath': row[5],
                    'Year': row[6],
                    'Rating': row[7],
                    '_last_synced': row[8],  # For mtime comparison in delta scan
                    '_normalized_path': row[9],  # For cross-provider path matching
                })
            return songs
    except Exception as e:
        logger.debug(f"DB cache lookup failed (expected during first scan): {e}")
        return []


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
    # Replace backslashes first so Linux doesn't treat them as literal chars
    path = path.replace('\\', '/')
    p = PurePosixPath(path)

    # Make relative if absolute and base_path provided
    if base_path:
        base_path = base_path.replace('\\', '/')
        base = PurePosixPath(base_path)
        if p.is_absolute():
            try:
                p = p.relative_to(base)
            except ValueError:
                pass  # Not relative to base, keep as-is

    # Convert to POSIX style
    normalized = p.as_posix()

    return normalized.strip()


def file_path_hash(normalized_path: str) -> str:
    """Generate SHA-256 hash of normalized file path for use as item_id."""
    return hashlib.sha256(normalized_path.encode('utf-8')).hexdigest()


def extract_metadata(file_path: str) -> Dict:
    """
    Extract metadata from an audio file using mutagen.

    Returns a dict with keys: title, artist, album, album_artist, track_number, year, genre, rating
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
        'rating': None,
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

        # Extract rating from non-easy tags (requires re-opening without easy=True)
        metadata['rating'] = _extract_rating(file_path)

    except Exception as e:
        logger.warning(f"Error extracting metadata from {file_path}: {e}")

    return metadata


def _extract_rating(file_path: str) -> Optional[int]:
    """
    Extract rating from audio file tags.

    Supports:
    - ID3 POPM (Popularimeter) frame for MP3 (0-255 scale, converted to 0-5)
    - ID3 TXXX:RATING frame for MP3
    - FLAC/OGG Vorbis RATING comment
    - MP4/M4A rating

    Returns rating on 0-5 scale, or None if not found.
    """
    if not MUTAGEN_AVAILABLE:
        return None

    try:
        audio = MutagenFile(file_path, easy=False)
        if audio is None:
            return None

        ext = os.path.splitext(file_path)[1].lower()

        # MP3 files - check ID3 tags
        if ext == '.mp3' and audio.tags:
            # Try POPM (Popularimeter) frame first
            for key in audio.tags.keys():
                if key.startswith('POPM'):
                    popm = audio.tags[key]
                    if hasattr(popm, 'rating'):
                        # POPM rating is 0-255, convert to 0-5 scale
                        # Common mappings: 0=0, 1=1, 64=2, 128=3, 196=4, 255=5
                        popm_rating = popm.rating
                        if popm_rating == 0:
                            return 0
                        elif popm_rating <= 31:
                            return 1
                        elif popm_rating <= 95:
                            return 2
                        elif popm_rating <= 159:
                            return 3
                        elif popm_rating <= 223:
                            return 4
                        else:
                            return 5

            # Try TXXX:RATING frame
            for key in audio.tags.keys():
                if key.startswith('TXXX') and 'RATING' in str(key).upper():
                    try:
                        rating_str = str(audio.tags[key].text[0])
                        rating = int(float(rating_str))
                        # Normalize to 0-5 if needed
                        if rating > 5:
                            rating = min(5, rating // 20)  # Assume 0-100 scale
                        return max(0, min(5, rating))
                    except (ValueError, IndexError, AttributeError):
                        pass

        # FLAC files - check Vorbis comments
        elif ext == '.flac' and hasattr(audio, 'tags') and audio.tags:
            for key in ['RATING', 'FMPS_RATING', 'rating']:
                if key in audio.tags:
                    try:
                        rating_str = audio.tags[key][0]
                        rating = float(rating_str)
                        # FMPS uses 0-1 scale, convert to 0-5
                        if rating <= 1:
                            return round(rating * 5)
                        # Direct 0-5 scale
                        return max(0, min(5, int(rating)))
                    except (ValueError, IndexError):
                        pass

        # OGG files - similar to FLAC
        elif ext in ('.ogg', '.opus') and hasattr(audio, 'tags') and audio.tags:
            for key in ['RATING', 'FMPS_RATING', 'rating']:
                if key in audio.tags:
                    try:
                        rating_str = audio.tags[key][0]
                        rating = float(rating_str)
                        if rating <= 1:
                            return round(rating * 5)
                        return max(0, min(5, int(rating)))
                    except (ValueError, IndexError):
                        pass

        # M4A/MP4 files
        elif ext in ('.m4a', '.mp4') and hasattr(audio, 'tags') and audio.tags:
            # iTunes rating stored in 'rtng' atom (0-100) or as custom tag
            if 'rtng' in audio.tags:
                try:
                    rating = audio.tags['rtng'][0]
                    # iTunes uses 0-100 scale
                    return max(0, min(5, rating // 20))
                except (IndexError, TypeError):
                    pass
            # Check for custom rating tags
            for key in audio.tags.keys():
                if 'rating' in str(key).lower():
                    try:
                        rating = int(audio.tags[key][0])
                        if rating > 5:
                            rating = min(5, rating // 20)
                        return max(0, min(5, rating))
                    except (ValueError, IndexError, TypeError):
                        pass

    except Exception as e:
        logger.debug(f"Error extracting rating from {file_path}: {e}")

    return None


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
        'FilePath': file_path,  # Alias for analysis.py compatibility
        'RelativePath': normalized_path,
        'TrackNumber': metadata['track_number'],
        'Year': metadata['year'],
        'Genre': metadata['genre'],
        'Duration': metadata['duration'],
        'Rating': metadata.get('rating'),
        'OriginalAlbumArtist': metadata['album_artist'],  # For analysis.py album_artist update
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
    """Fetch all audio files from the music directory.

    Uses a delta scan strategy: first does a quick os.walk to collect file paths,
    then loads cached songs from the DB, and only extracts metadata (mutagen) for
    files not already in the cache. This makes subsequent scans fast even over NAS/SMB.
    """
    cfg = get_config()
    music_dir = cfg['music_directory']
    supported = set(fmt.lower() if fmt.startswith('.') else f'.{fmt.lower()}'
                    for fmt in cfg['supported_formats'])
    scan_subdirs = cfg['scan_subdirectories']

    if not os.path.isdir(music_dir):
        logger.error(f"Music directory not found: {music_dir}")
        return []

    # Phase 1: Quick path scan (no metadata extraction)
    logger.info(f"Scanning local music directory: {music_dir}")
    all_paths = []
    try:
        if scan_subdirs:
            for root, _, files in os.walk(music_dir):
                for filename in files:
                    if os.path.splitext(filename)[1].lower() in supported:
                        all_paths.append(os.path.join(root, filename))
        else:
            for filename in os.listdir(music_dir):
                if os.path.splitext(filename)[1].lower() in supported:
                    full_path = os.path.join(music_dir, filename)
                    if os.path.isfile(full_path):
                        all_paths.append(full_path)
    except Exception as e:
        logger.error(f"Error scanning music directory: {e}", exc_info=True)
        return []

    logger.info(f"Found {len(all_paths)} audio files on filesystem")

    # Phase 2: Load DB cache and determine delta (new files + modified files)
    cached_songs = _get_songs_from_db()
    # Index cache by both the original path AND the normalized path so we can
    # match even when the DB stores a different provider's path (e.g., Jellyfin's
    # /data/Music I/... vs LocalFiles' /music/...).  Normalized paths are
    # lowercase and prefix-stripped, so we normalize filesystem paths the same way.
    cached_by_path = {}
    cached_by_norm = {}
    for s in cached_songs:
        path = s.get('Path') or s.get('FilePath')
        if path:
            cached_by_path[path] = s
        norm = s.get('_normalized_path')
        if norm:
            cached_by_norm[norm] = s

    def _normalize_fs_path(fs_path):
        """Normalize a filesystem path the same way as track.normalized_path."""
        p = fs_path.replace('\\', '/')
        # Strip common mount prefixes (must match normalize_path_deterministic)
        for prefix in ('/media/music/', '/media/', '/mnt/media/music/', '/mnt/media/',
                       '/mnt/music/', '/mnt/data/music/', '/mnt/data/', '/mnt/',
                       '/data/music/', '/data/', '/music/', '/share/music/', '/share/',
                       '/volume1/music/', '/volume1/', '/srv/music/', '/srv/',
                       '/home/music/', '/storage/music/', '/opt/music/', '/nas/music/',
                       '/library/music/'):
            if p.lower().startswith(prefix):
                p = p[len(prefix):]
                break
        return p.lower().lstrip('/')

    fs_paths = set(all_paths)
    rescan_paths = []  # New or modified files that need metadata extraction
    active_cached = []  # Cached songs still valid

    for path in all_paths:
        cached = cached_by_path.get(path) or cached_by_norm.get(_normalize_fs_path(path))
        if cached is None:
            # New file — not in cache
            rescan_paths.append(path)
        else:
            # Existing file — check if modified since last sync
            last_synced = cached.get('_last_synced')
            if last_synced:
                try:
                    file_mtime = os.stat(path).st_mtime
                    synced_ts = last_synced.timestamp() if hasattr(last_synced, 'timestamp') else float(last_synced)
                    if file_mtime > synced_ts:
                        rescan_paths.append(path)
                        continue
                except OSError:
                    pass  # Can't stat — use cached version
            # Override Path/FilePath with the actual filesystem path so that
            # downstream code (e.g., cross-provider hash linking) uses the
            # LocalFiles path, not the primary provider's path from score.file_path.
            cached['Path'] = path
            cached['FilePath'] = path
            active_cached.append(cached)

    if not rescan_paths:
        logger.info(f"Delta scan: 0 new/modified files, using {len(active_cached)} cached songs")
        return active_cached

    new_count = sum(1 for p in rescan_paths if p not in cached_by_path and _normalize_fs_path(p) not in cached_by_norm)
    modified_count = len(rescan_paths) - new_count
    logger.info(f"Delta scan: {new_count} new + {modified_count} modified files to process, {len(active_cached)} cached")

    # Phase 3: Extract metadata only for new/modified files
    new_songs = []
    for full_path in rescan_paths:
        try:
            song = _format_song(full_path, music_dir)
            new_songs.append(song)
        except Exception as e:
            logger.warning(f"Error processing {full_path}: {e}")

    all_songs = active_cached + new_songs
    logger.info(f"Total: {len(all_songs)} audio files ({len(active_cached)} cached + {len(new_songs)} new/modified)")
    return all_songs


def get_recent_albums(limit: int) -> List[Dict]:
    """
    Get recently modified albums from the local music directory.

    For local files, we group songs by album and return the most recently
    modified albums based on the newest file in each album.
    Uses DB cache when available to avoid rescanning the filesystem.
    """
    cfg = get_config()
    music_dir = cfg['music_directory']

    # Use DB cache first to avoid expensive filesystem scan
    all_songs = _get_songs_from_db()
    if not all_songs:
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
    Uses DB cache when available to avoid rescanning the filesystem.
    """
    all_songs = _get_songs_from_db()
    if not all_songs:
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


def create_playlist(base_name: str, item_ids: List[str], config_override: Dict = None) -> Optional[str]:
    """
    Create an M3U playlist file.

    item_ids are the file path hashes - we need to look up the actual paths.
    """
    cfg = get_config(overrides=config_override)
    playlist_dir = cfg['playlist_directory']
    music_dir = cfg['music_directory']

    # Ensure playlist directory exists
    os.makedirs(playlist_dir, exist_ok=True)

    # Build a lookup from item_id to file path.
    # Use get_all_songs() which returns actual filesystem paths (case-correct)
    # via the delta scan. _get_songs_from_db() may return score.file_path from
    # the primary provider (e.g. Jellyfin's /data/Music I/...) which doesn't
    # exist on the LocalFiles mount.
    all_songs = get_all_songs()
    id_to_path = {song['Id']: song.get('Path') or song.get('FilePath', '') for song in all_songs}

    # Resolve paths
    paths = []
    for item_id in item_ids:
        if item_id in id_to_path:
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

    # Sanitize base_name to prevent path traversal
    safe_base_name = os.path.basename(base_name)
    if not safe_base_name:
        logger.error("Invalid playlist base_name")
        return None

    # Write M3U file
    playlist_name = f"{safe_base_name}.m3u"
    playlist_path = os.path.join(playlist_dir, playlist_name)

    # Verify resolved path stays within playlist_dir (prevent symlink traversal)
    real_path = os.path.realpath(playlist_path)
    real_dir = os.path.realpath(playlist_dir)
    if not real_path.startswith(real_dir + os.sep) and real_path != real_dir:
        logger.warning(f"Path traversal attempt blocked in create_playlist: {base_name}")
        return None

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

    # Sanitize playlist_id: strip path separators to prevent path traversal
    safe_id = os.path.basename(playlist_id)
    if not safe_id or safe_id != playlist_id:
        logger.warning(f"Rejected playlist_id with path components: {playlist_id}")
        return False

    playlist_path = os.path.join(playlist_dir, safe_id)

    # Verify resolved path stays within playlist_dir
    real_path = os.path.realpath(playlist_path)
    real_dir = os.path.realpath(playlist_dir)
    if not real_path.startswith(real_dir + os.sep) and real_path != real_dir:
        logger.warning(f"Path traversal attempt blocked: {playlist_id}")
        return False

    if not os.path.exists(playlist_path):
        logger.warning(f"Playlist file not found: {playlist_path}")
        return False

    try:
        os.remove(playlist_path)
        logger.info(f"Deleted playlist: {safe_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting playlist: {e}")
        return False


def create_instant_playlist(playlist_name: str, item_ids: List[str], user_creds=None, server_config=None) -> Optional[Dict]:
    """Create an instant playlist (same as regular playlist for local files)."""
    sc = server_config or {}
    config_override = {}
    if sc.get('music_directory'):
        config_override['music_directory'] = sc['music_directory']
    if sc.get('playlist_directory'):
        config_override['playlist_directory'] = sc['playlist_directory']
    final_name = f"{playlist_name.strip()}_instant"
    result = create_playlist(final_name, item_ids, config_override=config_override or None)
    if result:
        return {'Id': result, 'Name': final_name}
    return None


def get_top_played_songs(limit: int, user_creds=None, server_config=None) -> List[Dict]:
    """Not supported for local files - no play history tracking."""
    logger.warning("get_top_played_songs is not supported for local files provider")
    return []


def get_last_played_time(item_id: str, user_creds=None, server_config=None):
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
            },
            {
                'name': 'music_path_prefix',
                'label': 'Music Path Prefix',
                'type': 'text',
                'required': False,
                'description': 'Folder prefix to strip for cross-provider matching. Leave empty if paths start directly with artist folders.'
            }
        ]
    }
