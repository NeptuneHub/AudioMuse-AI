# tasks/mediaserver_mpd.py

import logging
import os
import shutil
from datetime import datetime
import config
import requests # <-- ADDED: Needed for HTTP downloads
import random   # <-- ADDED: For shuffling albums

# Add the MPD client library dependency
# NOTE: This implementation requires the 'python-mpd2' library.
# Install it with: pip install python-mpd2
try:
    import mpd
except ImportError:
    # Handle the case where the library isn't installed.
    # You might want to log a more prominent warning or exit if MPD is the configured server type.
    pass

logger = logging.getLogger(__name__)

# ##############################################################################
# MPD (MUSIC PLAYER DAEMON) IMPLEMENTATION
# ##############################################################################

def _connect():
    """Establishes a connection to the MPD server."""
    # Set use_unicode=True to ensure all communication with the server,
    # including file paths, is handled as UTF-8.
    client = mpd.MPDClient(use_unicode=True)
    client.timeout = 60
    client.idletimeout = 30


    try:
        logger.info(f"Calling MPD connect('{config.MPD_HOST}', {config.MPD_PORT}, timeout=None)")
        client.connect(config.MPD_HOST, config.MPD_PORT)
        logger.info(f"Successfully connected to MPD server. Status: {client.status()}")

        if config.MPD_PASSWORD:
            logger.info("Authenticating with MPD password")
            client.password(config.MPD_PASSWORD)

        return client
    except Exception as e:
        logger.error(f"Failed to connect or configure MPD server: {e}", exc_info=True)
        _disconnect_safely(client)
        return None

def _format_song(song_dict):
    """Formats an MPD song dictionary to the standard format used in this script."""
    # The 'Id' will be the file path, which is unique.
    return {
        'Id': song_dict.get('file'),
        'Name': song_dict.get('title', os.path.basename(song_dict.get('file', ''))),
        'AlbumArtist': song_dict.get('albumartist'),
        'Artist': song_dict.get('artist'),
        'Album': song_dict.get('album'),
        'Path': song_dict.get('file'),
        'last-modified': song_dict.get('last-modified')
    }

def _disconnect_safely(client):
    """Safely closes and disconnects the MPD client."""
    if not client:
        return
    try:
        client.close()
        client.disconnect()
    except (mpd.ConnectionError, IOError, BrokenPipeError):
        pass # Ignore errors on disconnect, as the connection might already be lost.

def get_recent_albums(limit):
    """
    [EFFICIENT VERSION] Fetches a random selection of albums from MPD.
    
    NOTE: Finding the chronologically "most recent" albums requires a full
    scan of every song in the library, which is extremely slow on large collections.
    This function provides a fast and practical alternative by returning a random
    sample, which is much better for discovering content to analyze.
    """
    client = _connect()
    if not client:
        return []

    albums = []
    try:
        logger.info("Fetching a random selection of albums for analysis...")
        
        # This is a very fast command that gets all unique album names.
        album_names = client.list('album')
        
        # If the user wants all albums, don't shuffle, just format.
        fetch_all = (limit == 0)
        if fetch_all:
            logger.info(f"Formatting all {len(album_names)} albums.")
            albums_to_process = album_names
        else:
            logger.info(f"Found {len(album_names)} total albums. Shuffling to select {limit} random ones.")
            random.shuffle(album_names)
            albums_to_process = album_names[:limit]

        # Format the selected albums into the expected dictionary structure.
        # We use a placeholder date as the true 'last_modified' is unknown without a full scan.
        now = datetime.now()
        albums = [{'Id': name, 'Name': name, 'last_modified': now} for name in albums_to_process]
        
        logger.info(f"Selected {len(albums)} albums to process.")

    except Exception as e:
        logger.error(f"MPD get_recent_albums failed: {e}", exc_info=True)
    finally:
        _disconnect_safely(client)

    return albums

def get_tracks_from_album(album_id):
    """Fetches all audio tracks for a given album name from MPD. album_id is the album name."""
    client = _connect()
    if not client:
        return []
        
    tracks = []
    try:
        # Use client.find("album", album_id) to search by metadata tag.
        songs = client.find("album", album_id)
        tracks = [_format_song(s) for s in songs if 'file' in s]
        logger.info(f"Found {len(tracks)} tracks for album '{album_id}'.")
    except Exception as e:
        logger.error(f"MPD get_tracks_from_album failed for album '{album_id}': {e}", exc_info=True)
    finally:
        _disconnect_safely(client)
    return tracks

def download_track(temp_dir, item):
    """
    Downloads a track from a remote MPD server using its built-in HTTP streamer.
    This function assumes MPD's HTTP stream is available on port 8000.
    """
    try:
        track_path = item.get('Path')
        if not track_path:
            logger.error("MPD item has no 'Path' attribute to download.")
            return None

        # Construct the HTTP stream URL for the track.
        # This uses the same MPD_HOST but assumes port 8000 for the HTTP stream.
        # The path needs to be URL-encoded to handle special characters.
        from urllib.parse import quote
        encoded_path = quote(track_path)
        download_url = f"http://{config.MPD_HOST}:8000/{encoded_path}"

        logger.info(f"Downloading track from URL: {download_url}")

        # Use requests library to download the file in a streaming fashion
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()  # This will raise an exception for bad status codes (4xx or 5xx)

            # Create a safe local filename
            file_extension = os.path.splitext(track_path)[1]
            track_id = os.path.basename(track_path).replace(file_extension, '')
            local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")

            # Write the content to the local file in chunks
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded '{item.get('Name', 'Unknown')}' to '{local_filename}'")
            return local_filename

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP download failed for track {item.get('Name', 'Unknown')}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during download of {item.get('Name', 'Unknown')}: {e}", exc_info=True)
        return None

def get_all_songs():
    """Fetches all songs from MPD using a robust, song-by-song method."""
    client = _connect()
    if not client:
        return []
        
    all_formatted_songs = []
    try:
        logger.info("Fetching all songs from MPD database (robust method)...")
        
        all_files = client.list('file')
        logger.info(f"Found {len(all_files)} files to process.")

        for i, file_path_dict in enumerate(all_files):
            try:
                # FIX: The list command returns a list of dicts. We need the value from the 'file' key.
                file_path_str = file_path_dict.get('file')
                if not file_path_str:
                    continue
                
                song_info_list = client.listallinfo(file_path_str)
                if song_info_list and 'file' in song_info_list[0]:
                    all_formatted_songs.append(_format_song(song_info_list[0]))
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"Formatted {i+1}/{len(all_files)} songs...")
            except Exception:
                # Ignore errors for individual files (e.g., playlist files, etc.)
                pass

        logger.info(f"Successfully formatted {len(all_formatted_songs)} songs from MPD")
        
    except Exception as e:
        logger.error(f"MPD get_all_songs failed: {e}", exc_info=True)
    finally:
        _disconnect_safely(client)
    return all_formatted_songs

def get_playlist_by_name(playlist_name):
    """Finds an MPD playlist by its exact name."""
    client = _connect()
    if not client:
        return None
        
    try:
        playlists = client.listplaylists()
        for p in playlists:
            if p.get('playlist') == playlist_name:
                return {'Id': playlist_name, 'Name': playlist_name}
    except Exception as e:
        logger.error(f"MPD get_playlist_by_name failed for '{playlist_name}': {e}", exc_info=True)
    finally:
        _disconnect_safely(client)
    return None

def create_playlist(base_name, item_ids):
    """Creates a new playlist on MPD. item_ids are file paths."""
    client = _connect()
    if not client:
        return

    try:
        # Check if playlist exists and clear it, otherwise MPD appends.
        if any(p.get('playlist') == base_name for p in client.listplaylists()):
             client.playlistclear(base_name)
             logger.info(f"Cleared existing MPD playlist '{base_name}'.")

        for item_path in item_ids:
            client.playlistadd(base_name, item_path)
        logger.info(f"✅ Created/updated MPD playlist '{base_name}' with {len(item_ids)} songs.")
    except Exception as e:
        logger.error(f"Exception creating MPD playlist '{base_name}': {e}", exc_info=True)
    finally:
        _disconnect_safely(client)

def get_all_playlists():
    """Fetches all playlists from MPD."""
    client = _connect()
    if not client:
        return []
        
    playlists = []
    try:
        mpd_playlists = client.listplaylists()
        playlists = [{'Id': p.get('playlist'), 'Name': p.get('playlist')} for p in mpd_playlists]
    except Exception as e:
        logger.error(f"MPD get_all_playlists failed: {e}", exc_info=True)
    finally:
        _disconnect_safely(client)
    return playlists

def delete_playlist(playlist_id):
    """Deletes a playlist on MPD. playlist_id is the playlist name."""
    client = _connect()
    if not client:
        return False
        
    success = False
    try:
        client.rm(playlist_id)
        logger.info(f"🗑️ Deleted MPD playlist: {playlist_id}")
        success = True
    except Exception as e:
        logger.error(f"Exception deleting MPD playlist '{playlist_id}': {e}", exc_info=True)
    finally:
        _disconnect_safely(client)
    return success

# --- User-specific MPD functions (STUBS) ---
# MPD is a single-user daemon and does not track play counts or last played times by default.
def get_top_played_songs(limit, user_creds=None):
    """Not supported by MPD. Returns an empty list."""
    logger.warning("get_top_played_songs is not supported by the MPD backend.")
    return []

def get_last_played_time(item_id, user_creds=None):
    """Not supported by MPD. Returns None."""
    logger.warning("get_last_played_time is not supported by the MPD backend.")
    return None

def create_instant_playlist(playlist_name, item_ids, user_creds=None, add_instant_suffix=True):
    """Creates a new instant playlist on MPD."""
    final_playlist_name = f"{playlist_name.strip()}_instant" if add_instant_suffix else playlist_name.strip()
    # For MPD, this is the same as a regular playlist. user_creds are ignored.
    create_playlist(final_playlist_name, item_ids)
    # The return value for this function in other implementations is a dict.
    return {'Id': final_playlist_name, 'Name': final_playlist_name}

