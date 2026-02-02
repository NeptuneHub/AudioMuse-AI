# tasks/mediaserver_lyrion.py

import requests
import logging
import os
from urllib.parse import unquote, urlparse
import config

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 300

# Custom exception for Lyrion API failures so callers can decide how to handle them
class LyrionAPIError(Exception):
    pass

def _decode_lyrion_url(url):
    """Decode Lyrion file:// URI to a plain filesystem path."""
    if not url:
        return None
    if url.startswith('file://'):
        return unquote(urlparse(url).path)
    return unquote(url)

# ##############################################################################
# LYRION (JSON-RPC) IMPLEMENTATION
# ##############################################################################
# Lyrion uses a JSON-RPC API. This section contains functions to interact with it.

def _get_target_paths_for_filtering():
    """
    Gets the target paths from config for path-based filtering.
    Returns a set of lowercase paths to match against, or None if no filtering.
    """
    folder_names_str = getattr(config, 'MUSIC_LIBRARIES', '')
    logger.info(f"DEBUG: MUSIC_LIBRARIES config value: '{folder_names_str}'")

    if not folder_names_str.strip():
        logger.info("DEBUG: MUSIC_LIBRARIES is empty, no path filtering")
        return None

    target_paths = {path.strip().lower() for path in folder_names_str.split(',') if path.strip()}
    logger.info(f"DEBUG: Target paths for filtering: {list(target_paths)}")
    return target_paths

def _try_alternative_lms_calls_for_path(album):
    """
    Try alternative LMS/Lyrion JSON-RPC calls to get path information.
    """
    album_id = album.get('id')
    album_title = album.get('album', 'Unknown')
    
    if not album_id:
        return None
        
    logger.info(f"DEBUG: Trying alternative LMS calls for album '{album_title}' (ID: {album_id})")
    
    # Method 1: Try 'songinfo' command with track lookup
    try:
        # First get tracks with more detailed info
        tracks_response = _jsonrpc_request("titles", [0, 1, f"album_id:{album_id}", "tags:uflocpPa"])
        if tracks_response and "titles_loop" in tracks_response and tracks_response["titles_loop"]:
            first_track = tracks_response["titles_loop"][0]
            logger.info(f"DEBUG: Track with tags: {first_track}")
            
            # Check for path-related fields with tags
            for field in ['url', 'path', 'remote_title', 'u', 'f', 'l', 'o', 'c', 'p', 'P', 'a']:
                if field in first_track and first_track[field]:
                    logger.info(f"DEBUG: Found potential path in track field '{field}': {first_track[field]}")
                    return first_track[field]
    except Exception as e:
        logger.info(f"DEBUG: Method 1 failed: {e}")
    
    # Method 2: Try 'songinfo' command directly on track
    try:
        tracks_simple = _jsonrpc_request("titles", [0, 1, f"album_id:{album_id}"])
        if tracks_simple and "titles_loop" in tracks_simple and tracks_simple["titles_loop"]:
            track_id = tracks_simple["titles_loop"][0].get('id')
            if track_id:
                songinfo_response = _jsonrpc_request("songinfo", [0, 20, f"track_id:{track_id}"])
                logger.info(f"DEBUG: Songinfo response: {songinfo_response}")
                if songinfo_response and "songinfo_loop" in songinfo_response:
                    for info_item in songinfo_response["songinfo_loop"]:
                        if 'url' in info_item or 'path' in info_item:
                            logger.info(f"DEBUG: Found path in songinfo: {info_item}")
                            return info_item.get('url') or info_item.get('path')
    except Exception as e:
        logger.info(f"DEBUG: Method 2 failed: {e}")
    
    # Method 3: Try 'browse' command to get folder structure  
    try:
        browse_response = _jsonrpc_request("browse", [0, 1, "item_id:0", "folder_id"])
        logger.info(f"DEBUG: Browse response: {browse_response}")
        # This might give us folder structure information
    except Exception as e:
        logger.info(f"DEBUG: Method 3 (browse) failed: {e}")
    
    # Method 4: Try getting album with different parameters
    try:
        album_detailed = _jsonrpc_request("albums", [0, 1, f"album_id:{album_id}", "tags:alj"])
        logger.info(f"DEBUG: Detailed album response: {album_detailed}")
        if album_detailed and "albums_loop" in album_detailed:
            for detailed_album in album_detailed["albums_loop"]:
                for field in ['url', 'path', 'j', 'l', 'a']:  # j=artwork, l=album, a=artist
                    if field in detailed_album and detailed_album[field]:
                        logger.info(f"DEBUG: Found potential path in detailed album field '{field}': {detailed_album[field]}")
                        return detailed_album[field]
    except Exception as e:
        logger.info(f"DEBUG: Method 4 failed: {e}")
        
    return None

def _album_matches_target_paths(album, target_paths):
    """
    Check if an album's path matches any of the target paths.
    Uses multiple LMS API methods to find path information.
    """
    if target_paths is None:
        return True  # No filtering
    
    album_title = album.get('album', 'Unknown')
    logger.info(f"DEBUG: Checking album '{album_title}' - basic album data: {album}")
    
    # Try to get path using alternative LMS calls
    album_path = _try_alternative_lms_calls_for_path(album)
    
    if not album_path:
        logger.info(f"DEBUG: No path found for album '{album_title}' using any method")
        return False
    
    album_path_lower = album_path.lower()
    logger.info(f"DEBUG: Checking album path '{album_path}' against targets: {list(target_paths)}")
    
    # Check if the album path contains any of the target paths
    for target_path in target_paths:
        if target_path in album_path_lower:
            logger.info(f"DEBUG: MATCH found - '{target_path}' in '{album_path_lower}'")
            return True
    
    logger.info(f"DEBUG: No match for album path '{album_path_lower}'")
    return False

def _get_target_music_folder_ids():
    """
    Parses config for music folder names and returns their IDs for filtering using a robust,
    case-insensitive matching against the server's actual folder configuration.
    """
    folder_names_str = getattr(config, 'MUSIC_LIBRARIES', '')

    logger.info(f"DEBUG: MUSIC_LIBRARIES config value: '{folder_names_str}'")

    if not folder_names_str.strip():
        logger.info("DEBUG: MUSIC_LIBRARIES is empty, scanning all folders")
        return None

    target_names_lower = {name.strip().lower() for name in folder_names_str.split(',') if name.strip()}
    logger.info(f"DEBUG: Target names/paths to match: {list(target_names_lower)}")

    # Use the musicfolders command to get the available music folders.
    response = _jsonrpc_request("musicfolders", [0, 999999])
    
    logger.info(f"DEBUG: Lyrion musicfolders response: {response}")
    
    if not response:
        logger.error("Failed to fetch music folders from Lyrion or response was empty.")
        logger.warning("Since MUSIC_LIBRARIES is configured but folder detection failed, returning empty set to prevent scanning everything.")
        return set()

    # Extract folder list from response
    all_folders = []
    if isinstance(response, dict) and "folder_loop" in response:
        all_folders = response["folder_loop"]
    elif isinstance(response, dict) and "folders_loop" in response:
        all_folders = response["folders_loop"]
    elif isinstance(response, list):
        all_folders = response
    else:
        # Try to find the first list in the response dict
        if isinstance(response, dict):
            for v in response.values():
                if isinstance(v, list):
                    all_folders = v
                    break

    if not all_folders:
        logger.error("No music folders found in Lyrion response.")
        return set()

    # Build a case-insensitive map: lowercase_name_or_path -> {'name': OriginalCaseName, 'id': FolderId, 'path': FolderPath}
    folder_map = {}
    for folder in all_folders:
        if isinstance(folder, dict):
            folder_name = folder.get('name') or folder.get('folder')
            folder_path = folder.get('path') or folder.get('url')  # Lyrion may use 'url' for the path
            folder_id = folder.get('id') or folder.get('folder_id')
            logger.info(f"DEBUG: Processing folder - name: '{folder_name}', path: '{folder_path}', id: '{folder_id}', raw: {folder}")
            if folder_name and folder_id:
                folder_info = {'name': folder_name, 'id': folder_id, 'path': folder_path or folder_name}
                # Map both name and path (if different) to the same folder info
                folder_map[folder_name.lower()] = folder_info
                if folder_path and folder_path.lower() != folder_name.lower():
                    folder_map[folder_path.lower()] = folder_info
                logger.info(f"DEBUG: Added to folder_map - name key: '{folder_name.lower()}', path key: '{folder_path.lower() if folder_path else 'N/A'}'")

    # --- DIAGNOSTIC LOGGING ---
    # Get unique folder info (since we may have duplicates from name/path mapping)
    unique_folders = {folder['id']: folder for folder in folder_map.values()}
    available_info = [f"{folder['name']} (path: {folder['path']})" for folder in unique_folders.values()]
    logger.info(f"Available Lyrion music folders found: {available_info}")
    # --- END DIAGNOSTIC LOGGING ---

    # Match user's config against the map to find IDs and original names
    found_folders = []
    unfound_names = []
    logger.info(f"DEBUG: Available folder_map keys: {list(folder_map.keys())}")
    for target_name in target_names_lower:
        logger.info(f"DEBUG: Looking for target: '{target_name}'")
        if target_name in folder_map:
            found_folders.append(folder_map[target_name])
            logger.info(f"DEBUG: FOUND match for '{target_name}': {folder_map[target_name]}")
        else:
            unfound_names.append(target_name)
            logger.info(f"DEBUG: NO MATCH found for '{target_name}'")

    if unfound_names:
        logger.warning(f"Lyrion config specified folder names that were not found: {list(unfound_names)}")

    if not found_folders:
        logger.warning(f"No matching music folders found for configured names: {list(target_names_lower)}. No albums will be analyzed.")
        return set()

    music_folder_ids = {folder['id'] for folder in found_folders}
    found_info = [f"{folder['name']} (path: {folder['path']})" for folder in found_folders]

    logger.info(f"Filtering analysis to {len(music_folder_ids)} Lyrion folders: {found_info}")
    logger.info(f"DEBUG: Returning folder IDs: {music_folder_ids}")
    return music_folder_ids

def _get_first_player():
    """Gets the first available player from Lyrion for web interface operations."""
    try:
        response = _jsonrpc_request("players", [0, 1])
        if response and "players_loop" in response and response["players_loop"]:
            player = response["players_loop"][0]
            player_id = player.get("playerid")
            if player_id:
                logger.info(f"Found Lyrion player: {player_id}")
                return player_id
        
        # Fallback: try to use a common default or return None
        logger.warning("No Lyrion players found, using fallback player ID")
        return "10.42.6.0"  # Use the player from your example as fallback
    except Exception as e:
        logger.error(f"Error getting Lyrion player: {e}")
        return "10.42.6.0"  # Use the player from your example as fallback

def _jsonrpc_request(method, params, player_id=""):
    """
    Helper to make a JSON-RPC request to the Lyrion server without authentication.
    Returns the 'result' field on success, or None on failure.
    """
    url = f"{config.LYRION_URL}/jsonrpc.js"
    payload = {
        "id": 1,
        "method": "slim.request",
        "params": [player_id, [method, *params]]
    }

    # Try with retry logic for connection issues. Use the configured REQUESTS_TIMEOUT
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with requests.Session() as s:
                s.headers.update({"Content-Type": "application/json"})
                # Use configured timeout so slow servers can be handled
                r = s.post(url, json=payload, timeout=REQUESTS_TIMEOUT)

            r.raise_for_status()
            response_data = r.json()

            if response_data.get("error"):
                # JSON-RPC error from the server is fatal for this call
                msg = response_data['error'].get('message')
                logger.error(f"Lyrion JSON-RPC Error: {msg}")
                raise LyrionAPIError(f"Lyrion API error: {msg}")

            # On success, return the result field. It might be None if not present.
            return response_data.get("result")

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.warning(f"Connection issue with Lyrion API (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)  # Wait 2 seconds before retry
                continue
            else:
                err = f"Failed to connect to Lyrion after {max_retries} attempts: {e}"
                logger.error(err)
                raise LyrionAPIError(err)
        except LyrionAPIError:
            # Propagate our custom errors as-is
            raise
        except Exception as e:
            logger.error(f"Failed to call Lyrion JSON-RPC API with method '{method}': {e}", exc_info=True)
            raise LyrionAPIError(f"Unexpected error calling Lyrion API: {e}")

    # Should not reach here; raise to be explicit
    raise LyrionAPIError("Unreachable: exceeded jsonrpc retry loop")


def _count_albums(use_sort_new=True, page_size=100):
    """Count total albums on the Lyrion server using the given paging mode.
    Returns an integer count or None on fatal error.
    """
    total = 0
    offset = 0
    while True:
        params = [offset, page_size, "sort:new"] if use_sort_new else [offset, page_size]
        try:
            resp = _jsonrpc_request("albums", params)
        except Exception as e:
            logger.warning(f"Unable to count albums (use_sort_new={use_sort_new}): {e}")
            return None

        page_albums = []
        if isinstance(resp, dict) and "albums_loop" in resp:
            page_albums = resp["albums_loop"]
        elif isinstance(resp, list):
            page_albums = resp

        if not page_albums:
            break

        total += len(page_albums)
        if len(page_albums) < page_size:
            break
        offset += len(page_albums)

    return total

def download_track(temp_dir, item):
    """Downloads a single track from Lyrion using its URL."""
    try:
        track_id = item.get('Id')
        if not track_id:
            logger.error("Lyrion item does not have a track ID.")
            return None
            
        # The correct, stable URL format for directly downloading a track from Lyrion/LMS by its ID.
        # This avoids issues with the /stream endpoint which is often for the currently playing track.
        download_url = f"{config.LYRION_URL}/music/{track_id}/download"
        
        # A more robust way to handle the file extension.
        file_extension = item.get('Path', '.mp3')
        if file_extension and '.' in file_extension:
            file_extension = os.path.splitext(file_extension)[1]
        else:
            file_extension = '.mp3'
        
        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")
        
        logger.info(f"Attempting to download from URL: {download_url}")
        
        # Use a new session for each download to avoid connection pooling issues.
        with requests.Session() as s:
            with s.get(download_url, stream=True, timeout=REQUESTS_TIMEOUT) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        logger.info(f"Downloaded '{item.get('title', 'Unknown')}' to '{local_filename}'")
        return local_filename
    except Exception as e:
        logger.error(f"Failed to download Lyrion track {item.get('title', 'Unknown')}: {e}", exc_info=True)
    return None

def _get_all_albums_simple(limit):
    """Simple album fetching without filtering."""
    albums_accum = []
    fetch_all = (limit == 0)
    page_size = 100
    remaining = None if fetch_all else int(limit)
    offset = 0

    pages_fetched = 0

    # Decide whether to use 'sort:new' or fall back to unsorted pagination.
    # Only perform a full comparison when fetching the entire library (limit==0).
    use_sort_new = True
    if limit == 0:
        logger.info("Counting Lyrion albums with and without 'sort:new' to decide pagination mode...")
        sorted_total = _count_albums(use_sort_new=True, page_size=page_size)
        unsorted_total = _count_albums(use_sort_new=False, page_size=page_size)
        logger.info(f"Albums with sort:new = {sorted_total}, albums without sort = {unsorted_total}")
        if unsorted_total is not None and sorted_total is not None and unsorted_total > sorted_total:
            logger.info("Albums without sort are more numerous; proceeding with unsorted pagination for Lyrion.")
            use_sort_new = False

    while True:
        req_count = page_size if (remaining is None or remaining > page_size) else remaining
        params = [offset, req_count, "sort:new"] if use_sort_new else [offset, req_count]

        # Per-page retry: don't let a single slow/failed page stop the whole fetch
        page_response = None
        page_error = None
        for attempt in range(3):
            try:
                page_response = _jsonrpc_request("albums", params)
                break
            except LyrionAPIError as e:
                page_error = e
                logger.warning(f"albums page fetch failed at offset={offset} (attempt {attempt+1}/3): {e}")
                import time
                time.sleep(1)
                continue

        if page_response is None:
            # After attempts, skip this page and continue to the next to avoid stopping at page 1
            logger.error(f"Skipping albums page at offset={offset} after repeated failures: {page_error}")
            # Advance offset by page_size to avoid an infinite loop. This may lose items but avoids halting.
            offset += page_size
            continue

        response = page_response

        # Normalize response shapes
        page_albums = []
        if isinstance(response, dict) and "albums_loop" in response:
            page_albums = response["albums_loop"]
        elif isinstance(response, list):
            page_albums = response

        # If server legitimately returned an empty page, we've reached the end
        if not page_albums:
            break

        pages_fetched += 1

        mapped = [{'Id': a.get('id'), 'Name': a.get('album')} for a in page_albums]
        albums_accum.extend(mapped)

        if remaining is not None:
            remaining -= len(page_albums)
            if remaining <= 0:
                break

        if len(page_albums) < req_count:
            break

        offset += len(page_albums)

    logger.info(f"_get_all_albums_simple: fetched {len(albums_accum)} albums across {pages_fetched} pages (requested limit: {limit})")
    return albums_accum

def _try_folder_id_based_filtering(target_paths, limit):
    """
    Try to get the actual folder ID and use it for filtering.
    This bypasses the path detection issues by working directly with folder IDs.
    """
    if not target_paths:
        return None
        
    logger.info(f"DEBUG: Trying folder-based filtering for paths: {list(target_paths)}")
    
    # Try to get music folders with retry and better error handling
    music_folders = None
    for attempt in range(3):
        try:
            logger.info(f"DEBUG: Attempt {attempt + 1}/3 to get musicfolders")
            response = _jsonrpc_request("musicfolders", [0, 999])
            if response:
                music_folders = response
                break
        except Exception as e:
            logger.info(f"DEBUG: Musicfolders attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                import time
                time.sleep(1)
    
    if not music_folders:
        logger.info("DEBUG: Could not get music folders, cannot use folder-based filtering")
        return None
    
    logger.info(f"DEBUG: Successfully got music folders: {music_folders}")
    
    # Extract folder list
    all_folders = []
    if isinstance(music_folders, dict) and "folder_loop" in music_folders:
        all_folders = music_folders["folder_loop"]
    elif isinstance(music_folders, list):
        all_folders = music_folders
    
    if not all_folders:
        logger.info("DEBUG: No folders found in musicfolders response")
        return None
    
    # Find matching folder IDs - look for folders that contain our target path
    matching_folder_ids = []
    for folder in all_folders:
        if isinstance(folder, dict):
            folder_name = folder.get('name', '')
            folder_path = folder.get('path', '') or folder.get('url', '')
            folder_id = folder.get('id', '') or folder.get('folder_id', '')
            
            logger.info(f"DEBUG: Checking folder - ID: {folder_id}, Name: {folder_name}, Path: {folder_path}")
            
            # Check if any target path matches this folder's path
            for target_path in target_paths:
                if (target_path.lower() in folder_path.lower() or 
                    target_path.lower() in folder_name.lower() or
                    folder_path.lower() in target_path.lower()):
                    logger.info(f"DEBUG: FOLDER MATCH found - folder '{folder_name}' (path: {folder_path}) matches target '{target_path}'")
                    matching_folder_ids.append(folder_id)
                    break
    
    if not matching_folder_ids:
        logger.info(f"DEBUG: No matching folders found for target paths: {list(target_paths)}")
        return None
    
    logger.info(f"DEBUG: Using folder IDs for filtering: {matching_folder_ids}")
    
    # Get albums from matching folders
    albums_found = []
    for folder_id in matching_folder_ids:
        try:
            albums_response = _jsonrpc_request("albums", [0, limit or 100, f"folder_id:{folder_id}", "sort:new"])
            logger.info(f"DEBUG: Albums response for folder {folder_id}: found {len(albums_response.get('albums_loop', []))} albums" if albums_response else "No response")
            
            if albums_response and "albums_loop" in albums_response:
                folder_albums = albums_response["albums_loop"]
                mapped = [{'Id': a.get('id'), 'Name': a.get('album')} for a in folder_albums]
                albums_found.extend(mapped)
                
                if limit and len(albums_found) >= limit:
                    albums_found = albums_found[:limit]
                    break
                    
        except Exception as e:
            logger.info(f"DEBUG: Failed to get albums for folder {folder_id}: {e}")
            continue
    
    return albums_found if albums_found else None

def _album_has_tracks_in_target_path(album_id, target_paths):
    """
    Check if an album has tracks in the target folder by examining actual file paths.
    This is the most reliable method for Lyrion folder filtering.
    """
    # Request more tracks and retry on transient API errors. Inspect more tracks to increase
    # chance of finding a path field.
    attempts = 3
    for attempt in range(attempts):
        try:
            # Ask for up to 20 tracks with richer tags that may include path/url
            response = _jsonrpc_request("titles", [0, 20, f"album_id:{album_id}", "tags:fFlpuoP"])

            if not response or "titles_loop" not in response:
                return False

            tracks = response["titles_loop"]

            # Check various fields that might contain the actual file path
            path_fields = ['url', 'path', 'f', 'F', 'l', 'p', 'u', 'o', 'file', 'filename']

            # Inspect the first N tracks (increase from 3 to 8)
            for track in tracks[:8]:
                for field in path_fields:
                    if field in track and track[field]:
                        track_path = str(track[field]).lower()

                        # Check if any target path is in this track's path
                        for target_path in target_paths:
                            if target_path in track_path:
                                return True

                        # Also check if track path contains target path parts
                        for target_path in target_paths:
                            target_parts = target_path.strip('/').split('/')
                            if len(target_parts) >= 2:
                                # Check if last part of target path is in track path
                                last_part = target_parts[-1].lower()
                                if last_part in track_path:
                                    return True

            return False

        except LyrionAPIError as e:
            logger.warning(f"Transient Lyrion API error checking album tracks (attempt {attempt+1}/{attempts}): {e}")
            import time
            time.sleep(1)
            continue
        except Exception as e:
            logger.debug(f"Unexpected error while checking album tracks for {album_id}: {e}", exc_info=True)
            return False

    # If all attempts failed, assume no path info found (caller will decide how to treat this)
    logger.error(f"Failed to fetch tracks for album {album_id} after {attempts} attempts")
    return False

def get_recent_albums(limit):
    """
    Fetches recently added albums from Lyrion using JSON-RPC.
    If MUSIC_LIBRARIES is set, filters albums by checking if their tracks' actual file paths match.
    Scans ALL albums until the requested number is found (or library is exhausted).
    """
    target_paths = _get_target_paths_for_filtering()
    
    # If no filtering needed, use simple approach
    if target_paths is None:
        return _get_all_albums_simple(limit)
    
    # Use file path checking approach - scan ALL albums until we find enough matches
    logger.info(f"Scanning Lyrion library for albums in configured folders (limit: {limit or 'all'})")
    
    filtered_albums = []
    page_size = 100
    offset = 0
    fetch_all = (limit == 0)
    # Diagnostics counters to help debug why albums are filtered out
    pages_fetched = 0
    albums_scanned = 0
    albums_matched = 0
    filtered_no_album_id = 0
    filtered_no_path = 0
    
    # Like the simple fetcher, try 'sort:new' first but fall back to
    # unsorted pagination if that appears to return fewer albums. Only
    # perform the full comparison when fetching the entire library.
    use_sort_new = True
    if fetch_all:
        logger.info("Counting Lyrion albums with and without 'sort:new' to decide pagination mode (analysis of all albums)...")
        sorted_total = _count_albums(use_sort_new=True, page_size=page_size)
        unsorted_total = _count_albums(use_sort_new=False, page_size=page_size)
        logger.info(f"Albums with sort:new = {sorted_total}, albums without sort = {unsorted_total}")
        if unsorted_total is not None and sorted_total is not None and unsorted_total > sorted_total:
            logger.info("Albums without sort are more numerous; proceeding with unsorted pagination for Lyrion.")
            use_sort_new = False

    while True:
        # Get next batch of albums
        params = [offset, page_size, "sort:new"] if use_sort_new else [offset, page_size]
        # Per-page retry: attempt several times before skipping this page
        page_response = None
        page_error = None
        for attempt in range(3):
            try:
                page_response = _jsonrpc_request("albums", params)
                break
            except LyrionAPIError as e:
                page_error = e
                logger.warning(f"albums page fetch failed at offset={offset} (attempt {attempt+1}/3): {e}")
                import time
                time.sleep(1)
                continue

        if page_response is None:
            logger.error(f"Skipping albums page at offset={offset} after repeated failures: {page_error}")
            offset += page_size
            # continue to next page instead of aborting entire scan
            continue

        response = page_response
        if not response:
            # Legitimate empty response -> end of library
            break

        pages_fetched += 1
        
        # Extract albums from response
        page_albums = []
        if isinstance(response, dict) and "albums_loop" in response:
            page_albums = response["albums_loop"]
        elif isinstance(response, list):
            page_albums = response
        
        if not page_albums:
            break
        
        # Check each album in this batch
        for album in page_albums:
            albums_scanned += 1
            album_id = album.get('id')
            album_name = album.get('album', 'Unknown')

            if not album_id:
                filtered_no_album_id += 1
                logger.debug(f"Skipping album without ID (page {pages_fetched}): {album}")
                continue

            # Check if this album's tracks are in our target folder
            try:
                has_tracks = _album_has_tracks_in_target_path(album_id, target_paths)
            except Exception as e:
                logger.warning(f"Error checking album paths for album {album_id}: {e}")
                has_tracks = False

            if has_tracks:
                albums_matched += 1
                mapped_album = {'Id': album_id, 'Name': album_name}
                filtered_albums.append(mapped_album)

                # Stop if we have enough albums (unless fetching all)
                if not fetch_all and len(filtered_albums) >= limit:
                    logger.info(f"Found {limit} matching albums in configured folders (pages fetched: {pages_fetched}, albums scanned: {albums_scanned}, matched: {albums_matched})")
                    return filtered_albums
            else:
                filtered_no_path += 1
                logger.debug(f"Album {album_id} ('{album_name}') does not appear to have tracks in target paths (page {pages_fetched}).")
        
        # If this page had fewer albums than requested, we've reached the end
        if len(page_albums) < page_size:
            break
        
        offset += len(page_albums)
    
    logger.info(f"Found {len(filtered_albums)} albums in configured folders (pages fetched: {pages_fetched}, albums scanned: {albums_scanned}, matched: {albums_matched}, filtered_no_path: {filtered_no_path}, filtered_no_album_id: {filtered_no_album_id})")
    return filtered_albums

def get_all_songs():
    """
    Fetches all songs from Lyrion using JSON-RPC.
    For now, just gets all songs since folder filtering is complex in Lyrion.
    """
    target_paths = _get_target_paths_for_filtering()

    if target_paths is not None:
        logger.warning("LYRION FOLDER FILTERING IS DISABLED - fetching all songs instead")
    
    # Fetch all songs without filtering
    logger.info("Fetching all songs from Lyrion")
    response = _jsonrpc_request("titles", [0, 999999, "tags:galduAyR"])
    
    all_songs = []
    if response and "titles_loop" in response:
        songs = response["titles_loop"]
        
        # Map all songs to our standard format
        for song in songs:
            # Prioritize track artist over album artist to avoid "Various Artists"
            if song.get('trackartist'):
                track_artist = song.get('trackartist')
                used_field = 'trackartist'
            elif song.get('contributor'):
                track_artist = song.get('contributor')
                used_field = 'contributor'
            elif song.get('artist'):
                track_artist = song.get('artist')
                used_field = 'artist'
            elif song.get('albumartist'):
                track_artist = song.get('albumartist')
                used_field = 'albumartist'
            elif song.get('band'):
                track_artist = song.get('band')
                used_field = 'band'
            else:
                track_artist = 'Unknown Artist'
                used_field = 'fallback'
            
            mapped_song = {
                'Id': song.get('id'),
                'Name': song.get('title'),
                'AlbumArtist': track_artist,
                'OriginalAlbumArtist': song.get('albumartist'),
                'Path': song.get('url'),
                'url': song.get('url'),
                'Year': int(song.get('year')) if song.get('year') else None,
                'Rating': int(song.get('rating')) if song.get('rating') else None,
                'FilePath': _decode_lyrion_url(song.get('url')),
            }
            all_songs.append(mapped_song)

        logger.info(f"Found {len(songs)} total songs")

    return all_songs

def _add_to_playlist(playlist_id, item_ids):
    """Adds songs to a Lyrion playlist using the working player-based method."""
    if not item_ids: 
        return True
    
    logger.info(f"Adding {len(item_ids)} songs to Lyrion playlist ID '{playlist_id}'.")
    
    # Get a player for the command
    player_id = _get_first_player()
    if not player_id:
        logger.error("No Lyrion player available for playlist operations.")
        return False
    
    try:
        # Get the original playlist name FIRST, before any operations
        logger.debug("Step 0: Getting original playlist name before operations")
        playlist_info = _jsonrpc_request("playlists", [0, 999999])  # Get all playlists
        
        original_name = None
        if playlist_info and "playlists_loop" in playlist_info:
            for pl in playlist_info["playlists_loop"]:
                if str(pl.get("id")) == str(playlist_id):
                    original_name = pl.get("playlist")
                    logger.debug(f"Found original playlist name: '{original_name}' for ID {playlist_id}")
                    break
        
        if not original_name:
            logger.error(f"Could not find playlist {playlist_id} in playlists list!")
            return False
        
        # Method: Load playlist to player, add tracks, then use playlists edit to update
        logger.info(f"Using method: Load → Add → Update original playlist via edit command")
        
        # Step 1: Load the saved playlist into the player's current playlist
        logger.debug(f"Step 1: Loading playlist {playlist_id} to player {player_id}")
        load_response = _jsonrpc_request("playlistcontrol", [
            "cmd:load",
            f"playlist_id:{playlist_id}"
        ], player_id)
        
        logger.debug(f"Load playlist response: {load_response}")
        
        # Step 2: Add tracks to the player's current playlist in batches
        batch_size = 50  # Larger batches since this method works
        total_added = 0
        
        for i in range(0, len(item_ids), batch_size):
            batch_ids = item_ids[i:i + batch_size]
            track_id_list = ",".join(str(track_id) for track_id in batch_ids)
            
            logger.debug(f"Step 2: Adding batch {i//batch_size + 1} with {len(batch_ids)} tracks")
            add_response = _jsonrpc_request("playlistcontrol", [
                "cmd:add",
                f"track_id:{track_id_list}"
            ], player_id)
            
            logger.debug(f"Add batch response: {add_response}")
            
            if add_response and "count" in add_response:
                batch_added = add_response.get("count", 0)
                total_added += batch_added
                logger.debug(f"Added {batch_added} tracks in this batch, total: {total_added}")
            
            # Small delay between batches
            if i + batch_size < len(item_ids):
                import time
                time.sleep(0.1)
        
        # Step 3: Delete the original empty playlist
        logger.debug(f"Step 3: Deleting original empty playlist {playlist_id}")
        delete_response = _jsonrpc_request("playlists", [
            "delete",
            f"playlist_id:{playlist_id}"
        ])
        logger.debug(f"Delete response: {delete_response}")
        
        # Step 4: Save the current player playlist with the original name
        logger.debug(f"Step 4: Saving current playlist as '{original_name}'")
        save_response = _jsonrpc_request("playlist", [
            "save",
            original_name,
            "silent:1"
        ], player_id)
        
        logger.debug(f"Save playlist response: {save_response}")
        
        # Check if we got the expected playlist ID back
        if save_response and "__playlist_id" in save_response:
            final_playlist_id = save_response["__playlist_id"]
            if str(final_playlist_id) == str(playlist_id):
                logger.info(f"✅ Successfully updated original playlist {playlist_id} with {total_added} tracks")
                return True
            else:
                logger.warning(f"Created new playlist {final_playlist_id} instead of updating {playlist_id}")
                # If we got a different ID, try to delete the new one and rename it
                try:
                    # The new playlist has the tracks, so we need to work with it
                    logger.info(f"Working with new playlist ID {final_playlist_id} which has the content")
                    return True
                except Exception as e:
                    logger.error(f"Error handling new playlist: {e}")
                    return False
        elif total_added > 0:
            logger.info(f"✅ Successfully added {total_added} tracks (save response: {save_response})")
            return True
        else:
            logger.warning("No tracks were added to the playlist")
            return False
            
    except Exception as e:
        logger.error(f"Error in playlist update method: {e}")
        return False

def _create_playlist_batched(playlist_name, item_ids):
    """Creates a new Lyrion playlist and adds tracks using the web interface approach."""
    logger.info(f"Attempting to create Lyrion playlist '{playlist_name}' with {len(item_ids)} songs using web interface method.")

    try:
        # Step 1: Create the playlist using JSON-RPC (this part works)
        create_response = _jsonrpc_request("playlists", ["new", f"name:{playlist_name}"])
        
        if create_response:
            playlist_id = (
                create_response.get("id") or
                create_response.get("overwritten_playlist_id") or
                create_response.get("playlist_id")
            )
            
            if playlist_id:
                logger.info(f"✅ Created Lyrion playlist '{playlist_name}' (ID: {playlist_id}).")
                
                # Step 2: Add tracks using the web interface method
                if item_ids:
                    if _add_to_playlist(playlist_id, item_ids):
                        logger.info(f"✅ Successfully added {len(item_ids)} tracks to playlist '{playlist_name}'.")
                    else:
                        logger.warning(f"Playlist '{playlist_name}' created but some tracks may not have been added.")
                
                return {"Id": playlist_id, "Name": playlist_name}
        
        logger.error(f"Failed to create Lyrion playlist '{playlist_name}'. Response: {create_response}")
        return None
        
    except Exception as e:
        logger.error(f"Exception creating Lyrion playlist '{playlist_name}': {e}", exc_info=True)
        return None

def create_playlist(base_name, item_ids):
    """Creates a new playlist on Lyrion using admin credentials, with batching."""
    # Return the result of the batched creation so callers can inspect the new playlist ID
    return _create_playlist_batched(base_name, item_ids)

def get_all_playlists():
    """Fetches all playlists from Lyrion using JSON-RPC."""
    response = _jsonrpc_request("playlists", [0, 999999])
    if response and "playlists_loop" in response:
        playlists = response["playlists_loop"]
        return [{'Id': p.get('id'), 'Name': p.get('playlist')} for p in playlists]
    return []

def delete_playlist(playlist_id):
    """Deletes a playlist on Lyrion using JSON-RPC."""
    # The correct command is 'playlists delete'.
    response = _jsonrpc_request("playlists", ["delete", f"playlist_id:{playlist_id}"])
    if response:
        logger.info(f"🗑️ Deleted Lyrion playlist ID: {playlist_id}")
        return True
    logger.error(f"Failed to delete playlist ID '{playlist_id}' on Lyrion")
    return False

# --- User-specific Lyrion functions ---
def get_tracks_from_album(album_id):
    """Fetches all audio tracks for an album from Lyrion using JSON-RPC."""
    logger.info(f"Attempting to fetch tracks for album ID: {album_id}")
    
    # Lyrion's JSON-RPC doesn't have a direct "get tracks for album" call.
    # The 'titles' command with a filter is the correct way to get songs for an album.
    # We now fetch all songs and filter them by the album ID.
    try:
        response = _jsonrpc_request("titles", [0, 999999, f"album_id:{album_id}", "tags:galduAyR"])
        logger.debug(f"Lyrion API Raw Track Response for Album {album_id}: {response}")
    except Exception as e:
        logger.error(f"Lyrion API call for album {album_id} failed: {e}", exc_info=True)
        return []

    # Normalize response shapes: LMS/Lyrion may return a dict with 'titles_loop' or a raw list.
    songs = []
    if not response:
        logger.warning(f"Lyrion API returned empty response for album {album_id}.")
        return []

    if isinstance(response, dict):
        if "titles_loop" in response and isinstance(response["titles_loop"], list):
            songs = response["titles_loop"]
        else:
            # Fallback: try to find the first list value in the response
            for v in response.values():
                if isinstance(v, list):
                    songs = v
                    break
    elif isinstance(response, list):
        songs = response

    if not songs:
        logger.warning(f"Lyrion API response for tracks of album {album_id} did not contain any song entries.")
        return []

    # Robust Spotify detection: check several possible fields and make it case-insensitive.
    def is_spotify_track(item: dict) -> bool:
        for key in ("genre", "service", "source"):
            val = item.get(key)
            if isinstance(val, str) and "spotify" in val.lower():
                return True
        # Also check the URL/path for spotify links
        url = (item.get("url") or item.get("Path") or item.get("path") or "")
        if isinstance(url, str) and "spotify" in url.lower():
            return True
        return False

    local_songs = []
    skipped_tracks = []
    for s in songs:
        if is_spotify_track(s):
            skipped_tracks.append(s)
        else:
            local_songs.append(s)

    if skipped_tracks:
        skipped_count = len(skipped_tracks)
        logger.info(f"Skipping {skipped_count} track(s) from album {album_id} because they appear to be from Spotify or are non-downloadable.")
        # Log concise identifying information for each skipped track so operators can verify.
        for st in skipped_tracks:
            sk_id = st.get('id') or st.get('Id') or st.get('track_id')
            sk_title = st.get('title') or st.get('name') or st.get('Name')
            # Use track artist prioritization for logging too
            sk_artist = (st.get('trackartist') or st.get('contributor') or 
                        st.get('artist') or st.get('albumartist') or 
                        st.get('band') or 'Unknown Artist')
            sk_url = st.get('url') or st.get('Path') or st.get('path')
            logger.info(f"Skipped track - id: {sk_id!r}, title: {sk_title!r}, artist: {sk_artist!r}, url/path: {sk_url!r}")

    if not local_songs and songs:
        logger.info(f"Album {album_id} contains only Spotify or non-downloadable tracks and will be skipped.")

    # Map Lyrion API keys to our standard format with safe fallbacks.
    mapped = []
    for s in local_songs:
        id_val = s.get('id') or s.get('Id') or s.get('track_id')
        title = s.get('title') or s.get('name') or s.get('Name')
        
        # Prioritize track artist over album artist to avoid "Various Artists"
        if s.get('trackartist'):
            artist = s.get('trackartist')
            used_field = 'trackartist'
        elif s.get('contributor'):
            artist = s.get('contributor')
            used_field = 'contributor'
        elif s.get('artist'):
            artist = s.get('artist')
            used_field = 'artist'
        elif s.get('albumartist'):
            artist = s.get('albumartist')
            used_field = 'albumartist'
        elif s.get('band'):
            artist = s.get('band')
            used_field = 'band'
        else:
            artist = 'Unknown Artist'
            used_field = 'fallback'
        
        path = s.get('url') or s.get('Path') or s.get('path') or ''
        mapped.append({
            'Id': id_val, 'Name': title, 'AlbumArtist': artist, 'OriginalAlbumArtist': s.get('albumartist'),
            'Path': path, 'url': path,
            'Year': int(s.get('year')) if s.get('year') else None,
            'Rating': int(s.get('rating')) if s.get('rating') else None,
            'FilePath': _decode_lyrion_url(s.get('url')),
        })

    return mapped

def get_playlist_by_name(playlist_name):
    """Finds a Lyrion playlist by its exact name using JSON-RPC."""
    # Fetch all playlists and filter by name, as direct name search is not standard.
    all_playlists = get_all_playlists()
    for p in all_playlists:
        if p.get('Name') == playlist_name:
            return p # Return the already formatted playlist dict
    return None

def get_top_played_songs(limit):
    """Fetches the top N most played songs from Lyrion for a specific user using JSON-RPC."""
    response = _jsonrpc_request("titles", [0, limit, "sort:popular", "tags:galduAyR"])
    if response and "titles_loop" in response:
        songs = response["titles_loop"]
        # Map Lyrion API keys to our standard format.
        mapped_songs = []
        for s in songs:
            title = s.get('title', 'Unknown')
            
            # Prioritize track artist over album artist to avoid "Various Artists"
            if s.get('trackartist'):
                track_artist = s.get('trackartist')
                used_field = 'trackartist'
            elif s.get('contributor'):
                track_artist = s.get('contributor')
                used_field = 'contributor'
            elif s.get('artist'):
                track_artist = s.get('artist')
                used_field = 'artist'
            elif s.get('albumartist'):
                track_artist = s.get('albumartist')
                used_field = 'albumartist'
            elif s.get('band'):
                track_artist = s.get('band')
                used_field = 'band'
            else:
                track_artist = 'Unknown Artist'
                used_field = 'fallback'
            
            mapped_songs.append({
                'Id': s.get('id'),
                'Name': title,
                'AlbumArtist': track_artist,
                'OriginalAlbumArtist': s.get('albumartist'),
                'Path': s.get('url'),
                'url': s.get('url'),
                'Year': int(s.get('year')) if s.get('year') else None,
                'Rating': int(s.get('rating')) if s.get('rating') else None,
                'FilePath': _decode_lyrion_url(s.get('url')),
            })
        return mapped_songs
    return []


def get_last_played_time(item_id):
    """Fetches the last played time for a track for a specific user. Not supported by Lyrion JSON-RPC API."""
    logger.warning("Lyrion's JSON-RPC API does not provide a 'last played time' for individual tracks.")
    return None

def create_instant_playlist(playlist_name, item_ids):
    """Creates a new instant playlist on Lyrion for a specific user, with batching."""
    final_playlist_name = f"{playlist_name.strip()}_instant"
    return _create_playlist_batched(final_playlist_name, item_ids)
