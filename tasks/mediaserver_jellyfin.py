# tasks/mediaserver_jellyfin.py

import requests
import logging
import os
import config

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 300

# ##############################################################################
# JELLYFIN IMPLEMENTATION
# ##############################################################################

def _get_target_library_ids():
    """
    Parses config for library names and returns their IDs for filtering using a robust,
    case-insensitive matching against the server's actual library configuration.
    """
    library_names_str = getattr(config, 'MUSIC_LIBRARIES', '')

    if not library_names_str.strip():
        return None

    target_names_lower = {name.strip().lower() for name in library_names_str.split(',') if name.strip()}

    # Use the /Library/VirtualFolders endpoint as it provides the canonical system configuration.
    url = f"{config.JELLYFIN_URL}/Library/VirtualFolders"
    try:
        r = requests.get(url, headers=config.HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        all_libraries = r.json()

        # Build a case-insensitive map: lowercase_name -> {'name': OriginalCaseName, 'id': ItemId}
        library_map = {
            lib['Name'].lower(): {'name': lib['Name'], 'id': lib['ItemId']}
            for lib in all_libraries
            if lib.get('CollectionType') == 'music'
        }

        # --- DIAGNOSTIC LOGGING ---
        available_music_libraries = [lib['name'] for lib in library_map.values()]
        logger.info(f"Available Jellyfin music libraries found: {available_music_libraries}")
        # --- END DIAGNOSTIC LOGGING ---

        # Match user's config against the map to find IDs and original names
        found_libraries = []
        unfound_names = []
        for target_name in target_names_lower:
            if target_name in library_map:
                found_libraries.append(library_map[target_name])
            else:
                unfound_names.append(target_name)

        if unfound_names:
            logger.warning(f"Jellyfin config specified library names that were not found: {list(unfound_names)}")

        if not found_libraries:
            logger.warning(f"No matching music libraries found for configured names: {list(target_names_lower)}. No albums will be analyzed.")
            return set()

        music_library_ids = {lib['id'] for lib in found_libraries}
        found_names_original_case = [lib['name'] for lib in found_libraries]

        logger.info(f"Filtering analysis to {len(music_library_ids)} Jellyfin libraries: {found_names_original_case}")
        return music_library_ids

    except Exception as e:
        logger.error(f"Failed to fetch or parse Jellyfin virtual folders at '{url}': {e}", exc_info=True)
        return set()


def _jellyfin_get_users(token):
    """Fetches a list of all users from Jellyfin using a provided token."""
    url = f"{config.JELLYFIN_URL}/Users"
    headers = {"X-Emby-Token": token}
    try:
        r = requests.get(url, headers=headers, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Jellyfin get_users failed: {e}", exc_info=True)
        return None

def resolve_user(identifier, token):
    """
    Resolves a Jellyfin username to a User ID.
    If the identifier doesn't match any username, it's returned as is, assuming it's already an ID.
    """
    users = _jellyfin_get_users(token)
    if users:
        for user in users:
            if user.get('Name', '').lower() == identifier.lower():
                logger.info(f"Matched username '{identifier}' to User ID '{user['Id']}'.")
                return user['Id']
    
    logger.info(f"No username match for '{identifier}'. Assuming it is a User ID.")
    return identifier # Return original identifier if no match is found

# --- ADMIN/GLOBAL JELLYFIN FUNCTIONS ---
def get_recent_albums(limit):
    """
    Fetches a list of the most recently added albums from Jellyfin using pagination.
    Uses global admin credentials.
    If MUSIC_LIBRARIES is set, it will only return albums from those libraries.
    """
    target_library_ids = _get_target_library_ids()
    
    # Case 1: Config is set, but no matching libraries were found. Scan nothing.
    if isinstance(target_library_ids, set) and not target_library_ids:
        logger.warning("Library filtering is active, but no matching libraries were found on the server. Returning no albums.")
        return []

    all_albums = []
    fetch_all = (limit == 0)

    # Case 2: Config is NOT set (is None). Scan all albums from the user's root without ParentId.
    if target_library_ids is None:
        logger.info("Scanning all Jellyfin libraries for recent albums.")
        start_index = 0
        page_size = 500
        while True:
            # We fetch full pages and apply the limit only after collecting and sorting.
            url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
            params = {
                "IncludeItemTypes": "MusicAlbum", "SortBy": "DateCreated", "SortOrder": "Descending",
                "Recursive": True, "Limit": page_size, "StartIndex": start_index
            }
            try:
                r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
                r.raise_for_status()
                response_data = r.json()
                albums_on_page = response_data.get("Items", [])
                
                if not albums_on_page:
                    break
                
                all_albums.extend(albums_on_page)
                start_index += len(albums_on_page)

                if len(albums_on_page) < page_size:
                    break
            except Exception as e:
                logger.error(f"Jellyfin get_recent_albums failed during 'scan all': {e}", exc_info=True)
                break
    
    # Case 3: Config is set and we have library IDs. Scan each of these libraries by using their ID as ParentId.
    else:
        logger.info(f"Scanning {len(target_library_ids)} specific Jellyfin libraries for recent albums.")
        for library_id in target_library_ids:
            start_index = 0
            page_size = 500
            while True: # Paginate through the current library
                url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
                params = {
                    "IncludeItemTypes": "MusicAlbum", "SortBy": "DateCreated", "SortOrder": "Descending",
                    "Recursive": True, "Limit": page_size, "StartIndex": start_index,
                    "ParentId": library_id
                }
                try:
                    r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
                    r.raise_for_status()
                    response_data = r.json()
                    albums_on_page = response_data.get("Items", [])
                    
                    if not albums_on_page:
                        break
                    
                    all_albums.extend(albums_on_page)
                    start_index += len(albums_on_page)

                    if len(albums_on_page) < page_size:
                        break
                except Exception as e:
                    logger.error(f"Jellyfin get_recent_albums failed for library ID {library_id}: {e}", exc_info=True)
                    break

    # After fetching, a final sort and trim is needed only if we fetched from multiple libraries.
    if target_library_ids is not None and len(target_library_ids) > 1:
        all_albums.sort(key=lambda x: x.get('DateCreated', ''), reverse=True)

    # Apply the final limit if one was specified
    if not fetch_all:
        return all_albums[:limit]
        
    return all_albums

def get_tracks_from_album(album_id):
    """Fetches all audio tracks for a given album ID from Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"ParentId": album_id, "IncludeItemTypes": "Audio"}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        items = r.json().get("Items", [])
        
        # Apply artist field prioritization to each track
        for item in items:
            title = item.get('Name', 'Unknown')
            artist_name, artist_id = _select_best_artist(item, title)
            item['AlbumArtist'] = artist_name
            item['ArtistId'] = artist_id
        
        return items
    except Exception as e:
        logger.error(f"Jellyfin get_tracks_from_album failed for album {album_id}: {e}", exc_info=True)
        return []

def download_track(temp_dir, item):
    """Downloads a single track from Jellyfin using admin credentials."""
    try:
        track_id = item['Id']
        
        # Try to get format from Container field first (most reliable)
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
        
        download_url = f"{config.JELLYFIN_URL}/Items/{track_id}/Download"
        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")
        with requests.get(download_url, headers=config.HEADERS, stream=True, timeout=REQUESTS_TIMEOUT) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        logger.info(f"Downloaded '{item['Name']}' to '{local_filename}'")
        return local_filename
    except Exception as e:
        logger.error(f"Failed to download track {item.get('Name', 'Unknown')}: {e}", exc_info=True)
        return None

def _select_best_artist(item, title="Unknown"):
    """
    Selects the best artist field from Jellyfin item, prioritizing track artists over album artists.
    This helps avoid "Various Artists" issues in compilation albums.
    Returns tuple: (artist_name, artist_id)
    """
    # Priority: Artists array (track artists) > AlbumArtist > fallback
    # Jellyfin/Emby provides ArtistItems array with Id and Name
    if item.get('ArtistItems') and len(item['ArtistItems']) > 0:
        track_artist = item['ArtistItems'][0].get('Name', 'Unknown Artist')
        artist_id = item['ArtistItems'][0].get('Id')
        used_field = 'ArtistItems[0]'
    elif item.get('Artists') and len(item['Artists']) > 0:
        track_artist = item['Artists'][0]  # Take first artist if multiple
        artist_id = None
        used_field = 'Artists[0]'
    elif item.get('AlbumArtist'):
        track_artist = item['AlbumArtist']
        artist_id = None
        used_field = 'AlbumArtist'
    else:
        track_artist = 'Unknown Artist'
        artist_id = None
        used_field = 'fallback'
    
    return track_artist, artist_id

def get_all_songs():
    """Fetches all songs from Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "Audio", "Recursive": True}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        items = r.json().get("Items", [])
        
        # Apply artist field prioritization to each item
        for item in items:
            title = item.get('Name', 'Unknown')
            artist_name, artist_id = _select_best_artist(item, title)
            item['AlbumArtist'] = artist_name
            item['ArtistId'] = artist_id
        
        return items
    except Exception as e:
        logger.error(f"Jellyfin get_all_songs failed: {e}", exc_info=True)
        return []

def get_playlist_by_name(playlist_name):
    """Finds a Jellyfin playlist by its exact name using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True, "Name": playlist_name}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        playlists = r.json().get("Items", [])
        return playlists[0] if playlists else None
    except Exception as e:
        logger.error(f"Jellyfin get_playlist_by_name failed for '{playlist_name}': {e}", exc_info=True)
        return None

def create_playlist(base_name, item_ids):
    """Creates a new playlist on Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Playlists"
    body = {"Name": base_name, "Ids": item_ids, "UserId": config.JELLYFIN_USER_ID}
    try:
        r = requests.post(url, headers=config.HEADERS, json=body, timeout=REQUESTS_TIMEOUT)
        if r.ok: logger.info("✅ Created Jellyfin playlist '%s'", base_name)
    except Exception as e:
        logger.error("Exception creating Jellyfin playlist '%s': %s", base_name, e, exc_info=True)

def get_all_playlists():
    """Fetches all playlists from Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(f"Jellyfin get_all_playlists failed: {e}", exc_info=True)
        return []

def delete_playlist(playlist_id):
    """Deletes a playlist on Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Items/{playlist_id}"
    try:
        r = requests.delete(url, headers=config.HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Exception deleting Jellyfin playlist ID {playlist_id}: {e}", exc_info=True)
        return False

# --- USER-SPECIFIC JELLYFIN FUNCTIONS ---
def get_top_played_songs(limit, user_creds=None):
    """Fetches the top N most played songs from Jellyfin for a specific user."""
    user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
    token = user_creds.get('token') if user_creds else config.JELLYFIN_TOKEN
    if not user_id or not token: raise ValueError("Jellyfin User ID and Token are required.")

    url = f"{config.JELLYFIN_URL}/Users/{user_id}/Items"
    headers = {"X-Emby-Token": token}
    params = {"IncludeItemTypes": "Audio", "SortBy": "PlayCount", "SortOrder": "Descending", "Recursive": True, "Limit": limit, "Fields": "UserData,Path"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        items = r.json().get("Items", [])
        
        # Apply artist field prioritization to each item
        for item in items:
            title = item.get('Name', 'Unknown')
            artist_name, artist_id = _select_best_artist(item, title)
            item['AlbumArtist'] = artist_name
            item['ArtistId'] = artist_id
        
        return items
    except Exception as e:
        logger.error(f"Jellyfin get_all_songs failed: {e}", exc_info=True)
        return []

def get_last_played_time(item_id, user_creds=None):
    """Fetches the last played time for a specific track from Jellyfin for a specific user."""
    user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
    token = user_creds.get('token') if user_creds else config.JELLYFIN_TOKEN
    if not user_id or not token: raise ValueError("Jellyfin User ID and Token are required.")

    url = f"{config.JELLYFIN_URL}/Users/{user_id}/Items/{item_id}"
    headers = {"X-Emby-Token": token}
    params = {"Fields": "UserData"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("UserData", {}).get("LastPlayedDate")
    except Exception as e:
        logger.error(f"Jellyfin get_last_played_time failed for item {item_id}, user {user_id}: {e}", exc_info=True)
        return None

def create_instant_playlist(playlist_name, item_ids, user_creds=None):
    """Creates a new instant playlist on Jellyfin for a specific user."""
    # Treat empty token ("") as not provided and fall back to admin token from config
    token = config.JELLYFIN_TOKEN
    if user_creds and isinstance(user_creds, dict) and user_creds.get('token'):
        token = user_creds.get('token')
    if not token:
        # No token available even after fallback
        raise ValueError("Jellyfin Token is required.")

    # Treat empty user_identifier as not provided and fall back to admin user id
    identifier = config.JELLYFIN_USER_ID
    if user_creds and isinstance(user_creds, dict) and user_creds.get('user_identifier'):
        identifier = user_creds.get('user_identifier')
    if not identifier:
        raise ValueError("Jellyfin User Identifier is required.")

    user_id = resolve_user(identifier, token)
    
    final_playlist_name = f"{playlist_name.strip()}_instant"
    url = f"{config.JELLYFIN_URL}/Playlists"
    headers = {"X-Emby-Token": token}
    body = {"Name": final_playlist_name, "Ids": item_ids, "UserId": user_id}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Exception creating Jellyfin instant playlist '%s' for user %s: %s", playlist_name, user_id, e, exc_info=True)
        return None

