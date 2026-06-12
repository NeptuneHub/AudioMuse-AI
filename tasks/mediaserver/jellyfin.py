# tasks/mediaserver/jellyfin.py

from . import http as requests
import logging
import os
import config

from .helper import detect_path_format

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 300
JELLYFIN_PLAYLIST_BATCH_SIZE = 100

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


def list_libraries(user_creds=None):
    """List all music libraries exposed by a Jellyfin server.

    Unlike `_get_target_library_ids()`, this does NOT read `config.MUSIC_LIBRARIES`
    and does NOT filter — it returns every music library the server reports, so the
    UI can render a checkbox list. Accepts optional `user_creds` so the setup
    wizard test flow and the migration assistant can probe a target without
    mutating global config.
    """
    base_url = (user_creds.get('url') if user_creds and user_creds.get('url') else config.JELLYFIN_URL).rstrip('/')
    url = f"{base_url}/Library/VirtualFolders"
    try:
        r = requests.get(url, headers=_jellyfin_headers_from_creds(user_creds), timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        all_libraries = r.json() or []
        return [
            {'id': lib.get('ItemId'), 'name': lib.get('Name')}
            for lib in all_libraries
            if isinstance(lib, dict) and lib.get('CollectionType') == 'music' and lib.get('ItemId') and lib.get('Name')
        ]
    except Exception as e:
        logger.error(f"Jellyfin list_libraries failed at '{url}': {e}", exc_info=True)
        return []


def _jellyfin_base_url(user_creds=None):
    return (user_creds.get('url') if user_creds and user_creds.get('url') else config.JELLYFIN_URL).rstrip('/')


def _jellyfin_headers_from_creds(user_creds=None):
    headers = dict(getattr(config, 'HEADERS', {}) or {})
    token = user_creds.get('token') if user_creds else getattr(config, 'JELLYFIN_TOKEN', None)
    if token:
        headers['X-Emby-Token'] = token
    return headers


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
                albums_on_page = response_data.get("Items") or []
                
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
                    albums_on_page = response_data.get("Items") or []
                    
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

def get_tracks_from_album(album_id, user_creds=None):
    """Fetches all audio tracks for a given album ID from Jellyfin using admin or override credentials."""
    user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
    url = f"{_jellyfin_base_url(user_creds)}/Users/{user_id}/Items"
    params = {
        "ParentId": album_id,
        "IncludeItemTypes": "Audio",
        "Fields": "Path,ProductionYear,IndexNumber,ParentIndexNumber,AlbumArtist,Album,ArtistItems,Artists",
    }
    try:
        r = requests.get(url, headers=_jellyfin_headers_from_creds(user_creds), params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        items = r.json().get("Items") or []

        # Apply artist field prioritization to each track
        for item in items:
            item['OriginalAlbumArtist'] = item.get('AlbumArtist')
            title = item.get('Name', 'Unknown')
            artist_name, artist_id = _select_best_artist(item, title)
            item['AlbumArtist'] = artist_name
            item['ArtistId'] = artist_id
            item['Year'] = item.get('ProductionYear')
            item['FilePath'] = item.get('Path')

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

def get_all_songs(user_creds=None):
    """Fetches all songs from Jellyfin using admin or override credentials, paginated.

    Jellyfin 10.11.x scales poorly on a single unbounded ``/Items`` query: for a
    large library (tens of thousands of tracks) it exceeds the HTTP read timeout
    even at very high ``REQUESTS_TIMEOUT`` values (issue #523). Paginating keeps
    each request's server-side cost bounded.

    On a page failure this RAISES rather than returning a partial or empty list.
    The result feeds the migration matcher, where any score row missing from the
    returned set is deleted as an orphan by the execute step, so a silently
    truncated scan would destroy real analysis data. The migration probe routes
    already wrap this call and surface the error to the user.
    """
    user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
    url = f"{_jellyfin_base_url(user_creds)}/Users/{user_id}/Items"
    all_items = []
    start_index = 0
    limit = 500  # smaller than Emby's 1000 due to Jellyfin 10.11.x per-request DB cost

    while True:
        params = {
            "IncludeItemTypes": "Audio",
            "Recursive": True,
            "StartIndex": start_index,
            "Limit": limit,
            "Fields": "Path,ProductionYear,IndexNumber,ParentIndexNumber,AlbumArtist,Album,ArtistItems,Artists",
        }
        try:
            r = requests.get(url, headers=_jellyfin_headers_from_creds(user_creds), params=params, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
            items = r.json().get("Items") or []

            # Apply artist field prioritization to each item
            for item in items:
                item['OriginalAlbumArtist'] = item.get('AlbumArtist')
                title = item.get('Name', 'Unknown')
                artist_name, artist_id = _select_best_artist(item, title)
                item['AlbumArtist'] = artist_name
                item['ArtistId'] = artist_id
                item['Year'] = item.get('ProductionYear')
                item['FilePath'] = item.get('Path')

            all_items.extend(items)

            if len(items) < limit:
                # Last (short) page reached.
                break

            start_index += limit
        except Exception as e:
            logger.error(f"Jellyfin get_all_songs failed at index {start_index}: {e}", exc_info=True)
            raise

    return all_items


def search_albums(query, user_creds=None):
    """Search Jellyfin albums using admin or override credentials."""
    user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
    url = f"{_jellyfin_base_url(user_creds)}/Users/{user_id}/Items"
    params = {
        "IncludeItemTypes": "MusicAlbum",
        "Recursive": True,
        "SearchTerm": query,
        "Limit": 10,
        "Fields": "ChildCount,ProductionYear,AlbumArtist",
    }
    try:
        r = requests.get(url, headers=_jellyfin_headers_from_creds(user_creds), params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        items = r.json().get("Items") or []
        return [
            {
                'id':          item.get('Id'),
                'name':        item.get('Name'),
                'artist':      item.get('AlbumArtist'),
                'year':        item.get('ProductionYear'),
                'track_count': item.get('ChildCount'),
            }
            for item in items
        ]
    except Exception as e:
        logger.error(f"Jellyfin search_albums failed: {e}", exc_info=True)
        return []


def test_connection(user_creds=None):
    """Test Jellyfin connectivity using admin or override credentials."""
    try:
        user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
        url = f"{_jellyfin_base_url(user_creds)}/Users/{user_id}/Items"
        params = {
            "IncludeItemTypes": "Audio",
            "Recursive": True,
            "Fields": "Path,ProductionYear,IndexNumber,ParentIndexNumber,AlbumArtist,Album,ArtistItems,Artists",
            "StartIndex": 0,
            "Limit": 100,
        }
        r = requests.get(url, headers=_jellyfin_headers_from_creds(user_creds), params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        items = r.json().get('Items', []) or []
        sample = []
        for item in items:
            track_artist, _ = _select_best_artist(item, item.get('Name', 'Unknown'))
            sample.append({
                'Id': item.get('Id'),
                'Path': item.get('Path'),
                'Name': item.get('Name'),
                'AlbumArtist': track_artist,
            })
        path_format = detect_path_format(sample)
        return {
            'ok': True,
            'error': None,
            'sample_count': len(sample),
            'path_format': path_format,
            'warnings': [],
        }
    except Exception as e:
        logger.warning(f"Jellyfin test_connection failed: {e}")
        return {'ok': False, 'error': str(e), 'sample_count': 0, 'path_format': 'none', 'warnings': []}


def get_playlist_by_name(playlist_name):
    """Finds a Jellyfin playlist by its exact name using admin credentials.

    Jellyfin's /Users/{userId}/Items endpoint silently ignores the Name query
    parameter and returns every playlist regardless of value, so we have to
    filter client-side by exact match (mirrors the Emby version).
    """
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        playlists = r.json().get("Items") or []
        for playlist in playlists:
            if playlist.get("Name") == playlist_name:
                return playlist
        return None
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
        return r.json().get("Items") or []
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
    params = {"IncludeItemTypes": "Audio", "SortBy": "PlayCount", "SortOrder": "Descending", "Recursive": True, "Limit": limit, "Fields": "UserData,Path,ProductionYear"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        items = r.json().get("Items") or []

        # Apply artist field prioritization to each item
        for item in items:
            item['OriginalAlbumArtist'] = item.get('AlbumArtist')
            title = item.get('Name', 'Unknown')
            artist_name, artist_id = _select_best_artist(item, title)
            item['AlbumArtist'] = artist_name
            item['ArtistId'] = artist_id
            item['Year'] = item.get('ProductionYear')
            item['FilePath'] = item.get('Path')

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

def get_lyrics(track_id: str, timeout: float = 2.5):
    """Fetch embedded lyrics from Jellyfin for a given track ID.

    Uses the Jellyfin Lyrics API (available since Jellyfin 10.8).
    Returns plain text (newline-separated lines) or None.
    """
    try:
        url = f"{config.JELLYFIN_URL}/Audio/{track_id}/Lyrics"
        r = requests.get(url, headers=config.HEADERS, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # Response: {"Lyrics": [{"Text": "line", "Start": 0}, ...]}
        lyrics_lines = data.get('Lyrics') or []
        if not lyrics_lines:
            return None
        text = '\n'.join(line.get('Text', '') for line in lyrics_lines if line.get('Text'))
        return text.strip() or None
    except Exception as exc:
        logger.debug('Jellyfin get_lyrics failed for %s: %s', track_id, exc)
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


def _get_playlist_entry_ids(playlist_id):
    """Fetches every PlaylistItemId for an existing Jellyfin playlist (admin creds).

    Each playlist *entry* has both the underlying audio item's ``Id`` and a separate
    ``PlaylistItemId``. Removal via ``DELETE /Playlists/{Id}/Items?entryIds=…`` requires
    the latter — passing the audio Id will silently no-op.
    """
    url = f"{config.JELLYFIN_URL}/Playlists/{playlist_id}/Items"
    params = {"UserId": config.JELLYFIN_USER_ID}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        items = r.json().get("Items") or []
        entry_ids = [it.get("PlaylistItemId") for it in items if it.get("PlaylistItemId")]
        if len(entry_ids) != len(items):
            logger.warning(
                f"Jellyfin _get_playlist_entry_ids: playlist {playlist_id} had "
                f"{len(items) - len(entry_ids)} items missing PlaylistItemId — they will not be removed"
            )
        return entry_ids
    except Exception as e:
        logger.error(f"Jellyfin _get_playlist_entry_ids failed for {playlist_id}: {e}", exc_info=True)
        return None


def _remove_playlist_entries(playlist_id, entry_ids):
    """DELETEs entries from a Jellyfin playlist in batches. Raises on HTTP failure;
    callers must wrap in try/except if they want to handle the failure (e.g. fall
    back to delete-and-recreate on Jellyfin < 10.11)."""
    if not entry_ids:
        return
    url = f"{config.JELLYFIN_URL}/Playlists/{playlist_id}/Items"
    for i in range(0, len(entry_ids), JELLYFIN_PLAYLIST_BATCH_SIZE):
        batch = entry_ids[i:i + JELLYFIN_PLAYLIST_BATCH_SIZE]
        params = {"entryIds": ",".join(batch)}
        r = requests.delete(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()


def _add_items_to_playlist(playlist_id, item_ids):
    """POSTs items to a Jellyfin playlist in batches (admin creds). Returns True on full success."""
    if not item_ids:
        return True
    url = f"{config.JELLYFIN_URL}/Playlists/{playlist_id}/Items"
    for i in range(0, len(item_ids), JELLYFIN_PLAYLIST_BATCH_SIZE):
        batch = item_ids[i:i + JELLYFIN_PLAYLIST_BATCH_SIZE]
        params = {"ids": ",".join(batch), "userId": config.JELLYFIN_USER_ID}
        try:
            r = requests.post(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
        except Exception as e:
            logger.error(
                f"Jellyfin _add_items_to_playlist: batch starting at {i} failed for playlist {playlist_id}: {e}",
                exc_info=True,
            )
            return False
    return True


def _create_fresh_playlist(playlist_name, item_ids):
    """POST a new Jellyfin playlist with ``item_ids``. Returns the playlist dict or None."""
    url = f"{config.JELLYFIN_URL}/Playlists"
    first_batch = item_ids[:JELLYFIN_PLAYLIST_BATCH_SIZE]
    rest = item_ids[JELLYFIN_PLAYLIST_BATCH_SIZE:]
    body = {"Name": playlist_name, "Ids": first_batch, "UserId": config.JELLYFIN_USER_ID}
    try:
        r = requests.post(url, headers=config.HEADERS, json=body, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        created = r.json()
    except Exception as e:
        logger.error(f"Jellyfin _create_fresh_playlist: create failed for '{playlist_name}': {e}", exc_info=True)
        return None

    new_id = created.get("Id")
    if not new_id:
        logger.error(f"Jellyfin _create_fresh_playlist: created '{playlist_name}' but response had no Id")
        return None

    if rest and not _add_items_to_playlist(new_id, rest):
        logger.error(f"Jellyfin _create_fresh_playlist: created '{playlist_name}' but failed to add overflow tracks")

    logger.info(f"✅ Jellyfin: created playlist '{playlist_name}' (Id={new_id}) with {len(item_ids)} tracks")
    return {**created, 'Id': new_id, 'Name': created.get('Name', playlist_name)}


def create_or_replace_playlist(playlist_name, item_ids, user_creds=None):
    """Cron-only upsert: create the playlist if missing, or replace its contents.

    Tries to preserve the playlist Id by clearing then repopulating in place. Falls back
    to deleting the whole playlist and recreating it when the in-place clear fails — this
    is the case on Jellyfin < 10.11 with API-token auth (jellyfin/jellyfin#13476, server
    fix shipped in 10.11.0). The fallback yields a new Id every cron tick; upgrade
    Jellyfin to 10.11+ to keep stable Ids.

    Uses admin credentials. ``user_creds`` is accepted for dispatcher signature parity but
    not currently used (cron always runs as admin). Returns the playlist dict (with 'Id'/'Name')
    or None on failure.
    """
    if not item_ids:
        return None

    existing = get_playlist_by_name(playlist_name)
    if not existing:
        return _create_fresh_playlist(playlist_name, item_ids)

    playlist_id = existing.get("Id")
    if not playlist_id:
        logger.error(f"Jellyfin create_or_replace_playlist: existing playlist '{playlist_name}' has no Id")
        return None

    entry_ids = _get_playlist_entry_ids(playlist_id)
    if entry_ids is None:
        return None

    try:
        _remove_playlist_entries(playlist_id, entry_ids)
    except Exception:
        logger.info(
            f"Reuse of existing playlist '{playlist_name}' not supported from the Music Server, going to create a new one."
        )
        if not delete_playlist(playlist_id):
            logger.error(
                f"Jellyfin: failed to delete playlist '{playlist_name}' (Id={playlist_id}) for fallback recreate"
            )
            return None
        return _create_fresh_playlist(playlist_name, item_ids)

    if not _add_items_to_playlist(playlist_id, item_ids):
        # Items were already cleared above; signal failure so the cron handler doesn't log success.
        logger.error(f"Jellyfin create_or_replace_playlist: failed to add tracks to playlist {playlist_id}")
        return None

    logger.info(f"✅ Jellyfin: replaced contents of playlist '{playlist_name}' (Id={playlist_id}, tracks={len(item_ids)})")
    return {**existing, 'Id': playlist_id, 'Name': existing.get('Name', playlist_name)}

