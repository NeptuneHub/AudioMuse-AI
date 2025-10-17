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
    Fetches recent albums from Jellyfin, aligned with other media servers behavior:
    - limit = 0: Returns ALL albums + standalone tracks (comprehensive discovery)
    - limit > 0: Returns ONLY real albums (no standalone tracks)
    
    This matches Navidrome and Lyrion behavior where specific limits focus on albums only.
    """
    if limit == 0:
        # Special case: limit=0 means get everything (albums + standalone tracks)
        return get_recent_music_items(limit)
    else:
        # Normal case: get only real albums, no standalone tracks
        return _get_recent_albums_only(limit)

def get_comprehensive_music_discovery(limit=0):
    """
    Convenience function for comprehensive music discovery including standalone tracks.
    Always returns both albums and standalone tracks as pseudo-albums.
    Use this when you want to ensure no music is missed, regardless of metadata completeness.
    """
    return get_recent_music_items(limit)

def _get_recent_standalone_tracks(limit, target_library_ids=None):
    """
    Fetches recent standalone audio tracks that are not properly organized in albums.
    This captures orphaned tracks, loose files, and tracks with missing album metadata.
    """
    if target_library_ids is not None and isinstance(target_library_ids, set) and not target_library_ids:
        logger.info("Library filtering is active but no matching libraries found. Skipping standalone tracks.")
        return []

    all_tracks = []
    fetch_all = (limit == 0)

    # Case 1: No library filtering - scan all libraries
    if target_library_ids is None:
        logger.info("Scanning all Jellyfin libraries for recent standalone tracks.")
        start_index = 0
        page_size = 500
        while True:
            url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
            params = {
                "IncludeItemTypes": "Audio", "SortBy": "DateCreated", "SortOrder": "Descending",
                "Recursive": True, "Limit": page_size, "StartIndex": start_index,
                "Fields": "ParentId,Path,DateCreated"  # Include fields to check album relationship
            }
            try:
                r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
                r.raise_for_status()
                response_data = r.json()
                tracks_on_page = response_data.get("Items", [])
                
                if not tracks_on_page:
                    break

                # Filter for tracks that don't have a proper album parent
                standalone_tracks = []
                for track in tracks_on_page:
                    # Check if track has a proper album parent by trying to get parent info
                    parent_id = track.get('ParentId')
                    if not parent_id:
                        # No parent - definitely standalone
                        standalone_tracks.append(track)
                    else:
                        # Check if parent is actually an album (not just a folder)
                        try:
                            parent_url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items/{parent_id}"
                            parent_r = requests.get(parent_url, headers=config.HEADERS, timeout=REQUESTS_TIMEOUT)
                            if parent_r.ok:
                                parent_info = parent_r.json()
                                # If parent is not a MusicAlbum, treat track as standalone
                                if parent_info.get('Type') != 'MusicAlbum':
                                    standalone_tracks.append(track)
                        except:
                            # If we can't check parent, assume it's standalone to be safe
                            standalone_tracks.append(track)

                all_tracks.extend(standalone_tracks)
                start_index += len(tracks_on_page)
                
                if not fetch_all and len(all_tracks) >= limit:
                    all_tracks = all_tracks[:limit]
                    break

                if len(tracks_on_page) < page_size:
                    break
            except Exception as e:
                logger.error(f"Jellyfin get_recent_standalone_tracks failed: {e}", exc_info=True)
                break

    # Case 2: Library filtering - scan specific libraries
    else:
        logger.info(f"Scanning {len(target_library_ids)} specific Jellyfin libraries for recent standalone tracks.")
        for library_id in target_library_ids:
            start_index = 0
            page_size = 500
            while True:
                url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
                params = {
                    "IncludeItemTypes": "Audio", "SortBy": "DateCreated", "SortOrder": "Descending",
                    "Recursive": True, "Limit": page_size, "StartIndex": start_index,
                    "ParentId": library_id, "Fields": "ParentId,Path,DateCreated"
                }
                try:
                    r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
                    r.raise_for_status()
                    response_data = r.json()
                    tracks_on_page = response_data.get("Items", [])
                    
                    if not tracks_on_page:
                        break

                    # Apply same standalone filtering logic
                    standalone_tracks = []
                    for track in tracks_on_page:
                        parent_id = track.get('ParentId')
                        if not parent_id or parent_id == library_id:
                            # No parent or parent is the library itself - standalone
                            standalone_tracks.append(track)
                        else:
                            # Check if parent is actually an album
                            try:
                                parent_url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items/{parent_id}"
                                parent_r = requests.get(parent_url, headers=config.HEADERS, timeout=REQUESTS_TIMEOUT)
                                if parent_r.ok:
                                    parent_info = parent_r.json()
                                    if parent_info.get('Type') != 'MusicAlbum':
                                        standalone_tracks.append(track)
                            except:
                                standalone_tracks.append(track)

                    all_tracks.extend(standalone_tracks)
                    start_index += len(tracks_on_page)
                    
                    if not fetch_all and len(all_tracks) >= limit:
                        all_tracks = all_tracks[:limit]
                        break

                    if len(tracks_on_page) < page_size:
                        break
                except Exception as e:
                    logger.error(f"Jellyfin get_recent_standalone_tracks failed for library ID {library_id}: {e}", exc_info=True)
                    break

    # Apply artist field prioritization to standalone tracks
    for track in all_tracks:
        title = track.get('Name', 'Unknown')
        track['AlbumArtist'] = _select_best_artist(track, title)

    if all_tracks:
        logger.info(f"Found {len(all_tracks)} recent standalone tracks (not in albums)")
    
    return all_tracks

def _get_recent_albums_only(limit):
    """
    Original implementation: Fetches ONLY albums from Jellyfin (no standalone tracks).
    This is kept as a separate function in case the original behavior is needed.
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
        logger.info("Scanning all Jellyfin libraries for recent albums (albums only).")
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
                logger.error(f"Jellyfin _get_recent_albums_only failed during 'scan all': {e}", exc_info=True)
                break
    
    # Case 3: Config is set and we have library IDs. Scan each of these libraries by using their ID as ParentId.
    else:
        logger.info(f"Scanning {len(target_library_ids)} specific Jellyfin libraries for recent albums (albums only).")
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
                    logger.error(f"Jellyfin _get_recent_albums_only failed for library ID {library_id}: {e}", exc_info=True)
                    break

    # After fetching, a final sort and trim is needed only if we fetched from multiple libraries.
    if target_library_ids is not None and len(target_library_ids) > 1:
        all_albums.sort(key=lambda x: x.get('DateCreated', ''), reverse=True)

    # Apply the final limit if one was specified
    if not fetch_all:
        return all_albums[:limit]
        
    return all_albums

def get_recent_music_items(limit):
    """
    Gets both recent albums AND recent standalone tracks that aren't properly organized in albums.
    This ensures no music is missed during analysis, even if metadata is incomplete.
    Returns a list combining album objects and standalone track objects.
    """
    target_library_ids = _get_target_library_ids()
    
    # Get recent albums (existing functionality)
    albums = _get_recent_albums_only(limit)
    
    # Get recent standalone tracks (new functionality) 
    # Use the same limit to get a reasonable number of standalone tracks
    standalone_limit = min(limit, 100) if limit > 0 else 100  # Cap standalone tracks at 100
    standalone_tracks = _get_recent_standalone_tracks(standalone_limit, target_library_ids)
    
    # Create pseudo-albums for standalone tracks to maintain compatibility with analysis workflow
    pseudo_albums = []
    for track in standalone_tracks:
        # Create a pseudo-album containing just this one track
        pseudo_album = {
            'Id': f"standalone_{track['Id']}",  # Unique pseudo-album ID
            'Name': f"Standalone: {track.get('Name', 'Unknown')}",
            'Type': 'PseudoAlbum',  # Mark as pseudo-album
            'StandaloneTrack': track,  # Embed the track data
            'DateCreated': track.get('DateCreated', ''),
            'AlbumArtist': track.get('AlbumArtist', 'Unknown Artist')
        }
        pseudo_albums.append(pseudo_album)
    
    # Combine albums and pseudo-albums
    all_items = albums + pseudo_albums
    
    # Sort by date if we have multiple sources
    if albums and pseudo_albums:
        all_items.sort(key=lambda x: x.get('DateCreated', ''), reverse=True)
    
    # Apply final limit if specified
    if limit > 0:
        all_items = all_items[:limit]
    
    if pseudo_albums:
        logger.info(f"Found {len(albums)} regular albums and {len(pseudo_albums)} standalone tracks (combined into {len(all_items)} total items)")
    
    return all_items

def get_tracks_from_album(album_id):
    """Fetches all audio tracks for a given album ID from Jellyfin using admin credentials."""
    # Check if this is a pseudo-album for a standalone track
    if str(album_id).startswith('standalone_'):
        # Extract the real track ID from the pseudo-album ID
        real_track_id = album_id.replace('standalone_', '')
        
        # Get the track directly by its ID
        url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items/{real_track_id}"
        try:
            r = requests.get(url, headers=config.HEADERS, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
            track_item = r.json()
            
            # Apply artist field prioritization
            title = track_item.get('Name', 'Unknown')
            track_item['AlbumArtist'] = _select_best_artist(track_item, title)
            
            return [track_item]  # Return as single-item list to maintain compatibility
        except Exception as e:
            logger.error(f"Jellyfin get_tracks_from_album failed for standalone track {real_track_id}: {e}", exc_info=True)
            return []
    
    # Normal album handling
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"ParentId": album_id, "IncludeItemTypes": "Audio"}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        items = r.json().get("Items", [])
        
        # Apply artist field prioritization to each track
        for item in items:
            title = item.get('Name', 'Unknown')
            item['AlbumArtist'] = _select_best_artist(item, title)
        
        return items
    except Exception as e:
        logger.error(f"Jellyfin get_tracks_from_album failed for album {album_id}: {e}", exc_info=True)
        return []

def download_track(temp_dir, item):
    """Downloads a single track from Jellyfin using admin credentials."""
    try:
        track_id = item['Id']
        file_extension = os.path.splitext(item.get('Path', ''))[1] or '.tmp'
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
    """
    # Priority: Artists array (track artists) > AlbumArtist > fallback
    if item.get('Artists') and len(item['Artists']) > 0:
        track_artist = item['Artists'][0]  # Take first artist if multiple
        used_field = 'Artists[0]'
    elif item.get('AlbumArtist'):
        track_artist = item['AlbumArtist']
        used_field = 'AlbumArtist'
    else:
        track_artist = 'Unknown Artist'
        used_field = 'fallback'
    
    return track_artist

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
            item['AlbumArtist'] = _select_best_artist(item, title)
        
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
        
        # Apply artist field prioritization to each track
        for item in items:
            title = item.get('Name', 'Unknown')
            item['AlbumArtist'] = _select_best_artist(item, title)
        
        return items
    except Exception as e:
        logger.error(f"Jellyfin get_top_played_songs failed for user {user_id}: {e}", exc_info=True)
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
    token = user_creds.get('token') if user_creds else config.JELLYFIN_TOKEN
    if not token: raise ValueError("Jellyfin Token is required.")
    
    identifier = user_creds.get('user_identifier') if user_creds else config.JELLYFIN_USER_ID
    if not identifier: raise ValueError("Jellyfin User Identifier is required.")

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

