# tasks/mediaserver_navidrome.py

import requests
import logging
import os
import random
import config

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 300
NAVIDROME_API_BATCH_SIZE = 40

# Global cache to store pseudo-album tracks for retrieval by get_tracks_from_album
_pseudo_album_cache = {}


def _is_standalone_track(song_item):
    """Returns True when the Navidrome song lacks album metadata entirely."""
    album_id = song_item.get('albumId') or song_item.get('album_id')
    if album_id:
        return False
    return True

# ##############################################################################
# NAVIDROME (SUBSONIC API) IMPLEMENTATION
# ##############################################################################

def _get_target_music_folder_ids():
    """
    Parses config for music folder names and returns their IDs for filtering using a robust,
    case-insensitive matching against the server's actual folder configuration.
    """
    folder_names_str = getattr(config, 'MUSIC_LIBRARIES', '')

    if not folder_names_str.strip():
        return None

    target_names_lower = {name.strip().lower() for name in folder_names_str.split(',') if name.strip()}

    # Use the getMusicFolders endpoint to get the available music folders.
    response = _navidrome_request("getMusicFolders")
    
    if not (response and "musicFolders" in response and "musicFolder" in response["musicFolders"]):
        logger.error("Failed to fetch music folders from Navidrome or response format unexpected.")
        return set()

    all_folders = response["musicFolders"]["musicFolder"]

    # Build a case-insensitive map: lowercase_name -> {'name': OriginalCaseName, 'id': FolderId}
    folder_map = {
        folder['name'].lower(): {'name': folder['name'], 'id': folder['id']}
        for folder in all_folders
        if isinstance(folder, dict) and 'name' in folder and 'id' in folder
    }

    # --- DIAGNOSTIC LOGGING ---
    available_music_folders = [folder['name'] for folder in folder_map.values()]
    logger.info(f"Available Navidrome music folders found: {available_music_folders}")
    # --- END DIAGNOSTIC LOGGING ---

    # Match user's config against the map to find IDs and original names
    found_folders = []
    unfound_names = []
    for target_name in target_names_lower:
        if target_name in folder_map:
            found_folders.append(folder_map[target_name])
        else:
            unfound_names.append(target_name)

    if unfound_names:
        logger.warning(f"Navidrome config specified folder names that were not found: {list(unfound_names)}")

    if not found_folders:
        logger.warning(f"No matching music folders found for configured names: {list(target_names_lower)}. No albums will be analyzed.")
        return set()

    music_folder_ids = {folder['id'] for folder in found_folders}
    found_names_original_case = [folder['name'] for folder in found_folders]

    logger.info(f"Filtering analysis to {len(music_folder_ids)} Navidrome folders: {found_names_original_case}")
    return music_folder_ids

def get_navidrome_auth_params(username=None, password=None):
    """Generates Navidrome auth params, using provided creds or falling back to global config."""
    auth_user = username or config.NAVIDROME_USER
    auth_pass = password or config.NAVIDROME_PASSWORD
    if not auth_user or not auth_pass: 
        logger.warning("Navidrome User or Password is not configured.")
        return {}
    hex_encoded_password = auth_pass.encode('utf-8').hex()
    return {"u": auth_user, "p": f"enc:{hex_encoded_password}", "v": "1.16.1", "c": config.APP_VERSION, "f": "json"}

def _navidrome_request(endpoint, params=None, method='get', stream=False, user_creds=None):
    """
    Helper to make Navidrome API requests. It sends all parameters in the URL's
    query string, which is the expected behavior for Subsonic APIs, but can cause
    issues with very long parameter lists (e.g., creating large playlists).
    """
    params = params or {}
    auth_params = get_navidrome_auth_params(
        username=user_creds.get('user') if user_creds else None,
        password=user_creds.get('password') if user_creds else None
    )
    if not auth_params:
        logger.error("Navidrome credentials not configured. Cannot make API call.")
        return None

    url = f"{config.NAVIDROME_URL}/rest/{endpoint}.view"
    all_params = {**auth_params, **params}

    try:
        r = requests.request(method, url, params=all_params, timeout=REQUESTS_TIMEOUT, stream=stream)
        r.raise_for_status()

        if stream:
            return r
            
        subsonic_response = r.json().get("subsonic-response", {})
        if subsonic_response.get("status") == "failed":
            error = subsonic_response.get("error", {})
            logger.error(f"Navidrome API Error on '{endpoint}': {error.get('message')}")
            return None
        return subsonic_response
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Navidrome API endpoint '{endpoint}': {e}", exc_info=True)
        return None

def download_track(temp_dir, item):
    """Downloads a single track from Navidrome using admin credentials."""
    try:
        track_id = item['id'] 
        file_extension = os.path.splitext(item.get('path', ''))[1] or '.tmp'
        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")
        
        response = _navidrome_request("stream", params={"id": track_id}, stream=True)
        if response:
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded '{item.get('title', 'Unknown')}' to '{local_filename}'")
            return local_filename
    except Exception as e:
        logger.error(f"Failed to download Navidrome track {item.get('title', 'Unknown')}: {e}", exc_info=True)
    return None

def get_recent_albums(limit):
    """
    Fetches recent albums from Navidrome, with comprehensive discovery when limit=0:
    - limit = 0: Returns ALL albums + standalone tracks (comprehensive discovery)
    - limit > 0: Returns ONLY real albums (no standalone tracks)
    
    This ensures consistent behavior across all media servers.
    """
    if limit == 0:
        # Special case: limit=0 means get everything (albums + standalone tracks)
        return get_recent_music_items(limit)
    else:
        # Normal case: get only real albums, no standalone tracks
        return _get_recent_albums_only(limit)

def _get_recent_albums_only(limit):
    """
    Original implementation: Fetches ONLY albums from Navidrome (no standalone tracks).
    This is kept as a separate function for when album-only behavior is needed.
    """
    target_folder_ids = _get_target_music_folder_ids()
    
    # Case 1: Config is set, but no matching folders were found. Scan nothing.
    if isinstance(target_folder_ids, set) and not target_folder_ids:
        logger.warning("Folder filtering is active, but no matching folders were found on the server. Returning no albums.")
        return []

    all_albums = []
    fetch_all = (limit == 0)

    # Case 2: Config is NOT set (is None). Scan all albums without musicFolderId filter.
    if target_folder_ids is None:
        logger.info("Scanning all Navidrome music folders for recent albums.")
        offset = 0
        page_size = 500
        while True:
            size_to_fetch = page_size if fetch_all else min(page_size, limit - len(all_albums))
            if size_to_fetch <= 0: break

            params = {"type": "newest", "size": size_to_fetch, "offset": offset}
            response = _navidrome_request("getAlbumList2", params)

            if response and "albumList2" in response and "album" in response["albumList2"]:
                albums = response["albumList2"]["album"]
                if not albums: break 

                all_albums.extend([{**a, 'Id': a.get('id'), 'Name': a.get('name')} for a in albums])
                offset += len(albums)

                if len(albums) < size_to_fetch: break
            else:
                logger.error("Failed to fetch recent albums page from Navidrome.")
                break

    # Case 3: Config is set and we have folder IDs. Scan each of these folders by using musicFolderId.
    else:
        logger.info(f"Scanning {len(target_folder_ids)} specific Navidrome music folders for recent albums.")
        for folder_id in target_folder_ids:
            offset = 0
            page_size = 500
            while True: # Paginate through the current folder
                size_to_fetch = page_size if fetch_all else min(page_size, limit - len(all_albums))
                if size_to_fetch <= 0: break

                params = {"type": "newest", "size": size_to_fetch, "offset": offset, "musicFolderId": folder_id}
                response = _navidrome_request("getAlbumList2", params)

                if response and "albumList2" in response and "album" in response["albumList2"]:
                    albums = response["albumList2"]["album"]
                    if not albums: break 

                    all_albums.extend([{**a, 'Id': a.get('id'), 'Name': a.get('name')} for a in albums])
                    offset += len(albums)

                    if len(albums) < size_to_fetch: break
                else:
                    logger.error(f"Failed to fetch recent albums page from Navidrome folder ID {folder_id}.")
                    break

    # After fetching, a final sort and trim is needed only if we fetched from multiple folders.
    if target_folder_ids is not None and len(target_folder_ids) > 1:
        # Sort by newest first (assuming albums have a 'created' or similar field)
        # Note: Navidrome album objects may not have a direct creation date field in the API response
        # The albums should already be sorted by the API, but we ensure consistency
        pass  # Albums from getAlbumList2 with type="newest" should already be properly sorted

    # Apply the final limit if one was specified
    if not fetch_all:
        return all_albums[:limit]
        
    return all_albums

def get_recent_music_items(limit):
    """
    Gets both recent albums AND recent standalone tracks for comprehensive discovery.
    This ensures no music is missed during analysis, even if metadata is incomplete.
    Returns a list combining album objects and standalone track objects.
    """
    _pseudo_album_cache.clear()
    target_folder_ids = _get_target_music_folder_ids()
    
    # Get recent albums (existing functionality)
    albums = _get_recent_albums_only(limit)
    
    # Get recent standalone tracks
    standalone_tracks = _get_recent_standalone_tracks(limit, target_folder_ids)
    
    # Combine and sort by date
    all_items = albums + standalone_tracks
    all_items.sort(key=lambda x: x.get('created', x.get('DateCreated', '')), reverse=True)
    
    # Apply final limit if specified
    if limit > 0:
        return all_items[:limit]
    
    return all_items

def _get_recent_standalone_tracks(limit, target_folder_ids=None):
    """
    Fetches recent standalone audio tracks that might not be properly organized in albums.
    This captures orphaned tracks, loose files, and tracks with missing album metadata.
    """
    if target_folder_ids is not None and isinstance(target_folder_ids, set) and not target_folder_ids:
        logger.info("Folder filtering is active but no matching folders found. Skipping standalone tracks.")
        return []

    logger.info("Fetching recent standalone tracks from Navidrome...")
    
    all_tracks = []
    fetch_all = (limit == 0)
    
    try:
        # Get recent songs using getSongList
        if target_folder_ids is None:
            # No folder filtering
            offset = 0
            page_size = 500
            while True:
                size_to_fetch = page_size if fetch_all else min(page_size, limit - len(all_tracks))
                if size_to_fetch <= 0:
                    break
                    
                params = {"type": "newest", "size": size_to_fetch, "offset": offset}
                response = _navidrome_request("getSongList", params)
                
                if response and "songList" in response and "song" in response["songList"]:
                    songs = response["songList"]["song"]
                    if not songs:
                        break

                    standalone_songs = [s for s in songs if _is_standalone_track(s)]
                    if not standalone_songs:
                        offset += len(songs)
                        if len(songs) < size_to_fetch:
                            break
                        continue

                    # Group standalone tracks by artist into pseudo-albums
                    pseudo_albums = _group_tracks_into_pseudo_albums(standalone_songs)
                    all_tracks.extend(pseudo_albums)
                    
                    offset += len(songs)
                    if len(songs) < size_to_fetch:
                        break
                else:
                    break
        else:
            # With folder filtering
            for folder_id in target_folder_ids:
                offset = 0
                page_size = 500
                while True:
                    size_to_fetch = page_size if fetch_all else min(page_size, limit - len(all_tracks))
                    if size_to_fetch <= 0:
                        break
                        
                    params = {"type": "newest", "size": size_to_fetch, "offset": offset, "musicFolderId": folder_id}
                    response = _navidrome_request("getSongList", params)
                    
                    if response and "songList" in response and "song" in response["songList"]:
                        songs = response["songList"]["song"]
                        if not songs:
                            break

                        standalone_songs = [s for s in songs if _is_standalone_track(s)]
                        if not standalone_songs:
                            offset += len(songs)
                            if len(songs) < size_to_fetch:
                                break
                            continue

                        # Group standalone tracks by artist into pseudo-albums
                        pseudo_albums = _group_tracks_into_pseudo_albums(standalone_songs)
                        all_tracks.extend(pseudo_albums)
                        
                        offset += len(songs)
                        if len(songs) < size_to_fetch:
                            break
                    else:
                        break
                        
    except Exception as e:
        logger.error(f"Failed to fetch standalone tracks from Navidrome: {e}", exc_info=True)
    
    return all_tracks

def _group_tracks_into_pseudo_albums(songs):
    """
    Groups standalone tracks by artist into pseudo-albums for compatibility with album-based workflow.
    """
    artist_groups = {}
    
    for song in songs:
        artist = _select_best_artist(song, song.get('title', 'Unknown'))
        
        if artist not in artist_groups:
            pseudo_id = f"standalone-{artist.replace(' ', '-').lower()}"
            artist_groups[artist] = {
                'Id': pseudo_id,
                'Name': f"🎵 Standalone Tracks - {artist}",
                'created': song.get('created', ''),
                'DateCreated': song.get('created', ''),
                'artist': artist,
                'artistId': song.get('artistId', ''),
                'tracks': []
            }
        
        artist_groups[artist]['tracks'].append(song)
        
        # Use the most recent track's date for the pseudo-album
        if song.get('created', '') > artist_groups[artist]['created']:
            artist_groups[artist]['created'] = song.get('created', '')
            artist_groups[artist]['DateCreated'] = song.get('created', '')
    
    # Store tracks in global cache for retrieval by get_tracks_from_album
    for artist, album_data in artist_groups.items():
        _pseudo_album_cache[album_data['Id']] = album_data['tracks']
    
    return list(artist_groups.values())

def get_comprehensive_music_discovery(limit=0):
    """
    Convenience function for comprehensive music discovery including standalone tracks.
    Always returns both albums and standalone tracks as pseudo-albums.
    """
    return get_recent_music_items(limit)

def _select_best_artist(song_item, title="Unknown"):
    """
    Selects the best artist field from Navidrome song item, prioritizing track artists over album artists.
    This helps avoid "Various Artists" issues in compilation albums.
    """
    # Priority: artist (track artist) > albumArtist > fallback
    if song_item.get('artist'):
        track_artist = song_item['artist']
        used_field = 'artist'
    elif song_item.get('albumArtist'):
        track_artist = song_item['albumArtist']
        used_field = 'albumArtist'
    else:
        track_artist = 'Unknown Artist'
        used_field = 'fallback'
    
    return track_artist

def get_all_songs():
    """
    Fetches all songs from Navidrome using admin credentials.
    If MUSIC_LIBRARIES is set, it will only return songs from those folders.
    """
    target_folder_ids = _get_target_music_folder_ids()
    
    # Case 1: Config is set, but no matching folders were found. Return no songs.
    if isinstance(target_folder_ids, set) and not target_folder_ids:
        logger.warning("Folder filtering is active, but no matching folders were found on the server. Returning no songs.")
        return []

    all_songs = []
    
    # Case 2: Config is NOT set (is None). Scan all songs without folder filter.
    if target_folder_ids is None:
        logger.info("Fetching all songs from all Navidrome music folders.")
        offset = 0
        limit = 500
        while True:
            params = {"query": '', "songCount": limit, "songOffset": offset}
            response = _navidrome_request("search3", params)
            if response and "searchResult3" in response and "song" in response["searchResult3"]:
                songs = response["searchResult3"]["song"]
                if not songs: break
                
                # Apply artist field prioritization to each song
                for s in songs:
                    title = s.get('title', 'Unknown')
                    artist = _select_best_artist(s, title)
                    all_songs.append({
                        'Id': s.get('id'), 
                        'Name': title, 
                        'AlbumArtist': artist, 
                        'Path': s.get('path')
                    })
                
                offset += len(songs)
                if len(songs) < limit: break
            else:
                logger.error("Failed to fetch all songs from Navidrome.")
                break

    # Case 3: Config is set and we have folder IDs. Get albums from folders, then songs from albums.
    else:
        logger.info(f"Fetching songs from {len(target_folder_ids)} specific Navidrome music folders.")
        
        # First, get all albums from the specified folders
        target_albums = []
        for folder_id in target_folder_ids:
            offset = 0
            page_size = 500
            while True:
                params = {"type": "newest", "size": page_size, "offset": offset, "musicFolderId": folder_id}
                response = _navidrome_request("getAlbumList2", params)
                
                if response and "albumList2" in response and "album" in response["albumList2"]:
                    albums = response["albumList2"]["album"]
                    if not albums: break
                    
                    target_albums.extend(albums)
                    offset += len(albums)
                    
                    if len(albums) < page_size: break
                else:
                    logger.error(f"Failed to fetch albums from Navidrome folder ID {folder_id}.")
                    break
        
        logger.info(f"Found {len(target_albums)} albums in specified folders. Getting songs from these albums.")
        
        # Now get songs from each album
        for album in target_albums:
            album_id = album.get('id')
            if not album_id: continue
            
            album_songs = get_tracks_from_album(album_id)
            for song in album_songs:
                # Convert to the expected format
                all_songs.append({
                    'Id': song.get('Id'), 
                    'Name': song.get('Name'), 
                    'AlbumArtist': song.get('AlbumArtist'), 
                    'Path': song.get('Path')
                })

    return all_songs

def _add_to_playlist(playlist_id, item_ids, user_creds=None):
    """
    Adds a list of songs to an existing Navidrome playlist in batches.
    Uses the 'updatePlaylist' endpoint.
    """
    if not item_ids:
        return True

    logger.info(f"Adding {len(item_ids)} songs to Navidrome playlist ID {playlist_id} in batches.")
    for i in range(0, len(item_ids), NAVIDROME_API_BATCH_SIZE):
        batch_ids = item_ids[i:i + NAVIDROME_API_BATCH_SIZE]
        params = {"playlistId": playlist_id, "songIdToAdd": batch_ids}
        
        # Note: updatePlaylist uses a POST method.
        response = _navidrome_request("updatePlaylist", params, method='post', user_creds=user_creds)
        
        if not (response and response.get("status") == "ok"):
            logger.error(f"Failed to add batch of {len(batch_ids)} songs to playlist {playlist_id}.")
            return False
    logger.info(f"Successfully added all songs to playlist {playlist_id}.")
    return True

def _create_playlist_batched(playlist_name, item_ids, user_creds=None):
    """
    Creates a new playlist on Navidrome. Handles large numbers of
    songs by batching and captures the new playlist ID directly from the
    creation response to avoid race conditions.
    """
    # If no songs are provided, create an empty playlist.
    if not item_ids:
        item_ids = []

    # --- Create the playlist and capture the response ---
    ids_for_creation = item_ids[:NAVIDROME_API_BATCH_SIZE]
    ids_to_add_later = item_ids[NAVIDROME_API_BATCH_SIZE:]

    create_params = {"name": playlist_name, "songId": ids_for_creation}
    create_response = _navidrome_request("createPlaylist", create_params, method='post', user_creds=user_creds)

    # --- Extract playlist object directly from the creation response ---
    if not (create_response and create_response.get("status") == "ok" and "playlist" in create_response):
        logger.error(f"Failed to create Navidrome playlist '{playlist_name}' or API response was malformed.")
        return None

    new_playlist = create_response["playlist"]
    new_playlist_id = new_playlist.get("id")

    if not new_playlist_id:
        logger.error(f"Navidrome playlist '{playlist_name}' was created, but the response did not contain an ID.")
        return None

    logger.info(f"✅ Created Navidrome playlist '{playlist_name}' (ID: {new_playlist_id}) with the first {len(ids_for_creation)} songs.")

    # If there are more songs to add, use the ID we just got
    if ids_to_add_later:
        if not _add_to_playlist(new_playlist_id, ids_to_add_later, user_creds):
            logger.error(f"Failed to add all songs to the new playlist '{playlist_name}'. The playlist was created but may be incomplete.")
            # We still return the playlist object, as it was created.
    
    # Standardize the keys to match what the rest of the app expects ('Id' with capital I)
    new_playlist['Id'] = new_playlist.get('id')
    new_playlist['Name'] = new_playlist.get('name')
    
    return new_playlist


def create_playlist(base_name, item_ids):
    """Creates a new playlist on Navidrome using admin credentials, with batching."""
    _create_playlist_batched(base_name, item_ids, user_creds=None)


def get_all_playlists():
    """Fetches all playlists from Navidrome using admin credentials."""
    response = _navidrome_request("getPlaylists")
    if response and "playlists" in response and "playlist" in response["playlists"]:
        return [{**p, 'Id': p.get('id'), 'Name': p.get('name')} for p in response["playlists"]["playlist"]]
    return []

def delete_playlist(playlist_id):
    """Deletes a playlist on Navidrome using admin credentials."""
    response = _navidrome_request("deletePlaylist", {"id": playlist_id}, method='post')
    if response and response.get("status") == "ok":
        logger.info(f"🗑️ Deleted Navidrome playlist ID: {playlist_id}")
        return True
    logger.error(f"Failed to delete playlist ID '{playlist_id}' on Navidrome")
    return False

# --- USER-SPECIFIC NAVIDROME FUNCTIONS ---
def get_tracks_from_album(album_id, user_creds=None):
    """Fetches all audio tracks for an album. Uses specific user_creds if provided."""
    # Check if this is a pseudo-album for standalone tracks
    if str(album_id).startswith('standalone-'):
        # Look up tracks in the pseudo-album cache
        if album_id in _pseudo_album_cache:
            songs = _pseudo_album_cache[album_id]
            
            # Apply artist field prioritization to each song
            result = []
            for s in songs:
                title = s.get('title', 'Unknown')
                artist = _select_best_artist(s, title)
                result.append({
                    **s, 
                    'Id': s.get('id'), 
                    'Name': title, 
                    'AlbumArtist': artist, 
                    'Path': s.get('path')
                })
            return result
        else:
            logger.warning(f"Pseudo-album {album_id} not found in cache. This may indicate a workflow issue.")
            return []
        
    params = {"id": album_id}
    response = _navidrome_request("getAlbum", params, user_creds=user_creds)
    if response and "album" in response and "song" in response["album"]:
        songs = response["album"]["song"]
        
        # Apply artist field prioritization to each song
        result = []
        for s in songs:
            title = s.get('title', 'Unknown')
            artist = _select_best_artist(s, title)
            result.append({
                **s, 
                'Id': s.get('id'), 
                'Name': title, 
                'AlbumArtist': artist, 
                'Path': s.get('path')
            })
        return result
    return []

def get_playlist_by_name(playlist_name, user_creds=None):
    """
    Finds a Navidrome playlist by its exact name. Returns the first match found.
    This is primarily used for checking if a playlist exists before deletion.
    """
    response = _navidrome_request("getPlaylists", user_creds=user_creds)
    if not (response and "playlists" in response and "playlist" in response["playlists"]):
        return None

    # Find the first playlist that matches the name exactly.
    for playlist_summary in response["playlists"]["playlist"]:
        if playlist_summary.get("name") == playlist_name:
            # For the purpose of checking existence and getting an ID for deletion,
            # the summary object is sufficient.
            return playlist_summary
    
    return None # No match found

def get_top_played_songs(limit, user_creds):
    """Fetches the top N most played songs from Navidrome for a specific user."""
    all_top_songs = []
    num_albums_to_fetch = (limit // 10) + 10
    params = {"type": "frequent", "size": num_albums_to_fetch}
    response = _navidrome_request("getAlbumList2", params, user_creds=user_creds)
    if response and "albumList2" in response and "album" in response["albumList2"]:
        for album in response["albumList2"]["album"]:
            tracks = get_tracks_from_album(album.get("id"), user_creds=user_creds)
            if tracks: all_top_songs.extend(tracks)
    return random.sample(all_top_songs, limit) if len(all_top_songs) > limit else all_top_songs

def get_last_played_time(item_id, user_creds):
    """Fetches the last played time for a track for a specific user."""
    response = _navidrome_request("getSong", {"id": item_id}, user_creds=user_creds)
    if response and "song" in response: return response["song"].get("lastPlayed")
    return None

def create_instant_playlist(playlist_name, item_ids, user_creds):
    """Creates a new instant playlist on Navidrome for a specific user, with batching."""
    final_playlist_name = f"{playlist_name.strip()}_instant"
    return _create_playlist_batched(final_playlist_name, item_ids, user_creds)
