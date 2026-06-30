from . import http as requests
import logging
import os
from urllib.parse import unquote, urlparse
import config

from .helper import detect_path_format, is_auth_error

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 300


class LyrionAPIError(Exception):
    pass


def _decode_lyrion_url(url):
    if not url:
        return None
    if url.startswith('file://'):
        return unquote(urlparse(url).path)
    return unquote(url)


_LYRION_REMOTE_SERVICES = ('spotify', 'qobuz', 'tidal', 'wimp', 'youtube', 'deezer')


def _lyrion_is_remote(item):
    if not isinstance(item, dict):
        return False
    for key in ('url', 'path', 'Path'):
        val = item.get(key)
        if isinstance(val, str) and val:
            lower = val.lower()
            for svc in _LYRION_REMOTE_SERVICES:
                if svc in lower:
                    return True
    for key in ('genre', 'type', 'service', 'source'):
        val = item.get(key)
        if isinstance(val, str) and val:
            lower = val.lower()
            for svc in _LYRION_REMOTE_SERVICES:
                if svc in lower:
                    return True
    return False


def _lyrion_track(item):
    if not isinstance(item, dict):
        return {
            'id': None,
            'path': None,
            'title': None,
            'artist': None,
            'album_artist': None,
            'album': None,
            'year': None,
            'track_number': None,
            'disc_number': None,
        }

    def _try(*keys):
        for key in keys:
            value = item.get(key)
            if value is not None:
                return value
        return None

    year = _try('year', 'Year')
    if isinstance(year, str):
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = None

    return {
        'id': _try('id', 'track_id'),
        'path': _try('url', 'path'),
        'title': _try('title', 'name'),
        'artist': _try('artist', 'albumartist'),
        'album_artist': _try('albumartist', 'artist'),
        'album': _try('album', 'title'),
        'year': year,
        'track_number': _try('track', 'index', 'trackNumber'),
        'disc_number': _try('disc', 'disc_number', 'parentIndexNumber'),
    }


def _lyrion_titles_response(response):
    if not response:
        return []
    if isinstance(response, dict):
        titles = response.get('titles_loop')
        if isinstance(titles, list):
            return titles
        for value in response.values():
            if isinstance(value, list):
                return value
        return []
    if isinstance(response, list):
        return response
    return []


def _get_target_paths_for_filtering():
    folder_names_str = getattr(config, 'MUSIC_LIBRARIES', '')
    logger.info(f"DEBUG: MUSIC_LIBRARIES config value: '{folder_names_str}'")

    if not folder_names_str.strip():
        logger.info("DEBUG: MUSIC_LIBRARIES is empty, no path filtering")
        return None

    target_paths = {path.strip().lower() for path in folder_names_str.split(',') if path.strip()}
    logger.info(f"DEBUG: Target paths for filtering: {list(target_paths)}")
    return target_paths


def _get_target_music_folder_ids():
    folder_names_str = getattr(config, 'MUSIC_LIBRARIES', '')

    logger.info(f"DEBUG: MUSIC_LIBRARIES config value: '{folder_names_str}'")

    if not folder_names_str.strip():
        logger.info("DEBUG: MUSIC_LIBRARIES is empty, scanning all folders")
        return None

    target_names_lower = {
        name.strip().lower() for name in folder_names_str.split(',') if name.strip()
    }
    logger.info(f"DEBUG: Target names/paths to match: {list(target_names_lower)}")

    response = _jsonrpc_request("musicfolders", [0, 999999])

    logger.info(f"DEBUG: Lyrion musicfolders response: {response}")

    if not response:
        logger.error("Failed to fetch music folders from Lyrion or response was empty.")
        logger.warning(
            "Since MUSIC_LIBRARIES is configured but folder detection failed, returning empty set to prevent scanning everything."
        )
        return set()

    all_folders = []
    if isinstance(response, dict) and "folder_loop" in response:
        all_folders = response["folder_loop"]
    elif isinstance(response, dict) and "folders_loop" in response:
        all_folders = response["folders_loop"]
    elif isinstance(response, list):
        all_folders = response
    else:
        if isinstance(response, dict):
            for v in response.values():
                if isinstance(v, list):
                    all_folders = v
                    break

    if not all_folders:
        logger.error("No music folders found in Lyrion response.")
        return set()

    folder_map = {}
    for folder in all_folders:
        if isinstance(folder, dict):
            folder_name = folder.get('name') or folder.get('folder')
            folder_path = folder.get('path') or folder.get('url')
            folder_id = folder.get('id') or folder.get('folder_id')
            logger.info(
                f"DEBUG: Processing folder - name: '{folder_name}', path: '{folder_path}', id: '{folder_id}', raw: {folder}"
            )
            if folder_name and folder_id:
                folder_info = {
                    'name': folder_name,
                    'id': folder_id,
                    'path': folder_path or folder_name,
                }
                folder_map[folder_name.lower()] = folder_info
                if folder_path and folder_path.lower() != folder_name.lower():
                    folder_map[folder_path.lower()] = folder_info
                logger.info(
                    f"DEBUG: Added to folder_map - name key: '{folder_name.lower()}', path key: '{folder_path.lower() if folder_path else 'N/A'}'"
                )

    unique_folders = {folder['id']: folder for folder in folder_map.values()}
    available_info = [
        f"{folder['name']} (path: {folder['path']})" for folder in unique_folders.values()
    ]
    logger.info(f"Available Lyrion music folders found: {available_info}")

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
        logger.warning(
            f"Lyrion config specified folder names that were not found: {list(unfound_names)}"
        )

    if not found_folders:
        logger.warning(
            f"No matching music folders found for configured names: {list(target_names_lower)}. No albums will be analyzed."
        )
        return set()

    music_folder_ids = {folder['id'] for folder in found_folders}
    found_info = [f"{folder['name']} (path: {folder['path']})" for folder in found_folders]

    logger.info(f"Filtering analysis to {len(music_folder_ids)} Lyrion folders: {found_info}")
    logger.info(f"DEBUG: Returning folder IDs: {music_folder_ids}")
    return music_folder_ids


def list_libraries(user_creds=None):
    response = _jsonrpc_request("musicfolder", [0, 999999], user_creds=user_creds)
    if not response:
        return []

    all_folders = []
    if isinstance(response, dict):
        if "folder_loop" in response:
            all_folders = response["folder_loop"]
        elif "folders_loop" in response:
            all_folders = response["folders_loop"]
        else:
            for v in response.values():
                if isinstance(v, list):
                    all_folders = v
                    break
    elif isinstance(response, list):
        all_folders = response

    libraries = []
    for folder in all_folders or []:
        if not isinstance(folder, dict):
            continue
        folder_id = folder.get('id') or folder.get('folder_id')
        folder_name = folder.get('filename') or folder.get('name') or folder.get('folder')
        folder_path = folder.get('path') or folder.get('url')
        if folder_id is None or not folder_name:
            continue
        display_name = folder_path or folder_name
        libraries.append({'id': str(folder_id), 'name': display_name})
    return libraries


def _get_first_player():
    try:
        response = _jsonrpc_request("players", [0, 1])
        if response and "players_loop" in response and response["players_loop"]:
            player = response["players_loop"][0]
            player_id = player.get("playerid")
            if player_id:
                logger.info(f"Found Lyrion player: {player_id}")
                return player_id

        logger.warning("No Lyrion players found, using fallback player ID")
        return "10.42.6.0"
    except Exception as e:
        logger.exception(f"Error getting Lyrion player: {e}")
        return "10.42.6.0"


def _jsonrpc_request(method, params, player_id="", user_creds=None, timeout=None):
    base_url = (
        user_creds.get('url') if user_creds and user_creds.get('url') else config.LYRION_URL
    ).rstrip('/')
    url = f"{base_url}/jsonrpc.js"
    payload = {"id": 1, "method": "slim.request", "params": [player_id, [method, *params]]}
    auth = None
    if user_creds:
        user = user_creds.get('user')
        password = user_creds.get('password')
        if user or password:
            auth = (user or '', password or '')

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with requests.Session() as s:
                s.headers.update({"Content-Type": "application/json"})
                r = s.post(url, json=payload, timeout=timeout or REQUESTS_TIMEOUT, auth=auth)

            r.raise_for_status()
            response_data = r.json()

            if response_data.get("error"):
                msg = response_data['error'].get('message')
                logger.error(f"Lyrion JSON-RPC Error: {msg}")
                raise LyrionAPIError(f"Lyrion API error: {msg}")

            return response_data.get("result")

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.warning(
                f"Connection issue with Lyrion API (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                import time

                time.sleep(2)
                continue
            else:
                err = f"Failed to connect to Lyrion after {max_retries} attempts: {e}"
                logger.exception(err)
                raise LyrionAPIError(err)
        except LyrionAPIError:
            raise
        except Exception as e:
            logger.exception(f"Failed to call Lyrion JSON-RPC API with method '{method}'")
            raise LyrionAPIError(f"Unexpected error calling Lyrion API: {e}")

    raise LyrionAPIError("Unreachable: exceeded jsonrpc retry loop")


def _count_albums(use_sort_new=True, page_size=100):
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
    try:
        track_id = item.get('Id')
        if not track_id:
            logger.error("Lyrion item does not have a track ID.")
            return None

        download_url = f"{config.LYRION_URL}/music/{track_id}/download"

        file_extension = item.get('Path', '.mp3')
        if file_extension and '.' in file_extension:
            file_extension = os.path.splitext(file_extension)[1]
        else:
            file_extension = '.mp3'

        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")

        logger.info(f"Attempting to download from URL: {download_url}")

        with requests.Session() as s:
            with s.get(download_url, stream=True, timeout=REQUESTS_TIMEOUT) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        logger.info(f"Downloaded '{item.get('title', 'Unknown')}' to '{local_filename}'")
        return local_filename
    except Exception:
        logger.exception(f"Failed to download Lyrion track {item.get('title', 'Unknown')}")
    return None


def _get_all_albums_simple(limit):
    albums_accum = []
    fetch_all = limit == 0
    page_size = 100
    remaining = None if fetch_all else int(limit)
    offset = 0

    pages_fetched = 0

    use_sort_new = True
    if limit == 0:
        logger.info(
            "Counting Lyrion albums with and without 'sort:new' to decide pagination mode..."
        )
        sorted_total = _count_albums(use_sort_new=True, page_size=page_size)
        unsorted_total = _count_albums(use_sort_new=False, page_size=page_size)
        logger.info(
            f"Albums with sort:new = {sorted_total}, albums without sort = {unsorted_total}"
        )
        if (
            unsorted_total is not None
            and sorted_total is not None
            and unsorted_total > sorted_total
        ):
            logger.info(
                "Albums without sort are more numerous; proceeding with unsorted pagination for Lyrion."
            )
            use_sort_new = False

    while True:
        req_count = page_size if (remaining is None or remaining > page_size) else remaining
        params = [offset, req_count, "sort:new"] if use_sort_new else [offset, req_count]

        page_response = None
        page_error = None
        for attempt in range(3):
            try:
                page_response = _jsonrpc_request("albums", params)
                break
            except LyrionAPIError as e:
                page_error = e
                logger.warning(
                    f"albums page fetch failed at offset={offset} (attempt {attempt + 1}/3): {e}"
                )
                import time

                time.sleep(1)
                continue

        if page_response is None:
            logger.error(
                f"Skipping albums page at offset={offset} after repeated failures: {page_error}"
            )
            offset += page_size
            continue

        response = page_response

        page_albums = []
        if isinstance(response, dict) and "albums_loop" in response:
            page_albums = response["albums_loop"]
        elif isinstance(response, list):
            page_albums = response

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

    logger.info(
        f"_get_all_albums_simple: fetched {len(albums_accum)} albums across {pages_fetched} pages (requested limit: {limit})"
    )
    return albums_accum


def _album_has_tracks_in_target_path(album_id, target_paths):
    attempts = 3
    for attempt in range(attempts):
        try:
            response = _jsonrpc_request("titles", [0, 20, f"album_id:{album_id}", "tags:fFlpuoP"])

            if not response or "titles_loop" not in response:
                return False

            tracks = response["titles_loop"]

            path_fields = ['url', 'path', 'f', 'F', 'l', 'p', 'u', 'o', 'file', 'filename']

            for track in tracks[:8]:
                for field in path_fields:
                    if field in track and track[field]:
                        track_path = str(track[field]).lower()

                        for target_path in target_paths:
                            if target_path in track_path:
                                return True

                        for target_path in target_paths:
                            target_parts = target_path.strip('/').split('/')
                            if len(target_parts) >= 2:
                                last_part = target_parts[-1].lower()
                                if last_part in track_path:
                                    return True

            return False

        except LyrionAPIError as e:
            logger.warning(
                f"Transient Lyrion API error checking album tracks (attempt {attempt + 1}/{attempts}): {e}"
            )
            import time

            time.sleep(1)
            continue
        except Exception as e:
            logger.debug(
                f"Unexpected error while checking album tracks for {album_id}: {e}", exc_info=True
            )
            return False

    logger.error(f"Failed to fetch tracks for album {album_id} after {attempts} attempts")
    return False


def get_recent_albums(limit):
    target_paths = _get_target_paths_for_filtering()

    if target_paths is None:
        return _get_all_albums_simple(limit)

    logger.info(
        f"Scanning Lyrion library for albums in configured folders (limit: {limit or 'all'})"
    )

    filtered_albums = []
    page_size = 100
    offset = 0
    fetch_all = limit == 0
    pages_fetched = 0
    albums_scanned = 0
    albums_matched = 0
    filtered_no_album_id = 0
    filtered_no_path = 0

    use_sort_new = True
    if fetch_all:
        logger.info(
            "Counting Lyrion albums with and without 'sort:new' to decide pagination mode (analysis of all albums)..."
        )
        sorted_total = _count_albums(use_sort_new=True, page_size=page_size)
        unsorted_total = _count_albums(use_sort_new=False, page_size=page_size)
        logger.info(
            f"Albums with sort:new = {sorted_total}, albums without sort = {unsorted_total}"
        )
        if (
            unsorted_total is not None
            and sorted_total is not None
            and unsorted_total > sorted_total
        ):
            logger.info(
                "Albums without sort are more numerous; proceeding with unsorted pagination for Lyrion."
            )
            use_sort_new = False

    while True:
        params = [offset, page_size, "sort:new"] if use_sort_new else [offset, page_size]
        page_response = None
        page_error = None
        for attempt in range(3):
            try:
                page_response = _jsonrpc_request("albums", params)
                break
            except LyrionAPIError as e:
                page_error = e
                logger.warning(
                    f"albums page fetch failed at offset={offset} (attempt {attempt + 1}/3): {e}"
                )
                import time

                time.sleep(1)
                continue

        if page_response is None:
            logger.error(
                f"Skipping albums page at offset={offset} after repeated failures: {page_error}"
            )
            offset += page_size
            continue

        response = page_response
        if not response:
            break

        pages_fetched += 1

        page_albums = []
        if isinstance(response, dict) and "albums_loop" in response:
            page_albums = response["albums_loop"]
        elif isinstance(response, list):
            page_albums = response

        if not page_albums:
            break

        for album in page_albums:
            albums_scanned += 1
            album_id = album.get('id')
            album_name = album.get('album', 'Unknown')

            if not album_id:
                filtered_no_album_id += 1
                logger.debug(f"Skipping album without ID (page {pages_fetched}): {album}")
                continue

            try:
                has_tracks = _album_has_tracks_in_target_path(album_id, target_paths)
            except Exception as e:
                logger.warning(f"Error checking album paths for album {album_id}: {e}")
                has_tracks = False

            if has_tracks:
                albums_matched += 1
                mapped_album = {'Id': album_id, 'Name': album_name}
                filtered_albums.append(mapped_album)

                if not fetch_all and len(filtered_albums) >= limit:
                    logger.info(
                        f"Found {limit} matching albums in configured folders (pages fetched: {pages_fetched}, albums scanned: {albums_scanned}, matched: {albums_matched})"
                    )
                    return filtered_albums
            else:
                filtered_no_path += 1
                logger.debug(
                    f"Album {album_id} ('{album_name}') does not appear to have tracks in target paths (page {pages_fetched})."
                )

        if len(page_albums) < page_size:
            break

        offset += len(page_albums)

    logger.info(
        f"Found {len(filtered_albums)} albums in configured folders (pages fetched: {pages_fetched}, albums scanned: {albums_scanned}, matched: {albums_matched}, filtered_no_path: {filtered_no_path}, filtered_no_album_id: {filtered_no_album_id})"
    )
    return filtered_albums


def get_all_songs(user_creds=None):
    target_paths = _get_target_paths_for_filtering()

    if target_paths is not None:
        logger.warning("LYRION FOLDER FILTERING IS DISABLED - fetching all songs instead")

    logger.info("Fetching all songs from Lyrion")
    response = _jsonrpc_request("titles", [0, 999999, "tags:galduAyR"], user_creds=user_creds)

    all_songs = []
    if response and "titles_loop" in response:
        songs = response["titles_loop"]

        for song in songs:
            if song.get('trackartist'):
                track_artist = song.get('trackartist')
            elif song.get('contributor'):
                track_artist = song.get('contributor')
            elif song.get('artist'):
                track_artist = song.get('artist')
            elif song.get('albumartist'):
                track_artist = song.get('albumartist')
            elif song.get('band'):
                track_artist = song.get('band')
            else:
                track_artist = 'Unknown Artist'

            mapped_song = {
                'Id': song.get('id'),
                'Name': song.get('title'),
                'AlbumArtist': track_artist,
                'OriginalAlbumArtist': song.get('albumartist'),
                'Album': song.get('album'),
                'Path': song.get('url'),
                'url': song.get('url'),
                'Year': int(song.get('year')) if song.get('year') else None,
                'Rating': int(int(song.get('rating')) / 20) if song.get('rating') else None,
                'FilePath': _decode_lyrion_url(song.get('url')),
            }
            all_songs.append(mapped_song)

        logger.info(f"Found {len(songs)} total songs")

    return all_songs


def search_albums(query, user_creds=None):
    body = _jsonrpc_request(
        "albums", [0, 10, f"search:{query}", "tags:lyja"], user_creds=user_creds
    )
    if not body:
        return []
    albums = []
    if isinstance(body, dict):
        albums = body.get('albums_loop') or []
    elif isinstance(body, list):
        albums = body
    out = []
    for a in albums:
        year = a.get('year')
        if not isinstance(year, int):
            try:
                year = int(year) if year not in (None, '') else None
            except (TypeError, ValueError):
                year = None
        out.append(
            {
                'id': str(a.get('id', '')) or None,
                'name': a.get('album') or a.get('title'),
                'artist': a.get('artist') or a.get('albumartist'),
                'year': year,
                'track_count': a.get('tracks') or a.get('count'),
            }
        )
    return out


def test_connection(user_creds=None):
    warnings = []
    try:
        body = _jsonrpc_request("titles", [0, 100, "tags:galduAyR"], user_creds=user_creds)
        if body is None:
            return {
                'ok': False,
                'error': 'Lyrion test_connection failed',
                'auth_failed': False,
                'sample_count': 0,
                'path_format': 'none',
                'warnings': [],
            }
        raws = _lyrion_titles_response(body)
        if not raws:
            body = _jsonrpc_request("titles", [0, 100], user_creds=user_creds)
            raws = _lyrion_titles_response(body)
    except Exception as e:
        logger.warning(f"Lyrion test_connection failed: {e}")
        return {
            'ok': False,
            'error': str(e),
            'auth_failed': is_auth_error(e),
            'sample_count': 0,
            'path_format': 'none',
            'warnings': [],
        }

    sample = []
    for r in raws:
        if _lyrion_is_remote(r):
            continue
        sample.append(_lyrion_track(r))

    path_format = detect_path_format(sample)
    if path_format != 'absolute':
        warnings.append(
            'Lyrion is returning relative paths, stream URIs, or no paths at all. '
            'Automatic path-based matching may be unreliable for this library. '
            'Expect to manually match most albums in Step 4.'
        )
    return {
        'ok': True,
        'error': None,
        'sample_count': len(sample),
        'path_format': path_format,
        'warnings': warnings,
    }


def _add_to_playlist(playlist_id, item_ids):
    if not item_ids:
        return True

    logger.info(f"Adding {len(item_ids)} songs to Lyrion playlist ID '{playlist_id}'.")

    player_id = _get_first_player()
    if not player_id:
        logger.error("No Lyrion player available for playlist operations.")
        return False

    try:
        logger.debug("Step 0: Getting original playlist name before operations")
        playlist_info = _jsonrpc_request("playlists", [0, 999999])

        original_name = None
        if playlist_info and "playlists_loop" in playlist_info:
            for pl in playlist_info["playlists_loop"]:
                if str(pl.get("id")) == str(playlist_id):
                    original_name = pl.get("playlist")
                    logger.debug(
                        f"Found original playlist name: '{original_name}' for ID {playlist_id}"
                    )
                    break

        if not original_name:
            logger.error(f"Could not find playlist {playlist_id} in playlists list!")
            return False

        logger.info("Using method: Load -> Add -> Update original playlist via edit command")

        logger.debug(f"Step 1: Loading playlist {playlist_id} to player {player_id}")
        load_response = _jsonrpc_request(
            "playlistcontrol", ["cmd:load", f"playlist_id:{playlist_id}"], player_id
        )

        logger.debug(f"Load playlist response: {load_response}")

        batch_size = 50
        total_added = 0

        for i in range(0, len(item_ids), batch_size):
            batch_ids = item_ids[i : i + batch_size]
            track_id_list = ",".join(str(track_id) for track_id in batch_ids)

            logger.debug(f"Step 2: Adding batch {i // batch_size + 1} with {len(batch_ids)} tracks")
            add_response = _jsonrpc_request(
                "playlistcontrol", ["cmd:add", f"track_id:{track_id_list}"], player_id
            )

            logger.debug(f"Add batch response: {add_response}")

            if add_response and "count" in add_response:
                batch_added = add_response.get("count", 0)
                total_added += batch_added
                logger.debug(f"Added {batch_added} tracks in this batch, total: {total_added}")

            if i + batch_size < len(item_ids):
                import time

                time.sleep(0.1)

        logger.debug(f"Step 3: Deleting original empty playlist {playlist_id}")
        delete_response = _jsonrpc_request("playlists", ["delete", f"playlist_id:{playlist_id}"])
        logger.debug(f"Delete response: {delete_response}")

        logger.debug(f"Step 4: Saving current playlist as '{original_name}'")
        save_response = _jsonrpc_request("playlist", ["save", original_name, "silent:1"], player_id)

        logger.debug(f"Save playlist response: {save_response}")

        if save_response and "__playlist_id" in save_response:
            final_playlist_id = save_response["__playlist_id"]
            if str(final_playlist_id) == str(playlist_id):
                logger.info(
                    f"Successfully updated original playlist {playlist_id} with {total_added} tracks"
                )
                return True
            else:
                logger.warning(
                    f"Created new playlist {final_playlist_id} instead of updating {playlist_id}"
                )
                try:
                    logger.info(
                        f"Working with new playlist ID {final_playlist_id} which has the content"
                    )
                    return True
                except Exception as e:
                    logger.exception(f"Error handling new playlist: {e}")
                    return False
        elif total_added > 0:
            logger.info(f"Successfully added {total_added} tracks (save response: {save_response})")
            return True
        else:
            logger.warning("No tracks were added to the playlist")
            return False

    except Exception as e:
        logger.exception(f"Error in playlist update method: {e}")
        return False


def _create_playlist_batched(playlist_name, item_ids):
    logger.info(
        f"Attempting to create Lyrion playlist '{playlist_name}' with {len(item_ids)} songs using web interface method."
    )

    try:
        create_response = _jsonrpc_request("playlists", ["new", f"name:{playlist_name}"])

        if create_response:
            playlist_id = (
                create_response.get("id")
                or create_response.get("overwritten_playlist_id")
                or create_response.get("playlist_id")
            )

            if playlist_id:
                logger.info(f"Created Lyrion playlist '{playlist_name}' (ID: {playlist_id}).")

                if item_ids:
                    if _add_to_playlist(playlist_id, item_ids):
                        logger.info(
                            f"Successfully added {len(item_ids)} tracks to playlist '{playlist_name}'."
                        )
                    else:
                        logger.warning(
                            f"Playlist '{playlist_name}' created but some tracks may not have been added."
                        )

                return {"Id": playlist_id, "Name": playlist_name}

        logger.error(
            f"Failed to create Lyrion playlist '{playlist_name}'. Response: {create_response}"
        )
        return None

    except Exception as e:
        logger.exception(f"Exception creating Lyrion playlist '{playlist_name}': {e}")
        return None


def create_playlist(base_name, item_ids):
    return _create_playlist_batched(base_name, item_ids)


def get_all_playlists():
    response = _jsonrpc_request("playlists", [0, 999999])
    if response and "playlists_loop" in response:
        playlists = response["playlists_loop"]
        return [{'Id': p.get('id'), 'Name': p.get('playlist')} for p in playlists]
    return []


def delete_playlist(playlist_id):
    response = _jsonrpc_request("playlists", ["delete", f"playlist_id:{playlist_id}"])
    if response is not None:
        logger.info(f"Deleted Lyrion playlist ID: {playlist_id}")
        return True
    logger.error(f"Failed to delete playlist ID '{playlist_id}' on Lyrion")
    return False


def get_tracks_from_album(album_id, user_creds=None):
    logger.info(f"Attempting to fetch tracks for album ID: {album_id}")

    try:
        response = _jsonrpc_request(
            "titles", [0, 999999, f"album_id:{album_id}", "tags:galduAyR"], user_creds=user_creds
        )
        logger.debug(f"Lyrion API Raw Track Response for Album {album_id}: {response}")
    except Exception as e:
        logger.exception(f"Lyrion API call for album {album_id} failed: {e}")
        return []

    songs = []
    if not response:
        logger.warning(f"Lyrion API returned empty response for album {album_id}.")
        return []

    if isinstance(response, dict):
        if "titles_loop" in response and isinstance(response["titles_loop"], list):
            songs = response["titles_loop"]
        else:
            for v in response.values():
                if isinstance(v, list):
                    songs = v
                    break
    elif isinstance(response, list):
        songs = response

    if not songs:
        logger.warning(
            f"Lyrion API response for tracks of album {album_id} did not contain any song entries."
        )
        return []

    local_songs = []
    skipped_tracks = []
    for s in songs:
        if _lyrion_is_remote(s):
            skipped_tracks.append(s)
        else:
            local_songs.append(s)

    if skipped_tracks:
        skipped_count = len(skipped_tracks)
        logger.info(
            f"Skipping {skipped_count} track(s) from album {album_id} because they appear to be from a remote streaming service (Spotify, Qobuz, Tidal, Wimp, YouTube, Deezer, ...) or are non-downloadable."
        )
        for st in skipped_tracks:
            sk_id = st.get('id') or st.get('Id') or st.get('track_id')
            sk_title = st.get('title') or st.get('name') or st.get('Name')
            sk_artist = (
                st.get('trackartist')
                or st.get('contributor')
                or st.get('artist')
                or st.get('albumartist')
                or st.get('band')
                or 'Unknown Artist'
            )
            sk_url = st.get('url') or st.get('Path') or st.get('path')
            logger.info(
                f"Skipped track - id: {sk_id!r}, title: {sk_title!r}, artist: {sk_artist!r}, url/path: {sk_url!r}"
            )

    if not local_songs and songs:
        logger.info(
            f"Album {album_id} contains only remote streaming-service tracks (Spotify, Qobuz, Tidal, Wimp, YouTube, Deezer, ...) or non-downloadable tracks and will be skipped."
        )

    mapped = []
    for s in local_songs:
        id_val = s.get('id') or s.get('Id') or s.get('track_id')
        title = s.get('title') or s.get('name') or s.get('Name')

        if s.get('trackartist'):
            artist = s.get('trackartist')
        elif s.get('contributor'):
            artist = s.get('contributor')
        elif s.get('artist'):
            artist = s.get('artist')
        elif s.get('albumartist'):
            artist = s.get('albumartist')
        elif s.get('band'):
            artist = s.get('band')
        else:
            artist = 'Unknown Artist'

        path = s.get('url') or s.get('Path') or s.get('path') or ''
        mapped.append(
            {
                'Id': id_val,
                'Name': title,
                'AlbumArtist': artist,
                'OriginalAlbumArtist': s.get('albumartist'),
                'Album': s.get('album'),
                'Path': path,
                'url': path,
                'Year': int(s.get('year')) if s.get('year') else None,
                'Rating': int(int(s.get('rating')) / 20) if s.get('rating') else None,
                'FilePath': _decode_lyrion_url(s.get('url')),
            }
        )

    return mapped


def get_playlist_by_name(playlist_name):
    all_playlists = get_all_playlists()
    for p in all_playlists:
        if p.get('Name') == playlist_name:
            return p
    return None


def get_playlist_track_ids(playlist_id):
    try:
        response = _jsonrpc_request(
            "playlists", ["tracks", 0, 999999, f"playlist_id:{playlist_id}", "tags:u"]
        )
    except Exception as e:
        logger.exception(f"Lyrion get_playlist_track_ids failed for {playlist_id}: {e}")
        return []
    if not response:
        return []
    loop = []
    if isinstance(response, dict):
        if isinstance(response.get("playlisttracks_loop"), list):
            loop = response["playlisttracks_loop"]
        else:
            for v in response.values():
                if isinstance(v, list):
                    loop = v
                    break
    elif isinstance(response, list):
        loop = response
    return [str(t.get("id")) for t in loop if isinstance(t, dict) and t.get("id")]


def get_top_played_songs(limit):
    response = _jsonrpc_request("titles", [0, limit, "sort:popular", "tags:galduAyR"])
    if response and "titles_loop" in response:
        songs = response["titles_loop"]
        mapped_songs = []
        for s in songs:
            title = s.get('title', 'Unknown')

            if s.get('trackartist'):
                track_artist = s.get('trackartist')
            elif s.get('contributor'):
                track_artist = s.get('contributor')
            elif s.get('artist'):
                track_artist = s.get('artist')
            elif s.get('albumartist'):
                track_artist = s.get('albumartist')
            elif s.get('band'):
                track_artist = s.get('band')
            else:
                track_artist = 'Unknown Artist'

            mapped_songs.append(
                {
                    'Id': s.get('id'),
                    'Name': title,
                    'AlbumArtist': track_artist,
                    'OriginalAlbumArtist': s.get('albumartist'),
                    'Album': s.get('album'),
                    'Path': s.get('url'),
                    'url': s.get('url'),
                    'Year': int(s.get('year')) if s.get('year') else None,
                    'Rating': int(int(s.get('rating')) / 20) if s.get('rating') else None,
                    'FilePath': _decode_lyrion_url(s.get('url')),
                }
            )
        return mapped_songs
    return []


def get_last_played_time(item_id):
    logger.warning(
        "Lyrion's JSON-RPC API does not provide a 'last played time' for individual tracks."
    )
    return None


def get_lyrics(track_id: str, timeout: float = 2.5):
    try:
        result = _jsonrpc_request(
            'songinfo',
            [0, 100, f'track_id:{track_id}', 'tags:w'],
            timeout=timeout,
        )
        if not result:
            return None
        for entry in result.get('songinfo_loop', []):
            lyrics = entry.get('lyrics') or entry.get('Lyrics')
            if lyrics:
                return str(lyrics).strip() or None
        return None
    except Exception as exc:
        logger.debug('Lyrion get_lyrics failed for %s: %s', track_id, exc)
        return None


def create_instant_playlist(playlist_name, item_ids):
    final_playlist_name = f"{playlist_name.strip()}_instant"
    return _create_playlist_batched(final_playlist_name, item_ids)


def create_or_replace_playlist(playlist_name, item_ids, user_creds=None):
    if not item_ids:
        return None

    existing = get_playlist_by_name(playlist_name)
    if existing:
        old_id = existing.get('Id')
        if old_id and not delete_playlist(old_id):
            logger.error(
                f"Lyrion create_or_replace_playlist: failed to delete existing '{playlist_name}' "
                f"(id={old_id}); aborting to avoid creating a duplicate"
            )
            return None

    return _create_playlist_batched(playlist_name, item_ids)
