# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Jellyfin backend for the AudioMuse-AI media-server abstraction.

Implements the provider interface against a Jellyfin server's REST API using
MediaBrowser token auth. Dispatched by tasks/mediaserver/__init__.py when
config.MEDIASERVER_TYPE == 'jellyfin'.

Main Features:
* Fetches recent albums, album tracks, and downloads tracks.
* Creates/updates playlists and reads last-played times.
"""

from . import http as requests
import logging
import os
import config
from config import jellyfin_auth_header

from .helper import detect_path_format, detect_download_extension, is_auth_error
from .helper import select_best_artist as _select_best_artist

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 300
JELLYFIN_PLAYLIST_BATCH_SIZE = 100


def _get_target_library_ids():
    library_names_str = getattr(config, 'MUSIC_LIBRARIES', '')

    if not library_names_str.strip():
        return None

    target_names_lower = {
        name.strip().lower() for name in library_names_str.split(',') if name.strip()
    }

    url = f"{config.JELLYFIN_URL}/Library/VirtualFolders"
    try:
        r = requests.get(url, headers=config.HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        all_libraries = r.json()

        library_map = {
            lib['Name'].lower(): {'name': lib['Name'], 'id': lib['ItemId']}
            for lib in all_libraries
            if lib.get('CollectionType') == 'music'
        }

        available_music_libraries = [lib['name'] for lib in library_map.values()]
        logger.info(f"Available Jellyfin music libraries found: {available_music_libraries}")

        found_libraries = []
        unfound_names = []
        for target_name in target_names_lower:
            if target_name in library_map:
                found_libraries.append(library_map[target_name])
            else:
                unfound_names.append(target_name)

        if unfound_names:
            logger.warning(
                f"Jellyfin config specified library names that were not found: {list(unfound_names)}"
            )

        if not found_libraries:
            logger.warning(
                f"No matching music libraries found for configured names: {list(target_names_lower)}. No albums will be analyzed."
            )
            return set()

        music_library_ids = {lib['id'] for lib in found_libraries}
        found_names_original_case = [lib['name'] for lib in found_libraries]

        logger.info(
            f"Filtering analysis to {len(music_library_ids)} Jellyfin libraries: {found_names_original_case}"
        )
        return music_library_ids

    except Exception as e:
        logger.exception(
            f"Failed to fetch or parse Jellyfin virtual folders at '{url}': {e}"
        )
        return set()


def list_libraries(user_creds=None):
    base_url = (
        user_creds.get('url') if user_creds and user_creds.get('url') else config.JELLYFIN_URL
    ).rstrip('/')
    url = f"{base_url}/Library/VirtualFolders"
    try:
        r = requests.get(
            url, headers=_jellyfin_headers_from_creds(user_creds), timeout=REQUESTS_TIMEOUT
        )
        r.raise_for_status()
        all_libraries = r.json() or []
        return [
            {'id': lib.get('ItemId'), 'name': lib.get('Name')}
            for lib in all_libraries
            if isinstance(lib, dict)
            and lib.get('CollectionType') == 'music'
            and lib.get('ItemId')
            and lib.get('Name')
        ]
    except Exception as e:
        logger.exception(f"Jellyfin list_libraries failed at '{url}': {e}")
        return []


def _jellyfin_base_url(user_creds=None):
    return (
        user_creds.get('url') if user_creds and user_creds.get('url') else config.JELLYFIN_URL
    ).rstrip('/')


def _jellyfin_headers_from_creds(user_creds=None):
    token = (user_creds.get('token') if user_creds else None) or getattr(
        config, 'JELLYFIN_TOKEN', None
    )
    return jellyfin_auth_header(token)


def _jellyfin_get_users(token):
    url = f"{config.JELLYFIN_URL}/Users"
    headers = jellyfin_auth_header(token)
    try:
        r = requests.get(url, headers=headers, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.exception(f"Jellyfin get_users failed: {e}")
        return None


def resolve_user(identifier, token):
    users = _jellyfin_get_users(token)
    if users:
        for user in users:
            if user.get('Name', '').lower() == identifier.lower():
                logger.info(f"Matched username '{identifier}' to User ID '{user['Id']}'.")
                return user['Id']

    logger.info(f"No username match for '{identifier}'. Assuming it is a User ID.")
    return identifier


def get_recent_albums(limit):
    target_library_ids = _get_target_library_ids()

    if isinstance(target_library_ids, set) and not target_library_ids:
        logger.warning(
            "Library filtering is active, but no matching libraries were found on the server. Returning no albums."
        )
        return []

    all_albums = []
    fetch_all = limit == 0

    if target_library_ids is None:
        logger.info("Scanning all Jellyfin libraries for recent albums.")
        start_index = 0
        page_size = 500
        while True:
            url = f"{config.JELLYFIN_URL}/Items"
            params = {
                "userId": config.JELLYFIN_USER_ID,
                "IncludeItemTypes": "MusicAlbum",
                "SortBy": "DateCreated",
                "SortOrder": "Descending",
                "Recursive": True,
                "Limit": page_size,
                "StartIndex": start_index,
            }
            try:
                r = requests.get(
                    url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT
                )
                r.raise_for_status()
                response_data = r.json()
                albums_on_page = response_data.get("Items") or []

                if not albums_on_page:
                    break

                all_albums.extend(albums_on_page)
                start_index += len(albums_on_page)

                if len(albums_on_page) < page_size:
                    break
            except Exception:
                logger.exception("Jellyfin get_recent_albums failed during 'scan all'")
                break

    else:
        logger.info(
            f"Scanning {len(target_library_ids)} specific Jellyfin libraries for recent albums."
        )
        for library_id in target_library_ids:
            start_index = 0
            page_size = 500
            while True:
                url = f"{config.JELLYFIN_URL}/Items"
                params = {
                    "userId": config.JELLYFIN_USER_ID,
                    "IncludeItemTypes": "MusicAlbum",
                    "SortBy": "DateCreated",
                    "SortOrder": "Descending",
                    "Recursive": True,
                    "Limit": page_size,
                    "StartIndex": start_index,
                    "ParentId": library_id,
                }
                try:
                    r = requests.get(
                        url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT
                    )
                    r.raise_for_status()
                    response_data = r.json()
                    albums_on_page = response_data.get("Items") or []

                    if not albums_on_page:
                        break

                    all_albums.extend(albums_on_page)
                    start_index += len(albums_on_page)

                    if len(albums_on_page) < page_size:
                        break
                except Exception:
                    logger.exception(
                        f"Jellyfin get_recent_albums failed for library ID {library_id}"
                    )
                    break

    if target_library_ids is not None and len(target_library_ids) > 1:
        all_albums.sort(key=lambda x: x.get('DateCreated', ''), reverse=True)

    if not fetch_all:
        return all_albums[:limit]

    return all_albums


def get_tracks_from_album(album_id, user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
    url = f"{_jellyfin_base_url(user_creds)}/Items"
    params = {
        "userId": user_id,
        "ParentId": album_id,
        "IncludeItemTypes": "Audio",
        "Fields": "Path,ProductionYear,IndexNumber,ParentIndexNumber,AlbumArtist,Album,ArtistItems,Artists",
    }
    try:
        r = requests.get(
            url,
            headers=_jellyfin_headers_from_creds(user_creds),
            params=params,
            timeout=REQUESTS_TIMEOUT,
        )
        r.raise_for_status()
        items = r.json().get("Items") or []

        for item in items:
            item['OriginalAlbumArtist'] = item.get('AlbumArtist')
            title = item.get('Name', 'Unknown')
            artist_name, artist_id = _select_best_artist(item, title)
            item['AlbumArtist'] = artist_name
            item['ArtistId'] = artist_id
            item['Year'] = item.get('ProductionYear')
            item['FilePath'] = item.get('Path')

        return items
    except Exception:
        logger.exception(f"Jellyfin get_tracks_from_album failed for album {album_id}")
        return []


def download_track(temp_dir, item):
    try:
        track_id = item['Id']
        file_extension = detect_download_extension(item)
        download_url = f"{config.JELLYFIN_URL}/Items/{track_id}/Download"
        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")
        with requests.get(
            download_url, headers=config.HEADERS, stream=True, timeout=REQUESTS_TIMEOUT
        ) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Downloaded '{item['Name']}' to '{local_filename}'")
        return local_filename
    except Exception as e:
        logger.exception(f"Failed to download track {item.get('Name', 'Unknown')}: {e}")
        return None


def get_all_songs(user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
    url = f"{_jellyfin_base_url(user_creds)}/Items"
    all_items = []
    start_index = 0
    limit = 500

    while True:
        params = {
            "userId": user_id,
            "IncludeItemTypes": "Audio",
            "Recursive": True,
            "StartIndex": start_index,
            "Limit": limit,
            "Fields": "Path,ProductionYear,IndexNumber,ParentIndexNumber,AlbumArtist,Album,ArtistItems,Artists",
        }
        try:
            r = requests.get(
                url,
                headers=_jellyfin_headers_from_creds(user_creds),
                params=params,
                timeout=REQUESTS_TIMEOUT,
            )
            r.raise_for_status()
            items = r.json().get("Items") or []

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
                break

            start_index += limit
        except Exception:
            logger.exception(f"Jellyfin get_all_songs failed at index {start_index}")
            raise

    return all_items


def search_albums(query, user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
    url = f"{_jellyfin_base_url(user_creds)}/Items"
    params = {
        "userId": user_id,
        "IncludeItemTypes": "MusicAlbum",
        "Recursive": True,
        "SearchTerm": query,
        "Limit": 10,
        "Fields": "ChildCount,ProductionYear,AlbumArtist",
    }
    try:
        r = requests.get(
            url,
            headers=_jellyfin_headers_from_creds(user_creds),
            params=params,
            timeout=REQUESTS_TIMEOUT,
        )
        r.raise_for_status()
        items = r.json().get("Items") or []
        return [
            {
                'id': item.get('Id'),
                'name': item.get('Name'),
                'artist': item.get('AlbumArtist'),
                'year': item.get('ProductionYear'),
                'track_count': item.get('ChildCount'),
            }
            for item in items
        ]
    except Exception as e:
        logger.exception(f"Jellyfin search_albums failed: {e}")
        return []


def test_connection(user_creds=None):
    try:
        user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
        url = f"{_jellyfin_base_url(user_creds)}/Items"
        params = {
            "userId": user_id,
            "IncludeItemTypes": "Audio",
            "Recursive": True,
            "Fields": "Path,ProductionYear,IndexNumber,ParentIndexNumber,AlbumArtist,Album,ArtistItems,Artists",
            "StartIndex": 0,
            "Limit": 100,
        }
        r = requests.get(
            url,
            headers=_jellyfin_headers_from_creds(user_creds),
            params=params,
            timeout=REQUESTS_TIMEOUT,
        )
        r.raise_for_status()
        items = r.json().get('Items', []) or []
        sample = []
        for item in items:
            track_artist, _ = _select_best_artist(item, item.get('Name', 'Unknown'))
            sample.append(
                {
                    'Id': item.get('Id'),
                    'Path': item.get('Path'),
                    'Name': item.get('Name'),
                    'AlbumArtist': track_artist,
                }
            )
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
        return {
            'ok': False,
            'error': str(e),
            'auth_failed': is_auth_error(e),
            'sample_count': 0,
            'path_format': 'none',
            'warnings': [],
        }


def get_playlist_by_name(playlist_name):
    url = f"{config.JELLYFIN_URL}/Items"
    params = {"userId": config.JELLYFIN_USER_ID, "IncludeItemTypes": "Playlist", "Recursive": True}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        playlists = r.json().get("Items") or []
        for playlist in playlists:
            if playlist.get("Name") == playlist_name:
                return playlist
        return None
    except Exception:
        logger.exception(f"Jellyfin get_playlist_by_name failed for '{playlist_name}'")
        return None


def create_playlist(base_name, item_ids):
    url = f"{config.JELLYFIN_URL}/Playlists"
    body = {"Name": base_name, "Ids": item_ids, "UserId": config.JELLYFIN_USER_ID}
    try:
        r = requests.post(url, headers=config.HEADERS, json=body, timeout=REQUESTS_TIMEOUT)
        if r.ok:
            logger.info("Created Jellyfin playlist '%s'", base_name)
    except Exception as e:
        logger.exception("Exception creating Jellyfin playlist '%s': %s", base_name, e)


def get_all_playlists():
    url = f"{config.JELLYFIN_URL}/Items"
    params = {"userId": config.JELLYFIN_USER_ID, "IncludeItemTypes": "Playlist", "Recursive": True}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("Items") or []
    except Exception as e:
        logger.exception(f"Jellyfin get_all_playlists failed: {e}")
        return []


def delete_playlist(playlist_id):
    url = f"{config.JELLYFIN_URL}/Items/{playlist_id}"
    try:
        r = requests.delete(url, headers=config.HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.exception(f"Exception deleting Jellyfin playlist ID {playlist_id}: {e}")
        return False


def get_top_played_songs(limit, user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
    token = user_creds.get('token') if user_creds else config.JELLYFIN_TOKEN
    if not user_id or not token:
        raise ValueError("Jellyfin User ID and Token are required.")

    url = f"{config.JELLYFIN_URL}/Items"
    headers = jellyfin_auth_header(token)
    params = {
        "userId": user_id,
        "IncludeItemTypes": "Audio",
        "SortBy": "PlayCount",
        "SortOrder": "Descending",
        "Recursive": True,
        "Limit": limit,
        "Fields": "UserData,Path,ProductionYear",
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        items = r.json().get("Items") or []

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
        logger.exception(f"Jellyfin get_all_songs failed: {e}")
        return []


def get_last_played_time(item_id, user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
    token = user_creds.get('token') if user_creds else config.JELLYFIN_TOKEN
    if not user_id or not token:
        raise ValueError("Jellyfin User ID and Token are required.")

    url = f"{config.JELLYFIN_URL}/Items/{item_id}"
    headers = jellyfin_auth_header(token)
    params = {"userId": user_id, "Fields": "UserData"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("UserData", {}).get("LastPlayedDate")
    except Exception:
        logger.exception(
            f"Jellyfin get_last_played_time failed for item {item_id}, user {user_id}"
        )
        return None


def get_lyrics(track_id: str, timeout: float = 2.5):
    try:
        url = f"{config.JELLYFIN_URL}/Audio/{track_id}/Lyrics"
        r = requests.get(url, headers=config.HEADERS, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        lyrics_lines = data.get('Lyrics') or []
        if not lyrics_lines:
            return None
        text = '\n'.join(line.get('Text', '') for line in lyrics_lines if line.get('Text'))
        return text.strip() or None
    except Exception as exc:
        logger.debug('Jellyfin get_lyrics failed for %s: %s', track_id, exc)
        return None


def create_instant_playlist(playlist_name, item_ids, user_creds=None):
    token = config.JELLYFIN_TOKEN
    if user_creds and isinstance(user_creds, dict) and user_creds.get('token'):
        token = user_creds.get('token')
    if not token:
        raise ValueError("Jellyfin Token is required.")

    identifier = config.JELLYFIN_USER_ID
    if user_creds and isinstance(user_creds, dict) and user_creds.get('user_identifier'):
        identifier = user_creds.get('user_identifier')
    if not identifier:
        raise ValueError("Jellyfin User Identifier is required.")

    user_id = resolve_user(identifier, token)

    final_playlist_name = f"{playlist_name.strip()}_instant"
    url = f"{config.JELLYFIN_URL}/Playlists"
    headers = jellyfin_auth_header(token)
    body = {"Name": final_playlist_name, "Ids": item_ids, "UserId": user_id}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        logger.exception(
            "Exception creating Jellyfin instant playlist '%s' for user %s",
            playlist_name,
            user_id,
        )
        return None


def _fetch_playlist_items(playlist_id, user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.JELLYFIN_USER_ID
    headers = (
        jellyfin_auth_header(user_creds.get('token'))
        if user_creds and user_creds.get('token')
        else config.HEADERS
    )
    url = f"{config.JELLYFIN_URL}/Playlists/{playlist_id}/Items"
    params = {"UserId": user_id}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("Items") or []
    except Exception as e:
        logger.exception(f"Jellyfin _fetch_playlist_items failed for {playlist_id}: {e}")
        return None


def _get_playlist_entry_ids(playlist_id):
    items = _fetch_playlist_items(playlist_id)
    if items is None:
        return None
    entry_ids = [it.get("PlaylistItemId") for it in items if it.get("PlaylistItemId")]
    if len(entry_ids) != len(items):
        logger.warning(
            f"Jellyfin _get_playlist_entry_ids: playlist {playlist_id} had "
            f"{len(items) - len(entry_ids)} items missing PlaylistItemId - they will not be removed"
        )
    return entry_ids


def get_playlist_track_ids(playlist_id, user_creds=None):
    items = _fetch_playlist_items(playlist_id, user_creds=user_creds)
    if not items:
        return []
    return [str(it.get("Id")) for it in items if it.get("Id")]


def _remove_playlist_entries(playlist_id, entry_ids):
    if not entry_ids:
        return
    url = f"{config.JELLYFIN_URL}/Playlists/{playlist_id}/Items"
    for i in range(0, len(entry_ids), JELLYFIN_PLAYLIST_BATCH_SIZE):
        batch = entry_ids[i : i + JELLYFIN_PLAYLIST_BATCH_SIZE]
        params = {"entryIds": ",".join(batch)}
        r = requests.delete(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()


def _add_items_to_playlist(playlist_id, item_ids):
    if not item_ids:
        return True
    url = f"{config.JELLYFIN_URL}/Playlists/{playlist_id}/Items"
    for i in range(0, len(item_ids), JELLYFIN_PLAYLIST_BATCH_SIZE):
        batch = item_ids[i : i + JELLYFIN_PLAYLIST_BATCH_SIZE]
        params = {"ids": ",".join(batch), "userId": config.JELLYFIN_USER_ID}
        try:
            r = requests.post(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
        except Exception as e:
            logger.exception(
                f"Jellyfin _add_items_to_playlist: batch starting at {i} failed for playlist {playlist_id}: {e}",
            )
            return False
    return True


def _create_fresh_playlist(playlist_name, item_ids):
    url = f"{config.JELLYFIN_URL}/Playlists"
    first_batch = item_ids[:JELLYFIN_PLAYLIST_BATCH_SIZE]
    rest = item_ids[JELLYFIN_PLAYLIST_BATCH_SIZE:]
    body = {"Name": playlist_name, "Ids": first_batch, "UserId": config.JELLYFIN_USER_ID}
    try:
        r = requests.post(url, headers=config.HEADERS, json=body, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        created = r.json()
    except Exception:
        logger.exception(
            f"Jellyfin _create_fresh_playlist: create failed for '{playlist_name}'"
        )
        return None

    new_id = created.get("Id")
    if not new_id:
        logger.error(
            f"Jellyfin _create_fresh_playlist: created '{playlist_name}' but response had no Id"
        )
        return None

    if rest and not _add_items_to_playlist(new_id, rest):
        logger.error(
            f"Jellyfin _create_fresh_playlist: created '{playlist_name}' but failed to add overflow tracks"
        )

    logger.info(
        f"Jellyfin: created playlist '{playlist_name}' (Id={new_id}) with {len(item_ids)} tracks"
    )
    return {**created, 'Id': new_id, 'Name': created.get('Name', playlist_name)}


def create_or_replace_playlist(playlist_name, item_ids, user_creds=None):
    if not item_ids:
        return None

    existing = get_playlist_by_name(playlist_name)
    if not existing:
        return _create_fresh_playlist(playlist_name, item_ids)

    playlist_id = existing.get("Id")
    if not playlist_id:
        logger.error(
            f"Jellyfin create_or_replace_playlist: existing playlist '{playlist_name}' has no Id"
        )
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
            logger.exception(
                f"Jellyfin: failed to delete playlist '{playlist_name}' (Id={playlist_id}) for fallback recreate"
            )
            return None
        return _create_fresh_playlist(playlist_name, item_ids)

    if not _add_items_to_playlist(playlist_id, item_ids):
        logger.error(
            f"Jellyfin create_or_replace_playlist: failed to add tracks to playlist {playlist_id}"
        )
        return None

    logger.info(
        f"Jellyfin: replaced contents of playlist '{playlist_name}' (Id={playlist_id}, tracks={len(item_ids)})"
    )
    return {**existing, 'Id': playlist_id, 'Name': existing.get('Name', playlist_name)}
