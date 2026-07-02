# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Emby backend for the AudioMuse-AI media-server abstraction.

Implements the provider interface against an Emby server's REST API (base path
http[s]://host:port/emby/..., docs at dev.emby.media). Dispatched by
tasks/mediaserver/__init__.py when config.MEDIASERVER_TYPE == 'emby'.

Main Features:
* Fetches recent albums, album tracks, downloads, and manages playlists.
* Handles the VirtualFolders endpoint, which returns a list (not a dict).
"""

from . import http as requests
import logging
import os
import config

from .helper import detect_path_format, detect_download_extension, is_auth_error
from .helper import select_best_artist as _select_best_artist

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 300
EMBY_PLAYLIST_BATCH_SIZE = 100
_TRACK_FIELDS = "Path,ProductionYear,IndexNumber,ParentIndexNumber,AlbumArtist,Album,ArtistItems,Artists"


def _get_target_library_ids():
    library_names_str = config.MUSIC_LIBRARIES

    if not library_names_str.strip():
        return None

    target_names_lower = {
        name.strip().lower() for name in library_names_str.split(',') if name.strip()
    }

    url = f"{config.EMBY_URL}/emby/Library/VirtualFolders"
    try:
        r = requests.get(url, headers=config.HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()

        all_libraries = r.json()
        if not isinstance(all_libraries, list):
            logger.warning(
                f"Unexpected response type from Emby: {type(all_libraries)} - expected a list."
            )
            all_libraries = []

        library_map = {
            lib['Name'].lower(): {'name': lib['Name'], 'id': lib['ItemId']}
            for lib in all_libraries
            if lib.get('CollectionType') == 'music'
        }

        available_music_libraries = [lib['name'] for lib in library_map.values()]
        logger.info(f"Available Emby music libraries found: {available_music_libraries}")

        found_libraries = []
        unfound_names = []
        for target_name in target_names_lower:
            if target_name in library_map:
                found_libraries.append(library_map[target_name])
            else:
                unfound_names.append(target_name)

        if unfound_names:
            logger.warning(
                f"Emby config specified library names that were not found: {list(unfound_names)}"
            )

        if not found_libraries:
            logger.warning(
                f"No matching music libraries found for configured names: {list(target_names_lower)}. No albums will be analyzed."
            )
            return set()

        music_library_ids = {lib['id'] for lib in found_libraries}
        found_names_original_case = [lib['name'] for lib in found_libraries]

        logger.info(
            f"Filtering analysis to {len(music_library_ids)} Emby libraries: {found_names_original_case}"
        )
        return music_library_ids

    except Exception:
        logger.exception(
            f"Failed to fetch or parse Emby virtual folders at '{url}'"
        )
        return set()


def list_libraries(user_creds=None):
    base_url = (
        user_creds.get('url') if user_creds and user_creds.get('url') else config.EMBY_URL
    ).rstrip('/')
    url = f"{base_url}/emby/Library/VirtualFolders"
    try:
        r = requests.get(
            url, headers=_emby_headers_from_creds(user_creds), timeout=REQUESTS_TIMEOUT
        )
        r.raise_for_status()
        all_libraries = r.json() or []
        if not isinstance(all_libraries, list):
            return []
        return [
            {'id': lib.get('ItemId'), 'name': lib.get('Name')}
            for lib in all_libraries
            if isinstance(lib, dict)
            and lib.get('CollectionType') == 'music'
            and lib.get('ItemId')
            and lib.get('Name')
        ]
    except Exception:
        logger.exception(f"Emby list_libraries failed at '{url}'")
        return []


def _emby_base_url(user_creds=None):
    return (
        user_creds.get('url') if user_creds and user_creds.get('url') else config.EMBY_URL
    ).rstrip('/')


def _emby_headers_from_creds(user_creds=None):
    headers = dict(config.HEADERS or {})
    token = user_creds.get('token') if user_creds else config.EMBY_TOKEN
    if token:
        headers['X-Emby-Token'] = token
    return headers


def _emby_get_users(token):
    url = f"{config.EMBY_URL}/emby/Users"
    headers = {"X-Emby-Token": token}
    try:
        r = requests.get(url, headers=headers, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        logger.exception("Emby get_users failed")
        return None


def resolve_user(identifier, token):
    users = _emby_get_users(token)
    if users:
        for user in users:
            if user.get('Name', '').lower() == identifier.lower():
                logger.info(f"Matched username '{identifier}' to User ID '{user['Id']}'.")
                return user['Id']

    logger.info(f"No username match for '{identifier}'. Assuming it is a User ID.")
    return identifier


def get_recent_albums(limit):
    if limit == 0:
        return get_recent_music_items(limit)
    else:
        return _get_recent_albums_only(limit)


def _get_recent_standalone_tracks(limit, target_library_ids=None, user_creds=None):
    """
    Fetches recent standalone audio tracks that are not properly organized in albums.
    This captures orphaned tracks, loose files, and tracks with missing album metadata.
    """
    user_id = user_creds.get('user_id') if user_creds else config.EMBY_USER_ID
    if (
        target_library_ids is not None
        and isinstance(target_library_ids, set)
        and not target_library_ids
    ):
        logger.info(
            "Library filtering is active but no matching libraries found. Skipping standalone tracks."
        )
        return []

    all_tracks = []
    fetch_all = limit == 0

    if target_library_ids is None:
        logger.info("Scanning all Emby libraries for recent standalone tracks.")
        start_index = 0
        page_size = 500
        while True:
            url = f"{config.EMBY_URL}/emby/Users/{user_id}/Items"
            params = {
                "IncludeItemTypes": "Audio",
                "SortBy": "DateCreated",
                "SortOrder": "Descending",
                "Recursive": True,
                "Limit": page_size,
                "StartIndex": start_index,
                "Fields": "ParentId,Path,DateCreated",
            }
            try:
                r = requests.get(
                    url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT
                )
                r.raise_for_status()
                response_data = r.json()
                tracks_on_page = response_data.get("Items") or []

                if not tracks_on_page:
                    break

                standalone_tracks = []
                for track in tracks_on_page:
                    parent_id = track.get('ParentId')
                    if not parent_id:
                        standalone_tracks.append(track)
                    else:
                        try:
                            parent_url = f"{config.EMBY_URL}/emby/Users/{user_id}/Items/{parent_id}"
                            parent_r = requests.get(
                                parent_url, headers=config.HEADERS, timeout=REQUESTS_TIMEOUT
                            )
                            if parent_r.ok:
                                parent_info = parent_r.json()
                                if parent_info.get('Type') != 'MusicAlbum':
                                    standalone_tracks.append(track)
                        except Exception:
                            standalone_tracks.append(track)

                all_tracks.extend(standalone_tracks)
                start_index += len(tracks_on_page)

                if not fetch_all and len(all_tracks) >= limit:
                    all_tracks = all_tracks[:limit]
                    break

                if len(tracks_on_page) < page_size:
                    break
            except Exception:
                logger.exception("Emby get_recent_standalone_tracks failed")
                break

    else:
        logger.info(
            f"Scanning {len(target_library_ids)} specific Emby libraries for recent standalone tracks."
        )
        for library_id in target_library_ids:
            start_index = 0
            page_size = 500
            while True:
                url = f"{config.EMBY_URL}/emby/Users/{user_id}/Items"
                params = {
                    "IncludeItemTypes": "Audio",
                    "SortBy": "DateCreated",
                    "SortOrder": "Descending",
                    "Recursive": True,
                    "Limit": page_size,
                    "StartIndex": start_index,
                    "ParentId": library_id,
                    "Fields": "ParentId,Path,DateCreated",
                }
                try:
                    r = requests.get(
                        url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT
                    )
                    r.raise_for_status()
                    response_data = r.json()
                    tracks_on_page = response_data.get("Items") or []

                    if not tracks_on_page:
                        break

                    standalone_tracks = []
                    for track in tracks_on_page:
                        parent_id = track.get('ParentId')
                        if not parent_id or parent_id == library_id:
                            standalone_tracks.append(track)
                        else:
                            try:
                                parent_url = (
                                    f"{config.EMBY_URL}/emby/Users/{user_id}/Items/{parent_id}"
                                )
                                parent_r = requests.get(
                                    parent_url, headers=config.HEADERS, timeout=REQUESTS_TIMEOUT
                                )
                                if parent_r.ok:
                                    parent_info = parent_r.json()
                                    if parent_info.get('Type') != 'MusicAlbum':
                                        standalone_tracks.append(track)
                            except Exception:
                                standalone_tracks.append(track)

                    all_tracks.extend(standalone_tracks)
                    start_index += len(tracks_on_page)

                    if not fetch_all and len(all_tracks) >= limit:
                        all_tracks = all_tracks[:limit]
                        break

                    if len(tracks_on_page) < page_size:
                        break
                except Exception:
                    logger.exception(
                        f"Emby get_recent_standalone_tracks failed for library ID {library_id}",
                    )
                    break

    for track in all_tracks:
        track['OriginalAlbumArtist'] = track.get('AlbumArtist')
        title = track.get('Name', 'Unknown')
        artist_name, artist_id = _select_best_artist(track, title)
        track['AlbumArtist'] = artist_name
        track['ArtistId'] = artist_id

    if all_tracks:
        logger.info(f"Found {len(all_tracks)} recent standalone tracks (not in albums)")

    return all_tracks


def _get_recent_albums_only(limit, user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.EMBY_USER_ID
    target_library_ids = _get_target_library_ids()

    if isinstance(target_library_ids, set) and not target_library_ids:
        logger.warning(
            "Library filtering is active, but no matching libraries were found on the server. Returning no albums."
        )
        return []

    all_albums = []
    fetch_all = limit == 0

    if target_library_ids is None:
        logger.info("Scanning all Emby libraries for recent albums (albums only).")
        start_index = 0
        page_size = 500
        while True:
            url = f"{config.EMBY_URL}/emby/Users/{user_id}/Items"
            params = {
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
                logger.exception(
                    "Emby _get_recent_albums_only failed during 'scan all'"
                )
                break

    else:
        logger.info(
            f"Scanning {len(target_library_ids)} specific Emby libraries for recent albums (albums only)."
        )
        for library_id in target_library_ids:
            start_index = 0
            page_size = 500
            while True:
                url = f"{config.EMBY_URL}/emby/Users/{user_id}/Items"
                params = {
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
                        f"Emby _get_recent_albums_only failed for library ID {library_id}",
                    )
                    break

    if target_library_ids is not None and len(target_library_ids) > 1:
        all_albums.sort(key=lambda x: x.get('DateCreated', ''), reverse=True)

    if not fetch_all:
        return all_albums[:limit]

    return all_albums


def get_recent_music_items(limit):
    target_library_ids = _get_target_library_ids()

    albums = _get_recent_albums_only(limit)

    standalone_limit = min(limit, 100) if limit > 0 else 100
    standalone_tracks = _get_recent_standalone_tracks(standalone_limit, target_library_ids)

    pseudo_albums = []
    for track in standalone_tracks:
        pseudo_album = {
            'Id': f"standalone_{track['Id']}",
            'Name': f"Standalone: {track.get('Name', 'Unknown')}",
            'Type': 'PseudoAlbum',
            'StandaloneTrack': track,
            'DateCreated': track.get('DateCreated', ''),
            'AlbumArtist': track.get('AlbumArtist', 'Unknown Artist'),
        }
        pseudo_albums.append(pseudo_album)

    all_items = albums + pseudo_albums

    if albums and pseudo_albums:
        all_items.sort(key=lambda x: x.get('DateCreated', ''), reverse=True)

    if limit > 0:
        all_items = all_items[:limit]

    if pseudo_albums:
        logger.info(
            f"Found {len(albums)} regular albums and {len(pseudo_albums)} standalone tracks (combined into {len(all_items)} total items)"
        )

    return all_items


def get_tracks_from_album(album_id, user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.EMBY_USER_ID
    if str(album_id).startswith('standalone_'):
        real_track_id = album_id.replace('standalone_', '')

        url = f"{_emby_base_url(user_creds)}/emby/Users/{user_id}/Items/{real_track_id}"
        params = {
            "Fields": _TRACK_FIELDS
        }
        try:
            r = requests.get(
                url,
                headers=_emby_headers_from_creds(user_creds),
                params=params,
                timeout=REQUESTS_TIMEOUT,
            )
            r.raise_for_status()
            track_item = r.json()

            track_item['OriginalAlbumArtist'] = track_item.get('AlbumArtist')
            title = track_item.get('Name', 'Unknown')
            artist_name, artist_id = _select_best_artist(track_item, title)
            track_item['AlbumArtist'] = artist_name
            track_item['ArtistId'] = artist_id
            track_item['Year'] = track_item.get('ProductionYear')
            track_item['FilePath'] = track_item.get('Path')

            return [track_item]
        except Exception:
            logger.exception(
                f"Emby get_tracks_from_album failed for standalone track {real_track_id}",
            )
            return []

    url = f"{_emby_base_url(user_creds)}/emby/Users/{user_id}/Items"
    params = {
        "ParentId": album_id,
        "IncludeItemTypes": "Audio",
        "Fields": _TRACK_FIELDS,
    }
    try:
        r = requests.get(
            url,
            headers=_emby_headers_from_creds(user_creds),
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
        logger.exception(f"Emby get_tracks_from_album failed for album {album_id}")
        return []


def download_track(temp_dir, item):
    try:
        track_id = item['Id']
        file_extension = detect_download_extension(item)
        download_url = f"{config.EMBY_URL}/emby/Items/{track_id}/Download"
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
    except Exception:
        logger.exception(f"Failed to download track {item.get('Name', 'Unknown')}")
        return None


def get_all_songs(user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.EMBY_USER_ID
    url = f"{_emby_base_url(user_creds)}/emby/Users/{user_id}/Items"
    all_items = []
    start_index = 0
    limit = 1000

    while True:
        params = {
            "IncludeItemTypes": "Audio",
            "Recursive": True,
            "StartIndex": start_index,
            "Limit": limit,
            "Fields": "UserData,Path,ProductionYear,IndexNumber,ParentIndexNumber,AlbumArtist,Album,ArtistItems,Artists",
        }
        try:
            r = requests.get(
                url,
                headers=_emby_headers_from_creds(user_creds),
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
            logger.exception(f"Emby get_all_songs failed at index {start_index}")
            raise

    return all_items


def search_albums(query, user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.EMBY_USER_ID
    url = f"{_emby_base_url(user_creds)}/emby/Users/{user_id}/Items"
    params = {
        "IncludeItemTypes": "MusicAlbum",
        "Recursive": True,
        "SearchTerm": query,
        "Limit": 10,
        "Fields": "ChildCount,ProductionYear,AlbumArtist",
    }
    try:
        r = requests.get(
            url,
            headers=_emby_headers_from_creds(user_creds),
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
    except Exception:
        logger.exception("Emby search_albums failed")
        return []


def test_connection(user_creds=None):
    try:
        user_id = user_creds.get('user_id') if user_creds else config.EMBY_USER_ID
        url = f"{_emby_base_url(user_creds)}/emby/Users/{user_id}/Items"
        params = {
            "IncludeItemTypes": "Audio",
            "Recursive": True,
            "Fields": _TRACK_FIELDS,
            "StartIndex": 0,
            "Limit": 100,
        }
        r = requests.get(
            url,
            headers=_emby_headers_from_creds(user_creds),
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
        logger.warning(f"Emby test_connection failed: {e}")
        return {
            'ok': False,
            'error': str(e),
            'auth_failed': is_auth_error(e),
            'sample_count': 0,
            'path_format': 'none',
            'warnings': [],
        }


def get_playlist_by_name(playlist_name, user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.EMBY_USER_ID
    url = f"{config.EMBY_URL}/emby/Users/{user_id}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        playlists = r.json().get("Items") or []

        for playlist in playlists:
            if playlist.get("Name") == playlist_name:
                return playlist

        return None

    except Exception:
        logger.exception(f"Emby get_playlist_by_name failed for '{playlist_name}'")
        return None


def create_playlist(playlist_name, item_ids, user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.EMBY_USER_ID
    token = (user_creds.get('token') if user_creds else None) or config.EMBY_TOKEN
    if not token:
        raise ValueError("Emby Token is required and could not be found.")
    if not user_id:
        raise ValueError("Emby User Identifier is required and could not be found.")

    try:
        final_playlist_name = f"{playlist_name.strip()}"

        ids_param = (
            ",".join(item_ids) if isinstance(item_ids, (list, set, tuple)) else str(item_ids)
        )
        url = (
            f"{config.EMBY_URL}/emby/Playlists"
            f"?Name={requests.utils.quote(final_playlist_name)}"
            f"&Ids={requests.utils.quote(ids_param)}"
            f"&UserId={user_id}"
            f"&MediaType=Audio"
        )

        headers = {"X-Emby-Token": token}

        r = requests.post(url, headers=headers, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()

        logger.info("Successfully created playlist '%s' for user %s.", final_playlist_name, user_id)
        return r.json()

    except requests.exceptions.RequestException:
        logger.exception(
            "HTTP Exception creating Emby playlist '%s' for user %s",
            playlist_name,
            user_id,
        )
        return None

    except Exception:
        logger.exception(
            "Generic exception creating Emby playlist '%s' for user %s",
            playlist_name,
            user_id,
        )
        return None


def get_all_playlists(user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.EMBY_USER_ID
    url = f"{config.EMBY_URL}/emby/Users/{user_id}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True}
    try:
        r = requests.get(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("Items") or []
    except Exception:
        logger.exception("Emby get_all_playlists failed")
        return []


def delete_playlist(playlist_id):
    url = f"{config.EMBY_URL}/emby/Items/Delete"
    params = {"Ids": playlist_id}
    try:
        r = requests.post(url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception:
        logger.exception(f"Exception deleting Emby playlist ID {playlist_id}")
        return False


def get_top_played_songs(limit, user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.EMBY_USER_ID
    token = user_creds.get('token') if user_creds else config.EMBY_TOKEN
    if not user_id or not token:
        raise ValueError("Emby User ID and Token are required.")

    url = f"{config.EMBY_URL}/emby/Users/{user_id}/Items"
    headers = {"X-Emby-Token": token}
    params = {
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
    except Exception:
        logger.exception(f"Emby get_top_played_songs failed for user {user_id}")
        return []


def get_last_played_time(item_id, user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.EMBY_USER_ID
    token = user_creds.get('token') if user_creds else config.EMBY_TOKEN
    if not user_id or not token:
        raise ValueError("Emby User ID and Token are required.")

    url = f"{config.EMBY_URL}/emby/Users/{user_id}/Items/{item_id}"
    headers = {"X-Emby-Token": token}
    params = {"Fields": "UserData"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("UserData", {}).get("LastPlayedDate")
    except Exception:
        logger.exception(
            f"Emby get_last_played_time failed for item {item_id}, user {user_id}",
        )
        return None


def get_lyrics(track_id: str, timeout: float = 2.5):
    try:
        url = f"{config.EMBY_URL}/emby/Items/{track_id}/Lyrics"
        r = requests.get(url, headers=config.HEADERS, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        lyrics_lines = data.get('Lyrics') or []
        if not lyrics_lines:
            return None
        text = '\n'.join(line.get('Text', '') for line in lyrics_lines if line.get('Text'))
        return text.strip() or None
    except Exception as exc:
        logger.debug('Emby get_lyrics failed for %s: %s', track_id, exc)
        return None


def create_instant_playlist(playlist_name, item_ids, user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.EMBY_USER_ID
    token = (user_creds.get('token') if user_creds else None) or config.EMBY_TOKEN
    if not token:
        raise ValueError("Emby Token is required and could not be found.")
    if not user_id:
        raise ValueError("Emby User user_id is required and could not be found.")

    try:
        final_playlist_name = f"{playlist_name.strip()}_instant"

        ids_param = (
            ",".join(item_ids) if isinstance(item_ids, (list, set, tuple)) else str(item_ids)
        )
        url = (
            f"{config.EMBY_URL}/emby/Playlists"
            f"?Name={requests.utils.quote(final_playlist_name)}"
            f"&Ids={requests.utils.quote(ids_param)}"
            f"&UserId={user_id}"
            f"&MediaType=Audio"
        )

        headers = {"X-Emby-Token": token}

        r = requests.post(url, headers=headers, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()

        logger.info("Successfully created playlist '%s' for user %s.", final_playlist_name, user_id)
        return r.json()

    except requests.exceptions.RequestException:
        logger.exception(
            "HTTP Exception creating Emby playlist '%s' for user %s",
            playlist_name,
            user_id,
        )
        return None

    except Exception:
        logger.exception(
            "Generic exception creating Emby playlist '%s' for user %s",
            playlist_name,
            user_id,
        )
        return None


def _fetch_playlist_items(playlist_id, user_id, headers):
    url = f"{config.EMBY_URL}/emby/Playlists/{playlist_id}/Items"
    params = {"UserId": user_id}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("Items") or []
    except Exception:
        logger.exception(f"Emby _fetch_playlist_items failed for {playlist_id}")
        return None


def _get_playlist_entry_ids(playlist_id, user_id, headers):
    items = _fetch_playlist_items(playlist_id, user_id, headers)
    if items is None:
        return None
    entry_ids = [it.get("PlaylistItemId") for it in items if it.get("PlaylistItemId")]
    if len(entry_ids) != len(items):
        logger.warning(
            f"Emby _get_playlist_entry_ids: playlist {playlist_id} had "
            f"{len(items) - len(entry_ids)} items missing PlaylistItemId - they will not be removed"
        )
    return entry_ids


def get_playlist_track_ids(playlist_id, user_creds=None):
    user_id = user_creds.get('user_id') if user_creds else config.EMBY_USER_ID
    headers = {"X-Emby-Token": user_creds.get('token')} if user_creds else config.HEADERS
    items = _fetch_playlist_items(playlist_id, user_id, headers)
    if not items:
        return []
    return [str(it.get("Id")) for it in items if it.get("Id")]


def _remove_playlist_entries(playlist_id, entry_ids, headers):
    if not entry_ids:
        return True
    url = f"{config.EMBY_URL}/emby/Playlists/{playlist_id}/Items"
    for i in range(0, len(entry_ids), EMBY_PLAYLIST_BATCH_SIZE):
        batch = entry_ids[i : i + EMBY_PLAYLIST_BATCH_SIZE]
        params = {"EntryIds": ",".join(batch)}
        try:
            r = requests.delete(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
        except Exception:
            logger.exception(
                f"Emby _remove_playlist_entries: batch starting at {i} failed for playlist {playlist_id}",
            )
            return False
    return True


def _add_items_to_playlist(playlist_id, item_ids, user_id, headers):
    if not item_ids:
        return True
    url = f"{config.EMBY_URL}/emby/Playlists/{playlist_id}/Items"
    for i in range(0, len(item_ids), EMBY_PLAYLIST_BATCH_SIZE):
        batch = item_ids[i : i + EMBY_PLAYLIST_BATCH_SIZE]
        params = {"Ids": ",".join(batch), "UserId": user_id}
        try:
            r = requests.post(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
        except Exception:
            logger.exception(
                f"Emby _add_items_to_playlist: batch starting at {i} failed for playlist {playlist_id}",
            )
            return False
    return True


def create_or_replace_playlist(playlist_name, item_ids, user_creds=None):
    if not item_ids:
        return None

    user_id = user_creds.get('user_id') if user_creds else config.EMBY_USER_ID
    token = (user_creds.get('token') if user_creds else None) or config.EMBY_TOKEN
    if not token or not user_id:
        logger.error("Emby create_or_replace_playlist: token or user_id missing")
        return None
    headers = {"X-Emby-Token": token}

    existing = get_playlist_by_name(playlist_name, user_creds=user_creds)
    if not existing:
        first_batch = item_ids[:EMBY_PLAYLIST_BATCH_SIZE]
        rest = item_ids[EMBY_PLAYLIST_BATCH_SIZE:]
        ids_param = ",".join(first_batch)
        url = (
            f"{config.EMBY_URL}/emby/Playlists"
            f"?Name={requests.utils.quote(playlist_name)}"
            f"&Ids={requests.utils.quote(ids_param)}"
            f"&UserId={user_id}"
            f"&MediaType=Audio"
        )
        try:
            r = requests.post(url, headers=headers, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
            created = r.json()
        except Exception:
            logger.exception(
                f"Emby create_or_replace_playlist: create failed for '{playlist_name}'",
            )
            return None

        new_id = created.get("Id")
        if not new_id:
            logger.error(
                f"Emby create_or_replace_playlist: created '{playlist_name}' but response had no Id"
            )
            return None

        if rest and not _add_items_to_playlist(new_id, rest, user_id, headers):
            logger.error(
                f"Emby create_or_replace_playlist: created '{playlist_name}' but failed to add overflow tracks"
            )

        logger.info(
            f"OK Emby: created playlist '{playlist_name}' (Id={new_id}) with {len(item_ids)} tracks"
        )
        return {**created, 'Id': new_id, 'Name': created.get('Name', playlist_name)}

    playlist_id = existing.get("Id")
    if not playlist_id:
        logger.error(
            f"Emby create_or_replace_playlist: existing playlist '{playlist_name}' has no Id"
        )
        return None

    entry_ids = _get_playlist_entry_ids(playlist_id, user_id, headers)
    if entry_ids is None:
        return None

    if not _remove_playlist_entries(playlist_id, entry_ids, headers):
        return None

    if not _add_items_to_playlist(playlist_id, item_ids, user_id, headers):
        logger.error(
            f"Emby create_or_replace_playlist: failed to add tracks to playlist {playlist_id}"
        )
        return None

    logger.info(
        f"OK Emby: replaced contents of playlist '{playlist_name}' (Id={playlist_id}, tracks={len(item_ids)})"
    )
    return {**existing, 'Id': playlist_id, 'Name': existing.get('Name', playlist_name)}
