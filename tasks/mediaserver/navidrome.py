# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Navidrome (Subsonic API) backend for the AudioMuse-AI media-server abstraction.

Implements the provider interface against a Navidrome/Subsonic server.
Dispatched by tasks/mediaserver/__init__.py when
config.MEDIASERVER_TYPE == 'navidrome'.

Main Features:
* Fetches albums/tracks, downloads, and manages playlists via the Subsonic API.
* Coerces single-dict responses to lists (Subsonic returns a dict, not a list,
  when only one item exists).
"""

from . import http as requests
import logging
import os
import random
import re
import config

from .helper import detect_path_format

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 300
NAVIDROME_API_BATCH_SIZE = 40


def _get_target_music_folder_ids(user_creds=None):
    folder_names_str = config.MUSIC_LIBRARIES

    if not folder_names_str.strip():
        return None

    target_names_lower = {
        name.strip().lower() for name in folder_names_str.split(',') if name.strip()
    }

    response = _navidrome_request("getMusicFolders", user_creds=user_creds)

    if not (response and "musicFolders" in response and "musicFolder" in response["musicFolders"]):
        logger.error("Failed to fetch music folders from Navidrome or response format unexpected.")
        return set()

    all_folders = _coerce_to_list(response["musicFolders"]["musicFolder"])

    folder_map = {
        folder['name'].lower(): {'name': folder['name'], 'id': folder['id']}
        for folder in all_folders
        if isinstance(folder, dict) and 'name' in folder and 'id' in folder
    }

    available_music_folders = [folder['name'] for folder in folder_map.values()]
    logger.info(f"Available Navidrome music folders found: {available_music_folders}")

    found_folders = []
    unfound_names = []
    for target_name in target_names_lower:
        if target_name in folder_map:
            found_folders.append(folder_map[target_name])
        else:
            unfound_names.append(target_name)

    if unfound_names:
        logger.warning(
            f"Navidrome config specified folder names that were not found: {list(unfound_names)}"
        )

    if not found_folders:
        logger.warning(
            f"No matching music folders found for configured names: {list(target_names_lower)}. No albums will be analyzed."
        )
        return set()

    music_folder_ids = {folder['id'] for folder in found_folders}
    found_names_original_case = [folder['name'] for folder in found_folders]

    logger.info(
        f"Filtering analysis to {len(music_folder_ids)} Navidrome folders: {found_names_original_case}"
    )
    return music_folder_ids


def list_libraries(user_creds=None):
    response = _navidrome_request("getMusicFolders", user_creds=user_creds)
    if not (response and "musicFolders" in response and "musicFolder" in response["musicFolders"]):
        return []
    all_folders = _coerce_to_list(response["musicFolders"]["musicFolder"])
    return [
        {'id': str(f['id']), 'name': f['name']}
        for f in all_folders
        if isinstance(f, dict) and 'id' in f and 'name' in f
    ]


def get_navidrome_auth_params(username=None, password=None):
    auth_user = username or config.NAVIDROME_USER
    auth_pass = password or config.NAVIDROME_PASSWORD
    if not auth_user or not auth_pass:
        logger.warning("Navidrome User or Password is not configured.")
        return {}
    hex_encoded_password = auth_pass.encode('utf-8').hex()
    return {
        "u": auth_user,
        "p": f"enc:{hex_encoded_password}",
        "v": "1.16.1",
        "c": "AudioMuse-AI",
        "f": "json",
    }


_SUBSONIC_AUTH_ERROR_CODES = {40, 41, 42, 43, 44}

_SECRET_QUERY_PARAM = re.compile(r'(?i)([?&][pst]=)[^&\s]*')


def _redact_navidrome_secrets(text):
    return _SECRET_QUERY_PARAM.sub(r'\1[REDACTED]', str(text))


def _navidrome_request_ex(
    endpoint, params=None, method='get', stream=False, user_creds=None, timeout=None
):
    params = params or {}
    auth_params = get_navidrome_auth_params(
        username=user_creds.get('user') if user_creds else None,
        password=user_creds.get('password') if user_creds else None,
    )
    if not auth_params:
        msg = "Navidrome username or password is not configured."
        logger.error(f"{msg} Cannot make API call.")
        return None, {'kind': 'config', 'message': msg}

    base_url = (
        user_creds.get('url') if user_creds and user_creds.get('url') else config.NAVIDROME_URL
    ).rstrip('/')
    url = f"{base_url}/rest/{endpoint}.view"
    all_params = {**auth_params, **params}

    try:
        r = requests.request(
            method, url, params=all_params, timeout=timeout or REQUESTS_TIMEOUT, stream=stream
        )
        r.raise_for_status()

        if stream:
            return r, None

        subsonic_response = r.json().get("subsonic-response", {})
        if subsonic_response.get("status") == "failed":
            error = subsonic_response.get("error", {}) or {}
            message = error.get("message") or "Navidrome returned an error."
            logger.error(f"Navidrome API Error on '{endpoint}': {message}")
            kind = 'auth' if error.get("code") in _SUBSONIC_AUTH_ERROR_CODES else 'server'
            return None, {'kind': kind, 'message': message}
        return subsonic_response, None

    except requests.exceptions.RequestException as e:
        safe = _redact_navidrome_secrets(e)
        logger.error(f"Error calling Navidrome API endpoint '{endpoint}': {safe}")  # noqa: TRY400 - .exception would leak the unredacted URL creds via the traceback
        return None, {'kind': 'network', 'message': safe}
    except Exception as e:
        safe = _redact_navidrome_secrets(e)
        logger.error(f"Unexpected error handling Navidrome response for '{endpoint}': {safe}")  # noqa: TRY400 - .exception would leak the unredacted URL creds via the traceback
        return None, {'kind': 'server', 'message': safe}


def _navidrome_request(
    endpoint, params=None, method='get', stream=False, user_creds=None, timeout=None
):
    data, _ = _navidrome_request_ex(
        endpoint,
        params=params,
        method=method,
        stream=stream,
        user_creds=user_creds,
        timeout=timeout,
    )
    return data


def download_track(temp_dir, item):
    try:
        track_id = item['id']

        file_extension = '.tmp'
        try:
            suffix = item.get('suffix')
            if suffix and isinstance(suffix, str) and suffix.strip():
                safe_suffix = suffix.strip().replace('/', '').replace('\\', '')
                if safe_suffix:
                    file_extension = f".{safe_suffix}"
                    logger.debug(f"Using suffix field for format: {file_extension}")
            elif item.get('path'):
                file_extension = os.path.splitext(item['path'])[1] or '.tmp'
        except Exception as e:
            logger.debug(f"Error getting format from suffix/path, using .tmp: {e}")

        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")

        response = _navidrome_request("stream", params={"id": track_id}, stream=True)
        if response:
            with response:
                with open(local_filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info(f"Downloaded '{item.get('title', 'Unknown')}' to '{local_filename}'")
            return local_filename
    except Exception as e:
        logger.error(  # noqa: TRY400 - .exception would leak the unredacted URL creds via the traceback
            f"Failed to download Navidrome track {item.get('title', 'Unknown')}: {_redact_navidrome_secrets(e)}"
        )  # noqa: TRY400 - .exception would leak the unredacted URL creds via the traceback
    return None


def get_recent_albums(limit):
    target_folder_ids = _get_target_music_folder_ids()

    if isinstance(target_folder_ids, set) and not target_folder_ids:
        logger.warning(
            "Folder filtering is active, but no matching folders were found on the server. Returning no albums."
        )
        return []

    all_albums = []
    fetch_all = limit == 0

    if target_folder_ids is None:
        logger.info("Scanning all Navidrome music folders for recent albums.")
        offset = 0
        page_size = 500
        while True:
            size_to_fetch = page_size if fetch_all else min(page_size, limit - len(all_albums))
            if size_to_fetch <= 0:
                break

            params = {"type": "newest", "size": size_to_fetch, "offset": offset}
            response = _navidrome_request("getAlbumList2", params)

            if response and "albumList2" in response and "album" in response["albumList2"]:
                albums = response["albumList2"]["album"]
                if not albums:
                    break

                all_albums.extend([{**a, 'Id': a.get('id'), 'Name': a.get('name')} for a in albums])
                offset += len(albums)

                if len(albums) < size_to_fetch:
                    break
            else:
                logger.error("Failed to fetch recent albums page from Navidrome.")
                break

    else:
        logger.info(
            f"Scanning {len(target_folder_ids)} specific Navidrome music folders for recent albums."
        )
        for folder_id in target_folder_ids:
            offset = 0
            page_size = 500
            while True:
                size_to_fetch = page_size if fetch_all else min(page_size, limit - len(all_albums))
                if size_to_fetch <= 0:
                    break

                params = {
                    "type": "newest",
                    "size": size_to_fetch,
                    "offset": offset,
                    "musicFolderId": folder_id,
                }
                response = _navidrome_request("getAlbumList2", params)

                if response and "albumList2" in response and "album" in response["albumList2"]:
                    albums = response["albumList2"]["album"]
                    if not albums:
                        break

                    all_albums.extend(
                        [{**a, 'Id': a.get('id'), 'Name': a.get('name')} for a in albums]
                    )
                    offset += len(albums)

                    if len(albums) < size_to_fetch:
                        break
                else:
                    logger.error(
                        f"Failed to fetch recent albums page from Navidrome folder ID {folder_id}."
                    )
                    break

    if not fetch_all:
        return all_albums[:limit]

    return all_albums


def _select_best_artist(song_item, title="Unknown"):
    if song_item.get('artist'):
        track_artist = song_item['artist']
        artist_id = song_item.get('artistId')
    elif song_item.get('albumArtist'):
        track_artist = song_item['albumArtist']
        artist_id = song_item.get('albumArtistId')
    else:
        track_artist = 'Unknown Artist'
        artist_id = None

    return track_artist, artist_id


def get_all_songs(user_creds=None, apply_filter=True):
    target_folder_ids = (
        _get_target_music_folder_ids(user_creds=user_creds) if apply_filter else None
    )

    if isinstance(target_folder_ids, set) and not target_folder_ids:
        logger.warning(
            "Folder filtering is active, but no matching folders were found on the server. Returning no songs."
        )
        return []

    all_songs = []

    if target_folder_ids is None:
        logger.info("Fetching all songs from all Navidrome music folders.")
        offset = 0
        limit = 500
        while True:
            params = {"query": '', "songCount": limit, "songOffset": offset}
            response = _navidrome_request("search3", params, user_creds=user_creds)
            if response and "searchResult3" in response and "song" in response["searchResult3"]:
                songs = response["searchResult3"]["song"]
                if not songs:
                    break

                for s in songs:
                    title = s.get('title', 'Unknown')
                    artist_name = s.get('artist', 'Unknown Artist')
                    artist_id = s.get('artistId')
                    raw_path = s.get('path') or s.get('url')
                    all_songs.append(
                        {
                            'Id': s.get('id'),
                            'Name': title,
                            'AlbumArtist': artist_name,
                            'ArtistId': artist_id,
                            'OriginalAlbumArtist': s.get('displayAlbumArtist')
                            or s.get('albumArtist'),
                            'Album': s.get('album'),
                            'Path': raw_path,
                            'Year': s.get('year'),
                            'Rating': s.get('userRating') if s.get('userRating') else None,
                            'FilePath': raw_path,
                        }
                    )

                offset += len(songs)
                if len(songs) < limit:
                    break
            else:
                logger.error("Failed to fetch all songs from Navidrome.")
                break

    else:
        logger.info(
            f"Fetching songs from {len(target_folder_ids)} specific Navidrome music folders."
        )

        target_albums = []
        for folder_id in target_folder_ids:
            offset = 0
            page_size = 500
            while True:
                params = {
                    "type": "newest",
                    "size": page_size,
                    "offset": offset,
                    "musicFolderId": folder_id,
                }
                response = _navidrome_request("getAlbumList2", params)

                if response and "albumList2" in response and "album" in response["albumList2"]:
                    albums = response["albumList2"]["album"]
                    if not albums:
                        break

                    target_albums.extend(albums)
                    offset += len(albums)

                    if len(albums) < page_size:
                        break
                else:
                    logger.error(f"Failed to fetch albums from Navidrome folder ID {folder_id}.")
                    break

        logger.info(
            f"Found {len(target_albums)} albums in specified folders. Getting songs from these albums."
        )

        for album in target_albums:
            album_id = album.get('id')
            if not album_id:
                continue

            album_songs = get_tracks_from_album(album_id, user_creds=user_creds)
            for song in album_songs:
                all_songs.append(
                    {
                        'Id': song.get('Id'),
                        'Name': song.get('Name'),
                        'AlbumArtist': song.get('AlbumArtist'),
                        'ArtistId': song.get('ArtistId'),
                        'OriginalAlbumArtist': song.get('OriginalAlbumArtist'),
                        'Album': song.get('Album'),
                        'Path': song.get('Path'),
                        'Year': song.get('Year'),
                        'Rating': song.get('Rating'),
                        'FilePath': song.get('FilePath'),
                    }
                )

    return all_songs


def search_albums(query, user_creds=None):
    body = _navidrome_request(
        "search3",
        {
            "query": query,
            "albumCount": 10,
            "songCount": 0,
            "artistCount": 0,
        },
        user_creds=user_creds,
    )
    if not body:
        return []
    albums = ((body.get('searchResult3') or {}).get('album')) or []
    return [
        {
            'id': a.get('id'),
            'name': a.get('name') or a.get('title'),
            'artist': a.get('artist'),
            'year': a.get('year'),
            'track_count': a.get('songCount'),
        }
        for a in albums
    ]


def _coerce_to_list(value):
    if isinstance(value, dict):
        return [value]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return []


def test_connection(user_creds=None):
    warnings = []
    body, err = _navidrome_request_ex(
        "search3",
        {
            "query": '',
            "songCount": 100,
            "songOffset": 0,
            "artistCount": 0,
            "albumCount": 0,
        },
        user_creds=user_creds,
    )
    if not body:
        return {
            'ok': False,
            'error': (err or {}).get('message') or 'Navidrome test_connection failed',
            'auth_failed': bool(err and err.get('kind') == 'auth'),
            'sample_count': 0,
            'path_format': 'none',
            'warnings': [],
        }
    songs = _coerce_to_list((body.get('searchResult3') or {}).get('song'))

    sample = []
    for s in songs:
        if not isinstance(s, dict):
            continue
        title = s.get('title', 'Unknown')
        track_artist = s.get('artist') or s.get('albumArtist') or 'Unknown Artist'
        sample.append(
            {
                'Id': s.get('id'),
                'Path': s.get('path') or s.get('url'),
                'Name': title,
                'AlbumArtist': s.get('albumArtist') or s.get('artist'),
                'artist': track_artist,
                'url': s.get('url'),
            }
        )
    path_format = detect_path_format(sample)
    if path_format != 'absolute':
        warnings.append(
            'Navidrome is returning relative paths or no paths at all. '
            'This happens when "Report Real Path" is disabled in Navidrome '
            '(Settings > Players > AudioMuse-AI [python-requests]). '
            'Automatic path-based matching will not work well. Enable Report '
            'Real Path and re-test, or you will need to manually match most '
            'albums in Step 4.'
        )
    return {
        'ok': True,
        'error': None,
        'sample_count': len(sample),
        'path_format': path_format,
        'warnings': warnings,
    }


def _add_to_playlist(playlist_id, item_ids, user_creds=None):
    if not item_ids:
        return True

    logger.info(f"Adding {len(item_ids)} songs to Navidrome playlist ID {playlist_id} in batches.")
    for i in range(0, len(item_ids), NAVIDROME_API_BATCH_SIZE):
        batch_ids = item_ids[i : i + NAVIDROME_API_BATCH_SIZE]
        params = {
            "playlistId": playlist_id,
            "songIdToAdd": batch_ids,
            "public": "true",
        }

        response = _navidrome_request(
            "updatePlaylist", params, method='post', user_creds=user_creds
        )

        if not (response and response.get("status") == "ok"):
            logger.error(
                f"Failed to add batch of {len(batch_ids)} songs to playlist {playlist_id}."
            )
            return False
    logger.info(f"Successfully added all songs to playlist {playlist_id}.")
    return True


def _create_playlist_batched(playlist_name, item_ids, user_creds=None):
    if not item_ids:
        item_ids = []

    ids_for_creation = item_ids[:NAVIDROME_API_BATCH_SIZE]
    ids_to_add_later = item_ids[NAVIDROME_API_BATCH_SIZE:]

    create_params = {
        "name": playlist_name,
        "songId": ids_for_creation,
    }
    create_response = _navidrome_request(
        "createPlaylist", create_params, method='post', user_creds=user_creds
    )

    if not (
        create_response and create_response.get("status") == "ok" and "playlist" in create_response
    ):
        logger.error(
            f"Failed to create Navidrome playlist '{playlist_name}' or API response was malformed."
        )
        return None

    new_playlist = create_response["playlist"]
    new_playlist_id = new_playlist.get("id")

    if not new_playlist_id:
        logger.error(
            f"Navidrome playlist '{playlist_name}' was created, but the response did not contain an ID."
        )
        return None

    logger.info(
        f"Created Navidrome playlist '{playlist_name}' (ID: {new_playlist_id}) with the first {len(ids_for_creation)} songs."
    )

    update_response = _navidrome_request(
        "updatePlaylist",
        {"playlistId": new_playlist_id, "public": "true"},
        method='post',
        user_creds=user_creds,
    )
    if not (update_response and update_response.get("status") == "ok"):
        logger.error(
            f"Failed to set playlist '{playlist_name}' public after creation via updatePlaylist."
        )

    if ids_to_add_later:
        if not _add_to_playlist(new_playlist_id, ids_to_add_later, user_creds):
            logger.error(
                f"Failed to add all songs to the new playlist '{playlist_name}'. The playlist was created but may be incomplete."
            )

    new_playlist['Id'] = new_playlist.get('id')
    new_playlist['Name'] = new_playlist.get('name')

    return new_playlist


def create_playlist(base_name, item_ids):
    _create_playlist_batched(base_name, item_ids, user_creds=None)


def get_all_playlists():
    response = _navidrome_request("getPlaylists")
    if response and "playlists" in response and "playlist" in response["playlists"]:
        return [
            {**p, 'Id': p.get('id'), 'Name': p.get('name')}
            for p in response["playlists"]["playlist"]
        ]
    return []


def delete_playlist(playlist_id):
    response = _navidrome_request("deletePlaylist", {"id": playlist_id}, method='post')
    if response and response.get("status") == "ok":
        logger.info(f"Deleted Navidrome playlist ID: {playlist_id}")
        return True
    logger.error(f"Failed to delete playlist ID '{playlist_id}' on Navidrome")
    return False


def get_tracks_from_album(album_id, user_creds=None):
    params = {"id": album_id}
    response = _navidrome_request("getAlbum", params, user_creds=user_creds)
    if response and "album" in response and "song" in response["album"]:
        songs = response["album"]["song"]

        result = []
        for s in songs:
            title = s.get('title', 'Unknown')
            artist, artist_id = _select_best_artist(s, title)
            logger.debug(
                f"getAlbum track '{title}': artist='{artist}', artist_id='{artist_id}', raw_artistId='{s.get('artistId')}', raw_albumArtistId='{s.get('albumArtistId')}'"
            )
            raw_path = s.get('path') or s.get('url')
            result.append(
                {
                    **s,
                    'Id': s.get('id'),
                    'Name': title,
                    'AlbumArtist': artist,
                    'ArtistId': artist_id,
                    'OriginalAlbumArtist': s.get('displayAlbumArtist') or s.get('albumArtist'),
                    'Album': s.get('album'),
                    'Path': raw_path,
                    'Year': s.get('year'),
                    'Rating': s.get('userRating') if s.get('userRating') else None,
                    'FilePath': raw_path,
                }
            )
        return result
    return []


def get_playlist_by_name(playlist_name, user_creds=None):
    response = _navidrome_request("getPlaylists", user_creds=user_creds)
    if not (response and "playlists" in response and "playlist" in response["playlists"]):
        return None

    for playlist_summary in response["playlists"]["playlist"]:
        if playlist_summary.get("name") == playlist_name:
            return playlist_summary

    return None


def _get_playlist_detail(playlist_id, user_creds=None):
    detail = _navidrome_request("getPlaylist", {"id": playlist_id}, user_creds=user_creds)
    if not (detail and "playlist" in detail):
        return None
    return detail["playlist"]


def get_playlist_track_ids(playlist_id, user_creds=None):
    playlist = _get_playlist_detail(playlist_id, user_creds=user_creds)
    if not playlist:
        return []
    entries = playlist.get("entry")
    if isinstance(entries, dict):
        entries = [entries]
    elif not isinstance(entries, list):
        entries = []
    return [str(e.get("id")) for e in entries if e.get("id")]


def get_top_played_songs(limit, user_creds):
    all_top_songs = []
    per_album_cap = max(1, config.SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM)
    num_albums_to_fetch = (limit // per_album_cap) + 10
    params = {"type": "frequent", "size": num_albums_to_fetch}
    response = _navidrome_request("getAlbumList2", params, user_creds=user_creds)
    if response and "albumList2" in response and "album" in response["albumList2"]:
        albums = response["albumList2"]["album"]
        if isinstance(albums, dict):
            albums = [albums]
        elif not isinstance(albums, list):
            albums = []
        for album in albums:
            tracks = get_tracks_from_album(album.get("id"), user_creds=user_creds)
            if not tracks:
                continue
            if len(tracks) > per_album_cap:
                tracks = random.sample(tracks, per_album_cap)
            all_top_songs.extend(tracks)
    all_top_songs.sort(
        key=lambda song: song.get('played') or song.get('lastPlayed') or '', reverse=True
    )
    return all_top_songs[:limit]


def get_last_played_time(item_id, user_creds):
    response = _navidrome_request("getSong", {"id": item_id}, user_creds=user_creds)
    if response and "song" in response:
        song = response.get("song")
        if isinstance(song, dict):
            return song.get("played") or song.get("lastPlayed")
    return None


def get_lyrics(track_id: str, timeout: float = 2.5):
    try:
        response = _navidrome_request('getLyricsBySongId', params={'id': track_id}, timeout=timeout)
        if not response:
            return None
        structured = response.get('lyricsList', {}).get('structuredLyrics', [])
        if not structured:
            return None
        chosen = next((e for e in structured if not e.get('synced')), structured[0])
        lines = [ln.get('value', '') for ln in chosen.get('line', []) if ln.get('value')]
        text = '\n'.join(lines)
        return text.strip() or None
    except Exception as exc:
        logger.debug('Navidrome get_lyrics failed for %s: %s', track_id, exc)
        return None


def create_instant_playlist(playlist_name, item_ids, user_creds):
    final_playlist_name = f"{playlist_name.strip()}_instant"
    return _create_playlist_batched(final_playlist_name, item_ids, user_creds)


def _clear_playlist_items(playlist_id, user_creds=None):
    playlist = _get_playlist_detail(playlist_id, user_creds=user_creds)
    if not playlist:
        logger.error(f"Navidrome _clear_playlist_items: failed to fetch playlist {playlist_id}")
        return False

    song_count = int(playlist.get("songCount") or 0)
    if song_count == 0:
        return True

    indices = list(range(song_count - 1, -1, -1))
    for i in range(0, len(indices), NAVIDROME_API_BATCH_SIZE):
        batch = indices[i : i + NAVIDROME_API_BATCH_SIZE]
        params = {
            "playlistId": playlist_id,
            "songIndexToRemove": batch,
        }
        response = _navidrome_request(
            "updatePlaylist", params, method='post', user_creds=user_creds
        )
        if not (response and response.get("status") == "ok"):
            logger.error(
                f"Navidrome _clear_playlist_items: failed to remove batch starting at index {batch[0]}"
            )
            return False
    return True


def create_or_replace_playlist(playlist_name, item_ids, user_creds=None):
    if not item_ids:
        return None

    existing = get_playlist_by_name(playlist_name, user_creds=user_creds)
    if not existing:
        return _create_playlist_batched(playlist_name, item_ids, user_creds=user_creds)

    playlist_id = existing.get("id")
    if not playlist_id:
        logger.error(
            f"Navidrome create_or_replace_playlist: existing playlist '{playlist_name}' has no id"
        )
        return None

    if not _clear_playlist_items(playlist_id, user_creds=user_creds):
        return None

    if not _add_to_playlist(playlist_id, item_ids, user_creds=user_creds):
        logger.error(
            f"Navidrome create_or_replace_playlist: failed to add tracks to playlist {playlist_id}"
        )
        return None

    return {
        **existing,
        'Id': playlist_id,
        'Name': existing.get('name') or playlist_name,
    }
