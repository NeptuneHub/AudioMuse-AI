# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""K7 backend for the AudioMuse-AI media-server abstraction.

Implements the provider interface against a K7 server's REST API using
X-Api-Key auth. Dispatched by tasks/mediaserver/__init__.py when
config.MEDIASERVER_TYPE == 'k7'.

K7 wire JSON is camelCase with string enums. Responses are normalised to the
Jellyfin-style PascalCase fields the rest of AudioMuse expects (Id, Name, ...).

Main Features:
* Honours MUSIC_LIBRARIES via /api/libraries (music media type only).
* Fetches recent albums, album tracks and all songs with page-based pagination.
* Downloads tracks via direct-stream, falling back to downloads/prepare when
  StreamAccess rejects ApiKey clients.
* Reads play stats/lyrics and manages playlists through the K7 REST API.
"""

from . import http as requests
import logging
import os
import time
import uuid

import config
from . import context
from .helper import detect_path_format, is_auth_error

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 300
K7_PAGE_SIZE = 500
# Stable device id for downloads/prepare (ApiKey clients need an ephemeral download).
_AUDIOMUSE_DEVICE_ID = "a11d1000-0000-4000-8000-00000000a0d7"


def _pick(data, *keys, default=None):
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    lower = {str(k).lower(): v for k, v in data.items()}
    for key in keys:
        value = lower.get(str(key).lower())
        if value is not None:
            return value
    return default


def _items(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return _pick(payload, "items", "Items", default=[]) or []
    return []


def _is_music_library(media_type):
    return media_type in (2, "2", "Music", "music")


def _k7_base_url(user_creds=None):
    creds = context.active_creds(user_creds)
    url = (creds.get("url") if creds and creds.get("url") else None) or config.K7_URL
    return url.rstrip("/") if url else ""


def _k7_headers(user_creds=None):
    creds = context.active_creds(user_creds)
    api_key = (creds.get("api_key") if creds and creds.get("api_key") else None) or config.K7_API_KEY
    return {"X-Api-Key": api_key, "Accept": "application/json"}


def _get_target_library_ids():
    library_names_str = context.active_libraries(config.MUSIC_LIBRARIES)
    if not library_names_str.strip():
        return None

    target_names = {n.strip().lower() for n in library_names_str.split(",") if n.strip()}
    if not target_names:
        return None

    libraries = list_libraries()
    if not libraries:
        return set()

    matched = {
        lib["id"]
        for lib in libraries
        if (lib.get("name") or "").lower() in target_names
    }
    return matched if matched else set()


def _normalize_track(item):
    """Map a K7 track DTO onto the AudioMuse catalogue shape."""
    if not isinstance(item, dict):
        return item

    track_id = str(_pick(item, "id", "Id", default="") or "")
    title = _pick(item, "title", "Title", "Name", default="Unknown") or "Unknown"
    artist = _pick(item, "artistName", "ArtistName", "AlbumArtist", default="Unknown Artist")
    artist_id = _pick(item, "artistId", "ArtistId")
    album = _pick(item, "albumTitle", "AlbumTitle", "Album", default="") or ""
    release = _pick(item, "releaseDate", "ReleaseDate")
    year = None
    if isinstance(release, str) and len(release) >= 4:
        year = release[:4]
    elif _pick(item, "year", "Year") is not None:
        year = _pick(item, "year", "Year")

    indexed_file_id = _pick(item, "indexedFileId", "IndexedFileId")
    indexed_files = _pick(item, "indexedFiles", "IndexedFiles", default=[]) or []
    file_path = _pick(item, "path", "Path", "filePath", "FilePath")
    if not file_path and isinstance(indexed_files, list) and indexed_files:
        first = indexed_files[0] if isinstance(indexed_files[0], dict) else None
        if first:
            file_path = _pick(first, "path", "Path")
            if not indexed_file_id:
                indexed_file_id = _pick(first, "id", "Id")

    duration = _pick(item, "duration", "Duration", "DurationSeconds")

    item["Id"] = track_id
    item["Name"] = title
    item["AlbumArtist"] = artist or "Unknown Artist"
    item["ArtistId"] = str(artist_id) if artist_id else None
    item["Album"] = album
    item["Path"] = file_path
    item["FilePath"] = file_path
    item["Year"] = year
    item["IndexNumber"] = _pick(item, "trackNumber", "TrackNumber", "IndexNumber")
    item["IndexedFileId"] = str(indexed_file_id) if indexed_file_id else None
    item["Container"] = None
    if duration is not None:
        try:
            item["DurationSeconds"] = float(duration)
        except (TypeError, ValueError):
            pass
    return item


def _normalize_album(item):
    if not isinstance(item, dict):
        return item
    item["Id"] = str(_pick(item, "id", "Id", default="") or "")
    item["Name"] = _pick(item, "title", "Title", "Name", default="Unknown") or "Unknown"
    item["DateCreated"] = _pick(item, "created", "Created", "DateCreated", default="") or ""
    return item


def list_libraries(user_creds=None):
    user_creds = context.active_creds(user_creds)
    base_url = _k7_base_url(user_creds)
    url = f"{base_url}/api/libraries"
    try:
        r = requests.get(url, headers=_k7_headers(user_creds), timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        libraries = r.json() or []
        return [
            {
                "id": str(_pick(lib, "id", "Id")),
                "name": _pick(lib, "title", "Title", "Name"),
            }
            for lib in libraries
            if isinstance(lib, dict) and _is_music_library(_pick(lib, "mediaType", "MediaType"))
        ]
    except Exception:
        logger.exception("K7 list_libraries failed")
        return []


def test_connection(user_creds=None):
    user_creds = context.active_creds(user_creds)
    try:
        base_url = _k7_base_url(user_creds)
        url = f"{base_url}/api/medias"
        params = {
            "MediaTypes": "MusicTrack",
            "PageNumber": 1,
            "PageSize": 100,
        }
        r = requests.get(
            url, headers=_k7_headers(user_creds), params=params, timeout=REQUESTS_TIMEOUT
        )
        r.raise_for_status()
        items = _items(r.json())

        sample = []
        for item in items:
            track = _normalize_track(dict(item) if isinstance(item, dict) else item)
            sample.append(
                {
                    "Id": track.get("Id"),
                    "Path": track.get("Path"),
                    "Name": track.get("Name"),
                    "AlbumArtist": track.get("AlbumArtist"),
                }
            )

        path_format = detect_path_format(sample)
        return {
            "ok": True,
            "error": None,
            "sample_count": len(sample),
            "path_format": path_format,
            "warnings": [],
        }
    except Exception as e:
        if is_auth_error(e):
            logger.warning("K7 test_connection auth failed: %s", e)
        else:
            logger.warning("K7 test_connection failed: %s", e)
        return {
            "ok": False,
            "error": str(e),
            "sample_count": 0,
            "path_format": "none",
            "warnings": [],
        }


def get_recent_albums(limit):
    target_library_ids = _get_target_library_ids()
    if isinstance(target_library_ids, set) and not target_library_ids:
        logger.warning("K7: Library filtering active but no matching libraries found.")
        return []

    all_albums = []
    fetch_all = limit == 0
    page_number = 1

    library_params = {}
    if target_library_ids:
        library_params = {"LibraryIds": list(target_library_ids)}

    while True:
        url = f"{_k7_base_url()}/api/medias"
        params = {
            "MediaTypes": "MusicAlbum",
            "OrderBy": "CreatedDesc",
            "PageNumber": page_number,
            "PageSize": K7_PAGE_SIZE,
            **library_params,
        }
        try:
            r = requests.get(url, headers=_k7_headers(), params=params, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
            albums_on_page = _items(r.json())
            if not albums_on_page:
                break

            for album in albums_on_page:
                all_albums.append(_normalize_album(album))

            page_number += 1
            if len(albums_on_page) < K7_PAGE_SIZE:
                break
            if not fetch_all and len(all_albums) >= limit:
                break
        except Exception:
            logger.exception("K7 get_recent_albums failed")
            break

    if not fetch_all and limit > 0:
        all_albums = all_albums[:limit]
    return all_albums


def get_tracks_from_album(album_id, user_creds=None):
    user_creds = context.active_creds(user_creds)
    url = f"{_k7_base_url(user_creds)}/api/medias/{album_id}"
    try:
        r = requests.get(url, headers=_k7_headers(user_creds), timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        data = r.json() or {}
        tracks = _pick(data, "tracks", "Tracks", default=[]) or []
        return [_normalize_track(t) for t in tracks if isinstance(t, dict)]
    except Exception:
        logger.exception("K7 get_tracks_from_album failed for album %s", album_id)
        return []


def _content_type_to_ext(content_type):
    mapping = {
        "audio/flac": ".flac",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mp4": ".m4a",
        "audio/aac": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/vorbis": ".ogg",
        "audio/opus": ".opus",
    }
    ct = (content_type or "").split(";")[0].strip().lower()
    return mapping.get(ct, ".tmp")


def _download_via_direct_stream(temp_dir, track_id, indexed_file_id, title):
    download_url = f"{_k7_base_url()}/api/indexed-files/{indexed_file_id}/direct-stream"
    local_filename = os.path.join(temp_dir, f"{track_id}.tmp")
    with requests.get(
        download_url, headers=_k7_headers(), stream=True, timeout=REQUESTS_TIMEOUT
    ) as r:
        r.raise_for_status()
        ext = _content_type_to_ext(r.headers.get("Content-Type", ""))
        if ext and ext != ".tmp":
            local_filename = os.path.join(temp_dir, f"{track_id}{ext}")
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    logger.info("Downloaded '%s' via direct-stream to '%s'", title, local_filename)
    return local_filename


def _download_via_prepare(temp_dir, track_id, indexed_file_id, title):
    """Fallback when StreamAccess rejects ApiKey: prepare + file endpoints."""
    headers = _k7_headers()
    prepare_url = f"{_k7_base_url()}/api/downloads/prepare"
    body = {
        "indexedFileId": indexed_file_id,
        "deviceId": _AUDIOMUSE_DEVICE_ID,
    }
    r = requests.post(prepare_url, headers=headers, json=body, timeout=REQUESTS_TIMEOUT)
    r.raise_for_status()
    download = r.json() or {}
    download_id = _pick(download, "id", "Id")
    if not download_id:
        raise RuntimeError("K7 prepare download returned no id")

    status = _pick(download, "status", "Status", default="")
    deadline = time.monotonic() + 120
    while str(status).lower() not in ("ready", "failed") and time.monotonic() < deadline:
        time.sleep(0.5)
        status_r = requests.get(
            f"{_k7_base_url()}/api/downloads/{download_id}",
            headers=headers,
            timeout=REQUESTS_TIMEOUT,
        )
        status_r.raise_for_status()
        download = status_r.json() or {}
        status = _pick(download, "status", "Status", default="")

    if str(status).lower() != "ready":
        reason = _pick(download, "failureReason", "FailureReason", default=status)
        raise RuntimeError(f"K7 download not ready: {reason}")

    file_url = f"{_k7_base_url()}/api/downloads/{download_id}/file"
    local_filename = os.path.join(temp_dir, f"{track_id}.tmp")
    with requests.get(file_url, headers=headers, stream=True, timeout=REQUESTS_TIMEOUT) as fr:
        fr.raise_for_status()
        ext = _content_type_to_ext(
            fr.headers.get("Content-Type") or _pick(download, "contentType", "ContentType", default="")
        )
        if ext and ext != ".tmp":
            local_filename = os.path.join(temp_dir, f"{track_id}{ext}")
        with open(local_filename, "wb") as f:
            for chunk in fr.iter_content(chunk_size=8192):
                f.write(chunk)

    logger.info("Downloaded '%s' via prepare/file to '%s'", title, local_filename)
    return local_filename


def download_track(temp_dir, item):
    try:
        indexed_file_id = _pick(item, "IndexedFileId", "indexedFileId")
        track_id = str(_pick(item, "Id", "id", default=uuid.uuid4()) or uuid.uuid4())
        title = _pick(item, "Name", "title", "Title", default="Unknown") or "Unknown"

        if not indexed_file_id:
            logger.error("K7: Track '%s' has no IndexedFileId, cannot download.", title)
            return None

        try:
            return _download_via_direct_stream(temp_dir, track_id, indexed_file_id, title)
        except Exception as direct_err:
            if not is_auth_error(direct_err):
                raise
            logger.info(
                "K7 direct-stream rejected ApiKey for '%s'; falling back to downloads API",
                title,
            )
            return _download_via_prepare(temp_dir, track_id, indexed_file_id, title)
    except Exception:
        logger.exception("K7: Failed to download track %s", _pick(item, "Name", "title", default="Unknown"))
        return None


def get_all_songs(user_creds=None, apply_filter=True):
    user_creds = context.active_creds(user_creds)
    target_library_ids = _get_target_library_ids() if apply_filter else None
    if isinstance(target_library_ids, set) and not target_library_ids:
        logger.warning("K7: Library filtering active but no matching libraries found.")
        return []

    all_songs = []
    page_number = 1
    library_params = {}
    if target_library_ids:
        library_params = {"LibraryIds": list(target_library_ids)}

    while True:
        url = f"{_k7_base_url(user_creds)}/api/medias"
        params = {
            "MediaTypes": "MusicTrack",
            "PageNumber": page_number,
            "PageSize": K7_PAGE_SIZE,
            **library_params,
        }
        try:
            r = requests.get(
                url, headers=_k7_headers(user_creds), params=params, timeout=REQUESTS_TIMEOUT
            )
            r.raise_for_status()
            data = r.json() or {}
            items = _items(data)
            if not items:
                break

            for item in items:
                all_songs.append(_normalize_track(item))

            page_number += 1
            total_count = _pick(data, "totalCount", "TotalCount", default=0) or 0
            if page_number * K7_PAGE_SIZE > total_count + K7_PAGE_SIZE:
                break
            if len(items) < K7_PAGE_SIZE:
                break
        except Exception:
            logger.exception("K7 get_all_songs failed on page %s", page_number)
            break

    return all_songs


def search_albums(query, user_creds=None):
    user_creds = context.active_creds(user_creds)
    url = f"{_k7_base_url(user_creds)}/api/medias"
    params = {
        "MediaTypes": "MusicAlbum",
        "SearchText": query,
        "PageNumber": 1,
        "PageSize": 10,
    }
    try:
        r = requests.get(
            url, headers=_k7_headers(user_creds), params=params, timeout=REQUESTS_TIMEOUT
        )
        r.raise_for_status()
        items = _items(r.json())
        results = []
        for item in items:
            release = _pick(item, "releaseDate", "ReleaseDate")
            year = release[:4] if isinstance(release, str) and len(release) >= 4 else None
            results.append(
                {
                    "id": str(_pick(item, "id", "Id")),
                    "name": _pick(item, "title", "Title", "Name"),
                    "artist": _pick(item, "artistName", "ArtistName"),
                    "year": year,
                    "track_count": None,
                }
            )
        return results
    except Exception:
        logger.exception("K7 search_albums failed")
        return []


def get_all_playlists():
    all_playlists = []
    page_number = 1
    page_size = 200
    while True:
        url = f"{_k7_base_url()}/api/playlists"
        params = {"PageNumber": page_number, "PageSize": page_size, "MediaType": "MusicTrack"}
        try:
            r = requests.get(url, headers=_k7_headers(), params=params, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
            items = _items(r.json())
            if not items:
                break
            for item in items:
                all_playlists.append(
                    {
                        "Id": str(_pick(item, "id", "Id")),
                        "Name": _pick(item, "title", "Title", "Name", default="") or "",
                    }
                )
            if len(items) < page_size:
                break
            page_number += 1
        except Exception:
            logger.exception("K7 get_all_playlists failed")
            break
    return all_playlists


def delete_playlist(playlist_id):
    url = f"{_k7_base_url()}/api/playlists/{playlist_id}"
    try:
        r = requests.delete(url, headers=_k7_headers(), timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        logger.info("Deleted K7 playlist %s", playlist_id)
        return True
    except Exception:
        logger.exception("K7 delete_playlist failed for %s", playlist_id)
        return False


def get_playlist_by_name(playlist_name):
    for playlist in get_all_playlists():
        if playlist.get("Name") == playlist_name:
            return playlist
    return None


def get_playlist_track_ids(playlist_id, user_creds=None):
    user_creds = context.active_creds(user_creds)
    track_ids = []
    page_number = 1
    page_size = 200
    while True:
        url = f"{_k7_base_url(user_creds)}/api/playlists/{playlist_id}/items"
        params = {"PageNumber": page_number, "PageSize": page_size}
        try:
            r = requests.get(
                url, headers=_k7_headers(user_creds), params=params, timeout=REQUESTS_TIMEOUT
            )
            r.raise_for_status()
            items = _items(r.json())
            if not items:
                break
            for item in items:
                media_id = _pick(item, "mediaId", "MediaId")
                if media_id:
                    track_ids.append(str(media_id))
            if len(items) < page_size:
                break
            page_number += 1
        except Exception:
            logger.exception("K7 get_playlist_track_ids failed for %s", playlist_id)
            break
    return track_ids


def _create_playlist(playlist_name, item_ids, user_creds=None):
    user_creds = context.active_creds(user_creds)
    url = f"{_k7_base_url(user_creds)}/api/playlists"
    body = {"title": playlist_name, "mediaType": "MusicTrack"}
    try:
        r = requests.post(
            url, headers=_k7_headers(user_creds), json=body, timeout=REQUESTS_TIMEOUT
        )
        r.raise_for_status()
        playlist_id = str(r.json())
        if item_ids and playlist_id:
            _add_items_to_playlist(playlist_id, item_ids, user_creds)
        logger.info(
            "Created K7 playlist '%s' with %s tracks", playlist_name, len(item_ids or [])
        )
        return playlist_id
    except Exception:
        logger.exception("K7 create playlist failed for '%s'", playlist_name)
        return None


def create_playlist(base_name, item_ids):
    return _create_playlist(base_name, item_ids)


def create_instant_playlist(playlist_name, item_ids, user_creds=None):
    return _create_playlist(f"{playlist_name.strip()}_instant", item_ids, user_creds)


def create_or_replace_playlist(playlist_name, item_ids, user_creds=None):
    user_creds = context.active_creds(user_creds)
    existing = get_playlist_by_name(playlist_name)
    if existing:
        playlist_id = existing["Id"]
        _clear_playlist_items(playlist_id, user_creds)
        _add_items_to_playlist(playlist_id, item_ids, user_creds)
        logger.info("Replaced K7 playlist '%s' with %s tracks", playlist_name, len(item_ids or []))
        return playlist_id
    return _create_playlist(playlist_name, item_ids, user_creds)


def get_top_played_songs(limit, user_creds=None):
    user_creds = context.active_creds(user_creds)
    page_size = limit if limit and limit > 0 else 100

    top_url = f"{_k7_base_url(user_creds)}/api/music/top-tracks"
    try:
        params = {"Count": page_size}
        r = requests.get(
            top_url, headers=_k7_headers(user_creds), params=params, timeout=REQUESTS_TIMEOUT
        )
        if r.status_code == 200:
            payload = r.json() or []
            results = []
            for entry in payload if isinstance(payload, list) else []:
                track = _pick(entry, "track", "Track", default=entry) or {}
                play_count = _pick(entry, "playCount", "PlayCount", default=0) or 0
                track_id = str(_pick(track, "id", "Id", default="") or "")
                if not track_id:
                    continue
                results.append(
                    {
                        "Id": track_id,
                        "Name": _pick(track, "title", "Title", "Name", default="Unknown"),
                        "AlbumArtist": _pick(
                            track, "artistName", "ArtistName", default="Unknown Artist"
                        ),
                        "PlayCount": play_count,
                    }
                )
            if results:
                return results[:page_size]
    except Exception:
        logger.debug("K7 /api/music/top-tracks unavailable; falling back to medias sort", exc_info=True)

    url = f"{_k7_base_url(user_creds)}/api/medias"
    params = {
        "MediaTypes": "MusicTrack",
        "OrderBy": "PlayCountDesc",
        "PageNumber": 1,
        "PageSize": page_size,
    }
    try:
        r = requests.get(
            url, headers=_k7_headers(user_creds), params=params, timeout=REQUESTS_TIMEOUT
        )
        r.raise_for_status()
        items = _items(r.json())
        results = []
        for item in items:
            user_state = _pick(item, "userState", "UserState", default={}) or {}
            play_count = _pick(user_state, "playCount", "PlayCount", default=0) or 0
            track_id = str(_pick(item, "id", "Id", default="") or "")
            if not track_id:
                continue
            results.append(
                {
                    "Id": track_id,
                    "Name": _pick(item, "title", "Title", "Name", default="Unknown"),
                    "AlbumArtist": _pick(
                        item, "artistName", "ArtistName", default="Unknown Artist"
                    ),
                    "PlayCount": play_count,
                }
            )
        # Empty PlayCountDesc (fresh library) would fail the setup probe; retry unordered.
        if not results:
            params.pop("OrderBy", None)
            r = requests.get(
                url, headers=_k7_headers(user_creds), params=params, timeout=REQUESTS_TIMEOUT
            )
            r.raise_for_status()
            for item in _items(r.json()):
                track_id = str(_pick(item, "id", "Id", default="") or "")
                if not track_id:
                    continue
                results.append(
                    {
                        "Id": track_id,
                        "Name": _pick(item, "title", "Title", "Name", default="Unknown"),
                        "AlbumArtist": _pick(
                            item, "artistName", "ArtistName", default="Unknown Artist"
                        ),
                        "PlayCount": 0,
                    }
                )
        return results[:page_size]
    except Exception:
        logger.exception("K7 get_top_played_songs failed")
        return []


def get_last_played_time(item_id, user_creds=None):
    user_creds = context.active_creds(user_creds)
    url = f"{_k7_base_url(user_creds)}/api/medias/{item_id}"
    try:
        r = requests.get(url, headers=_k7_headers(user_creds), timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        data = r.json() or {}
        user_state = _pick(data, "userState", "UserState", default={}) or {}
        return _pick(user_state, "lastInteractedAt", "LastInteractedAt")
    except Exception:
        logger.exception("K7 get_last_played_time failed for %s", item_id)
        return None


def get_lyrics(track_id: str, timeout: float = 2.5):
    url = f"{_k7_base_url()}/api/medias/{track_id}"
    try:
        r = requests.get(url, headers=_k7_headers(), timeout=timeout or REQUESTS_TIMEOUT)
        r.raise_for_status()
        data = r.json() or {}
        lrc = _pick(data, "lyricsLrc", "LyricsLrc")
        plain = _pick(data, "lyrics", "Lyrics")
        return lrc or plain
    except Exception:
        logger.debug("K7 get_lyrics failed for %s", track_id, exc_info=True)
        return None


def _add_items_to_playlist(playlist_id, item_ids, user_creds=None):
    user_creds = context.active_creds(user_creds)
    headers = _k7_headers(user_creds)
    for media_id in item_ids or []:
        url = f"{_k7_base_url(user_creds)}/api/playlists/{playlist_id}/items"
        body = {"mediaId": str(media_id)}
        try:
            r = requests.post(url, headers=headers, json=body, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
        except Exception as e:
            logger.warning("K7: Failed to add item %s to playlist %s: %s", media_id, playlist_id, e)


def _clear_playlist_items(playlist_id, user_creds=None):
    user_creds = context.active_creds(user_creds)
    headers = _k7_headers(user_creds)
    page_number = 1
    page_size = 200
    all_item_ids = []

    while True:
        url = f"{_k7_base_url(user_creds)}/api/playlists/{playlist_id}/items"
        params = {"PageNumber": page_number, "PageSize": page_size}
        try:
            r = requests.get(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
            items = _items(r.json())
            if not items:
                break
            for item in items:
                item_id = _pick(item, "id", "Id")
                if item_id:
                    all_item_ids.append(str(item_id))
            if len(items) < page_size:
                break
            page_number += 1
        except Exception:
            logger.exception("K7: Failed to get playlist items for clear")
            break

    for item_id in all_item_ids:
        url = f"{_k7_base_url(user_creds)}/api/playlists/{playlist_id}/items/{item_id}"
        try:
            r = requests.delete(url, headers=headers, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
        except Exception as e:
            logger.warning("K7: Failed to remove playlist item %s: %s", item_id, e)
