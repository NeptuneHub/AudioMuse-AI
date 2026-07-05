# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Plex backend for the AudioMuse-AI media-server abstraction.

Implements the provider interface against a Plex Media Server's HTTP API
(http[s]://host:32400/..., docs at https://plexapi.dev/) using X-Plex-Token
auth and JSON responses. Dispatched by tasks/mediaserver/__init__.py when
config.MEDIASERVER_TYPE == 'plex'.

Main Features:
* Auto-discovers music library sections (type 'artist') and honours MUSIC_LIBRARIES.
* Fetches recent albums, album tracks and all songs with container pagination.
* Downloads tracks, reads play stats/lyrics and manages audio playlists via
  server:// metadata URIs.
"""

from datetime import datetime, timezone

from . import http as requests
import logging
import os
import config

from .helper import detect_path_format, detect_download_extension, is_auth_error

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 300
PLEX_PAGE_SIZE = 1000
PLEX_PLAYLIST_BATCH_SIZE = 100
PLEX_ALBUM_TYPE = 9
PLEX_TRACK_TYPE = 10
_LYRIC_STREAM_TYPE = 4

_MACHINE_ID_CACHE = {}


def _base_url(user_creds=None):
    return (
        user_creds.get('url') if user_creds and user_creds.get('url') else config.PLEX_URL
    ).rstrip('/')


def _headers(user_creds=None):
    token = (user_creds.get('token') if user_creds else None) or config.PLEX_TOKEN
    headers = {'Accept': 'application/json'}
    if token:
        headers['X-Plex-Token'] = token
    return headers


def _container(payload):
    if isinstance(payload, dict):
        mc = payload.get('MediaContainer')
        if isinstance(mc, dict):
            return mc
    return {}


def _first_part(item):
    media = item.get('Media') if isinstance(item, dict) else None
    if not media:
        return None
    parts = media[0].get('Part') if isinstance(media[0], dict) else None
    if not parts:
        return None
    return parts[0] if isinstance(parts[0], dict) else None


def _str_key(value):
    return str(value) if value is not None else None


def _normalize_track(item):
    part = _first_part(item)
    media = item.get('Media') or []
    grandparent = item.get('grandparentTitle')
    track_artist = item.get('originalTitle') or grandparent or 'Unknown Artist'
    return {
        'Id': _str_key(item.get('ratingKey')),
        'Name': item.get('title'),
        'AlbumArtist': track_artist,
        'OriginalAlbumArtist': grandparent,
        'ArtistId': _str_key(item.get('grandparentRatingKey')),
        'Album': item.get('parentTitle'),
        'Year': item.get('parentYear') or item.get('year'),
        'IndexNumber': item.get('index'),
        'ParentIndexNumber': item.get('parentIndex'),
        'Path': part.get('file') if part else None,
        'FilePath': part.get('file') if part else None,
        'Container': media[0].get('container') if media and isinstance(media[0], dict) else None,
        'PartKey': part.get('key') if part else None,
    }


def _normalize_album(item):
    return {
        'Id': _str_key(item.get('ratingKey')),
        'Name': item.get('title'),
        'AlbumArtist': item.get('parentTitle') or 'Unknown Artist',
        'Year': item.get('year'),
        'DateCreated': item.get('addedAt') or 0,
    }


def _music_sections(user_creds=None):
    url = f"{_base_url(user_creds)}/library/sections"
    try:
        r = requests.get(url, headers=_headers(user_creds), timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        directories = _container(r.json()).get('Directory') or []
        return [
            {'id': d.get('key'), 'name': d.get('title')}
            for d in directories
            if isinstance(d, dict)
            and d.get('type') == 'artist'
            and d.get('key')
            and d.get('title')
        ]
    except Exception:
        logger.exception(f"Plex _music_sections failed at '{url}'")
        return []


def list_libraries(user_creds=None):
    return _music_sections(user_creds)


def _target_sections(user_creds=None):
    sections = _music_sections(user_creds)
    names_str = config.MUSIC_LIBRARIES
    if not names_str or not names_str.strip():
        return sections

    wanted = {name.strip().lower() for name in names_str.split(',') if name.strip()}
    matched = [s for s in sections if (s.get('name') or '').lower() in wanted]
    if not matched:
        logger.warning(
            f"Plex: no music sections matched MUSIC_LIBRARIES={names_str!r}; nothing will be scanned."
        )
    return matched


def _paged_metadata(path, params, user_creds=None, page_size=PLEX_PAGE_SIZE, max_items=None):
    url = f"{_base_url(user_creds)}{path}"
    collected = []
    start = 0
    while True:
        headers = _headers(user_creds)
        headers['X-Plex-Container-Start'] = str(start)
        headers['X-Plex-Container-Size'] = str(page_size)
        r = requests.get(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        container = _container(r.json())
        items = container.get('Metadata') or []
        collected.extend(items)

        if max_items is not None and len(collected) >= max_items:
            return collected[:max_items]
        if len(items) < page_size:
            break

        total = container.get('totalSize')
        start += len(items)
        if total is not None and start >= int(total):
            break
    return collected


def get_recent_albums(limit):
    sections = _target_sections()
    if not sections:
        return []

    fetch_all = limit == 0
    page_size = PLEX_PAGE_SIZE if fetch_all else max(1, min(limit, PLEX_PAGE_SIZE))
    max_items = None if fetch_all else limit

    all_albums = []
    for section in sections:
        try:
            items = _paged_metadata(
                f"/library/sections/{section['id']}/all",
                {'type': PLEX_ALBUM_TYPE, 'sort': 'addedAt:desc'},
                page_size=page_size,
                max_items=max_items,
            )
        except Exception:
            logger.exception(f"Plex get_recent_albums failed for section {section['id']}")
            continue
        all_albums.extend(_normalize_album(it) for it in items)

    if len(sections) > 1:
        all_albums.sort(key=lambda a: a.get('DateCreated') or 0, reverse=True)

    if not fetch_all:
        return all_albums[:limit]
    return all_albums


def get_tracks_from_album(album_id, user_creds=None):
    url = f"{_base_url(user_creds)}/library/metadata/{album_id}/children"
    try:
        r = requests.get(url, headers=_headers(user_creds), timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        items = _container(r.json()).get('Metadata') or []
        return [_normalize_track(it) for it in items]
    except Exception:
        logger.exception(f"Plex get_tracks_from_album failed for album {album_id}")
        return []


def _resolve_part(track_id, user_creds=None):
    url = f"{_base_url(user_creds)}/library/metadata/{track_id}"
    r = requests.get(url, headers=_headers(user_creds), timeout=REQUESTS_TIMEOUT)
    r.raise_for_status()
    items = _container(r.json()).get('Metadata') or []
    if not items:
        return None, None
    item = items[0]
    part = _first_part(item)
    media = item.get('Media') or []
    container = media[0].get('container') if media and isinstance(media[0], dict) else None
    return (part.get('key') if part else None), container


def download_track(temp_dir, item):
    try:
        track_id = item.get('Id') or item.get('id')
        part_key = item.get('PartKey')
        container = item.get('Container')

        if not part_key:
            part_key, resolved_container = _resolve_part(track_id)
            container = container or resolved_container

        if not part_key:
            logger.error(f"Plex download_track: no media part found for track {track_id}")
            return None

        file_extension = detect_download_extension(
            {'Container': container, 'Path': item.get('Path')}
        )
        download_url = f"{_base_url()}{part_key}"
        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")
        with requests.get(
            download_url,
            headers=_headers(),
            params={'download': 1},
            stream=True,
            timeout=REQUESTS_TIMEOUT,
        ) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Downloaded '{item.get('Name', 'Unknown')}' to '{local_filename}'")
        return local_filename
    except Exception:
        logger.exception(f"Failed to download track {item.get('Name', 'Unknown')}")
        return None


def get_all_songs(user_creds=None):
    sections = _target_sections(user_creds)
    all_items = []
    for section in sections:
        try:
            items = _paged_metadata(
                f"/library/sections/{section['id']}/all",
                {'type': PLEX_TRACK_TYPE},
                user_creds=user_creds,
            )
        except Exception:
            logger.exception(f"Plex get_all_songs failed for section {section['id']}")
            raise
        all_items.extend(_normalize_track(it) for it in items)
    return all_items


def search_albums(query, user_creds=None):
    results = []
    for section in _target_sections(user_creds):
        url = f"{_base_url(user_creds)}/library/sections/{section['id']}/all"
        params = {'type': PLEX_ALBUM_TYPE, 'title': query}
        try:
            r = requests.get(
                url, headers=_headers(user_creds), params=params, timeout=REQUESTS_TIMEOUT
            )
            r.raise_for_status()
            items = _container(r.json()).get('Metadata') or []
        except Exception:
            logger.exception(f"Plex search_albums failed for section {section['id']}")
            continue
        for it in items:
            results.append(
                {
                    'id': _str_key(it.get('ratingKey')),
                    'name': it.get('title'),
                    'artist': it.get('parentTitle'),
                    'year': it.get('year'),
                    'track_count': it.get('leafCount'),
                }
            )
            if len(results) >= 10:
                return results
    return results


def test_connection(user_creds=None):
    try:
        base = _base_url(user_creds)
        r = requests.get(
            f"{base}/library/sections", headers=_headers(user_creds), timeout=REQUESTS_TIMEOUT
        )
        r.raise_for_status()
        directories = _container(r.json()).get('Directory') or []
        music = [
            d for d in directories
            if isinstance(d, dict) and d.get('type') == 'artist' and d.get('key')
        ]
        if not music:
            return {
                'ok': False,
                'error': 'No Plex music library (artist section) found on this server.',
                'auth_failed': False,
                'sample_count': 0,
                'path_format': 'none',
                'warnings': [],
            }

        headers = _headers(user_creds)
        headers['X-Plex-Container-Start'] = '0'
        headers['X-Plex-Container-Size'] = '100'
        r2 = requests.get(
            f"{base}/library/sections/{music[0]['key']}/all",
            headers=headers,
            params={'type': PLEX_TRACK_TYPE},
            timeout=REQUESTS_TIMEOUT,
        )
        r2.raise_for_status()
        items = _container(r2.json()).get('Metadata') or []
        sample = []
        for it in items:
            track = _normalize_track(it)
            sample.append(
                {
                    'Id': track['Id'],
                    'Path': track['Path'],
                    'Name': track['Name'],
                    'AlbumArtist': track['AlbumArtist'],
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
        logger.warning(f"Plex test_connection failed: {e}")
        return {
            'ok': False,
            'error': str(e),
            'auth_failed': is_auth_error(e),
            'sample_count': 0,
            'path_format': 'none',
            'warnings': [],
        }


def _list_playlists(user_creds=None):
    url = f"{_base_url(user_creds)}/playlists"
    try:
        r = requests.get(
            url,
            headers=_headers(user_creds),
            params={'playlistType': 'audio'},
            timeout=REQUESTS_TIMEOUT,
        )
        r.raise_for_status()
        items = _container(r.json()).get('Metadata') or []
        return [
            {'Id': _str_key(p.get('ratingKey')), 'Name': p.get('title')}
            for p in items
            if p.get('ratingKey') is not None
        ]
    except Exception:
        logger.exception("Plex list playlists failed")
        return []


def get_all_playlists():
    return _list_playlists()


def _find_playlist(playlist_name, user_creds=None):
    for playlist in _list_playlists(user_creds):
        if playlist.get('Name') == playlist_name:
            return playlist
    return None


def get_playlist_by_name(playlist_name):
    return _find_playlist(playlist_name)


def _delete_playlist(playlist_id, user_creds=None):
    url = f"{_base_url(user_creds)}/playlists/{playlist_id}"
    try:
        r = requests.delete(url, headers=_headers(user_creds), timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception:
        logger.exception(f"Exception deleting Plex playlist ID {playlist_id}")
        return False


def delete_playlist(playlist_id):
    return _delete_playlist(playlist_id)


def get_playlist_track_ids(playlist_id, user_creds=None):
    url = f"{_base_url(user_creds)}/playlists/{playlist_id}/items"
    try:
        r = requests.get(url, headers=_headers(user_creds), timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        items = _container(r.json()).get('Metadata') or []
        return [_str_key(it.get('ratingKey')) for it in items if it.get('ratingKey') is not None]
    except Exception:
        logger.exception(f"Plex get_playlist_track_ids failed for {playlist_id}")
        return []


def _machine_identifier(user_creds=None):
    base = _base_url(user_creds)
    cached = _MACHINE_ID_CACHE.get(base)
    if cached:
        return cached
    r = requests.get(f"{base}/identity", headers=_headers(user_creds), timeout=REQUESTS_TIMEOUT)
    r.raise_for_status()
    machine_id = _container(r.json()).get('machineIdentifier')
    if machine_id:
        _MACHINE_ID_CACHE[base] = machine_id
    return machine_id


def _metadata_uri(machine_id, item_ids):
    joined = ",".join(str(i) for i in item_ids)
    return f"server://{machine_id}/com.plexapp.plugins.library/library/metadata/{joined}"


def _add_items(playlist_id, item_ids, machine_id, user_creds=None):
    base = _base_url(user_creds)
    for i in range(0, len(item_ids), PLEX_PLAYLIST_BATCH_SIZE):
        batch = item_ids[i : i + PLEX_PLAYLIST_BATCH_SIZE]
        params = {'uri': _metadata_uri(machine_id, batch)}
        try:
            r = requests.post(
                f"{base}/playlists/{playlist_id}/items",
                headers=_headers(user_creds),
                params=params,
                timeout=REQUESTS_TIMEOUT,
            )
            r.raise_for_status()
        except Exception:
            logger.exception(
                f"Plex _add_items: batch starting at {i} failed for playlist {playlist_id}"
            )
            return False
    return True


def _create_playlist_batched(title, item_ids, user_creds=None):
    if not item_ids:
        return None

    base = _base_url(user_creds)
    machine_id = _machine_identifier(user_creds)
    first_batch = item_ids[:PLEX_PLAYLIST_BATCH_SIZE]
    rest = item_ids[PLEX_PLAYLIST_BATCH_SIZE:]

    params = {
        'type': 'audio',
        'title': title,
        'smart': 0,
        'uri': _metadata_uri(machine_id, first_batch),
    }
    r = requests.post(
        f"{base}/playlists", headers=_headers(user_creds), params=params, timeout=REQUESTS_TIMEOUT
    )
    r.raise_for_status()
    created = (_container(r.json()).get('Metadata') or [{}])[0]
    new_id = created.get('ratingKey')
    if not new_id:
        logger.error(f"Plex _create_playlist_batched: created '{title}' but response had no ratingKey")
        return None

    if rest and not _add_items(new_id, rest, machine_id, user_creds):
        logger.error(f"Plex _create_playlist_batched: created '{title}' but failed to add overflow tracks")

    return {'Id': _str_key(new_id), 'Name': created.get('title', title)}


def create_playlist(base_name, item_ids):
    try:
        _create_playlist_batched(base_name, list(item_ids))
    except Exception:
        logger.exception("Exception creating Plex playlist '%s'", base_name)


def create_instant_playlist(playlist_name, item_ids, user_creds=None):
    try:
        return _create_playlist_batched(
            f"{playlist_name.strip()}_instant", list(item_ids), user_creds
        )
    except Exception:
        logger.exception("Exception creating Plex instant playlist '%s'", playlist_name)
        return None


def create_or_replace_playlist(playlist_name, item_ids, user_creds=None):
    if not item_ids:
        return None

    ids = list(item_ids)
    existing = _find_playlist(playlist_name, user_creds)
    if existing and existing.get('Id') and not _delete_playlist(existing['Id'], user_creds):
        logger.error(
            f"Plex create_or_replace_playlist: failed to delete existing '{playlist_name}'"
        )
        return None

    try:
        result = _create_playlist_batched(playlist_name, ids, user_creds)
    except Exception:
        logger.exception(f"Plex create_or_replace_playlist: create failed for '{playlist_name}'")
        return None

    if result:
        logger.info(f"Plex: wrote playlist '{playlist_name}' with {len(ids)} tracks")
    return result


def get_top_played_songs(limit, user_creds=None):
    sections = _target_sections(user_creds)
    page_size = str(limit) if limit and limit > 0 else str(PLEX_PAGE_SIZE)

    scored = []
    for section in sections:
        url = f"{_base_url(user_creds)}/library/sections/{section['id']}/all"
        headers = _headers(user_creds)
        headers['X-Plex-Container-Start'] = '0'
        headers['X-Plex-Container-Size'] = page_size
        try:
            r = requests.get(
                url,
                headers=headers,
                params={'type': PLEX_TRACK_TYPE, 'sort': 'viewCount:desc'},
                timeout=REQUESTS_TIMEOUT,
            )
            r.raise_for_status()
            items = _container(r.json()).get('Metadata') or []
        except Exception:
            logger.exception(f"Plex get_top_played_songs failed for section {section['id']}")
            continue
        for it in items:
            scored.append((it.get('viewCount') or 0, _normalize_track(it)))

    scored.sort(key=lambda entry: entry[0], reverse=True)
    tracks = [track for _, track in scored]
    if limit and limit > 0:
        return tracks[:limit]
    return tracks


def get_last_played_time(item_id, user_creds=None):
    url = f"{_base_url(user_creds)}/library/metadata/{item_id}"
    try:
        r = requests.get(url, headers=_headers(user_creds), timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        items = _container(r.json()).get('Metadata') or []
        if not items:
            return None
        return _epoch_to_iso(items[0].get('lastViewedAt'))
    except Exception:
        logger.exception(f"Plex get_last_played_time failed for item {item_id}")
        return None


def _epoch_to_iso(epoch):
    if not epoch:
        return None
    try:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime(
            '%Y-%m-%dT%H:%M:%S.000Z'
        )
    except (TypeError, ValueError, OSError):
        return None


def _find_lyric_stream_key(item):
    for media in item.get('Media') or []:
        if not isinstance(media, dict):
            continue
        for part in media.get('Part') or []:
            if not isinstance(part, dict):
                continue
            for stream in part.get('Stream') or []:
                if isinstance(stream, dict) and stream.get('streamType') == _LYRIC_STREAM_TYPE:
                    key = stream.get('key')
                    if key:
                        return key
    return None


def get_lyrics(track_id, timeout=2.5):
    try:
        base = _base_url()
        r = requests.get(
            f"{base}/library/metadata/{track_id}", headers=_headers(), timeout=timeout
        )
        r.raise_for_status()
        items = _container(r.json()).get('Metadata') or []
        if not items:
            return None
        stream_key = _find_lyric_stream_key(items[0])
        if not stream_key:
            return None
        lr = requests.get(f"{base}{stream_key}", headers=_headers(), timeout=timeout)
        lr.raise_for_status()
        text = (lr.text or '').strip()
        return text or None
    except Exception as exc:
        logger.debug('Plex get_lyrics failed for %s: %s', track_id, exc)
        return None
