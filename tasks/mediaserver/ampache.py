# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Ampache media-server backend, speaking Ampache's own JSON API.

Ampache also serves a Subsonic API, so it can be driven through the ``navidrome``
backend instead. This backend exists because the native API answers in one call
what Subsonic needs several for, and it returns fields Subsonic has no room for
(replay gain, r128, multiple artists, stream format).

Two things differ from the Subsonic path and both matter to callers:

* Track ids here are Ampache's own row ids (``1``), not the prefixed Subsonic
  form (``so-1``). A library analysed through one backend is therefore keyed
  differently from the same library analysed through the other.
* Auth is a handshake that returns a session token, rather than credentials sent
  on every request. The token is cached per server and re-issued on expiry.

Main Features:
* Trades credentials for a session token that holds either an API key or a
  time-salted password hash in the same field. The token is cached per
  CREDENTIAL SET (a rotated password never reuses the old session) and
  re-handshaked once when a session lapses mid-run; the API-key attempt is only
  made for a key-shaped secret, so a real password never travels in a query
  string. The caller's timeout bounds the handshake too, not just the data call.
* Fetches catalogues, recent albums, album tracks, search results and the whole
  song list with pagination, honouring MUSIC_LIBRARIES by resolving it to
  Ampache catalog ids, pushing that filter into the server query and still
  enforcing it locally. Recent albums page on until the filter yields the
  requested count instead of filtering one server-limited page.
* Downloads the original file rather than a transcoded stream, refusing a
  response that carries an Ampache JSON error under HTTP 200 instead of audio.
* Reads play stats and lyrics, and manages playlists through the shared
  dispatcher contract: creation returns the ``{'Id', 'Name'}`` dict callers
  dereference, a playlist that received none of its tracks is not reported as
  created, and a failed delete aborts the replace rather than leaving two
  playlists under one name.
"""

from . import http as requests
import hashlib
import logging
import os
import re
import threading
import time

import config
from . import context
from .helper import detect_download_extension, detect_path_format

logger = logging.getLogger(__name__)

_TOKEN_TTL_SECONDS = 3000
_HANDSHAKE_TIMEOUT_SECONDS = 30
_REQUEST_TIMEOUT_SECONDS = 60
_PAGE_SIZE = 500
_SESSION_EXPIRED_CODE = '4701'
_AUTH_ERROR_CODES = ('4742', '4704')

_token_cache = {}
_token_lock = threading.Lock()

_SECRET_QUERY_PARAM = re.compile(r'(?i)([?&](?:auth|passphrase|password)=)[^&\s]*')
_API_KEY_SHAPE = re.compile(r'\A[0-9a-fA-F]{32,}\Z')

_CATALOGUE_KEYS = (
    'Id', 'Name', 'AlbumArtist', 'ArtistId', 'OriginalAlbumArtist', 'Album',
    'Path', 'FilePath', 'Year', 'Rating', 'DurationSeconds',
)


def _redact_ampache_secrets(text):
    return _SECRET_QUERY_PARAM.sub(r'\1[REDACTED]', str(text))


def _creds(user_creds=None):
    user_creds = context.active_creds(user_creds) or {}
    url = (user_creds.get('url') or config.AMPACHE_URL or '').rstrip('/')
    user = user_creds.get('user') or config.AMPACHE_USER
    password = user_creds.get('password') or config.AMPACHE_PASSWORD
    return url, user, password


def _cache_key(url, user, password):
    secret = hashlib.sha256((password or '').encode('utf-8')).hexdigest()
    return f"{url}|{user}|{secret}"


def _handshake_attempts(user, password, timestamp, passphrase):
    attempts = []
    if user:
        attempts.append(
            {'action': 'handshake', 'user': user, 'timestamp': timestamp, 'auth': passphrase}
        )
    if not user or _API_KEY_SHAPE.match(password or ''):
        attempts.append({'action': 'handshake', 'auth': password})
    return attempts


def _handshake(url, user, password, timeout=None):
    timestamp = int(time.time())
    pass_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    passphrase = hashlib.sha256(f"{timestamp}{pass_hash}".encode('utf-8')).hexdigest()

    body = None
    for params in _handshake_attempts(user, password, timestamp, passphrase):
        params['version'] = '8.0.0'
        try:
            response = requests.get(
                f"{url}/server/json.server.php",
                params=params,
                timeout=timeout or _HANDSHAKE_TIMEOUT_SECONDS,
            )
            body = response.json()
        except Exception as e:
            logger.error(  # noqa: TRY400 - .exception would leak the unredacted URL creds via the traceback
                f"Ampache handshake failed: {_redact_ampache_secrets(e)}"
            )
            return None, {'kind': 'network', 'message': str(_redact_ampache_secrets(e))}

        if isinstance(body, dict) and body.get('auth'):
            return body, None

    error = (body or {}).get('error') if isinstance(body, dict) else None
    message = (error or {}).get('message') or 'Ampache handshake was rejected'
    return None, {'kind': 'auth', 'message': message}


def _token(user_creds=None, force=False, timeout=None):
    url, user, password = _creds(user_creds)
    if not url or not password:
        logger.warning("Ampache URL or password is not configured.")
        return None, None, {'kind': 'config', 'message': 'Ampache URL or password is not configured.'}

    key = _cache_key(url, user, password)
    with _token_lock:
        cached = _token_cache.get(key)
        if cached and not force and cached['expires'] > time.time():
            return url, cached['token'], None

    body, err = _handshake(url, user, password, timeout=timeout)
    if not body:
        return url, None, err

    with _token_lock:
        _token_cache[key] = {'token': body['auth'], 'expires': time.time() + _TOKEN_TTL_SECONDS}
    return url, body['auth'], None


def _error_from_body(body):
    if not isinstance(body, dict):
        return None, ''
    error = body.get('error')
    if not error:
        return None, ''
    return error, str(error.get('errorCode') or error.get('code') or '')


def _stream_error_body(response):
    try:
        content_type = str((response.headers or {}).get('Content-Type') or '')
    except Exception:
        return None
    if 'json' not in content_type.lower():
        return None
    try:
        return response.json()
    except Exception:
        return None


def _close_quietly(response):
    try:
        response.close()
    except Exception:
        logger.debug("Ampache: closing an errored stream response failed.", exc_info=True)


def _fetch(url, action, token, params, stream, timeout):
    all_params = {'action': action, 'auth': token, 'version': '8.0.0', **(params or {})}
    try:
        response = requests.get(
            f"{url}/server/json.server.php",
            params=all_params,
            stream=stream,
            timeout=timeout or _REQUEST_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.error(  # noqa: TRY400 - .exception would leak the unredacted URL creds via the traceback
            f"Ampache request '{action}' failed: {_redact_ampache_secrets(e)}"
        )
        return None, {'kind': 'network', 'message': str(_redact_ampache_secrets(e))}
    return response, None


def _stream_payload(action, response):
    try:
        response.raise_for_status()
    except Exception as e:
        logger.error(  # noqa: TRY400 - .exception would leak the unredacted URL creds via the traceback
            f"Ampache stream '{action}' failed: {_redact_ampache_secrets(e)}"
        )
        return None, None, {'kind': 'network', 'message': str(_redact_ampache_secrets(e))}

    body = _stream_error_body(response)
    if body is None:
        return response, None, None
    _close_quietly(response)
    return None, body, None


def _body_error(action, body, stream):
    error, code = _error_from_body(body)
    if error:
        if code == _SESSION_EXPIRED_CODE:
            return 'retry', None
        kind = 'auth' if code in _AUTH_ERROR_CODES else 'api'
        message = error.get('errorMessage') or error.get('message') or 'Ampache error'
        return 'error', {'kind': kind, 'message': message}
    if stream:
        return 'error', {
            'kind': 'api',
            'message': f"Ampache returned JSON instead of audio for '{action}'",
        }
    return 'ok', None


def _request_ex(action, params=None, stream=False, user_creds=None, timeout=None):
    for attempt in (0, 1):
        url, token, err = _token(user_creds, force=bool(attempt), timeout=timeout)
        if not token:
            return None, err

        response, err = _fetch(url, action, token, params, stream, timeout)
        if err:
            return None, err

        if stream:
            passthrough, body, err = _stream_payload(action, response)
            if err:
                return None, err
            if passthrough is not None:
                return passthrough, None
        else:
            try:
                body = response.json()
            except Exception as e:
                return None, {'kind': 'parse', 'message': str(_redact_ampache_secrets(e))}

        verdict, err = _body_error(action, body, stream)
        if verdict == 'retry':
            if attempt == 0:
                continue
            break
        if verdict == 'error':
            return None, err
        return body, None

    return None, {'kind': 'auth', 'message': 'Ampache session could not be renewed'}


def _request(action, params=None, stream=False, user_creds=None, timeout=None):
    body, _err = _request_ex(action, params, stream=stream, user_creds=user_creds, timeout=timeout)
    return body


def _target_catalog_ids(user_creds=None):
    libraries = (context.active_libraries(config.MUSIC_LIBRARIES) or '').strip()
    if not libraries:
        return None

    wanted = {name.strip().lower() for name in libraries.split(',') if name.strip()}
    if not wanted:
        return None

    body = _request('catalogs', {'filter': 'music'}, user_creds=user_creds)
    catalogs = (body or {}).get('catalog') or []
    ids = {
        str(c.get('id'))
        for c in catalogs
        if str(c.get('name', '')).lower() in wanted or str(c.get('id')) in wanted
    }
    if not ids:
        logger.warning("Ampache library filter matched no catalogs; returning no songs.")
    return ids


def list_libraries(user_creds=None):
    body = _request('catalogs', {'filter': 'music'}, user_creds=user_creds)
    catalogs = (body or {}).get('catalog') or []
    return [{'id': str(c.get('id')), 'name': c.get('name') or f"Catalog {c.get('id')}"} for c in catalogs]


def _map_song(song):
    artist = (song.get('artist') or {}) if isinstance(song.get('artist'), dict) else {}
    albumartist = (song.get('albumartist') or {}) if isinstance(song.get('albumartist'), dict) else {}
    album = (song.get('album') or {}) if isinstance(song.get('album'), dict) else {}
    path = song.get('filename') or ''

    return {
        **song,
        'Id': str(song.get('id')),
        'Name': song.get('title') or song.get('name') or 'Unknown',
        'AlbumArtist': albumartist.get('name') or artist.get('name') or 'Unknown',
        'ArtistId': str(artist.get('id')) if artist.get('id') is not None else None,
        'OriginalAlbumArtist': albumartist.get('name'),
        'Album': album.get('name'),
        'Path': path,
        'FilePath': path,
        'Year': song.get('year'),
        'Rating': song.get('rating') or None,
        'DurationSeconds': song.get('time'),
        'suffix': song.get('format') or song.get('stream_format'),
        'title': song.get('title'),
    }


def _map_catalogue_song(song):
    mapped = _map_song(song)
    return {key: mapped[key] for key in _CATALOGUE_KEYS}


def _catalog_filter_params(catalog_ids):
    params = {'type': 'song', 'operator': 'or'}
    for index, catalog_id in enumerate(sorted(catalog_ids), start=1):
        params[f'rule_{index}'] = 'catalog'
        params[f'rule_{index}_operator'] = 0
        params[f'rule_{index}_input'] = catalog_id
    return params


def _catalogue_query(catalog_ids):
    if catalog_ids:
        return 'advanced_search', _catalog_filter_params(catalog_ids)
    return 'songs', {}


def _in_catalogs(row, catalog_ids):
    return catalog_ids is None or str(row.get('catalog')) in catalog_ids


def _should_retry_unfiltered(action, err, offset, collected):
    message = (err or {}).get('message') or 'unknown error'
    if action == 'advanced_search' and not collected:
        logger.warning(
            "Ampache advanced_search catalog filter failed (%s); falling back to a "
            "full song fetch filtered locally.",
            message,
        )
        return True
    logger.error(
        "AMPACHE CATALOGUE FETCH FAILED at offset %d after %d songs (%s). The "
        "returned catalogue is INCOMPLETE - do not treat missing tracks as deleted.",
        offset, collected, message,
    )
    return False


def get_all_songs(user_creds=None, apply_filter=True):
    catalog_ids = _target_catalog_ids(user_creds=user_creds) if apply_filter else None
    if isinstance(catalog_ids, set) and not catalog_ids:
        return []

    action, base_params = _catalogue_query(catalog_ids)

    songs = []
    offset = 0
    while True:
        params = {**base_params, 'offset': offset, 'limit': _PAGE_SIZE}
        body, err = _request_ex(action, params, user_creds=user_creds)
        if body is None:
            if not _should_retry_unfiltered(action, err, offset, len(songs)):
                break
            action, base_params = 'songs', {}
            continue

        rows = body.get('song') or []
        if not rows:
            break

        songs.extend(_map_catalogue_song(r) for r in rows if _in_catalogs(r, catalog_ids))
        offset += len(rows)
        if len(rows) < _PAGE_SIZE:
            break

    logger.info(f"Fetched {len(songs)} songs from Ampache.")
    return songs


def _map_album(album):
    artist = (album.get('artist') or {}) if isinstance(album.get('artist'), dict) else {}
    return {
        **album,
        'Id': str(album.get('id')),
        'Name': album.get('name'),
        'AlbumArtist': artist.get('name'),
    }


def get_recent_albums(limit):
    fetch_all = not limit or int(limit) <= 0
    wanted = 0 if fetch_all else int(limit)

    catalog_ids = _target_catalog_ids()
    if isinstance(catalog_ids, set) and not catalog_ids:
        return []

    mapped = []
    offset = 0
    page = _PAGE_SIZE if catalog_ids else max(wanted, 1)
    while True:
        params = {'offset': offset, 'limit': 0 if fetch_all else page, 'sort': 'addition_time,DESC'}
        albums = (_request('albums', params) or {}).get('album') or []
        if not albums:
            break

        mapped.extend(_map_album(a) for a in albums if _in_catalogs(a, catalog_ids))
        offset += len(albums)
        if fetch_all or len(albums) < page or len(mapped) >= wanted:
            break

    return mapped if fetch_all else mapped[:wanted]


def get_tracks_from_album(album_id, user_creds=None):
    body = _request('album_songs', {'filter': album_id}, user_creds=user_creds)
    return [_map_song(s) for s in ((body or {}).get('song') or [])]


def search_albums(query, user_creds=None):
    body = _request(
        'advanced_search',
        {
            'type': 'album',
            'operator': 'and',
            'rule_1': 'title',
            'rule_1_operator': 0,
            'rule_1_input': query,
            'limit': 100,
        },
        user_creds=user_creds,
    )
    albums = (body or {}).get('album') or []
    return [{**a, 'Id': str(a.get('id')), 'Name': a.get('name')} for a in albums]


def download_track(temp_dir, item):
    try:
        track_id = item.get('id') or item.get('Id')
        file_extension = detect_download_extension(
            {**item, 'Container': item.get('suffix') or item.get('format')}
        )
        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")

        response = _request('download', {'id': track_id, 'type': 'song'}, stream=True)
        if response is None:
            return None

        with response:
            with open(local_filename, 'wb') as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    handle.write(chunk)

        logger.info(f"Downloaded '{item.get('Name') or item.get('title') or 'Unknown'}' to '{local_filename}'")
        return local_filename
    except Exception as e:
        logger.error(  # noqa: TRY400 - .exception would leak the unredacted URL creds via the traceback
            f"Failed to download Ampache track {item.get('Name', 'Unknown')}: {_redact_ampache_secrets(e)}"
        )
    return None


def test_connection(user_creds=None):
    warnings = []
    body, err = _request_ex('songs', {'limit': 100}, user_creds=user_creds)
    if body is None:
        return {
            'ok': False,
            'error': (err or {}).get('message') or 'Ampache test_connection failed',
            'auth_failed': bool(err and err.get('kind') == 'auth'),
            'sample_count': 0,
            'path_format': 'none',
            'warnings': warnings,
        }

    songs = [_map_song(s) for s in (body.get('song') or [])]
    path_format = detect_path_format(songs)
    if path_format != 'absolute':
        warnings.append(
            'Ampache is returning relative paths or no paths at all. This happens when '
            'the catalog was added with a relative path, or when the API user cannot '
            'read the filename field. Automatic path-based matching will not work well, '
            'so you will need to manually match most albums in Step 4.'
        )
    return {
        'ok': True,
        'error': None,
        'auth_failed': False,
        'sample_count': len(songs),
        'path_format': path_format,
        'warnings': warnings,
    }


def _playlists(user_creds=None):
    body = _request('playlists', {'limit': 0}, user_creds=user_creds)
    playlists = (body or {}).get('playlist') or []
    return [{**p, 'Id': str(p.get('id')), 'Name': p.get('name')} for p in playlists]


def get_all_playlists():
    return _playlists()


def get_playlist_by_name(playlist_name, user_creds=None):
    for playlist in _playlists(user_creds=user_creds):
        if playlist.get('Name') == playlist_name:
            return playlist
    return None


def get_playlist_track_ids(playlist_id, user_creds=None):
    body = _request('playlist_songs', {'filter': playlist_id, 'limit': 0}, user_creds=user_creds)
    return [str(s.get('id')) for s in ((body or {}).get('song') or [])]


def delete_playlist(playlist_id):
    return _request('playlist_delete', {'filter': playlist_id}) is not None


def create_playlist(base_name, item_ids):
    body = _request('playlist_create', {'name': base_name, 'type': 'private'})
    playlist = (body or {}).get('playlist') or {}
    playlist_id = playlist.get('id') or (body or {}).get('id')
    if not playlist_id:
        logger.error(f"Ampache refused to create playlist '{base_name}'.")
        return None

    wanted = list(item_ids or [])
    added = 0
    for item_id in wanted:
        if _request('playlist_add_song', {'filter': playlist_id, 'song': item_id, 'check': 1}) is not None:
            added += 1

    if wanted and not added:
        logger.error(
            "AMPACHE PLAYLIST '%s' (id=%s) RECEIVED NONE OF ITS %d TRACKS - every "
            "playlist_add_song call was rejected. The playlist exists on the server but "
            "is EMPTY; the Ampache error is logged above.",
            base_name, playlist_id, len(wanted),
        )
        return None
    if added < len(wanted):
        logger.error(
            "AMPACHE PLAYLIST '%s' (id=%s) IS INCOMPLETE: Ampache rejected %d of its %d tracks.",
            base_name, playlist_id, len(wanted) - added, len(wanted),
        )

    return {'Id': str(playlist_id), 'Name': base_name}


def create_instant_playlist(playlist_name, item_ids, user_creds=None):
    return create_playlist(f"{playlist_name.strip()}_instant", item_ids)


def create_or_replace_playlist(playlist_name, item_ids, user_creds=None):
    user_creds = context.active_creds(user_creds)
    existing = get_playlist_by_name(playlist_name, user_creds=user_creds)
    if existing and existing.get('Id') and not delete_playlist(existing['Id']):
        logger.error(
            f"Ampache create_or_replace_playlist: failed to delete existing "
            f"'{playlist_name}' (id={existing['Id']}); aborting to avoid creating a duplicate"
        )
        return None
    return create_playlist(playlist_name, item_ids)


def get_top_played_songs(limit, user_creds):
    body = _request('stats', {'type': 'song', 'filter': 'frequent', 'limit': limit}, user_creds=user_creds)
    return [_map_song(s) for s in ((body or {}).get('song') or [])]


def get_last_played_time(item_id, user_creds):
    return None


def get_lyrics(track_id: str, timeout: float = 2.5):
    body = _request('song', {'filter': track_id}, timeout=timeout)
    songs = (body or {}).get('song') or []
    if isinstance(songs, list) and songs:
        return songs[0].get('lyrics') or None
    return None
