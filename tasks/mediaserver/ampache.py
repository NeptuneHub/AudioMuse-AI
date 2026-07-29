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
* Trades credentials for a session token that accepts either an API key or a
  time-salted password hash in the same field, caching it per server and
  re-handshaking once when a session lapses mid-run.
* Fetches catalogues, recent albums, album tracks, search results and the whole
  song list with pagination, honouring MUSIC_LIBRARIES by resolving it to
  Ampache catalog ids.
* Downloads the original file rather than a transcoded stream, reads play stats
  and lyrics, and manages playlists through the shared dispatcher contract.
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
from .helper import detect_path_format

logger = logging.getLogger(__name__)

# Ampache expires idle sessions server-side; re-handshake a little before that so
# a long analysis run never fails midway on a token that lapsed between calls.
_TOKEN_TTL_SECONDS = 3000
_token_cache = {}
_token_lock = threading.Lock()

_SECRET_QUERY_PARAM = re.compile(r'(?i)([?&](?:auth|passphrase|password)=)[^&\s]*')


def _redact_ampache_secrets(text):
    return _SECRET_QUERY_PARAM.sub(r'\1[REDACTED]', str(text))


def _creds(user_creds=None):
    user_creds = context.active_creds(user_creds) or {}
    url = (user_creds.get('url') or config.AMPACHE_URL or '').rstrip('/')
    user = user_creds.get('user') or config.AMPACHE_USER
    password = user_creds.get('password') or config.AMPACHE_PASSWORD
    return url, user, password


def _cache_key(url, user):
    return f"{url}|{user}"


def _handshake(url, user, password):
    """Trade credentials for a session token.

    Ampache accepts either an API key or a time-salted password hash as ``auth``.
    An API key is passed through untouched; anything else is treated as a
    password and hashed as ``sha256(timestamp + sha256(password))``, which is
    what lets the same field hold either.
    """
    timestamp = int(time.time())
    pass_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    passphrase = hashlib.sha256(f"{timestamp}{pass_hash}".encode('utf-8')).hexdigest()

    for params in (
        {'action': 'handshake', 'user': user, 'timestamp': timestamp, 'auth': passphrase},
        # An API key needs neither user nor timestamp; try it second so a real
        # password is never sent as a key.
        {'action': 'handshake', 'auth': password},
    ):
        params['version'] = '8.0.0'
        try:
            response = requests.get(f"{url}/server/json.server.php", params=params, timeout=30)
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


def _token(user_creds=None, force=False):
    url, user, password = _creds(user_creds)
    if not url or not password:
        logger.warning("Ampache URL or password is not configured.")
        return None, None, {'kind': 'config', 'message': 'Ampache URL or password is not configured.'}

    key = _cache_key(url, user)
    with _token_lock:
        cached = _token_cache.get(key)
        if cached and not force and cached['expires'] > time.time():
            return url, cached['token'], None

    body, err = _handshake(url, user, password)
    if not body:
        return url, None, err

    with _token_lock:
        _token_cache[key] = {'token': body['auth'], 'expires': time.time() + _TOKEN_TTL_SECONDS}
    return url, body['auth'], None


def _request_ex(action, params=None, stream=False, user_creds=None, timeout=None):
    """Call one Ampache action, re-handshaking once if the session has lapsed."""
    for attempt in (0, 1):
        url, token, err = _token(user_creds, force=bool(attempt))
        if not token:
            return None, err

        all_params = {'action': action, 'auth': token, 'version': '8.0.0', **(params or {})}
        try:
            response = requests.get(
                f"{url}/server/json.server.php",
                params=all_params,
                stream=stream,
                timeout=timeout or 60,
            )
        except Exception as e:
            logger.error(  # noqa: TRY400 - .exception would leak the unredacted URL creds via the traceback
                f"Ampache request '{action}' failed: {_redact_ampache_secrets(e)}"
            )
            return None, {'kind': 'network', 'message': str(_redact_ampache_secrets(e))}

        if stream:
            try:
                response.raise_for_status()
            except Exception as e:
                logger.error(  # noqa: TRY400 - .exception would leak the unredacted URL creds via the traceback
                    f"Ampache stream '{action}' failed: {_redact_ampache_secrets(e)}"
                )
                return None, {'kind': 'network', 'message': str(_redact_ampache_secrets(e))}
            return response, None

        try:
            body = response.json()
        except Exception as e:
            return None, {'kind': 'parse', 'message': str(_redact_ampache_secrets(e))}

        error = body.get('error') if isinstance(body, dict) else None
        if error:
            code = str(error.get('errorCode') or error.get('code') or '')
            # 4701 is Ampache's "session expired"; anything else is not worth a retry.
            if code == '4701' and attempt == 0:
                continue
            kind = 'auth' if code in ('4701', '4742', '4704') else 'api'
            return None, {'kind': kind, 'message': error.get('errorMessage') or error.get('message') or 'Ampache error'}

        return body, None

    return None, {'kind': 'auth', 'message': 'Ampache session could not be renewed'}


def _request(action, params=None, stream=False, user_creds=None, timeout=None):
    body, _err = _request_ex(action, params, stream=stream, user_creds=user_creds, timeout=timeout)
    return body


def _target_catalog_ids(user_creds=None):
    """Catalog ids the configured library filter selects, or None for everything."""
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
    return [{'Id': str(c.get('id')), 'Name': c.get('name') or f"Catalog {c.get('id')}"} for c in catalogs]


def _map_song(song):
    """Normalise one Ampache song row into the shape every backend returns."""
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
        # `suffix` is what download_track uses to name the temp file; Ampache
        # calls the same thing `format`.
        'suffix': song.get('format') or song.get('stream_format'),
        'title': song.get('title'),
    }


def get_all_songs(user_creds=None, apply_filter=True):
    catalog_ids = _target_catalog_ids(user_creds=user_creds) if apply_filter else None
    if isinstance(catalog_ids, set) and not catalog_ids:
        return []

    songs = []
    offset = 0
    page = 500
    while True:
        params = {'offset': offset, 'limit': page}
        body = _request('songs', params, user_creds=user_creds)
        rows = (body or {}).get('song') or []
        if not rows:
            break

        for row in rows:
            if catalog_ids is not None and str(row.get('catalog')) not in catalog_ids:
                continue
            songs.append(_map_song(row))

        offset += len(rows)
        if len(rows) < page:
            break

    logger.info(f"Fetched {len(songs)} songs from Ampache.")
    return songs


def get_recent_albums(limit):
    fetch_all = not limit or int(limit) <= 0
    params = {'limit': 0 if fetch_all else int(limit), 'sort': 'addition_time,DESC'}
    body = _request('albums', params)
    albums = (body or {}).get('album') or []

    catalog_ids = _target_catalog_ids()
    if isinstance(catalog_ids, set):
        if not catalog_ids:
            return []
        albums = [a for a in albums if str(a.get('catalog')) in catalog_ids]

    mapped = [
        {
            **a,
            'Id': str(a.get('id')),
            'Name': a.get('name'),
            'AlbumArtist': ((a.get('artist') or {}) if isinstance(a.get('artist'), dict) else {}).get('name'),
        }
        for a in albums
    ]
    return mapped if fetch_all else mapped[: int(limit)]


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
    """Stream one track to disk, returning the local path."""
    try:
        track_id = item.get('id') or item.get('Id')

        suffix = item.get('suffix') or item.get('format')
        if suffix and isinstance(suffix, str) and suffix.strip():
            file_extension = '.' + suffix.strip().replace('/', '').replace('\\', '')
        elif item.get('Path'):
            file_extension = os.path.splitext(item['Path'])[1] or '.tmp'
        else:
            file_extension = '.tmp'

        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")

        # `download` hands back the original file; `stream` would transcode it and
        # analysis must see the real audio.
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
    return {
        'ok': True,
        'error': None,
        'auth_failed': False,
        'sample_count': len(songs),
        'path_format': detect_path_format(songs),
        'warnings': warnings,
    }


def get_all_playlists():
    body = _request('playlists', {'limit': 0})
    playlists = (body or {}).get('playlist') or []
    return [{**p, 'Id': str(p.get('id')), 'Name': p.get('name')} for p in playlists]


def get_playlist_by_name(playlist_name, user_creds=None):
    for playlist in get_all_playlists():
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

    for item_id in item_ids:
        _request('playlist_add_song', {'filter': playlist_id, 'song': item_id, 'check': 1})

    return str(playlist_id)


def create_instant_playlist(playlist_name, item_ids, user_creds=None):
    return create_playlist(f"{playlist_name.strip()}_instant", item_ids)


def create_or_replace_playlist(playlist_name, item_ids, user_creds=None):
    user_creds = context.active_creds(user_creds)
    existing = get_playlist_by_name(playlist_name, user_creds=user_creds)
    if existing:
        delete_playlist(existing['Id'])
    return create_playlist(playlist_name, item_ids)


def get_top_played_songs(limit, user_creds):
    body = _request('stats', {'type': 'song', 'filter': 'frequent', 'limit': limit}, user_creds=user_creds)
    return [_map_song(s) for s in ((body or {}).get('song') or [])]


def get_last_played_time(item_id, user_creds):
    """Ampache exposes no per-track last-played timestamp, so callers get None."""
    return None


def get_lyrics(track_id: str, timeout: float = 2.5):
    body = _request('song', {'filter': track_id}, timeout=timeout)
    songs = (body or {}).get('song') or []
    if isinstance(songs, list) and songs:
        return songs[0].get('lyrics') or None
    return None
