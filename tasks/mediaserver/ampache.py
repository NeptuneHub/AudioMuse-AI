# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Ampache media-server backend, speaking Ampache's own JSON API.

Requires Ampache API 8 or newer: album browsing depends on the catalogue id being
present on album objects and on the ``cond`` browse filter. test_connection warns
when a server reports an older API rather than letting an analysis quietly find
nothing.

Ampache also serves a Subsonic API, so it can be driven through the ``navidrome``
backend instead. This backend exists because the native API answers in one call
what Subsonic needs several for, and it returns fields Subsonic has no room for
(replay gain, r128, multiple artists, stream format).

Two things differ from the Subsonic path and both matter to callers:

* Track ids here are Ampache's own row ids (``1``), not the prefixed Subsonic
  form (``so-1``). A library analysed through one backend is therefore keyed
  differently from the same library analysed through the other.
* Auth is a session token rather than credentials sent on every request. An API
  key skips the handshake entirely - Ampache takes it from an Authorization header
  and opens the session itself. A password must handshake; that token is cached
  per server, its window slides forward on every call the server accepts - the way
  Ampache's own session does - and it is re-issued only once the window has lapsed.

Main Features:
* Prefers an API key sent as ``Authorization: Bearer``, which Ampache resolves
  with findByApiKey and answers by creating the session server-side - so there is
  no handshake, and no secret in the query string or the web server's access log.
  Tried on the first request per credential set and remembered, falling back to the
  handshake on a server that refuses it. A password cannot use this path.
* Trades credentials for a session token that holds either an API key or a
  time-salted password hash in the same field. The token is cached per
  CREDENTIAL SET (a rotated password never reuses the old session) and
  re-handshaked once when a session lapses mid-run; the API-key attempt is only
  made for a key-shaped secret, so a real password never travels in a query
  string. The caller's timeout bounds the handshake too, not just the data call.
* Fetches catalogues, recent albums, album tracks, search results and the whole
  song list with pagination, honouring MUSIC_LIBRARIES by resolving it to Ampache
  catalog ids and pushing that filter into the server query as
  ``cond=catalog,<id>`` - one browse per catalogue, since conditions combine
  rather than alternate - while still enforcing it locally, because Ampache
  IGNORES a condition it does not understand instead of refusing it. Every page
  asks for a real page size so "analyse every album" is not capped at the
  server's first page.
* Downloads the original file rather than a transcoded stream, refusing a
  response that carries an Ampache JSON error under HTTP 200 instead of audio.
* Answers "which tracks are on this album" with a ``browse`` when only ids are
  wanted, which returns a name array instead of hydrating every song. The analysis
  dispatcher needs ids and a count to decide what to enqueue and nothing more.
  Anything that cannot be trusted - an unknown catalogue id, a total that does not
  match the rows returned - falls back to the full ``album_songs`` fetch, because a
  short list would make an unfinished album look complete.
* Serves lyrics from the album fetch that already returned them. Ampache
  serialises a single ``song`` and an ``album_songs`` row through the same
  songs_array, so asking per track re-paid that row's whole hydration cost to
  re-read one field. A caller with no album context still falls back to ``song``.
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
from datetime import datetime, timezone

import config
from . import context
from .helper import detect_download_extension, detect_path_format

logger = logging.getLogger(__name__)

_API_VERSION = '8.0.0'
# Ampache 8 is the floor, not a preference: album browsing needs the catalogue id
# on album objects and the `cond` browse filter, and older servers answer the same
# calls differently (or not at all). test_connection warns when a server reports
# an older API so the setup wizard says so instead of an analysis finding nothing.
_MIN_API_MAJOR = 8

# Fallback session window, used only when a handshake reports no usable
# session_expire. The real length comes from the server.
_TOKEN_TTL_SECONDS = 3000
# Ampache extends a session's expiry on every authenticated call, so the cached
# window slides forward on each success rather than counting down from the
# handshake. Renewing at a fraction of the window keeps a request from leaving
# just after the server has dropped the session.
_SESSION_RENEW_MARGIN = 0.9
_HANDSHAKE_TIMEOUT_SECONDS = 30
_REQUEST_TIMEOUT_SECONDS = 60
_PAGE_SIZE = 500
_SESSION_EXPIRED_CODE = '4701'
_AUTH_ERROR_CODES = ('4742', '4704')

_token_cache = {}
_token_lock = threading.Lock()

# Ampache reads an API key straight from an Authorization header and creates the
# session itself, so a key-shaped secret needs no handshake at all. Whether a
# given server accepts that is discovered on the first real request per credential
# set and remembered here - True: header auth works, False: it was refused and the
# handshake is used instead. Deliberately not configurable: there is nothing for an
# operator to decide that the first response does not answer.
_header_auth = {}
_header_auth_lock = threading.Lock()

# Ampache serialises a single `song` and an `album_songs` row through the SAME
# Json8_Data::songs_array, so an album fetch has already returned the lyrics the
# lyrics stage would otherwise ask for one track at a time - and that per-song call
# repeats every bit of hydration (rating, userflag, art, album and artist lookups)
# the album fetch just paid for. The rows are kept here for the lyrics stage, which
# runs in the same job and process as the album fetch that filled it.
#
# Keyed by track id alone, with no credential set in the key: `lyrics` comes off the
# song row itself, not from the calling user (unlike rating and flag, which
# songs_array resolves per user), so it does not vary between callers.
_album_lyrics = {}
_album_lyrics_lock = threading.Lock()
# A whole-library run walks every album in one job, so the cache needs a ceiling.
_LYRICS_CACHE_MAX = 5000
# A cached None means "Ampache says this song has no lyrics", which is an answer and
# needs no request. Only an absent entry justifies one, so absence needs a sentinel
# that None cannot be confused with.
_LYRICS_UNCACHED = object()

# Stock Ampache 8 requires `catalog` on a sub-type browse alongside `filter`
# (BrowseMethod.php: `foreach (['filter', 'catalog'] as $parameter)`), so the album's
# catalogue id has to be known before its songs can be browsed. Album discovery
# already reads that field off every album row - the API-8 gate and the library
# filter both depend on it - so it is recorded here for the dispatch loop, which runs
# in the same process as the discovery that fills it.
_album_catalogs = {}
_album_catalogs_lock = threading.Lock()
# Bounded by the library's album count in practice. Unlike the lyrics cache this is
# NOT cleared wholesale on overflow: entries are added during discovery and read
# later, so clearing would strand albums the loop has not reached yet. Overflow stops
# recording instead, and the albums past the ceiling fall back to the heavier fetch.
_ALBUM_CATALOG_CACHE_MAX = 100000

_SECRET_QUERY_PARAM = re.compile(r'(?i)([?&](?:auth|passphrase|password)=)[^&\s]*')
_API_KEY_SHAPE = re.compile(r'\A[0-9a-fA-F]{32,}\Z')

_CATALOGUE_KEYS = (
    'Id', 'Name', 'AlbumArtist', 'ArtistId', 'OriginalAlbumArtist', 'Album',
    'Path', 'FilePath', 'Year', 'Rating', 'DurationSeconds',
)


def _redact_ampache_secrets(text):
    return _SECRET_QUERY_PARAM.sub(r'\1[REDACTED]', str(text))


def _log_error(message, error):
    """Log a failure WITHOUT its traceback, redacting the exception text.

    Every Ampache call carries its session token (and the handshake its passphrase)
    in the query string, so a traceback - which prints the request URL verbatim.
    Logging lives in this helper rather than inline.
    """
    logger.error("%s: %s", message, _redact_ampache_secrets(error))


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
        params['version'] = _API_VERSION
        try:
            response = requests.get(
                f"{url}/server/json.server.php",
                params=params,
                timeout=timeout or _HANDSHAKE_TIMEOUT_SECONDS,
            )
            body = response.json()
        except Exception as e:
            _log_error("Ampache handshake failed", e)
            return None, {'kind': 'network', 'message': str(_redact_ampache_secrets(e))}

        if isinstance(body, dict) and body.get('auth'):
            return body, None

    error = (body or {}).get('error') if isinstance(body, dict) else None
    message = (error or {}).get('message') or 'Ampache handshake was rejected'
    return None, {'kind': 'auth', 'message': message}


def _session_lifetime(body):
    """How long a fresh session lasts, in seconds, from the handshake body.

    Ampache reports an absolute ISO-8601 ``session_expire``. What the cache needs
    is a LENGTH, because that expiry slides forward on every authenticated call,
    so the timestamp is converted once and the duration reapplied on each
    success. Falls back to the conservative default when the field is missing,
    unparseable, or implausibly short - an older server, a proxy that rewrites
    it, or a clock disagreement between us and the server.
    """
    raw = str((body or {}).get('session_expire') or '').strip()
    if not raw:
        return _TOKEN_TTL_SECONDS
    try:
        expires_at = datetime.fromisoformat(raw.replace('Z', '+00:00'))
    except ValueError:
        return _TOKEN_TTL_SECONDS
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    seconds = (expires_at - datetime.now(timezone.utc)).total_seconds()
    return seconds if seconds >= 60 else _TOKEN_TTL_SECONDS


def _token(user_creds=None, force=False, timeout=None):
    url, user, password = _creds(user_creds)
    if not url or not password:
        logger.warning("Ampache URL or password is not configured.")
        return None, None, {'kind': 'config', 'message': 'Ampache URL or password is not configured.'}

    key = _cache_key(url, user, password)
    with _token_lock:
        cached = _token_cache.get(key)
        # A header-auth entry exists for the version gate and may hold no token,
        # so require one rather than handing back an empty session.
        if cached and not force and cached.get('token') and cached.get('expires', 0) > time.time():
            return url, cached['token'], None

    body, err = _handshake(url, user, password, timeout=timeout)
    if not body:
        return url, None, err

    lifetime = _session_lifetime(body)
    with _token_lock:
        _token_cache[key] = {
            'token': body['auth'],
            # Kept so every later success can reapply the same window.
            'lifetime': lifetime,
            'expires': time.time() + lifetime * _SESSION_RENEW_MARGIN,
            # The handshake already reports the API version, so the version gate
            # needs no extra round trip.
            'api': str(body.get('api') or body.get('version') or ''),
        }
    return url, body['auth'], None


def _extend_session(user_creds=None):
    """Slide the cached expiry forward after a call the server accepted.

    Ampache resets a session's expiry on every authenticated request, so a token
    that is kept busy never needs re-issuing. Without this the cache counts down
    from the handshake instead, and a long analysis re-handshakes on a timer while
    the session it already holds is still perfectly valid - once per window, per
    worker process, for the whole run.

    Only called on success. A session that really has lapsed is what the 4701
    retry in ``_request_ex`` is for.
    """
    url, user, password = _creds(user_creds)
    if not url:
        return
    key = _cache_key(url, user, password)
    with _token_lock:
        cached = _token_cache.get(key)
        if not cached:
            return
        lifetime = cached.get('lifetime') or _TOKEN_TTL_SECONDS
        cached['expires'] = time.time() + lifetime * _SESSION_RENEW_MARGIN


def _header_auth_state(url, user, password):
    with _header_auth_lock:
        return _header_auth.get(_cache_key(url, user, password))


def _remember_header_auth(url, user, password, works):
    with _header_auth_lock:
        _header_auth[_cache_key(url, user, password)] = works


def _bootstrap_header_auth(url, user, password, timeout=None):
    """Settle header auth for these credentials with a single bearer ping.

    A bearer-authenticated ``ping`` answers with everything the handshake would
    have returned - ``api``, ``session_expire`` and an ``auth`` token - so one call
    both proves the key is accepted and fills the cache the version gate and the
    expiry window read. It REPLACES the handshake rather than adding to it, and it
    is cheaper: no password on the wire, and nothing secret in the query string.

    Returns True when header auth is available. A refusal is remembered so the
    handshake is used from then on; a network failure is not, because it says
    nothing about whether the header would be accepted.
    """
    response, err = _fetch(url, 'ping', None, None, False, timeout, api_key=password)
    if err:
        return False

    try:
        body = response.json()
    except Exception:
        body = None
    if not isinstance(body, dict):
        body = {}
    error, _code = _error_from_body(body)

    # An accepted bearer ping carries the server_details payload. Ampache answers
    # an unauthenticated ping with server/version only, so requiring `api` is what
    # separates "the key was taken" from "the endpoint merely replied".
    if error or not body.get('api'):
        _remember_header_auth(url, user, password, False)
        logger.info(
            "Ampache would not take the API key as a bearer token (%s); using the "
            "handshake for this server instead.",
            (error or {}).get('message') or (error or {}).get('errorMessage') or 'no api in ping',
        )
        return False

    lifetime = _session_lifetime(body)
    with _token_lock:
        _token_cache[_cache_key(url, user, password)] = {
            'token': str(body.get('auth') or ''),
            'lifetime': lifetime,
            'expires': time.time() + lifetime * _SESSION_RENEW_MARGIN,
            'api': str(body.get('api') or body.get('version') or ''),
        }
    _remember_header_auth(url, user, password, True)
    logger.info(
        "Ampache accepted the API key in an Authorization header (API %s), so this "
        "server needs no handshake - it opens the session itself.",
        body.get('api'),
    )
    return True


def _header_auth_ready(url, user, password, timeout=None):
    """True when requests for these credentials should carry a bearer header.

    Only a key-shaped secret qualifies: Ampache resolves a bearer token with
    findByApiKey, so a real password would just be refused - and being refused
    means having put the password on the wire for nothing.
    """
    if not url or not password or not _API_KEY_SHAPE.match(password):
        return False
    state = _header_auth_state(url, user, password)
    if state is not None:
        return state
    return _bootstrap_header_auth(url, user, password, timeout)


def _api_major(version):
    """Major API version from either the '8.0.0' or the packed '600000' form."""
    text = str(version or '').strip()
    if '.' in text:
        head = text.split('.', 1)[0]
        return int(head) if head.isdigit() else None
    if text.isdigit():
        return int(text[0]) if len(text) >= 4 else int(text)
    return None


def _cached_api_version(user_creds=None):
    url, user, password = _creds(user_creds)
    key = _cache_key(url, user, password)
    with _token_lock:
        cached = _token_cache.get(key) or {}
    # Filled by the handshake, or by the bearer ping in header-auth mode.
    return cached.get('api')


def _api_version_warning(user_creds=None):
    """Warn when the server is older than the API this backend is written against."""
    major = _api_major(_cached_api_version(user_creds))
    if major is None or major >= _MIN_API_MAJOR:
        return None
    return (
        f'This server reports Ampache API {major}, but AudioMuse-AI targets API '
        f'{_MIN_API_MAJOR} or newer. Album browsing needs the catalogue id on album '
        'objects and the "cond" browse filter that API 8 provides, so on an older '
        'server an analysis can find no albums or ignore your library selection. '
        'Upgrade Ampache before relying on this connection.'
    )


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


def _fetch(url, action, token, params, stream, timeout, api_key=None):
    all_params = {'action': action, 'version': _API_VERSION, **(params or {})}
    headers = None
    if api_key:
        # Ampache only looks at the Authorization header when there is NO auth
        # parameter at all (ApiHandler: `if (!isset($input['auth']))`), so the
        # token has to be absent from the query string, not merely empty.
        headers = {'Authorization': f'Bearer {api_key}'}
    else:
        all_params['auth'] = token
    try:
        response = requests.get(
            f"{url}/server/json.server.php",
            params=all_params,
            headers=headers,
            stream=stream,
            timeout=timeout or _REQUEST_TIMEOUT_SECONDS,
        )
    except Exception as e:
        _log_error(f"Ampache request '{action}' failed", e)
        return None, {'kind': 'network', 'message': str(_redact_ampache_secrets(e))}
    return response, None


def _stream_payload(action, response):
    try:
        response.raise_for_status()
    except Exception as e:
        _log_error(f"Ampache stream '{action}' failed", e)
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


def _response_payload(action, response, stream):
    """Split a response into (passthrough, parsed body, error).

    A streamed response that really is audio is handed back as ``passthrough``
    for the caller to consume; anything else is parsed so the Ampache error
    envelope (which arrives under HTTP 200) can be inspected.
    """
    if stream:
        return _stream_payload(action, response)
    try:
        return None, response.json(), None
    except Exception as e:
        return None, None, {'kind': 'parse', 'message': str(_redact_ampache_secrets(e))}


def _evaluate(action, response, stream, user_creds, extend):
    """A fetched response as ('ok' | 'retry' | 'error', payload, error)."""
    passthrough, body, err = _response_payload(action, response, stream)
    if err:
        return 'error', None, err
    if passthrough is not None:
        if extend:
            _extend_session(user_creds)
        return 'ok', passthrough, None

    verdict, err = _body_error(action, body, stream)
    if verdict == 'ok':
        if extend:
            _extend_session(user_creds)
        return 'ok', body, None
    return verdict, None, err


def _attempt_request(action, params, stream, user_creds, timeout, force):
    """One authenticate-and-fetch attempt, reporting 'ok' / 'retry' / 'error'.

    'retry' means Ampache reported an expired session, which the caller answers
    by re-authenticating once with ``force=True``.
    """
    url, user, password = _creds(user_creds)
    if _header_auth_ready(url, user, password, timeout):
        # Nothing to slide: Ampache opens and extends the session for a
        # header-authenticated call itself, so the cached window is only there for
        # the version gate.
        response, err = _fetch(url, action, None, params, stream, timeout, api_key=password)
        if err:
            return 'error', None, err
        return _evaluate(action, response, stream, user_creds, extend=False)

    url, token, err = _token(user_creds, force=force, timeout=timeout)
    if not token:
        return 'error', None, err

    response, err = _fetch(url, action, token, params, stream, timeout)
    if err:
        return 'error', None, err
    return _evaluate(action, response, stream, user_creds, extend=True)


def _request_ex(action, params=None, stream=False, user_creds=None, timeout=None):
    for attempt in (0, 1):
        verdict, payload, err = _attempt_request(
            action, params, stream, user_creds, timeout, force=bool(attempt)
        )
        if verdict == 'ok':
            return payload, None
        if verdict == 'error':
            return None, err

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


def _page_size():
    """Rows per browse page, from config so a fast server can raise it.

    Bigger pages cut request overhead but NOT the server's cost, which is
    per-song: Ampache hydrates every row and looks up its rating, art, album and
    artist names individually. Raising this shortens the request count, not the
    work, and makes each request longer.
    """
    size = getattr(config, 'AMPACHE_PAGE_SIZE', _PAGE_SIZE)
    try:
        size = int(size)
    except (TypeError, ValueError):
        return _PAGE_SIZE
    return size if size > 0 else _PAGE_SIZE


def _catalogue_query_plan(catalog_ids):
    """One song browse per target catalogue, or a single unfiltered browse.

    A plain ``songs`` browse with ``cond=catalog,<id>`` compiles to an indexed
    ``song.catalog = <id>`` with no join, where the equivalent ``advanced_search``
    with the catalogues OR'd together builds a much heavier query for the same
    rows. Album discovery already browses one catalogue at a time for this reason,
    and ``cond`` conditions combine rather than alternate, so several catalogues
    are one browse each either way.

    Each entry is ``(params, catalog_id)``; ``catalog_id`` is None when nothing is
    being filtered and there is therefore nothing to verify.
    """
    if not catalog_ids:
        return [({}, None)]
    return [({'cond': f'catalog,{catalog_id}'}, catalog_id) for catalog_id in sorted(catalog_ids)]


def _cond_trusted(rows, catalog_id):
    """False when a ``cond=catalog`` browse returned rows it should not have.

    Ampache IGNORES a ``cond`` it does not understand instead of refusing it, so a
    browse that should be one catalogue can quietly be the whole library. That
    matters more here than for albums: with one browse per catalogue, an ignored
    condition would walk the entire library once per selected catalogue. Rows
    carry their catalogue id, so the first page settles it. A page that reports no
    catalogue id cannot be verified, which is treated the same as being ignored.
    """
    reported = [row for row in rows if 'catalog' in row]
    if not reported:
        return False
    return all(str(row.get('catalog')) == str(catalog_id) for row in reported)


def _in_catalogs(row, catalog_ids):
    return catalog_ids is None or str(row.get('catalog')) in catalog_ids


def _filter_album_rows(rows, catalog_ids):
    """Keep the rows inside the target catalogues; report a page that names none.

    Ampache IGNORES a ``cond`` it does not understand instead of failing, so a
    filtered browse can come back as the whole library. Re-checking each row's
    catalogue id is therefore not belt-and-braces, it is what stops an ignored
    filter from silently analysing everything. A page that reports no catalogue
    ids at all means the server cannot express the filter (API 8 added the field),
    which the caller surfaces rather than passing off as an empty library.

    Returns ``(kept rows, server reported no catalogue ids)``.
    """
    if not catalog_ids:
        return rows, False
    reported = [row for row in rows if 'catalog' in row]
    if not reported:
        return [], True
    return [row for row in reported if str(row.get('catalog')) in catalog_ids], False


def _log_catalogue_fetch_failure(err, offset, collected):
    logger.error(
        "AMPACHE CATALOGUE FETCH FAILED at offset %d after %d songs (%s). The "
        "returned catalogue is INCOMPLETE - do not treat missing tracks as deleted.",
        offset, collected, (err or {}).get('message') or 'unknown error',
    )


def _collect_songs_for(params, catalog_id, catalog_ids, songs, user_creds):
    """Page one browse into ``songs``, reporting how it ended.

    ``'ok'``        - the browse was walked to the end.
    ``'untrusted'`` - the catalogue condition was ignored, so the whole plan has
                      to be abandoned rather than repeated per catalogue.
    ``'failed'``    - a page could not be fetched twice running, so what has been
                      collected is INCOMPLETE.

    A failed page is retried once before giving up. That matters: the previous
    behaviour abandoned the catalogue filter on the first failure and re-walked the
    entire unfiltered library, so one transient error on page one cost a full-
    library enumeration.
    """
    offset = 0
    retried = False
    page = _page_size()
    while True:
        body, err = _request_ex(
            'songs', {**params, 'offset': offset, 'limit': page}, user_creds=user_creds
        )
        if body is None:
            if not retried:
                retried = True
                logger.warning(
                    "Ampache song browse failed at offset %d (%s); retrying that page "
                    "once before giving up on it.",
                    offset, (err or {}).get('message') or 'unknown error',
                )
                continue
            _log_catalogue_fetch_failure(err, offset, len(songs))
            return 'failed'
        retried = False

        rows = body.get('song') or []
        if not rows:
            return 'ok'
        if catalog_id is not None and offset == 0 and not _cond_trusted(rows, catalog_id):
            return 'untrusted'

        songs.extend(_map_catalogue_song(r) for r in rows if _in_catalogs(r, catalog_ids))
        offset += len(rows)
        if len(rows) < page:
            return 'ok'


def get_all_songs(user_creds=None, apply_filter=True):
    catalog_ids = _target_catalog_ids(user_creds=user_creds) if apply_filter else None
    if isinstance(catalog_ids, set) and not catalog_ids:
        return []

    songs = []
    for params, catalog_id in _catalogue_query_plan(catalog_ids):
        verdict = _collect_songs_for(params, catalog_id, catalog_ids, songs, user_creds)
        if verdict == 'failed':
            break
        if verdict == 'untrusted':
            logger.warning(
                "Ampache ignored the catalog condition on a song browse, so browsing "
                "per catalogue would walk the whole library once for each of %d "
                "catalogues. Falling back to one unfiltered fetch filtered locally.",
                len(catalog_ids or ()),
            )
            # Start clean: anything already collected came from a browse that was
            # not filtering, and the replacement walk covers the same ground from
            # offset 0. Reusing the partial list would double-count it.
            songs.clear()
            _collect_songs_for({}, None, catalog_ids, songs, user_creds)
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


def _album_query_plan(catalog_ids):
    """One album browse per target catalogue, or a single unfiltered browse.

    Ampache filters a browse with ``cond=<field>,<value>``, and conditions
    combine, so several catalogues are not expressible as one ``cond``. Issuing
    one filtered browse per catalogue keeps each response's newest-first order
    intact and merges cleanly; the common case of a single catalogue stays a
    single query.
    """
    if not catalog_ids:
        return [{}]
    return [{'cond': f'catalog,{catalog_id}'} for catalog_id in sorted(catalog_ids)]


def _remember_album_catalogs(rows):
    """Record each album's catalogue id so its songs can be browsed later."""
    fresh = {
        str(row.get('id')): str(row.get('catalog'))
        for row in rows
        if isinstance(row, dict) and row.get('id') is not None and row.get('catalog') is not None
    }
    if not fresh:
        return
    with _album_catalogs_lock:
        if len(_album_catalogs) >= _ALBUM_CATALOG_CACHE_MAX:
            return
        _album_catalogs.update(fresh)


def _cached_album_catalog(album_id):
    with _album_catalogs_lock:
        return _album_catalogs.get(str(album_id))


def _log_album_fetch_failure(err, offset, collected):
    logger.error(
        "AMPACHE ALBUM FETCH FAILED at offset %d after %d albums (%s). Album "
        "discovery is INCOMPLETE - do not read this as an empty library. With a "
        "library filter set, check that this server supports the 'cond' browse "
        "filter on the albums action.",
        offset, collected, (err or {}).get('message') or 'unknown error',
    )


def _album_page(base_params, offset, page, collected):
    params = {**base_params, 'offset': offset, 'limit': page, 'sort': 'addition_time,DESC'}
    body, err = _request_ex('albums', params)
    if body is None:
        _log_album_fetch_failure(err, offset, collected)
        return None
    return body.get('album') or []


def _collect_albums_for(base_params, catalog_ids, fetch_all, wanted, collected):
    """Page one browse into ``collected``, keyed by album id so merges dedupe.

    Every page asks for a real page size: ``limit=0`` is not a portable "no
    limit" on Ampache, and combining it with a single pass meant an install
    configured to analyse EVERY album only ever saw the server's first page.
    """
    offset = 0
    page = _page_size() if (fetch_all or catalog_ids) else max(wanted, 1)
    while True:
        rows = _album_page(base_params, offset, page, len(collected))
        if not rows:
            return
        kept, unreported = _filter_album_rows(rows, catalog_ids)
        if unreported:
            logger.error(
                "AMPACHE REPORTED NO CATALOGUE IDS on its albums, so the library "
                "filter %s cannot be verified and album discovery is stopping rather "
                "than analysing the whole library. Ampache API %s or newer is "
                "required for a library-filtered install.",
                sorted(catalog_ids), _MIN_API_MAJOR,
            )
            return
        _remember_album_catalogs(kept)
        for row in kept:
            collected.setdefault(str(row.get('id')), _map_album(row))
        offset += len(rows)
        if len(rows) < page or (not fetch_all and len(collected) >= wanted):
            return


def _collect_recent_albums(catalog_ids, fetch_all, wanted):
    collected = {}
    for base_params in _album_query_plan(catalog_ids):
        _collect_albums_for(base_params, catalog_ids, fetch_all, wanted, collected)
        if not fetch_all and len(collected) >= wanted:
            break
    return list(collected.values())


def get_recent_albums(limit):
    fetch_all = not limit or int(limit) <= 0
    wanted = 0 if fetch_all else int(limit)

    catalog_ids = _target_catalog_ids()
    if isinstance(catalog_ids, set) and not catalog_ids:
        return []

    mapped = _collect_recent_albums(catalog_ids, fetch_all, wanted)
    if not mapped:
        logger.warning(
            "AMPACHE RETURNED NO ALBUMS%s, so there is nothing to analyse. Treat this "
            "as a configuration or API problem unless the server really is empty.",
            f" for catalogs {sorted(catalog_ids)}" if catalog_ids else '',
        )
    return mapped if fetch_all else mapped[:wanted]


def _remember_album_lyrics(rows):
    """Keep the lyrics an ``album_songs`` response already carried.

    Only a row that actually carries the field is remembered. ``lyrics: null`` is
    Ampache answering "this song has none", which is worth caching; a row missing the
    key entirely says nothing and must still fall back to a request.
    """
    fresh = {
        str(row.get('id')): row.get('lyrics') or None
        for row in rows
        if isinstance(row, dict) and 'lyrics' in row and row.get('id') is not None
    }
    if not fresh:
        return
    with _album_lyrics_lock:
        # Dropped wholesale rather than by age: the lyrics stage reads an entry in the
        # same job, right after the album is fetched, and the album just fetched
        # survives because it is added after the clear. A dropped entry costs one
        # fallback request, never a wrong answer.
        if len(_album_lyrics) + len(fresh) > _LYRICS_CACHE_MAX:
            _album_lyrics.clear()
        _album_lyrics.update(fresh)


def _cached_album_lyrics(track_id):
    with _album_lyrics_lock:
        return _album_lyrics.get(str(track_id), _LYRICS_UNCACHED)


def get_tracks_from_album(album_id, user_creds=None):
    body = _request('album_songs', {'filter': album_id}, user_creds=user_creds)
    rows = (body or {}).get('song') or []
    _remember_album_lyrics(rows)
    return [_map_song(s) for s in rows]


def _browse_total_count(body, ids):
    """True when the browse reported a total that matches the rows it returned.

    A ``total_count`` that disagrees means the answer was truncated (or is not the
    shape expected), and a short list is worse here than a failed request: the
    dispatch loop compares this count against the number of tracks already analysed,
    so a truncated album looks finished and is silently skipped. An absent
    ``total_count`` is treated the same way, since there is then nothing to check
    against.
    """
    try:
        return int(body.get('total_count')) == len(ids)
    except (TypeError, ValueError):
        return False


def _browse_album_track_ids(album_id, user_creds=None):
    """Song ids for one album via ``browse``, or None when it cannot be trusted.

    ``browse`` answers through ``Catalog::get_name_array`` - id, name, prefix and
    basename, nothing else - so it skips the per-song hydration (rating, userflag, art,
    album name, artist names) that makes an ``album_songs`` row expensive to produce.
    A caller that only needs ids and a count should not pay for the rest.

    Deliberately sends no ``offset``/``limit``: the album is asked for whole, and a
    reply that does not account for every row is rejected rather than paged. Returns
    None for every case a caller must not read as "this album has no tracks", so the
    fallback stays responsible for that distinction.
    """
    params = {'type': 'album', 'filter': album_id}
    catalog_id = _cached_album_catalog(album_id)
    if catalog_id:
        params['catalog'] = catalog_id
    body, err = _request_ex('browse', params, user_creds=user_creds)
    if body is None:
        logger.debug(
            "Ampache browse for album %s failed (%s); falling back to album_songs.",
            album_id, (err or {}).get('message') or 'unknown error',
        )
        return None

    rows = body.get('browse')
    if not isinstance(rows, list) or not rows:
        return None
    ids = [
        str(row.get('id'))
        for row in rows
        if isinstance(row, dict) and row.get('id') is not None
    ]
    if not ids or not _browse_total_count(body, ids):
        logger.debug(
            "Ampache browse for album %s returned %d usable ids for a reported total "
            "of %r; falling back to album_songs rather than risk a short list.",
            album_id, len(ids), body.get('total_count'),
        )
        return None
    return ids


def get_album_track_ids(album_id, user_creds=None):
    """Track ids for one album, without serialising each track's metadata.

    Falls back to the full ``album_songs`` fetch whenever ``browse`` cannot be
    trusted - a server that will not browse without a catalogue id it never reported,
    an album whose catalogue is unknown, or any answer that does not account for every
    row. The fallback returns the same ids, just more expensively, so a refusal costs
    speed and never correctness.
    """
    ids = _browse_album_track_ids(album_id, user_creds=user_creds)
    if ids is not None:
        return ids
    return [
        str(track.get('Id') or track.get('id'))
        for track in (get_tracks_from_album(album_id, user_creds=user_creds) or [])
    ]


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
        _log_error(f"Failed to download Ampache track {item.get('Name', 'Unknown')}", e)
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

    version_warning = _api_version_warning(user_creds)
    if version_warning:
        warnings.append(version_warning)

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


def delete_playlist(playlist_id, user_creds=None):
    return _request('playlist_delete', {'filter': playlist_id}, user_creds=user_creds) is not None


def create_playlist(base_name, item_ids, user_creds=None):
    body = _request(
        'playlist_create', {'name': base_name, 'type': 'private'}, user_creds=user_creds
    )
    playlist = (body or {}).get('playlist') or {}
    playlist_id = playlist.get('id') or (body or {}).get('id')
    if not playlist_id:
        logger.error(f"Ampache refused to create playlist '{base_name}'.")
        return None

    wanted = list(item_ids or [])
    added = 0
    for item_id in wanted:
        if _request(
            'playlist_add_song',
            {'filter': playlist_id, 'song': item_id, 'check': 1},
            user_creds=user_creds,
        ) is not None:
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
    return create_playlist(
        f"{playlist_name.strip()}_instant", item_ids, user_creds=context.active_creds(user_creds)
    )


def create_or_replace_playlist(playlist_name, item_ids, user_creds=None):
    user_creds = context.active_creds(user_creds)
    existing = get_playlist_by_name(playlist_name, user_creds=user_creds)
    if existing and existing.get('Id') and not delete_playlist(
        existing['Id'], user_creds=user_creds
    ):
        logger.error(
            f"Ampache create_or_replace_playlist: failed to delete existing "
            f"'{playlist_name}' (id={existing['Id']}); aborting to avoid creating a duplicate"
        )
        return None
    return create_playlist(playlist_name, item_ids, user_creds=user_creds)


def get_top_played_songs(limit, user_creds):
    body = _request('stats', {'type': 'song', 'filter': 'frequent', 'limit': limit}, user_creds=user_creds)
    return [_map_song(s) for s in ((body or {}).get('song') or [])]


def get_last_played_time(_item_id, _user_creds=None):
    """Not available: Ampache reports play stats per library, not per track.

    ``stats`` can rank songs by play count but exposes no per-song "last played
    at" timestamp, so there is nothing to return. The parameters are part of the
    dispatcher contract and are deliberately unused (leading underscores).
    Recency-weighted callers such as the Sonic Fingerprint treat ``None`` as
    "unknown" and fall back to play counts.
    """
    logger.debug("Ampache exposes no per-track last-played timestamp; returning None.")
    return None


def get_lyrics(track_id: str, timeout: float = 2.5):
    """Lyrics for one track, preferring what the album fetch already returned.

    The analysis path reaches every track through ``get_tracks_from_album``, whose
    response carries this field already, so the common case costs no request at all.
    A fetch that never happened (a caller with no album context, or an evicted entry)
    still falls back to the single-song call.
    """
    cached = _cached_album_lyrics(track_id)
    if cached is not _LYRICS_UNCACHED:
        return cached
    body = _request('song', {'filter': track_id}, timeout=timeout)
    songs = (body or {}).get('song') or []
    if isinstance(songs, list) and songs:
        return songs[0].get('lyrics') or None
    return None
