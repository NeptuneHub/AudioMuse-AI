"""Direct HTTP probes for target media servers, for the provider migration tool.

This module deliberately does NOT read ``config.py`` globals. Every function
takes a ``creds`` dict so the migration tool can test and migrate to a new
provider without mutating the running app's active configuration.

Supported providers:
  * ``jellyfin`` / ``emby`` — X-Emby-Token header API, identical shape
  * ``navidrome``           — Subsonic JSON API
  * ``lyrion``              — Logitech Media Server JSON-RPC (``/jsonrpc.js``)
  * ``mpd``                 — Socket protocol (stub — raises NotImplementedError)

Unified track dict shape returned by ``fetch_all_tracks`` and
``get_album_tracks``::

    {
      'id':           str,   # provider's native item id
      'path':         str|None,
      'title':        str|None,
      'artist':       str|None,
      'album_artist': str|None,
      'album':        str|None,
      'year':         int|None,
      'track_number': int|None,
      'disc_number':  int|None,
    }

Unified album dict shape returned by ``search_albums``::

    {
      'id':          str,
      'name':        str,
      'artist':      str|None,
      'year':        int|None,
      'track_count': int|None,
    }

``test_connection`` returns::

    {
      'ok':           bool,
      'error':        str|None,
      'sample_count': int,
      'path_format':  'absolute'|'relative'|'none'|'mixed',
      'warnings':     list[str],
    }
"""
import logging
from urllib.parse import unquote, urlparse

import requests

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 60
_SAMPLE_LIMIT = 100


# ---------------------------------------------------------------------------
# Path-format detection
# ---------------------------------------------------------------------------

def _detect_path_format(tracks):
    """Classify a sample of tracks by whether their ``path`` fields look
    absolute, relative, missing, or mixed.

    Used by ``test_connection`` to warn Navidrome users when "Report Real Path"
    is disabled (all paths come back as relative or None).
    """
    with_path = [t for t in tracks if t.get('path')]
    if not with_path:
        return 'none'
    abs_count = sum(1 for t in with_path if str(t['path']).startswith('/'))
    ratio = abs_count / len(with_path)
    # Strongly skewed either way → call it that. Anything in between → mixed.
    if ratio >= 0.8:
        return 'absolute'
    if ratio <= 0.2:
        return 'relative'
    return 'mixed'


# ---------------------------------------------------------------------------
# Jellyfin / Emby
# ---------------------------------------------------------------------------

def _jellyfin_headers(creds):
    return {'X-Emby-Token': creds.get('token', '')}


def _jellyfin_track(item):
    """Translate a Jellyfin/Emby Items entry into the unified shape."""
    return {
        'id':           item.get('Id'),
        'path':         item.get('Path'),
        'title':        item.get('Name'),
        'artist':       item.get('AlbumArtist'),
        'album_artist': item.get('AlbumArtist'),
        'album':        item.get('Album'),
        'year':         item.get('ProductionYear'),
        'track_number': item.get('IndexNumber'),
        'disc_number':  item.get('ParentIndexNumber'),
    }


def _jellyfin_fetch_page(creds, start_index=0, limit=None):
    """Single page of /Users/{uid}/Items with StartIndex/Limit.

    Returns (items, total_record_count). ``limit=None`` lets Jellyfin
    return whatever it considers a page (effectively "everything from
    StartIndex") which is fine when we're walking page-by-page.
    """
    url = f"{creds['url'].rstrip('/')}/Users/{creds['user_id']}/Items"
    params = {
        'IncludeItemTypes': 'Audio',
        'Recursive': 'true',
        'Fields': 'Path,ProductionYear,IndexNumber,ParentIndexNumber,AlbumArtist,Album',
        'StartIndex': start_index,
    }
    if limit is not None:
        params['Limit'] = limit
    r = requests.get(url, headers=_jellyfin_headers(creds), params=params, timeout=REQUESTS_TIMEOUT)
    r.raise_for_status()
    body = r.json() or {}
    items = body.get('Items') or []
    total = body.get('TotalRecordCount')
    return items, total


def _jellyfin_fetch_all(creds):
    """Walk every audio item with StartIndex/Limit pagination.

    Large Jellyfin libraries time out if we ask for everything in one
    request (no Limit), so we page at 500/call. Stops when a page
    returns fewer items than the page size or when cumulative reaches
    TotalRecordCount.
    """
    all_tracks = []
    start = 0
    page_size = 500
    while True:
        items, total = _jellyfin_fetch_page(creds, start_index=start, limit=page_size)
        if not items:
            break
        all_tracks.extend(_jellyfin_track(i) for i in items)
        if len(items) < page_size:
            break
        start += len(items)
        if total is not None and start >= total:
            break
    return all_tracks


def _jellyfin_search_albums(creds, query):
    url = f"{creds['url'].rstrip('/')}/Users/{creds['user_id']}/Items"
    params = {
        'IncludeItemTypes': 'MusicAlbum',
        'Recursive': 'true',
        'SearchTerm': query,
        'Limit': 10,
        'Fields': 'ChildCount,ProductionYear,AlbumArtist',
    }
    r = requests.get(url, headers=_jellyfin_headers(creds), params=params, timeout=REQUESTS_TIMEOUT)
    r.raise_for_status()
    items = r.json().get('Items', []) or []
    return [
        {
            'id':          item.get('Id'),
            'name':        item.get('Name'),
            'artist':      item.get('AlbumArtist'),
            'year':        item.get('ProductionYear'),
            'track_count': item.get('ChildCount'),
        }
        for item in items
    ]


def _jellyfin_get_album_tracks(creds, album_id):
    url = f"{creds['url'].rstrip('/')}/Users/{creds['user_id']}/Items"
    params = {
        'ParentId': album_id,
        'IncludeItemTypes': 'Audio',
        'Fields': 'Path,ProductionYear,IndexNumber,ParentIndexNumber,AlbumArtist,Album',
    }
    r = requests.get(url, headers=_jellyfin_headers(creds), params=params, timeout=REQUESTS_TIMEOUT)
    r.raise_for_status()
    items = r.json().get('Items', []) or []
    return [_jellyfin_track(i) for i in items]


def _jellyfin_test_connection(creds):
    # Sample a single small page instead of the full library — big
    # libraries took >60s to serve the Recursive=true, no-Limit call
    # and the wizard saw Read timed out.
    try:
        items, _total = _jellyfin_fetch_page(creds, start_index=0, limit=_SAMPLE_LIMIT)
    except Exception as e:
        logger.warning("Jellyfin/Emby probe failed: %s", e)
        return {'ok': False, 'error': str(e), 'sample_count': 0,
                'path_format': 'none', 'warnings': []}
    sample = [_jellyfin_track(i) for i in items]
    return {
        'ok':           True,
        'error':        None,
        'sample_count': len(sample),
        'path_format':  _detect_path_format(sample),
        'warnings':     [],
    }


# ---------------------------------------------------------------------------
# Navidrome (Subsonic JSON API)
# ---------------------------------------------------------------------------

def _navidrome_auth_params(creds):
    user = creds.get('user', '')
    password = creds.get('password', '')
    hex_pw = password.encode('utf-8').hex()
    return {
        'u': user,
        'p': f'enc:{hex_pw}',
        'v': '1.16.1',
        'c': 'AudioMuse-AI',
        'f': 'json',
    }


def _navidrome_request(creds, endpoint, extra_params=None):
    url = f"{creds['url'].rstrip('/')}/rest/{endpoint}.view"
    params = dict(_navidrome_auth_params(creds))
    if extra_params:
        params.update(extra_params)
    r = requests.get(url, params=params, timeout=REQUESTS_TIMEOUT)
    r.raise_for_status()
    body = r.json().get('subsonic-response', {}) or {}
    if body.get('status') == 'failed':
        err = body.get('error', {})
        raise RuntimeError(f"Navidrome API error: {err.get('message', 'unknown')}")
    return body


def _navidrome_track(song):
    return {
        'id':           song.get('id'),
        'path':         song.get('path'),
        'title':        song.get('title'),
        'artist':       song.get('artist'),
        'album_artist': song.get('albumArtist') or song.get('artist'),
        'album':        song.get('album'),
        'year':         song.get('year'),
        'track_number': song.get('track'),
        'disc_number':  song.get('discNumber'),
    }


def _navidrome_fetch_all(creds):
    all_tracks = []
    offset = 0
    page_size = 500
    while True:
        body = _navidrome_request(creds, 'search3', {
            'query': '',
            'songCount': page_size,
            'songOffset': offset,
            'artistCount': 0,
            'albumCount': 0,
        })
        songs = ((body.get('searchResult3') or {}).get('song')) or []
        if not songs:
            break
        all_tracks.extend(_navidrome_track(s) for s in songs)
        if len(songs) < page_size:
            break
        offset += len(songs)
    return all_tracks


def _navidrome_search_albums(creds, query):
    body = _navidrome_request(creds, 'search3', {
        'query': query,
        'albumCount': 10,
        'songCount': 0,
        'artistCount': 0,
    })
    albums = ((body.get('searchResult3') or {}).get('album')) or []
    return [
        {
            'id':          a.get('id'),
            'name':        a.get('name') or a.get('title'),
            'artist':      a.get('artist'),
            'year':        a.get('year'),
            'track_count': a.get('songCount'),
        }
        for a in albums
    ]


def _navidrome_get_album_tracks(creds, album_id):
    body = _navidrome_request(creds, 'getAlbum', {'id': album_id})
    album = body.get('album') or {}
    songs = album.get('song') or []
    return [_navidrome_track(s) for s in songs]


def _navidrome_test_connection(creds):
    warnings = []
    try:
        # Fetch just one page for the sample
        body = _navidrome_request(creds, 'search3', {
            'query': '',
            'songCount': _SAMPLE_LIMIT,
            'songOffset': 0,
            'artistCount': 0,
            'albumCount': 0,
        })
    except Exception as e:
        logger.warning("Navidrome probe failed: %s", e)
        return {'ok': False, 'error': str(e), 'sample_count': 0,
                'path_format': 'none', 'warnings': []}

    songs = ((body.get('searchResult3') or {}).get('song')) or []
    sample = [_navidrome_track(s) for s in songs]
    path_format = _detect_path_format(sample)

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
        'ok':           True,
        'error':        None,
        'sample_count': len(sample),
        'path_format':  path_format,
        'warnings':     warnings,
    }


# ---------------------------------------------------------------------------
# Lyrion (Logitech Media Server — JSON-RPC at /jsonrpc.js)
# ---------------------------------------------------------------------------
#
# LMS tag letters used in the ``titles`` query:
#   g = genre          a = artist     l = album     d = duration
#   u = url (file://)  A = albumartist  y = year     R = rating
#
# The request shape is always::
#
#     POST {url}/jsonrpc.js
#     {"id":1,"method":"slim.request","params":["",[<method>, <arg1>, <arg2>, ...]]}
#
# Most LMS installs have no auth, but self-hosted setups may gate it with HTTP
# basic auth — so ``_lyrion_jsonrpc`` honors ``creds['user']``/``creds['password']``
# when either is present.
#
# ``fetch_all_tracks`` skips tracks whose ``url`` or ``genre``/``service`` field
# contains "spotify" — those are streamed-from-Spotify entries that have no
# local file path and can't be matched for migration.


def _lyrion_jsonrpc(creds, method, params):
    """POST one slim.request to Lyrion and return the parsed JSON body."""
    base = (creds.get('url') or '').rstrip('/')
    if not base:
        raise ValueError('Lyrion creds missing "url"')
    url = f"{base}/jsonrpc.js"
    payload = {
        'id':     1,
        'method': 'slim.request',
        'params': ['', [method, *params]],
    }
    auth = None
    user, password = creds.get('user'), creds.get('password')
    if user or password:
        auth = (user or '', password or '')
    r = requests.post(url, json=payload, timeout=REQUESTS_TIMEOUT, auth=auth)
    r.raise_for_status()
    return r.json() or {}


def _lyrion_decode_url(url):
    """Turn a Lyrion ``url`` field into a filesystem path if possible.

    LMS returns ``file:///music/artist/album/track.flac`` for local tracks
    and leaves remote streams as plain ``http://`` or ``spotify:`` URIs.
    We return filesystem paths as-is and pass other schemes through so the
    caller can decide (the matcher only path-matches strings that start
    with ``/``).
    """
    if not url:
        return None
    if url.startswith('file://'):
        return unquote(urlparse(url).path) or None
    return unquote(url)


def _lyrion_is_spotify(raw):
    """Return True for LMS entries that point at Spotify instead of a file.

    Those rows show up in the ``titles`` response when the user has the
    Spotty plugin installed. They have no usable file path, so we drop them
    before they reach the matcher."""
    url = (raw.get('url') or '').lower()
    if 'spotify' in url:
        return True
    for fld in ('genre', 'service', 'source'):
        val = raw.get(fld)
        if isinstance(val, str) and 'spotify' in val.lower():
            return True
    return False


def _lyrion_track(raw):
    """Translate a titles_loop entry into the unified track dict shape."""
    track_artist = (
        raw.get('trackartist')
        or raw.get('artist')
        or raw.get('albumartist')
    )
    album_artist = raw.get('albumartist') or raw.get('artist')
    year = raw.get('year')
    if not isinstance(year, int):
        try:
            year = int(year) if year not in (None, '') else None
        except (TypeError, ValueError):
            year = None
    return {
        'id':           str(raw.get('id', '')) or None,
        'path':         _lyrion_decode_url(raw.get('url')),
        'title':        raw.get('title'),
        'artist':       track_artist,
        'album_artist': album_artist,
        'album':        raw.get('album'),
        'year':         year,
        'track_number': raw.get('tracknum'),
        'disc_number':  raw.get('disc'),
    }


def _lyrion_fetch_all(creds):
    """Walk every track via paginated ``titles`` calls (500/page)."""
    all_tracks = []
    offset = 0
    page_size = 500
    while True:
        body = _lyrion_jsonrpc(creds, 'titles', [offset, page_size, 'tags:galduAyR'])
        result = body.get('result') or {}
        page = result.get('titles_loop') or []
        if not page:
            break
        for raw in page:
            if _lyrion_is_spotify(raw):
                continue
            all_tracks.append(_lyrion_track(raw))
        if len(page) < page_size:
            break
        offset += len(page)
    return all_tracks


def _lyrion_search_albums(creds, query):
    body = _lyrion_jsonrpc(creds, 'albums', [0, 10, f'search:{query}', 'tags:lyja'])
    result = body.get('result') or {}
    albums = result.get('albums_loop') or []
    out = []
    for a in albums:
        year = a.get('year')
        if not isinstance(year, int):
            try:
                year = int(year) if year not in (None, '') else None
            except (TypeError, ValueError):
                year = None
        out.append({
            'id':          str(a.get('id', '')) or None,
            'name':        a.get('album') or a.get('title'),
            'artist':      a.get('artist') or a.get('albumartist'),
            'year':        year,
            'track_count': a.get('tracks') or a.get('count'),
        })
    return out


def _lyrion_get_album_tracks(creds, album_id):
    body = _lyrion_jsonrpc(
        creds, 'titles',
        [0, 999999, f'album_id:{album_id}', 'tags:galduAyR'],
    )
    result = body.get('result') or {}
    page = result.get('titles_loop') or []
    return [_lyrion_track(raw) for raw in page if not _lyrion_is_spotify(raw)]


def _lyrion_test_connection(creds):
    warnings = []
    try:
        body = _lyrion_jsonrpc(creds, 'titles', [0, _SAMPLE_LIMIT, 'tags:galduAyR'])
    except Exception as e:
        logger.warning("Lyrion probe failed: %s", e)
        return {'ok': False, 'error': str(e), 'sample_count': 0,
                'path_format': 'none', 'warnings': []}

    result = body.get('result') or {}
    raws = result.get('titles_loop') or []
    sample = [_lyrion_track(r) for r in raws if not _lyrion_is_spotify(r)]
    path_format = _detect_path_format(sample)

    if path_format != 'absolute':
        warnings.append(
            'Lyrion is returning relative paths, stream URIs, or no paths at '
            'all. Automatic path-based matching may be unreliable for this '
            'library — expect to manually match most albums in Step 4. '
            'Metadata-based matching still works.'
        )

    return {
        'ok':           True,
        'error':        None,
        'sample_count': len(sample),
        'path_format':  path_format,
        'warnings':     warnings,
    }


# ---------------------------------------------------------------------------
# MPD — stub
# ---------------------------------------------------------------------------

def _mpd_not_supported(*_args, **_kwargs):
    raise NotImplementedError(
        'MPD migration is not yet supported by the migration tool. '
        'MPD stores file paths relative to its music_directory and requires '
        'a socket-protocol client. Planned for a later release.'
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_DISPATCH = {
    'jellyfin': {
        'fetch_all_tracks': _jellyfin_fetch_all,
        'search_albums':    _jellyfin_search_albums,
        'get_album_tracks': _jellyfin_get_album_tracks,
        'test_connection':  _jellyfin_test_connection,
    },
    'emby': {
        # Emby is a Jellyfin fork with the same /Users/{uid}/Items endpoint.
        'fetch_all_tracks': _jellyfin_fetch_all,
        'search_albums':    _jellyfin_search_albums,
        'get_album_tracks': _jellyfin_get_album_tracks,
        'test_connection':  _jellyfin_test_connection,
    },
    'navidrome': {
        'fetch_all_tracks': _navidrome_fetch_all,
        'search_albums':    _navidrome_search_albums,
        'get_album_tracks': _navidrome_get_album_tracks,
        'test_connection':  _navidrome_test_connection,
    },
    'lyrion': {
        'fetch_all_tracks': _lyrion_fetch_all,
        'search_albums':    _lyrion_search_albums,
        'get_album_tracks': _lyrion_get_album_tracks,
        'test_connection':  _lyrion_test_connection,
    },
    'mpd': {
        'fetch_all_tracks': _mpd_not_supported,
        'search_albums':    _mpd_not_supported,
        'get_album_tracks': _mpd_not_supported,
        'test_connection':  _mpd_not_supported,
    },
}


def _dispatch(provider_type, op):
    t = (provider_type or '').lower()
    if t not in _DISPATCH:
        raise ValueError(
            f"Provider type '{provider_type}' not supported by migration probe. "
            f"Supported: {sorted(_DISPATCH.keys())}"
        )
    return _DISPATCH[t][op]


def fetch_all_tracks(provider_type, creds):
    return _dispatch(provider_type, 'fetch_all_tracks')(creds)


def search_albums(provider_type, creds, query):
    return _dispatch(provider_type, 'search_albums')(creds, query)


def get_album_tracks(provider_type, creds, album_id):
    return _dispatch(provider_type, 'get_album_tracks')(creds, album_id)


def test_connection(provider_type, creds):
    return _dispatch(provider_type, 'test_connection')(creds)
