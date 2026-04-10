"""Direct HTTP probes for target media servers, for the provider migration tool.

This module deliberately does NOT read ``config.py`` globals. Every function
takes a ``creds`` dict so the migration tool can test and migrate to a new
provider without mutating the running app's active configuration.

Supported providers:
  * ``jellyfin`` / ``emby`` — X-Emby-Token header API, identical shape
  * ``navidrome``           — Subsonic JSON API
  * ``lyrion``              — JSON-RPC (stub — raises NotImplementedError)
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


def _jellyfin_fetch_all(creds):
    url = f"{creds['url'].rstrip('/')}/Users/{creds['user_id']}/Items"
    params = {
        'IncludeItemTypes': 'Audio',
        'Recursive': 'true',
        'Fields': 'Path,ProductionYear,IndexNumber,ParentIndexNumber,AlbumArtist,Album',
    }
    r = requests.get(url, headers=_jellyfin_headers(creds), params=params, timeout=REQUESTS_TIMEOUT)
    r.raise_for_status()
    items = r.json().get('Items', []) or []
    return [_jellyfin_track(i) for i in items]


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
    try:
        tracks = _jellyfin_fetch_all(creds)
    except Exception as e:
        logger.warning("Jellyfin/Emby probe failed: %s", e)
        return {'ok': False, 'error': str(e), 'sample_count': 0,
                'path_format': 'none', 'warnings': []}
    sample = tracks[:_SAMPLE_LIMIT]
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
            '(Settings > Personal > Subsonic). Automatic path-based matching '
            'will not work well. Enable Report Real Path and re-test, or you '
            'will need to manually match most albums in Step 4.'
        )

    return {
        'ok':           True,
        'error':        None,
        'sample_count': len(sample),
        'path_format':  path_format,
        'warnings':     warnings,
    }


# ---------------------------------------------------------------------------
# Lyrion / MPD — stubs
# ---------------------------------------------------------------------------

def _lyrion_not_supported(*_args, **_kwargs):
    raise NotImplementedError(
        'Lyrion migration is not yet supported by the migration tool. '
        'As a workaround, migrate to Jellyfin or Navidrome first, then to Lyrion.'
    )


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
        'fetch_all_tracks': _lyrion_not_supported,
        'search_albums':    _lyrion_not_supported,
        'get_album_tracks': _lyrion_not_supported,
        'test_connection':  _lyrion_not_supported,
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
