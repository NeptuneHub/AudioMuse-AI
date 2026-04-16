"""Provider migration probe wrappers for target media servers.

This module deliberately does NOT read ``config.py`` globals. Every function
takes a ``creds`` dict so the migration tool can test and migrate to a new
provider without mutating the running app's active configuration.

Supported providers:
  * ``jellyfin`` / ``emby`` — X-Emby-Token header API, identical shape
  * ``navidrome``           — Subsonic JSON API
  * ``lyrion``              — Logitech Media Server JSON-RPC (``/jsonrpc.js``)
 
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
from tasks.mediaserver_helper import detect_path_format as _detect_path_format
from tasks import mediaserver


def _normalize_track(item):
    """Convert provider-specific song dicts into the unified track shape."""
    if item is None:
        return {
            'id': None,
            'path': None,
            'title': None,
            'artist': None,
            'album_artist': None,
            'album': None,
            'year': None,
            'track_number': None,
            'disc_number': None,
        }

    def _try(*keys):
        for key in keys:
            value = item.get(key)
            if value is not None:
                return value
        return None

    year = _try('Year', 'year')
    if isinstance(year, str):
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = None

    return {
        'id': _try('Id', 'id', 'track_id'),
        'path': _try('Path', 'path', 'url'),
        'title': _try('Name', 'name', 'title'),
        'artist': _try('AlbumArtist', 'artist', 'author'),
        'album_artist': _try('OriginalAlbumArtist', 'albumArtist', 'AlbumArtist'),
        'album': _try('Album', 'album'),
        'year': year,
        'track_number': _try('IndexNumber', 'track_number', 'track'),
        'disc_number': _try('ParentIndexNumber', 'disc_number', 'disc'),
    }


_SUPPORTED_PROVIDERS = {'jellyfin', 'emby', 'navidrome', 'lyrion'}


def _normalize_provider_type(provider_type):
    t = (provider_type or '').lower()
    if t not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Provider type '{provider_type}' not supported by migration probe. "
            f"Supported: {sorted(_SUPPORTED_PROVIDERS)}"
        )
    return t


def fetch_all_tracks(provider_type, creds):
    t = _normalize_provider_type(provider_type)
    items = mediaserver.get_all_songs(user_creds=creds, provider_type=t)
    return [_normalize_track(item) for item in items or []]


def search_albums(provider_type, creds, query):
    t = _normalize_provider_type(provider_type)
    return mediaserver.search_albums(query, user_creds=creds, provider_type=t)


def get_album_tracks(provider_type, creds, album_id):
    t = _normalize_provider_type(provider_type)
    items = mediaserver.get_tracks_from_album(album_id, user_creds=creds, provider_type=t)
    return [_normalize_track(item) for item in items or []]


def test_connection(provider_type, creds):
    t = _normalize_provider_type(provider_type)
    return mediaserver.test_connection(user_creds=creds, provider_type=t)
