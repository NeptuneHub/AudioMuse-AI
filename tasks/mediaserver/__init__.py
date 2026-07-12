# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Central media-server dispatcher and public API for AudioMuse-AI.

Single abstraction over every supported external media server: higher layers
call these dispatcher functions, which delegate to the active backend selected
by config.MEDIASERVER_TYPE.

Main Features:
* Lazily imports and dispatches to the active provider backend (jellyfin, emby,
  navidrome, lyrion, plex), so importing this package does not load inactive backends.
* Centralizes the provider-agnostic public API; shared HTTP and metadata parsing
  live in http.py and helper.py.
"""

import logging
import os
from importlib import import_module

import config

from . import context
from .context import use_server

logger = logging.getLogger(__name__)

_PROVIDER_NAMES = ('jellyfin', 'navidrome', 'lyrion', 'emby', 'plex')
_warned_unsupported = set()

_PLAYLIST_NAME_REQUIRED = "Playlist name is required."
_TRACK_IDS_REQUIRED = "Track IDs are required."

_PUBLIC_SERVER_API = (
    'resolve_emby_jellyfin_user', 'delete_playlists_by_suffix', 'delete_automatic_playlists',
    'get_recent_albums', 'get_tracks_from_album', 'download_track', 'get_all_songs',
    'list_libraries', 'search_albums', 'test_connection', 'get_playlist_by_name',
    'get_all_playlists', 'get_playlist_track_ids', 'create_playlist', 'create_instant_playlist',
    'create_or_replace_playlist', 'get_top_played_songs', 'get_last_played_time', 'get_lyrics',
)


def _provider(provider_type=None):
    name = provider_type or context.active_type(config.MEDIASERVER_TYPE)
    if name not in _PROVIDER_NAMES:
        if name not in _warned_unsupported:
            _warned_unsupported.add(name)
            logger.warning(
                "Unsupported MEDIASERVER_TYPE %r (supported: %s); media-server operations are no-ops.",
                name,
                ', '.join(_PROVIDER_NAMES),
            )
        return None
    return import_module('.' + name, __name__)


def resolve_emby_jellyfin_user(identifier, token):
    if context.active_type(config.MEDIASERVER_TYPE) in ('jellyfin', 'emby'):
        return _provider().resolve_user(identifier, token)
    return []


def _delete_matching_playlists(playlists_to_check, delete_function, suffix):
    deleted_count = 0
    for p in playlists_to_check:
        playlist_id = p.get('Id') or p.get('id')
        try:
            if p.get('Name', '').endswith(suffix) and delete_function(playlist_id):
                deleted_count += 1
        except Exception:
            logger.exception(
                f"Failed to delete playlist {playlist_id}; continuing with the remaining playlists."
            )
    return deleted_count


def delete_playlists_by_suffix(suffix):
    logger.info(f"Starting deletion of all '{suffix}' playlists.")
    deleted_count = 0

    provider = _provider()
    if provider is not None:
        deleted_count = _delete_matching_playlists(
            provider.get_all_playlists(), provider.delete_playlist, suffix
        )

    logger.info(f"Finished deletion. Deleted {deleted_count} playlists.")


def delete_automatic_playlists():
    delete_playlists_by_suffix('_automatic')


def get_recent_albums(limit):
    provider = _provider()
    if provider is None:
        return []
    return provider.get_recent_albums(limit)


def get_tracks_from_album(album_id, user_creds=None, provider_type=None):
    user_creds = context.active_creds(user_creds)
    provider = _provider(provider_type)
    if provider is None:
        return []
    return provider.get_tracks_from_album(album_id, user_creds=user_creds)


def download_track(temp_dir, item):
    provider = _provider()
    downloaded_path = provider.download_track(temp_dir, item) if provider is not None else None

    if not downloaded_path:
        return None

    if downloaded_path.endswith('.tmp'):
        try:
            if not os.path.exists(downloaded_path):
                logger.warning(f"Downloaded file does not exist: {downloaded_path}")
                return downloaded_path

            detected_ext = _detect_audio_format(downloaded_path)
            if detected_ext and detected_ext != '.tmp':
                new_path = downloaded_path[: -len('.tmp')] + detected_ext
                if os.path.exists(new_path):
                    logger.warning(f"Target file already exists, keeping .tmp: {new_path}")
                    return downloaded_path
                os.rename(downloaded_path, new_path)
                logger.info(
                    f"Detected format and renamed: {os.path.basename(downloaded_path)} -> {os.path.basename(new_path)}"
                )
                return new_path
        except Exception as e:
            logger.debug(
                f"Format detection failed for {os.path.basename(downloaded_path)}, keeping .tmp: {e}"
            )

    return downloaded_path


def _detect_audio_format(filepath):
    try:
        with open(filepath, 'rb') as f:
            header = f.read(12)

            if len(header) < 4:
                return '.tmp'

            if header[:4] == b'fLaC':
                return '.flac'

            if header[:3] == b'ID3' or (
                len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0
            ):
                return '.mp3'

            if header[:4] == b'OggS':
                return '.ogg'

            if header[:4] == b'RIFF' and len(header) >= 12 and header[8:12] == b'WAVE':
                return '.wav'

            if len(header) >= 8 and header[4:8] == b'ftyp':
                return '.m4a'

            if header[:4] == b'\x30\x26\xb2\x75':
                return '.wma'

            logger.debug(f"Unknown audio format, header: {header[:4].hex()}")
            return '.tmp'

    except Exception as e:
        logger.debug(f"Error detecting audio format: {e}")
        return '.tmp'


def get_all_songs(user_creds=None, provider_type=None, apply_filter=True):
    user_creds = context.active_creds(user_creds)
    provider_type = provider_type or context.active_type(config.MEDIASERVER_TYPE)
    provider = _provider(provider_type)
    if provider is None:
        return []
    if provider_type == 'navidrome':
        return provider.get_all_songs(user_creds=user_creds, apply_filter=apply_filter)
    return provider.get_all_songs(user_creds=user_creds)


def list_libraries(user_creds=None, provider_type=None):
    user_creds = context.active_creds(user_creds)
    provider = _provider(provider_type)
    if provider is None:
        return {'libraries': [], 'unsupported': True}
    return {'libraries': provider.list_libraries(user_creds=user_creds), 'unsupported': False}


def search_albums(query, user_creds=None, provider_type=None):
    user_creds = context.active_creds(user_creds)
    provider = _provider(provider_type)
    if provider is None:
        return []
    return provider.search_albums(query, user_creds=user_creds)


def test_connection(user_creds=None, provider_type=None):
    user_creds = context.active_creds(user_creds)
    provider_type = provider_type or context.active_type(config.MEDIASERVER_TYPE)
    provider = _provider(provider_type)
    if provider is None:
        return {
            'ok': False,
            'error': f"Provider '{provider_type}' not supported",
            'sample_count': 0,
            'path_format': 'none',
            'warnings': [],
        }
    return provider.test_connection(user_creds=user_creds)


def get_playlist_by_name(playlist_name):
    if not playlist_name:
        raise ValueError(_PLAYLIST_NAME_REQUIRED)
    provider = _provider()
    if provider is None:
        return None
    return provider.get_playlist_by_name(playlist_name)


def get_all_playlists():
    provider = _provider()
    if provider is None:
        return []
    return provider.get_all_playlists()


def get_playlist_track_ids(playlist_id, user_creds=None):
    if not playlist_id:
        return []
    user_creds = context.active_creds(user_creds)
    provider = _provider()
    if provider is None:
        return []
    if context.active_type(config.MEDIASERVER_TYPE) == 'lyrion':
        return provider.get_playlist_track_ids(playlist_id)
    return provider.get_playlist_track_ids(playlist_id, user_creds=user_creds)


def _to_server_ids(item_ids):
    """Translate canonical catalogue ids to the active (or default) server's
    real track ids, dropping any the server does not have. This is the SINGLE
    translation point for playlist creation: callers pass canonical ids and must
    never pre-translate. Legacy provider ids map to themselves on the default
    server, so mixed catalogues pass through unchanged."""
    from .registry import translate_ids

    server_id = context.active_server_id()
    try:
        mapping = translate_ids(item_ids, server_id)
    except Exception:
        try:
            from database import connect_raw
            raw = connect_raw()
            try:
                mapping = translate_ids(item_ids, server_id, conn=raw)
            finally:
                raw.close()
        except Exception:
            logger.exception("Playlist id translation failed; sending ids unchanged")
            return list(item_ids)
    translated = [mapping[str(i)] for i in item_ids if str(i) in mapping]
    if item_ids and not translated:
        logger.warning(
            "Playlist id translation dropped all %d input ids for server %s; "
            "the provider call becomes a no-op",
            len(item_ids), server_id or 'default',
        )
    return translated


def create_playlist(base_name, item_ids):
    if not base_name:
        raise ValueError(_PLAYLIST_NAME_REQUIRED)
    if not item_ids:
        raise ValueError(_TRACK_IDS_REQUIRED)
    item_ids = _to_server_ids(item_ids)
    provider = _provider()
    if provider is not None and item_ids:
        provider.create_playlist(base_name, item_ids)


def create_instant_playlist(playlist_name, item_ids, user_creds=None):
    if not playlist_name:
        raise ValueError(_PLAYLIST_NAME_REQUIRED)
    if not item_ids:
        raise ValueError(_TRACK_IDS_REQUIRED)

    user_creds = context.active_creds(user_creds)
    item_ids = _to_server_ids(item_ids)
    if not item_ids:
        return None
    provider = _provider()
    if provider is None:
        return None
    if context.active_type(config.MEDIASERVER_TYPE) == 'lyrion':
        return provider.create_instant_playlist(playlist_name, item_ids)
    return provider.create_instant_playlist(playlist_name, item_ids, user_creds)


def create_or_replace_playlist(playlist_name, item_ids, user_creds=None):
    if not playlist_name:
        raise ValueError(_PLAYLIST_NAME_REQUIRED)
    if not item_ids:
        raise ValueError(_TRACK_IDS_REQUIRED)

    user_creds = context.active_creds(user_creds)
    item_ids = _to_server_ids(item_ids)
    if not item_ids:
        return None
    provider = _provider()
    if provider is None:
        raise NotImplementedError(
            f"create_or_replace_playlist not supported for MEDIASERVER_TYPE={context.active_type(config.MEDIASERVER_TYPE)!r}"
        )
    return provider.create_or_replace_playlist(playlist_name, item_ids, user_creds)


def get_top_played_songs(limit, user_creds=None):
    user_creds = context.active_creds(user_creds)
    provider = _provider()
    if provider is None:
        return []
    if context.active_type(config.MEDIASERVER_TYPE) == 'lyrion':
        return provider.get_top_played_songs(limit)
    return provider.get_top_played_songs(limit, user_creds)


def get_last_played_time(item_id, user_creds=None):
    user_creds = context.active_creds(user_creds)
    provider = _provider()
    if provider is None:
        return None
    if context.active_type(config.MEDIASERVER_TYPE) == 'lyrion':
        return provider.get_last_played_time(item_id)
    return provider.get_last_played_time(item_id, user_creds)


def get_lyrics(track_id: str, timeout: float = 2.5):
    provider = _provider()
    if provider is None:
        return None
    return provider.get_lyrics(track_id, timeout=timeout)


class BoundServer:
    """A media-server dispatcher bound to one registry server via the active context.

    Every public dispatcher function is exposed as a method that runs inside the
    bound server's context, so ``for_server(sid).get_all_songs()`` targets that
    server while the module-level functions keep targeting the config default.
    """

    def __init__(self, server_context, server_id=None):
        self._ctx = server_context
        self.server_id = server_id

    def __getattr__(self, name):
        if name in _PUBLIC_SERVER_API:
            fn = globals()[name]
            ctx = self._ctx

            def _bound(*args, **kwargs):
                with use_server(ctx):
                    return fn(*args, **kwargs)

            return _bound
        raise AttributeError(name)


def for_server(server_id, conn=None):
    """Return a BoundServer for ``server_id`` (config default when it is the default/None).

    Pass ``conn`` (a raw psycopg2 connection) when calling from a worker where no
    Flask application context exists; the registry lookup uses it instead of the
    request-scoped connection.
    """
    from .registry import context_for
    return BoundServer(context_for(server_id, conn), server_id)
