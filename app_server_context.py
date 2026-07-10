# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared helpers for the optional per-request media-server selection.

Every existing API keeps working unchanged; callers may additionally pass a
``server`` id (query string or JSON body) to target a specific configured
server. When absent, the default server is used, so the historical behaviour is
preserved byte-for-byte.

Main Features:
* ``resolve_request_server_id`` reads the optional ``server`` parameter.
* ``create_instant_playlist_for_server`` translates canonical track ids to the
  target server's ids and creates the playlist there (identity for the default).
* Credential masking and a template-friendly server list for the UI.
"""

import logging

from flask import request

logger = logging.getLogger(__name__)

_SECRET_CRED_KEYS = ('token', 'password')
CRED_MASK = '__unchanged__'


def resolve_request_server_id(data=None):
    """Return the requested server id, or None for the default server.

    Raises ValueError if a server id is provided but is not in the registry.
    """
    from tasks.mediaserver import registry

    server_id = None
    if isinstance(data, dict):
        server_id = data.get('server') or data.get('server_id')
    if not server_id:
        server_id = request.args.get('server') or request.args.get('server_id')
    if not server_id:
        return None
    if registry.get_server(server_id) is None:
        raise ValueError(f"Unknown server '{server_id}'")
    return server_id


def is_default_server(server_id):
    from tasks.mediaserver import registry
    if not server_id:
        return True
    return server_id == registry.get_default_server_id()


def needs_translation(server_id):
    """True when the request's ids must be translated before hitting a server.

    Always for a secondary server; also for the default server once the catalogue
    id is the content fingerprint (item_id no longer equals the server's id).
    """
    import config
    if config.CATALOG_FINGERPRINT_AS_ID:
        return True
    return not is_default_server(server_id)


def create_instant_playlist_for_server(playlist_name, item_ids, server_id):
    """Create a playlist on ``server_id``, translating canonical ids first.

    Returns ``{'result', 'requested', 'mapped', 'skipped'}``. When no translation
    is needed (default server, catalogue id == server id) it is a straight
    passthrough.
    """
    from tasks import mediaserver
    from tasks.mediaserver import registry

    requested = len(item_ids)
    if not needs_translation(server_id):
        result = mediaserver.create_instant_playlist(playlist_name, item_ids)
        return {'result': result, 'requested': requested, 'mapped': requested, 'skipped': 0}

    translated = registry.translate_ids(item_ids, server_id)
    target_ids = [translated[i] for i in item_ids if i in translated]
    skipped = requested - len(target_ids)
    if not target_ids:
        raise ValueError("None of the selected tracks are available on the target server.")
    result = mediaserver.for_server(server_id).create_instant_playlist(playlist_name, target_ids)
    return {'result': result, 'requested': requested, 'mapped': len(target_ids), 'skipped': skipped}


def available_ids_for_server(item_ids, server_id):
    """Return the subset of item_ids that exist on ``server_id``.

    All ids for the default server (or an unset server); for a secondary server
    only the ids present in its track mapping, since a user may not have the same
    songs on every server.
    """
    if is_default_server(server_id):
        return list(item_ids)
    from tasks.mediaserver import registry
    mapping = registry.translate_ids(item_ids, server_id)
    return [i for i in item_ids if i in mapping]


def overfetch_limit(requested_n, multiplier=2):
    """How many rows to request from the index for this request.

    The default server has the whole catalogue, so ``requested_n`` is returned
    unchanged. When a specific secondary server is selected, return
    ``requested_n * multiplier`` so the caller over-fetches; after dropping the
    tracks that server does not have, the trimmed list still fills to the
    requested size.
    """
    if requested_n is None:
        return None
    try:
        server_id = resolve_request_server_id()
        if is_default_server(server_id):
            return requested_n
    except Exception:
        return requested_n
    return requested_n * multiplier


def scope_results(rows, requested_n=None, id_key='item_id'):
    """Drop rows not on the selected server, then trim to ``requested_n``.

    A no-op for the default server (returns the first ``requested_n`` rows, which
    is what the endpoint already fetched), so behaviour is unchanged there.
    """
    filtered = filter_rows_for_request_server(rows, id_key)
    if requested_n is not None and requested_n >= 0:
        return filtered[:requested_n]
    return filtered


def filter_rows_for_request_server(rows, id_key='item_id'):
    """Drop result rows whose track is not on the request's selected server.

    Returns ``rows`` unchanged for the default server. ``id_key`` is a dict key or
    a callable that extracts the canonical item_id from a row.
    """
    if not rows:
        return rows
    try:
        server_id = resolve_request_server_id()
        if is_default_server(server_id):
            return rows
    except Exception:
        return rows
    from tasks.mediaserver import registry

    def _get(row):
        if callable(id_key):
            return id_key(row)
        if isinstance(row, dict):
            return row.get(id_key)
        return None

    ids = [i for i in (_get(r) for r in rows) if i]
    mapping = registry.translate_ids(ids, server_id)
    return [r for r in rows if _get(r) in mapping]


def mask_creds(creds):
    """Return a copy of a creds dict with secret fields replaced by a sentinel."""
    masked = {}
    for key, value in (creds or {}).items():
        if key in _SECRET_CRED_KEYS and value:
            masked[key] = CRED_MASK
        else:
            masked[key] = value
    return masked


def merge_creds(existing, incoming):
    """Merge incoming creds over existing, preserving secrets left as the mask."""
    merged = dict(existing or {})
    for key, value in (incoming or {}).items():
        if key in _SECRET_CRED_KEYS and value == CRED_MASK:
            continue
        merged[key] = value
    return merged


def server_public_dict(server):
    """A registry server row with creds masked, for API/UI responses."""
    return {
        'server_id': server['server_id'],
        'name': server['name'],
        'server_type': server['server_type'],
        'creds': mask_creds(server['creds']),
        'music_libraries': server['music_libraries'],
        'is_default': server['is_default'],
        'enabled': server['enabled'],
    }


def servers_for_ui():
    """List of masked servers plus the default id, for templates and the API."""
    import config
    from tasks.mediaserver import registry

    try:
        servers = registry.list_servers()
    except Exception:
        logger.exception("Failed to list media servers for the UI")
        servers = []
    return {
        'servers': [server_public_dict(s) for s in servers],
        'default_id': next((s['server_id'] for s in servers if s['is_default']), None),
        'multi_server_enabled': config.MULTI_SERVER_ENABLED,
    }
