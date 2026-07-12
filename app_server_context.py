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
* ``resolve_input_item_id(s)`` canonicalize caller-supplied seed/track ids
  (provider ids in, canonical ids out) before they reach the shared indexes.
* ``create_instant_playlist_for_server`` translates canonical track ids to the
  target server's ids and creates the playlist there (identity for the default).
* Credential masking and a template-friendly server list for the UI.
"""

import logging
from contextlib import contextmanager

from flask import request

logger = logging.getLogger(__name__)

_SECRET_CRED_KEYS = ('token', 'password')
CRED_MASK = '__unchanged__'


def resolve_request_server_id(data=None):
    """Return the requested server id, or None for the default server.

    The ``server`` parameter accepts either the configured display NAME (the
    friendly value users see in the setup wizard) or the internal server id.
    Raises ValueError when the value matches no configured server.
    """
    from tasks.mediaserver import registry

    if data is None and request.method in ('POST', 'PUT', 'PATCH'):
        data = request.get_json(silent=True)
    requested = None
    if isinstance(data, dict):
        requested = data.get('server') or data.get('server_id')
    if not requested:
        requested = request.args.get('server') or request.args.get('server_id')
    if not requested:
        return None
    server = registry.get_server(requested) or registry.get_server_by_name(requested)
    if server is None:
        raise ValueError(f"Unknown server '{requested}'")
    if not server.get('enabled', True):
        raise ValueError(f"Server '{server['name']}' is disabled")
    return server['server_id']


def is_default_server(server_id):
    from tasks.mediaserver import registry
    if not server_id:
        return True
    return server_id == registry.get_default_server_id()


def resolve_input_item_ids(item_ids, data=None):
    """Canonicalize caller-supplied track ids for the request's selected server.

    The request-level face of ``registry.canonical_input_ids``: provider ids from
    the selected (or default) server become canonical catalogue ids before they
    reach the shared indexes; canonical or unknown ids pass through unchanged.
    """
    ids = [str(i) for i in (item_ids or []) if i]
    if not ids:
        return {}
    from tasks.mediaserver import registry

    server_id = resolve_request_server_id(data)
    try:
        return registry.canonical_input_ids(ids, server_id)
    except Exception:
        logger.exception("Request input id resolution failed; using ids as-is")
        return {i: i for i in ids}


def resolve_input_item_id(raw_id, data=None):
    """Single-id convenience over ``resolve_input_item_ids``."""
    if not raw_id:
        return raw_id
    return resolve_input_item_ids([raw_id], data).get(str(raw_id), str(raw_id))


def resolve_artist_identifier(identifier, data=None):
    """Turn a selected-server artist id into the shared artist name."""
    if not identifier:
        return identifier
    from tasks.mediaserver import registry

    server_id = resolve_request_server_id(data) or registry.get_default_server_id()
    return registry.artist_names_for_ids([identifier], server_id).get(
        str(identifier), identifier
    )


def scope_artist_results(rows, requested_n=None):
    """Keep artists represented on the selected server and expose its IDs/counts."""
    if not rows:
        return rows
    from tasks.mediaserver import registry

    server_id = resolve_request_server_id() or registry.get_default_server_id()
    names = [row.get('artist') for row in rows if row.get('artist')]
    counts = registry.artist_track_counts(names, server_id)
    ids = registry.artist_ids_for_names(names, server_id)
    scoped = []
    for source in rows:
        name = source.get('artist')
        if not counts.get(name):
            continue
        row = dict(source)
        row['artist_id'] = ids.get(name)
        row['track_count'] = counts[name]
        scoped.append(row)
    return scoped[:requested_n] if requested_n is not None else scoped


@contextmanager
def use_request_server(data=None):
    """Bind provider calls to the request's selected server for one block."""
    from tasks.mediaserver import context, registry

    server_id = resolve_request_server_id(data)
    server_ctx = registry.context_for(server_id) if server_id else None
    with context.use_server(server_ctx):
        yield server_id


def create_instant_playlist_for_server(playlist_name, item_ids, server_id, user_creds=None):
    """Create a playlist on ``server_id`` (default when None).

    The canonical ids are passed through UNTRANSLATED: the mediaserver dispatcher
    is the single place that translates them to the target server's real ids
    (translating here too would translate twice and send wrong ids). The mapping
    is still consulted first to report how many tracks the server has and to
    fail clearly when it has none. Returns ``{'result', 'requested', 'mapped',
    'skipped'}``.
    """
    from tasks import mediaserver
    from tasks.mediaserver import registry

    requested = len(item_ids)
    available = registry.translate_ids(item_ids, server_id)
    mapped = sum(1 for i in item_ids if str(i) in available)
    skipped = requested - mapped
    if not mapped:
        raise ValueError("None of the selected tracks are available on the target server.")
    result = mediaserver.for_server(server_id).create_instant_playlist(
        playlist_name, item_ids, user_creds
    )
    return {'result': result, 'requested': requested, 'mapped': mapped, 'skipped': skipped}


def scope_results(rows, requested_n=None, id_key='item_id'):
    """Drop rows not on the selected server, then trim to ``requested_n``.

    Filtering applies to the default too because any server may be a subset.
    """
    filtered = filter_rows_for_request_server(rows, id_key)
    if requested_n is not None and requested_n >= 0:
        return filtered[:requested_n]
    return filtered


def filter_rows_for_request_server(rows, id_key='item_id'):
    """Drop result rows whose track is not on the request's selected server.

    ``id_key`` is a dict key or a callable that extracts the canonical item_id.
    Raises ValueError for an unknown or disabled ``server`` parameter so the
    endpoint can answer 400; a registry failure fails open (rows unchanged).
    Single-server installs with no explicit selection skip translation entirely.
    """
    if not rows:
        return rows
    server_id = resolve_request_server_id()
    from tasks.mediaserver import registry

    if server_id is None and not registry.has_secondary_servers():
        return rows

    def _get(row):
        if callable(id_key):
            return id_key(row)
        if isinstance(row, dict):
            return row.get(id_key)
        return None

    ids = [i for i in (_get(r) for r in rows) if i]
    try:
        mapping = registry.translate_ids(ids, server_id)
    except Exception:
        logger.exception("Server availability filtering failed; returning rows unfiltered")
        return rows
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
    from tasks.mediaserver import registry

    try:
        servers = registry.list_servers()
    except Exception:
        logger.exception("Failed to list media servers for the UI")
        servers = []
    return {
        'servers': [server_public_dict(s) for s in servers],
        'default_id': next((s['server_id'] for s in servers if s['is_default']), None),
        'multi_server_enabled': True,
    }
