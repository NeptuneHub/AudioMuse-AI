# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Centralized per-server availability mask shared by the similarity indexes.

Every disk-paged index stores ONE union of all servers' canonical ids; a query
scoped to a server filters candidates through an availability mask built from
track_server_map. This module owns the scope resolution and the mask builder so
every index reuses the same logic instead of reimplementing it.

Main Features:
* active_availability_scope resolves the request/job server, failing closed to
  a sentinel scope on an unknown server and open to the union on infra errors
* available_item_ids returns the set of ids a server may see (the default
  server also keeps every legacy, non-fingerprint id)
* build_availability_mask returns the same set as a bool array aligned to an
  ordered id list, for indexes that address candidates by position
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def active_availability_scope():
    """Return the current request/job server, or None for union/background scope.

    An unknown or disabled requested server maps to a fail-closed sentinel scope;
    any other resolution error fails open to None (union scope).
    """
    try:
        from tasks.mediaserver import context

        active = context.active_server_id()
        if active:
            return str(active)
    except Exception:
        pass
    try:
        from flask import has_request_context

        if not has_request_context():
            return None
        from app_server_context import resolve_request_server_id
        from tasks.mediaserver import registry

        requested = resolve_request_server_id()
        return str(requested or registry.get_default_server_id() or '') or None
    except ValueError:
        return '__invalid_server__'
    except Exception:
        logger.exception("Could not resolve request availability scope")
        return None


def _fetch_available(server_id, item_ids, conn_factory):
    conn = conn_factory()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT is_default, updated_at FROM music_servers WHERE server_id = %s",
            (server_id,),
        )
        row = cur.fetchone()
        is_default = bool(row[0]) if row else False
        cur.execute(
            "SELECT item_id FROM track_server_map WHERE server_id = %s",
            (server_id,),
        )
        available = {str(r[0]) for r in cur.fetchall()}
    if is_default:
        from tasks.simhash import is_fingerprint_id

        available.update(i for i in item_ids if not is_fingerprint_id(i))
    return available


def available_item_ids(server_id, item_ids, conn_factory):
    """frozenset of item_ids visible on ``server_id``, or None for union scope.

    The default server keeps its mapped rows plus every legacy (non-fingerprint)
    id; a secondary server keeps exactly its track_server_map rows.
    """
    if server_id is None:
        return None
    return frozenset(_fetch_available(server_id, list(item_ids), conn_factory))


def build_availability_mask(server_id, item_ids, conn_factory):
    """bool ndarray aligned to ``item_ids``: True where the id is visible.

    Returns None for a None server (union scope, no filtering).
    """
    if server_id is None:
        return None
    ids = list(item_ids)
    available = _fetch_available(server_id, ids, conn_factory)
    return np.fromiter((i in available for i in ids), dtype=np.bool_, count=len(ids))
