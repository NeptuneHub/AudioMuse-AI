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
* invalidate_availability_caches drops the cached masks of every index that
  keeps one, so a mapping change reaches all of them from one call
* AvailabilityCache is the per-server, per-index-build memo with a 30 s life
  that every index keeps in front of the mask builder, and
  single_server_mask_unneeded the shared rule that skips the mask altogether
  for a lone default server over legacy ids
"""

import importlib
import logging
import threading
import time

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


def single_server_mask_unneeded(server_id, has_canonical_ids):
    try:
        from tasks.mediaserver import registry

        return bool(
            str(server_id) == str(registry.get_default_server_id() or '')
            and not registry.has_secondary_servers()
            and not has_canonical_ids
        )
    except Exception:
        logger.debug("Single-server availability fast path failed.", exc_info=True)
        return False


class AvailabilityCache:
    def __init__(self, ttl_seconds=30.0):
        self._ttl = float(ttl_seconds)
        self._lock = threading.Lock()
        self._entries = {}

    def get(self, server_id, scope, build):
        key = (str(server_id), scope)
        now = time.monotonic()
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None and now - cached[0] < self._ttl:
                return cached[1]
        value = build()
        with self._lock:
            for stale in [k for k, v in self._entries.items() if now - v[0] >= self._ttl]:
                self._entries.pop(stale, None)
            self._entries[key] = (now, value)
        return value

    def invalidate(self, server_id=None):
        with self._lock:
            if server_id is None:
                self._entries.clear()
                return
            for key in [key for key in self._entries if key[0] == str(server_id)]:
                self._entries.pop(key, None)

    def __len__(self):
        with self._lock:
            return len(self._entries)


_MASK_OWNERS = ('tasks.paged_ivf', 'tasks.hyperbolic_index', 'tasks.neural_fingerprint_index')


def invalidate_availability_caches(server_id=None):
    for module_name in _MASK_OWNERS:
        try:
            importlib.import_module(module_name).invalidate_availability_cache(server_id)
        except Exception:
            logger.debug("Availability-cache invalidation failed for %s", module_name, exc_info=True)
