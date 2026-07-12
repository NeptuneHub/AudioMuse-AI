# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Registry of configured media servers and cross-server track-id mapping.

Persists every configured media server (the ``music_servers`` table) plus the
per-track mapping from a canonical library ``item_id`` to that track's id on
each server (the ``track_server_map`` table). The default server's
credentials are mirrored from the global ``config`` (still edited by the setup
wizard), so single-server installs keep one source of truth and behave exactly
as before; secondary servers store their own credentials in the registry.

Main Features:
* CRUD over the server registry with a single enforced default server.
* Builds the default server's creds from config and keeps its row in sync.
* Resolves a normalized server context by id and translates canonical item_ids
  to a target server's provider track ids (legacy raw ids may use identity on
  the default server; canonical ids always require a mapping).
* ``canonical_input_ids`` is the single input-side resolver turning
  caller-supplied provider ids into canonical ids (fail-open pass-through).
"""

import logging
import uuid

from psycopg2.extras import DictCursor, Json, execute_values

import config
from database import get_db

logger = logging.getLogger(__name__)

_COLUMNS = (
    "server_id", "name", "server_type", "creds",
    "music_libraries", "is_default", "enabled",
)


def creds_from_config(server_type):
    """Build a ``user_creds`` dict for ``server_type`` from the config globals."""
    creds = {}
    for field in config.MEDIASERVER_FIELDS_BY_TYPE.get(server_type, []):
        key = config.MEDIASERVER_CRED_KEY_BY_FIELD.get(field)
        if key:
            creds[key] = getattr(config, field, "") or ""
    return creds


def normalize_row(row):
    """Turn a DB row (dict-like) into the context dict provider backends consume."""
    if row is None:
        return None
    return {
        "server_id": row["server_id"],
        "name": row["name"],
        "server_type": row["server_type"],
        "creds": dict(row["creds"] or {}),
        "music_libraries": row["music_libraries"] or "",
        "is_default": bool(row["is_default"]),
        "enabled": bool(row["enabled"]),
    }


def _rows(db, where="", params=()):
    cur = db.cursor(cursor_factory=DictCursor)
    try:
        cur.execute(
            "SELECT server_id, name, server_type, creds, music_libraries, is_default, enabled "
            "FROM music_servers " + where,
            params,
        )
        return [normalize_row(r) for r in cur.fetchall()]
    finally:
        cur.close()


def list_servers(conn=None):
    db = conn or get_db()
    return _rows(db, "ORDER BY is_default DESC, name ASC")


def get_server(server_id, conn=None):
    if not server_id:
        return None
    db = conn or get_db()
    rows = _rows(db, "WHERE server_id = %s", (server_id,))
    return rows[0] if rows else None


def get_server_by_name(name, conn=None):
    """Find a server by its user-facing display name (case-insensitive)."""
    if not name:
        return None
    db = conn or get_db()
    rows = _rows(db, "WHERE lower(name) = lower(%s) ORDER BY name ASC LIMIT 1", (name,))
    return rows[0] if rows else None


def get_default_server(conn=None):
    db = conn or get_db()
    rows = _rows(db, "WHERE is_default ORDER BY name ASC LIMIT 1")
    return rows[0] if rows else None


def get_default_server_id(conn=None):
    server = get_default_server(conn)
    return server["server_id"] if server else None


def context_for(server_id, conn=None):
    """Return the context dict for ``server_id``, or None to mean 'use config default'.

    Returning None for the default server keeps its code path byte-identical to
    the historical single-server behaviour (provider backends fall back to config).
    """
    db = conn or get_db()
    default = get_default_server(db)
    default_id = default["server_id"] if default else None
    if not server_id or server_id == default_id:
        return None
    return get_server(server_id, db)


def _clear_default(db):
    cur = db.cursor()
    try:
        cur.execute("UPDATE music_servers SET is_default = FALSE, updated_at = now() WHERE is_default")
    finally:
        cur.close()


def add_server(name, server_type, creds, music_libraries="", enabled=True, make_default=False, conn=None):
    db = conn or get_db()
    server_id = uuid.uuid4().hex
    cur = db.cursor()
    try:
        if make_default:
            _clear_default(db)
        cur.execute(
            "INSERT INTO music_servers "
            "(server_id, name, server_type, creds, music_libraries, is_default, enabled) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (server_id, name, server_type, Json(dict(creds or {})),
             music_libraries or "", bool(make_default), bool(enabled)),
        )
        db.commit()
        return server_id
    finally:
        cur.close()


def update_server(server_id, name=None, server_type=None, creds=None,
                  music_libraries=None, enabled=None, conn=None):
    db = conn or get_db()
    sets, params = [], []
    if name is not None:
        sets.append("name = %s")
        params.append(name)
    if server_type is not None:
        sets.append("server_type = %s")
        params.append(server_type)
    if creds is not None:
        sets.append("creds = %s")
        params.append(Json(dict(creds)))
    if music_libraries is not None:
        sets.append("music_libraries = %s")
        params.append(music_libraries)
    if enabled is not None:
        sets.append("enabled = %s")
        params.append(bool(enabled))
    if not sets:
        return
    sets.append("updated_at = now()")
    params.append(server_id)
    cur = db.cursor()
    try:
        cur.execute("UPDATE music_servers SET " + ", ".join(sets) + " WHERE server_id = %s", params)
        db.commit()
    finally:
        cur.close()


def set_default(server_id, conn=None):
    db = conn or get_db()
    cur = db.cursor()
    try:
        _clear_default(db)
        cur.execute(
            "UPDATE music_servers SET is_default = TRUE, enabled = TRUE, updated_at = now() WHERE server_id = %s",
            (server_id,),
        )
        db.commit()
    finally:
        cur.close()


def delete_server(server_id, conn=None):
    """Delete a secondary server. Refuses to delete the default server."""
    db = conn or get_db()
    server = get_server(server_id, db)
    if server is None:
        return False
    if server["is_default"]:
        raise ValueError("Cannot delete the default server; set another server as default first.")
    cur = db.cursor()
    try:
        cur.execute("DELETE FROM music_servers WHERE server_id = %s", (server_id,))
        db.commit()
        try:
            from tasks.paged_ivf import invalidate_availability_cache
            invalidate_availability_cache(server_id)
        except Exception:
            logger.debug("Availability-cache invalidation failed", exc_info=True)
        return True
    finally:
        cur.close()


def sync_default_from_config(conn=None):
    """Overwrite the default server's type/creds/library filter from config.

    Called after the setup wizard saves media-server settings so the default
    registry row always mirrors the global config the wizard edits. Creates the
    default row when the registry is still empty.
    """
    db = conn or get_db()
    server_type = config.MEDIASERVER_TYPE
    creds = creds_from_config(server_type)
    libraries = config.MUSIC_LIBRARIES or ""
    default = get_default_server(db)
    if default is None:
        cur = db.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM music_servers")
            has_any = cur.fetchone()[0] > 0
        finally:
            cur.close()
        add_server(
            name=_default_server_name(server_type),
            server_type=server_type,
            creds=creds,
            music_libraries=libraries,
            enabled=True,
            make_default=not has_any,
            conn=db,
        )
        return
    update_server(
        default["server_id"],
        server_type=server_type,
        creds=creds,
        music_libraries=libraries,
        conn=db,
    )


def _default_server_name(server_type):
    return (server_type or "media server").capitalize()


def translate_ids(item_ids, server_id=None, conn=None):
    """Map canonical library item_ids to their ids on ``server_id``.

    Returns ``{item_id: provider_track_id}`` containing only the ids that exist
    on the target server. For the default server (or an unset server_id) the
    mapping is the identity, since ``score.item_id`` already holds its ids.
    """
    ids = [str(i) for i in dict.fromkeys(item_ids) if i]
    if not ids:
        return {}
    db = conn or get_db()
    default = get_default_server(db)
    default_id = default["server_id"] if default else None
    is_default = (not server_id) or server_id == default_id
    target = server_id or default_id
    if target is None:
        from tasks.audio_fingerprint import is_fingerprint_id
        return {i: i for i in ids if not is_fingerprint_id(i)}
    cur = db.cursor()
    try:
        cur.execute(
            "SELECT item_id, provider_track_id FROM track_server_map "
            "WHERE server_id = %s AND item_id = ANY(%s)",
            (target, ids),
        )
        mapped = {r[0]: r[1] for r in cur.fetchall()}
    finally:
        cur.close()
    if is_default:
        from tasks.audio_fingerprint import is_fingerprint_id
        return {
            i: mapped.get(i, i)
            for i in ids
            if i in mapped or not is_fingerprint_id(i)
        }
    return mapped


def reverse_translate_ids(provider_ids, server_id=None, conn=None):
    """Map a server's real track ids back to the canonical catalogue item_ids.

    The inverse of ``translate_ids``: returns ``{provider_id: item_id}`` for the
    ids known on ``server_id`` (default server when None). On the default server
    unknown ids fall back to themselves, since legacy rows still use the
    provider id as their catalogue id.
    """
    ids = [str(i) for i in dict.fromkeys(provider_ids) if i]
    if not ids:
        return {}
    db = conn or get_db()
    default = get_default_server(db)
    default_id = default["server_id"] if default else None
    is_default = (not server_id) or server_id == default_id
    target = server_id or default_id
    if target is None:
        return {i: i for i in ids}
    cur = db.cursor()
    try:
        cur.execute(
            "SELECT provider_track_id, item_id FROM track_server_map "
            "WHERE server_id = %s AND provider_track_id = ANY(%s)",
            (target, ids),
        )
        mapped = {r[0]: r[1] for r in cur.fetchall()}
    finally:
        cur.close()
    if is_default:
        return {i: mapped.get(i, i) for i in ids}
    return mapped


def canonical_input_ids(item_ids, server_id=None, conn=None):
    """Resolve caller-supplied track ids to canonical catalogue ids.

    The single input-side resolver: a provider id known on ``server_id`` (active
    or default when None) maps to its canonical id; canonical or unknown ids pass
    through unchanged, so every feature accepts either form. Never raises - a
    registry failure falls back to the ids as given.
    """
    ids = [str(i) for i in item_ids if i]
    if not ids:
        return {}
    try:
        mapped = reverse_translate_ids(ids, server_id, conn=conn)
    except Exception:
        logger.exception("Input id resolution failed; using ids as-is")
        return {i: i for i in ids}
    return {i: mapped.get(i, i) for i in ids}


def upsert_track_maps(server_id, mapping, conn=None):
    """Bulk-upsert ``{item_id: (provider_track_id, match_tier)}`` for a server."""
    if not mapping:
        return 0
    db = conn or get_db()
    rows = []
    for item_id, value in mapping.items():
        if isinstance(value, (tuple, list)):
            provider_track_id, match_tier = value[0], (value[1] if len(value) > 1 else None)
        else:
            provider_track_id, match_tier = value, None
        if provider_track_id is None or provider_track_id == '':
            continue
        rows.append((str(item_id), server_id, str(provider_track_id), match_tier))
    if not rows:
        return 0
    cur = db.cursor()
    try:
        execute_values(
            cur,
            "INSERT INTO track_server_map "
            "(item_id, server_id, provider_track_id, match_tier, updated_at) VALUES %s "
            "ON CONFLICT (item_id, server_id) DO UPDATE SET "
            "provider_track_id = EXCLUDED.provider_track_id, "
            "match_tier = EXCLUDED.match_tier, updated_at = now()",
            rows,
            template="(%s, %s, %s, %s, now())",
            page_size=5000,
        )
        db.commit()
        try:
            from tasks.paged_ivf import invalidate_availability_cache
            invalidate_availability_cache(server_id)
        except Exception:
            logger.debug("Availability-cache invalidation failed", exc_info=True)
        return len(rows)
    finally:
        cur.close()


def mapped_count(server_id, conn=None):
    """Number of canonical tracks that have a mapping on ``server_id``."""
    db = conn or get_db()
    cur = db.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM track_server_map WHERE server_id = %s", (server_id,))
        return cur.fetchone()[0]
    finally:
        cur.close()
