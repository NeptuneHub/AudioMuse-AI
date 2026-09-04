# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Registry of configured media servers and cross-server track-id mapping.

Persists every configured media server (the music_servers table) plus the
per-track mapping from a canonical library item_id to that track's id on each
server (the track_server_map table). The registry is the ONLY persistent home of
media-server settings, the default server included: config module globals are a
read-only projection of the default row, the setup wizard writes here, and
init_db migrates legacy app_config rows in once and deletes them.

Main Features:
* CRUD over the server registry with a single enforced default server.
* Resolves a normalized server context by id and translates canonical item_ids
  to a target server's provider track ids.
* canonical_input_ids is the single input-side resolver (fail-open pass-through).
* servers_for_scope / has_secondary_servers resolve which servers a
  multi-server operation should touch, and for_each_server runs one callable
  once per server in that scope with each server's context bound: a per-server
  failure is logged and the loop continues, but a scope where every server
  failed raises. It returns the single result for a one-server scope, else the
  per-server results in scope order.
* Self-heals the default server: context_for compares the config projection
  against the default row and binds the row when they disagree, so a stale boot
  never talks to the wrong machine.
* upsert_track_maps hands each mapping the Chromaprint already stored for its
  canonical track, in the SAME transaction that writes the mapping. Matching a
  track and knowing its fingerprint are one step, so a sweep always carries the
  fingerprint with the mapping. The inherit rides a SAVEPOINT and can only ever
  be skipped: the mapping write is the job and always stands.
"""

import logging
import threading
import time
import uuid

import psycopg2
from psycopg2.extras import DictCursor, Json, execute_values

import config
from database import (
    get_db,
    inherit_chromaprints_from_staged_maps,
    missing_required_creds,
)
from sanitization import sanitize_string_for_db, sanitize_string_for_db_loud

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_TTL = 10.0
_default_cache = {'expires': 0.0, 'row': None, 'secondary_expires': 0.0, 'secondary': None}
_default_cache_lock = threading.Lock()
_warned_once = {'projection': False, 'row': False}


def invalidate_server_cache():
    with _default_cache_lock:
        _default_cache['expires'] = 0.0
        _default_cache['row'] = None
        _default_cache['secondary_expires'] = 0.0
        _default_cache['secondary'] = None
        _warned_once['projection'] = False
        _warned_once['row'] = False


def _warn_once(key):
    with _default_cache_lock:
        if _warned_once[key]:
            return False
        _warned_once[key] = True
        return True


def _rollback(db):
    try:
        db.rollback()
    except Exception:
        logger.debug("Rollback failed (connection may be closed)", exc_info=True)


def creds_from_config(server_type):
    """Build a ``user_creds`` dict for ``server_type`` from the config globals.

    Only used to seed the registry from a legacy install's env/config at first
    boot (init_db migration); at runtime the registry row is the source and the
    config globals are its projection.
    """
    creds = {}
    for field in config.MEDIASERVER_FIELDS_BY_TYPE.get(server_type, []):
        key = config.MEDIASERVER_CRED_KEY_BY_FIELD.get(field)
        if key:
            creds[key] = getattr(config, field) or ""
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
    }


def _rows(db, where="", params=()):
    cur = db.cursor(cursor_factory=DictCursor)
    try:
        cur.execute(
            "SELECT server_id, name, server_type, creds, music_libraries, is_default "
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
    if conn is None:
        now = time.monotonic()
        with _default_cache_lock:
            if _default_cache['expires'] > now:
                row = _default_cache['row']
                return dict(row) if row else None
    db = conn or get_db()
    rows = _rows(db, "WHERE is_default ORDER BY name ASC LIMIT 1")
    result = rows[0] if rows else None
    if conn is None:
        with _default_cache_lock:
            _default_cache['row'] = dict(result) if result else None
            _default_cache['expires'] = time.monotonic() + _DEFAULT_CACHE_TTL
    return result


def get_default_server_id(conn=None):
    server = get_default_server(conn)
    return server["server_id"] if server else None


def servers_for_scope(scope, conn=None):
    """Resolve which servers an operation with ``scope`` should touch.

    Returns a list of normalized server dicts; a ``None`` element means 'legacy
    config default, bind no context'. An empty/unreadable registry yields
    ``[None]`` so single-server installs behave exactly as before - but a lost
    database connection propagates, so a batch run fails (and is retried)
    instead of silently shrinking to a default-only run that reports SUCCESS.
    ``scope == 'default'`` returns only the default server; ``'all'`` (or any
    falsy scope) returns every configured server; anything else is treated as
    one specific server's id or display name and returns just that server
    (``[]`` when it matches nothing).
    """
    try:
        servers = list_servers(conn)
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        raise
    except Exception:
        logger.exception("Server registry unavailable; using the legacy config default")
        return [None]
    if not servers:
        return [None]
    if scope == 'default':
        return [s for s in servers if s['is_default']]
    if scope and scope != 'all':
        wanted = str(scope)
        matched = [
            s for s in servers
            if s['server_id'] == wanted or s['name'].lower() == wanted.lower()
        ]
        if not matched:
            logger.warning("Server scope %r matches no configured server", scope)
        return matched
    return servers


def has_secondary_servers(conn=None):
    """True when any non-default server exists (cached like the default row)."""
    if conn is None:
        now = time.monotonic()
        with _default_cache_lock:
            if _default_cache['secondary_expires'] > now:
                return bool(_default_cache['secondary'])
    db = conn or get_db()
    cur = db.cursor()
    try:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM music_servers WHERE NOT is_default)"
        )
        result = bool(cur.fetchone()[0])
    finally:
        cur.close()
    if conn is None:
        with _default_cache_lock:
            _default_cache['secondary'] = result
            _default_cache['secondary_expires'] = time.monotonic() + _DEFAULT_CACHE_TTL
    return result


def _config_projection_lost(default):
    server_type = (default.get('server_type') or '').strip().lower()
    if (config.MEDIASERVER_TYPE or '').strip().lower() != server_type:
        return True
    if (config.MUSIC_LIBRARIES or '') != (default.get('music_libraries') or ''):
        return True
    row_creds = default.get('creds') or {}
    projected = creds_from_config(server_type)
    return any(
        value != (row_creds.get(key) or '') for key, value in projected.items()
    )


def _default_context(default):
    if default is None:
        return None
    missing = missing_required_creds(default['server_type'], default['creds'])
    if missing:
        if _warn_once('row'):
            logger.error(
                "Default server '%s' has no %s stored in the registry, so nothing can "
                "reach it. Fix it in the setup wizard.",
                default['name'], ', '.join(missing),
            )
        return None
    if not _config_projection_lost(default):
        return None
    if _warn_once('projection'):
        logger.warning(
            "This process's config does not match default server '%s' (it started "
            "before the database had those settings, or they changed since); binding "
            "the registry row directly. Restart the process to reload the config "
            "module.",
            default['name'],
        )
    return default


def context_for(server_id, conn=None):
    """Return the context dict for ``server_id``, or None to mean 'use config default'.

    Returning None for the default server keeps its code path byte-identical to
    the historical single-server behaviour (provider backends fall back to config)
    - but ONLY while those config globals still agree with the default row. When
    they do not (a worker that imported config before the database was reachable,
    or that never saw a later wizard change), the row itself is bound so the call
    reaches the right server instead of an empty URL or a stale one. A default row
    with no usable credentials is never bound: the config fallback is left in place
    and the operator is told to fix the row.
    """
    db = conn or get_db()
    default = get_default_server(db if conn is not None else None)
    default_id = default["server_id"] if default else None
    if not server_id or server_id == default_id:
        return _default_context(default)
    server = get_server(server_id, db)
    if server is not None:
        # Providers read a missing credential from the config globals, which are
        # the DEFAULT server's projection - so an incomplete secondary would
        # quietly talk to the wrong machine. The API refuses to store one, but
        # say so loudly if a legacy row ever gets here.
        missing = missing_required_creds(server['server_type'], server['creds'])
        if missing:
            logger.error(
                "Server '%s' is missing required credentials (%s); its provider "
                "calls would fall back to the default server. Fix it in the setup "
                "wizard.",
                server['name'], ', '.join(missing),
            )
    return server


def bind(server, conn=None):
    """Context manager binding one server row from ``servers_for_scope``.

    ``with registry.bind(server):`` is the ONE way a per-server loop targets its
    server. ``None`` (the legacy config default) binds nothing, exactly as
    ``context_for`` resolves it. Every scheduled feature that iterates servers -
    analysis, clustering, cleaning, radio, sonic fingerprint, plugin cron tasks -
    uses this instead of re-deriving the context itself: an omitted or wrong
    binding silently talks to the DEFAULT server, which is invisible until a
    playlist lands on the wrong machine.
    """
    from . import context as ms_context

    server_id = server['server_id'] if server else None
    if not server_id:
        return ms_context.use_server(None)
    ctx = context_for(server_id, conn)
    if ctx is None:
        if not server.get('is_default') and server_id != get_default_server_id(conn):
            raise ValueError(f"Unknown music server '{server_id}'")
        ctx = {'server_id': server_id}
    return ms_context.use_server(ctx)


def for_each_server(scope, func, *args, **kwargs):
    servers = servers_for_scope(scope)
    results = []
    failures = []
    for server in servers:
        name = server['name'] if server else 'default server'
        try:
            with bind(server):
                results.append(func(*args, **kwargs))
        except Exception as exc:
            logger.exception('Operation failed on %s; continuing', name)
            failures.append(f'{name}: {exc}')
    if failures and not results:
        raise RuntimeError('Operation failed on every server: ' + '; '.join(failures))
    if failures:
        logger.warning(
            'Operation completed on %d/%d servers (%s)',
            len(results), len(servers), '; '.join(failures),
        )
    return results[0] if len(results) == 1 else results


def _clear_default(db):
    cur = db.cursor()
    try:
        cur.execute("UPDATE music_servers SET is_default = FALSE, updated_at = now() WHERE is_default")
    finally:
        cur.close()


def add_server(name, server_type, creds, music_libraries="", make_default=False, conn=None):
    db = conn or get_db()
    server_id = uuid.uuid4().hex
    cur = db.cursor()
    try:
        cur.execute("SELECT EXISTS (SELECT 1 FROM music_servers WHERE is_default)")
        make_default = bool(make_default or not cur.fetchone()[0])
        if make_default:
            _clear_default(db)
        cur.execute(
            "INSERT INTO music_servers "
            "(server_id, name, server_type, creds, music_libraries, is_default) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (server_id, name, server_type, Json(dict(creds or {})),
             music_libraries or "", bool(make_default)),
        )
        db.commit()
        invalidate_server_cache()
        return server_id
    except Exception:
        _rollback(db)
        raise
    finally:
        cur.close()


def update_server(server_id, name=None, server_type=None, creds=None,
                  music_libraries=None, conn=None):
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
    if not sets:
        return
    sets.append("updated_at = now()")
    params.append(server_id)
    cur = db.cursor()
    try:
        cur.execute("UPDATE music_servers SET " + ", ".join(sets) + " WHERE server_id = %s", params)
        db.commit()
        invalidate_server_cache()
    except Exception:
        _rollback(db)
        raise
    finally:
        cur.close()


def set_default(server_id, conn=None):
    db = conn or get_db()
    cur = db.cursor()
    try:
        _clear_default(db)
        cur.execute(
            "UPDATE music_servers SET is_default = TRUE, updated_at = now() WHERE server_id = %s",
            (server_id,),
        )
        if not cur.rowcount:
            # The row vanished between the caller's check and this write (a
            # concurrent delete). Committing now would clear the old default and
            # promote nothing, leaving the install with NO default server.
            raise ValueError(f"Server '{server_id}' no longer exists; the default was not changed.")
        db.commit()
        invalidate_server_cache()
    except Exception:
        _rollback(db)
        raise
    finally:
        cur.close()


def delete_server(server_id, conn=None):
    db = conn or get_db()
    server = get_server(server_id, db)
    if server is None:
        return False
    if server["is_default"]:
        raise ValueError("Cannot delete the default server; set another server as default first.")
    cur = db.cursor()
    try:
        cur.execute("DELETE FROM playlist WHERE server_id = %s", (server_id,))
        cur.execute("DELETE FROM music_servers WHERE server_id = %s", (server_id,))
        db.commit()
        invalidate_server_cache()
        try:
            from tasks.paged_ivf import invalidate_availability_cache
            invalidate_availability_cache(server_id)
        except Exception:
            logger.debug("Availability-cache invalidation failed", exc_info=True)
        return True
    except Exception:
        _rollback(db)
        raise
    finally:
        cur.close()


def save_default_server_settings(server_type, creds, music_libraries="", conn=None):
    """Persist the setup wizard's media-server settings into the default row.

    The registry is the ONLY home of these settings; config module globals are a
    read-only projection refreshed from this row. Creates the default row when
    the registry is still empty (fresh install saving the wizard for the first
    time).
    """
    db = conn or get_db()
    server_type = (server_type or "").strip().lower()
    default = get_default_server(db if conn is not None else None)
    if default is None:
        add_server(
            name=_default_server_name(server_type),
            server_type=server_type,
            creds=creds,
            music_libraries=music_libraries or "",
            make_default=False,
            conn=db,
        )
        return
    update_server(
        default["server_id"],
        server_type=server_type,
        creds=creds,
        music_libraries=music_libraries or "",
        conn=db,
    )


def _default_server_name(server_type):
    return (server_type or "media server").capitalize()


def availability_sql(alias='s'):
    """SQL fragment: the aliased score row is present on a server.

    Takes two params in order: ``server_id`` and ``include_legacy_default``
    (bool). A row counts as present when it has a track_server_map row for the
    server, or - when the flag is true, i.e. the target is the default server -
    when it is a legacy provider-keyed row (non ``fp_`` id) predating
    canonicalization. The ONE spelling of this rule; every query filtering by
    server availability must build it here.
    """
    return (
        "(EXISTS (SELECT 1 FROM track_server_map availability "
        f"WHERE availability.item_id = {alias}.item_id AND availability.server_id = %s) "
        f"OR (%s AND left({alias}.item_id, 3) <> 'fp_'))"
    )


_TRANSLATE_IDS_CHUNK = 5000


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
    default = get_default_server(db if conn is not None else None)
    default_id = default["server_id"] if default else None
    is_default = (not server_id) or server_id == default_id
    target = server_id or default_id
    if target is None:
        from tasks.simhash import is_fingerprint_id
        return {i: i for i in ids if not is_fingerprint_id(i)}
    cur = db.cursor()
    mapped = {}
    try:
        # N provider tracks may map to one item_id on a server (duplicate files);
        # pick ONE deterministically - strongest match tier, then the smallest
        # provider id as a stable tiebreak - so a playlist target never changes
        # between runs.
        for start in range(0, len(ids), _TRANSLATE_IDS_CHUNK):
            chunk = ids[start:start + _TRANSLATE_IDS_CHUNK]
            cur.execute(
                "SELECT DISTINCT ON (item_id) item_id, provider_track_id "
                "FROM track_server_map WHERE server_id = %s AND item_id = ANY(%s) "
                "ORDER BY item_id, "
                "  CASE match_tier "
                "    WHEN 'fingerprint' THEN 0 WHEN 'path' THEN 1 WHEN 'tail' THEN 2 "
                "    WHEN 'exact_meta' THEN 3 WHEN 'default' THEN 4 WHEN 'norm_meta' THEN 5 "
                "    WHEN 'title_artist' THEN 6 WHEN 'analysis' THEN 7 ELSE 8 END, "
                "  provider_track_id",
                (target, chunk),
            )
            mapped.update({r[0]: r[1] for r in cur.fetchall()})
    finally:
        cur.close()
    if is_default:
        from tasks.simhash import is_fingerprint_id
        dropped = sum(1 for i in ids if is_fingerprint_id(i) and i not in mapped)
        if dropped:
            logger.info(
                "%d canonical ids have no mapping on the default server (not on it, "
                "or awaiting a re-analysis after a split); dropped from translation",
                dropped,
            )
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
    default = get_default_server(db if conn is not None else None)
    default_id = default["server_id"] if default else None
    is_default = (not server_id) or server_id == default_id
    target = server_id or default_id
    if target is None:
        return {i: i for i in ids}
    clean_of = {i: sanitize_string_for_db(i) for i in ids}
    lookup = [c for c in dict.fromkeys(clean_of.values()) if c]
    by_clean = {}
    if lookup:
        cur = db.cursor()
        try:
            cur.execute(
                "SELECT provider_track_id, item_id FROM track_server_map "
                "WHERE server_id = %s AND provider_track_id = ANY(%s)",
                (target, lookup),
            )
            by_clean = {r[0]: r[1] for r in cur.fetchall()}
        finally:
            cur.close()
    if is_default:
        return {i: by_clean.get(clean_of[i], i) for i in ids}
    return {i: by_clean[clean_of[i]] for i in ids if clean_of[i] in by_clean}


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


def _staged_map_upsert(db, rows, stage_ddl, stage_insert, conflict_delete,
                       final_insert, final_template, pre_commit=None):
    """Shared transactional scaffold for the *_server_map bulk upserts.

    Stages ``rows`` into a temp table, deletes rows conflicting on the
    alternate unique key, ON CONFLICT-upserts the batch, then runs the optional
    ``pre_commit(cur)`` hook and commits; any failure rolls back.
    """
    cur = db.cursor()
    try:
        cur.execute(stage_ddl)
        execute_values(cur, stage_insert, rows, page_size=5000)
        cur.execute(conflict_delete)
        execute_values(cur, final_insert, rows, template=final_template, page_size=5000)
        if pre_commit is not None:
            pre_commit(cur)
        db.commit()
        return len(rows)
    except Exception:
        _rollback(db)
        raise
    finally:
        cur.close()


def upsert_artist_maps(server_id, mapping, conn=None):
    """Bulk-upsert ``{artist_name: provider_artist_id}`` for one server."""
    cleaned = {}
    for name, provider_id in (mapping or {}).items():
        if name and provider_id and server_id:
            clean_name = sanitize_string_for_db_loud(name, 'artist_name')
            clean_id = sanitize_string_for_db_loud(provider_id, 'provider_artist_id')
            if not clean_name or not clean_id:
                continue
            rank = (0 if clean_name == str(name) else 1, clean_id)
            prev = cleaned.get(clean_name)
            if prev is None or rank < prev:
                cleaned[clean_name] = rank
    rows_by_provider = {}
    for clean_name, (_dirty, clean_id) in cleaned.items():
        rows_by_provider[clean_id] = (clean_name, str(server_id), clean_id)
    rows = list(rows_by_provider.values())
    if not rows:
        return 0
    db = conn or get_db()
    return _staged_map_upsert(
        db,
        rows,
        stage_ddl=(
            "CREATE TEMP TABLE incoming_artist_server_map "
            "(artist_name TEXT, server_id TEXT, provider_artist_id TEXT) ON COMMIT DROP"
        ),
        stage_insert="INSERT INTO incoming_artist_server_map VALUES %s",
        conflict_delete=(
            "DELETE FROM artist_server_map current USING incoming_artist_server_map incoming "
            "WHERE current.server_id = incoming.server_id "
            "AND current.provider_artist_id = incoming.provider_artist_id "
            "AND current.artist_name <> incoming.artist_name"
        ),
        final_insert=(
            "INSERT INTO artist_server_map "
            "(artist_name, server_id, provider_artist_id, updated_at) VALUES %s "
            "ON CONFLICT (artist_name, server_id) DO UPDATE SET "
            "provider_artist_id = EXCLUDED.provider_artist_id, updated_at = now()"
        ),
        final_template="(%s, %s, %s, now())",
    )


def _artist_lookup(values, server_id, conn, map_columns):
    """Directional artist_server_map lookup; ``map_columns`` is a constant
    (source, result) column pair, never caller input. The default server's ids now
    live in artist_server_map like every other server's (the legacy artist_mapping
    table was folded in and dropped at startup), so there is no fallback."""
    wanted = [str(value) for value in dict.fromkeys(values) if value]
    if not wanted:
        return {}
    clean_of = {value: sanitize_string_for_db(value) for value in wanted}
    lookup = [c for c in dict.fromkeys(clean_of.values()) if c]
    if not lookup:
        return {}
    db = conn or get_db()
    target = server_id or get_default_server_id(db)
    if not target:
        return {}
    src, dst = map_columns
    cur = db.cursor()
    try:
        cur.execute(
            f"SELECT {src}, {dst} FROM artist_server_map "
            f"WHERE server_id = %s AND {src} = ANY(%s)",
            (target, lookup),
        )
        by_clean = {str(row[0]): str(row[1]) for row in cur.fetchall()}
    finally:
        cur.close()
    return {
        value: by_clean[clean_of[value]]
        for value in wanted
        if clean_of[value] in by_clean
    }


def artist_names_for_ids(provider_artist_ids, server_id=None, conn=None):
    return _artist_lookup(
        provider_artist_ids, server_id, conn,
        map_columns=('provider_artist_id', 'artist_name'),
    )


def artist_ids_for_names(artist_names, server_id=None, conn=None):
    return _artist_lookup(
        artist_names, server_id, conn,
        map_columns=('artist_name', 'provider_artist_id'),
    )


def artist_track_counts(artist_names, server_id=None, conn=None):
    """Selected-server track counts for artist names."""
    names = [str(value) for value in artist_names if value]
    if not names:
        return {}
    db = conn or get_db()
    default_id = get_default_server_id(db)
    target = server_id or default_id
    if not target:
        return {}
    cur = db.cursor()
    try:
        cur.execute(
            "SELECT s.author, COUNT(*) FROM score s WHERE s.author = ANY(%s) AND "
            + availability_sql('s')
            + " GROUP BY s.author",
            (names, target, target == default_id),
        )
        return {str(row[0]): int(row[1]) for row in cur.fetchall()}
    finally:
        cur.close()


def upsert_track_maps(server_id, mapping, conn=None):
    """Bulk-upsert ``{provider_track_id: (item_id, match_tier, file_path)}``.

    N provider tracks may map to one canonical item_id on a server (duplicate
    files sharing one fingerprint). Keyed by ``provider_track_id``, the table's
    PRIMARY KEY column, so every provider row survives instead of colliding on
    ``(item_id, server_id)``.

    ``file_path`` is THIS server's path for the file and is optional: a caller
    that does not know it passes a 2-tuple and the stored path is left alone
    (COALESCE), so a path-less writer can never erase a path a sweep recorded.
    """
    if not mapping:
        return 0
    db = conn or get_db()
    rows_by_provider = {}
    for provider_track_id, value in mapping.items():
        if isinstance(value, (tuple, list)):
            item_id = value[0]
            match_tier = value[1] if len(value) > 1 else None
            file_path = value[2] if len(value) > 2 else None
        else:
            item_id, match_tier, file_path = value, None, None
        if provider_track_id is None or provider_track_id == '':
            continue
        if item_id is None or item_id == '':
            continue
        provider_track_id = sanitize_string_for_db_loud(
            provider_track_id, 'provider_track_id'
        )
        if not provider_track_id:
            continue
        file_path = (
            (sanitize_string_for_db_loud(file_path, 'file_path') or None)
            if file_path else None
        )
        rows_by_provider[provider_track_id] = (
            str(item_id), server_id, provider_track_id, match_tier, file_path,
        )
    rows = list(rows_by_provider.values())
    if not rows:
        return 0

    def _touch_server(cur):
        cur.execute(
            "UPDATE music_servers SET updated_at = now() WHERE server_id = %s",
            (server_id,),
        )
        if not config.CHROMAPRINT_COLLECTION_ENABLED:
            return
        inherited = inherit_chromaprints_from_staged_maps(cur)
        if inherited:
            logger.info(
                "%d of the %d mapping(s) written for server %s inherited the "
                "Chromaprint already stored for their canonical track",
                inherited, len(rows), server_id,
            )

    def _run():
        return _staged_map_upsert(
            db,
            rows,
            stage_ddl=(
                "CREATE TEMP TABLE incoming_track_server_map "
                "(item_id TEXT, server_id TEXT, provider_track_id TEXT, match_tier TEXT, "
                "file_path TEXT) ON COMMIT DROP"
            ),
            stage_insert="INSERT INTO incoming_track_server_map VALUES %s",
            conflict_delete=(
                "DELETE FROM track_server_map current USING incoming_track_server_map incoming "
                "WHERE current.server_id = incoming.server_id "
                "AND current.provider_track_id = incoming.provider_track_id "
                "AND current.item_id <> incoming.item_id"
            ),
            final_insert=(
                "INSERT INTO track_server_map "
                "(item_id, server_id, provider_track_id, match_tier, file_path, updated_at) "
                "VALUES %s "
                "ON CONFLICT (server_id, provider_track_id) DO UPDATE SET "
                "item_id = EXCLUDED.item_id, "
                "match_tier = EXCLUDED.match_tier, "
                "file_path = COALESCE(EXCLUDED.file_path, track_server_map.file_path), "
                "updated_at = now()"
            ),
            final_template="(%s, %s, %s, %s, %s, now())",
            pre_commit=_touch_server,
        )

    try:
        written = _run()
    except (psycopg2.errors.InvalidColumnReference, psycopg2.errors.UniqueViolation) as exc:
        # Two distinct broken schemas reach here, and only one of them raises 42P10.
        #  - The (server_id, provider_track_id) key is missing entirely: the ON
        #    CONFLICT arbiter is unknown, so Postgres raises InvalidColumnReference.
        #  - The PK swap failed halfway, leaving the NEW unique index AND the OLD
        #    (item_id, server_id) primary key both enforced. The arbiter then
        #    resolves fine, so 42P10 never fires - but an N:1 insert (the whole
        #    point of the relaxed key) dies with a UniqueViolation on the surviving
        #    old PK, which nothing used to catch.
        logger.warning(
            "track_server_map does not have the (server_id, provider_track_id) "
            "primary key; ensuring the schema and retrying the upsert."
        )
        from database import ensure_track_server_map_schema, track_server_map_pk_columns
        ensure_track_server_map_schema(db)
        columns = track_server_map_pk_columns(db)
        if columns != ['server_id', 'provider_track_id']:
            # ensure_ returns True even when the swap silently rolled back, so trust
            # the catalog, not the boolean. Retrying here would hit the same 23505.
            raise RuntimeError(
                "track_server_map primary key is still %s; the relaxation migration "
                "did not complete. Check the container logs from startup." % (columns,)
            ) from exc
        written = _run()
    try:
        from tasks.paged_ivf import invalidate_availability_cache
        invalidate_availability_cache(server_id)
    except Exception:
        logger.debug("Availability-cache invalidation failed", exc_info=True)
    return written


def mapped_count(server_id, conn=None):
    """Number of canonical tracks that have a mapping on ``server_id``."""
    db = conn or get_db()
    cur = db.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM track_server_map WHERE server_id = %s", (server_id,))
        return cur.fetchone()[0]
    finally:
        cur.close()
