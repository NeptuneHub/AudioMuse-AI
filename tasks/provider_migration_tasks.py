# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Orchestrate the media-provider migration as RQ jobs.

Drives the multi-step migration flow whose dry-run and source-refresh phases run
as RQ jobs polled by the UI; delegates track matching to provider_migration_matcher
and reuses the app's core routines under an app context.

Main Features:
* Under an advisory lock, pauses and drains workers, then rewrites item ids and
  index id-maps inside a single transaction that drops and re-adds the score
  foreign keys, applies new provider metadata, and updates app_config.
* Reads target metadata from the migration_target_meta side table and builds
  the old->new id mapping via indexed per-album queries; reloads state and
  index tables after commit.
"""

import json
import logging
import re
import time

from sanitization import sanitize_string_for_db as _sanitize_text

logger = logging.getLogger(__name__)


_ADVISORY_LOCK_KEY = 7421536190082003

_DRAIN_TIMEOUT_SECONDS = 60

_MIG_TMP_PREFIX = '__audiomuse_mig_tmp__'


def rewrite_id_map_json(id_map_json, mapping):
    if not id_map_json:
        return id_map_json
    try:
        m = json.loads(id_map_json)
    except Exception:
        logger.warning("Could not parse id_map_json, leaving it unchanged")
        return id_map_json
    if isinstance(m, dict):
        rewritten = {}
        for k, v in m.items():
            if v in mapping:
                rewritten[k] = mapping[v]
        return json.dumps(rewritten)
    if isinstance(m, list):
        rewritten = [mapping[v] if v in mapping else None for v in m]
        return json.dumps(rewritten)
    logger.warning(
        "id_map_json has unexpected top-level type %s, leaving it unchanged",
        type(m).__name__,
    )
    return id_map_json


def find_fk(cur, table, column, ref_table='score', ref_column='item_id'):
    cur.execute(
        """
        SELECT tc.constraint_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_name = kcu.table_name
        JOIN information_schema.constraint_column_usage ccu
          ON tc.constraint_name = ccu.constraint_name
        WHERE tc.table_name = %s
          AND tc.constraint_type = 'FOREIGN KEY'
          AND kcu.column_name = %s
          AND ccu.table_name = %s
          AND ccu.column_name = %s
        LIMIT 1
        """,
        (table, column, ref_table, ref_column),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _get_dedicated_conn():
    import psycopg2
    import config  # noqa: F401  (lazy so tests don't need live env vars)

    return psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        dbname=config.POSTGRES_DB,
    )


def _get_redis():
    from app_helper import redis_conn

    return redis_conn


def _drain_workers_or_timeout(seconds=_DRAIN_TIMEOUT_SECONDS):
    deadline = time.time() + seconds
    while time.time() < deadline:
        time.sleep(1)
        break


def execute_provider_migration(session_id):
    logger.info("provider migration: starting session %s", session_id)

    redis = _get_redis()
    redis.set('migration:paused', '1', ex=3600)
    try:
        _pause_and_drain_workers(redis)

        conn = _get_dedicated_conn()
        try:
            conn.autocommit = False
        except Exception:
            pass

        cur = conn.cursor()

        session = _load_session(cur, session_id)
        target_type = session['target_type']
        target_creds = session['target_creds']
        state = session['state']

        if session['status'] != 'dry_run_ready':
            raise RuntimeError(
                f"Cannot execute migration: session {session_id} is in status "
                f"'{session['status']}', expected 'dry_run_ready'"
            )

        mapping = _merge_mapping(state)
        new_meta = _load_new_meta_from_table(cur, session_id)
        selected_libraries = state.get('selected_libraries')
        logger.info("provider migration: %d tracks will be rewritten", len(mapping))

        fk_embedding = find_fk(cur, 'embedding', 'item_id')
        fk_clap_embedding = find_fk(cur, 'clap_embedding', 'item_id')
        cur.execute("SELECT to_regclass('public.lyrics_embedding') IS NOT NULL")
        lyrics_exists = bool(cur.fetchone()[0])
        fk_lyrics_embedding = find_fk(cur, 'lyrics_embedding', 'item_id') if lyrics_exists else None

        try:
            index_rebuild_needed = _run_migration_transaction(
                cur=cur,
                mapping=mapping,
                new_meta=new_meta,
                fk_embedding=fk_embedding,
                fk_clap_embedding=fk_clap_embedding,
                fk_lyrics_embedding=fk_lyrics_embedding,
                lyrics_exists=lyrics_exists,
                target_type=target_type,
                target_creds=target_creds,
                session_id=session_id,
                selected_libraries=selected_libraries,
            )
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise

        _post_commit_reload(redis)

        return {
            'ok': True,
            'matched': len(mapping),
            'index_rebuild_needed': bool(index_rebuild_needed),
        }
    finally:
        try:
            redis.delete('migration:paused')
        except Exception:
            pass


def _pause_and_drain_workers(redis):
    try:
        from rq import Worker  # pragma: no cover - optional import in tests

        for w in Worker.all(connection=redis):
            try:
                w.send_stop_signal()
            except Exception as e:
                logger.debug("worker stop signal failed (ignored): %s", e)
    except Exception as e:
        logger.debug("rq worker enumeration failed (ignored): %s", e)
    _drain_workers_or_timeout()


def _load_session(cur, session_id):
    cur.execute(
        "SELECT id, target_type, target_creds, state, status FROM migration_session WHERE id = %s",
        (session_id,),
    )
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"migration_session {session_id} not found")
    _id, target_type, target_creds_json, state_json, status = row
    try:
        creds = (
            json.loads(target_creds_json)
            if isinstance(target_creds_json, str)
            else target_creds_json
        )
    except Exception:
        creds = {}
    try:
        state = json.loads(state_json) if isinstance(state_json, str) else state_json
    except Exception:
        state = {}
    return {
        'id': _id,
        'target_type': target_type,
        'target_creds': creds,
        'state': state or {},
        'status': status,
    }


def build_mapping(state):
    dry = (state.get('dry_run') or {}).get('matches') or {}
    manual = state.get('manual_matches') or {}
    manual_unmatches = set(state.get('manual_unmatches') or [])

    merged = {}
    for old_id, new_id in dry.items():
        if old_id in manual_unmatches:
            continue
        merged[old_id] = new_id
    merged.update(manual)

    seen_new = {}
    deduped = {}
    dropped = []
    for old_id, new_id in merged.items():
        key = str(new_id)
        if key in seen_new:
            dropped.append((old_id, new_id, seen_new[key]))
            continue
        seen_new[key] = old_id
        deduped[old_id] = new_id
    return deduped, dropped


def _merge_mapping(state):
    deduped, dropped = build_mapping(state)
    if dropped:
        logger.warning(
            "provider migration: dropped %d mapping(s) that collided on "
            "new_id (multiple source rows pointed at the same target id); "
            "those rows will be orphaned on execute. First 10: %s",
            len(dropped),
            dropped[:10],
        )
    return deduped


def _load_new_meta_from_table(cur, session_id):
    cur.execute("SELECT to_regclass('public.migration_target_meta')")
    if cur.fetchone()[0] is None:
        logger.warning(
            "provider migration: migration_target_meta does not exist; item ids "
            "will be rewritten but the target's path/title/artist/album will not "
            "be applied to the catalogue"
        )
        return {}
    cur.execute(
        "SELECT new_id, path, title, artist, album, album_artist, year "
        "FROM migration_target_meta WHERE session_id = %s",
        (session_id,),
    )
    out = {}
    for r in cur.fetchall() or []:
        out[r[0]] = {
            'path': r[1],
            'title': r[2],
            'artist': r[3],
            'album': r[4],
            'album_artist': r[5],
            'year': r[6],
        }
    if not out:
        logger.warning(
            "provider migration: session %s has no target metadata rows; the "
            "catalogue keeps the SOURCE provider's metadata (re-run the dry run "
            "to collect it again)",
            session_id,
        )
    return out


def _populate_migration_map_table(cur, mapping):
    cur.execute(
        "CREATE TEMP TABLE item_id_migration_map ("
        " old_id TEXT PRIMARY KEY, "
        " new_id TEXT NOT NULL UNIQUE"
        ") ON COMMIT DROP"
    )
    _rows = list(mapping.items())
    for i in range(0, len(_rows), 1000):
        chunk = _rows[i : i + 1000]
        placeholders = ",".join(["(%s,%s)"] * len(chunk))
        flat = [v for pair in chunk for v in pair]
        cur.execute(
            "INSERT INTO item_id_migration_map (old_id, new_id) VALUES " + placeholders,  # nosec B608 - %s-placeholder string only; values are bound params
            flat,
        )
    cur.execute("ANALYZE item_id_migration_map")


def _drop_fk_constraints(cur, fk_embedding, fk_clap_embedding, lyrics_exists, fk_lyrics_embedding):
    if fk_embedding:
        cur.execute(f"ALTER TABLE embedding DROP CONSTRAINT {fk_embedding}")
    if fk_clap_embedding:
        cur.execute(f"ALTER TABLE clap_embedding DROP CONSTRAINT {fk_clap_embedding}")
    if lyrics_exists and fk_lyrics_embedding:
        cur.execute(f"ALTER TABLE lyrics_embedding DROP CONSTRAINT {fk_lyrics_embedding}")


def _readd_fk_constraints(cur, fk_embedding, fk_clap_embedding, lyrics_exists, fk_lyrics_embedding):
    if fk_embedding:
        cur.execute(
            f"ALTER TABLE embedding ADD CONSTRAINT {fk_embedding} "
            f"FOREIGN KEY (item_id) REFERENCES score(item_id) ON DELETE CASCADE"
        )
    if fk_clap_embedding:
        cur.execute(
            f"ALTER TABLE clap_embedding ADD CONSTRAINT {fk_clap_embedding} "
            f"FOREIGN KEY (item_id) REFERENCES score(item_id) ON DELETE CASCADE"
        )
    if lyrics_exists and fk_lyrics_embedding:
        cur.execute(
            f"ALTER TABLE lyrics_embedding ADD CONSTRAINT {fk_lyrics_embedding} "
            f"FOREIGN KEY (item_id) REFERENCES score(item_id) ON DELETE CASCADE"
        )


def _rewrite_item_ids(cur, lyrics_exists):
    prefix = _MIG_TMP_PREFIX
    for table, alias in (
        ("score", "s"),
        ("playlist", "p"),
        ("embedding", "e"),
        ("clap_embedding", "e"),
    ):
        cur.execute(
            f"UPDATE {table} {alias} SET item_id = %s || m.new_id "
            f"FROM item_id_migration_map m WHERE {alias}.item_id = m.old_id",
            (prefix,),
        )
        cur.execute(
            f"UPDATE {table} {alias} SET item_id = m.new_id "
            f"FROM item_id_migration_map m "
            f"WHERE {alias}.item_id = %s || m.new_id",
            (prefix,),
        )
    if lyrics_exists:
        cur.execute(
            "UPDATE lyrics_embedding e SET item_id = %s || m.new_id "
            "FROM item_id_migration_map m WHERE e.item_id = m.old_id",
            (prefix,),
        )
        cur.execute(
            "UPDATE lyrics_embedding e SET item_id = m.new_id "
            "FROM item_id_migration_map m "
            "WHERE e.item_id = %s || m.new_id",
            (prefix,),
        )


def _apply_new_meta(cur, new_meta):
    if not new_meta:
        return
    cur.execute(
        "CREATE TEMP TABLE migration_new_meta ("
        " new_id TEXT PRIMARY KEY, "
        " new_path TEXT, new_title TEXT, new_artist TEXT, "
        " new_album TEXT, new_album_artist TEXT, new_year INTEGER"
        ") ON COMMIT DROP"
    )
    _metas = list(new_meta.items())
    for i in range(0, len(_metas), 500):
        chunk = _metas[i : i + 500]
        placeholders = ",".join(["(%s,%s,%s,%s,%s,%s,%s)"] * len(chunk))
        flat = []
        for new_id, meta in chunk:
            flat.extend(
                (
                    _sanitize_text(new_id),
                    _sanitize_text(meta.get('path')),
                    _sanitize_text(meta.get('title')),
                    _sanitize_text(meta.get('artist')),
                    _sanitize_text(meta.get('album')),
                    _sanitize_text(meta.get('album_artist')),
                    meta.get('year'),
                )
            )
        cur.execute(
            "INSERT INTO migration_new_meta "
            "(new_id, new_path, new_title, new_artist, new_album, new_album_artist, new_year) "
            "VALUES " + placeholders,  # nosec B608 - %s-placeholder string only; values are bound params
            flat,
        )
    cur.execute(
        "UPDATE score s SET "
        "  file_path    = COALESCE(n.new_path,         s.file_path), "
        "  title        = COALESCE(n.new_title,        s.title), "
        "  author       = COALESCE(n.new_artist,       s.author), "
        "  album        = COALESCE(n.new_album,        s.album), "
        "  album_artist = COALESCE(n.new_album_artist, s.album_artist), "
        "  year         = COALESCE(n.new_year,         s.year) "
        "FROM migration_new_meta n WHERE s.item_id = n.new_id"
    )


def _rewrite_index_id_maps(cur, mapping):
    from tasks.index_build_helpers import rewrite_segmented_id_map

    _seg_base = re.compile(r"^(.*)_\d+_\d+$")
    index_rebuild_needed = []
    for table in ('voyager_index_data', 'map_projection_data'):
        cur.execute("SELECT to_regclass(%s)", (table,))
        if cur.fetchone()[0] is None:
            continue
        cur.execute(f"SELECT DISTINCT index_name FROM {table}")
        bases = set()
        for (name,) in cur.fetchall() or []:
            m = _seg_base.match(name)
            bases.add(m.group(1) if m else name)
        for base in sorted(bases):
            try:
                rewrite_segmented_id_map(
                    cur, table, base, lambda j: rewrite_id_map_json(j, mapping)
                )
            except ValueError:
                logger.warning(
                    "provider migration: id_map for '%s' in %s could not be "
                    "relabelled in place; dropping the stale index so it "
                    "rebuilds on the next analysis",
                    base,
                    table,
                    exc_info=True,
                )
                like_pattern = base.replace("_", r"\_") + r"\_%\_%"
                cur.execute(
                    f"DELETE FROM {table} WHERE index_name = %s OR index_name LIKE %s ESCAPE '\\'",
                    (base, like_pattern),
                )
                index_rebuild_needed.append(f"{table}:{base}")
    return index_rebuild_needed


def _clear_index_tables(cur):
    for table in (
        'ivf_cell',
        'ivf_dir',
        'artist_index_data',
        'artist_metadata_data',
        'artist_component_projection',
        'artist_mapping',
    ):
        cur.execute("SELECT to_regclass(%s)", (table,))
        if cur.fetchone()[0] is not None:
            cur.execute(f"DELETE FROM {table}")


def _clear_default_server_artist_map(cur):
    """Drop the default server's artist ids: they belong to the OLD provider.

    The migration repoints the default server at a new provider, so every
    ``provider_artist_id`` stored for it is now a dead id - the same reason the
    legacy ``artist_mapping`` table is cleared above. Track ids are repointed
    instead of dropped because the matcher produced a new id for each one;
    artists have no such mapping, so they are cleared and the next analysis
    rebuilds them. Secondary servers did not migrate: their rows stay.
    """
    cur.execute("SELECT to_regclass('public.artist_server_map')")
    if cur.fetchone()[0] is None:
        return
    cur.execute(
        "DELETE FROM artist_server_map a USING music_servers s "
        "WHERE s.is_default AND a.server_id = s.server_id"
    )
    if cur.rowcount:
        logger.info(
            "provider migration: cleared %d stale artist id(s) of the default server",
            cur.rowcount,
        )


def _run_migration_transaction(
    cur,
    mapping,
    new_meta,
    fk_embedding,
    fk_clap_embedding,
    fk_lyrics_embedding,
    lyrics_exists,
    target_type,
    target_creds,
    session_id,
    selected_libraries=None,
):
    cur.execute("SELECT pg_advisory_xact_lock(%s)", (_ADVISORY_LOCK_KEY,))

    _populate_migration_map_table(cur, mapping)

    cur.execute(
        "DELETE FROM score s WHERE NOT EXISTS "
        "(SELECT 1 FROM item_id_migration_map m WHERE m.old_id = s.item_id)"
    )

    _drop_fk_constraints(cur, fk_embedding, fk_clap_embedding, lyrics_exists, fk_lyrics_embedding)
    _rewrite_item_ids(cur, lyrics_exists)
    _readd_fk_constraints(cur, fk_embedding, fk_clap_embedding, lyrics_exists, fk_lyrics_embedding)

    cur.execute(
        "UPDATE track_server_map t SET provider_track_id = m.new_id, updated_at = now() "
        "FROM item_id_migration_map m, music_servers s "
        "WHERE s.is_default AND t.server_id = s.server_id AND t.item_id = m.new_id"
    )

    _apply_new_meta(cur, new_meta)

    index_rebuild_needed = _rewrite_index_id_maps(cur, mapping)

    _clear_index_tables(cur)
    _clear_default_server_artist_map(cur)

    _write_provider_to_default_server(
        cur, target_type, target_creds, selected_libraries=selected_libraries
    )
    _purge_media_keys_from_app_config(cur)

    cur.execute(
        "UPDATE migration_session SET status = 'completed', completed_at = NOW() WHERE id = %s",
        (session_id,),
    )

    return index_rebuild_needed


def _cleaned_libraries_value(selected_libraries):
    cleaned = [str(name).strip() for name in (selected_libraries or []) if str(name).strip()]
    cleaned = [name for name in cleaned if ',' not in name]
    return ','.join(cleaned)


def _write_provider_to_default_server(cur, target_type, target_creds, selected_libraries=None):
    """Point the music_servers default row at the migration target.

    The registry row is the source of truth the config globals are projected
    from (and init_db deletes the mediaserver app_config keys on boot), so
    without this update the provider switch would silently revert to the old
    server on the next config refresh. When there is no default row at all the
    row is CREATED: silently updating nothing would leave the whole install with
    a rewritten catalogue and no server to reach it with.
    """
    import uuid as _uuid

    from psycopg2.extras import Json

    cur.execute("SELECT to_regclass('public.music_servers') IS NOT NULL")
    if not cur.fetchone()[0]:
        return
    creds = Json(dict(target_creds or {}))
    libraries = _cleaned_libraries_value(selected_libraries)
    cur.execute(
        "UPDATE music_servers SET server_type = %s, creds = %s, music_libraries = %s, "
        "track_count = NULL, updated_at = now() WHERE is_default",
        (target_type, creds, libraries),
    )
    if cur.rowcount:
        logger.info(
            "provider migration: music_servers default row now targets '%s'",
            target_type,
        )
        return
    cur.execute(
        "INSERT INTO music_servers "
        "(server_id, name, server_type, creds, music_libraries, is_default) "
        "VALUES (%s, %s, %s, %s, %s, TRUE)",
        (_uuid.uuid4().hex, (target_type or 'media server').capitalize(),
         target_type, creds, libraries),
    )
    logger.warning(
        "provider migration: no default server existed; created one for '%s'",
        target_type,
    )


def _purge_media_keys_from_app_config(cur):
    """Drop any media-server rows a legacy install still has in app_config.

    The registry is the ONLY home of these settings, so the migration writes it
    and clears the legacy copies instead of maintaining a second one, which
    would leave a stale provider - credentials included - behind until the next
    restart. Boot does the same, through the same single implementation.
    """
    from database import purge_media_keys_from_app_config

    removed = purge_media_keys_from_app_config(cur)
    if removed:
        logger.info(
            "provider migration: removed %d legacy media-server key(s) from app_config",
            removed,
        )


def _post_commit_reload(redis):
    try:
        from tasks.mediaserver import registry

        registry.invalidate_server_cache()
    except Exception as e:
        logger.warning("registry cache invalidation failed: %s", e)
    try:
        import config

        config.refresh_config()
    except Exception as e:
        logger.warning("config.refresh_config() failed: %s", e)
    try:
        import restart_manager

        restart_manager.publish_restart_request()
    except Exception as e:
        logger.warning("restart_manager.publish_restart_request() failed: %s", e)


def dry_run_provider_migration(session_id, allow_title_artist_only=False):
    from app import app

    with app.app_context():
        import app_provider_migration

        return app_provider_migration.run_dry_run_core(
            session_id, allow_title_artist_only=allow_title_artist_only
        )


def source_refresh_provider_migration(session_id):
    from app import app

    with app.app_context():
        import app_provider_migration

        return app_provider_migration.run_source_refresh_core(session_id)
