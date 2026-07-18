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
* The centralized catalogue is NEVER touched: `score` rows, their canonical fp_2
  ids and their embeddings survive every migration. A provider swap only rewrites
  `track_server_map` for the default server - matched songs get the target's track
  id, unmatched songs are unbound from that server - so no song's analysis is ever
  lost and the similarity indexes stay valid (no rebuild).
* Under an advisory lock, repoints the default server's mappings, refreshes song
  metadata from the new provider, clears the old provider's artist ids, and points
  the music_servers default row at the target.
* Reads target metadata from the migration_target_meta side table and builds
  the old->new id mapping via indexed per-album queries; reloads state after commit.
"""

import json
import logging

from rq import get_current_job

from config import (
    TASK_STATUS_STARTED,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
)
from sanitization import sanitize_string_for_db as _sanitize_text

logger = logging.getLogger(__name__)


_ADVISORY_LOCK_KEY = 7421536190082003

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


# The migration itself no longer rewrites score.item_id, so it has no need to drop
# these. The one-time fingerprint canonicalization DOES relabel item_ids (that is its
# whole purpose) and imports both from here.
def _drop_fk_constraints(cur, fk_embedding, fk_clap_embedding, lyrics_exists, fk_lyrics_embedding):
    """Drop the embedding cascades. IF EXISTS, so a caller may pass the name it
    INTENDS the constraint to have rather than only a name it found."""
    if fk_embedding:
        cur.execute(f"ALTER TABLE embedding DROP CONSTRAINT IF EXISTS {fk_embedding}")
    if fk_clap_embedding:
        cur.execute(f"ALTER TABLE clap_embedding DROP CONSTRAINT IF EXISTS {fk_clap_embedding}")
    if lyrics_exists and fk_lyrics_embedding:
        cur.execute(
            f"ALTER TABLE lyrics_embedding DROP CONSTRAINT IF EXISTS {fk_lyrics_embedding}"
        )


def _readd_fk_constraints(cur, fk_embedding, fk_clap_embedding, lyrics_exists, fk_lyrics_embedding):
    """Re-add the embedding cascades UNCONDITIONALLY.

    This used to re-add only what ``find_fk`` had found. A schema whose constraint
    was missing (or merely named unexpectedly) therefore came out of the rewrite
    with NO cascade at all, and nothing said so: deleting a score row would then
    silently orphan its embeddings forever. Creating the constraint is the correct
    end state whether or not one was there to begin with, so the caller passes the
    name it wants and this always produces it.
    """
    for table, name in (
        ('embedding', fk_embedding),
        ('clap_embedding', fk_clap_embedding),
        ('lyrics_embedding', fk_lyrics_embedding if lyrics_exists else None),
    ):
        if not name:
            continue
        cur.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {name}")
        cur.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {name} "
            f"FOREIGN KEY (item_id) REFERENCES score(item_id) ON DELETE CASCADE"
        )


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


MIGRATION_TASK_TYPE = 'provider_migration'


def _migration_task_id():
    job = get_current_job()
    return job.id if job else None


def _report_migration(task_id, status, progress, message, details=None):
    """Write the migration's task_status row, so it is visible and mutually exclusive.

    Silently a no-op outside an RQ job (the dry-run helpers call the same code from
    a request thread in tests).
    """
    if not task_id:
        return
    try:
        from flask_app import app
        from database import save_task_status

        payload = {'message': message, 'status_message': message}
        if details:
            payload.update(details)
        with app.app_context():
            save_task_status(
                task_id, MIGRATION_TASK_TYPE, status, progress=progress, details=payload
            )
    except Exception:
        logger.exception("Could not record provider-migration task status")


def execute_provider_migration(session_id):
    """Repoint the DEFAULT server's mappings at a new provider. The catalogue stays.

    Writes task_status rows under the 'provider_migration' type so the run is visible
    and get_active_main_task can keep an analysis or a sweep from writing
    track_server_map underneath it. It used to claim it "paused and drained the
    workers" instead, which did nothing at all: send_stop_signal does not exist in
    RQ 2.x, the drain loop broke out after one second, and the migration:paused key
    had no reader anywhere.
    """
    logger.info("provider migration: starting session %s", session_id)

    redis = _get_redis()
    task_id = _migration_task_id()
    _report_migration(task_id, TASK_STATUS_STARTED, 0, "Provider migration started...")
    try:
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
        logger.info(
            "provider migration: %d songs will be repointed at the new provider; "
            "the catalogue itself is not touched", len(mapping),
        )

        try:
            index_rebuild_needed = _run_migration_transaction(
                cur=cur,
                mapping=mapping,
                new_meta=new_meta,
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

        summary = {
            'ok': True,
            'matched': len(mapping),
            'index_rebuild_needed': bool(index_rebuild_needed),
        }
        _report_migration(
            task_id, TASK_STATUS_SUCCESS, 100,
            f"Provider migration complete: {len(mapping)} tracks repointed.",
            details=summary,
        )
        return summary
    except Exception:
        logger.exception("Provider migration failed for session %s", session_id)
        _report_migration(
            task_id, TASK_STATUS_FAILURE, 100,
            "Provider migration failed; check the container logs.",
        )
        raise


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
    # Joined through the migration map on the CANONICAL id: score.item_id is the
    # content hash and is never rewritten, so it cannot be matched against the
    # target's provider id directly (it could, back when item_id WAS the provider id).
    # This refreshes the song's metadata from the new provider; it does not move it.
    cur.execute(
        "UPDATE score s SET "
        "  title        = COALESCE(n.new_title,        s.title), "
        "  author       = COALESCE(n.new_artist,       s.author), "
        "  album        = COALESCE(n.new_album,        s.album), "
        "  album_artist = COALESCE(n.new_album_artist, s.album_artist), "
        "  year         = COALESCE(n.new_year,         s.year) "
        "FROM migration_new_meta n, item_id_migration_map m "
        "WHERE m.new_id = n.new_id AND s.item_id = m.old_id"
    )
    cur.execute(
        "UPDATE track_server_map t SET file_path = n.new_path, updated_at = now() "
        "FROM migration_new_meta n, item_id_migration_map m, music_servers s "
        "WHERE m.new_id = n.new_id AND t.item_id = m.old_id "
        "AND s.is_default AND t.server_id = s.server_id "
        "AND n.new_path IS NOT NULL"
    )


def _clear_default_server_artist_map(cur):
    """Drop the default server's ARTIST ids: they belong to the OLD provider.

    Artist ids are the one thing the matcher cannot repoint - it produces a new id
    per TRACK, and artists have no such mapping - so they are cleared and the next
    analysis rebuilds them. Secondary servers did not migrate: their rows stay.

    Only ids are cleared, never analysis. The track similarity indexes are NOT
    touched: they are keyed by the canonical item_id, which a migration no longer
    changes, so every one of them stays valid across the provider swap.
    """
    cur.execute("SELECT to_regclass('public.artist_server_map')")
    if cur.fetchone()[0] is not None:
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
    target_type,
    target_creds,
    session_id,
    selected_libraries=None,
):
    """Repoint the DEFAULT server's mappings at a new provider. Nothing else.

    `score` is the centralized catalogue: one row per distinct recording, keyed by
    the fp_2 content hash of its own audio. That hash is a property of the AUDIO, not
    of any server, so a migration cannot change it and must never delete it. A song's
    analysis (its MusiCNN, CLAP and lyrics embeddings) is expensive and irreplaceable;
    a provider swap is just a change of where the file happens to live.

    So all a migration does is rewrite `track_server_map` for the default server:
      - matched     -> its provider_track_id becomes the target's id
      - unmatched   -> its mapping row is DROPPED (the song is unbound from this
                       server, exactly as if you had deleted it from the library),
                       while the score row, its embeddings and its mappings to any
                       OTHER server survive untouched
    An unbound song is hidden from that server's results by the availability filter,
    and if the file ever comes back a sweep re-binds it with no re-analysis.

    This used to DELETE unmatched score rows and REWRITE score.item_id into the target
    provider's track id - the pre-canonicalization design, where item_id WAS the
    provider id. In a union catalogue that destroyed the canonical ids outright, took
    the embeddings with them via ON DELETE CASCADE, and forced a full index rebuild.
    Nothing here touches score.item_id now, so the similarity indexes stay valid and
    no rebuild is needed.
    """
    cur.execute("SELECT pg_advisory_xact_lock(%s)", (_ADVISORY_LOCK_KEY,))

    _populate_migration_map_table(cur, mapping)

    # Unbind what the target does not have. The catalogue row stays.
    cur.execute(
        "DELETE FROM track_server_map t USING music_servers s "
        "WHERE s.is_default AND t.server_id = s.server_id "
        "AND NOT EXISTS (SELECT 1 FROM item_id_migration_map m WHERE m.old_id = t.item_id)"
    )

    # N duplicate files of one song each own a default-server row. The repoint below
    # would stamp them all with the SAME target id and violate the
    # (server_id, provider_track_id) key, so collapse to one row per song first: the
    # target provider gives exactly one id per matched song.
    cur.execute(
        "DELETE FROM track_server_map t USING music_servers s "
        "WHERE s.is_default AND t.server_id = s.server_id "
        "AND t.ctid <> (SELECT min(t2.ctid) FROM track_server_map t2 "
        "WHERE t2.item_id = t.item_id AND t2.server_id = t.server_id)"
    )

    # The song keeps its canonical id; only the provider id it is reachable by on the
    # default server changes. In TWO passes, via a prefix no provider id can collide
    # with: (server_id, provider_track_id) is a plain unique index, which Postgres
    # enforces row by row inside a single UPDATE. A re-point onto the SAME provider
    # (a library rebuild) permutes the ids rather than replacing them, so a one-pass
    # UPDATE trips the index the moment it assigns an id another row still holds.
    # Every unmatched and duplicate row is already gone above, so the only rows left
    # on this server are the matched ones: the prefixed values are distinct, and no
    # unprefixed row survives to collide with them.
    cur.execute(
        "UPDATE track_server_map t SET provider_track_id = %s || m.new_id, "
        "match_tier = 'default', updated_at = now() "
        "FROM item_id_migration_map m, music_servers s "
        "WHERE s.is_default AND t.server_id = s.server_id AND t.item_id = m.old_id",
        (_MIG_TMP_PREFIX,),
    )
    cur.execute(
        "UPDATE track_server_map t "
        "SET provider_track_id = substr(t.provider_track_id, %s) "
        "FROM music_servers s "
        "WHERE s.is_default AND t.server_id = s.server_id "
        "AND t.provider_track_id LIKE %s",
        (len(_MIG_TMP_PREFIX) + 1, _MIG_TMP_PREFIX + '%'),
    )

    _apply_new_meta(cur, new_meta)
    _clear_default_server_artist_map(cur)

    _write_provider_to_default_server(
        cur, target_type, target_creds, selected_libraries=selected_libraries
    )
    _purge_media_keys_from_app_config(cur)

    cur.execute(
        "UPDATE migration_session SET status = 'completed', completed_at = NOW() WHERE id = %s",
        (session_id,),
    )

    # item_ids never moved, so every similarity index still points at the right songs.
    return False


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
