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
        host=getattr(config, 'POSTGRES_HOST', 'localhost'),
        port=getattr(config, 'POSTGRES_PORT', '5432'),
        user=getattr(config, 'POSTGRES_USER', 'postgres'),
        password=getattr(config, 'POSTGRES_PASSWORD', ''),
        dbname=getattr(config, 'POSTGRES_DB', 'postgres'),
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

    _apply_new_meta(cur, new_meta)

    index_rebuild_needed = _rewrite_index_id_maps(cur, mapping)

    _clear_index_tables(cur)

    _write_provider_to_app_config(
        cur, target_type, target_creds, selected_libraries=selected_libraries
    )

    cur.execute(
        "UPDATE migration_session SET status = 'completed', completed_at = NOW() WHERE id = %s",
        (session_id,),
    )

    return index_rebuild_needed


_CREDS_TO_CONFIG = {
    'jellyfin': {'url': 'JELLYFIN_URL', 'user_id': 'JELLYFIN_USER_ID', 'token': 'JELLYFIN_TOKEN'},
    'emby': {'url': 'EMBY_URL', 'user_id': 'EMBY_USER_ID', 'token': 'EMBY_TOKEN'},
    'navidrome': {
        'url': 'NAVIDROME_URL',
        'user': 'NAVIDROME_USER',
        'password': 'NAVIDROME_PASSWORD',
    },
    'lyrion': {'url': 'LYRION_URL'},
}


def _write_provider_to_app_config(cur, target_type, target_creds, selected_libraries=None):
    import config as cfg

    cur.execute("SELECT pg_advisory_lock(726354821)")
    try:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'app_config')"
        )
        if not cur.fetchone()[0]:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS app_config ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
    finally:
        cur.execute("SELECT pg_advisory_unlock(726354821)")

    values = {'MEDIASERVER_TYPE': target_type}
    key_map = _CREDS_TO_CONFIG.get(target_type, {})
    for cred_key, config_key in key_map.items():
        val = target_creds.get(cred_key)
        if val is not None:
            values[config_key] = str(val)

    for key, value in values.items():
        cur.execute(
            "INSERT INTO app_config (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, "
            "updated_at = CURRENT_TIMESTAMP",
            (_sanitize_text(key), _sanitize_text(value)),
        )

    cleaned = [str(name).strip() for name in (selected_libraries or []) if str(name).strip()]
    cleaned = [name for name in cleaned if ',' not in name]
    ml_value = ','.join(cleaned)
    if ml_value:
        cur.execute(
            "INSERT INTO app_config (key, value) VALUES ('MUSIC_LIBRARIES', %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, "
            "updated_at = CURRENT_TIMESTAMP",
            (_sanitize_text(ml_value),),
        )
    else:
        cur.execute("DELETE FROM app_config WHERE key = 'MUSIC_LIBRARIES'")

    obsolete = cfg.MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE.get(target_type, [])
    if obsolete:
        cur.execute(
            "DELETE FROM app_config WHERE key = ANY(%s)",
            (list(obsolete),),
        )


def _post_commit_reload(redis):
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
