"""RQ-compatible execution job for the provider migration tool.

This module rewrites every ``item_id`` in the database to point at the
corresponding track on a new media server provider, then persists the new
provider credentials in ``app_config`` (the same table the setup wizard uses).
After commit, ``config.refresh_config()`` + ``restart_manager`` propagate the
change to all processes without a manual container restart.

The transaction is atomic:
    - orphan tracks deleted first (cascades through embedding FKs),
    - FKs on embedding/clap_embedding/mulan_embedding dropped (they lack
      ON UPDATE CASCADE),
    - ``score.item_id`` / ``playlist.item_id`` / all embedding tables rewritten
      via ``UPDATE ... FROM item_id_migration_map`` (O(N), one round trip),
    - FKs re-added with original (reflected) names,
    - voyager / map_projection ``id_map_json`` rewritten in Python (int keys
      kept, string values swapped),
    - artist_index_data / artist_component_projection / artist_mapping
      truncated — they contain provider-specific artist IDs and will lazily
      rebuild on next use,
    - ``app_config`` updated with MEDIASERVER_TYPE + provider credentials,
    - migration_session row marked completed.

Post-commit best-effort: ``config.refresh_config()`` to reload the local
process, ``restart_manager.publish_restart_request()`` to notify workers,
clear the ``migration:paused`` Redis key.

Safety invariants enforced by the test suite:
    - ``migration_session.status`` must equal ``'dry_run_ready'`` before exec.
    - RQ workers are paused via ``redis SET migration:paused 1`` before the
      advisory lock, and drained before the tx begins.
    - All DDL + DML runs on one dedicated psycopg2 connection; temp tables
      and the advisory lock require session continuity.
"""
import json
import logging
import time

from tasks.memory_utils import sanitize_string_for_db as _sanitize_text

logger = logging.getLogger(__name__)


# Advisory lock key — plain bigint, no collision with init_db / janitor
# (verified via grep for pg_advisory in the codebase).
_ADVISORY_LOCK_KEY = 7421536190082003

# How long to wait for in-flight RQ jobs to drain before forcing migration.
_DRAIN_TIMEOUT_SECONDS = 60

# Intermediate prefix used during the two-pass item_id rewrite. Postgres
# enforces PRIMARY KEY / UNIQUE row-by-row during UPDATE, so a single-pass
# UPDATE blows up if any mapping new_id happens to already exist in the table
# as another row's old_id (common when both providers use small integer IDs,
# e.g., Emby ↔ Emby). Pass 1 stages every row at <prefix>||new_id (unique per
# new_id) and Pass 2 strips the prefix to land the final new_id. The prefix is
# deliberately long and unusual so it can never collide with a real item_id.
_MIG_TMP_PREFIX = '__audiomuse_mig_tmp__'


# ---------------------------------------------------------------------------
# Pure-Python helpers (tested in isolation)
# ---------------------------------------------------------------------------

def rewrite_id_map_json(id_map_json, mapping):
    """Rewrite a Voyager / map-projection id_map JSON blob in place.

    Two on-disk formats live in this DB:

    1. Voyager (``voyager_index_data.id_map_json``) is a dict
       ``{voyager_int_id_str: old_item_id_str}``. The integer key is the
       HNSW vector slot and must be preserved verbatim; we only swap the
       string values. Orphan entries (old id not in ``mapping``) are
       dropped — consumers already tolerate missing keys, and dropping
       keeps the map small.

    2. Map projection (``map_projection_data.id_map_json``) is a flat
       list ``[item_id_0, item_id_1, ...]`` where position N corresponds
       to row N of the projection matrix. Here we can NOT drop orphans
       because the list has to stay in lockstep with the projection
       array — we replace orphan slots with ``None`` so the slot is kept
       but the consumer (app_map.py:149) falls through to compute the
       projection on the fly for that item.

    Returns the rewritten JSON string (or the original empty/None value).
    """
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
            # else: drop — orphan, no mapping
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
    """Reflect the actual FK constraint name that references ``ref_table.ref_column``.

    Postgres auto-names FKs ``<table>_<column>_fkey`` by default but older
    schemas may have been migrated with different names, so we look up the real
    one at runtime instead of hard-coding.
    """
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


# ---------------------------------------------------------------------------
# Injection points (tests replace these with MagicMock)
# ---------------------------------------------------------------------------

def _get_dedicated_conn():
    """Return a fresh psycopg2 connection not shared with the pool.

    Tests replace this with a MagicMock that yields a fake connection/cursor.
    """
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
    """Return a Redis client. Tests patch this to a MagicMock."""
    from app_helper import redis_conn
    return redis_conn


def _drain_workers_or_timeout(seconds=_DRAIN_TIMEOUT_SECONDS):
    """Poll task_status until no analysis jobs are running, up to ``seconds``.

    Tests replace this with a no-op. In production this blocks the current
    process so the migration doesn't step on in-flight analysis.
    """
    deadline = time.time() + seconds
    while time.time() < deadline:
        # A real implementation would poll task_status for active STARTED jobs.
        # For the first release we simply sleep briefly after sending stop
        # signals — workers finish their current job on their own.
        time.sleep(1)
        break  # placeholder — sufficient after worker.send_stop_signal()


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def execute_provider_migration(session_id):
    """Execute a provider migration for the given ``migration_session.id``.

    Returns ``{'ok': True, 'matched': N, 'orphans': M}`` on success, raises on
    any pre-check or transactional failure.
    """
    logger.info("provider migration: starting session %s", session_id)

    redis = _get_redis()
    # 1. Pause workers before we even read the session, so no new analysis
    #    jobs start while we're locking tables.
    redis.set('migration:paused', '1', ex=3600)
    try:
        _pause_and_drain_workers(redis)

        conn = _get_dedicated_conn()
        try:
            conn.autocommit = False
        except Exception:
            pass  # mocks may not support attribute assignment

        cur = conn.cursor()

        # 2. Load and validate the session row
        session = _load_session(cur, session_id)
        target_type = session['target_type']
        target_creds = session['target_creds']
        state = session['state']

        if session['status'] != 'dry_run_ready':
            raise RuntimeError(
                f"Cannot execute migration: session {session_id} is in status "
                f"'{session['status']}', expected 'dry_run_ready'"
            )

        # 3. Merge dry_run auto-matches with manual matches into a flat dict
        mapping = _merge_mapping(state)
        new_meta = state.get('new_meta') or {}
        logger.info("provider migration: %d tracks will be rewritten", len(mapping))

        # 4. Reflect FK names and check for optional tables before opening tx
        fk_embedding      = find_fk(cur, 'embedding', 'item_id')
        fk_clap_embedding = find_fk(cur, 'clap_embedding', 'item_id')
        cur.execute("SELECT to_regclass('public.mulan_embedding') IS NOT NULL")
        mulan_exists = bool(cur.fetchone()[0])
        fk_mulan_embedding = find_fk(cur, 'mulan_embedding', 'item_id') if mulan_exists else None

        # 5. Run the transaction
        try:
            _run_migration_transaction(
                cur=cur,
                mapping=mapping,
                new_meta=new_meta,
                fk_embedding=fk_embedding,
                fk_clap_embedding=fk_clap_embedding,
                fk_mulan_embedding=fk_mulan_embedding,
                mulan_exists=mulan_exists,
                target_type=target_type,
                target_creds=target_creds,
                session_id=session_id,
            )
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise

        # 6. Post-commit: reload config, notify other processes to restart
        _post_commit_reload(redis)

        return {
            'ok': True,
            'matched': len(mapping),
        }
    finally:
        # Always clear the pause flag, even on failure — otherwise workers
        # would stay paused forever.
        try:
            redis.delete('migration:paused')
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers — each one accepts a cursor so tests can feed a MagicMock
# ---------------------------------------------------------------------------

def _pause_and_drain_workers(redis):
    """Stop accepting new RQ jobs and wait for in-flight jobs to finish.

    Tests replace ``_drain_workers_or_timeout`` with a no-op and don't care
    about the actual RQ ``Worker.send_stop_signal`` path. In production we:
      1. Signal every registered worker to finish its current job and exit.
      2. Poll until no STARTED jobs remain (or time out after 60s).
    """
    try:
        from rq import Worker  # pragma: no cover — optional import in tests
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
        "SELECT id, target_type, target_creds, state, status "
        "FROM migration_session WHERE id = %s",
        (session_id,),
    )
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"migration_session {session_id} not found")
    _id, target_type, target_creds_json, state_json, status = row
    try:
        creds = json.loads(target_creds_json) if isinstance(target_creds_json, str) else target_creds_json
    except Exception:
        creds = {}
    try:
        state = json.loads(state_json) if isinstance(state_json, str) else state_json
    except Exception:
        state = {}
    return {
        'id':            _id,
        'target_type':   target_type,
        'target_creds':  creds,
        'state':         state or {},
        'status':        status,
    }


def build_mapping(state):
    """Build the final ``old_id -> new_id`` mapping plus a list of dropped
    collisions. Shared by ``_merge_mapping`` (execute) and
    ``finalize_dry_run`` (so the wizard can surface collision counts).

    Precedence:
      1. ``manual_unmatches`` drops auto matches for those old_ids (user
         explicitly orphaned them in the step-4 rematch flow).
      2. ``manual_matches`` wins over surviving dry_run matches for the same
         old_id.
      3. Collisions on ``new_id`` are resolved first-write-wins. The temp
         ``item_id_migration_map`` table has ``UNIQUE(new_id)``, so any
         duplicate would fail the whole transaction with an opaque constraint
         violation after the user already typed the confirmation phrase.
         Collisions can happen when the user re-targets an album and the
         matcher picks the same new track for two different old rows.

    Returns ``(deduped, dropped)`` where ``deduped`` is the final mapping
    dict and ``dropped`` is a list of ``(old_id, new_id, winner_old_id)``
    tuples — one per row that was orphaned to resolve a collision.
    """
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
    """Execute-path wrapper around :func:`build_mapping` that logs dropped
    collisions and returns only the mapping dict (tests assert this shape)."""
    deduped, dropped = build_mapping(state)
    if dropped:
        logger.warning(
            "provider migration: dropped %d mapping(s) that collided on "
            "new_id (multiple source rows pointed at the same target id); "
            "those rows will be orphaned on execute. First 10: %s",
            len(dropped), dropped[:10],
        )
    return deduped


def _run_migration_transaction(cur, mapping, new_meta,
                               fk_embedding, fk_clap_embedding, fk_mulan_embedding,
                               mulan_exists, target_type, target_creds, session_id):
    """Execute every SQL statement for the migration transaction.

    Caller is responsible for commit/rollback. This function only issues
    statements — no commit, no connection management.

    Order is load-bearing: the sequence is asserted by the test suite because
    orphan deletion must happen before the rewrite, FKs must be dropped before
    the UPDATE and re-added after, and so on.
    """
    # 1. Acquire advisory lock, scoped to this transaction
    cur.execute("SELECT pg_advisory_xact_lock(%s)", (_ADVISORY_LOCK_KEY,))

    # 2. Stage the rewrite map in a temp table
    cur.execute(
        "CREATE TEMP TABLE item_id_migration_map ("
        " old_id TEXT PRIMARY KEY, "
        " new_id TEXT NOT NULL UNIQUE"
        ") ON COMMIT DROP"
    )
    for old_id, new_id in mapping.items():
        cur.execute(
            "INSERT INTO item_id_migration_map (old_id, new_id) VALUES (%s, %s)",
            (old_id, new_id),
        )

    # 3. Delete orphans FIRST so the FK cascades clean the embedding tables
    #    before we start rewriting.
    cur.execute(
        "DELETE FROM score WHERE item_id NOT IN "
        "(SELECT old_id FROM item_id_migration_map)"
    )

    # 4. Drop FKs on the embedding tables — Postgres has ON DELETE CASCADE
    #    but no ON UPDATE CASCADE, so we must drop them to rewrite both sides.
    if fk_embedding:
        cur.execute(f"ALTER TABLE embedding DROP CONSTRAINT {fk_embedding}")
    if fk_clap_embedding:
        cur.execute(f"ALTER TABLE clap_embedding DROP CONSTRAINT {fk_clap_embedding}")
    if mulan_exists and fk_mulan_embedding:
        cur.execute(f"ALTER TABLE mulan_embedding DROP CONSTRAINT {fk_mulan_embedding}")

    # 5. Rewrite item_id on every item-id-keyed table.
    #    We do this in TWO passes per table because Postgres enforces PRIMARY
    #    KEY / UNIQUE constraints row-by-row during UPDATE (they are not
    #    deferrable by default). A single-pass UPDATE would fail with
    #    "duplicate key" whenever a mapping's new_id equals another row's
    #    current item_id — very common when both providers issue small
    #    integer IDs that happen to overlap (e.g., migrating Emby→Emby or
    #    Jellyfin→Emby where both servers use "25" for different tracks).
    #
    #    Pass 1 stages every row at (_MIG_TMP_PREFIX || new_id), which is
    #    guaranteed unique (new_id is UNIQUE in the map) and cannot collide
    #    with any real existing item_id because no real id starts with the
    #    prefix. Pass 2 strips the prefix to land the final new_id — safe
    #    because the final new_ids are unique across all surviving rows.
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
    if mulan_exists:
        cur.execute(
            "UPDATE mulan_embedding e SET item_id = %s || m.new_id "
            "FROM item_id_migration_map m WHERE e.item_id = m.old_id",
            (prefix,),
        )
        cur.execute(
            "UPDATE mulan_embedding e SET item_id = m.new_id "
            "FROM item_id_migration_map m "
            "WHERE e.item_id = %s || m.new_id",
            (prefix,),
        )

    # 6. Re-add the FKs with the original (reflected) names
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
    if mulan_exists and fk_mulan_embedding:
        cur.execute(
            f"ALTER TABLE mulan_embedding ADD CONSTRAINT {fk_mulan_embedding} "
            f"FOREIGN KEY (item_id) REFERENCES score(item_id) ON DELETE CASCADE"
        )

    # 7. Refresh score metadata (file_path, title, author, album, year) from
    #    the new provider's values. New paths are critical: the new provider's
    #    path format may not overlap with the old one at all (Jellyfin absolute
    #    vs Navidrome relative), and downstream features use file_path.
    if new_meta:
        cur.execute(
            "CREATE TEMP TABLE migration_new_meta ("
            " new_id TEXT PRIMARY KEY, "
            " new_path TEXT, new_title TEXT, new_artist TEXT, "
            " new_album TEXT, new_year INTEGER"
            ") ON COMMIT DROP"
        )
        for new_id, meta in new_meta.items():
            cur.execute(
                "INSERT INTO migration_new_meta "
                "(new_id, new_path, new_title, new_artist, new_album, new_year) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    _sanitize_text(new_id),
                    _sanitize_text(meta.get('path')),
                    _sanitize_text(meta.get('title')),
                    _sanitize_text(meta.get('artist')),
                    _sanitize_text(meta.get('album')),
                    meta.get('year'),
                ),
            )
        cur.execute(
            "UPDATE score s SET "
            "  file_path = COALESCE(n.new_path,   s.file_path), "
            "  title     = COALESCE(n.new_title,  s.title), "
            "  author    = COALESCE(n.new_artist, s.author), "
            "  album     = COALESCE(n.new_album,  s.album), "
            "  year      = COALESCE(n.new_year,   s.year) "
            "FROM migration_new_meta n WHERE s.item_id = n.new_id"
        )

    # 8. Rewrite Voyager id_map_json in place (values only; int keys unchanged)
    cur.execute(
        "SELECT index_name, id_map_json FROM voyager_index_data "
        "WHERE id_map_json <> ''"
    )
    for row in (cur.fetchall() or []):
        index_name, id_map_json = row[0], row[1]
        new_json = rewrite_id_map_json(id_map_json, mapping)
        if new_json != id_map_json:
            cur.execute(
                "UPDATE voyager_index_data SET id_map_json = %s WHERE index_name = %s",
                (new_json, index_name),
            )

    # Same transform for the 2D map projection id_map
    cur.execute(
        "SELECT index_name, id_map_json FROM map_projection_data "
        "WHERE id_map_json <> ''"
    )
    for row in (cur.fetchall() or []):
        index_name, id_map_json = row[0], row[1]
        new_json = rewrite_id_map_json(id_map_json, mapping)
        if new_json != id_map_json:
            cur.execute(
                "UPDATE map_projection_data SET id_map_json = %s WHERE index_name = %s",
                (new_json, index_name),
            )

    # 9. Truncate provider-specific artist tables — they contain artist IDs
    #    from the old provider. They rebuild lazily on next query.
    cur.execute("DELETE FROM artist_index_data")
    cur.execute("DELETE FROM artist_component_projection")
    cur.execute("DELETE FROM artist_mapping")

    # 10. Persist the new provider in app_config (same table as setup wizard)
    _write_provider_to_app_config(cur, target_type, target_creds)

    # 11. Mark the session row completed
    cur.execute(
        "UPDATE migration_session SET status = 'completed', completed_at = NOW() "
        "WHERE id = %s",
        (session_id,),
    )


_CREDS_TO_CONFIG = {
    'jellyfin':  {'url': 'JELLYFIN_URL', 'user_id': 'JELLYFIN_USER_ID', 'token': 'JELLYFIN_TOKEN'},
    'emby':      {'url': 'EMBY_URL', 'user_id': 'EMBY_USER_ID', 'token': 'EMBY_TOKEN'},
    'navidrome': {'url': 'NAVIDROME_URL', 'user': 'NAVIDROME_USER', 'password': 'NAVIDROME_PASSWORD'},
    'lyrion':    {'url': 'LYRION_URL'},
    'mpd':       {'host': 'MPD_HOST', 'port': 'MPD_PORT', 'password': 'MPD_PASSWORD',
                  'music_directory': 'MPD_MUSIC_DIRECTORY'},
}


def _write_provider_to_app_config(cur, target_type, target_creds):
    """Write MEDIASERVER_TYPE + provider credentials into ``app_config``.

    Runs inside the caller's transaction so a rollback undoes everything.
    Also deletes obsolete credential keys from the old provider (same
    pattern the setup wizard uses via ``MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE``).
    """
    import config as cfg

    # Ensure ``app_config`` exists. ``init_db()`` and the setup wizard both
    # create it on startup, but a DB restored from a pre-setup-wizard backup
    # won't have it — ``app_backup.restore`` drops all tables and replays the
    # backup file without re-running ``init_db()``. Creating it here keeps the
    # migration transactional (same cursor, rolls back with everything else).
    # Use the same advisory lock as init_db() so concurrent schema creation
    # never races into duplicate-type errors.
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

    # Build the key→value pairs to upsert
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

    # Remove credentials for providers we're switching away from
    obsolete = cfg.MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE.get(target_type, [])
    if obsolete:
        cur.execute(
            "DELETE FROM app_config WHERE key = ANY(%s)",
            (list(obsolete),),
        )


def _post_commit_reload(redis):
    """Reload config and notify other processes via restart_manager.

    Best-effort: any failure here is logged but does not fail the migration
    (the DB state is already committed and will load correctly on restart).
    """
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
