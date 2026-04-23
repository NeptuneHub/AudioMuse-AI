# app_helper.py
import json
import logging
import os
import time
import psycopg2
from psycopg2.extras import DictCursor
import numpy as np
from flask import g

# RQ imports
from redis import Redis
from rq import Queue
from rq.job import Job, JobStatus
from rq.exceptions import NoSuchJobError

# Import from main app
# We import 'app' to use its context (e.g., for logging)
# Note: get_db, redis_conn will now be defined *in this file*.

# Import configuration
from config import DATABASE_URL, REDIS_URL

# Import RQ specifics
from rq.command import send_stop_job_command

logger = logging.getLogger(__name__)
# Import app object after it's defined to break circular dependency
# Avoid importing the Flask `app` object here to prevent circular imports.
# Use the module-level `logger` defined above for logging instead of `app.logger`.

# In-memory cache for the precomputed 2D map projection (optional)
MAP_PROJECTION_CACHE = None

# In-memory cache for the precomputed 2D artist component projections
ARTIST_PROJECTION_CACHE = None

# --- Constants ---
MAX_LOG_ENTRIES_STORED = 10 # Max number of recent log entries to store in the database per task

# --- RQ Setup ---
# Enhanced Redis connection settings for remote server stability:
# - socket_connect_timeout: max time to establish connection
# - socket_timeout: max time for socket operations (read/write)
# - socket_keepalive: enables TCP keepalive to prevent idle connection drops
# - health_check_interval: seconds between health checks on idle connections
# - retry_on_timeout: automatically retry on timeout errors
redis_conn = Redis.from_url(
    REDIS_URL, 
    socket_connect_timeout=30,
    socket_timeout=60,
    socket_keepalive=True,
    health_check_interval=30,
    retry_on_timeout=True
)
# FIX: result_ttl removed - caused jobs to disappear from Redis before monitor_and_clear_jobs could track them
# This was breaking the throttle mechanism causing all jobs to launch at once
rq_queue_high = Queue('high', connection=redis_conn, default_timeout=-1) # High priority for main tasks
rq_queue_default = Queue('default', connection=redis_conn, default_timeout=-1) # Default queue for sub-tasks

# --- Database Setup (PostgreSQL) ---
def get_db():
    if 'db' not in g:
        try:
            g.db = psycopg2.connect(
                DATABASE_URL,
                connect_timeout=30,        # Time to establish connection (increased from 15)
                keepalives_idle=600,       # Start keepalives after 10 min idle
                keepalives_interval=30,    # Send keepalive every 30 sec
                keepalives_count=3,        # 3 failed keepalives = dead connection
                options='-c statement_timeout=600000'  # 10 min query timeout (600 seconds)
            )
        except psycopg2.OperationalError as e:
            logger.error(f"Failed to connect to database: {e}")
            raise # Re-raise to ensure the operation that needed the DB fails clearly
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    with db.cursor() as cur:
        # Serialize concurrent init_db() runs across gunicorn workers/containers.
        # Multiple workers racing on CREATE EXTENSION / CREATE OR REPLACE FUNCTION
        # causes Postgres "tuple concurrently updated" errors on pg_proc/pg_extension.
        # A session-level advisory lock forces other workers to wait here.
        # The key is an arbitrary stable bigint specific to this app's init.
        # Safety: session-level advisory locks are auto-released by Postgres
        # when the connection ends (normal close, crash, kill, or network drop),
        # so this lock can NEVER leak permanently even if init_db() raises.
        cur.execute("SELECT pg_advisory_lock(726354821)")
        try:
            # Enable extensions to fix and assist in searches
            cur.execute('CREATE EXTENSION IF NOT EXISTS unaccent')
            cur.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
            # Create 'score' table
            cur.execute("CREATE TABLE IF NOT EXISTS score (item_id TEXT PRIMARY KEY, title TEXT, author TEXT, album TEXT, album_artist TEXT, tempo REAL, key TEXT, scale TEXT, mood_vector TEXT)")
            # Add 'energy' column if not exists
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'energy')")
            if not cur.fetchone()[0]:
                logger.info("Adding 'energy' column to 'score' table.")
                cur.execute("ALTER TABLE score ADD COLUMN energy REAL")
            # Add 'other_features' column if not exists
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'other_features')")
            if not cur.fetchone()[0]:
                logger.info("Adding 'other_features' column to 'score' table.")
                cur.execute("ALTER TABLE score ADD COLUMN other_features TEXT")
            # Add 'album' column if not exists
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'album')")
            if not cur.fetchone()[0]:
                logger.info("Adding 'album' column to 'score' table.")
                cur.execute("ALTER TABLE score ADD COLUMN album TEXT")
            # Add 'album_artist' column if not exists
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'album_artist')")
            if not cur.fetchone()[0]:
                logger.info("Adding 'album_artist' column to 'score' table.")
                cur.execute("ALTER TABLE score ADD COLUMN album_artist TEXT")
            # Add 'year' column if not exists
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'year')")
            if not cur.fetchone()[0]:
                logger.info("Adding 'year' column to 'score' table.")
                cur.execute("ALTER TABLE score ADD COLUMN year INTEGER")
            # Add 'rating' column if not exists
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'rating')")
            if not cur.fetchone()[0]:
                logger.info("Adding 'rating' column to 'score' table.")
                cur.execute("ALTER TABLE score ADD COLUMN rating INTEGER")
            # Add 'file_path' column if not exists
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'file_path')")
            if not cur.fetchone()[0]:
                logger.info("Adding 'file_path' column to 'score' table.")
                cur.execute("ALTER TABLE score ADD COLUMN file_path TEXT")
        
            # Ensure we have a searchable, accent-stripped `search_u` column.
            # Postgres does not allow generated columns to call `unaccent()` (it's not marked immutable),
            # so we store the value in a normal column and keep it in sync via trigger.
            cur.execute("SELECT is_generated FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'search_u'")
            row = cur.fetchone()
            search_u_generated = (row and row[0] == 'ALWAYS')

            if search_u_generated:
                logger.info("Dropping legacy generated 'search_u' column to replace it with a trigger-updated column.")
                cur.execute("ALTER TABLE score DROP COLUMN IF EXISTS search_u")
                row = None

            # Create plain `search_u` column if missing
            if not row:
                logger.info("Adding 'search_u' column to 'score' table.")
                cur.execute("ALTER TABLE score ADD COLUMN search_u TEXT")

            # Create helper function for accent stripping (safe to run multiple times)
            cur.execute("CREATE OR REPLACE FUNCTION immutable_unaccent(text) RETURNS text LANGUAGE sql IMMUTABLE AS $$ SELECT public.unaccent($1) $$;")

            # Create/replace trigger function to keep search_u in sync
            cur.execute("""
                CREATE OR REPLACE FUNCTION score_search_u_sync() RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN
                    NEW.search_u := lower(immutable_unaccent(concat_ws(' ', NEW.title, NEW.author, NEW.album)));
                    RETURN NEW;
                END;
                $$;
            """)

            # Attach trigger to update search_u on insert/update
            # Note: Postgres doesn't support CREATE TRIGGER IF NOT EXISTS, so we drop and recreate.
            cur.execute("DROP TRIGGER IF EXISTS score_search_u_sync_trigger ON score")
            cur.execute("""
                CREATE TRIGGER score_search_u_sync_trigger
                BEFORE INSERT OR UPDATE ON score
                FOR EACH ROW
                EXECUTE FUNCTION score_search_u_sync();
            """)

            # Backfill existing rows (ensures proper value for pre-existing data)
            # This is safe to run repeatedly.
            cur.execute("UPDATE score SET search_u = lower(immutable_unaccent(concat_ws(' ', title, author, album))) WHERE search_u IS NULL")

            # Create index on 'score' to assist in searches
            cur.execute("CREATE INDEX IF NOT EXISTS score_search_u_trgm ON score USING gin (search_u gin_trgm_ops)")

            # Create 'playlist' table
            cur.execute("CREATE TABLE IF NOT EXISTS playlist (id SERIAL PRIMARY KEY, playlist_name TEXT, item_id TEXT, title TEXT, author TEXT, UNIQUE (playlist_name, item_id))")
            # Create 'task_status' table
            cur.execute("CREATE TABLE IF NOT EXISTS task_status (id SERIAL PRIMARY KEY, task_id TEXT UNIQUE NOT NULL, parent_task_id TEXT, task_type TEXT NOT NULL, sub_type_identifier TEXT, status TEXT, progress INTEGER DEFAULT 0, details TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            # Migrate 'start_time' and 'end_time' columns
            for col_name in ['start_time', 'end_time']:
                cur.execute("SELECT data_type FROM information_schema.columns WHERE table_name = 'task_status' AND column_name = %s", (col_name,))
                if not cur.fetchone(): cur.execute(f"ALTER TABLE task_status ADD COLUMN {col_name} DOUBLE PRECISION")
            # Create 'task_history' table — a small, persistent log of the last
            # completed/cancelled MAIN tasks. Survives the global Cancel button
            # which wipes `task_status`. Capped to the most recent 10 rows.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS task_history (
                    id SERIAL PRIMARY KEY,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    task_id TEXT,
                    task_type TEXT,
                    status TEXT,
                    duration_seconds DOUBLE PRECISION,
                    note TEXT
                )
            """)
            # Create 'embedding' table
            cur.execute("CREATE TABLE IF NOT EXISTS embedding (item_id TEXT PRIMARY KEY, FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE)")
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'embedding' AND column_name = 'embedding')")
            if not cur.fetchone()[0]: cur.execute("ALTER TABLE embedding ADD COLUMN embedding BYTEA")
            # Create 'clap_embedding' table for CLAP text search embeddings
            cur.execute("CREATE TABLE IF NOT EXISTS clap_embedding (item_id TEXT PRIMARY KEY, FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE)")
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'clap_embedding' AND column_name = 'embedding')")
            if not cur.fetchone()[0]: cur.execute("ALTER TABLE clap_embedding ADD COLUMN embedding BYTEA")
            # Create 'mulan_embedding' table only if MuLan is enabled
            from config import MULAN_ENABLED
            if MULAN_ENABLED:
                cur.execute("CREATE TABLE IF NOT EXISTS mulan_embedding (item_id TEXT PRIMARY KEY, FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE)")
                cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'mulan_embedding' AND column_name = 'embedding')")
                if not cur.fetchone()[0]: cur.execute("ALTER TABLE mulan_embedding ADD COLUMN embedding BYTEA")
            # Create 'voyager_index_data' table
            cur.execute("CREATE TABLE IF NOT EXISTS voyager_index_data (index_name VARCHAR(255) PRIMARY KEY, index_data BYTEA NOT NULL, id_map_json TEXT NOT NULL, embedding_dimension INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            # Create 'clap_index_data' table for stored CLAP text search indexes
            cur.execute("CREATE TABLE IF NOT EXISTS clap_index_data (index_name VARCHAR(255) PRIMARY KEY, index_data BYTEA NOT NULL, id_map_json TEXT NOT NULL, embedding_dimension INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            # Create 'artist_index_data' table for artist GMM-based HNSW index
            cur.execute("CREATE TABLE IF NOT EXISTS artist_index_data (index_name VARCHAR(255) PRIMARY KEY, index_data BYTEA NOT NULL, artist_map_json TEXT NOT NULL, gmm_params_json TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            # Create 'map_projection_data' table for precomputed 2D map projections
            cur.execute("CREATE TABLE IF NOT EXISTS map_projection_data (index_name VARCHAR(255) PRIMARY KEY, projection_data BYTEA NOT NULL, id_map_json TEXT NOT NULL, embedding_dimension INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            # Create 'artist_component_projection' table for precomputed 2D artist component projections
            cur.execute("CREATE TABLE IF NOT EXISTS artist_component_projection (index_name VARCHAR(255) PRIMARY KEY, projection_data BYTEA NOT NULL, artist_component_map_json TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            # Create 'cron' table to hold scheduled jobs (very small and simple)
            cur.execute("CREATE TABLE IF NOT EXISTS cron (id SERIAL PRIMARY KEY, name TEXT, task_type TEXT NOT NULL, cron_expr TEXT NOT NULL, enabled BOOLEAN DEFAULT FALSE, last_run DOUBLE PRECISION, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            # Create 'audiomuse_users' table. Every account (including the
            # install-time admin) lives here. 'role' is 'admin' or 'user'.
            cur.execute("CREATE TABLE IF NOT EXISTS audiomuse_users (id SERIAL PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            # Lightweight migration for installs that already have the table without a role column.
            cur.execute("ALTER TABLE audiomuse_users ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'user'")
            # Create 'dashboard_stats' singleton table (id fixed to 1) that holds
            # precomputed content/library aggregates and index counts. Refreshed
            # at app startup and hourly by a background job so the dashboard
            # does not have to scan the whole `score` table on every poll.
            cur.execute(
                "CREATE TABLE IF NOT EXISTS dashboard_stats ("
                "id INTEGER PRIMARY KEY, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "content JSONB NOT NULL DEFAULT '{}'::jsonb, "
                "indexes JSONB NOT NULL DEFAULT '[]'::jsonb, "
                "CONSTRAINT dashboard_stats_singleton CHECK (id = 1))"
            )
            # Ensure older restored DBs still have the primary key constraint.
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.table_constraints "
                "WHERE table_name = 'dashboard_stats' AND constraint_type = 'PRIMARY KEY'"
            )
            row = cur.fetchone()
            if row and row[0] == 0:
                logger.info("Cleaning dashboard_stats and adding missing primary key constraint to dashboard_stats.id")
                cur.execute("DELETE FROM dashboard_stats")
                cur.execute("ALTER TABLE dashboard_stats ADD CONSTRAINT dashboard_stats_pkey PRIMARY KEY (id)")
            # Create 'artist_mapping' table to map artist names to media server artist IDs
            cur.execute("CREATE TABLE IF NOT EXISTS artist_mapping (artist_name TEXT PRIMARY KEY, artist_id TEXT)")
            # Create application configuration table to persist setup values.
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'app_config')"
            )
            if not cur.fetchone()[0]:
                cur.execute(
                    "CREATE TABLE app_config ("
                    "key TEXT PRIMARY KEY, value TEXT NOT NULL, "
                    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                )
            # Create 'alchemy_anchors' table to persist named user anchors for reuse
            cur.execute("CREATE TABLE IF NOT EXISTS alchemy_anchors (id SERIAL PRIMARY KEY, name TEXT UNIQUE NOT NULL, centroid JSONB NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            # Provider migration tool: wizard session state (one row per migration attempt)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS migration_session (
                    id           SERIAL PRIMARY KEY,
                    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    status       TEXT NOT NULL DEFAULT 'in_progress',
                    source_type  TEXT NOT NULL,
                    target_type  TEXT NOT NULL,
                    target_creds TEXT NOT NULL,
                    state        JSONB NOT NULL DEFAULT '{}'
                )
            """)
            # Create 'text_search_queries' table for precomputed CLAP text search queries
            cur.execute("""
                CREATE TABLE IF NOT EXISTS text_search_queries (
                    id SERIAL PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    score REAL NOT NULL,
                    rank INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(rank)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_text_search_queries_rank ON text_search_queries(rank)")
        
            # Insert default queries if table is empty
            cur.execute("SELECT COUNT(*) FROM text_search_queries")
            count = cur.fetchone()[0]
        
            if count == 0:
                default_queries = [
                    "female vocal romantic trap",
                    "synth indie pop raspy",
                    "sad hard rock male vocal",
                    "funk falsetto energetic",
                    "groovy sax blues",
                    "classical relaxed piano",
                    "belting jazz happy",
                    "tabla afrobeat fast-paced",
                    "harmonized vocals slow-paced electronica",
                    "autotuned gospel excited",
                    "breathy aggressive house",
                    "smooth folk mid-tempo",
                    "deep voice r&b dark",
                    "punk guitar angry",
                    "metal choir dreamy",
                    "chant reggae trumpet",
                    "high-pitched brass hip-hop",
                    "disco whispered drum machine",
                    "happy whispered indie pop",
                    "synth energetic raspy",
                    "rock slow-paced cello",
                    "falsetto jazz excited",
                    "r&b male vocal romantic",
                    "harmonized vocals dark trap",
                    "smooth blues sax",
                    "high-pitched fast-paced soul",
                    "female vocal sad hip-hop",
                    "congas aggressive soul",
                    "mid-tempo afrobeat autotuned",
                    "belting funk groovy",
                    "angry alternative breathy",
                    "gospel choir steelpan",
                    "viola relaxed folk",
                    "dreamy rhodes metal",
                    "acoustic guitar country chant",
                    "deep voice orchestra reggae",
                    "fast-paced synth progressive rock",
                    "hard rock raspy romantic",
                    "fast-paced electric guitar progressive rock",
                    "hard rock aggressive breathy",
                    "rock high-pitched energetic",
                    "autotuned energetic hip-hop",
                    "raspy fast-paced blues",
                    "belting electronica energetic",
                    "whispered indie pop aggressive",
                    "harmonized vocals aggressive synth",
                    "orchestra whispered romantic",
                    "belting mid-tempo progressive rock",
                    "autotuned pop mid-tempo",
                    "pop energetic synthesizer"
                ]
            
                for rank, query in enumerate(default_queries, start=1):
                    cur.execute("""
                        INSERT INTO text_search_queries (query_text, score, rank, created_at)
                        VALUES (%s, %s, %s, NOW())
                    """, (query, 1.0, rank))
            
                logger.info(f"Inserted {len(default_queries)} default DCLAP search queries")
        
            db.commit()
            # Release the advisory lock acquired at the top of init_db().
        finally:
            cur.execute("SELECT pg_advisory_unlock(726354821)")

# --- Status Constants ---
TASK_STATUS_PENDING = "PENDING"
TASK_STATUS_STARTED = "STARTED"
TASK_STATUS_PROGRESS = "PROGRESS"
TASK_STATUS_SUCCESS = "SUCCESS"
TASK_STATUS_FAILURE = "FAILURE"
TASK_STATUS_REVOKED = "REVOKED"

# --- DB Cleanup Utility ---
def clean_up_previous_main_tasks():
    """
    Cleans up all previous main tasks before a new one starts.
    - Archives tasks in SUCCESS state.
    - Archives stale tasks stuck in PENDING, STARTED, or PROGRESS states.
    - DELETES all child tasks associated with archived parent tasks to prevent DB bloat.
    A main task is identified by having a NULL parent_task_id.
    """
    db = get_db() # This now calls the function within this file
    cur = db.cursor(cursor_factory=DictCursor)
    logger.info("Starting cleanup of all previous main tasks.")
    
    non_terminal_statuses = (TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS, TASK_STATUS_SUCCESS)
    
    try:
        cur.execute("SELECT task_id, status, details, task_type, start_time, end_time FROM task_status WHERE status IN %s AND parent_task_id IS NULL", (non_terminal_statuses,))
        tasks_to_archive = cur.fetchall()

        archived_count = 0
        deleted_children_count = 0
        
        for task_row in tasks_to_archive:
            task_id = task_row['task_id']
            original_status = task_row['status']
            
            original_details_json = task_row['details']
            original_status_message = f"Task was in '{original_status}' state."

            original_details_dict = None
            if original_details_json:
                try:
                    original_details_dict = json.loads(original_details_json)
                    original_status_message = original_details_dict.get("status_message", original_status_message)
                except (json.JSONDecodeError, TypeError):
                     logger.warning(f"Could not parse original details for task {task_id} during archival.")

            # Record into persistent history BEFORE deleting children — the
            # note builder needs to query subtasks (e.g. tracks_analyzed).
            try:
                duration_s = None
                if task_row['start_time'] is not None:
                    end = task_row['end_time'] if task_row['end_time'] is not None else time.time()
                    duration_s = max(0.0, float(end) - float(task_row['start_time']))
                final_status = TASK_STATUS_SUCCESS if original_status == TASK_STATUS_SUCCESS else TASK_STATUS_REVOKED
                record_task_history(
                    task_id, task_row['task_type'], final_status,
                    duration_s, details=original_details_dict,
                )
            except Exception as e_hist:
                logger.debug(f"history record skipped during archive of {task_id}: {e_hist}")

            if original_status == TASK_STATUS_SUCCESS:
                archival_reason = "New main task started, old successful task archived."
            else:
                archival_reason = f"New main task started, stale task (status: {original_status}) has been archived."

            archived_details = {
                "log": [f"[Archived] {archival_reason}. Original summary: {original_status_message}"],
                "original_status_before_archival": original_status,
                "archival_reason": archival_reason
            }
            archived_details_json = json.dumps(archived_details)

            with db.cursor() as update_cur:
                # First, delete all child tasks to prevent DB bloat and avoid counting old tasks
                update_cur.execute(
                    "DELETE FROM task_status WHERE parent_task_id = %s",
                    (task_id,)
                )
                children_deleted = update_cur.rowcount
                deleted_children_count += children_deleted
                
                if children_deleted > 0:
                    logger.info(f"Deleted {children_deleted} child tasks for parent task {task_id}")
                
                # Then archive the parent task
                update_cur.execute(
                    "UPDATE task_status SET status = %s, details = %s, progress = 100, timestamp = NOW() WHERE task_id = %s AND status = %s",
                    (TASK_STATUS_REVOKED, archived_details_json, task_id, original_status)
                )
            archived_count += 1

        if archived_count > 0:
            db.commit()
            logger.info(f"Archived {archived_count} previous main tasks and deleted {deleted_children_count} child tasks.")
        else:
            logger.info("No previous main tasks found to clean up.")
    except Exception as e_main_clean:
        db.rollback()
        logger.error(f"Error during the main task cleanup process: {e_main_clean}")
    finally:
        cur.close()


# ---------------------------------------------------------------------------
# Task history (separate from task_status — survives the global Cancel button)
# ---------------------------------------------------------------------------

TASK_HISTORY_MAX_ROWS = 10


def _build_task_note(task_type, details_obj, db):
    """Build a short, human-readable note for a finished task.

    Looks at the ``details`` JSON we stored on the main task and, when needed,
    queries subtasks to compute a meaningful number (e.g. total songs analyzed
    across all album_analysis subtasks)."""
    if not isinstance(details_obj, dict):
        details_obj = {}
    t = (task_type or '').lower()

    try:
        if 'analysis' in t:
            # Prefer summing tracks_analyzed from album_analysis subtasks.
            try:
                with db.cursor() as cur:
                    cur.execute(
                        "SELECT details FROM task_status WHERE parent_task_id = %s AND status = 'SUCCESS'",
                        (details_obj.get('_task_id') or '',),
                    )
                    rows = cur.fetchall()
            except Exception:
                rows = []
            songs = 0
            for (d,) in rows or []:
                if not d:
                    continue
                try:
                    obj = json.loads(d)
                    if isinstance(obj, dict):
                        v = obj.get('tracks_analyzed')
                        if isinstance(v, (int, float)):
                            songs += int(v)
                except Exception:
                    continue
            if songs > 0:
                return f"Songs analyzed: {songs}"
            # Fallback to album-level info from the main task details.
            albums = details_obj.get('albums_completed') or details_obj.get('total_albums_processed')
            if albums:
                return f"Albums analyzed: {albums}"
            return ''

        if 'clean' in t:
            for k in ('tracks_deleted', 'orphans_removed', 'songs_cleaned',
                     'tracks_removed', 'deleted_count', 'cleaned_tracks'):
                v = details_obj.get(k)
                if isinstance(v, (int, float)):
                    return f"Songs cleaned: {int(v)}"
            return ''

        if 'cluster' in t:
            sampled = (details_obj.get('best_params') or {}).get('initial_subset_size') \
                if isinstance(details_obj.get('best_params'), dict) else None
            if sampled is None:
                sampled = details_obj.get('sampled_songs') or details_obj.get('num_sampled_songs')
            n_clusters = details_obj.get('num_playlists_created') or details_obj.get('num_clusters')
            parts = []
            if sampled:
                parts.append(f"sampled: {int(sampled)}")
            if n_clusters:
                parts.append(f"clusters: {int(n_clusters)}")
            return ' • '.join(parts)
    except Exception as e:
        logger.debug(f"task note builder failed for type={task_type}: {e}")
    return ''


def record_task_history(task_id, task_type, status, duration_seconds=None, note=None, details=None):
    """Insert a row into ``task_history`` and trim the table to the most
    recent ``TASK_HISTORY_MAX_ROWS`` entries.

    Safe to call from anywhere; never raises. ``details`` (dict or None) is
    used to build a default ``note`` when one is not provided explicitly.
    If a short note cannot be inferred, fall back to the task's final
    status_message or message text when available.

    The history table is treated as immutable per task_id: once a task has
    been recorded, we do not insert a second history row for the same task.
    """
    if not task_id:
        return
    try:
        db = get_db()
        # If no note was supplied, try to infer one from details.
        if note is None:
            details_obj = details if isinstance(details, dict) else {}
            # Pass task_id through so the analysis branch can query subtasks.
            details_obj = dict(details_obj)
            details_obj['_task_id'] = task_id
            note = _build_task_note(task_type, details_obj, db) or ''
            if not note:
                note = details_obj.get('status_message') or details_obj.get('message') or ''

        with db.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM task_history WHERE task_id = %s LIMIT 1",
                (task_id,)
            )
            if cur.fetchone():
                return
            cur.execute(
                """
                INSERT INTO task_history (task_id, task_type, status, duration_seconds, note)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (task_id, task_type, status, duration_seconds, note),
            )
            # Trim — keep only the most recent rows.
            cur.execute(
                """
                DELETE FROM task_history
                WHERE id NOT IN (
                    SELECT id FROM task_history ORDER BY recorded_at DESC, id DESC LIMIT %s
                )
                """,
                (TASK_HISTORY_MAX_ROWS,),
            )
        db.commit()
    except Exception as e:
        logger.warning(f"record_task_history failed for {task_id}: {e}")
        try:
            db.rollback()
        except Exception:
            pass


def get_active_main_task(task_type=None):
    """Return the currently active main task.

    If task_type is provided, only return an active task of that type.
    If task_type is None, return any active main task.
    """
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    non_terminal_statuses = (TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS)

    if task_type:
        cur.execute("""
            SELECT task_id, task_type, status, details
            FROM task_status
            WHERE task_type = %s AND status IN %s AND parent_task_id IS NULL
            ORDER BY timestamp DESC
            LIMIT 1
        """, (task_type, non_terminal_statuses))
    else:
        cur.execute("""
            SELECT task_id, task_type, status, details
            FROM task_status
            WHERE status IN %s AND parent_task_id IS NULL
            ORDER BY timestamp DESC
            LIMIT 1
        """, (non_terminal_statuses,))

    active_task = cur.fetchone()
    cur.close()
    return dict(active_task) if active_task else None


# --- DB Utility Functions (used by tasks.py and API) ---
def save_task_status(task_id, task_type, status=TASK_STATUS_PENDING, parent_task_id=None, sub_type_identifier=None, progress=0, details=None):
    """
    Saves or updates a task's status in the database, using Unix timestamps for start and end times.
    """
    db = get_db() # This now calls the function within this file
    cur = db.cursor()
    current_unix_time = time.time()

    if details is not None and isinstance(details, dict):
        # Log truncation logic remains the same
        if status != TASK_STATUS_SUCCESS and 'log' in details and isinstance(details['log'], list):
            log_list = details['log']
            if len(log_list) > MAX_LOG_ENTRIES_STORED:
                original_log_length = len(log_list)
                details['log'] = log_list[-MAX_LOG_ENTRIES_STORED:]
                details['log_storage_info'] = f"Log in DB truncated to last {MAX_LOG_ENTRIES_STORED} entries. Original length: {original_log_length}."
            else:
                details.pop('log_storage_info', None)
        elif status == TASK_STATUS_SUCCESS:
            details.pop('log_storage_info', None)
            if 'log' not in details or not isinstance(details.get('log'), list) or not details.get('log'):
                details['log'] = ["Task completed successfully."]

    details_json = json.dumps(details) if details is not None else None
    
    try:
        # This query now handles start_time and end_time using Unix timestamps
        cur.execute("""
            INSERT INTO task_status (task_id, parent_task_id, task_type, sub_type_identifier, status, progress, details, timestamp, start_time, end_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), %s, CASE WHEN %s IN ('SUCCESS', 'FAILURE', 'REVOKED') THEN %s ELSE NULL END)
            ON CONFLICT (task_id) DO UPDATE SET
                status = EXCLUDED.status,
                parent_task_id = EXCLUDED.parent_task_id,
                sub_type_identifier = EXCLUDED.sub_type_identifier,
                progress = EXCLUDED.progress,
                details = EXCLUDED.details,
                timestamp = NOW(),
                start_time = COALESCE(task_status.start_time, %s),
                end_time = CASE
                                WHEN EXCLUDED.status IN ('SUCCESS', 'FAILURE', 'REVOKED') AND task_status.end_time IS NULL
                                THEN %s
                                ELSE task_status.end_time
                           END
        """, (task_id, parent_task_id, task_type, sub_type_identifier, status, progress, details_json, current_unix_time, status, current_unix_time, current_unix_time, current_unix_time))
        db.commit()
    except psycopg2.Error as e:
        logger.error(f"DB Error saving task status for {task_id}: {e}")
        try:
            db.rollback()
            logger.info(f"DB transaction rolled back for task status update of {task_id}.")
        except psycopg2.Error as rb_e:
            logger.error(f"DB Error during rollback for task status {task_id}: {rb_e}")
    finally:
        cur.close()

    # Record persistent history for MAIN tasks that just reached a terminal state.
    # Skip the synthetic 'unknown' placeholder inserted by the global cancel
    # path (app_helper.cancel_all_jobs) — it has no real type and would show
    # up as an 'unknown' row in the dashboard's recent activity table.
    try:
        if (
            parent_task_id is None
            and status in (TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)
            and task_type and task_type != 'unknown'
        ):
            duration_s = None
            try:
                hist_cur = db.cursor()
                hist_cur.execute(
                    "SELECT start_time, end_time FROM task_status WHERE task_id = %s",
                    (task_id,),
                )
                row = hist_cur.fetchone()
                hist_cur.close()
                if row and row[0] is not None:
                    end = row[1] if row[1] is not None else current_unix_time
                    duration_s = max(0.0, float(end) - float(row[0]))
            except Exception:
                pass
            record_task_history(task_id, task_type, status, duration_s, details=details)
    except Exception as e_hist:
        logger.debug(f"history record skipped for {task_id}: {e_hist}")


def get_task_info_from_db(task_id):
    """Fetches task info from DB and calculates running time in Python."""
    db = get_db() # This now calls the function within this file
    cur = db.cursor(cursor_factory=DictCursor)
    # Fetch raw columns including the Unix timestamps
    cur.execute("""
        SELECT 
            task_id, parent_task_id, task_type, sub_type_identifier, status, progress, details, timestamp, start_time, end_time
        FROM task_status 
        WHERE task_id = %s
    """, (task_id,))
    row = cur.fetchone()
    cur.close()
    if not row:
        return None
    
    row_dict = dict(row)
    current_unix_time = time.time()
    
    start_time = row_dict.get('start_time')
    end_time = row_dict.get('end_time')

    # If start_time is null (old record or pre-start), duration is 0.
    if start_time is None:
        row_dict['running_time_seconds'] = 0.0
    else:
        # If end_time is null, task is running. Use current time.
        effective_end_time = end_time if end_time is not None else current_unix_time
        row_dict['running_time_seconds'] = max(0, effective_end_time - start_time)
        
    return row_dict

def get_child_tasks_from_db(parent_task_id):
    """Fetches all child tasks for a given parent_task_id from the database."""
    conn = get_db() # This now calls the function within this file
    cur = conn.cursor(cursor_factory=DictCursor)
    # MODIFIED: Select the 'details' column as well for the final check.
    cur.execute("SELECT task_id, status, sub_type_identifier, details FROM task_status WHERE parent_task_id = %s", (parent_task_id,))
    tasks = cur.fetchall()
    cur.close()
    # DictCursor returns a list of dictionary-like objects, convert to plain dicts
    return [dict(row) for row in tasks]

def track_exists(item_id):
    """
    Checks if a track exists in the database AND has been analyzed for key features.
    in both the 'score' and 'embedding' tables.
    Returns True if:
    1. The track exists in 'score' table and 'other_features', 'energy', 'mood_vector', and 'tempo' are populated.
    2. The track exists in the 'embedding' table.
    Returns False otherwise, indicating a re-analysis is needed.
    """
    conn = get_db() # This now calls the function within this file
    cur = conn.cursor()
    cur.execute("""
        SELECT s.item_id
        FROM score s
        JOIN embedding e ON s.item_id = e.item_id
        WHERE s.item_id = %s
          AND s.other_features IS NOT NULL AND s.other_features != ''
          AND s.energy IS NOT NULL
          AND s.mood_vector IS NOT NULL AND s.mood_vector != ''
          AND s.tempo IS NOT NULL
    """, (item_id,))
    row = cur.fetchone()
    cur.close()
    return row is not None

def save_track_analysis_and_embedding(item_id, title, author, tempo, key, scale, moods, embedding_vector, energy=None, other_features=None, album=None, album_artist=None, year=None, rating=None, file_path=None):
    """Saves track analysis and embedding in a single transaction."""
    
    def _sanitize_string(s, max_length=1000, field_name="field"):
        """Sanitize string for PostgreSQL insertion."""
        if s is None:
            return None
        
        # Ensure it's a string
        if not isinstance(s, str):
            try:
                s = str(s)
            except Exception:
                logger.warning(f"Could not convert {field_name} to string, using empty string")
                return ""
        
        # Remove problematic characters
        # NUL byte (0x00) - PostgreSQL cannot store
        s = s.replace('\x00', '')
        
        # Remove other control characters that could cause issues
        # Keep only printable ASCII, space, tab, newline, and common Unicode
        s = ''.join(char for char in s if char.isprintable() or char in '\n\t ')
        
        # Truncate to max length to prevent overly long strings
        if len(s) > max_length:
            logger.warning(f"{field_name} truncated from {len(s)} to {max_length} characters")
            s = s[:max_length]
        
        # Strip leading/trailing whitespace
        s = s.strip()
        
        return s
    
    # Sanitize all string inputs with field-specific limits
    title = _sanitize_string(title, max_length=500, field_name="title")
    author = _sanitize_string(author, max_length=200, field_name="author")
    album = _sanitize_string(album, max_length=200, field_name="album")
    album_artist = _sanitize_string(album_artist, max_length=200, field_name="album_artist")
    key = _sanitize_string(key, max_length=10, field_name="key")
    scale = _sanitize_string(scale, max_length=10, field_name="scale")
    other_features = _sanitize_string(other_features, max_length=2000, field_name="other_features")

    # year: parse from various date formats and validate
    def _parse_year_from_date(year_value):
        """
        Parse year from various date formats.
        Supports: YYYY, YYYY-MM-DD, MM-DD-YYYY, DD-MM-YYYY (with - or / separators)
        """
        if year_value is None:
            return None

        year_str = str(year_value).strip()
        if not year_str:
            return None

        # Try parsing as pure integer first (YYYY)
        try:
            year = int(year_str)
            if 1000 <= year <= 2100:
                return year
        except (ValueError, TypeError):
            pass

        # Normalize separators
        normalized = year_str.replace('/', '-')
        parts = normalized.split('-')

        if len(parts) == 3:
            try:
                # YYYY-MM-DD format
                if len(parts[0]) == 4:
                    year = int(parts[0])
                    if 1000 <= year <= 2100:
                        return year

                # MM-DD-YYYY or DD-MM-YYYY format
                if len(parts[2]) == 4:
                    year = int(parts[2])
                    if 1000 <= year <= 2100:
                        return year

                # 2-digit year (MM-DD-YY)
                if len(parts[2]) == 2:
                    year = int(parts[2])
                    year += 2000 if year < 30 else 1900
                    if 1000 <= year <= 2100:
                        return year
            except (ValueError, TypeError, IndexError):
                pass

        return None

    year = _parse_year_from_date(year)

    # rating: validate as integer 0-5 (5-star rating system)
    if rating is not None:
        try:
            rating = int(rating)
            if rating < 0 or rating > 5:
                rating = None
        except (ValueError, TypeError):
            rating = None

    file_path = _sanitize_string(file_path, max_length=1000, field_name="file_path")

    mood_str = ','.join(f"{k}:{v:.3f}" for k, v in moods.items())
    
    conn = get_db() # This now calls the function within this file
    cur = conn.cursor()
    try:
        # Save analysis to score table
        cur.execute("""
            INSERT INTO score (item_id, title, author, tempo, key, scale, mood_vector, energy, other_features, album, album_artist, year, rating, file_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (item_id) DO UPDATE SET
                title = EXCLUDED.title,
                author = EXCLUDED.author,
                tempo = EXCLUDED.tempo,
                key = EXCLUDED.key,
                scale = EXCLUDED.scale,
                mood_vector = EXCLUDED.mood_vector,
                energy = EXCLUDED.energy,
                other_features = EXCLUDED.other_features,
                album = EXCLUDED.album,
                album_artist = EXCLUDED.album_artist,
                year = EXCLUDED.year,
                rating = EXCLUDED.rating,
                file_path = EXCLUDED.file_path
        """, (item_id, title, author, tempo, key, scale, mood_str, energy, other_features, album, album_artist, year, rating, file_path))

        # Save embedding
        if isinstance(embedding_vector, np.ndarray) and embedding_vector.size > 0:
            embedding_blob = embedding_vector.astype(np.float32).tobytes()
            cur.execute("""
                INSERT INTO embedding (item_id, embedding) VALUES (%s, %s)
                ON CONFLICT (item_id) DO UPDATE SET embedding = EXCLUDED.embedding
            """, (item_id, psycopg2.Binary(embedding_blob)))

        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("Error saving track analysis and embedding for %s: %s", item_id, e)
        raise
    finally:
        cur.close()

def save_clap_embedding(item_id, clap_embedding_vector):
    """Saves CLAP embedding for a track."""
    if clap_embedding_vector is None or (isinstance(clap_embedding_vector, np.ndarray) and clap_embedding_vector.size == 0):
        return
    
    conn = get_db()
    cur = conn.cursor()
    try:
        embedding_blob = clap_embedding_vector.astype(np.float32).tobytes()
        cur.execute("""
            INSERT INTO clap_embedding (item_id, embedding) VALUES (%s, %s)
            ON CONFLICT (item_id) DO UPDATE SET embedding = EXCLUDED.embedding
        """, (item_id, psycopg2.Binary(embedding_blob)))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving CLAP embedding for {item_id}: {e}")
        raise
    finally:
        cur.close()


def get_clap_embedding(item_id):
    """Load CLAP embedding for a track from the database.
    
    Returns:
        numpy array (512-dim float32) or None if not found
    """
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT embedding FROM clap_embedding WHERE item_id = %s", (item_id,))
        row = cur.fetchone()
        if row and row[0]:
            return np.frombuffer(row[0], dtype=np.float32)
        return None
    except Exception as e:
        logger.error(f"Error loading CLAP embedding for {item_id}: {e}")
        return None
    finally:
        cur.close()


def save_mulan_embedding(item_id, mulan_embedding_vector):
    """Saves MuLan embedding for a track."""
    if mulan_embedding_vector is None or (isinstance(mulan_embedding_vector, np.ndarray) and mulan_embedding_vector.size == 0):
        return
    
    conn = get_db()
    cur = conn.cursor()
    try:
        embedding_blob = mulan_embedding_vector.astype(np.float32).tobytes()
        cur.execute("""
            INSERT INTO mulan_embedding (item_id, embedding) VALUES (%s, %s)
            ON CONFLICT (item_id) DO UPDATE SET embedding = EXCLUDED.embedding
        """, (item_id, psycopg2.Binary(embedding_blob)))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving MuLan embedding for {item_id}: {e}")
        raise
    finally:
        cur.close()

def get_all_tracks():
    """Fetches all tracks and their embeddings from the database."""
    conn = get_db() # This now calls the function within this file
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute("""
        SELECT s.item_id, s.title, s.author, s.tempo, s.key, s.scale, s.mood_vector, s.energy, s.other_features, s.year, s.rating, s.file_path, e.embedding
        FROM score s
        LEFT JOIN embedding e ON s.item_id = e.item_id
    """)
    rows = cur.fetchall()
    cur.close()
    
    # Convert DictRow objects to regular dicts to allow adding new keys.
    processed_rows = []
    for row in rows:
        row_dict = dict(row)
        if row_dict.get('embedding'):
            # Use np.frombuffer to convert the binary data back to a numpy array
            row_dict['embedding_vector'] = np.frombuffer(row_dict['embedding'], dtype=np.float32)
        else:
            row_dict['embedding_vector'] = np.array([]) # Use a consistent name
        processed_rows.append(row_dict)
        
    return processed_rows

def get_tracks_by_ids(item_ids_list):
    """Fetches full track data (including embeddings) for a specific list of item_ids."""
    if not item_ids_list:
        return []
    conn = get_db() # This now calls the function within this file
    cur = conn.cursor(cursor_factory=DictCursor)
    
    # Convert item_ids to strings to match the text type in database
    item_ids_str = [str(item_id) for item_id in item_ids_list]
    
    query = """
        SELECT s.item_id, s.title, s.author, s.album, s.album_artist, s.tempo, s.key, s.scale, s.mood_vector, s.energy, s.other_features, s.year, s.rating, s.file_path, e.embedding
        FROM score s
        LEFT JOIN embedding e ON s.item_id = e.item_id
        WHERE s.item_id IN %s
    """
    cur.execute(query, (tuple(item_ids_str),))
    rows = cur.fetchall()
    cur.close()

    # Convert DictRow objects to regular dicts to allow adding new keys.
    processed_rows = []
    for row in rows:
        row_dict = dict(row)
        if row_dict.get('embedding'):
            row_dict['embedding_vector'] = np.frombuffer(row_dict['embedding'], dtype=np.float32)
        else:
            row_dict['embedding_vector'] = np.array([])
        processed_rows.append(row_dict)
    
    return processed_rows

def get_score_data_by_ids(item_ids_list):
    """Fetches only score-related data (excluding embeddings) for a specific list of item_ids."""
    if not item_ids_list:
        return []
    conn = get_db() # This now calls the function within this file
    cur = conn.cursor(cursor_factory=DictCursor)
    query = """
        SELECT s.item_id, s.title, s.author, s.album, s.album_artist, s.tempo, s.key, s.scale, s.mood_vector, s.energy, s.other_features, s.year, s.rating, s.file_path
        FROM score s
        WHERE s.item_id IN %s
    """
    try:
        cur.execute(query, (tuple(item_ids_list),))
        rows = cur.fetchall()
    except Exception as e:
        logger.error(f"Error fetching score data by IDs: {e}")
        rows = [] # Return empty list on error
    finally:
        cur.close()
    return [dict(row) for row in rows]


def save_alchemy_anchor(name, centroid):
    """Save a named anchor centroid into DB."""
    if not name or not centroid or not isinstance(centroid, list):
        raise ValueError('Anchor name and centroid list are required.')
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    try:
        centroid_json = json.dumps(centroid)
        cur.execute(
            "INSERT INTO alchemy_anchors (name, centroid) VALUES (%s, %s) "
            "ON CONFLICT (name) DO UPDATE SET centroid = EXCLUDED.centroid, created_at = NOW() "
            "RETURNING id, name, created_at",
            (name, centroid_json)
        )
        row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save alchemy anchor '{name}': {e}")
        return None
    finally:
        cur.close()


def get_alchemy_anchors():
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    try:
        cur.execute("SELECT id, name, created_at FROM alchemy_anchors ORDER BY created_at DESC")
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to load alchemy anchors: {e}")
        return []
    finally:
        cur.close()


def delete_alchemy_anchor(anchor_id):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM alchemy_anchors WHERE id = %s", (anchor_id,))
        conn.commit()
        return cur.rowcount > 0
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to delete alchemy anchor id={anchor_id}: {e}")
        return False
    finally:
        cur.close()


def get_alchemy_anchor_by_id(anchor_id):
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    try:
        cur.execute("SELECT id, name, centroid, created_at FROM alchemy_anchors WHERE id = %s", (anchor_id,))
        row = cur.fetchone()
        if not row:
            return None
        anchor = dict(row)
        if isinstance(anchor.get('centroid'), str):
            try:
                anchor['centroid'] = json.loads(anchor['centroid'])
            except Exception:
                anchor['centroid'] = None
        return anchor
    except Exception as e:
        logger.error(f"Failed to fetch alchemy anchor id={anchor_id}: {e}")
        return None
    finally:
        cur.close()


def update_alchemy_anchor_name(anchor_id, name):
    if not name or not isinstance(name, str):
        return None
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    try:
        cur.execute(
            "UPDATE alchemy_anchors SET name = %s WHERE id = %s RETURNING id, name",
            (name.strip(), anchor_id)
        )
        row = cur.fetchone()
        conn.commit()
        if not row:
            return None
        return dict(row)
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to rename alchemy anchor id={anchor_id}: {e}")
        return None
    finally:
        cur.close()


def save_map_projection(index_name, id_map, projection_array):
    """
    Save a precomputed 2D projection into the map_projection_data table.
    projection_array: numpy array of shape (N,2), dtype=float32
    id_map: JSON-serializable list/dict mapping rows to item_ids
    """
    conn = get_db()
    cur = conn.cursor()
    try:
        blob = projection_array.astype(np.float32).tobytes()
        id_map_json = json.dumps(id_map)
        cur.execute("""
            INSERT INTO map_projection_data (index_name, projection_data, id_map_json, embedding_dimension)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (index_name) DO UPDATE SET projection_data = EXCLUDED.projection_data, id_map_json = EXCLUDED.id_map_json, embedding_dimension = EXCLUDED.embedding_dimension, created_at = NOW()
        """, (index_name, psycopg2.Binary(blob), id_map_json, projection_array.shape[1] if projection_array.ndim == 2 else 0))
        conn.commit()
        try:
            size_bytes = len(blob)
            id_count = len(id_map) if hasattr(id_map, '__len__') else None
            logger.info(f"Saved map projection '{index_name}' to DB: {size_bytes} bytes, ids={id_count}")
        except Exception:
            # non-critical logging error
            logger.debug("Saved map projection but failed to compute size/id_count for log.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save map projection: {e}")
        raise
    finally:
        cur.close()


def load_map_projection(index_name, force_reload=False):
    """Load precomputed projection from DB. Returns (id_map, numpy_array) or (None, None)"""
    global MAP_PROJECTION_CACHE
    # Try cache first (unless force_reload is True)
    if not force_reload and MAP_PROJECTION_CACHE and MAP_PROJECTION_CACHE.get('index_name') == index_name:
        logger.info(f"Map projection '{index_name}' already loaded in cache. Skipping reload.")
        return MAP_PROJECTION_CACHE.get('id_map'), MAP_PROJECTION_CACHE.get('projection')

    logger.info(f"Attempting to load map projection '{index_name}' from database into memory...")
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT projection_data, id_map_json FROM map_projection_data WHERE index_name = %s", (index_name,))
        row = cur.fetchone()
        if not row:
            logger.warning(f"Map projection '{index_name}' not found in the database. Cache will be empty.")
            return None, None
        proj_blob, id_map_json = row[0], row[1]
        proj = np.frombuffer(proj_blob, dtype=np.float32)
        # infer shape as (-1,2) if length divisible by 2
        if proj.size % 2 == 0:
            proj = proj.reshape((-1, 2))
        id_map = json.loads(id_map_json)
        MAP_PROJECTION_CACHE = {'index_name': index_name, 'id_map': id_map, 'projection': proj}
        logger.info(f"Map projection '{index_name}' with {len(id_map)} items loaded successfully into memory.")
        return id_map, proj
    except Exception as e:
        logger.error(f"Failed to load map projection: {e}", exc_info=True)
        return None, None
    finally:
        cur.close()


def build_and_store_map_projection(index_name='main_map'):
    """Compute 2D projection for all tracks and store it. Uses available projection helpers if present.
    Returns True on success.
    """
    # Import local projection helpers to avoid circular imports
    try:
        from tasks.song_alchemy import _project_with_umap, _project_to_2d
    except Exception:
        _project_with_umap = None
        _project_to_2d = None

    rows = get_all_tracks()
    # collect embeddings and ids
    ids = []
    embs = []
    for r in rows:
        v = r.get('embedding_vector')
        if v is not None and v.size:
            ids.append(r['item_id'])
            embs.append(v)
    if not embs:
        logger.info('No embeddings available to build map projection.')
        return False

    mat = np.vstack(embs)
    projections = None
    try:
        logger.info(f"Starting to build map projection: {mat.shape[0]} embeddings found.")
        if _project_with_umap is not None:
            projections = _project_with_umap([v for v in mat])
    except Exception as e:
        logger.warning(f"UMAP projection failed during build: {e}")
        projections = None

    if projections is None:
        try:
            if _project_to_2d is not None:
                projections = _project_to_2d([v for v in mat])
        except Exception as e:
            logger.warning(f"PCA projection failed during build: {e}")
            projections = None

    if projections is None:
        projections = np.zeros((mat.shape[0], 2), dtype=np.float32)
    else:
        projections = np.array(projections, dtype=np.float32)
    logger.info(f"Computed projection shape: {projections.shape}")

    # Save to DB
    try:
        save_map_projection(index_name, ids, projections)
        # update in-memory cache
        global MAP_PROJECTION_CACHE
        MAP_PROJECTION_CACHE = {'index_name': index_name, 'id_map': ids, 'projection': projections}
        # Note: Caller (analysis task) is responsible for publishing reload message after all builds complete
        return True
    except Exception as e:
        logger.error(f"Failed to build and store map projection: {e}")
        return False


def load_artist_projection(index_name='artist_map', force_reload=False):
    """Load precomputed artist component projection from DB. 
    Returns (artist_component_map, numpy_array) or (None, None).
    artist_component_map format: [{'artist_id': '...', 'component_idx': 0, 'weight': 0.3}, ...]
    """
    global ARTIST_PROJECTION_CACHE
    # Try cache first (unless force_reload is True)
    if not force_reload and ARTIST_PROJECTION_CACHE and ARTIST_PROJECTION_CACHE.get('index_name') == index_name:
        logger.info(f"Artist projection '{index_name}' already loaded in cache. Skipping reload.")
        return ARTIST_PROJECTION_CACHE.get('component_map'), ARTIST_PROJECTION_CACHE.get('projection')

    logger.info(f"Attempting to load artist projection '{index_name}' from database into memory...")
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT projection_data, artist_component_map_json FROM artist_component_projection WHERE index_name = %s", (index_name,))
        row = cur.fetchone()
        if not row:
            logger.warning(f"Artist projection '{index_name}' not found in the database. Cache will be empty.")
            return None, None
        proj_blob, component_map_json = row[0], row[1]
        proj = np.frombuffer(proj_blob, dtype=np.float32)
        # infer shape as (-1,2) if length divisible by 2
        if proj.size % 2 == 0:
            proj = proj.reshape((-1, 2))
        component_map = json.loads(component_map_json)
        ARTIST_PROJECTION_CACHE = {'index_name': index_name, 'component_map': component_map, 'projection': proj}
        logger.info(f"Artist projection '{index_name}' with {len(component_map)} components loaded successfully into memory.")
        return component_map, proj
    except Exception as e:
        logger.error(f"Failed to load artist projection: {e}", exc_info=True)
        return None, None
    finally:
        cur.close()


def save_artist_projection(index_name, component_map, projections):
    """Save artist component projection to database.
    component_map: [{'artist_id': '...', 'component_idx': 0, 'weight': 0.3}, ...]
    projections: numpy array of shape (N, 2)
    """
    conn = get_db()
    cur = conn.cursor()
    try:
        component_map_json = json.dumps(component_map)
        proj_blob = projections.astype(np.float32).tobytes()
        cur.execute("INSERT INTO artist_component_projection (index_name, projection_data, artist_component_map_json) VALUES (%s, %s, %s) ON CONFLICT (index_name) DO UPDATE SET projection_data = EXCLUDED.projection_data, artist_component_map_json = EXCLUDED.artist_component_map_json, created_at = CURRENT_TIMESTAMP", (index_name, proj_blob, component_map_json))
        conn.commit()
        logger.info(f"Saved artist projection '{index_name}' with {len(component_map)} components to database.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save artist projection: {e}", exc_info=True)
    finally:
        cur.close()


def build_and_store_artist_projection(index_name='artist_map'):
    """Compute 2D projection for all artist GMM components and store it.
    This will be called during analysis to create the artist component map.
    Returns True on success.
    """
    from tasks.artist_gmm_manager import artist_gmm_params, load_artist_index_for_querying
    from tasks.song_alchemy import _project_with_umap, _project_to_2d
    
    # Always reload artist GMM params from database (force reload to ensure fresh data)
    load_artist_index_for_querying(force_reload=True)
    
    # Re-import after loading to get the updated global variable
    from tasks.artist_gmm_manager import artist_gmm_params as loaded_params
    
    if not loaded_params:
        logger.warning("No artist GMM params available to build artist projection.")
        return False
    
    # Collect all artist component vectors
    component_map = []
    vectors = []
    
    for artist_name, gmm in loaded_params.items():
        means = np.array(gmm['means'])  # Shape: [n_components, embedding_dim]
        weights = np.array(gmm['weights'])  # Shape: [n_components]
        
        # Get artist_id (use artist_name if no mapping exists)
        from app_helper_artist import get_artist_id_by_name
        artist_id = get_artist_id_by_name(artist_name) or artist_name
        
        for comp_idx in range(len(means)):
            component_map.append({
                'artist_id': artist_id,
                'artist_name': artist_name,
                'component_idx': comp_idx,
                'weight': float(weights[comp_idx])
            })
            vectors.append(means[comp_idx])
    
    if not vectors:
        logger.info('No artist component vectors available to build projection.')
        return False
    
    mat = np.vstack(vectors)
    projections = None
    
    try:
        logger.info(f"Starting to build artist projection: {mat.shape[0]} component vectors found.")
        # Try UMAP first
        if _project_with_umap is not None:
            projections = _project_with_umap([v for v in mat])
    except Exception as e:
        logger.warning(f"UMAP projection failed for artist components: {e}")
        projections = None
    
    # Fallback to PCA
    if projections is None:
        try:
            if _project_to_2d is not None:
                projections = _project_to_2d([v for v in mat])
        except Exception as e:
            logger.warning(f"PCA projection failed for artist components: {e}")
            projections = None
    
    if projections is None:
        projections = np.zeros((mat.shape[0], 2), dtype=np.float32)
    else:
        projections = np.array(projections, dtype=np.float32)
    
    logger.info(f"Computed artist projection shape: {projections.shape}")
    
    try:
        save_artist_projection(index_name, component_map, projections)
        # Update in-memory cache
        global ARTIST_PROJECTION_CACHE
        ARTIST_PROJECTION_CACHE = {'index_name': index_name, 'component_map': component_map, 'projection': projections}
        # Note: Caller (analysis task) is responsible for publishing reload message after all builds complete
        return True
    except Exception as e:
        logger.error(f"Failed to build and store artist projection: {e}")
        return False


def update_playlist_table(playlists): # Removed db_path
    conn = get_db() # This now calls the function within this file
    cur = conn.cursor()
    try:
        # Clear all previous conceptual playlists to reflect only the current run.
        cur.execute("DELETE FROM playlist")
        for name, cluster in playlists.items():
            for item_id, title, author in cluster:
                cur.execute("INSERT INTO playlist (playlist_name, item_id, title, author) VALUES (%s, %s, %s, %s) ON CONFLICT (playlist_name, item_id) DO NOTHING", (name, item_id, title, author))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("Error updating playlist table: %s", e)
    finally:
        cur.close()

def cancel_job_and_children_recursive(job_id, task_type_from_db=None, reason="Task cancellation processed by API."):
    """Helper to cancel a job and its children based on DB records.

    NOTE: Minimal global behavior — when invoked from the API cancel endpoint we clear RQ queues,
    attempt to stop all jobs known to RQ, delete all rows in `task_status`, and insert a single
    REVOKED row for the requested `job_id` (so UI sees one canonical cancelled task).
    This keeps the function signature unchanged and is intentionally simple and destructive (as requested).
    """
    cancelled_count = 0

    # --- Scan RQ for job ids to cancel ---
    job_ids = set()
    for q in (rq_queue_high, rq_queue_default):
        try:
            ids = getattr(q, 'job_ids', None)
            if ids is None:
                key = f"rq:queue:{getattr(q, 'name', '')}"
                raw = redis_conn.lrange(key, 0, -1)
                ids = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in raw]
            job_ids.update([str(i) for i in ids if i is not None])
        except Exception as e_q:
            logger.warning(f"Could not read queue {getattr(q, 'name', '<unknown>')}: {e_q}")

    # Include job ids from RQ job keys (covers started jobs)
    try:
        raw_keys = redis_conn.keys('rq:job:*')
        for k in raw_keys:
            kstr = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            parts = kstr.split(':')
            if len(parts) >= 3:
                jid = ':'.join(parts[2:])
                job_ids.add(jid)
    except Exception as e_keys:
        logger.warning(f"Could not list rq job keys: {e_keys}")

    # Attempt to cancel/stop all discovered jobs
    for jid in job_ids:
        try:
            try:
                j = Job.fetch(jid, connection=redis_conn)
                if not j.is_finished and not j.is_failed and not j.is_canceled:
                    if j.is_started:
                        send_stop_job_command(redis_conn, jid)
                    else:
                        j.cancel()
                    cancelled_count += 1
                    logger.info(f"Sent stop/cancel for job {jid} during global cancel")
            except NoSuchJobError:
                logger.debug(f"Job {jid} not found in RQ during global cancel")
        except Exception as e_j:
            logger.error(f"Error cancelling job {jid} during global cancel: {e_j}")

    # Try to clear the RQ queues using API (preferred) and fallback to key deletion if necessary
    try:
        for q in (rq_queue_high, rq_queue_default):
            try:
                if hasattr(q, 'empty'):
                    q.empty()
                    logger.info(f"Emptied queue {getattr(q, 'name', '<unknown>')} via Queue.empty() as part of global cancel")
                else:
                    key = f"rq:queue:{getattr(q, 'name', '')}"
                    redis_conn.delete(key)
                    logger.info(f"Deleted Redis key fallback for queue: {key} as part of global cancel")
            except Exception as e_q:
                logger.warning(f"Failed to empty queue {getattr(q, 'name', '<unknown>')} during global cancel: {e_q}")
    except Exception as e_qdel:
        logger.warning(f'Failed to clear queue lists during global cancel: {e_qdel}')

    # Consolidate DB: delete all task_status rows and insert a single REVOKED row for job_id
    db = get_db()
    cur = db.cursor()
    try:
        # Snapshot the in-flight main tasks into the persistent task_history
        # *before* we wipe task_status, so the dashboard's history table keeps
        # showing what was running when the user pressed Cancel.
        try:
            with db.cursor(cursor_factory=DictCursor) as snap_cur:
                snap_cur.execute(
                    "SELECT task_id, task_type, status, details, start_time, end_time "
                    "FROM task_status WHERE parent_task_id IS NULL"
                )
                now_ts = time.time()
                for r in snap_cur.fetchall():
                    duration_s = None
                    if r['start_time'] is not None:
                        end = r['end_time'] if r['end_time'] is not None else now_ts
                        duration_s = max(0.0, float(end) - float(r['start_time']))
                    details_obj = None
                    if r['details']:
                        try:
                            details_obj = json.loads(r['details'])
                        except Exception:
                            details_obj = None
                    # If the task was already in a terminal status, keep that one;
                    # otherwise mark it REVOKED.
                    final_status = r['status'] if r['status'] in (
                        TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED
                    ) else TASK_STATUS_REVOKED
                    record_task_history(
                        r['task_id'], r['task_type'], final_status,
                        duration_s, details=details_obj,
                    )
        except Exception as e_snap:
            logger.warning(f"Global cancel: failed snapshotting task_status into task_history: {e_snap}")

        cur.execute("DELETE FROM task_status")
        deleted = cur.rowcount
        db.commit()
        logger.info(f"Global cancel DB cleanup: deleted {deleted} task_status rows")
    except Exception as e_dbdel:
        db.rollback()
        logger.error(f"Error deleting task_status rows during global cancel: {e_dbdel}")
    finally:
        cur.close()

    try:
        # Ensure a single REVOKED row exists for job_id
        save_task_status(job_id, 'unknown', TASK_STATUS_REVOKED, progress=100, details={"message": reason, "origin": "global_cancel"})
    except Exception as e_save:
        logger.error(f"Failed to insert REVOKED recap row for {job_id}: {e_save}")

    return cancelled_count


# --- Auth / user-management helpers ---
# All auth logic (setup/auth/admin barriers, user CRUD, password hashing,
# JWT handling, the Flask routes) lives in ``app_auth``. The re-exports
# below keep the legacy ``from app_helper import ...`` paths working.
from app_auth import (  # noqa: E402  (intentional late import to avoid cycles)
    USER_ROLE_USER,
    USER_ROLE_ADMIN,
    check_setup_needed,
    check_auth_needed,
    check_admin_needed,
    is_admin_path,
    list_additional_users,
    count_admin_users,
    get_additional_user_by_id,
    create_additional_user,
    delete_additional_user_safe,
    verify_additional_user,
    upsert_admin_user,
    seed_admin_from_env,
)
