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
                options='-c statement_timeout=300000'  # 5 min query timeout (300 seconds)
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
        # Create 'playlist' table
        cur.execute("CREATE TABLE IF NOT EXISTS playlist (id SERIAL PRIMARY KEY, playlist_name TEXT, item_id TEXT, title TEXT, author TEXT, UNIQUE (playlist_name, item_id))")
        # Create 'task_status' table
        cur.execute("CREATE TABLE IF NOT EXISTS task_status (id SERIAL PRIMARY KEY, task_id TEXT UNIQUE NOT NULL, parent_task_id TEXT, task_type TEXT NOT NULL, sub_type_identifier TEXT, status TEXT, progress INTEGER DEFAULT 0, details TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        # Migrate 'start_time' and 'end_time' columns
        for col_name in ['start_time', 'end_time']:
            cur.execute("SELECT data_type FROM information_schema.columns WHERE table_name = 'task_status' AND column_name = %s", (col_name,))
            if not cur.fetchone(): cur.execute(f"ALTER TABLE task_status ADD COLUMN {col_name} DOUBLE PRECISION")
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
        # Create 'artist_index_data' table for artist GMM-based HNSW index
        cur.execute("CREATE TABLE IF NOT EXISTS artist_index_data (index_name VARCHAR(255) PRIMARY KEY, index_data BYTEA NOT NULL, artist_map_json TEXT NOT NULL, gmm_params_json TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        # Create 'map_projection_data' table for precomputed 2D map projections
        cur.execute("CREATE TABLE IF NOT EXISTS map_projection_data (index_name VARCHAR(255) PRIMARY KEY, projection_data BYTEA NOT NULL, id_map_json TEXT NOT NULL, embedding_dimension INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        # Create 'artist_component_projection' table for precomputed 2D artist component projections
        cur.execute("CREATE TABLE IF NOT EXISTS artist_component_projection (index_name VARCHAR(255) PRIMARY KEY, projection_data BYTEA NOT NULL, artist_component_map_json TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        # Create 'cron' table to hold scheduled jobs (very small and simple)
        cur.execute("CREATE TABLE IF NOT EXISTS cron (id SERIAL PRIMARY KEY, name TEXT, task_type TEXT NOT NULL, cron_expr TEXT NOT NULL, enabled BOOLEAN DEFAULT FALSE, last_run DOUBLE PRECISION, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        # Create 'artist_mapping' table to map artist names to media server artist IDs
        cur.execute("CREATE TABLE IF NOT EXISTS artist_mapping (artist_name TEXT PRIMARY KEY, artist_id TEXT)")
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
            
            logger.info(f"Inserted {len(default_queries)} default CLAP search queries")

        # =================================================================
        # MULTI-PROVIDER SUPPORT TABLES
        # =================================================================

        # Create 'provider' table - Registry of configured media providers
        cur.execute("""
            CREATE TABLE IF NOT EXISTS provider (
                id SERIAL PRIMARY KEY,
                provider_type VARCHAR(50) NOT NULL,
                name VARCHAR(255) NOT NULL,
                config JSONB NOT NULL DEFAULT '{}',
                enabled BOOLEAN DEFAULT TRUE,
                priority INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(provider_type, name)
            )
        """)

        # Create 'track' table - Stable track identity based on file path
        cur.execute("""
            CREATE TABLE IF NOT EXISTS track (
                id SERIAL PRIMARY KEY,
                file_path_hash VARCHAR(64) NOT NULL UNIQUE,
                file_path TEXT NOT NULL,
                normalized_path TEXT,
                file_size BIGINT,
                file_modified TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_track_file_path_hash ON track(file_path_hash)")
        # Add normalized_path column if it doesn't exist (migration for existing installs)
        cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'track' AND column_name = 'normalized_path')")
        if not cur.fetchone()[0]:
            logger.info("Adding 'normalized_path' column to 'track' table.")
            cur.execute("ALTER TABLE track ADD COLUMN normalized_path TEXT")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_track_normalized_path ON track(normalized_path)")

        # Create 'provider_track' table - Links provider item_ids to tracks
        cur.execute("""
            CREATE TABLE IF NOT EXISTS provider_track (
                id SERIAL PRIMARY KEY,
                provider_id INTEGER NOT NULL REFERENCES provider(id) ON DELETE CASCADE,
                track_id INTEGER NOT NULL REFERENCES track(id) ON DELETE CASCADE,
                item_id TEXT NOT NULL,
                title TEXT,
                artist TEXT,
                album TEXT,
                last_synced TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(provider_id, item_id),
                UNIQUE(provider_id, track_id)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_provider_track_item_id ON provider_track(item_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_provider_track_track_id ON provider_track(track_id)")

        # Create 'app_settings' table - Application configuration storage
        cur.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                key VARCHAR(255) PRIMARY KEY,
                value JSONB NOT NULL,
                category VARCHAR(100),
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add 'track_id' column to 'score' table if not exists (for multi-provider linking)
        cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'track_id')")
        if not cur.fetchone()[0]:
            logger.info("Adding 'track_id' column to 'score' table for multi-provider support.")
            cur.execute("ALTER TABLE score ADD COLUMN track_id INTEGER REFERENCES track(id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_score_track_id ON score(track_id)")

        # Add 'file_path' column to 'score' table if not exists (for file identification)
        cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'file_path')")
        if not cur.fetchone()[0]:
            logger.info("Adding 'file_path' column to 'score' table.")
            cur.execute("ALTER TABLE score ADD COLUMN file_path TEXT")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_score_file_path ON score(file_path)")

        # Performance indexes for hot queries (brainstorm, artist search)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_score_author_lower ON score(LOWER(author))")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_score_title_author_lower ON score(LOWER(title), LOWER(author))")

        # Insert default settings if app_settings is empty
        cur.execute("SELECT COUNT(*) FROM app_settings")
        if cur.fetchone()[0] == 0:
            default_settings = [
                ('setup_completed', 'false', 'system', 'Whether the setup wizard has been completed'),
                ('setup_version', '"1.0"', 'system', 'Version of the setup wizard last completed'),
                ('multi_provider_enabled', 'false', 'providers', 'Whether multi-provider mode is enabled'),
                ('primary_provider_id', 'null', 'providers', 'ID of the primary provider for playlist creation'),
                ('max_songs_per_artist_playlist', '5', 'ai', 'Max songs per artist in instant playlists'),
                ('playlist_energy_arc', 'false', 'ai', 'Enable energy arc shaping for playlist ordering'),
                ('ai_request_timeout', '300', 'ai', 'AI request timeout in seconds'),
            ]
            for key, value, category, description in default_settings:
                cur.execute("""
                    INSERT INTO app_settings (key, value, category, description)
                    VALUES (%s, %s::jsonb, %s, %s)
                    ON CONFLICT (key) DO NOTHING
                """, (key, value, category, description))
            logger.info("Inserted default app settings")

        # =================================================================
        # MIGRATIONS (guarded by app_settings keys)
        # =================================================================

        # Migration: Recompute track.file_path_hash with case-normalized paths
        cur.execute("SELECT value FROM app_settings WHERE key = 'migration_case_normalization_done'")
        if not cur.fetchone():
            try:
                import hashlib as _hashlib
                logger.info("Running migration: case normalization for track.file_path_hash")
                cur.execute("SELECT id, file_path, normalized_path FROM track")
                tracks_to_update = cur.fetchall()
                merged_count = 0
                updated_count = 0

                # Group tracks by their new normalized hash
                hash_groups = {}
                for track_id, file_path, old_normalized in tracks_to_update:
                    if not file_path:
                        continue
                    # Re-normalize with .lower() (normalize_provider_path now returns lowered)
                    new_normalized = normalize_provider_path(file_path)
                    if not new_normalized:
                        continue
                    new_hash = _hashlib.sha256(new_normalized.encode('utf-8')).hexdigest()
                    if new_hash not in hash_groups:
                        hash_groups[new_hash] = []
                    hash_groups[new_hash].append((track_id, new_normalized, new_hash))

                for new_hash, group in hash_groups.items():
                    if len(group) == 1:
                        # Simple update
                        track_id, new_normalized, new_hash = group[0]
                        cur.execute("""
                            UPDATE track SET file_path_hash = %s, normalized_path = %s, updated_at = NOW()
                            WHERE id = %s
                        """, (new_hash, new_normalized, track_id))
                        updated_count += 1
                    else:
                        # Merge: keep the first, redirect others' provider_track refs
                        canonical_track_id = group[0][0]
                        new_normalized = group[0][1]
                        cur.execute("""
                            UPDATE track SET file_path_hash = %s, normalized_path = %s, updated_at = NOW()
                            WHERE id = %s
                        """, (new_hash, new_normalized, canonical_track_id))
                        for dup_track_id, _, _ in group[1:]:
                            # Redirect provider_track references
                            cur.execute("""
                                UPDATE provider_track SET track_id = %s WHERE track_id = %s
                            """, (canonical_track_id, dup_track_id))
                            # Redirect score.track_id references
                            cur.execute("""
                                UPDATE score SET track_id = %s WHERE track_id = %s
                            """, (canonical_track_id, dup_track_id))
                            # Delete duplicate track
                            cur.execute("DELETE FROM track WHERE id = %s", (dup_track_id,))
                            merged_count += 1

                cur.execute("""
                    INSERT INTO app_settings (key, value, updated_at)
                    VALUES ('migration_case_normalization_done', 'true', NOW())
                    ON CONFLICT (key) DO NOTHING
                """)
                db.commit()
                logger.info(f"Case normalization migration complete: {updated_count} updated, {merged_count} merged")
            except Exception as e:
                db.rollback()
                logger.warning(f"Case normalization migration failed (will retry next startup): {e}")

        # Migration: Consolidate duplicate score rows (one per track_id)
        # MUST run AFTER all new code paths (link_provider_to_existing_track, updated remap, etc.)
        cur.execute("SELECT value FROM app_settings WHERE key = 'migration_dedup_score_rows_done'")
        if not cur.fetchone():
            try:
                logger.info("Running migration: consolidating duplicate score rows")
                # Find track_ids with multiple score rows
                cur.execute("""
                    SELECT track_id, COUNT(*) as cnt
                    FROM score
                    WHERE track_id IS NOT NULL
                    GROUP BY track_id
                    HAVING COUNT(*) > 1
                """)
                dup_groups = cur.fetchall()
                total_deleted = 0

                for track_id, cnt in dup_groups:
                    # Get all item_ids for this track_id
                    cur.execute("""
                        SELECT s.item_id FROM score s WHERE s.track_id = %s ORDER BY s.item_id
                    """, (track_id,))
                    item_ids = [row[0] for row in cur.fetchall()]

                    # Ensure each item_id has a provider_track entry
                    for item_id in item_ids:
                        cur.execute("""
                            SELECT 1 FROM provider_track WHERE item_id = %s
                        """, (item_id,))
                        if not cur.fetchone():
                            # Check if this item_id is the canonical one used in Voyager etc.
                            # Try to link it via score's file_path
                            cur.execute("SELECT file_path FROM score WHERE item_id = %s", (item_id,))
                            fp_row = cur.fetchone()
                            if fp_row and fp_row[0]:
                                logger.debug(f"Dedup migration: item {item_id} has no provider_track entry, creating placeholder")
                                # We can't create a full link without provider_id, skip this group
                                logger.warning(f"Skipping dedup for track_id {track_id}: item {item_id} has no provider_track mapping")
                                break
                    else:
                        # All items have provider_track entries — safe to dedup
                        # Pick canonical: prefer the one from the primary provider
                        primary_pid = get_primary_provider_id()
                        canonical_id = None
                        if primary_pid:
                            cur.execute("""
                                SELECT pt.item_id FROM provider_track pt
                                WHERE pt.track_id = %s AND pt.provider_id = %s
                                LIMIT 1
                            """, (track_id, primary_pid))
                            row = cur.fetchone()
                            if row:
                                canonical_id = row[0]
                        if not canonical_id:
                            canonical_id = item_ids[0]  # First alphabetically

                        # Delete non-canonical rows (in FK order)
                        non_canonical = [iid for iid in item_ids if iid != canonical_id]
                        if non_canonical:
                            cur.execute("DELETE FROM mulan_embedding WHERE item_id = ANY(%s)", (non_canonical,))
                            cur.execute("DELETE FROM clap_embedding WHERE item_id = ANY(%s)", (non_canonical,))
                            cur.execute("DELETE FROM embedding WHERE item_id = ANY(%s)", (non_canonical,))
                            cur.execute("DELETE FROM score WHERE item_id = ANY(%s)", (non_canonical,))
                            total_deleted += len(non_canonical)

                cur.execute("""
                    INSERT INTO app_settings (key, value, updated_at)
                    VALUES ('migration_dedup_score_rows_done', 'true', NOW())
                    ON CONFLICT (key) DO NOTHING
                """)
                db.commit()
                logger.info(f"Score dedup migration complete: {total_deleted} duplicate rows removed from {len(dup_groups)} track groups")
            except Exception as e:
                db.rollback()
                logger.warning(f"Score dedup migration failed (will retry next startup): {e}")

        # Migration: Encrypt existing unencrypted provider configs
        cur.execute("SELECT value FROM app_settings WHERE key = 'migration_encrypt_provider_configs_done'")
        if not cur.fetchone():
            try:
                cur.execute("SELECT id, config FROM provider")
                for row in cur.fetchall():
                    provider_id, pconfig = row
                    if pconfig and isinstance(pconfig, dict):
                        needs_encrypt = any(
                            k in pconfig and pconfig[k] and not str(pconfig[k]).startswith('gAAAAA')
                            for k in SENSITIVE_CONFIG_KEYS
                        )
                        if needs_encrypt:
                            encrypted = encrypt_provider_config(pconfig)
                            cur.execute("UPDATE provider SET config = %s WHERE id = %s",
                                        (json.dumps(encrypted), provider_id))
                cur.execute("""
                    INSERT INTO app_settings (key, value, updated_at)
                    VALUES ('migration_encrypt_provider_configs_done', 'true', NOW())
                    ON CONFLICT (key) DO NOTHING
                """)
                db.commit()
                logger.info("Encrypted existing provider configs")
            except Exception as e:
                db.rollback()
                logger.warning(f"Provider config encryption migration failed: {e}")

        db.commit()

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
        cur.execute("SELECT task_id, status, details, task_type FROM task_status WHERE status IN %s AND parent_task_id IS NULL", (non_terminal_statuses,))
        tasks_to_archive = cur.fetchall()

        archived_count = 0
        deleted_children_count = 0
        
        for task_row in tasks_to_archive:
            task_id = task_row['task_id']
            original_status = task_row['status']
            
            original_details_json = task_row['details']
            original_status_message = f"Task was in '{original_status}' state."

            if original_details_json:
                try:
                    original_details_dict = json.loads(original_details_json)
                    original_status_message = original_details_dict.get("status_message", original_status_message)
                except (json.JSONDecodeError, TypeError):
                     logger.warning(f"Could not parse original details for task {task_id} during archival.")

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

def save_track_analysis_and_embedding(item_id, title, author, tempo, key, scale, moods, embedding_vector, energy=None, other_features=None, album=None, album_artist=None, year=None, rating=None, file_path=None, provider_id=None):
    """Saves track analysis and embedding in a single transaction.

    Also creates/updates track linking for multi-provider support when file_path is provided.

    Args:
        item_id: Provider-specific track identifier
        title: Track title
        author: Artist name
        tempo: BPM
        key: Musical key
        scale: Major/Minor scale
        moods: Dict of mood labels and scores
        embedding_vector: numpy array of embeddings
        energy: Energy level (0.01-0.15)
        other_features: JSON string of additional features
        album: Album name
        album_artist: Album artist name
        year: Release year
        rating: User rating
        file_path: Full path to the audio file (for multi-provider track linking)
        provider_id: Optional provider ID for creating provider_track link
    """

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
    file_path = _sanitize_string(file_path, max_length=2000, field_name="file_path")

    # Normalize file_path for consistent cross-provider matching
    if file_path:
        normalized_fp = normalize_provider_path(file_path)
        if normalized_fp:
            file_path = normalized_fp

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

    mood_str = ','.join(f"{k}:{v:.3f}" for k, v in moods.items())

    conn = get_db() # This now calls the function within this file
    cur = conn.cursor()
    try:
        # Save analysis to score table (includes file_path for multi-provider linking)
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

    # Create track linking for multi-provider support (after main transaction commits)
    if file_path:
        try:
            # Get or create track record based on file path
            # Pass provider_id for provider-specific path normalization (music_path_prefix)
            track_id = get_or_create_track(file_path, provider_id=provider_id)
            if track_id:
                # Link score to track
                update_score_track_id(item_id, track_id)

                # Create provider_track link if provider_id is specified
                if provider_id:
                    link_provider_track(provider_id, track_id, item_id, title, author, album)
        except Exception as e:
            # Log but don't fail - track linking is supplementary
            logger.warning("Failed to create track linking for %s: %s", item_id, e)

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
    """Helper to cancel a job and its children based on DB records."""
    cancelled_count = 0

    # First, determine the task_type for the current job_id
    db_task_info = get_task_info_from_db(job_id)
    current_task_type = db_task_info.get('task_type') if db_task_info else task_type_from_db

    if not current_task_type:
        logger.warning(f"Could not determine task_type for job {job_id}. Cannot reliably mark as REVOKED in DB or cancel children.")
        try:
            Job.fetch(job_id, connection=redis_conn)
            send_stop_job_command(redis_conn, job_id)
            cancelled_count += 1
            logger.info(f"Job {job_id} (task_type unknown) stop command sent to RQ.")
        except NoSuchJobError:
            pass
        return cancelled_count

    # Mark as REVOKED in DB for the current job. This is the primary action.
    save_task_status(job_id, current_task_type, TASK_STATUS_REVOKED, progress=100, details={"message": reason})

    # Attempt to stop the job in RQ. This is a secondary action to interrupt a running process.
    action_taken_in_rq = False
    try:
        job_rq = Job.fetch(job_id, connection=redis_conn)
        current_rq_status = job_rq.get_status()
        logger.info(f"Job {job_id} (type: {current_task_type}) found in RQ with status: {current_rq_status}")

        if not job_rq.is_finished and not job_rq.is_failed and not job_rq.is_canceled:
            if job_rq.is_started:
                send_stop_job_command(redis_conn, job_id)
            else:
                job_rq.cancel()
            action_taken_in_rq = True
            logger.info(f"  Sent stop/cancel command for job {job_id} in RQ.")
        else:
            logger.info(f"  Job {job_id} is already in a terminal RQ state: {current_rq_status}.")

    except NoSuchJobError:
        logger.warning(f"Job {job_id} (type: {current_task_type}) not found in RQ, but marked as REVOKED in DB.")
    except Exception as e_rq_interaction:
        logger.error(f"Error interacting with RQ for job {job_id}: {e_rq_interaction}")

    if action_taken_in_rq:
        cancelled_count += 1

    # Recursively cancel children found in the database
    children_tasks = get_child_tasks_from_db(job_id)
    
    for child_task in children_tasks:
        child_job_id = child_task['task_id']
        # We only need to proceed if the child is not already in a terminal state
        child_db_info = get_task_info_from_db(child_job_id)
        if child_db_info and child_db_info.get('status') not in [TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED]:
             logger.info(f"Recursively cancelling child job: {child_job_id}")
             cancelled_count += cancel_job_and_children_recursive(child_job_id, reason="Cancelled due to parent task revocation.")

    return cancelled_count


# ##############################################################################
# MULTI-PROVIDER HELPER FUNCTIONS
# ##############################################################################

def get_primary_provider_id():
    """Get the primary provider ID from app_settings."""
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT value FROM app_settings WHERE key = 'primary_provider_id'")
        row = cur.fetchone()
        if row and row[0] is not None:
            try:
                # Value is stored as JSONB, could be int or null
                val = row[0]
                if isinstance(val, int):
                    return val
                if val is None or val == 'null':
                    return None
                return int(val)
            except (ValueError, TypeError):
                return None
        return None


def get_enabled_provider_ids():
    """Get list of enabled provider IDs ordered by priority (highest first)."""
    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            SELECT id FROM provider
            WHERE enabled = TRUE
            ORDER BY priority DESC, created_at ASC
        """)
        return [row[0] for row in cur.fetchall()]


def get_track_by_item_id(item_id, provider_id=None):
    """
    Look up a track by item_id with provider fallback logic.

    If provider_id is specified:
        - Look up in provider_track for that provider first
        - Fall back to score table if not in provider_track

    If provider_id is NOT specified (backward compatible mode):
        1. Try the primary provider first
        2. Try other enabled providers in priority order
        3. Fall back to direct score table lookup (legacy mode)

    Returns:
        dict with track info or None if not found
    """
    db = get_db()

    def lookup_in_score(item_id):
        """Direct lookup in score table (legacy mode)."""
        with db.cursor() as cur:
            cur.execute("""
                SELECT item_id, title, author, album, tempo, key, scale,
                       mood_vector, energy, other_features, file_path, track_id
                FROM score WHERE item_id = %s
            """, (item_id,))
            row = cur.fetchone()
            if row:
                return {
                    'item_id': row[0],
                    'title': row[1],
                    'author': row[2],
                    'album': row[3],
                    'tempo': row[4],
                    'key': row[5],
                    'scale': row[6],
                    'mood_vector': row[7],
                    'energy': row[8],
                    'other_features': row[9],
                    'file_path': row[10],
                    'track_id': row[11],
                    'provider_id': None  # Unknown provider in legacy mode
                }
        return None

    def lookup_via_provider(item_id, prov_id):
        """Look up via provider_track table."""
        with db.cursor() as cur:
            # First check provider_track
            cur.execute("""
                SELECT pt.item_id, pt.title, pt.artist, pt.album, pt.track_id,
                       s.tempo, s.key, s.scale, s.mood_vector, s.energy,
                       s.other_features, s.file_path
                FROM provider_track pt
                LEFT JOIN score s ON (
                    pt.item_id = s.item_id OR
                    (pt.track_id IS NOT NULL AND pt.track_id = s.track_id)
                )
                WHERE pt.provider_id = %s AND pt.item_id = %s
            """, (prov_id, item_id))
            row = cur.fetchone()
            if row:
                return {
                    'item_id': row[0],
                    'title': row[1],
                    'author': row[2],
                    'album': row[3],
                    'track_id': row[4],
                    'tempo': row[5],
                    'key': row[6],
                    'scale': row[7],
                    'mood_vector': row[8],
                    'energy': row[9],
                    'other_features': row[10],
                    'file_path': row[11],
                    'provider_id': prov_id
                }
        return None

    # If provider_id specified, try that provider first then fall back
    if provider_id is not None:
        result = lookup_via_provider(item_id, provider_id)
        if result:
            return result
        # Fall back to direct score lookup
        return lookup_in_score(item_id)

    # No provider specified - use fallback logic
    # 1. Try primary provider first
    primary_id = get_primary_provider_id()
    if primary_id:
        result = lookup_via_provider(item_id, primary_id)
        if result:
            return result

    # 2. Try other enabled providers in priority order
    enabled_ids = get_enabled_provider_ids()
    for prov_id in enabled_ids:
        if prov_id == primary_id:
            continue  # Already tried
        result = lookup_via_provider(item_id, prov_id)
        if result:
            return result

    # 3. Fall back to direct score table lookup (legacy/backward compatible)
    return lookup_in_score(item_id)


def get_tracks_by_item_ids(item_ids, provider_id=None):
    """
    Look up multiple tracks by item_ids with provider fallback logic.

    Args:
        item_ids: List of item IDs to look up
        provider_id: Optional provider ID to scope the lookup

    Returns:
        dict mapping item_id to track info
    """
    if not item_ids:
        return {}

    results = {}
    for item_id in item_ids:
        track = get_track_by_item_id(item_id, provider_id)
        if track:
            results[item_id] = track

    return results


def resolve_item_id_to_provider(item_id):
    """
    Resolve which provider(s) know about a given item_id.

    Returns:
        List of provider_ids that have this item_id,
        or empty list if only in score table (legacy)
    """
    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT provider_id FROM provider_track
            WHERE item_id = %s
        """, (item_id,))
        return [row[0] for row in cur.fetchall()]


def get_item_id_for_provider(file_path_or_track_id, provider_id):
    """
    Get the provider-specific item_id for a track.

    Useful when you have analysis data linked to one provider
    and need to find the equivalent track in another provider.

    Args:
        file_path_or_track_id: Either file path (str) or track_id (int)
        provider_id: The provider to look up in

    Returns:
        The item_id for that provider, or None if not found
    """
    db = get_db()
    with db.cursor() as cur:
        if isinstance(file_path_or_track_id, int):
            # Lookup by track_id
            cur.execute("""
                SELECT item_id FROM provider_track
                WHERE provider_id = %s AND track_id = %s
            """, (provider_id, file_path_or_track_id))
        else:
            # Lookup by file path - need to join through track table
            cur.execute("""
                SELECT pt.item_id FROM provider_track pt
                JOIN track t ON pt.track_id = t.id
                WHERE pt.provider_id = %s AND t.file_path = %s
            """, (provider_id, file_path_or_track_id))

        row = cur.fetchone()
        return row[0] if row else None


def is_multi_provider_mode():
    """Check if multi-provider mode is enabled."""
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT value FROM app_settings WHERE key = 'multi_provider_enabled'")
        row = cur.fetchone()
        if row:
            val = row[0]
            return val is True or val == True or val == 'true'
        return False


def set_primary_provider(provider_id):
    """Set the primary provider ID."""
    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            INSERT INTO app_settings (key, value, category, description, updated_at)
            VALUES ('primary_provider_id', %s, 'providers', 'ID of the primary provider', NOW())
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = NOW()
        """, (str(provider_id) if provider_id is not None else 'null',))
        db.commit()


# ##############################################################################
# CREDENTIAL ENCRYPTION
# ##############################################################################

SENSITIVE_CONFIG_KEYS = {'token', 'password', 'api_key'}

def _get_fernet():
    """Get or create a Fernet cipher using ENCRYPTION_KEY from config/env."""
    from cryptography.fernet import Fernet
    import config as _config
    key = getattr(_config, 'ENCRYPTION_KEY', None) or os.environ.get('ENCRYPTION_KEY')
    if not key:
        # Check app_settings for a previously generated key
        db = get_db()
        with db.cursor() as cur:
            cur.execute("SELECT value FROM app_settings WHERE key = 'encryption_key'")
            row = cur.fetchone()
            if row and row[0]:
                key = row[0] if isinstance(row[0], str) else str(row[0])
                # Strip JSON quotes if present
                key = key.strip('"')
        if not key:
            # Auto-generate and persist
            key = Fernet.generate_key().decode()
            db = get_db()
            with db.cursor() as cur:
                cur.execute("""
                    INSERT INTO app_settings (key, value, updated_at)
                    VALUES ('encryption_key', %s, NOW())
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """, (json.dumps(key),))
                db.commit()
            logger.info("Generated and stored new encryption key")
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_provider_config(config_dict):
    """Encrypt sensitive fields in a provider config dict before storage."""
    if not config_dict or not isinstance(config_dict, dict):
        return config_dict
    encrypted = dict(config_dict)
    try:
        f = _get_fernet()
        for k in SENSITIVE_CONFIG_KEYS:
            if k in encrypted and encrypted[k] and not str(encrypted[k]).startswith('gAAAAA'):
                encrypted[k] = f.encrypt(str(encrypted[k]).encode()).decode()
    except Exception as e:
        logger.error(f"Failed to encrypt provider config: {e}")
    return encrypted


def decrypt_provider_config(config_dict):
    """Decrypt sensitive fields in a provider config dict after retrieval."""
    if not config_dict or not isinstance(config_dict, dict):
        return config_dict
    decrypted = dict(config_dict)
    try:
        f = _get_fernet()
        for k in SENSITIVE_CONFIG_KEYS:
            if k in decrypted and decrypted[k] and str(decrypted[k]).startswith('gAAAAA'):
                try:
                    decrypted[k] = f.decrypt(str(decrypted[k]).encode()).decode()
                except Exception:
                    pass  # Not encrypted or wrong key, leave as-is
    except Exception as e:
        logger.error(f"Failed to decrypt provider config: {e}")
    return decrypted


# ##############################################################################
# TRACK LINKING FUNCTIONS - For multi-provider track identity
# ##############################################################################

def normalize_provider_path(file_path, provider_id=None):
    """
    Normalize a file path for cross-provider matching.

    Handles different provider path formats:
    - Jellyfin: /media/music/Library/Artist/Album/song.mp3
    - Navidrome: Artist/Album/song.mp3 (no library folder)
    - Lyrion: file:///music/Artist/Album/song.mp3
    - Local: /music/Artist/Album/song.mp3

    Normalization steps:
    1. Strip file:// URL prefix and URL-decode
    2. Strip common mount points (/media/music, /music, etc.)
    3. Strip provider-specific music_path_prefix from config (e.g., "MyLibrary/")
    4. Convert backslashes to forward slashes
    5. Remove leading slashes

    Args:
        file_path: The file path to normalize
        provider_id: Optional provider ID to get provider-specific path prefix

    Returns:
        Normalized relative path (e.g., "Artist/Album/song.mp3")
    """
    from urllib.parse import unquote

    if not file_path:
        return None

    normalized = file_path

    # Handle file:// URLs (Lyrion/LMS style)
    if normalized.startswith('file://'):
        normalized = normalized[7:]  # Remove 'file://'
        normalized = unquote(normalized)  # URL-decode

    # Convert Windows backslashes to forward slashes
    normalized = normalized.replace('\\', '/')

    # List of common mount point prefixes to strip (order matters - longer first)
    prefixes_to_strip = [
        '/media/music/',      # Common Jellyfin mount
        '/media/Media/',      # Alternate Jellyfin
        '/media/',            # Generic media mount
        '/mnt/media/music/',  # Mount point style
        '/mnt/media/',        # Mount point style
        '/mnt/music/',        # Mount point style
        '/mnt/data/music/',   # Data volume style
        '/mnt/data/',         # Data volume style
        '/mnt/',              # Generic mount
        '/data/music/',       # Data volume style
        '/data/',             # Data volume style
        '/music/',            # Direct music mount
        '/share/music/',      # NAS style
        '/share/',            # NAS style
        '/volume1/music/',    # Synology style
        '/volume1/',          # Synology style
        '/srv/music/',        # Some Linux systems
        '/srv/',              # Some Linux systems
        '/home/music/',       # Home directory music
        '/storage/music/',    # Some NAS/Android
        '/opt/music/',        # Some deployments
        '/nas/music/',        # NAS direct mount
        '/library/music/',    # macOS-style
    ]

    # Strip mount point prefixes (case-insensitive)
    lower_normalized = normalized.lower()
    for prefix in prefixes_to_strip:
        if lower_normalized.startswith(prefix.lower()):
            normalized = normalized[len(prefix):]
            break

    # Remove any remaining leading slashes
    normalized = normalized.lstrip('/')

    # Handle Windows absolute paths
    if len(normalized) > 1 and normalized[1] == ':':
        for marker in ['/music/', '/Music/', '/media/', '/Media/']:
            idx = normalized.find(marker)
            if idx != -1:
                normalized = normalized[idx + len(marker):]
                break

    # Strip provider-specific music_path_prefix if configured
    # This handles cases like Jellyfin including "MyLibrary/" but Navidrome not
    if provider_id:
        try:
            from app_setup import get_provider_by_id
            provider = get_provider_by_id(provider_id)
            if provider and provider.get('config'):
                music_prefix = provider['config'].get('music_path_prefix', '')
                if music_prefix:
                    music_prefix = music_prefix.replace('\\', '/').strip('/')
                    if music_prefix and normalized.lower().startswith(music_prefix.lower()):
                        # Strip the prefix plus any following slash
                        prefix_len = len(music_prefix)
                        if len(normalized) > prefix_len and normalized[prefix_len] == '/':
                            prefix_len += 1
                        normalized = normalized[prefix_len:]
        except Exception:
            pass  # Ignore errors - continue with standard normalization

    result = normalized.lstrip('/') if normalized else None
    return result.lower() if result else None


def _compute_file_path_hash(file_path, provider_id=None):
    """
    Compute SHA-256 hash of normalized file path for track identity.

    Args:
        file_path: The file path to hash
        provider_id: Optional provider ID for provider-specific normalization

    Returns:
        SHA-256 hash string or None if path is empty
    """
    import hashlib

    normalized = normalize_provider_path(file_path, provider_id)
    if not normalized:
        return None

    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def get_or_create_track(file_path, file_size=None, file_modified=None, provider_id=None):
    """
    Get or create a track record based on file path.

    The track table provides stable identity across providers based on file path.
    Uses provider-specific path normalization if provider_id is given.

    Args:
        file_path: Full or relative path to the audio file
        file_size: Optional file size in bytes
        file_modified: Optional file modification timestamp
        provider_id: Optional provider ID for path normalization (uses music_path_prefix from config)

    Returns:
        track_id (int) or None if file_path is empty
    """
    if not file_path:
        return None

    # Get normalized path (without provider-specific prefix)
    normalized_path = normalize_provider_path(file_path, provider_id)
    if not normalized_path:
        return None

    # Compute hash of the normalized path
    import hashlib
    file_path_hash = hashlib.sha256(normalized_path.encode('utf-8')).hexdigest()

    db = get_db()
    with db.cursor() as cur:
        # Try to get existing track by normalized path hash
        cur.execute("SELECT id FROM track WHERE file_path_hash = %s", (file_path_hash,))
        row = cur.fetchone()

        if row:
            track_id = row[0]
            # Update file info and normalized_path if provided
            updates = ["updated_at = NOW()"]
            values = []
            if file_size is not None:
                updates.append("file_size = %s")
                values.append(file_size)
            if file_modified is not None:
                updates.append("file_modified = %s")
                values.append(file_modified)
            # Always update normalized_path to latest
            updates.append("normalized_path = %s")
            values.append(normalized_path)
            values.append(track_id)
            cur.execute(f"UPDATE track SET {', '.join(updates)} WHERE id = %s", values)
            db.commit()
            return track_id

        # Create new track with normalized_path
        cur.execute("""
            INSERT INTO track (file_path_hash, file_path, normalized_path, file_size, file_modified)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (file_path_hash, file_path, normalized_path, file_size, file_modified))
        track_id = cur.fetchone()[0]
        db.commit()
        return track_id


def link_provider_track(provider_id, track_id, item_id, title=None, artist=None, album=None):
    """
    Link a provider's item_id to a track.

    Creates or updates the provider_track mapping that links a provider's
    native item_id to the stable track identity.

    Args:
        provider_id: ID of the provider
        track_id: ID of the track in the track table
        item_id: Provider's native item identifier
        title: Track title from this provider
        artist: Artist name from this provider
        album: Album name from this provider

    Returns:
        provider_track id or None on failure
    """
    if not provider_id or not track_id or not item_id:
        return None

    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            INSERT INTO provider_track (provider_id, track_id, item_id, title, artist, album, last_synced)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (provider_id, item_id) DO UPDATE SET
                track_id = EXCLUDED.track_id,
                title = COALESCE(EXCLUDED.title, provider_track.title),
                artist = COALESCE(EXCLUDED.artist, provider_track.artist),
                album = COALESCE(EXCLUDED.album, provider_track.album),
                last_synced = NOW()
            RETURNING id
        """, (provider_id, track_id, item_id, title, artist, album))
        result = cur.fetchone()
        db.commit()
        return result[0] if result else None


def update_score_track_id(item_id, track_id):
    """
    Update the track_id reference in the score table.

    This links the analysis data to the stable track identity.

    Args:
        item_id: The item_id in the score table
        track_id: The track_id to link to
    """
    if not item_id or not track_id:
        return

    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            UPDATE score SET track_id = %s WHERE item_id = %s AND (track_id IS NULL OR track_id != %s)
        """, (track_id, item_id, track_id))
        db.commit()


def get_track_by_file_path(file_path):
    """
    Get track info by file path.

    Args:
        file_path: Full or relative path to the audio file

    Returns:
        dict with track info or None if not found
    """
    if not file_path:
        return None

    file_path_hash = _compute_file_path_hash(file_path)
    if not file_path_hash:
        return None

    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            SELECT t.id, t.file_path, t.file_path_hash, t.file_size, t.file_modified,
                   s.item_id, s.title, s.author, s.album, s.tempo, s.key, s.scale,
                   s.mood_vector, s.energy, s.other_features
            FROM track t
            LEFT JOIN score s ON s.track_id = t.id
            WHERE t.file_path_hash = %s
        """, (file_path_hash,))
        row = cur.fetchone()
        if row:
            return {
                'track_id': row[0],
                'file_path': row[1],
                'file_path_hash': row[2],
                'file_size': row[3],
                'file_modified': row[4],
                'item_id': row[5],
                'title': row[6],
                'author': row[7],
                'album': row[8],
                'tempo': row[9],
                'key': row[10],
                'scale': row[11],
                'mood_vector': row[12],
                'energy': row[13],
                'other_features': row[14],
            }
        return None


def get_all_provider_item_ids_for_track(track_id):
    """
    Get all provider item_ids linked to a track.

    Args:
        track_id: The track ID

    Returns:
        List of dicts with provider_id, item_id, title, artist, album
    """
    if not track_id:
        return []

    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            SELECT pt.provider_id, pt.item_id, pt.title, pt.artist, pt.album,
                   p.provider_type, p.name as provider_name
            FROM provider_track pt
            JOIN provider p ON p.id = pt.provider_id
            WHERE pt.track_id = %s
        """, (track_id,))
        return [
            {
                'provider_id': row[0],
                'item_id': row[1],
                'title': row[2],
                'artist': row[3],
                'album': row[4],
                'provider_type': row[5],
                'provider_name': row[6],
            }
            for row in cur.fetchall()
        ]


def find_existing_analysis_by_file_path(file_path, provider_id=None, title=None, artist=None, album=None):
    """
    Find existing analysis data for a file path using cross-provider matching.

    This is used to check if a track has already been analyzed under a different
    provider's item_id, allowing reuse of analysis data in multi-provider setups.

    Uses a 3-tier lookup strategy:
    1. Hash-based match via track table (fastest, most reliable)
    2. Direct file_path match in score table (legacy fallback)
    3. Metadata match by title + artist + album (handles different mount points)

    Args:
        file_path: Full or relative path to the audio file
        provider_id: Optional provider ID for provider-specific path normalization
        title: Optional track title for metadata-based fallback matching
        artist: Optional artist name for metadata-based fallback matching
        album: Optional album name for metadata-based fallback matching

    Returns:
        dict with item_id and analysis status, or None if not found
    """
    if not file_path and not (title and artist and album):
        return None

    db = get_db()
    with db.cursor() as cur:
        # Fallback 1 & 2: Hash-based and direct file_path matching (require file_path)
        if file_path:
            file_path_hash = _compute_file_path_hash(file_path, provider_id)

            if file_path_hash:
                # First try via track table (proper linking)
                cur.execute("""
                    SELECT s.item_id, s.title, s.author, s.track_id,
                           (s.tempo IS NOT NULL) as has_musicnn,
                           EXISTS(SELECT 1 FROM embedding e WHERE e.item_id = s.item_id) as has_embedding,
                           EXISTS(SELECT 1 FROM clap_embedding ce WHERE ce.item_id = s.item_id) as has_clap
                    FROM track t
                    JOIN score s ON s.track_id = t.id
                    WHERE t.file_path_hash = %s
                """, (file_path_hash,))
                row = cur.fetchone()
                if row:
                    return {
                        'item_id': row[0],
                        'title': row[1],
                        'author': row[2],
                        'track_id': row[3],
                        'has_musicnn': row[4],
                        'has_embedding': row[5],
                        'has_clap': row[6],
                        'source': 'track_table'
                    }

            # Fall back to checking score.file_path directly (for legacy data)
            cur.execute("""
                SELECT s.item_id, s.title, s.author, s.track_id,
                       (s.tempo IS NOT NULL) as has_musicnn,
                       EXISTS(SELECT 1 FROM embedding e WHERE e.item_id = s.item_id) as has_embedding,
                       EXISTS(SELECT 1 FROM clap_embedding ce WHERE ce.item_id = s.item_id) as has_clap
                FROM score s
                WHERE s.file_path = %s
            """, (file_path,))
            row = cur.fetchone()
            if row:
                return {
                    'item_id': row[0],
                    'title': row[1],
                    'author': row[2],
                    'track_id': row[3],
                    'has_musicnn': row[4],
                    'has_embedding': row[5],
                    'has_clap': row[6],
                    'source': 'score_file_path'
                }

        # Fallback 3: Match by title + artist + album metadata
        # This handles cases where file paths differ across providers
        # (different mount points, relative vs absolute paths, etc.)
        if title and artist and album:
            cur.execute("""
                SELECT s.item_id, s.title, s.author, s.track_id,
                       (s.tempo IS NOT NULL) as has_musicnn,
                       EXISTS(SELECT 1 FROM embedding e WHERE e.item_id = s.item_id) as has_embedding,
                       EXISTS(SELECT 1 FROM clap_embedding ce WHERE ce.item_id = s.item_id) as has_clap
                FROM score s
                WHERE LOWER(s.title) = LOWER(%s)
                  AND LOWER(s.author) = LOWER(%s)
                  AND LOWER(s.album) = LOWER(%s)
                  AND s.track_id IS NOT NULL
                  AND s.tempo IS NOT NULL
                LIMIT 1
            """, (title, artist, album))
            row = cur.fetchone()
            if row:
                return {
                    'item_id': row[0],
                    'title': row[1],
                    'author': row[2],
                    'track_id': row[3],
                    'has_musicnn': row[4],
                    'has_embedding': row[5],
                    'has_clap': row[6],
                    'source': 'metadata_match'
                }

        return None


def link_provider_to_existing_track(file_path, provider_id, item_id, title=None, artist=None, album=None):
    """
    Link a new provider's item_id to an already-analyzed track via provider_track.

    Unlike copy_analysis_to_new_item(), this does NOT duplicate score/embedding rows.
    It only creates a provider_track mapping so the provider's item_id resolves
    to the existing canonical track.

    Args:
        file_path: File path of the track (used to find/create the track record)
        provider_id: ID of the provider being linked
        item_id: The provider's native item identifier
        title: Optional track title from this provider
        artist: Optional artist name from this provider
        album: Optional album name from this provider

    Returns:
        True if linking succeeded, False otherwise
    """
    if not file_path or not provider_id or not item_id:
        return False
    try:
        track_id = get_or_create_track(file_path, provider_id=provider_id)
        if not track_id:
            return False
        link_provider_track(provider_id, track_id, item_id, title, artist, album)
        logger.info(f"Linked provider {provider_id} item {item_id} to track {track_id} (no row duplication)")
        return True
    except Exception as e:
        logger.error(f"Failed to link provider track for {item_id}: {e}")
        return False


def copy_analysis_to_new_item(source_item_id, target_item_id, file_path=None, provider_id=None):
    """
    DEPRECATED: Use link_provider_to_existing_track() instead.

    Copy analysis data from one item_id to another.
    This duplicates score + embedding rows, which wastes storage and causes
    duplicate Voyager index entries. Kept temporarily for migration.

    Args:
        source_item_id: The item_id that has existing analysis
        target_item_id: The new item_id to copy analysis to
        file_path: Optional file path for track linking
        provider_id: Optional provider ID for track linking

    Returns:
        True if analysis was copied successfully, False otherwise
    """
    if not source_item_id or not target_item_id:
        return False

    if source_item_id == target_item_id:
        return True  # Nothing to copy

    db = get_db()
    try:
        with db.cursor() as cur:
            # Copy score data
            cur.execute("""
                INSERT INTO score (item_id, title, author, tempo, key, scale, mood_vector,
                                   energy, other_features, album, album_artist, year, rating, file_path, track_id)
                SELECT %s, title, author, tempo, key, scale, mood_vector,
                       energy, other_features, album, album_artist, year, rating, file_path, track_id
                FROM score WHERE item_id = %s
                ON CONFLICT (item_id) DO NOTHING
            """, (target_item_id, source_item_id))

            # Copy embedding
            cur.execute("""
                INSERT INTO embedding (item_id, embedding)
                SELECT %s, embedding FROM embedding WHERE item_id = %s
                ON CONFLICT (item_id) DO NOTHING
            """, (target_item_id, source_item_id))

            # Copy CLAP embedding if exists
            cur.execute("""
                INSERT INTO clap_embedding (item_id, embedding)
                SELECT %s, embedding FROM clap_embedding WHERE item_id = %s
                ON CONFLICT (item_id) DO NOTHING
            """, (target_item_id, source_item_id))

            # Copy MuLan embedding if exists
            cur.execute("""
                INSERT INTO mulan_embedding (item_id, embedding)
                SELECT %s, embedding FROM mulan_embedding WHERE item_id = %s
                ON CONFLICT (item_id) DO NOTHING
            """, (target_item_id, source_item_id))

            db.commit()

            # Create track linking for the new item
            if file_path:
                track_id = get_or_create_track(file_path, provider_id=provider_id)
                if track_id:
                    update_score_track_id(target_item_id, track_id)
                    if provider_id is not None:
                        link_provider_track(provider_id, track_id, target_item_id)

            logger.info(f"Copied analysis from {source_item_id} to {target_item_id}")
            return True

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to copy analysis from {source_item_id} to {target_item_id}: {e}")
        return False


def detect_music_path_prefix(sample_tracks, existing_normalized_paths=None, extra_sample_tracks=None):
    """
    Auto-detect the music_path_prefix for a new provider by comparing paths.

    Takes sample tracks from a new provider and compares their paths with
    existing normalized paths in the database to find the prefix difference.

    Args:
        sample_tracks: List of track dicts with 'title', 'artist', 'file_path' keys
        existing_normalized_paths: Optional dict mapping (title, artist) -> normalized_path
                                   If None, fetches from database
        extra_sample_tracks: Optional dict mapping provider_type -> list of track dicts
                            from previously tested providers (for setup wizard flow)

    Returns:
        dict with:
            - detected_prefix: The detected prefix to strip (or empty string)
            - confidence: 'high', 'medium', 'low', or 'none'
            - matches_found: Number of matching tracks found
            - sample_comparisons: List of example path comparisons
    """
    from urllib.parse import unquote

    if not sample_tracks:
        return {'detected_prefix': '', 'confidence': 'none', 'matches_found': 0, 'sample_comparisons': []}

    # Get existing normalized paths from database if not provided
    if existing_normalized_paths is None:
        existing_normalized_paths = {}
        db = get_db()
        with db.cursor() as cur:
            # Get normalized paths from track table
            cur.execute("""
                SELECT LOWER(pt.title), LOWER(pt.artist), t.normalized_path
                FROM provider_track pt
                JOIN track t ON pt.track_id = t.id
                WHERE t.normalized_path IS NOT NULL
                  AND pt.title IS NOT NULL
                  AND pt.artist IS NOT NULL
            """)
            for row in cur.fetchall():
                key = (row[0], row[1])
                if key not in existing_normalized_paths:
                    existing_normalized_paths[key] = row[2]

            # Also check score table for legacy data
            cur.execute("""
                SELECT LOWER(title), LOWER(author), file_path
                FROM score
                WHERE file_path IS NOT NULL
                  AND title IS NOT NULL
                  AND author IS NOT NULL
            """)
            for row in cur.fetchall():
                key = (row[0], row[1])
                if key not in existing_normalized_paths:
                    # Normalize the legacy path
                    normalized = normalize_provider_path(row[2], provider_id=None)
                    if normalized:
                        existing_normalized_paths[key] = normalized

    # Also add paths from extra_sample_tracks (from previously tested providers in setup wizard)
    if extra_sample_tracks:
        for provider_type, tracks in extra_sample_tracks.items():
            for track in tracks:
                title = track.get('title') or track.get('Name') or track.get('name')
                artist = track.get('artist') or track.get('Artist') or track.get('AlbumArtist')
                file_path = track.get('file_path') or track.get('Path') or track.get('path')
                if title and artist and file_path:
                    key = (title.lower(), artist.lower())
                    if key not in existing_normalized_paths:
                        normalized = normalize_provider_path(file_path, provider_id=None)
                        if normalized:
                            existing_normalized_paths[key] = normalized

    if not existing_normalized_paths:
        return {'detected_prefix': '', 'confidence': 'none', 'matches_found': 0,
                'sample_comparisons': [], 'message': 'No existing tracks to compare with',
                'had_existing_tracks': False}

    # Build a secondary index by filename for fallback matching
    # This helps when title extraction differs between providers (e.g., "01 - Song" vs "Song")
    existing_by_filename = {}
    for (title, artist), norm_path in existing_normalized_paths.items():
        if norm_path:
            # Extract filename from normalized path
            filename = norm_path.split('/')[-1].lower() if '/' in norm_path else norm_path.lower()
            if filename and filename not in existing_by_filename:
                existing_by_filename[filename] = norm_path

    # Normalize the sample paths (basic normalization without provider prefix)
    def basic_normalize(path):
        """Basic path normalization without provider-specific prefix stripping."""
        if not path:
            return None
        normalized = path
        # Handle file:// URLs
        if normalized.startswith('file://'):
            normalized = normalized[7:]
            normalized = unquote(normalized)
        normalized = normalized.replace('\\', '/')
        # Strip common mount points (synced with normalize_provider_path)
        prefixes = [
            '/media/music/', '/media/Media/', '/media/',
            '/mnt/media/music/', '/mnt/media/', '/mnt/music/',
            '/mnt/data/music/', '/mnt/data/', '/mnt/',
            '/data/music/', '/data/', '/music/',
            '/share/music/', '/share/',
            '/volume1/music/', '/volume1/',
            '/srv/music/', '/srv/',
            '/home/music/', '/storage/music/',
            '/opt/music/', '/nas/music/', '/library/music/',
        ]
        lower = normalized.lower()
        for prefix in prefixes:
            if lower.startswith(prefix.lower()):
                normalized = normalized[len(prefix):]
                break
        return normalized.lstrip('/')

    # Find matches and compare paths
    matches = []
    for track in sample_tracks:
        title = track.get('title') or track.get('Name') or track.get('name')
        artist = track.get('artist') or track.get('Artist') or track.get('AlbumArtist')
        file_path = track.get('file_path') or track.get('Path') or track.get('path')

        if not file_path:
            continue

        new_normalized = basic_normalize(file_path)
        if not new_normalized:
            continue

        # Try matching by title+artist first
        existing_path = None
        if title and artist:
            key = (title.lower(), artist.lower())
            existing_path = existing_normalized_paths.get(key)

        # Fallback: match by filename (helps when title extraction differs between providers)
        if not existing_path:
            new_filename = new_normalized.split('/')[-1].lower() if '/' in new_normalized else new_normalized.lower()
            existing_path = existing_by_filename.get(new_filename)

        if existing_path:
            matches.append({
                'title': title or '(unknown)',
                'artist': artist or '(unknown)',
                'new_path': new_normalized,
                'existing_path': existing_path
            })

    if not matches:
        return {'detected_prefix': '', 'confidence': 'none', 'matches_found': 0,
                'sample_comparisons': [], 'message': 'No matching tracks found between providers',
                'had_existing_tracks': True}

    # Detect prefix by finding common suffix (segment-based for robustness)
    prefix_candidates = {}
    sample_comparisons = []

    for match in matches[:20]:  # Limit analysis to first 20 matches
        new_path = match['new_path']
        existing_path = match['existing_path']

        # Check if new path ends with existing path (case-insensitive)
        if new_path.lower().endswith(existing_path.lower()):
            # The prefix is what's before the existing path
            prefix_len = len(new_path) - len(existing_path)
            prefix = new_path[:prefix_len].rstrip('/')
            if prefix:
                prefix_candidates[prefix] = prefix_candidates.get(prefix, 0) + 1

            sample_comparisons.append({
                'title': match['title'],
                'new_path': new_path,
                'existing_path': existing_path,
                'detected_prefix': prefix
            })
        elif existing_path.lower().endswith(new_path.lower()):
            # The existing path has a prefix (new provider doesn't have it)
            # This means the NEW provider needs no prefix, but existing does
            prefix_candidates[''] = prefix_candidates.get('', 0) + 1

            sample_comparisons.append({
                'title': match['title'],
                'new_path': new_path,
                'existing_path': existing_path,
                'detected_prefix': '(existing has prefix, new does not)'
            })
        else:
            # Segment-based longest-common-suffix matching
            # Handles cases where both providers have residual prefixes
            # (e.g., "multimedia/music/Artist/Album/song.mp3" vs "music/Artist/Album/song.mp3")
            new_segments = new_path.lower().split('/')
            existing_segments = existing_path.lower().split('/')

            common_count = 0
            for i in range(1, min(len(new_segments), len(existing_segments)) + 1):
                if new_segments[-i] == existing_segments[-i]:
                    common_count += 1
                else:
                    break

            if common_count >= 2:  # At least 2 matching segments (e.g., album/song)
                new_prefix = '/'.join(new_path.split('/')[:len(new_segments) - common_count])
                prefix_candidates[new_prefix] = prefix_candidates.get(new_prefix, 0) + 1

                sample_comparisons.append({
                    'title': match['title'],
                    'new_path': new_path,
                    'existing_path': existing_path,
                    'detected_prefix': new_prefix
                })

    if not prefix_candidates:
        return {'detected_prefix': '', 'confidence': 'low', 'matches_found': len(matches),
                'sample_comparisons': sample_comparisons[:5],
                'message': 'Could not detect consistent prefix pattern',
                'had_existing_tracks': True}

    # Find most common prefix
    most_common_prefix = max(prefix_candidates, key=prefix_candidates.get)
    occurrence_count = prefix_candidates[most_common_prefix]
    total_matches = len(matches)

    # Determine confidence
    if occurrence_count == total_matches and total_matches >= 3:
        confidence = 'high'
    elif occurrence_count >= total_matches * 0.8 and total_matches >= 2:
        confidence = 'medium'
    elif occurrence_count >= 1:
        confidence = 'low'
    else:
        confidence = 'none'

    return {
        'detected_prefix': most_common_prefix,
        'confidence': confidence,
        'matches_found': len(matches),
        'prefix_occurrences': occurrence_count,
        'sample_comparisons': sample_comparisons[:5],
        'had_existing_tracks': True
    }


def detect_path_format(sample_tracks):
    """
    Detect whether sample track paths are absolute or relative.

    Navidrome's "Report Real Path" setting controls this:
    - OFF (default): relative paths like "Artist/Album/Track.flac"
    - ON: absolute paths like "/music/Artist/Album/Track.flac"

    Args:
        sample_tracks: List of track dicts with 'file_path' key

    Returns:
        'absolute', 'relative', or 'unknown'
    """
    paths = [t.get('file_path', '') for t in sample_tracks if t]
    paths = [p for p in paths if p]
    if not paths:
        return 'unknown'
    absolute_count = sum(1 for p in paths if p.startswith('/'))
    return 'absolute' if absolute_count > len(paths) / 2 else 'relative'