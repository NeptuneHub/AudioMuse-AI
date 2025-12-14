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
from rq import Queue, Retry
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
redis_conn = Redis.from_url(REDIS_URL, socket_connect_timeout=15, socket_timeout=15)
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
        cur.execute("CREATE TABLE IF NOT EXISTS score (item_id TEXT PRIMARY KEY, title TEXT, author TEXT, tempo REAL, key TEXT, scale TEXT, mood_vector TEXT)")
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
                update_cur.execute(
                    "UPDATE task_status SET status = %s, details = %s, progress = 100, timestamp = NOW() WHERE task_id = %s AND status = %s",
                    (TASK_STATUS_REVOKED, archived_details_json, task_id, original_status)
                )
            archived_count += 1

        if archived_count > 0:
            db.commit()
            logger.info(f"Archived {archived_count} previous main tasks.")
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

def save_track_analysis_and_embedding(item_id, title, author, tempo, key, scale, moods, embedding_vector, energy=None, other_features=None):
    """Saves track analysis and embedding in a single transaction."""
    # Sanitize string inputs to remove NUL characters
    title = title.replace('\x00', '') if title else title
    author = author.replace('\x00', '') if author else author
    key = key.replace('\x00', '') if key else key
    scale = scale.replace('\x00', '') if scale else scale
    other_features = other_features.replace('\x00', '') if other_features else other_features

    mood_str = ','.join(f"{k}:{v:.3f}" for k, v in moods.items())
    
    conn = get_db() # This now calls the function within this file
    cur = conn.cursor()
    try:
        # Save analysis to score table
        cur.execute("""
            INSERT INTO score (item_id, title, author, tempo, key, scale, mood_vector, energy, other_features)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (item_id) DO UPDATE SET
                title = EXCLUDED.title,
                author = EXCLUDED.author,
                tempo = EXCLUDED.tempo,
                key = EXCLUDED.key,
                scale = EXCLUDED.scale,
                mood_vector = EXCLUDED.mood_vector,
                energy = EXCLUDED.energy,
                other_features = EXCLUDED.other_features
        """, (item_id, title, author, tempo, key, scale, mood_str, energy, other_features))

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

def get_all_tracks():
    """Fetches all tracks and their embeddings from the database."""
    conn = get_db() # This now calls the function within this file
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute("""
        SELECT s.item_id, s.title, s.author, s.tempo, s.key, s.scale, s.mood_vector, s.energy, s.other_features, e.embedding
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
        SELECT s.item_id, s.title, s.author, s.tempo, s.key, s.scale, s.mood_vector, s.energy, s.other_features, e.embedding
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
        SELECT s.item_id, s.title, s.author, s.tempo, s.key, s.scale, s.mood_vector, s.energy, s.other_features
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