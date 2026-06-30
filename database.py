
import json
import logging
import sys
import time

import numpy as np
import psycopg2
from flask import g
from psycopg2.extras import DictCursor

import config

logger = logging.getLogger(__name__)

from tz_helper import UTC_NOW_SQL

from sanitization import sanitize_db_field

from config import (
    TASK_STATUS_PENDING,
    TASK_STATUS_STARTED,
    TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
    TASK_STATUS_REVOKED,
)

TASK_HISTORY_MAX_ROWS = 10
MAX_LOG_ENTRIES_STORED = 10

MAP_PROJECTION_CACHE = None

_embedded_server = None


def get_db():
    if 'db' not in g:
        try:
            g.db = psycopg2.connect(
                config.DATABASE_URL,
                connect_timeout=30,
                keepalives_idle=600,
                keepalives_interval=30,
                keepalives_count=3,
                options='-c statement_timeout=600000'
            )
        except psycopg2.OperationalError as e:
            logger.exception(f"Failed to connect to database: {e}")
            raise
    return g.db


def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def start_embedded(data_dir):
    global _embedded_server
    import pgserver
    _embedded_server = pgserver.get_server(data_dir)
    return _embedded_server.get_uri()


def ensure_embedded_running(data_dir):
    global _embedded_server
    if _embedded_server is None:
        return start_embedded(data_dir)
    import pgserver
    from pathlib import Path
    try:
        pgserver.PostgresServer._instances.pop(Path(data_dir).expanduser().resolve(), None)
    except Exception:
        pass
    _embedded_server = pgserver.get_server(data_dir)
    return _embedded_server.get_uri()


def stop_embedded():
    global _embedded_server
    if _embedded_server is not None:
        _embedded_server.cleanup()
        _embedded_server = None



def _build_task_note(task_type, details_obj, db):
    if not isinstance(details_obj, dict):
        details_obj = {}
    t = (task_type or '').lower()

    try:
        if 'analysis' in t:
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
            return ' | '.join(parts)
    except Exception as e:
        logger.debug(f"task note builder failed for type={task_type}: {e}")
    return ''


def record_task_history(task_id, task_type, status, duration_seconds=None, note=None, details=None):
    if not task_id:
        return
    try:
        db = get_db()
        if note is None:
            details_obj = details if isinstance(details, dict) else {}
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
                f"""
                INSERT INTO task_history (task_id, task_type, status, duration_seconds, note, recorded_at)
                VALUES (%s, %s, %s, %s, %s, {UTC_NOW_SQL})
                """,
                (task_id, task_type, status, duration_seconds, note),
            )
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


def save_task_status(task_id, task_type, status=TASK_STATUS_PENDING, parent_task_id=None, sub_type_identifier=None, progress=0, details=None):
    db = get_db()
    current_unix_time = time.time()

    if details is not None and isinstance(details, dict):
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

    cur = db.cursor()
    try:
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
    except psycopg2.Error:
        logger.exception(f"DB Error saving task status for {task_id}")
        try:
            db.rollback()
            logger.info(f"DB transaction rolled back for task status update of {task_id}.")
        except psycopg2.Error:
            logger.exception(f"DB Error during rollback for task status {task_id}")
    finally:
        cur.close()

    try:
        if (
            parent_task_id is None
            and status in (TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)
            and task_type and task_type != 'unknown'
        ):
            duration_s = None
            try:
                with db.cursor() as hist_cur:
                    hist_cur.execute(
                        "SELECT start_time, end_time FROM task_status WHERE task_id = %s",
                        (task_id,),
                    )
                    row = hist_cur.fetchone()
                if row and row[0] is not None:
                    end = row[1] if row[1] is not None else current_unix_time
                    duration_s = max(0.0, float(end) - float(row[0]))
            except Exception:
                pass
            record_task_history(task_id, task_type, status, duration_s, details=details)
    except Exception as e_hist:
        logger.debug(f"history record skipped for {task_id}: {e_hist}")


def get_task_info_from_db(task_id):
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
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

    if start_time is None:
        row_dict['running_time_seconds'] = 0.0
    else:
        effective_end_time = end_time if end_time is not None else current_unix_time
        row_dict['running_time_seconds'] = max(0, effective_end_time - start_time)

    return row_dict



def get_score_data_by_ids(item_ids_list):
    if not item_ids_list:
        return []
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    query = """
        SELECT s.item_id, s.title, s.author, s.album, s.album_artist, s.tempo, s.key, s.scale, s.mood_vector, s.energy, s.other_features, s.year, s.rating, s.file_path
        FROM score s
        WHERE s.item_id IN %s
    """
    try:
        cur.execute(query, (tuple(item_ids_list),))
        rows = cur.fetchall()
    except Exception:
        logger.exception("Error fetching score data by IDs")
        rows = []
    finally:
        cur.close()
    return [dict(row) for row in rows]


def get_tracks_by_ids(item_ids_list):
    if not item_ids_list:
        return []
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)

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

    processed_rows = []
    for row in rows:
        row_dict = dict(row)
        if row_dict.get('embedding'):
            row_dict['embedding_vector'] = np.frombuffer(row_dict['embedding'], dtype=np.float32)
        else:
            row_dict['embedding_vector'] = np.array([])
        processed_rows.append(row_dict)

    return processed_rows


def load_map_projection(index_name, force_reload=False):
    global MAP_PROJECTION_CACHE
    if not force_reload and MAP_PROJECTION_CACHE and MAP_PROJECTION_CACHE.get('index_name') == index_name:
        logger.info(f"Map projection '{index_name}' already loaded in cache. Skipping reload.")
        return MAP_PROJECTION_CACHE.get('id_map'), MAP_PROJECTION_CACHE.get('projection')

    logger.info(f"Attempting to load map projection '{index_name}' from database into memory...")
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT projection_data, id_map_json FROM map_projection_data WHERE index_name = %s", (index_name,))
        row = cur.fetchone()
        if row and row[0] is not None:
            proj_blob, id_map_json = row[0], row[1]
        else:
            import re
            from tasks.index_build_helpers import reassemble_segmented_id_map
            cur.execute(
                "SELECT index_name, projection_data, id_map_json FROM map_projection_data WHERE index_name LIKE %s ESCAPE '\\'",
                (index_name.replace('_', r'\_') + r"\_%\_%",),
            )
            candidates = cur.fetchall()
            if not candidates:
                logger.warning(f"Map projection '{index_name}' not found in the database. Cache will be empty.")
                return None, None
            seg_pattern = re.compile(rf"^{re.escape(index_name)}_(\d+)_(\d+)$")
            parts = []
            total_expected = None
            for name, part_blob, part_id_map in candidates:
                m = seg_pattern.match(name)
                if not m:
                    continue
                part_no = int(m.group(1))
                total = int(m.group(2))
                if total_expected is None:
                    total_expected = total
                elif total_expected != total:
                    logger.error(f"Map projection segment total mismatch for '{index_name}' ({total_expected} vs {total}). Aborting load.")
                    return None, None
                parts.append((part_no, part_blob, part_id_map))
            if total_expected is None or len(parts) != total_expected:
                logger.error(f"Incomplete map projection segments for '{index_name}': expected {total_expected}, found {len(parts)}. Aborting load.")
                return None, None
            parts.sort(key=lambda p: p[0])
            proj_blob = b"".join(bytes(p[1]) for p in parts if p[1])
            id_map_json = reassemble_segmented_id_map((p[0], p[2]) for p in parts)
        proj = np.frombuffer(proj_blob, dtype=np.float32)
        if proj.size % 2 == 0:
            proj = proj.reshape((-1, 2))
        id_map = json.loads(id_map_json)
        MAP_PROJECTION_CACHE = {'index_name': index_name, 'id_map': id_map, 'projection': proj}
        logger.info(f"Map projection '{index_name}' with {len(id_map)} items loaded successfully into memory.")
        return id_map, proj
    except Exception:
        logger.exception("Failed to load map projection")
        return None, None
    finally:
        cur.close()



def save_track_analysis_and_embedding(item_id, title, author, tempo, key, scale, moods, embedding_vector, energy=None, other_features=None, album=None, album_artist=None, year=None, rating=None, file_path=None):

    title = sanitize_db_field(title, max_length=500, field_name="title")
    author = sanitize_db_field(author, max_length=200, field_name="author")
    album = sanitize_db_field(album, max_length=200, field_name="album")
    album_artist = sanitize_db_field(album_artist, max_length=200, field_name="album_artist")
    key = sanitize_db_field(key, max_length=10, field_name="key")
    scale = sanitize_db_field(scale, max_length=10, field_name="scale")
    other_features = sanitize_db_field(other_features, max_length=2000, field_name="other_features")

    def _parse_year_from_date(year_value):
        if year_value is None:
            return None

        year_str = str(year_value).strip()
        if not year_str:
            return None

        try:
            year = int(year_str)
            if 1000 <= year <= 2100:
                return year
        except (ValueError, TypeError):
            pass

        normalized = year_str.replace('/', '-')
        parts = normalized.split('-')

        if len(parts) == 3:
            try:
                if len(parts[0]) == 4:
                    year = int(parts[0])
                    if 1000 <= year <= 2100:
                        return year

                if len(parts[2]) == 4:
                    year = int(parts[2])
                    if 1000 <= year <= 2100:
                        return year

                if len(parts[2]) == 2:
                    year = int(parts[2])
                    year += 2000 if year < 30 else 1900
                    if 1000 <= year <= 2100:
                        return year
            except (ValueError, TypeError, IndexError):
                pass

        return None

    year = _parse_year_from_date(year)

    if rating is not None:
        try:
            rating = int(rating)
            if rating < 0 or rating > 5:
                rating = None
        except (ValueError, TypeError):
            rating = None

    file_path = sanitize_db_field(file_path, max_length=1000, field_name="file_path")

    mood_str = ','.join(f"{k}:{v:.3f}" for k, v in moods.items())

    conn = get_db()
    cur = conn.cursor()
    try:
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

        if isinstance(embedding_vector, np.ndarray) and embedding_vector.size > 0:
            embedding_blob = embedding_vector.astype(np.float32).tobytes()
            cur.execute("""
                INSERT INTO embedding (item_id, embedding) VALUES (%s, %s)
                ON CONFLICT (item_id) DO UPDATE SET embedding = EXCLUDED.embedding
            """, (item_id, psycopg2.Binary(embedding_blob)))

        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("Error saving track analysis and embedding for %s", item_id)
        raise
    finally:
        cur.close()


def save_clap_embedding(item_id, clap_embedding_vector):
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
    except Exception:
        conn.rollback()
        logger.exception(f"Error saving CLAP embedding for {item_id}")
        raise
    finally:
        cur.close()


def get_clap_embedding(item_id):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT embedding FROM clap_embedding WHERE item_id = %s", (item_id,))
        row = cur.fetchone()
        if row and row[0]:
            return np.frombuffer(row[0], dtype=np.float32)
        return None
    except Exception:
        logger.exception(f"Error loading CLAP embedding for {item_id}")
        return None
    finally:
        cur.close()


def save_lyrics_embedding(item_id, lyrics_embedding_vector, axis_vector=None):
    if lyrics_embedding_vector is None or (isinstance(lyrics_embedding_vector, np.ndarray) and lyrics_embedding_vector.size == 0):
        return

    conn = get_db()
    cur = conn.cursor()
    try:
        embedding_blob = lyrics_embedding_vector.astype(np.float32).tobytes() if isinstance(lyrics_embedding_vector, np.ndarray) else np.asarray(lyrics_embedding_vector, dtype=np.float32).tobytes()
        axis_blob = None
        if axis_vector is not None:
            arr = axis_vector if isinstance(axis_vector, np.ndarray) else np.asarray(axis_vector, dtype=np.float32)
            if arr.size > 0:
                axis_blob = arr.astype(np.float32, copy=False).tobytes()
        cur.execute("""
            INSERT INTO lyrics_embedding (item_id, embedding, axis_vector) VALUES (%s, %s, %s)
            ON CONFLICT (item_id) DO UPDATE SET embedding = EXCLUDED.embedding, axis_vector = EXCLUDED.axis_vector, updated_at = CURRENT_TIMESTAMP
        """, (item_id, psycopg2.Binary(embedding_blob),
              psycopg2.Binary(axis_blob) if axis_blob is not None else None))
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception(f"Error saving lyrics embedding for {item_id}")
        raise
    finally:
        cur.close()


ARTIST_PROJECTION_CACHE = None



def init_db():
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(726354821)")
        try:
            if sys.platform == 'win32':
                for ext in ('unaccent', 'pg_trgm'):
                    cur.execute("SAVEPOINT ext_create")
                    try:
                        cur.execute(f'CREATE EXTENSION IF NOT EXISTS {ext}')
                        cur.execute("RELEASE SAVEPOINT ext_create")
                    except Exception:
                        logger.warning("Extension %s not available -- skipping", ext)
                        cur.execute("ROLLBACK TO SAVEPOINT ext_create")
            else:
                cur.execute('CREATE EXTENSION IF NOT EXISTS unaccent')
                cur.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
            cur.execute("CREATE TABLE IF NOT EXISTS score (item_id TEXT PRIMARY KEY, title TEXT, author TEXT, album TEXT, album_artist TEXT, tempo REAL, key TEXT, scale TEXT, mood_vector TEXT)")
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'energy')")
            if not cur.fetchone()[0]:
                logger.info("Adding 'energy' column to 'score' table.")
                cur.execute("ALTER TABLE score ADD COLUMN energy REAL")
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'other_features')")
            if not cur.fetchone()[0]:
                logger.info("Adding 'other_features' column to 'score' table.")
                cur.execute("ALTER TABLE score ADD COLUMN other_features TEXT")
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'album')")
            if not cur.fetchone()[0]:
                logger.info("Adding 'album' column to 'score' table.")
                cur.execute("ALTER TABLE score ADD COLUMN album TEXT")
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'album_artist')")
            if not cur.fetchone()[0]:
                logger.info("Adding 'album_artist' column to 'score' table.")
                cur.execute("ALTER TABLE score ADD COLUMN album_artist TEXT")
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'year')")
            if not cur.fetchone()[0]:
                logger.info("Adding 'year' column to 'score' table.")
                cur.execute("ALTER TABLE score ADD COLUMN year INTEGER")
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'rating')")
            if not cur.fetchone()[0]:
                logger.info("Adding 'rating' column to 'score' table.")
                cur.execute("ALTER TABLE score ADD COLUMN rating INTEGER")
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'file_path')")
            if not cur.fetchone()[0]:
                logger.info("Adding 'file_path' column to 'score' table.")
                cur.execute("ALTER TABLE score ADD COLUMN file_path TEXT")

            cur.execute("SELECT is_generated FROM information_schema.columns WHERE table_name = 'score' AND column_name = 'search_u'")
            row = cur.fetchone()
            search_u_generated = (row and row[0] == 'ALWAYS')

            if search_u_generated:
                logger.info("Dropping legacy generated 'search_u' column to replace it with a trigger-updated column.")
                cur.execute("ALTER TABLE score DROP COLUMN IF EXISTS search_u")
                row = None

            if not row:
                logger.info("Adding 'search_u' column to 'score' table.")
                cur.execute("ALTER TABLE score ADD COLUMN search_u TEXT")

            if sys.platform == 'win32':
                cur.execute("SAVEPOINT search_setup")
                try:
                    cur.execute("CREATE OR REPLACE FUNCTION immutable_unaccent(text) RETURNS text LANGUAGE sql IMMUTABLE AS $$ SELECT public.unaccent($1) $$;")
                    cur.execute("""
                        CREATE OR REPLACE FUNCTION score_search_u_sync() RETURNS trigger LANGUAGE plpgsql AS $$
                        BEGIN
                            NEW.search_u := lower(immutable_unaccent(concat_ws(' ', NEW.title, NEW.author, NEW.album)));
                            RETURN NEW;
                        END;
                        $$;
                    """)
                    cur.execute("DROP TRIGGER IF EXISTS score_search_u_sync_trigger ON score")
                    cur.execute("""
                        CREATE TRIGGER score_search_u_sync_trigger
                        BEFORE INSERT OR UPDATE ON score
                        FOR EACH ROW
                        EXECUTE FUNCTION score_search_u_sync();
                    """)
                    cur.execute("UPDATE score SET search_u = lower(immutable_unaccent(concat_ws(' ', title, author, album))) WHERE search_u IS NULL")
                    cur.execute("CREATE INDEX IF NOT EXISTS score_search_u_trgm ON score USING gin (search_u gin_trgm_ops)")
                    cur.execute("RELEASE SAVEPOINT search_setup")
                except Exception:
                    logger.warning("unaccent/pg_trgm extensions not available -- accent-insensitive search disabled")
                    cur.execute("ROLLBACK TO SAVEPOINT search_setup")
            else:
                cur.execute("CREATE OR REPLACE FUNCTION immutable_unaccent(text) RETURNS text LANGUAGE sql IMMUTABLE AS $$ SELECT public.unaccent($1) $$;")
                cur.execute("""
                    CREATE OR REPLACE FUNCTION score_search_u_sync() RETURNS trigger LANGUAGE plpgsql AS $$
                    BEGIN
                        NEW.search_u := lower(immutable_unaccent(concat_ws(' ', NEW.title, NEW.author, NEW.album)));
                        RETURN NEW;
                    END;
                    $$;
                """)
                cur.execute("DROP TRIGGER IF EXISTS score_search_u_sync_trigger ON score")
                cur.execute("""
                    CREATE TRIGGER score_search_u_sync_trigger
                    BEFORE INSERT OR UPDATE ON score
                    FOR EACH ROW
                    EXECUTE FUNCTION score_search_u_sync();
                """)
                cur.execute("UPDATE score SET search_u = lower(immutable_unaccent(concat_ws(' ', title, author, album))) WHERE search_u IS NULL")
                cur.execute("CREATE INDEX IF NOT EXISTS score_search_u_trgm ON score USING gin (search_u gin_trgm_ops)")

            cur.execute("CREATE INDEX IF NOT EXISTS idx_score_album_artist_album ON score (album_artist, album)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_score_author ON score (author)")

            cur.execute("CREATE TABLE IF NOT EXISTS playlist (id SERIAL PRIMARY KEY, playlist_name TEXT, item_id TEXT, title TEXT, author TEXT, UNIQUE (playlist_name, item_id))")
            cur.execute("CREATE TABLE IF NOT EXISTS task_status (id SERIAL PRIMARY KEY, task_id TEXT UNIQUE NOT NULL, parent_task_id TEXT, task_type TEXT NOT NULL, sub_type_identifier TEXT, status TEXT, progress INTEGER DEFAULT 0, details TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_task_status_parent ON task_status (parent_task_id)")
            for col_name in ['start_time', 'end_time']:
                cur.execute("SELECT data_type FROM information_schema.columns WHERE table_name = 'task_status' AND column_name = %s", (col_name,))
                if not cur.fetchone(): cur.execute(f"ALTER TABLE task_status ADD COLUMN {col_name} DOUBLE PRECISION")
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
            cur.execute("CREATE TABLE IF NOT EXISTS embedding (item_id TEXT PRIMARY KEY, FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE)")
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'embedding' AND column_name = 'embedding')")
            if not cur.fetchone()[0]: cur.execute("ALTER TABLE embedding ADD COLUMN embedding BYTEA")
            cur.execute("CREATE TABLE IF NOT EXISTS lyrics_embedding (item_id TEXT PRIMARY KEY, FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE)")
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'lyrics_embedding' AND column_name = 'embedding')")
            if not cur.fetchone()[0]: cur.execute("ALTER TABLE lyrics_embedding ADD COLUMN embedding BYTEA")
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'lyrics_embedding' AND column_name = 'axis_vector')")
            if not cur.fetchone()[0]: cur.execute("ALTER TABLE lyrics_embedding ADD COLUMN axis_vector BYTEA")
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'lyrics_embedding' AND column_name = 'updated_at')")
            if not cur.fetchone()[0]: cur.execute("ALTER TABLE lyrics_embedding ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            cur.execute("CREATE TABLE IF NOT EXISTS clap_embedding (item_id TEXT PRIMARY KEY, FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE)")
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'clap_embedding' AND column_name = 'embedding')")
            if not cur.fetchone()[0]: cur.execute("ALTER TABLE clap_embedding ADD COLUMN embedding BYTEA")
            cur.execute("DROP TABLE IF EXISTS voyager_index_data")
            cur.execute("DROP TABLE IF EXISTS clap_index_data")
            cur.execute("DROP TABLE IF EXISTS lyrics_index_data")
            cur.execute("DROP TABLE IF EXISTS lyrics_axes_index_data")
            cur.execute("DROP TABLE IF EXISTS artist_index_data")
            cur.execute("CREATE TABLE IF NOT EXISTS artist_metadata_data (name VARCHAR(255) PRIMARY KEY, blob_data BYTEA NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            cur.execute("CREATE TABLE IF NOT EXISTS ivf_dir (name VARCHAR(255) PRIMARY KEY, blob_data BYTEA NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            cur.execute("CREATE TABLE IF NOT EXISTS ivf_cell (index_name VARCHAR(255) NOT NULL, cell_id INTEGER NOT NULL, cell_data BYTEA NOT NULL, PRIMARY KEY (index_name, cell_id))")
            cur.execute("ALTER TABLE ivf_cell ALTER COLUMN cell_data SET STORAGE EXTERNAL")
            cur.execute("CREATE TABLE IF NOT EXISTS map_projection_data (index_name VARCHAR(255) PRIMARY KEY, projection_data BYTEA NOT NULL, id_map_json TEXT NOT NULL, embedding_dimension INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            cur.execute("CREATE TABLE IF NOT EXISTS artist_component_projection (index_name VARCHAR(255) PRIMARY KEY, projection_data BYTEA NOT NULL, artist_component_map_json TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            cur.execute("CREATE TABLE IF NOT EXISTS cron (id SERIAL PRIMARY KEY, name TEXT, task_type TEXT NOT NULL, cron_expr TEXT NOT NULL, enabled BOOLEAN DEFAULT FALSE, last_run DOUBLE PRECISION, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            cur.execute("CREATE TABLE IF NOT EXISTS audiomuse_users (id SERIAL PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            cur.execute("ALTER TABLE audiomuse_users ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'user'")
            cur.execute(
                "CREATE TABLE IF NOT EXISTS dashboard_stats ("
                "id INTEGER PRIMARY KEY, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "content JSONB NOT NULL DEFAULT '{}'::jsonb, "
                "indexes JSONB NOT NULL DEFAULT '[]'::jsonb, "
                "CONSTRAINT dashboard_stats_singleton CHECK (id = 1))"
            )
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.table_constraints "
                "WHERE table_name = 'dashboard_stats' AND constraint_type = 'PRIMARY KEY'"
            )
            row = cur.fetchone()
            if row and row[0] == 0:
                logger.info("Cleaning dashboard_stats and adding missing primary key constraint to dashboard_stats.id")
                cur.execute("DELETE FROM dashboard_stats")
                cur.execute("ALTER TABLE dashboard_stats ADD CONSTRAINT dashboard_stats_pkey PRIMARY KEY (id)")
            cur.execute("CREATE TABLE IF NOT EXISTS artist_mapping (artist_name TEXT PRIMARY KEY, artist_id TEXT)")
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'app_config')"
            )
            if not cur.fetchone()[0]:
                cur.execute(
                    "CREATE TABLE app_config ("
                    "key TEXT PRIMARY KEY, value TEXT NOT NULL, "
                    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                )
            cur.execute("CREATE TABLE IF NOT EXISTS alchemy_anchors (id SERIAL PRIMARY KEY, name TEXT UNIQUE NOT NULL, centroid JSONB NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            cur.execute("CREATE TABLE IF NOT EXISTS alchemy_radios (id SERIAL PRIMARY KEY, anchor_id INTEGER UNIQUE NOT NULL REFERENCES alchemy_anchors(id) ON DELETE CASCADE, temperature DOUBLE PRECISION NOT NULL, n_results INTEGER NOT NULL, enabled BOOLEAN NOT NULL DEFAULT TRUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS migration_target_meta (
                    session_id   INTEGER NOT NULL REFERENCES migration_session(id) ON DELETE CASCADE,
                    new_id       TEXT NOT NULL,
                    path         TEXT,
                    title        TEXT,
                    artist       TEXT,
                    album        TEXT,
                    album_artist TEXT,
                    year         INTEGER,
                    PRIMARY KEY (session_id, new_id)
                )
            """)
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
        finally:
            cur.execute("SELECT pg_advisory_unlock(726354821)")




def clean_up_previous_main_tasks():
    db = get_db()
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
                update_cur.execute(
                    "DELETE FROM task_status WHERE parent_task_id = %s",
                    (task_id,)
                )
                children_deleted = update_cur.rowcount
                deleted_children_count += children_deleted

                if children_deleted > 0:
                    logger.info(f"Deleted {children_deleted} child tasks for parent task {task_id}")

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
    except Exception:
        db.rollback()
        logger.exception("Error during the main task cleanup process")
    finally:
        cur.close()

def get_active_main_task(task_type=None):
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

def get_child_tasks_from_db(parent_task_id):
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute("SELECT task_id, status, sub_type_identifier FROM task_status WHERE parent_task_id = %s", (parent_task_id,))
    tasks = cur.fetchall()
    cur.close()
    return [dict(row) for row in tasks]




def save_alchemy_anchor(name, centroid):
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
    except Exception:
        conn.rollback()
        logger.exception(f"Failed to save alchemy anchor '{name}'")
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
    except Exception:
        logger.exception("Failed to load alchemy anchors")
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
    except Exception:
        conn.rollback()
        logger.exception(f"Failed to delete alchemy anchor id={anchor_id}")
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
    except Exception:
        logger.exception(f"Failed to fetch alchemy anchor id={anchor_id}")
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
    except Exception:
        conn.rollback()
        logger.exception(f"Failed to rename alchemy anchor id={anchor_id}")
        return None
    finally:
        cur.close()

def get_alchemy_radios():
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    try:
        cur.execute(
            "SELECT r.id, r.anchor_id, a.name, r.temperature, r.n_results, r.enabled "
            "FROM alchemy_radios r JOIN alchemy_anchors a ON a.id = r.anchor_id "
            "ORDER BY a.name"
        )
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    except Exception:
        logger.exception("Failed to load alchemy radios")
        return []
    finally:
        cur.close()

def create_alchemy_radio(anchor_id, temperature, n_results, enabled=True):
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    try:
        cur.execute(
            "INSERT INTO alchemy_radios (anchor_id, temperature, n_results, enabled) "
            "VALUES (%s, %s, %s, %s) RETURNING id, anchor_id, temperature, n_results, enabled",
            (anchor_id, temperature, n_results, bool(enabled))
        )
        row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None
    except Exception:
        conn.rollback()
        logger.exception(f"Failed to create alchemy radio for anchor_id={anchor_id}")
        return None
    finally:
        cur.close()

def update_alchemy_radio(radio_id, temperature, n_results, enabled):
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    try:
        cur.execute(
            "UPDATE alchemy_radios SET temperature = %s, n_results = %s, enabled = %s "
            "WHERE id = %s RETURNING id, anchor_id, temperature, n_results, enabled",
            (temperature, n_results, bool(enabled), radio_id)
        )
        row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None
    except Exception:
        conn.rollback()
        logger.exception(f"Failed to update alchemy radio id={radio_id}")
        return None
    finally:
        cur.close()

def delete_alchemy_radio(radio_id):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM alchemy_radios WHERE id = %s", (radio_id,))
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        conn.rollback()
        logger.exception(f"Failed to delete alchemy radio id={radio_id}")
        return False
    finally:
        cur.close()




def save_map_projection(index_name, id_map, projection_array):
    conn = get_db()
    try:
        blob = projection_array.astype(np.float32).tobytes()
        if not blob:
            logger.info(f"Map projection '{index_name}' has no data; clearing existing store.")
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM map_projection_data WHERE index_name = %s OR index_name LIKE %s ESCAPE '\\'",
                    (index_name, index_name.replace('_', r'\_') + r"\_%\_%"),
                )
            conn.commit()
            return
        embedding_dim = projection_array.shape[1] if projection_array.ndim == 2 else 0
        from tasks.index_build_helpers import store_ivf_index_segmented
        store_ivf_index_segmented(
            conn,
            target_table="map_projection_data",
            index_name=index_name,
            index_bytes=blob,
            id_map=id_map,
            embedding_dimension=embedding_dim,
            binary_column="projection_data",
        )
        conn.commit()
        try:
            id_count = len(id_map) if hasattr(id_map, '__len__') else None
            logger.info(f"Saved map projection '{index_name}' to DB: {len(blob)} bytes, ids={id_count}")
        except Exception:
            logger.debug("Saved map projection but failed to compute size/id_count for log.")
    except Exception:
        conn.rollback()
        logger.exception("Failed to save map projection")
        raise

def load_artist_projection(index_name='artist_map', force_reload=False):
    global ARTIST_PROJECTION_CACHE
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
        if proj.size % 2 == 0:
            proj = proj.reshape((-1, 2))
        component_map = json.loads(component_map_json)
        ARTIST_PROJECTION_CACHE = {'index_name': index_name, 'component_map': component_map, 'projection': proj}
        logger.info(f"Artist projection '{index_name}' with {len(component_map)} components loaded successfully into memory.")
        return component_map, proj
    except Exception:
        logger.exception("Failed to load artist projection")
        return None, None
    finally:
        cur.close()

def save_artist_projection(index_name, component_map, projections):
    conn = get_db()
    cur = conn.cursor()
    try:
        component_map_json = json.dumps(component_map)
        proj_blob = projections.astype(np.float32).tobytes()
        cur.execute("INSERT INTO artist_component_projection (index_name, projection_data, artist_component_map_json) VALUES (%s, %s, %s) ON CONFLICT (index_name) DO UPDATE SET projection_data = EXCLUDED.projection_data, artist_component_map_json = EXCLUDED.artist_component_map_json, created_at = CURRENT_TIMESTAMP", (index_name, proj_blob, component_map_json))
        conn.commit()
        logger.info(f"Saved artist projection '{index_name}' with {len(component_map)} components to database.")
    except Exception:
        conn.rollback()
        logger.exception("Failed to save artist projection")
    finally:
        cur.close()




def update_playlist_table(playlists):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM playlist")
        for name, cluster in playlists.items():
            for item_id, title, author in cluster:
                cur.execute("INSERT INTO playlist (playlist_name, item_id, title, author) VALUES (%s, %s, %s, %s) ON CONFLICT (playlist_name, item_id) DO NOTHING", (name, item_id, title, author))
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("Error updating playlist table")
    finally:
        cur.close()
