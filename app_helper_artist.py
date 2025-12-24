# app_helper_artist.py
"""
Helper functions for artist mapping between names and IDs.
Separated to avoid circular imports.
"""

import logging
from app_helper import get_db
from tasks.memory_utils import sanitize_string_for_db

logger = logging.getLogger(__name__)

def upsert_artist_mapping(artist_name, artist_id):
    """
    Stores or updates the mapping between artist name and artist ID.
    If artist_name or artist_id is None/empty, does nothing.
    
    Sanitizes artist_name to remove NULL bytes (0x00) which PostgreSQL rejects.
    """
    if not artist_name or not artist_id:
        return
    
    # Sanitize artist name to remove NULL bytes and other problematic characters
    sanitized_name = sanitize_string_for_db(artist_name)
    if not sanitized_name:
        logger.warning(f"Artist name '{artist_name}' became empty after sanitization, skipping")
        return
    
    # Log if sanitization changed the name
    if sanitized_name != artist_name:
        logger.info(f"Sanitized artist name: '{artist_name}' → '{sanitized_name}'")
    
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO artist_mapping (artist_name, artist_id)
                VALUES (%s, %s)
                ON CONFLICT (artist_name)
                DO UPDATE SET artist_id = EXCLUDED.artist_id
            """, (sanitized_name, artist_id))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to upsert artist mapping for '{sanitized_name}': {e}")
        try:
            conn.rollback()
        except:
            pass

def get_artist_id_by_name(artist_name):
    """
    Retrieves the artist_id for a given artist_name.
    Returns None if not found.
    """
    if not artist_name:
        return None
    
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT artist_id FROM artist_mapping WHERE artist_name = %s", (artist_name,))
            row = cur.fetchone()
            return row[0] if row else None
    except Exception as e:
        logger.error(f"Failed to get artist_id for '{artist_name}': {e}")
        return None

def get_artist_name_by_id(artist_id):
    """
    Retrieves the artist_name for a given artist_id.
    Returns None if not found.
    """
    if not artist_id:
        return None
    
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT artist_name FROM artist_mapping WHERE artist_id = %s", (artist_id,))
            row = cur.fetchone()
            return row[0] if row else None
    except Exception as e:
        logger.error(f"Failed to get artist_name for '{artist_id}': {e}")
        return None
