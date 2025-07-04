import os
import json
import logging
import tempfile
import numpy as np
import time
from annoy import AnnoyIndex # type: ignore
import psycopg2
from psycopg2.extras import DictCursor
import requests # Added for Jellyfin interaction

from config import (
    EMBEDDING_DIMENSION, JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN,
    INDEX_NAME, NUM_TREES
) # Use a central config for this

logger = logging.getLogger(__name__)

# --- Global cache for the loaded Annoy index ---
# This will hold the index in memory for the web server process to prevent
# reloading from the database on every API call.
annoy_index = None
id_map = None


def build_and_store_annoy_index(db_conn):
    """
    Fetches all song embeddings, builds a new Annoy index, and stores it
    atomically in the 'annoy_index_data' table in PostgreSQL.
    """
    logger.info("Starting to build and store Annoy index...")
    cur = db_conn.cursor()
    try:
        logger.info("Fetching all embeddings from the database...")
        cur.execute("SELECT item_id, embedding FROM embedding")
        all_embeddings = cur.fetchall()

        if not all_embeddings:
            logger.warning("No embeddings found in DB. Annoy index will not be built.")
            return

        logger.info(f"Found {len(all_embeddings)} embeddings to index.")

        annoy_index = AnnoyIndex(EMBEDDING_DIMENSION, 'angular')
        id_map = {}
        annoy_item_index = 0
        for item_id, embedding_blob in all_embeddings:
            if embedding_blob is None:
                logger.warning(f"Skipping item_id {item_id}: embedding data is NULL.")
                continue
            
            embedding_vector = np.frombuffer(embedding_blob, dtype=np.float32)
            
            if embedding_vector.shape[0] != EMBEDDING_DIMENSION:
                logger.warning(f"Skipping item_id {item_id}: embedding dimension mismatch. "
                               f"Expected {EMBEDDING_DIMENSION}, got {embedding_vector.shape[0]}.")
                continue
            
            annoy_index.add_item(annoy_item_index, embedding_vector)
            id_map[annoy_item_index] = item_id
            annoy_item_index += 1

        if annoy_item_index == 0:
            logger.warning("No valid embeddings were found to add to the Annoy index. Aborting build process.")
            return

        logger.info(f"Building index with {annoy_item_index} items and {NUM_TREES} trees...")
        annoy_index.build(NUM_TREES)

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ann") as tmp:
                temp_file_path = tmp.name
            
            annoy_index.save(temp_file_path)

            with open(temp_file_path, 'rb') as f:
                index_binary_data = f.read()
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        logger.info(f"Annoy index binary data size to be stored: {len(index_binary_data)} bytes.")

        if not index_binary_data:
            logger.error("CRITICAL: Generated Annoy index file is empty. Aborting database storage.")
            return

        id_map_json = json.dumps(id_map)

        logger.info(f"Storing Annoy index '{INDEX_NAME}' in the database...")
        upsert_query = """
            INSERT INTO annoy_index_data (index_name, index_data, id_map_json, embedding_dimension, created_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (index_name) DO UPDATE SET
                index_data = EXCLUDED.index_data,
                id_map_json = EXCLUDED.id_map_json,
                embedding_dimension = EXCLUDED.embedding_dimension,
                created_at = CURRENT_TIMESTAMP;
        """
        cur.execute(upsert_query, (INDEX_NAME, psycopg2.Binary(index_binary_data), id_map_json, EMBEDDING_DIMENSION))
        db_conn.commit()
        logger.info("Annoy index build and database storage complete.")

    except Exception as e:
        logger.error("An error occurred during Annoy index build: %s", e, exc_info=True)
        db_conn.rollback()
    finally:
        cur.close()

def load_annoy_index_for_querying(force_reload=False):
    """
    Loads the Annoy index from the database into the global in-memory cache.
    """
    global annoy_index, id_map

    if annoy_index is not None and not force_reload:
        logger.info("Annoy index is already loaded in memory. Skipping reload.")
        return

    from app import get_db

    logger.info("Attempting to load Annoy index from database into memory...")
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT index_data, id_map_json, embedding_dimension FROM annoy_index_data WHERE index_name = %s", (INDEX_NAME,))
        record = cur.fetchone()

        if not record:
            logger.warning(f"Annoy index '{INDEX_NAME}' not found in the database. Cache will be empty.")
            annoy_index, id_map = None, None
            return
        
        index_binary_data, id_map_json, db_embedding_dim = record

        if not index_binary_data:
            logger.error(f"Annoy index '{INDEX_NAME}' data in database is empty.")
            annoy_index, id_map = None, None
            return

        if db_embedding_dim != EMBEDDING_DIMENSION:
            logger.error(f"FATAL: Annoy index dimension mismatch! DB has {db_embedding_dim}, config expects {EMBEDDING_DIMENSION}.")
            annoy_index, id_map = None, None
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".ann") as tmp:
            tmp.write(index_binary_data)
            temp_file_path = tmp.name
        
        logger.info(f"Loading index from temporary file: {temp_file_path}")
        loaded_index = AnnoyIndex(EMBEDDING_DIMENSION, 'angular')
        loaded_index.load(temp_file_path)
        os.remove(temp_file_path)

        annoy_index = loaded_index
        id_map = {int(k): v for k, v in json.loads(id_map_json).items()}

        logger.info(f"Annoy index with {len(id_map)} items loaded successfully into memory.")

    except Exception as e:
        logger.error("Failed to load Annoy index from database: %s", e, exc_info=True)
        annoy_index, id_map = None, None
    finally:
        cur.close()

def find_nearest_neighbors_by_id(target_item_id: str, n: int = 10):
    """
    Finds the N nearest neighbors for a given item_id using the globally cached index.
    """
    if annoy_index is None or id_map is None:
        raise RuntimeError("Annoy index is not loaded in memory. It may be missing, empty, or the server failed to load it on startup.")

    reverse_id_map = {v: k for k, v in id_map.items()}
    target_annoy_id = reverse_id_map.get(target_item_id)

    if target_annoy_id is None:
        logger.warning(f"Target item_id '{target_item_id}' not found in the loaded Annoy index map.")
        return []

    neighbor_annoy_ids, distances = annoy_index.get_nns_by_item(target_annoy_id, n + 1, include_distances=True)

    results = []
    for annoy_id, dist in zip(neighbor_annoy_ids, distances):
        if annoy_id != target_annoy_id:
            item_id = id_map.get(annoy_id)
            if item_id:
                results.append({"item_id": item_id, "distance": dist})

    return results[:n]

def get_item_id_by_title_and_artist(title: str, artist: str):
    """
    Finds the item_id for an exact title and artist match.
    Returns the item_id string or None if not found.
    """
    from app import get_db
    conn = get_db()
    # Use DictCursor to get column names
    cur = conn.cursor(cursor_factory=DictCursor)
    try:
        # CORRECTED TABLE NAME: Changed 'score_data' to 'score'
        query = "SELECT item_id FROM score WHERE title = %s AND author = %s LIMIT 1"
        cur.execute(query, (title, artist))
        result = cur.fetchone()
        if result:
            return result['item_id']
        return None
    except Exception as e:
        logger.error(f"Error fetching item_id for '{title}' by '{artist}': {e}", exc_info=True)
        return None
    finally:
        cur.close()

def search_tracks_by_title_and_artist(title_query: str, artist_query: str, limit: int = 15):
    """
    Searches for tracks using partial title and artist names for autocomplete.
    """
    from app import get_db
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    results = []
    try:
        # Build the query dynamically based on which fields are provided
        query_parts = []
        params = []
        
        if artist_query:
            query_parts.append("author ILIKE %s")
            params.append(f"%{artist_query}%")
            
        if title_query:
            query_parts.append("title ILIKE %s")
            params.append(f"%{title_query}%")

        if not query_parts:
            return [] # Don't run a query if no search terms are provided

        # Combine the query parts
        where_clause = " AND ".join(query_parts)
        
        # CORRECTED TABLE NAME: Changed 'score_data' to 'score'
        # We only need fields relevant for the autocomplete dropdown
        query = f"""
            SELECT item_id, title, author 
            FROM score 
            WHERE {where_clause}
            ORDER BY author, title 
            LIMIT %s
        """
        params.append(limit)
        
        cur.execute(query, tuple(params))
        
        # Convert rows to a list of dictionaries
        results = [dict(row) for row in cur.fetchall()]

    except Exception as e:
        logger.error(f"Error searching tracks with query '{title_query}', '{artist_query}': {e}", exc_info=True)
    finally:
        cur.close()
    
    return results


def create_jellyfin_playlist_from_ids(playlist_name: str, track_ids: list):
    """
    Creates a new playlist in Jellyfin with the provided name and track IDs.
    """
    if not all([JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN]):
        raise ValueError("Jellyfin server URL, User ID, or Token is not configured.")

    if not playlist_name or not playlist_name.strip():
        raise ValueError("Playlist name must be a non-empty string.")
    
    # Append _instant to the playlist name
    jellyfin_playlist_name = f"{playlist_name.strip()}_instant"
    
    headers = {"X-Emby-Token": JELLYFIN_TOKEN}
    body = {
        "Name": jellyfin_playlist_name,
        "Ids": track_ids,
        "UserId": JELLYFIN_USER_ID
    }
    
    url = f"{JELLYFIN_URL}/Playlists"
    
    logger.info(f"Attempting to create playlist '{jellyfin_playlist_name}' with {len(track_ids)} tracks on Jellyfin.")
    
    try:
        response = requests.post(url, headers=headers, json=body, timeout=60)
        response.raise_for_status()
        
        playlist_data = response.json()
        playlist_id = playlist_data.get('Id')
        
        if not playlist_id:
            raise Exception("Jellyfin API response did not include a playlist ID.")
            
        logger.info(f"✅ Successfully created playlist '{jellyfin_playlist_name}' with ID: {playlist_id}")
        return playlist_id

    except requests.exceptions.RequestException as e:
        logger.error(f"Error creating Jellyfin playlist '{jellyfin_playlist_name}': {e}", exc_info=True)
        raise Exception(f"Failed to communicate with Jellyfin: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred during playlist creation for '{jellyfin_playlist_name}': {e}", exc_info=True)
        raise
