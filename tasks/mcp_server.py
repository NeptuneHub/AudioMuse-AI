"""
MCP Server for AudioMuse-AI
Exposes database operations and AI tools as MCP resources for playlist generation.
"""
import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import psycopg2
from psycopg2.extras import DictCursor
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Create the MCP server
mcp_server = Server("audiomuse-ai")

# Thread pool for running sync database operations
executor = ThreadPoolExecutor(max_workers=5)


def get_db_connection():
    """Get database connection using config settings."""
    from config import DATABASE_URL
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


async def run_in_executor(func, *args):
    """Run a synchronous function in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)


def _artist_similarity_api_sync(artist: str, count: int, get_songs: int) -> List[Dict]:
    """Synchronous implementation of artist similarity API."""
    from tasks.artist_gmm_manager import find_similar_artists
    import re
    
    db_conn = get_db_connection()
    log_messages = []
    
    try:
        # STEP 1: Fuzzy lookup in database to find correct artist name
        log_messages.append(f"Looking up artist in database: '{artist}'")
        
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            # Try exact match first
            cur.execute("""
                SELECT DISTINCT author 
                FROM public.score 
                WHERE LOWER(author) = LOWER(%s)
                LIMIT 1
            """, (artist,))
            result = cur.fetchone()
            
            # If no exact match, try fuzzy ILIKE match
            if not result:
                # Normalize: remove spaces, dashes, slashes to handle variations
                # "AC DC" → "ACDC" matches "AC/DC" → "ACDC"
                # "blink-182" → "blink182" matches "blink‐182" → "blink182"
                artist_normalized = artist.replace(' ', '').replace('-', '').replace('‐', '').replace('/', '').replace("'", '')
                
                log_messages.append(f"No exact match, trying fuzzy search for normalized: '{artist_normalized}'")
                cur.execute("""
                    SELECT author, LENGTH(author) as len
                    FROM (
                        SELECT DISTINCT author
                        FROM public.score 
                        WHERE REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(author, ' ', ''), '-', ''), '‐', ''), '/', ''), '''', '') ILIKE %s
                    ) AS distinct_authors
                    ORDER BY len
                    LIMIT 1
                """, (f"%{artist_normalized}%",))
                result = cur.fetchone()
                if result:
                    log_messages.append(f"Fuzzy search found: '{result['author']}'")
                else:
                    log_messages.append(f"Fuzzy search returned no results for: '{artist_normalized}'")
            
            if result:
                db_artist_name = result['author']
                log_messages.append(f"Found in database: '{db_artist_name}'")
                artist = db_artist_name  # Use the correct database name
            else:
                log_messages.append(f"Artist not found in database, using original: '{artist}'")
        
        # STEP 2: Now call similarity API with correct artist name
        log_messages.append(f"Calling similarity API for: '{artist}'")
        similar_artists = find_similar_artists(artist, n=25)
        
        # ONLY if GMM search completely failed, try fallback strategies
        if not similar_artists:
            log_messages.append(f"Similarity API returned no results, trying fallback strategies")
            
            # Fallback 1: Try fuzzy matching in GMM index
            from tasks.artist_gmm_manager import reverse_artist_map
            if reverse_artist_map:
                artist_lower = artist.lower()
                matches = [
                    gmm_artist for gmm_artist in reverse_artist_map.keys()
                    if artist_lower in gmm_artist.lower()
                ]
                if matches:
                    # Use shortest match (most specific)
                    best_match = min(matches, key=len)
                    log_messages.append(f"Found fuzzy match in GMM index: '{best_match}' (from '{artist}')")
                    similar_artists = find_similar_artists(best_match, n=25)
            
            # Fallback 2: If still nothing, try removing special chars
            if not similar_artists:
                clean_artist = re.sub(r'[^\w\s]', '', artist).strip()
                if clean_artist != artist:
                    log_messages.append(f"Trying without special chars: '{clean_artist}'")
                    similar_artists = find_similar_artists(clean_artist, n=25)
        
        if not similar_artists:
            return {"songs": [], "message": "\n".join(log_messages) + f"\nNo similar artists found for '{artist}'"}
        
        artist_names = [a['artist'] for a in similar_artists[:count]]
        log_messages.append(f"Found {len(artist_names)} similar artists")
        
        # IMPORTANT: Include BOTH original artist AND similar artists
        all_artist_names = [artist] + artist_names
        log_messages.append(f"Searching songs from {len(all_artist_names)} artists (original + similar)")
        
        # Query songs from original artist + similar artists
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            placeholders = ','.join(['%s'] * len(all_artist_names))
            query = f"""
                SELECT item_id, title, author
                FROM (
                    SELECT DISTINCT item_id, title, author
                    FROM public.score
                    WHERE author IN ({placeholders})
                ) AS distinct_songs
                ORDER BY RANDOM()
                LIMIT %s
            """
            cur.execute(query, all_artist_names + [get_songs])
            results = cur.fetchall()
        
        songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author']} for r in results]
        log_messages.append(f"Retrieved {len(songs)} songs from original + similar artists")
        
        # Build component_matches to show which songs came from which artist
        component_matches = []
        for artist_name in all_artist_names:
            artist_songs = [s for s in songs if s['artist'] == artist_name]
            if artist_songs:
                component_matches.append({
                    "artist": artist_name,
                    "is_original": artist_name == artist,
                    "song_count": len(artist_songs),
                    "songs": artist_songs
                })
        
        return {
            "songs": songs, 
            "similar_artists": artist_names, 
            "component_matches": component_matches,
            "message": "\n".join(log_messages)
        }
    finally:
        db_conn.close()


def _artist_hits_query_sync(artist: str, ai_config: Dict, get_songs: int) -> List[Dict]:
    """Synchronous implementation of artist hits query using AI knowledge."""
    from ai import call_ai_for_chat
    
    db_conn = get_db_connection()
    log_messages = []
    
    try:
        log_messages.append(f"Using AI knowledge to suggest {artist}'s famous songs...")
        
        prompt = f"""You are a music expert. List the most famous and popular songs by the artist "{artist}".

CRITICAL REQUIREMENTS:
1. Return ONLY a JSON array of song titles
2. Include 15-25 of their most famous songs
3. Use exact song titles as they appear on albums
4. Format: ["Song Title 1", "Song Title 2", ...]
5. NO explanations, NO numbering, ONLY the JSON array

Example format:
["Song A", "Song B", "Song C"]

List the famous songs by {artist} now:"""
        
        raw_response = call_ai_for_chat(
            provider=ai_config['provider'],
            prompt=prompt,
            ollama_url=ai_config.get('ollama_url'),
            ollama_model_name=ai_config.get('ollama_model'),
            gemini_api_key=ai_config.get('gemini_key'),
            gemini_model_name=ai_config.get('gemini_model'),
            mistral_api_key=ai_config.get('mistral_key'),
            mistral_model_name=ai_config.get('mistral_model'),
            openai_server_url=ai_config.get('openai_url'),
            openai_model_name=ai_config.get('openai_model'),
            openai_api_key=ai_config.get('openai_key')
        )
        
        if raw_response.startswith("Error:"):
            return {"songs": [], "message": f"AI Error: {raw_response}"}
        
        # Parse AI response to extract song titles
        try:
            cleaned = raw_response.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0]
            cleaned = cleaned.strip()
            
            if "[" in cleaned and "]" in cleaned:
                start = cleaned.find("[")
                end = cleaned.rfind("]") + 1
                cleaned = cleaned[start:end]
            
            suggested_titles = json.loads(cleaned)
            log_messages.append(f"AI suggested {len(suggested_titles)} songs")
        except Exception as e:
            log_messages.append(f"Failed to parse AI response: {str(e)}")
            return {"songs": [], "message": "\n".join(log_messages)}
        
        # Query database for exact matches
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            found_songs = []
            for title in suggested_titles:
                cur.execute("""
                    SELECT item_id, title, author
                    FROM public.score
                    WHERE author = %s AND title ILIKE %s
                    LIMIT 1
                """, (artist, f"%{title}%"))
                result = cur.fetchone()
                if result:
                    found_songs.append({
                        "item_id": result['item_id'],
                        "title": result['title'],
                        "artist": result['author']
                    })
            
            # If we found some but not enough, add more random songs from this artist
            if len(found_songs) < get_songs:
                cur.execute("""
                    SELECT item_id, title, author
                    FROM public.score
                    WHERE author = %s
                    ORDER BY RANDOM()
                    LIMIT %s
                """, (artist, get_songs - len(found_songs)))
                additional = cur.fetchall()
                for r in additional:
                    found_songs.append({
                        "item_id": r['item_id'],
                        "title": r['title'],
                        "artist": r['author']
                    })
        
        log_messages.append(f"Found {len(found_songs)} songs by {artist}")
        return {"songs": found_songs, "message": "\n".join(log_messages)}
    finally:
        db_conn.close()


def _ai_brainstorm_sync(user_request: str, ai_config: Dict, get_songs: int) -> List[Dict]:
    """Use AI to brainstorm songs from its knowledge for ANY request when tools aren't enough."""
    from ai import call_ai_for_chat
    
    # Ensure get_songs is int (Gemini may return float)
    get_songs = int(get_songs) if get_songs is not None else 100
    
    db_conn = get_db_connection()
    log_messages = []
    
    try:
        log_messages.append(f"Using AI knowledge to brainstorm songs for: {user_request}")
        
        prompt = f"""You are a music expert with extensive knowledge of songs, artists, and music history. 

User request: "{user_request}"

TASK: Use your knowledge to suggest 25-35 specific songs (with exact artist names) that match this request.

Think about:
- If they want songs similar to an artist → suggest songs by that artist AND similar artists
- If they want a genre/mood → suggest famous songs in that genre/mood
- If they want popular/radio hits → suggest well-known mainstream songs
- If they want a time period → suggest songs from that era
- If they want a vibe → suggest songs that match that feeling

CRITICAL REQUIREMENTS:
1. Return ONLY a JSON array of objects
2. Each object MUST have "title" and "artist" fields
3. Be specific with exact song titles and artist names (as they appear in databases)
4. Include variety - different artists when possible
5. Format: [{{"title": "Song Name", "artist": "Artist Name"}}, ...]
6. NO explanations, NO numbering, ONLY the JSON array

Example format:
[
  {{"title": "All the Small Things", "artist": "blink-182"}},
  {{"title": "Basket Case", "artist": "Green Day"}},
  {{"title": "American Idiot", "artist": "Green Day"}}
]

Suggest songs for "{user_request}" now:"""
        
        raw_response = call_ai_for_chat(
            provider=ai_config['provider'],
            prompt=prompt,
            ollama_url=ai_config.get('ollama_url'),
            ollama_model_name=ai_config.get('ollama_model'),
            gemini_api_key=ai_config.get('gemini_key'),
            gemini_model_name=ai_config.get('gemini_model'),
            mistral_api_key=ai_config.get('mistral_key'),
            mistral_model_name=ai_config.get('mistral_model'),
            openai_server_url=ai_config.get('openai_url'),
            openai_model_name=ai_config.get('openai_model'),
            openai_api_key=ai_config.get('openai_key')
        )
        
        if raw_response.startswith("Error:"):
            return {"songs": [], "message": f"AI Error: {raw_response}"}
        
        # Parse AI response - robust extraction
        try:
            cleaned = raw_response.strip()
            
            # Remove markdown code blocks
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0]
            
            cleaned = cleaned.strip()
            
            # Extract JSON array even if surrounded by text
            if "[" in cleaned and "]" in cleaned:
                start = cleaned.find("[")
                end = cleaned.rfind("]") + 1
                cleaned = cleaned[start:end]
            
            # Replace single quotes with double quotes if needed
            cleaned = cleaned.replace("'\'", '"')
            
            # Try to parse
            song_list = json.loads(cleaned)
            
            if not isinstance(song_list, list):
                raise ValueError("Response is not a JSON array")
            
            log_messages.append(f"AI suggested {len(song_list)} songs")
        except Exception as e:
            log_messages.append(f"Failed to parse AI response: {str(e)}")
            log_messages.append(f"Raw AI response (first 500 chars): {raw_response[:500]}")
            return {"songs": [], "message": "\n".join(log_messages)}
        
        # Search database for these songs (FUZZY match)
        found_songs = []
        for item in song_list:
            title = item.get('title', '')
            artist = item.get('artist', '')
            
            if not title or not artist:
                continue
            
            with db_conn.cursor(cursor_factory=DictCursor) as cur:
                # Fuzzy search - match partial title OR artist
                cur.execute("""
                    SELECT item_id, title, author
                    FROM public.score
                    WHERE LOWER(title) LIKE LOWER(%s) 
                       OR LOWER(author) LIKE LOWER(%s)
                    ORDER BY 
                        CASE 
                            WHEN LOWER(title) LIKE LOWER(%s) AND LOWER(author) LIKE LOWER(%s) THEN 1
                            WHEN LOWER(title) LIKE LOWER(%s) THEN 2
                            WHEN LOWER(author) LIKE LOWER(%s) THEN 3
                            ELSE 4
                        END
                    LIMIT 3
                """, (f"%{title}%", f"%{artist}%", f"%{title}%", f"%{artist}%", f"%{title}%", f"%{artist}%"))
                results = cur.fetchall()
                
                for result in results:
                    song_dict = {
                        "item_id": result['item_id'],
                        "title": result['title'],
                        "artist": result['author']
                    }
                    # Avoid duplicates
                    if song_dict not in found_songs:
                        found_songs.append(song_dict)
            
            if len(found_songs) >= get_songs:
                break
        
        log_messages.append(f"Found {len(found_songs)} songs in database")
        
        return {"songs": found_songs, "ai_suggestions": len(song_list), "message": "\n".join(log_messages)}
    finally:
        db_conn.close()


def _song_similarity_api_sync(song_title: str, song_artist: str, get_songs: int) -> List[Dict]:
    """Synchronous implementation of song similarity API."""
    # Ensure get_songs is int (Gemini may return float)
    get_songs = int(get_songs) if get_songs is not None else 100
    
    db_conn = get_db_connection()
    log_messages = []
    
    try:
        # VALIDATION: Require BOTH title and artist
        if not song_title or not song_title.strip():
            return {
                "songs": [], 
                "message": "ERROR: song_similarity requires a valid song title! If you don't have a specific title, use ai_brainstorm instead."
            }
        if not song_artist or not song_artist.strip():
            return {
                "songs": [], 
                "message": "ERROR: song_similarity requires an artist name! Both title and artist are required."
            }
        
        log_messages.append(f"Looking up song in database: '{song_title}' by '{song_artist}'")
        
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            # STEP 1: Try exact match first
            cur.execute("""
                SELECT item_id, title, author FROM public.score
                WHERE LOWER(title) = LOWER(%s) AND LOWER(author) = LOWER(%s)
                LIMIT 1
            """, (song_title, song_artist))
            seed = cur.fetchone()
            
            # STEP 2: If no exact match, try fuzzy match with normalized text
            if not seed:
                log_messages.append(f"No exact match, trying fuzzy search...")
                # Normalize: remove spaces, dashes, slashes, apostrophes to handle variations
                title_normalized = song_title.replace(' ', '').replace('-', '').replace('‐', '').replace('/', '').replace("'", '')
                artist_normalized = song_artist.replace(' ', '').replace('-', '').replace('‐', '').replace('/', '').replace("'", '')
                
                cur.execute("""
                    SELECT item_id, title, author
                    FROM public.score
                    WHERE REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(title, ' ', ''), '-', ''), '‐', ''), '/', ''), '''', '') ILIKE %s 
                      AND REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(author, ' ', ''), '-', ''), '‐', ''), '/', ''), '''', '') ILIKE %s
                    ORDER BY LENGTH(title) + LENGTH(author)
                    LIMIT 1
                """, (f"%{title_normalized}%", f"%{artist_normalized}%"))
                seed = cur.fetchone()
            
            if not seed:
                return {"songs": [], "message": "\n".join(log_messages) + f"\nSong '{song_title}' by '{song_artist}' not found in database"}
            
            seed_id = seed['item_id']
            actual_title = seed['title']
            actual_artist = seed['author']
            log_messages.append(f"Found: '{actual_title}' by '{actual_artist}' (ID: {seed_id})")
            log_messages.append(f"Found seed song: {song_title} by {song_artist}")
        
        # Use Voyager index to find similar songs
        from tasks.voyager_manager import find_nearest_neighbors_by_id
        similar_results = find_nearest_neighbors_by_id(seed_id, n=get_songs + 1, eliminate_duplicates=False, radius_similarity=False)
        
        # Results have: item_id, distance (but NOT title/author)
        # Extract item_ids in order, excluding the seed song
        similar_ids = [r['item_id'] for r in similar_results if r['item_id'] != seed_id][:get_songs]
        
        # Fetch song details while preserving order
        if not similar_ids:
            songs = []
        else:
            # Create a mapping to preserve the order from Voyager
            id_to_order = {item_id: i for i, item_id in enumerate(similar_ids)}
            
            with db_conn.cursor(cursor_factory=DictCursor) as cur:
                placeholders = ','.join(['%s'] * len(similar_ids))
                cur.execute(f"""
                    SELECT item_id, title, author
                    FROM public.score
                    WHERE item_id IN ({placeholders})
                """, similar_ids)
                results = cur.fetchall()
            
            # Sort results by the original Voyager order
            sorted_results = sorted(results, key=lambda r: id_to_order.get(r['item_id'], 999999))
            songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author']} for r in sorted_results]
        
        log_messages.append(f"Retrieved {len(songs)} similar songs")
        
        return {"songs": songs, "message": "\n".join(log_messages)}
    finally:
        db_conn.close()


def _song_alchemy_sync(add_items: List[Dict], subtract_items: Optional[List[Dict]] = None, get_songs: int = 100) -> Dict:
    """
    Synchronous implementation of song alchemy - blend or subtract musical vibes.
    
    Args:
        add_items: List of items to ADD (blend). Each item: {'type': 'song'|'artist', 'id': '...'}
        subtract_items: Optional list of items to SUBTRACT. Each item: {'type': 'song'|'artist', 'id': '...'}
        get_songs: Number of results to return
    
    Returns:
        Dict with 'songs' (list of song dicts) and 'message' (log string)
    """
    from tasks.song_alchemy import song_alchemy
    
    log_messages = []
    
    try:
        log_messages.append(f"Song Alchemy: ADD {len(add_items)} items" + (f", SUBTRACT {len(subtract_items)} items" if subtract_items else ""))
        
        # Log what's being added
        for item in add_items:
            item_type = item.get('type', 'unknown')
            item_id = item.get('id', 'unknown')
            log_messages.append(f"  + ADD {item_type}: {item_id}")
        
        # Log what's being subtracted
        if subtract_items:
            for item in subtract_items:
                item_type = item.get('type', 'unknown')
                item_id = item.get('id', 'unknown')
                log_messages.append(f"  - SUBTRACT {item_type}: {item_id}")
        
        # Call song alchemy
        result = song_alchemy(
            add_items=add_items,
            subtract_items=subtract_items,
            n_results=get_songs
        )
        
        songs = result.get('results', [])
        log_messages.append(f"Retrieved {len(songs)} songs from alchemy")
        
        return {"songs": songs, "message": "\n".join(log_messages)}
        
    except Exception as e:
        logger.exception(f"Error in song alchemy: {e}")
        log_messages.append(f"Error: {str(e)}")
        return {"songs": [], "message": "\n".join(log_messages)}


def _database_genre_query_sync(
    genres: Optional[List[str]] = None, 
    get_songs: int = 100,
    moods: Optional[List[str]] = None,
    tempo_min: Optional[float] = None,
    tempo_max: Optional[float] = None,
    energy_min: Optional[float] = None,
    energy_max: Optional[float] = None,
    key: Optional[str] = None
) -> List[Dict]:
    """Synchronous implementation of flexible database search with multiple optional filters."""
    # Ensure get_songs is int (Gemini may return float)
    get_songs = int(get_songs) if get_songs is not None else 100
    
    db_conn = get_db_connection()
    log_messages = []
    
    try:
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            # Build conditions
            conditions = []
            params = []
            
            # Genre conditions (OR)
            if genres:
                genre_conditions = []
                for genre in genres:
                    genre_conditions.append("mood_vector LIKE %s")
                    params.append(f"%{genre}%")
                conditions.append("(" + " OR ".join(genre_conditions) + ")")
            
            # Mood/other_features conditions (AND if multiple moods)
            if moods:
                mood_conditions = []
                for mood in moods:
                    mood_conditions.append("other_features LIKE %s")
                    params.append(f"%{mood}%")
                if len(mood_conditions) == 1:
                    conditions.append(mood_conditions[0])
                else:
                    conditions.append("(" + " OR ".join(mood_conditions) + ")")
            
            # Numeric filters (AND)
            if tempo_min is not None:
                conditions.append("tempo >= %s")
                params.append(tempo_min)
            if tempo_max is not None:
                conditions.append("tempo <= %s")
                params.append(tempo_max)
            if energy_min is not None:
                conditions.append("energy >= %s")
                params.append(energy_min)
            if energy_max is not None:
                conditions.append("energy <= %s")
                params.append(energy_max)
            
            # Key filter
            if key:
                conditions.append("key = %s")
                params.append(key.upper())
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            params.append(get_songs)
            
            query = f"""
                SELECT DISTINCT item_id, title, author
                FROM (
                    SELECT item_id, title, author
                    FROM public.score
                    WHERE {where_clause}
                    ORDER BY RANDOM()
                ) AS randomized
                LIMIT %s
            """
            
            cur.execute(query, params)
            results = cur.fetchall()
        
        songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author']} for r in results]
        
        filters = []
        if genres:
            filters.append(f"genres: {', '.join(genres)}")
        if moods:
            filters.append(f"moods: {', '.join(moods)}")
        if tempo_min or tempo_max:
            filters.append(f"tempo: {tempo_min or 'any'}-{tempo_max or 'any'}")
        if energy_min or energy_max:
            filters.append(f"energy: {energy_min or 'any'}-{energy_max or 'any'}")
        if key:
            filters.append(f"key: {key}")
        
        log_messages.append(f"Found {len(songs)} songs matching {', '.join(filters) if filters else 'all criteria'}")
        
        return {"songs": songs, "message": "\n".join(log_messages)}
    finally:
        db_conn.close()


def _database_tempo_energy_query_sync(
    tempo_min: Optional[float],
    tempo_max: Optional[float],
    energy_min: Optional[float],
    energy_max: Optional[float],
    get_songs: int
) -> List[Dict]:
    """Synchronous implementation of tempo/energy query."""
    db_conn = get_db_connection()
    log_messages = []
    
    try:
        conditions = []
        params = []
        
        if tempo_min is not None:
            conditions.append("tempo >= %s")
            params.append(tempo_min)
        if tempo_max is not None:
            conditions.append("tempo <= %s")
            params.append(tempo_max)
        if energy_min is not None:
            conditions.append("energy >= %s")
            params.append(energy_min)
        if energy_max is not None:
            conditions.append("energy <= %s")
            params.append(energy_max)
        
        if not conditions:
            return {"songs": [], "message": "No tempo or energy criteria specified"}
        
        where_clause = " AND ".join(conditions)
        params.append(get_songs)
        
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            query = f"""
                SELECT DISTINCT item_id, title, author
                FROM (
                    SELECT item_id, title, author
                    FROM public.score
                    WHERE {where_clause}
                    ORDER BY RANDOM()
                ) AS randomized
                LIMIT %s
            """
            cur.execute(query, params)
            results = cur.fetchall()
        
        songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author']} for r in results]
        log_messages.append(f"Found {len(songs)} songs matching tempo/energy criteria")
        
        return {"songs": songs, "message": "\n".join(log_messages)}
    finally:
        db_conn.close()


def _vibe_match_sync(vibe_description: str, ai_config: Dict, get_songs: int) -> List[Dict]:
    """Synchronous implementation of vibe matching using AI."""
    from ai import call_ai_for_chat
    
    db_conn = get_db_connection()
    log_messages = []
    
    try:
        log_messages.append(f"Using AI to match vibe: {vibe_description}")
        
        prompt = f"""You are a music database expert. The user wants songs matching this vibe: "{vibe_description}"

Analyze this vibe and return a JSON object with search criteria for a music database.

Database schema:
- mood_vector: Contains genres like 'rock', 'pop', 'jazz', etc.
- other_features: Contains moods like 'danceable', 'party', 'relaxed', etc.
- energy: 0.01-0.15 (higher = more energetic)
- tempo: 40-200 BPM

Return ONLY this JSON structure:
{{
    "genres": ["genre1", "genre2"],
    "moods": ["mood1", "mood2"],
    "energy_min": 0.05,
    "energy_max": 0.12,
    "tempo_min": 100,
    "tempo_max": 140
}}

Return the JSON now:"""
        
        raw_response = call_ai_for_chat(
            provider=ai_config['provider'],
            prompt=prompt,
            ollama_url=ai_config.get('ollama_url'),
            ollama_model_name=ai_config.get('ollama_model'),
            gemini_api_key=ai_config.get('gemini_key'),
            gemini_model_name=ai_config.get('gemini_model'),
            mistral_api_key=ai_config.get('mistral_key'),
            mistral_model_name=ai_config.get('mistral_model'),
            openai_server_url=ai_config.get('openai_url'),
            openai_model_name=ai_config.get('openai_model'),
            openai_api_key=ai_config.get('openai_key')
        )
        
        if raw_response.startswith("Error:"):
            return {"songs": [], "message": f"AI Error: {raw_response}"}
        
        # Parse AI response
        try:
            cleaned = raw_response.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0]
            cleaned = cleaned.strip()
            
            if "{" in cleaned and "}" in cleaned:
                start = cleaned.find("{")
                end = cleaned.rfind("}") + 1
                cleaned = cleaned[start:end]
            
            criteria = json.loads(cleaned)
        except Exception as e:
            log_messages.append(f"Failed to parse AI response: {str(e)}")
            return {"songs": [], "message": "\n".join(log_messages)}
        
        # Build SQL query from criteria
        conditions = []
        params = []
        
        # Add genre conditions
        for genre in criteria.get('genres', []):
            conditions.append("mood_vector LIKE %s")
            params.append(f"%{genre}%")
        
        # Add mood conditions
        for mood in criteria.get('moods', []):
            conditions.append("other_features LIKE %s")
            params.append(f"%{mood}%")
        
        # Add energy/tempo conditions
        if 'energy_min' in criteria:
            conditions.append("energy >= %s")
            params.append(criteria['energy_min'])
        if 'energy_max' in criteria:
            conditions.append("energy <= %s")
            params.append(criteria['energy_max'])
        if 'tempo_min' in criteria:
            conditions.append("tempo >= %s")
            params.append(criteria['tempo_min'])
        if 'tempo_max' in criteria:
            conditions.append("tempo <= %s")
            params.append(criteria['tempo_max'])
        
        if not conditions:
            return {"songs": [], "message": "AI did not provide valid search criteria"}
        
        where_clause = " AND ".join(conditions)
        params.append(get_songs)
        
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            query = f"""
                SELECT DISTINCT item_id, title, author
                FROM (
                    SELECT item_id, title, author
                    FROM public.score
                    WHERE {where_clause}
                    ORDER BY RANDOM()
                ) AS randomized
                LIMIT %s
            """
            cur.execute(query, params)
            results = cur.fetchall()
        
        songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author']} for r in results]
        log_messages.append(f"Found {len(songs)} songs matching vibe criteria")
        
        return {"songs": songs, "criteria": criteria, "message": "\n".join(log_messages)}
    finally:
        db_conn.close()


def _explore_database_sync(
    artists: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    song_titles: Optional[List[str]] = None
) -> Dict:
    """Synchronous implementation of database exploration."""
    from tasks.chat_manager import explore_database_for_matches
    
    db_conn = get_db_connection()
    log_messages = []
    
    try:
        results = explore_database_for_matches(
            db_conn,
            artists or [],
            keywords or [],
            song_titles or [],
            log_messages
        )
        results['message'] = "\n".join(log_messages)
        return results
    finally:
        db_conn.close()


# ==================== MCP TOOL DEFINITIONS ====================

@mcp_server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available MCP tools - 4 CORE TOOLS."""
    return [
        Tool(
            name="artist_similarity",
            description="EXACT API: Find songs from similar artists (NOT the artist's own songs).",
            inputSchema={
                "type": "object",
                "properties": {
                    "artist": {
                        "type": "string",
                        "description": "Name of the artist"
                    },
                    "get_songs": {
                        "type": "integer",
                        "description": "Number of songs to retrieve",
                        "default": 100
                    }
                },
                "required": ["artist"]
            }
        ),
        Tool(
            name="song_similarity",
            description="EXACT API: Find songs similar to a specific song (requires title+artist).",
            inputSchema={
                "type": "object",
                "properties": {
                    "song_title": {
                        "type": "string",
                        "description": "Title of the song"
                    },
                    "song_artist": {
                        "type": "string",
                        "description": "Artist of the song"
                    },
                    "get_songs": {
                        "type": "integer",
                        "description": "Number of similar songs",
                        "default": 100
                    }
                },
                "required": ["song_title", "song_artist"]
            }
        ),
        Tool(
            name="search_database",
            description="EXACT DB: Search by genre/mood/tempo/energy filters. COMBINE all filters in ONE call!",
            inputSchema={
                "type": "object",
                "properties": {
                    "genres": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Genres (rock, pop, metal, jazz, etc.)"
                    },
                    "moods": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Moods (danceable, aggressive, happy, party, relaxed, sad)"
                    },
                    "tempo_min": {
                        "type": "number",
                        "description": "Min BPM (40-200)"
                    },
                    "tempo_max": {
                        "type": "number",
                        "description": "Max BPM (40-200)"
                    },
                    "energy_min": {
                        "type": "number",
                        "description": "Min energy (0.01-0.15)"
                    },
                    "energy_max": {
                        "type": "number",
                        "description": "Max energy (0.01-0.15)"
                    },
                    "key": {
                        "type": "string",
                        "description": "Musical key (C, D, E, F, G, A, B with # or b)"
                    },
                    "get_songs": {
                        "type": "integer",
                        "description": "Number of songs",
                        "default": 100
                    }
                }
            }
        ),
        Tool(
            name="song_alchemy",
            description="""VECTOR ARITHMETIC: Musical math - blend artists/songs or remove unwanted vibes.

WHEN TO USE THIS TOOL:
✅ USE for these patterns:
  - "Like X but NOT Y" → add X, subtract Y
  - "Mix/blend two artists" → add both artists
  - "Artist X meets Artist Y" → add both
  - "This song but calmer/faster/darker" → add song, subtract mood
  - "Remove unwanted vibe" → add wanted, subtract unwanted
  
❌ DO NOT USE for:
  - Simple artist search → use artist_similarity instead
  - Genre/mood search → use search_database instead
  - Just finding similar songs → use song_similarity instead
  
EXAMPLES:
  ✅ "Songs like The Beatles but not ballads" → add_items: Beatles (artist), subtract_items: slow ballad song
  ✅ "Radiohead meets Pink Floyd" → add_items: Radiohead (artist) + Pink Floyd (artist)
  ✅ "Energetic rock but not aggressive" → add_items: energetic rock songs, subtract_items: aggressive song
  ❌ "Find AC/DC songs" → WRONG! Use artist_similarity or artist_hits instead
  ❌ "Metal songs" → WRONG! Use search_database with genres=['metal'] instead

CRITICAL: You can ADD/SUBTRACT both ARTISTS and SONGS. Each item needs 'type' ('artist' or 'song') and 'id'.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "add_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["song", "artist"],
                                    "description": "Type of item: 'song' for specific songs, 'artist' for artist's style"
                                },
                                "id": {
                                    "type": "string",
                                    "description": "Item ID from database (item_id for songs, artist name for artists)"
                                }
                            },
                            "required": ["type", "id"]
                        },
                        "description": "Items to ADD (blend together). REQUIRED - at least one item."
                    },
                    "subtract_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["song", "artist"],
                                    "description": "Type of item: 'song' for specific songs, 'artist' for artist's style"
                                },
                                "id": {
                                    "type": "string",
                                    "description": "Item ID from database (item_id for songs, artist name for artists)"
                                }
                            },
                            "required": ["type", "id"]
                        },
                        "description": "Items to SUBTRACT (remove this vibe). OPTIONAL - use only for 'but not' requests."
                    },
                    "get_songs": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 100
                    }
                },
                "required": ["add_items"]
            }
        ),
        Tool(
            name="ai_brainstorm",
            description="AI KNOWLEDGE: Suggest songs for complex requests (artist's own songs, trending, era, etc.).",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_request": {
                        "type": "string",
                        "description": "The user's request (e.g., 'trending songs 2025', 'top radio hits', 'greatest rock songs')"
                    },
                    "ai_provider": {
                        "type": "string",
                        "description": "AI provider",
                        "enum": ["OLLAMA", "GEMINI", "MISTRAL", "OPENAI"]
                    },
                    "ai_config": {
                        "type": "object",
                        "description": "AI configuration"
                    },
                    "get_songs": {
                        "type": "integer",
                        "description": "Number of songs",
                        "default": 100
                    }
                },
                "required": ["user_request", "ai_provider", "ai_config"]
            }
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Handle tool calls from AI - 4 CORE TOOLS."""
    try:
        if name == "artist_similarity":
            result = await run_in_executor(
                _artist_similarity_api_sync,
                arguments['artist'],
                15,  # count - hardcoded
                arguments.get('get_songs', 100)
            )
        elif name == "song_similarity":
            result = await run_in_executor(
                _song_similarity_api_sync,
                arguments['song_title'],
                arguments['song_artist'],
                arguments.get('get_songs', 100)
            )
        elif name == "search_database":
            result = await run_in_executor(
                _database_genre_query_sync,
                arguments.get('genres'),
                arguments.get('get_songs', 100),
                arguments.get('moods'),
                arguments.get('tempo_min'),
                arguments.get('tempo_max'),
                arguments.get('energy_min'),
                arguments.get('energy_max'),
                arguments.get('key')
            )
        elif name == "ai_brainstorm":
            ai_config = {
                'provider': arguments['ai_provider'],
                **arguments['ai_config']
            }
            result = await run_in_executor(
                _ai_brainstorm_sync,
                arguments['user_request'],
                ai_config,
                arguments.get('get_songs', 100)
            )
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        
        # Return result as JSON
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    except Exception as e:
        logger.exception(f"Error executing tool {name}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


# ==================== SERVER RUNNER ====================

async def run_mcp_server():
    """Run the MCP server using stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options()
        )


def main():
    """Main entry point for the MCP server."""
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr  # MCP uses stdout for protocol, stderr for logs
    )
    
    logger.info("Starting AudioMuse-AI MCP Server...")
    asyncio.run(run_mcp_server())


if __name__ == "__main__":
    main()
