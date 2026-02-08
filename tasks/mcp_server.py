"""
Playlist Generation Tool Functions
Sync functions for playlist generation used by the web interface (chat.html).
Each function implements a specific search/query strategy.
"""
import logging
import json
import re
from typing import List, Dict, Optional
import psycopg2
from psycopg2.extras import DictCursor

logger = logging.getLogger(__name__)

# Cache for library context (refreshed once per app lifetime or on demand)
_library_context_cache = None


def get_db_connection():
    """Get database connection using config settings."""
    from config import DATABASE_URL
    return psycopg2.connect(DATABASE_URL)


def get_library_context(force_refresh: bool = False) -> Dict:
    """Query the database once to build a summary of the user's music library.

    Returns a dict with:
        total_songs, unique_artists, top_genres (list), year_min, year_max,
        has_ratings (bool), rated_songs_pct (float)
    """
    global _library_context_cache
    if _library_context_cache is not None and not force_refresh:
        return _library_context_cache

    db_conn = get_db_connection()
    try:
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            # Basic counts
            cur.execute("SELECT COUNT(*) AS cnt, COUNT(DISTINCT author) AS artists FROM public.score")
            row = cur.fetchone()
            total_songs = row['cnt']
            unique_artists = row['artists']

            # Year range
            cur.execute("SELECT MIN(year) AS ymin, MAX(year) AS ymax FROM public.score WHERE year IS NOT NULL AND year > 0")
            yr = cur.fetchone()
            year_min = yr['ymin']
            year_max = yr['ymax']

            # Rating coverage
            cur.execute("SELECT COUNT(*) AS rated FROM public.score WHERE rating IS NOT NULL AND rating > 0")
            rated_count = cur.fetchone()['rated']
            rated_pct = round(100.0 * rated_count / total_songs, 1) if total_songs > 0 else 0

            # Top genres from mood_vector (extract genre names and count occurrences)
            # mood_vector format: "rock:0.82,pop:0.45,..."
            cur.execute("""
                SELECT unnest(string_to_array(mood_vector, ',')) AS tag
                FROM public.score
                WHERE mood_vector IS NOT NULL AND mood_vector != ''
            """)
            genre_counts = {}
            for r in cur:
                tag = r['tag'].strip()
                if ':' in tag:
                    name = tag.split(':')[0].strip()
                    if name:
                        genre_counts[name] = genre_counts.get(name, 0) + 1
            top_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:15]

            # Available scales
            cur.execute("SELECT DISTINCT scale FROM public.score WHERE scale IS NOT NULL AND scale != '' ORDER BY scale")
            scales = [r['scale'] for r in cur.fetchall()]

            # Top moods from other_features (extract mood tags and count occurrences)
            # other_features format: "danceable, aggressive, happy" (comma-separated)
            cur.execute("""
                SELECT unnest(string_to_array(other_features, ',')) AS mood
                FROM public.score
                WHERE other_features IS NOT NULL AND other_features != ''
            """)
            mood_counts = {}
            for r in cur:
                mood = r['mood'].strip().lower()
                if mood:
                    mood_counts[mood] = mood_counts.get(mood, 0) + 1
            top_moods = sorted(mood_counts, key=mood_counts.get, reverse=True)[:10]

        ctx = {
            'total_songs': total_songs,
            'unique_artists': unique_artists,
            'top_genres': top_genres,
            'top_moods': top_moods,
            'year_min': year_min,
            'year_max': year_max,
            'has_ratings': rated_count > 0,
            'rated_songs_pct': rated_pct,
            'scales': scales,
        }
        _library_context_cache = ctx
        return ctx
    except Exception as e:
        logger.warning(f"Failed to get library context: {e}")
        return {
            'total_songs': 0, 'unique_artists': 0, 'top_genres': [],
            'top_moods': [], 'year_min': None, 'year_max': None,
            'has_ratings': False, 'rated_songs_pct': 0, 'scales': [],
        }
    finally:
        db_conn.close()


def _artist_similarity_api_sync(artist: str, count: int, get_songs: int) -> List[Dict]:
    """Synchronous implementation of artist similarity API."""
    from tasks.artist_gmm_manager import find_similar_artists
    
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


def _text_search_sync(description: str, tempo_filter: Optional[str], energy_filter: Optional[str], get_songs: int) -> Dict:
    """Synchronous implementation of CLAP text search with optional hybrid filtering."""
    from tasks.clap_text_search import search_by_text
    from config import CLAP_ENABLED
    
    db_conn = get_db_connection()
    log_messages = []
    
    try:
        if not CLAP_ENABLED:
            log_messages.append("CLAP text search is disabled")
            return {"songs": [], "message": "CLAP text search is not enabled. Please enable CLAP_ENABLED in config."}
        
        if not description:
            return {"songs": [], "message": "No description provided for text search"}
        
        log_messages.append(f"CLAP text search: '{description}'")
        
        # Get up to 100 songs from CLAP
        clap_results = search_by_text(description, limit=100)
        
        if not clap_results:
            log_messages.append("No results from CLAP text search")
            return {"songs": [], "message": "\n".join(log_messages)}
        
        log_messages.append(f"CLAP returned {len(clap_results)} songs")
        
        # If tempo/energy filters specified, apply hybrid filtering
        if tempo_filter or energy_filter:
            log_messages.append(f"Applying hybrid filters (tempo: {tempo_filter}, energy: {energy_filter})")
            
            # Get item_ids from CLAP results
            item_ids = [r['item_id'] for r in clap_results]
            
            # Define tempo ranges (BPM)
            tempo_ranges = {
                'slow': (0, 90),
                'medium': (90, 140),
                'fast': (140, 300)
            }
            
            # Define energy ranges (normalized)
            energy_ranges = {
                'low': (0, 0.05),
                'medium': (0.05, 0.10),
                'high': (0.10, 1.0)
            }
            
            # Build SQL filter conditions
            filter_conditions = []
            query_params = []
            
            if tempo_filter and tempo_filter in tempo_ranges:
                tempo_min, tempo_max = tempo_ranges[tempo_filter]
                filter_conditions.append("tempo >= %s AND tempo < %s")
                query_params.extend([tempo_min, tempo_max])
            
            if energy_filter and energy_filter in energy_ranges:
                energy_min, energy_max = energy_ranges[energy_filter]
                filter_conditions.append("energy >= %s AND energy < %s")
                query_params.extend([energy_min, energy_max])
            
            # Query database to filter by tempo/energy
            with db_conn.cursor(cursor_factory=DictCursor) as cur:
                placeholders = ','.join(['%s'] * len(item_ids))
                where_clause = ' AND '.join(filter_conditions)
                
                sql = f"""
                    SELECT item_id, title, author
                    FROM public.score
                    WHERE item_id IN ({placeholders})
                    AND {where_clause}
                """
                
                cur.execute(sql, item_ids + query_params)
                filtered_results = cur.fetchall()
            
            # Preserve CLAP similarity order for filtered results
            filtered_item_ids = {r['item_id'] for r in filtered_results}
            songs = [
                {"item_id": r['item_id'], "title": r['title'], "artist": r['author']}
                for r in clap_results
                if r['item_id'] in filtered_item_ids
            ]
            
            log_messages.append(f"Filtered to {len(songs)} songs matching tempo/energy criteria")
        else:
            # No filters - return CLAP results as-is
            songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author']} for r in clap_results]
            log_messages.append(f"Retrieved {len(songs)} songs from CLAP")
        
        return {"songs": songs[:get_songs], "message": "\n".join(log_messages)}
    except Exception as e:
        import traceback
        log_messages.append(f"Error in text search: {str(e)}")
        log_messages.append(traceback.format_exc())
        return {"songs": [], "message": "\n".join(log_messages)}
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
        
        # Search database for these songs using strict two-stage matching
        found_songs = []
        seen_ids = set()

        def _normalize(s: str) -> str:
            """Strip spaces, dashes, apostrophes for fuzzy comparison."""
            return re.sub(r"[\s\-\u2010\u2011\u2012\u2013\u2014/'\".,!?()]", '', s).lower()

        def _escape_like(s: str) -> str:
            """Escape LIKE wildcards to prevent injection."""
            return s.replace('%', r'\%').replace('_', r'\_')

        for item in song_list:
            title = item.get('title', '')
            artist = item.get('artist', '')

            if not title or not artist:
                continue

            with db_conn.cursor(cursor_factory=DictCursor) as cur:
                # Stage 1: Exact case-insensitive match on BOTH title AND artist
                cur.execute("""
                    SELECT item_id, title, author
                    FROM public.score
                    WHERE LOWER(title) = LOWER(%s) AND LOWER(author) = LOWER(%s)
                    LIMIT 1
                """, (title, artist))
                result = cur.fetchone()

                # Stage 2: Normalized fuzzy match requiring BOTH title AND artist to match
                if not result:
                    title_norm = _normalize(title)
                    artist_norm = _normalize(artist)
                    if title_norm and artist_norm:
                        cur.execute("""
                            SELECT item_id, title, author
                            FROM public.score
                            WHERE LOWER(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(title, ' ', ''), '-', ''), '''', ''), '.', ''), ',', ''))
                                  LIKE LOWER(%s)
                              AND LOWER(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(author, ' ', ''), '-', ''), '''', ''), '.', ''), ',', ''))
                                  LIKE LOWER(%s)
                            ORDER BY LENGTH(title) + LENGTH(author)
                            LIMIT 1
                        """, (f"%{_escape_like(title_norm)}%", f"%{_escape_like(artist_norm)}%"))
                        result = cur.fetchone()

                if result and result['item_id'] not in seen_ids:
                    found_songs.append({
                        "item_id": result['item_id'],
                        "title": result['title'],
                        "artist": result['author']
                    })
                    seen_ids.add(result['item_id'])

            if len(found_songs) >= get_songs:
                break

        log_messages.append(f"Found {len(found_songs)} songs in database (from {len(song_list)} AI suggestions)")

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
    key: Optional[str] = None,
    scale: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    min_rating: Optional[int] = None
) -> List[Dict]:
    """Synchronous implementation of flexible database search with multiple optional filters.

    Improvements over the original:
    - Genre matching uses regex to avoid substring false positives (e.g. 'rock' won't match 'indie rock')
    - Results are ordered by genre confidence score sum (relevance) instead of RANDOM()
    - Supports scale (major/minor), year range, and minimum rating filters
    """
    # Ensure get_songs is int (Gemini may return float)
    get_songs = int(get_songs) if get_songs is not None else 100

    db_conn = get_db_connection()
    log_messages = []

    try:
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            # Build conditions
            conditions = []
            params = []

            # Genre conditions (OR) - use regex to match whole genre names with confidence scores
            # mood_vector format: "rock:0.82,pop:0.45,indie rock:0.31"
            # We want "rock" to match "rock:0.82" but NOT "indie rock:0.31"
            has_genre_filter = False
            if genres:
                genre_conditions = []
                for genre in genres:
                    # Match genre at start of string or after comma, followed by colon
                    # PostgreSQL regex: (^|,)\s*rock:
                    genre_conditions.append("mood_vector ~* %s")
                    params.append(f"(^|,)\\s*{re.escape(genre)}:")
                conditions.append("(" + " OR ".join(genre_conditions) + ")")
                has_genre_filter = True

            # Mood/other_features conditions (OR)
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

            # Scale filter (major/minor)
            if scale:
                conditions.append("LOWER(scale) = LOWER(%s)")
                params.append(scale)

            # Year range filter
            if year_min is not None:
                conditions.append("year >= %s")
                params.append(int(year_min))
            if year_max is not None:
                conditions.append("year <= %s")
                params.append(int(year_max))

            # Minimum rating filter
            if min_rating is not None:
                conditions.append("rating >= %s")
                params.append(int(min_rating))

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            params.append(get_songs)

            # Use relevance ranking when genre filter is active, otherwise random
            if has_genre_filter:
                # Build a scoring expression that sums confidence scores for matched genres
                # For each requested genre, extract its score from mood_vector and sum them
                score_parts = []
                score_params = []
                for genre in genres:
                    # Extract the numeric score after 'genre:' using regex
                    score_parts.append("""
                        COALESCE(
                            CAST(
                                NULLIF(
                                    SUBSTRING(mood_vector FROM %s),
                                    ''
                                ) AS NUMERIC
                            ),
                            0
                        )
                    """)
                    # Regex to capture the score value: (?:^|,)\s*rock:(\d+\.?\d*)
                    score_params.append(f"(?:^|,)\\s*{re.escape(genre)}:(\\d+\\.?\\d*)")

                relevance_expr = " + ".join(score_parts)
                all_params = score_params + params

                query = f"""
                    SELECT DISTINCT item_id, title, author
                    FROM (
                        SELECT item_id, title, author,
                               ({relevance_expr}) AS relevance_score
                        FROM public.score
                        WHERE {where_clause}
                        ORDER BY relevance_score DESC, RANDOM()
                    ) AS ranked
                    LIMIT %s
                """
                cur.execute(query, all_params)
            else:
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
        if scale:
            filters.append(f"scale: {scale}")
        if year_min or year_max:
            filters.append(f"year: {year_min or 'any'}-{year_max or 'any'}")
        if min_rating:
            filters.append(f"min_rating: {min_rating}")

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
