"""
Chat Manager Module
Handles AI-powered playlist generation logic including:
- Action execution (artist queries, genre filtering, vibe matching, etc.)
- AI step processing and JSON parsing
- Database exploration for matching content
- SQL query generation for playlist creation
"""

import logging
import json
import re
from psycopg2.extras import DictCursor

logger = logging.getLogger(__name__)


def call_ai_step(step_name, prompt, ai_config, log_messages):
    """
    Helper function to call AI for a specific step and parse JSON response.
    Returns (parsed_dict, error_message).
    """
    from ai import call_ai_for_chat
    
    log_messages.append(f"\n--- {step_name} ---")
    
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
        log_messages.append(f"AI Error: {raw_response}")
        return None, raw_response
    
    log_messages.append(f"AI Response:\n{raw_response}")
    
    # Try to parse JSON from response with aggressive cleaning
    try:
        cleaned = raw_response.strip()
        
        # Remove markdown code blocks
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]
        
        cleaned = cleaned.strip()
        
        # Try to extract JSON object if there's text before/after
        if "{" in cleaned and "}" in cleaned:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            cleaned = cleaned[start:end]
        
        parsed = json.loads(cleaned)
        return parsed, None
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse JSON response: {str(e)}"
        log_messages.append(error_msg)
        log_messages.append(f"Attempted to parse: {cleaned[:200]}")
        return None, error_msg


def explore_database_for_matches(db_conn, expanded_artists, expanded_keywords, expanded_song_titles, log_messages):
    """
    Explores the database to find what actually exists.
    Returns dict with found_artists, artist_song_count, found_keywords, found_song_titles.
    """
    log_messages.append("\n--- Database Exploration ---")
    
    found_artists = []
    artist_song_count = 0
    found_keywords = []
    found_song_titles = []  # List of (title, artist) tuples
    
    try:
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            # PRIORITY: Search for specific song titles first (critical for temporal queries)
            if expanded_song_titles:
                log_messages.append(f"Searching for {len(expanded_song_titles)} specific song titles...")
                seen_song_titles = set()  # Track (title, author) pairs to avoid duplicates
                
                for song_title in expanded_song_titles:
                    # Create a simplified version for matching (remove all special chars, spaces, lowercase)
                    simplified = ''.join(c.lower() for c in song_title if c.isalnum())
                    
                    # Query with EXACT matching strategies (not partial substring)
                    query = """
                        SELECT DISTINCT title, author
                        FROM public.score
                        WHERE 
                            -- Strategy 1: Exact case-insensitive match
                            LOWER(title) = LOWER(%s)
                            -- Strategy 2: Exact alphanumeric-only match (handles special chars)
                            OR LOWER(REGEXP_REPLACE(title, '[^a-zA-Z0-9]', '', 'g')) = %s
                        LIMIT 10
                    """
                    
                    try:
                        cur.execute(query, (song_title, simplified))
                        results = cur.fetchall()
                        
                        for row in results:
                            title_author_pair = (row['title'], row['author'])
                            # Only add if not already seen (prevents duplicates)
                            if title_author_pair not in seen_song_titles:
                                found_song_titles.append(title_author_pair)
                                seen_song_titles.add(title_author_pair)
                    except Exception as e:
                        # If regex not supported, fall back to exact LOWER match
                        cur.execute("""
                            SELECT DISTINCT title, author
                            FROM public.score
                            WHERE LOWER(title) = LOWER(%s)
                            LIMIT 10
                        """, (song_title,))
                        results = cur.fetchall()
                        
                        for row in results:
                            title_author_pair = (row['title'], row['author'])
                            if title_author_pair not in seen_song_titles:
                                found_song_titles.append(title_author_pair)
                                seen_song_titles.add(title_author_pair)
                
                if found_song_titles:
                    log_messages.append(f"Found {len(found_song_titles)} unique matching song titles:")
                    for title, artist in found_song_titles[:10]:
                        log_messages.append(f"  - '{title}' by {artist}")
                else:
                    log_messages.append("⚠️ No matching song titles found in database")
            
            # Search for artists with aggressive fuzzy matching
            if expanded_artists:
                # For each artist, do a separate query to catch variations
                seen_artists = set()  # Avoid duplicates
                
                for artist in expanded_artists:
                    # Create a simplified version for matching (remove all special chars, spaces, lowercase)
                    # This will match: "Blink182" = "Blink-182" = "blink‐182" = "blink 182"
                    simplified = ''.join(c.lower() for c in artist if c.isalnum())
                    
                    # Query with multiple fuzzy matching strategies
                    query = """
                        SELECT DISTINCT author, COUNT(*) as song_count
                        FROM public.score
                        WHERE 
                            -- Strategy 1: Case-insensitive partial match
                            author ILIKE %s
                            -- Strategy 2: Simplified alphanumeric-only match
                            OR LOWER(REGEXP_REPLACE(author, '[^a-zA-Z0-9]', '', 'g')) LIKE %s
                        GROUP BY author
                    """
                    
                    try:
                        cur.execute(query, (f"%{artist}%", f"%{simplified}%"))
                        results = cur.fetchall()
                        
                        for row in results:
                            author_name = row['author']
                            if author_name not in seen_artists:
                                found_artists.append(author_name)
                                artist_song_count += row['song_count']
                                seen_artists.add(author_name)
                    except Exception as e:
                        # If regex not supported, fall back to simple ILIKE
                        log_messages.append(f"Fuzzy search failed for '{artist}', using simple match: {str(e)}")
                        cur.execute("""
                            SELECT DISTINCT author, COUNT(*) as song_count
                            FROM public.score
                            WHERE author ILIKE %s
                            GROUP BY author
                        """, (f"%{artist}%",))
                        results = cur.fetchall()
                        
                        for row in results:
                            author_name = row['author']
                            if author_name not in seen_artists:
                                found_artists.append(author_name)
                                artist_song_count += row['song_count']
                                seen_artists.add(author_name)
                
                # Sort by song count (most songs first)
                if found_artists:
                    # Get counts for sorting
                    artist_counts = {}
                    for artist in found_artists:
                        cur.execute("SELECT COUNT(*) as cnt FROM public.score WHERE author = %s", (artist,))
                        artist_counts[artist] = cur.fetchone()['cnt']
                    found_artists.sort(key=lambda x: artist_counts.get(x, 0), reverse=True)
                
                log_messages.append(f"Found {len(found_artists)} matching artists with {artist_song_count} total songs")
                if found_artists:
                    log_messages.append(f"Artists found: {', '.join(found_artists[:10])}")
            
            # Search for mood/genre keywords in mood_vector
            if expanded_keywords:
                for keyword in expanded_keywords:
                    # Check if this mood/genre exists in mood_vector
                    cur.execute("""
                        SELECT COUNT(*) as cnt
                        FROM public.score
                        WHERE mood_vector LIKE %s
                        LIMIT 1
                    """, (f"%{keyword}:%",))
                    result = cur.fetchone()
                    if result and result['cnt'] > 0:
                        found_keywords.append(keyword)
                
                if found_keywords:
                    log_messages.append(f"Found moods/genres: {', '.join(found_keywords)}")
    
    except Exception as e:
        logger.exception("Error during database exploration")
        log_messages.append(f"Database exploration error: {str(e)}")
    
    return {
        'found_artists': found_artists,
        'artist_song_count': artist_song_count,
        'found_keywords': found_keywords,
        'found_song_titles': found_song_titles
    }


def execute_action(action, db_conn, log_messages, ai_config=None):
    """
    Execute a single action from the AI's execution plan.
    Returns list of song results.
    ai_config is optional and only needed for artist_hits_query and vibe_match actions.
    """
    from ai import call_ai_for_chat
    
    action_type = action.get('action')
    params = action.get('params', {})
    get_songs_count = action.get('get_songs', 150)  # Higher default to account for duplicates
    
    log_messages.append(f"\n🤖 EXECUTING: {action_type}")
    log_messages.append(f"   Parameters: {params}")
    
    try:
        if action_type == "artist_similarity_api":
            from tasks.artist_gmm_manager import find_similar_artists
            artist = params.get('artist')
            
            similar_artists = find_similar_artists(artist, n=25)
            
            # If not found, try fuzzy matching by removing special chars
            if not similar_artists:
                artist_simplified = ''.join(c.lower() for c in artist if c.isalnum() or c.isspace())
                log_messages.append(f"   ⚠️ Artist '{artist}' not found, trying fuzzy match: '{artist_simplified}'")
                
                # Try to find the artist in database directly
                with db_conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute("""
                        SELECT DISTINCT author FROM public.score
                        WHERE LOWER(REGEXP_REPLACE(author, '[^a-zA-Z0-9 ]', '', 'g')) LIKE %s
                        LIMIT 1
                    """, (f'%{artist_simplified}%',))
                    fuzzy_result = cur.fetchone()
                
                if fuzzy_result:
                    fuzzy_artist = fuzzy_result['author']
                    log_messages.append(f"   ✓ Found fuzzy match: '{fuzzy_artist}'")
                    similar_artists = find_similar_artists(fuzzy_artist, n=25)
                
            if not similar_artists:
                log_messages.append(f"   ❌ Artist '{artist}' not found (even with fuzzy matching)")
                return []
            
            artist_names = [a['artist'] for a in similar_artists[:15]]
            log_messages.append(f"   ✓ Found {len(artist_names)} similar artists")
            
            # Query songs from these artists
            with db_conn.cursor(cursor_factory=DictCursor) as cur:
                placeholders = ','.join(['%s'] * len(artist_names))
                sql = f"""
                    SELECT DISTINCT item_id, title, author FROM (
                        SELECT item_id, title, author FROM public.score
                        WHERE author IN ({placeholders})
                        ORDER BY RANDOM()
                    ) AS randomized LIMIT %s
                """
                cur.execute(sql, artist_names + [get_songs_count])
                results = cur.fetchall()
                
            songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author']} for r in results]
            log_messages.append(f"   ✓ Retrieved {len(songs)} songs")
            return songs
        
        elif action_type == "artist_hits_query":
            """
            Get the artist's OWN famous songs.
            Uses AI knowledge to suggest their hits, then queries database for exact matches.
            Different from artist_similarity_api which returns songs from SIMILAR artists.
            """
            artist = params.get('artist')
            
            # Use AI to suggest this artist's famous songs
            log_messages.append(f"   🧠 Using AI knowledge to suggest {artist}'s famous songs...")
            
            # Check if ai_config is provided
            if not ai_config:
                log_messages.append(f"   ❌ AI config not provided for artist_hits_query")
                return []
            
            artist_hits_prompt = f"""You are a music expert. List the most famous, popular songs by the artist "{artist}".

CRITICAL: Return ONLY a JSON array of song titles - NO explanations, NO markdown, NO extra text.
These should be REAL songs that actually exist by this artist.
Format: ["Song Title 1", "Song Title 2", "Song Title 3", ...]

Provide 20-30 of their most well-known hits."""
            
            # Call AI to get song suggestions (supports all providers)
            ai_response = call_ai_for_chat(
                provider=ai_config['provider'],
                prompt=artist_hits_prompt,
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
            
            if ai_response.startswith("Error:"):
                log_messages.append(f"   ❌ AI call failed: {ai_response}")
                return []
            
            # Parse AI response to extract song titles
            try:
                # Try to find JSON array in response
                json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
                if json_match:
                    suggested_titles = json.loads(json_match.group(0))
                else:
                    log_messages.append(f"   ❌ Could not parse AI response: {ai_response[:200]}")
                    return []
            except Exception as e:
                log_messages.append(f"   ❌ JSON parse error: {str(e)}")
                return []
            
            log_messages.append(f"   ✓ AI suggested {len(suggested_titles)} songs")
            
            # Query database for exact title matches by this artist
            songs = []
            if suggested_titles:
                with db_conn.cursor(cursor_factory=DictCursor) as cur:
                    # Build query to find exact matches (title, artist)
                    # Use ILIKE for case-insensitive matching
                    or_conditions = []
                    params_list = []
                    for title in suggested_titles[:30]:  # Limit to avoid huge queries
                        or_conditions.append("(LOWER(title) = LOWER(%s) AND author = %s)")
                        params_list.extend([title, artist])
                    
                    if or_conditions:
                        where_clause = " OR ".join(or_conditions)
                        sql = f"""
                            SELECT DISTINCT item_id, title, author FROM public.score
                            WHERE {where_clause}
                            LIMIT %s
                        """
                        cur.execute(sql, params_list + [get_songs_count])
                        results = cur.fetchall()
                        songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author']} for r in results]
                        
                        log_messages.append(f"   ✓ Found {len(songs)} matching songs in database")
                        
                        # If we found fewer than expected, try fuzzy matching on artist name
                        if len(songs) < 20:
                            log_messages.append(f"   ⚠️ Only {len(songs)} exact matches, trying fuzzy artist matching...")
                            
                            # Find artist name variations in database
                            artist_simplified = ''.join(c.lower() for c in artist if c.isalnum() or c.isspace())
                            cur.execute("""
                                SELECT DISTINCT author FROM public.score
                                WHERE LOWER(REGEXP_REPLACE(author, '[^a-zA-Z0-9 ]', '', 'g')) LIKE %s
                                LIMIT 5
                            """, (f'%{artist_simplified}%',))
                            artist_variants = [r['author'] for r in cur.fetchall()]
                            
                            if artist_variants:
                                log_messages.append(f"   ✓ Found {len(artist_variants)} artist name variants: {artist_variants}")
                                
                                # Re-query with variants
                                or_conditions = []
                                params_list = []
                                for title in suggested_titles[:30]:
                                    for variant in artist_variants:
                                        or_conditions.append("(LOWER(title) = LOWER(%s) AND author = %s)")
                                        params_list.extend([title, variant])
                                
                                where_clause = " OR ".join(or_conditions)
                                sql = f"""
                                    SELECT DISTINCT item_id, title, author FROM public.score
                                    WHERE {where_clause}
                                    LIMIT %s
                                """
                                cur.execute(sql, params_list + [get_songs_count])
                                results = cur.fetchall()
                                songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author']} for r in results]
                                log_messages.append(f"   ✓ With fuzzy matching: {len(songs)} total songs found")
            
            return songs
            
        elif action_type == "artist_similarity_filtered":
            """
            Get songs from similar artists, then apply filters (genre, tempo, energy).
            This is a two-stage process: gather candidates, then filter them.
            """
            from tasks.artist_gmm_manager import find_similar_artists
            artist = params.get('artist')
            filters = params.get('filters', {})
            
            # Stage 1: Get similar artists (same as artist_similarity_api)
            similar_artists = find_similar_artists(artist, n=25)
            
            if not similar_artists:
                artist_simplified = ''.join(c.lower() for c in artist if c.isalnum() or c.isspace())
                log_messages.append(f"   ⚠️ Artist '{artist}' not found, trying fuzzy match: '{artist_simplified}'")
                
                with db_conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute("""
                        SELECT DISTINCT author FROM public.score
                        WHERE LOWER(REGEXP_REPLACE(author, '[^a-zA-Z0-9 ]', '', 'g')) LIKE %s
                        LIMIT 1
                    """, (f'%{artist_simplified}%',))
                    fuzzy_result = cur.fetchone()
                
                if fuzzy_result:
                    fuzzy_artist = fuzzy_result['author']
                    log_messages.append(f"   ✓ Found fuzzy match: '{fuzzy_artist}'")
                    similar_artists = find_similar_artists(fuzzy_artist, n=25)
                
            if not similar_artists:
                log_messages.append(f"   ❌ Artist '{artist}' not found")
                return []
            
            artist_names = [a['artist'] for a in similar_artists[:15]]
            log_messages.append(f"   ✓ Found {len(artist_names)} similar artists")
            
            # Stage 2: Query songs WITH filters applied
            with db_conn.cursor(cursor_factory=DictCursor) as cur:
                # Build WHERE clause with filters
                where_conditions = []
                params_list = []
                
                # Artist filter (always present)
                placeholders = ','.join(['%s'] * len(artist_names))
                where_conditions.append(f"author IN ({placeholders})")
                params_list.extend(artist_names)
                
                # Genre filter
                if 'genre' in filters and filters['genre']:
                    genre = filters['genre'].lower()
                    genre_mappings = {
                        'metal': ['metal', 'heavy metal', 'thrash metal'],
                        'rock': ['rock', 'hard rock', 'classic rock', 'alternative rock'],
                        'pop': ['pop', 'indie pop'],
                        'jazz': ['jazz'],
                        'blues': ['blues'],
                        'hip-hop': ['Hip-Hop', 'rap'],
                        'electronic': ['electronic', 'electronica'],
                    }
                    mood_filters = genre_mappings.get(genre, [genre])
                    genre_conditions = ' OR '.join(['mood_vector LIKE %s'] * len(mood_filters))
                    where_conditions.append(f"({genre_conditions})")
                    params_list.extend([f'%{mf}%' for mf in mood_filters])
                
                # Tempo filter
                if 'tempo' in filters and filters['tempo']:
                    tempo_level = filters['tempo'].lower()
                    tempo_conditions_map = {
                        'slow': 'tempo < 90',
                        'medium': 'tempo BETWEEN 90 AND 140',
                        'fast': 'tempo > 140',
                        'high': 'tempo > 140'
                    }
                    if tempo_level in tempo_conditions_map:
                        where_conditions.append(tempo_conditions_map[tempo_level])
                
                # Energy filter  
                if 'energy' in filters and filters['energy']:
                    energy_level = filters['energy'].lower()
                    energy_conditions_map = {
                        'low': 'energy < 0.05',
                        'medium': 'energy BETWEEN 0.05 AND 0.10',
                        'high': 'energy > 0.10'
                    }
                    if energy_level in energy_conditions_map:
                        where_conditions.append(energy_conditions_map[energy_level])
                
                where_clause = ' AND '.join(where_conditions)
                
                # Request more songs initially to compensate for filtering
                request_multiplier = 3 if filters else 1
                sql = f"""
                    SELECT DISTINCT item_id, title, author FROM (
                        SELECT item_id, title, author FROM public.score
                        WHERE {where_clause}
                        ORDER BY RANDOM()
                    ) AS randomized LIMIT %s
                """
                params_list.append(get_songs_count * request_multiplier)
                
                # Log the filters being applied
                filter_desc = []
                if 'genre' in filters and filters['genre']:
                    filter_desc.append(f"genre={filters['genre']}")
                if 'tempo' in filters and filters['tempo']:
                    filter_desc.append(f"tempo={filters['tempo']}")
                if 'energy' in filters and filters['energy']:
                    filter_desc.append(f"energy={filters['energy']}")
                
                if filter_desc:
                    log_messages.append(f"   🔍 Applying filters: {', '.join(filter_desc)}")
                    log_messages.append(f"   📝 SQL: SELECT ... WHERE author IN (...similar artists...) AND {' AND '.join([f for f in where_conditions[1:]])} ... LIMIT {get_songs_count * request_multiplier}")
                
                cur.execute(sql, params_list)
                results = cur.fetchall()
                
            songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author']} for r in results]
            log_messages.append(f"   ✓ Retrieved {len(songs)} filtered songs from similar artists")
            return songs
            
        elif action_type == "song_similarity_api":
            from tasks.voyager_manager import get_item_id_by_title_and_artist, find_nearest_neighbors_by_id
            song_title = params.get('song_title')
            song_artist = params.get('artist')
            
            target_item_id = get_item_id_by_title_and_artist(song_title, song_artist)
            if not target_item_id:
                log_messages.append(f"   ❌ Song '{song_title}' by '{song_artist}' not found")
                return []
            
            similar_songs = find_nearest_neighbors_by_id(target_item_id, n=get_songs_count)
            songs = [{"item_id": s['item_id'], "title": s['title'], "artist": s['author']} for s in similar_songs]
            log_messages.append(f"   ✓ Retrieved {len(songs)} similar songs")
            return songs
            
        elif action_type == "database_genre_query":
            genre = params.get('genre')
            tempo_level = params.get('tempo')  # Optional: slow/medium/fast
            energy_level = params.get('energy')  # Optional: low/medium/high
            
            # Handle None/null genre - query all songs
            if not genre:
                log_messages.append(f"   ⚠️ No genre specified, querying all songs randomly")
                with db_conn.cursor(cursor_factory=DictCursor) as cur:
                    sql = """
                        SELECT DISTINCT item_id, title, author FROM (
                            SELECT item_id, title, author FROM public.score
                            ORDER BY RANDOM()
                        ) AS randomized LIMIT %s
                    """
                    cur.execute(sql, [get_songs_count])
                    results = cur.fetchall()
                    
                songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author']} for r in results]
                log_messages.append(f"   ✓ Retrieved {len(songs)} random songs")
                return songs
            
            genre = genre.lower()
            
            genre_mappings = {
                'metal': ['metal', 'heavy metal', 'thrash metal', 'death metal', 'black metal'],
                'rock': ['rock', 'hard rock', 'classic rock', 'alternative rock', 'indie rock'],
                'pop': ['pop', 'indie pop'],
                'jazz': ['jazz'],
                'blues': ['blues'],
                'hip-hop': ['Hip-Hop', 'rap'],
                'rap': ['Hip-Hop', 'rap'],
                'electronic': ['electronic', 'electronica', 'electro'],
                'dance': ['dance', 'House'],
                'classical': ['classical'],
                'country': ['country'],
                'folk': ['folk'],
                'indie': ['indie', 'indie rock', 'indie pop'],
                'chill': ['chill', 'downtempo', 'ambient'],
                'ambient': ['ambient', 'chill'],
                'acoustic': ['acoustic', 'folk'],
                'downtempo': ['downtempo', 'chill']
            }
            
            mood_filters = genre_mappings.get(genre, [genre])
            
            # Build WHERE clause with optional tempo/energy filters
            where_conditions = []
            query_params = []
            
            # Genre filter (required)
            like_conditions = ' OR '.join(['mood_vector LIKE %s'] * len(mood_filters))
            where_conditions.append(f"({like_conditions})")
            query_params.extend([f'%{mf}%' for mf in mood_filters])
            
            # Optional tempo filter
            if tempo_level:
                tempo_level = tempo_level.lower()
                tempo_conditions = {
                    'slow': 'tempo < 90',
                    'medium': 'tempo BETWEEN 90 AND 140',
                    'fast': 'tempo > 140',
                    'high': 'tempo > 140'
                }
                if tempo_level in tempo_conditions:
                    where_conditions.append(tempo_conditions[tempo_level])
            
            # Optional energy filter
            if energy_level:
                energy_level = energy_level.lower()
                energy_conditions = {
                    'low': 'energy < 0.05',
                    'medium': 'energy BETWEEN 0.05 AND 0.10',
                    'high': 'energy > 0.10'
                }
                if energy_level in energy_conditions:
                    where_conditions.append(energy_conditions[energy_level])
            
            with db_conn.cursor(cursor_factory=DictCursor) as cur:
                where_clause = ' AND '.join(where_conditions)
                sql = f"""
                    SELECT DISTINCT item_id, title, author FROM (
                        SELECT item_id, title, author FROM public.score
                        WHERE {where_clause}
                        ORDER BY RANDOM()
                    ) AS randomized LIMIT %s
                """
                # Log the actual SQL query with interpolated values for debugging
                try:
                    # Safely interpolate parameters for logging (not for execution!)
                    log_params = [repr(p) for p in query_params]
                    log_where = where_clause
                    for param in log_params:
                        log_where = log_where.replace('%s', str(param), 1)
                    log_messages.append(f"   📝 SQL: SELECT ... WHERE {log_where} ... LIMIT {get_songs_count}")
                except:
                    # Fallback to placeholder version if interpolation fails
                    log_messages.append(f"   📝 SQL: SELECT ... WHERE {where_clause} ... LIMIT {get_songs_count}")
                
                cur.execute(sql, query_params + [get_songs_count])
                results = cur.fetchall()
                
            filter_desc = f"{genre}"
            if tempo_level:
                filter_desc += f" (tempo: {tempo_level})"
            if energy_level:
                filter_desc += f" (energy: {energy_level})"
            
            songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author']} for r in results]
            log_messages.append(f"   ✓ Retrieved {len(songs)} {filter_desc} songs")
            return songs
            
        elif action_type == "database_tempo_energy_query":
            tempo_level = params.get('tempo', 'medium').lower()  # slow/medium/fast/high
            energy_level = params.get('energy', 'medium').lower()  # low/medium/high
            
            # Map tempo levels to SQL conditions (tempo is in BPM, stored as REAL)
            # Based on actual data range: 40-180 BPM
            tempo_conditions = {
                'slow': 'tempo < 90',
                'medium': 'tempo BETWEEN 90 AND 140',
                'fast': 'tempo > 140',
                'high': 'tempo > 140'  # Synonym for 'fast'
            }
            
            # Map energy levels to SQL conditions (energy is normalized 0-0.15 based on actual data)
            energy_conditions = {
                'low': 'energy < 0.05',
                'medium': 'energy BETWEEN 0.05 AND 0.10',
                'high': 'energy > 0.10'
            }
            
            tempo_filter = tempo_conditions.get(tempo_level, 'tempo IS NOT NULL')
            energy_filter = energy_conditions.get(energy_level, 'energy IS NOT NULL')
            
            with db_conn.cursor(cursor_factory=DictCursor) as cur:
                sql = f"""
                    SELECT DISTINCT item_id, title, author FROM (
                        SELECT item_id, title, author FROM public.score
                        WHERE {tempo_filter} AND {energy_filter}
                        ORDER BY RANDOM()
                    ) AS randomized LIMIT %s
                """
                # Log the actual SQL query
                log_messages.append(f"   📝 SQL: SELECT item_id, title, author FROM public.score WHERE {tempo_filter} AND {energy_filter} ORDER BY RANDOM() LIMIT {get_songs_count}")
                cur.execute(sql, [get_songs_count])
                results = cur.fetchall()
                
            songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author']} for r in results]
            log_messages.append(f"   ✓ Retrieved {len(songs)} songs (tempo: {tempo_level}, energy: {energy_level})")
            return songs
            
        elif action_type == "vibe_match":
            """
            Creative/subjective query interpreter.
            Uses AI to reason about vibe/mood and translate to database filters.
            Examples: "songs that feel like rainy Sunday morning", "driving at night music"
            """
            vibe_description = params.get('vibe_description')
            
            if not ai_config:
                log_messages.append(f"   ❌ AI config not provided for vibe_match")
                return []
            
            log_messages.append(f"   🎨 Using AI to interpret vibe: '{vibe_description}'")
            
            # Ask AI to interpret the vibe and suggest filters
            vibe_prompt = f"""You are a music expert analyzing a subjective music request.

User's vibe request: "{vibe_description}"

Analyze this vibe and determine the musical characteristics that match it.
Consider:
- Mood/genre (e.g., chill, energetic, melancholic, upbeat)
- Tempo (slow/medium/fast)
- Energy level (low/medium/high)

Available genres in database: pop, rock, metal, jazz, blues, electronic, hip-hop, classical, folk, indie, ambient, chill, downtempo, acoustic

CRITICAL: Return ONLY a JSON object - NO markdown, NO explanations, NO extra text.
Format:
{{
  "genres": ["genre1", "genre2"],
  "tempo": "slow|medium|fast",
  "energy": "low|medium|high",
  "reasoning": "brief explanation of why these filters match the vibe"
}}

Example for "rainy Sunday morning":
{{
  "genres": ["chill", "ambient", "acoustic"],
  "tempo": "slow",
  "energy": "low",
  "reasoning": "Rainy Sunday mornings call for calm, introspective music with gentle tempo and low energy"
}}

Provide the JSON now:"""
            
            # Call AI to interpret vibe
            ai_response = call_ai_for_chat(
                provider=ai_config['provider'],
                prompt=vibe_prompt,
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
            
            if ai_response.startswith("Error:"):
                log_messages.append(f"   ❌ AI call failed: {ai_response}")
                return []
            
            # Parse AI response
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    vibe_analysis = json.loads(json_match.group(0))
                else:
                    log_messages.append(f"   ❌ Could not parse AI response: {ai_response[:200]}")
                    return []
            except Exception as e:
                log_messages.append(f"   ❌ JSON parse error: {str(e)}")
                return []
            
            genres = vibe_analysis.get('genres', [])
            tempo_level = vibe_analysis.get('tempo', 'medium')
            energy_level = vibe_analysis.get('energy', 'medium')
            reasoning = vibe_analysis.get('reasoning', '')
            
            log_messages.append(f"   🧠 AI Analysis: {reasoning}")
            log_messages.append(f"   🎵 Filters: genres={genres}, tempo={tempo_level}, energy={energy_level}")
            
            # Expanded genre mappings for vibe matching
            genre_mappings = {
                'metal': ['metal', 'heavy metal', 'thrash metal', 'death metal', 'black metal'],
                'rock': ['rock', 'hard rock', 'classic rock', 'alternative rock', 'indie rock'],
                'pop': ['pop', 'indie pop'],
                'jazz': ['jazz'],
                'blues': ['blues'],
                'hip-hop': ['Hip-Hop', 'rap'],
                'rap': ['Hip-Hop', 'rap'],
                'electronic': ['electronic', 'electronica', 'electro'],
                'dance': ['dance', 'House'],
                'classical': ['classical'],
                'country': ['country'],
                'folk': ['folk'],
                'indie': ['indie', 'indie rock', 'indie pop'],
                'chill': ['chill', 'downtempo', 'ambient'],
                'ambient': ['ambient', 'chill'],
                'acoustic': ['acoustic', 'folk'],
                'downtempo': ['downtempo', 'chill']
            }
            
            # Build WHERE clause
            where_conditions = []
            query_params = []
            
            # Genre filters (OR between genres)
            if genres:
                genre_likes = []
                for genre in genres:
                    mood_filters = genre_mappings.get(genre.lower(), [genre])
                    for mf in mood_filters:
                        genre_likes.append('mood_vector LIKE %s')
                        query_params.append(f'%{mf}%')
                
                if genre_likes:
                    where_conditions.append(f"({' OR '.join(genre_likes)})")
            
            # Tempo filter
            tempo_conditions = {
                'slow': 'tempo < 90',
                'medium': 'tempo BETWEEN 90 AND 140',
                'fast': 'tempo > 140',
                'high': 'tempo > 140'
            }
            if tempo_level and tempo_level.lower() in tempo_conditions:
                where_conditions.append(tempo_conditions[tempo_level.lower()])
            
            # Energy filter
            energy_conditions = {
                'low': 'energy < 0.05',
                'medium': 'energy BETWEEN 0.05 AND 0.10',
                'high': 'energy > 0.10'
            }
            if energy_level and energy_level.lower() in energy_conditions:
                where_conditions.append(energy_conditions[energy_level.lower()])
            
            # Execute query
            if not where_conditions:
                log_messages.append(f"   ⚠️ No filters generated, returning random songs")
                where_clause = "1=1"
            else:
                where_clause = ' AND '.join(where_conditions)
            
            with db_conn.cursor(cursor_factory=DictCursor) as cur:
                sql = f"""
                    SELECT DISTINCT item_id, title, author FROM (
                        SELECT item_id, title, author FROM public.score
                        WHERE {where_clause}
                        ORDER BY RANDOM()
                    ) AS randomized LIMIT %s
                """
                
                # Log interpolated SQL for debugging
                try:
                    log_params = [repr(p) for p in query_params]
                    log_where = where_clause
                    for param in log_params:
                        log_where = log_where.replace('%s', str(param), 1)
                    log_messages.append(f"   📝 SQL: SELECT ... WHERE {log_where} ... LIMIT {get_songs_count}")
                except:
                    log_messages.append(f"   📝 SQL: SELECT ... WHERE {where_clause} ... LIMIT {get_songs_count}")
                
                cur.execute(sql, query_params + [get_songs_count])
                results = cur.fetchall()
            
            songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author']} for r in results]
            log_messages.append(f"   ✓ Retrieved {len(songs)} songs matching vibe '{vibe_description}'")
            return songs
            
        else:
            log_messages.append(f"   ⚠️ Unknown action type: {action_type}")
            return []
            
    except Exception as e:
        logger.exception(f"Error executing action {action_type}")
        log_messages.append(f"   ❌ Error: {str(e)}")
        return []


def generate_final_sql_query(intent, strategy_info, found_artists, found_keywords, found_song_titles, target_count, ai_config, log_messages, user_requested_artists=None):
    """
    Generates the final SQL query based on the strategy and database exploration results.
    NOTE: This function is currently not used by the new action-based system but kept for reference/future use.
    """
    from ai import call_ai_for_chat
    
    log_messages.append("\n--- Generating Final SQL Query ---")
    
    strategy = strategy_info.get('strategy', 'fallback')
    
    # Determine if this is a temporal query (has specific song titles)
    is_temporal_query = len(found_song_titles) > 0
    
    # Check if this is an energy-based query (only add energy filter if explicitly mentioned)
    is_energy_query = any(keyword in intent.lower() for keyword in ['high energy', 'energetic', 'pumped', 'low energy', 'relaxing', 'calm'])
    
    # Separate user-requested artists from expanded ones using fuzzy matching
    priority_artists = []
    other_artists = []
    
    if user_requested_artists:
        for artist in found_artists:
            # Fuzzy match: remove all non-alphanumeric chars and compare lowercase
            artist_simplified = ''.join(c.lower() for c in artist if c.isalnum())
            is_requested = False
            
            for req in user_requested_artists:
                req_simplified = ''.join(c.lower() for c in req if c.isalnum())
                # Check if simplified versions match
                if req_simplified in artist_simplified or artist_simplified in req_simplified:
                    is_requested = True
                    break
            
            if is_requested:
                priority_artists.append(artist)
            else:
                other_artists.append(artist)
    else:
        other_artists = found_artists
    
    log_messages.append(f"Priority artists (user-requested): {', '.join(priority_artists) if priority_artists else 'None'}")
    log_messages.append(f"Additional similar artists: {', '.join(other_artists[:10]) if other_artists else 'None'}")
    
    if is_temporal_query:
        log_messages.append(f"**TEMPORAL QUERY DETECTED**: Found {len(found_song_titles)} specific song titles")
        log_messages.append(f"Song titles (sample): {', '.join([t[0] for t in found_song_titles[:5]])}")
    
    if is_energy_query:
        log_messages.append(f"**ENERGY QUERY DETECTED**: Will add energy filtering to SQL")
    
    # Build the final prompt for SQL generation
    final_prompt = f"""
You are a PostgreSQL expert. Generate a SQL query for the public.score table.

User Request Intent: {intent}
Strategy: {strategy}
Target Song Count: {target_count}
Is Energy Query: {is_energy_query}

{"**CRITICAL - TEMPORAL QUERY DETECTED**" if is_temporal_query else "**CRITICAL - GENRE/MOOD QUERY DETECTED**"}
{"This is a time-based query (e.g., 'recent years', '2020s'). The database has NO YEAR COLUMN." if is_temporal_query else "This is a GENRE/MOOD query (e.g., 'metal songs', 'chill music'). You MUST use mood_vector filtering."}
{"You MUST filter by specific SONG TITLES to achieve temporal filtering." if is_temporal_query else "You MUST use 'AND mood_vector LIKE %genre%' to filter by genre/mood."}

{"**CRITICAL - ENERGY QUERY DETECTED**" if is_energy_query else ""}
{"Add 'AND energy > 0.08' for high energy OR 'AND energy < 0.05' for low energy INSIDE the WHERE clause." if is_energy_query else ""}

**Database Findings:**
{"Specific Song Titles Found: " + str(len(found_song_titles)) if is_temporal_query else ""}
{("Sample titles: " + ', '.join([f"'{t[0]}' by {t[1]}" for t in found_song_titles[:10]])) if is_temporal_query else ""}

**Artist Prioritization:**
User EXPLICITLY requested these artists: {', '.join(user_requested_artists) if user_requested_artists else 'None - use all artists equally'}
Priority artists (MUST INCLUDE): {', '.join(priority_artists) if priority_artists else 'None'}
Additional similar artists (optional): {', '.join(other_artists[:15]) if other_artists else 'None'}

Available Moods/Genres: {', '.join(found_keywords) if found_keywords else 'None'}

Database Schema:
- item_id (text)
- title (text)
- author (text)
- tempo (numeric 40-200)
- mood_vector (text, format: 'pop:0.8,rock:0.3')
- other_features (text, format: 'danceable:0.7,party:0.6')
- energy (numeric 0-0.15, higher = more energetic)
- **NOTE: NO YEAR OR DATE COLUMN EXISTS**

**PROGRESSIVE FILTERING STRATEGY - CRITICAL:**
The goal is to return EXACTLY {target_count} songs. Start with minimal filters and add more ONLY if needed.

**Filtering Priority:**
- **TEMPORAL QUERIES** (has song titles): Artist filtering + Song title matching
- **GENRE/MOOD QUERIES** (no song titles): Artist filtering + mood_vector LIKE filter **REQUIRED**
- **ENERGY QUERIES**: Artist filtering + energy filter

**CRITICAL RULE FOR GENRE QUERIES:**
If this is NOT a temporal query (no song titles found), you **MUST** add mood_vector filtering:
- User asks for "metal" → Add: AND mood_vector LIKE '%metal%'
- User asks for "rock" → Add: AND mood_vector LIKE '%rock%'  
- User asks for "jazz" → Add: AND mood_vector LIKE '%jazz%'
- Use the found_keywords list to determine which genre to filter

**CRITICAL - ALL FILTERS MUST BE INSIDE THE SUBQUERY**:
- ✅ CORRECT: WHERE (author IN (...) AND mood_vector LIKE '%pop%') ORDER BY RANDOM()
- ❌ WRONG: WHERE author IN (...)) AS randomized WHERE mood_vector LIKE '%pop%'
- ❌ WRONG: WHERE author IN (...)) AS randomized WHERE energy > 0.08

**Query Building Rules:**
1. Return ONLY raw SQL (no markdown, no ```sql, no explanations)
2. **ABSOLUTELY CRITICAL**: ALL filters (mood_vector, energy, tempo) go INSIDE the subquery, BEFORE "ORDER BY RANDOM()"
3. **NEVER EVER** add WHERE clause after ") AS randomized" - the outer query is ALWAYS: SELECT DISTINCT item_id, title, author FROM (...) AS randomized LIMIT {target_count}
4. Use ONLY artists found in step 3 exploration (found_artists list)
5. For temporal queries: Use (title, author) IN tuples for exact matches
6. Use proper SQL escaping (single quotes as '')
7. **DOUBLE CHECK**: After writing your SQL, verify that mood_vector, energy, tempo filters are INSIDE the subquery, not outside

**FOR TEMPORAL QUERIES ("recent years", "2020s", "last decade"):**
- Use found song titles with artist names as exact (title, author) IN tuples
- Add OR author IN (...) for additional songs from those artists
- DO NOT add energy filters - this is a TIME-based query, not energy-based
- DO NOT add mood_vector filters - keep it simple to maximize results
- **IMPORTANT**: If you have many found artists (>10), use a simpler query:
  ```
  SELECT DISTINCT item_id, title, author FROM (
    SELECT item_id, title, author FROM public.score
    WHERE author IN ('Taylor Swift', 'Miley Cyrus', 'The Weeknd', ...)
    ORDER BY RANDOM()
  ) AS randomized LIMIT {target_count}
  ```

**FOR GENRE/MOOD QUERIES ("rock songs", "pop music", "jazz"):**
- Use author IN (...) for found artists
- Optionally add: AND mood_vector LIKE '%genre%' if that genre was found in database
- DO NOT add energy filters unless user explicitly asks for energy level

**FOR ENERGY QUERIES ("high energy", "relaxing music", "calm songs"):**
- Use author IN (...) for found artists  
- Add energy filter INSIDE WHERE clause: AND energy > 0.08 (high) or AND energy < 0.05 (low)
- Place filter BEFORE ORDER BY RANDOM()

Strategy Guidance: {strategy_info.get('sql_approach', 'Select diverse songs')}

**Example 1 - Temporal query ("top radio songs recent years"):**
SELECT DISTINCT item_id, title, author FROM (
  SELECT item_id, title, author FROM public.score
  WHERE (title, author) IN (
      ('Anti-Hero', 'Taylor Swift'),
      ('Flowers', 'Miley Cyrus'),
      ('As It Was', 'Harry Styles')
    )
    OR author IN ('Taylor Swift', 'Miley Cyrus', 'The Weeknd', 'Dua Lipa')
  ORDER BY RANDOM()
) AS randomized LIMIT {target_count}

**Example 2 - Genre query ("metal songs"):**
SELECT DISTINCT item_id, title, author FROM (
  SELECT item_id, title, author FROM public.score
  WHERE author IN ('Metallica', 'Iron Maiden', 'Slayer', 'Megadeth')
    AND (mood_vector LIKE '%metal%' OR mood_vector LIKE '%heavy metal%')
  ORDER BY RANDOM()
) AS randomized LIMIT {target_count}

**Example 3 - Energy query ("high energy workout music"):**
SELECT DISTINCT item_id, title, author FROM (
  SELECT item_id, title, author FROM public.score
  WHERE author IN ('David Guetta', 'Calvin Harris')
    AND energy > 0.08
  ORDER BY RANDOM()
) AS randomized LIMIT {target_count}

Generate the SQL query now:
"""
    
    raw_sql = call_ai_for_chat(
        provider=ai_config['provider'],
        prompt=final_prompt,
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
    
    if raw_sql.startswith("Error:"):
        log_messages.append(f"SQL Generation Error: {raw_sql}")
        return None, raw_sql
    
    log_messages.append(f"Generated SQL:\\n{raw_sql}")
    return raw_sql, None
