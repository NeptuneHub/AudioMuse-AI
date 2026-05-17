"""MCP tool implementations.

Synchronous bodies for every MCP tool dispatched by ``tasks.mcp_tools``.
Each function performs DB queries and/or AI calls and returns a result dict
shaped like ``{"songs": [...], "message": "..."}``.

This module contains *only* tool bodies and a small shared helper. Prompts
live in ``tasks.ai_prompts``; AI calls go through ``tasks.ai_api``; DB
connections come from ``tasks.mcp_helper.get_db_connection``.

Dependencies on heavy submodules (``tasks.artist_gmm_manager``,
``tasks.clap_text_search``, ``tasks.voyager_manager``, ``tasks.song_alchemy``,
``tasks.ai_api``) are imported lazily inside each tool function. Each of those
imports appears at most once across the module, so the lazy pattern is not
duplication -- it is the standard way to keep this module loadable in test
environments that stub those submodules via ``sys.modules``.
"""
import json
import logging
import re
import traceback
from typing import Dict, List, Optional

from psycopg2.extras import DictCursor

from tasks.mcp_helper import get_db_connection

logger = logging.getLogger(__name__)


def _reroute_mood_labels_from_genres(genres, moods):
    """Move any OTHER_FEATURE_LABELS values mistakenly passed as `genres` into `moods`.

    Small AI models often confuse moods (`danceable`, `aggressive`, `happy`, `party`,
    `relaxed`, `sad`) with genres. These labels live in `other_features`, never in
    `mood_vector`, so combining them as a genre filter with anything else will always
    return zero matches.

    Returns (new_genres, new_moods, log_message_or_None).
    """
    from config import OTHER_FEATURE_LABELS
    if not genres:
        return genres, moods, None
    mood_set = {m.lower() for m in OTHER_FEATURE_LABELS}
    rerouted = [g for g in genres if isinstance(g, str) and g.lower() in mood_set]
    if not rerouted:
        return genres, moods, None
    kept = [g for g in genres if not (isinstance(g, str) and g.lower() in mood_set)]
    new_moods = list(moods or [])
    existing_lower = {m.lower() for m in new_moods if isinstance(m, str)}
    for g in rerouted:
        canonical = g.lower()
        if canonical not in existing_lower:
            new_moods.append(canonical)
            existing_lower.add(canonical)
    msg = f"\u26a0\ufe0f Rerouted from genres to moods (not real genres): {', '.join(rerouted)}"
    return kept, new_moods, msg


def _artist_similarity_api_sync(artist: str, count: int, get_songs: int) -> Dict:
    """Find songs by an artist plus songs by similar artists (GMM-based)."""
    from tasks.artist_gmm_manager import find_similar_artists, reverse_artist_map

    db_conn = get_db_connection()
    log_messages = []

    try:
        # STEP 1: Fuzzy lookup in database to find correct artist name
        log_messages.append(f"Looking up artist in database: '{artist}'")

        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT author
                FROM public.score
                WHERE LOWER(author) = LOWER(%s)
                LIMIT 1
            """, (artist,))
            result = cur.fetchone()

            if not result:
                # Normalize: remove spaces, dashes, slashes to handle variations
                artist_normalized = artist.replace(' ', '').replace('-', '').replace('\u2010', '').replace('/', '').replace("'", '')

                log_messages.append(f"No exact match, trying fuzzy search for normalized: '{artist_normalized}'")
                cur.execute("""
                    SELECT author, LENGTH(author) as len
                    FROM (
                        SELECT DISTINCT author
                        FROM public.score
                        WHERE REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(author, ' ', ''), '-', ''), '\u2010', ''), '/', ''), '''', '') ILIKE %s
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
                artist = db_artist_name
            else:
                log_messages.append(f"Artist not found in database, using original: '{artist}'")

        # STEP 2: Now call similarity API with correct artist name
        log_messages.append(f"Calling similarity API for: '{artist}'")
        similar_artists = find_similar_artists(artist, n=25)

        if not similar_artists:
            log_messages.append("Similarity API returned no results, trying fallback strategies")

            if reverse_artist_map:
                artist_lower = artist.lower()
                matches = [
                    gmm_artist for gmm_artist in reverse_artist_map.keys()
                    if artist_lower in gmm_artist.lower()
                ]
                if matches:
                    best_match = min(matches, key=len)
                    log_messages.append(f"Found fuzzy match in GMM index: '{best_match}' (from '{artist}')")
                    similar_artists = find_similar_artists(best_match, n=25)

            if not similar_artists:
                clean_artist = re.sub(r'[^\w\s]', '', artist).strip()
                if clean_artist != artist:
                    log_messages.append(f"Trying without special chars: '{clean_artist}'")
                    similar_artists = find_similar_artists(clean_artist, n=25)

        if not similar_artists:
            return {"songs": [], "message": "\n".join(log_messages) + f"\nNo similar artists found for '{artist}'"}

        artist_names = [a['artist'] for a in similar_artists[:count]]
        log_messages.append(f"Found {len(artist_names)} similar artists")

        all_artist_names = [artist] + artist_names
        log_messages.append(f"Searching songs from {len(all_artist_names)} artists (original + similar)")

        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            placeholders = ','.join(['%s'] * len(all_artist_names))
            query = f"""
                SELECT item_id, title, author, album
                FROM (
                    SELECT DISTINCT item_id, title, author, album
                    FROM public.score
                    WHERE author IN ({placeholders})
                ) AS distinct_songs
                ORDER BY RANDOM()
                LIMIT %s
            """
            cur.execute(query, all_artist_names + [get_songs])
            results = cur.fetchall()

        songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author'], "album": r.get('album', '')} for r in results]
        log_messages.append(f"Retrieved {len(songs)} songs from original + similar artists")

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


def _text_search_sync(description: str, tempo_filter: Optional[str], energy_filter: Optional[str], get_songs: int) -> Dict:
    """CLAP text search with optional hybrid tempo/energy filtering."""
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

        clap_results = search_by_text(description, limit=100)

        if not clap_results:
            log_messages.append("No results from CLAP text search")
            return {"songs": [], "message": "\n".join(log_messages)}

        log_messages.append(f"CLAP returned {len(clap_results)} songs")

        if tempo_filter or energy_filter:
            log_messages.append(f"Applying hybrid filters (tempo: {tempo_filter}, energy: {energy_filter})")

            item_ids = [r['item_id'] for r in clap_results]

            tempo_ranges = {
                'slow': (0, 90),
                'medium': (90, 140),
                'fast': (140, 300)
            }
            energy_ranges = {
                'low': (0, 0.05),
                'medium': (0.05, 0.10),
                'high': (0.10, 1.0)
            }

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

            with db_conn.cursor(cursor_factory=DictCursor) as cur:
                placeholders = ','.join(['%s'] * len(item_ids))
                where_clause = ' AND '.join(filter_conditions)

                sql = f"""
                    SELECT item_id, title, author, album
                    FROM public.score
                    WHERE item_id IN ({placeholders})
                    AND {where_clause}
                """

                cur.execute(sql, item_ids + query_params)
                filtered_results = cur.fetchall()

            album_lookup = {r['item_id']: r.get('album', '') for r in filtered_results}
            filtered_item_ids = {r['item_id'] for r in filtered_results}
            songs = [
                {"item_id": r['item_id'], "title": r['title'], "artist": r['author'], "album": album_lookup.get(r['item_id'], '')}
                for r in clap_results
                if r['item_id'] in filtered_item_ids
            ]

            log_messages.append(f"Filtered to {len(songs)} songs matching tempo/energy criteria")
        else:
            songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author'], "album": r.get('album', '')} for r in clap_results]
            log_messages.append(f"Retrieved {len(songs)} songs from CLAP")

        # --- Genre keyword filter: remove off-genre CLAP results ---
        try:
            _GENRE_KEYWORDS = {
                'rock', 'metal', 'pop', 'jazz', 'blues', 'country', 'folk', 'punk',
                'hip-hop', 'rap', 'electronic', 'dance', 'reggae', 'soul', 'funk',
                'r&b', 'classical', 'indie', 'alternative', 'hard rock', 'heavy metal',
                'grunge', 'ska', 'latin', 'techno', 'house', 'ambient', 'new wave',
                'post-punk', 'shoegaze',
            }
            desc_lower = description.lower()
            matched_genres = [g for g in _GENRE_KEYWORDS if g in desc_lower]

            if matched_genres and songs:
                song_ids = [s['item_id'] for s in songs]
                with db_conn.cursor(cursor_factory=DictCursor) as cur:
                    ph = ','.join(['%s'] * len(song_ids))
                    genre_conditions = []
                    genre_params = []
                    for g in matched_genres:
                        genre_conditions.append("mood_vector ~* %s")
                        genre_params.append(f"(^|,)\\s*{re.escape(g)}:")
                    genre_where = " OR ".join(genre_conditions)
                    cur.execute(f"""
                        SELECT item_id FROM public.score
                        WHERE item_id IN ({ph})
                        AND ({genre_where})
                    """, song_ids + genre_params)
                    matching_ids = {r['item_id'] for r in cur.fetchall()}

                filtered = [s for s in songs if s['item_id'] in matching_ids]
                if len(filtered) >= len(songs) * 0.4:
                    removed = len(songs) - len(filtered)
                    if removed > 0:
                        log_messages.append(f"Genre keyword filter: removed {removed} off-genre songs (keywords: {', '.join(matched_genres[:3])})")
                    songs = filtered
        except Exception as e:
            logger.warning(f"CLAP genre filter failed (non-fatal): {e}")

        return {"songs": songs[:get_songs], "message": "\n".join(log_messages)}
    except Exception as e:
        log_messages.append(f"Error in text search: {str(e)}")
        log_messages.append(traceback.format_exc())
        return {"songs": [], "message": "\n".join(log_messages)}
    finally:
        db_conn.close()


def _song_similarity_api_sync(song_title: str, song_artist: str, get_songs: int) -> Dict:
    """Find similar songs to a (title, artist) seed via Voyager nearest neighbors."""
    from tasks.voyager_manager import find_nearest_neighbors_by_id

    get_songs = int(get_songs) if get_songs is not None else 100

    db_conn = get_db_connection()
    log_messages = []

    try:
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
            cur.execute("""
                SELECT item_id, title, author, album FROM public.score
                WHERE LOWER(title) = LOWER(%s) AND LOWER(author) = LOWER(%s)
                LIMIT 1
            """, (song_title, song_artist))
            seed = cur.fetchone()

            if not seed:
                log_messages.append("No exact match, trying fuzzy search...")
                title_normalized = song_title.replace(' ', '').replace('-', '').replace('\u2010', '').replace('/', '').replace("'", '')
                artist_normalized = song_artist.replace(' ', '').replace('-', '').replace('\u2010', '').replace('/', '').replace("'", '')

                cur.execute("""
                    SELECT item_id, title, author, album
                    FROM public.score
                    WHERE REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(title, ' ', ''), '-', ''), '\u2010', ''), '/', ''), '''', '') ILIKE %s
                      AND REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(author, ' ', ''), '-', ''), '\u2010', ''), '/', ''), '''', '') ILIKE %s
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

        similar_results = find_nearest_neighbors_by_id(seed_id, n=get_songs + 1, eliminate_duplicates=False, radius_similarity=False)

        similar_ids = [r['item_id'] for r in similar_results if r['item_id'] != seed_id][:get_songs]

        if not similar_ids:
            songs = []
        else:
            id_to_order = {item_id: i for i, item_id in enumerate(similar_ids)}

            with db_conn.cursor(cursor_factory=DictCursor) as cur:
                placeholders = ','.join(['%s'] * len(similar_ids))
                cur.execute(f"""
                    SELECT item_id, title, author, album
                    FROM public.score
                    WHERE item_id IN ({placeholders})
                """, similar_ids)
                results = cur.fetchall()

            sorted_results = sorted(results, key=lambda r: id_to_order.get(r['item_id'], 999999))
            songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author'], "album": r.get('album', '')} for r in sorted_results]

        log_messages.append(f"Retrieved {len(songs)} similar songs")

        return {"songs": songs, "message": "\n".join(log_messages)}
    finally:
        db_conn.close()


def _song_alchemy_sync(add_items: List[Dict], subtract_items: Optional[List[Dict]] = None, get_songs: int = 100) -> Dict:
    """Blend or subtract musical vibes via vector arithmetic over items."""
    from tasks.song_alchemy import song_alchemy

    log_messages = []

    try:
        log_messages.append(f"Song Alchemy: ADD {len(add_items)} items" + (f", SUBTRACT {len(subtract_items)} items" if subtract_items else ""))

        for item in add_items:
            item_type = item.get('type', 'unknown')
            item_id = item.get('id', 'unknown')
            log_messages.append(f"  + ADD {item_type}: {item_id}")

        if subtract_items:
            for item in subtract_items:
                item_type = item.get('type', 'unknown')
                item_id = item.get('id', 'unknown')
                log_messages.append(f"  - SUBTRACT {item_type}: {item_id}")

        result = song_alchemy(
            add_items=add_items,
            subtract_items=subtract_items,
            n_results=get_songs
        )

        raw_songs = result.get('results', [])
        songs = [{"item_id": s['item_id'], "title": s['title'], "artist": s.get('author', s.get('artist', '')), "album": s.get('album', '')} for s in raw_songs]
        log_messages.append(f"Retrieved {len(songs)} songs from alchemy")

        # --- Genre-coherence filter: remove off-genre results ---
        try:
            if songs and add_items:
                db_conn_gc = get_db_connection()
                try:
                    with db_conn_gc.cursor(cursor_factory=DictCursor) as cur:
                        seed_ids = []
                        for item in add_items:
                            item_type = item.get('type', 'artist')
                            item_id_val = item.get('id', '')
                            if item_type == 'artist':
                                cur.execute(
                                    "SELECT item_id FROM public.score WHERE LOWER(author) = LOWER(%s) LIMIT 10",
                                    (item_id_val,)
                                )
                                seed_ids.extend([r['item_id'] for r in cur.fetchall()])
                            elif item_type == 'song' and ' by ' in item_id_val:
                                parts = item_id_val.rsplit(' by ', 1)
                                cur.execute(
                                    "SELECT item_id FROM public.score WHERE LOWER(title) = LOWER(%s) AND LOWER(author) = LOWER(%s) LIMIT 1",
                                    (parts[0].strip(), parts[1].strip())
                                )
                                row = cur.fetchone()
                                if row:
                                    seed_ids.append(row['item_id'])

                        if seed_ids:
                            ph = ','.join(['%s'] * len(seed_ids))
                            cur.execute(f"""
                                SELECT unnest(string_to_array(mood_vector, ',')) AS tag
                                FROM public.score
                                WHERE item_id IN ({ph})
                                AND mood_vector IS NOT NULL AND mood_vector != ''
                            """, seed_ids)
                            seed_genre_scores = {}
                            for r in cur:
                                tag = r['tag'].strip()
                                if ':' in tag:
                                    name, score_str = tag.split(':', 1)
                                    name = name.strip()
                                    try:
                                        seed_genre_scores[name] = seed_genre_scores.get(name, 0) + float(score_str)
                                    except ValueError:
                                        pass
                            top_seed_genres = sorted(seed_genre_scores, key=seed_genre_scores.get, reverse=True)[:3]

                            if top_seed_genres:
                                result_ids = [s['item_id'] for s in songs]
                                ph2 = ','.join(['%s'] * len(result_ids))
                                cur.execute(f"""
                                    SELECT item_id, mood_vector
                                    FROM public.score
                                    WHERE item_id IN ({ph2})
                                """, result_ids)
                                result_genres = {}
                                for r in cur:
                                    mv = r['mood_vector'] or ''
                                    genres_found = {}
                                    for tag in mv.split(','):
                                        tag = tag.strip()
                                        if ':' in tag:
                                            gname, gscore = tag.split(':', 1)
                                            try:
                                                genres_found[gname.strip()] = float(gscore)
                                            except ValueError:
                                                pass
                                    result_genres[r['item_id']] = genres_found

                                filtered = []
                                for s in songs:
                                    sid = s['item_id']
                                    g = result_genres.get(sid, {})
                                    if not g:
                                        filtered.append(s)
                                    elif any(g.get(tg, 0) >= 0.2 for tg in top_seed_genres):
                                        filtered.append(s)

                                if len(filtered) >= len(songs) * 0.4:
                                    removed = len(songs) - len(filtered)
                                    if removed > 0:
                                        log_messages.append(f"Genre filter: removed {removed} off-genre songs (seed genres: {', '.join(top_seed_genres[:3])})")
                                    songs = filtered
                finally:
                    db_conn_gc.close()
        except Exception as e:
            logger.warning(f"Alchemy genre filter failed (non-fatal): {e}")

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
    min_rating: Optional[int] = None,
    album: Optional[str] = None,
    artist: Optional[str] = None
) -> Dict:
    """Flexible database search across genres, moods, tempo/energy, year, rating, etc.

    - Genre matching uses regex with confidence threshold to avoid substring false positives.
    - Results are ordered by genre/mood confidence score sum (relevance) when filters apply.
    """
    get_songs = int(get_songs) if get_songs is not None else 100

    db_conn = get_db_connection()
    log_messages = []

    # Reroute mood labels mistakenly passed as genres.
    genres, moods, reroute_msg = _reroute_mood_labels_from_genres(genres, moods)
    if reroute_msg:
        log_messages.append(reroute_msg)

    try:
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            conditions = []
            params = []

            has_genre_filter = False
            genre_confidence_threshold = 0.55
            if genres:
                genre_conditions = []
                for genre in genres:
                    genre_conditions.append(
                        "COALESCE(CAST(NULLIF(SUBSTRING(mood_vector FROM %s), '') AS NUMERIC), 0) >= %s"
                    )
                    params.append(f"(?i)(?:^|,)\\s*{re.escape(genre)}:(\\d+\\.?\\d*)")
                    params.append(genre_confidence_threshold)
                conditions.append("(" + " OR ".join(genre_conditions) + ")")
                has_genre_filter = True

            has_mood_filter = False
            mood_confidence_threshold = 0.6
            if moods:
                mood_conditions = []
                for mood in moods:
                    mood_conditions.append(
                        "COALESCE(CAST(NULLIF(SUBSTRING(other_features FROM %s), '') AS NUMERIC), 0) >= %s"
                    )
                    params.append(f"(?i)(?:^|,)\\s*{re.escape(mood)}:(\\d+\\.?\\d*)")
                    params.append(mood_confidence_threshold)
                if len(mood_conditions) == 1:
                    conditions.append(mood_conditions[0])
                else:
                    conditions.append("(" + " OR ".join(mood_conditions) + ")")
                has_mood_filter = True

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

            if key:
                conditions.append("key = %s")
                params.append(key.upper())

            if scale:
                conditions.append("LOWER(scale) = LOWER(%s)")
                params.append(scale)

            if year_min is not None:
                conditions.append("year >= %s")
                params.append(int(year_min))
            if year_max is not None:
                conditions.append("year <= %s")
                params.append(int(year_max))

            if min_rating is not None:
                conditions.append("rating >= %s")
                params.append(int(min_rating))

            if album:
                conditions.append("LOWER(album) LIKE LOWER(%s)")
                params.append(f"%{album}%")

            if artist:
                conditions.append("""
                    LOWER(REPLACE(REPLACE(REPLACE(REPLACE(author, '-', ''), '\u2010', ''), '/', ''), '''', ''))
                    =
                    LOWER(REPLACE(REPLACE(REPLACE(REPLACE(%s, '-', ''), '\u2010', ''), '/', ''), '''', ''))
                """)
                params.append(artist)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            params.append(get_songs)

            if has_genre_filter or has_mood_filter:
                score_parts = []
                score_params = []
                if has_genre_filter:
                    for genre in genres:
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
                        score_params.append(f"(?i)(?:^|,)\\s*{re.escape(genre)}:(\\d+\\.?\\d*)")
                if has_mood_filter:
                    for mood in moods:
                        score_parts.append("""
                            COALESCE(
                                CAST(
                                    NULLIF(
                                        SUBSTRING(other_features FROM %s),
                                        ''
                                    ) AS NUMERIC
                                ),
                                0
                            )
                        """)
                        score_params.append(f"(?i)(?:^|,)\\s*{re.escape(mood)}:(\\d+\\.?\\d*)")

                relevance_expr = " + ".join(score_parts)
                all_params = score_params + params

                query = f"""
                    SELECT DISTINCT item_id, title, author, album
                    FROM (
                        SELECT item_id, title, author, album,
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
                    SELECT DISTINCT item_id, title, author, album
                    FROM (
                        SELECT item_id, title, author, album
                        FROM public.score
                        WHERE {where_clause}
                        ORDER BY RANDOM()
                    ) AS randomized
                    LIMIT %s
                """
                cur.execute(query, params)

            results = cur.fetchall()

        songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author'], "album": r.get('album', '')} for r in results]

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
        if album:
            filters.append(f"album: {album}")
        if artist:
            filters.append(f"artist: {artist}")

        log_messages.append(f"Found {len(songs)} songs matching {', '.join(filters) if filters else 'all criteria'}")

        return {"songs": songs, "message": "\n".join(log_messages)}
    finally:
        db_conn.close()


def _ai_brainstorm_sync(user_request: str, ai_config: Dict, get_songs: int) -> Dict:
    """Use AI to brainstorm songs from world knowledge for any free-form request."""
    from tasks.ai_api import generate_text as _ai_generate_text
    from tasks.ai_prompts import build_ai_brainstorm_prompt

    get_songs = int(get_songs) if get_songs is not None else 100

    db_conn = get_db_connection()
    log_messages = []

    try:
        log_messages.append(f"Using AI knowledge to brainstorm songs for: {user_request}")

        prompt = build_ai_brainstorm_prompt(user_request)

        raw_response = _ai_generate_text(prompt, ai_config, skip_delay=True)

        if raw_response.startswith("Error:"):
            return {"songs": [], "message": f"AI Error: {raw_response}"}

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

            cleaned = cleaned.replace("'\'", '"')

            song_list = json.loads(cleaned)

            if not isinstance(song_list, list):
                raise ValueError("Response is not a JSON array")

            log_messages.append(f"AI suggested {len(song_list)} songs")
        except Exception as e:
            log_messages.append(f"Failed to parse AI response: {str(e)}")
            log_messages.append(f"Raw AI response (first 500 chars): {raw_response[:500]}")
            return {"songs": [], "message": "\n".join(log_messages)}

        found_songs = []
        seen_ids = set()

        def _normalize(s: str) -> str:
            return re.sub(r"[\s\-\u2010\u2011\u2012\u2013\u2014/'\".,!?()]", '', s).lower()

        def _escape_like(s: str) -> str:
            return s.replace('%', r'\%').replace('_', r'\_')

        valid_items = [(item.get('title', ''), item.get('artist', ''))
                       for item in song_list
                       if item.get('title') and item.get('artist')]

        stage2_items = []
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            if valid_items:
                values_params = []
                for title, artist in valid_items:
                    values_params.extend([title.lower(), artist.lower()])
                values_clause = ', '.join(['(%s, %s)'] * len(valid_items))
                cur.execute(f"""
                    SELECT item_id, title, author, album
                    FROM public.score
                    WHERE (LOWER(title), LOWER(author)) IN (VALUES {values_clause})
                """, values_params)
                exact_rows = cur.fetchall()

                exact_match_map = {}
                for row in exact_rows:
                    key = (row['title'].lower(), row['author'].lower())
                    if key not in exact_match_map:
                        exact_match_map[key] = row

                stage2_items = []
                for title, artist in valid_items:
                    key = (title.lower(), artist.lower())
                    result = exact_match_map.get(key)
                    if result and result['item_id'] not in seen_ids:
                        found_songs.append({
                            "item_id": result['item_id'],
                            "title": result['title'],
                            "artist": result['author'],
                            "album": result.get('album', '')
                        })
                        seen_ids.add(result['item_id'])
                    elif not result:
                        stage2_items.append((title, artist))

            if stage2_items:
                or_conditions = []
                fuzzy_params = []
                fuzzy_lookup_order = []
                for title, artist in stage2_items:
                    title_norm = _normalize(title)
                    artist_norm = _normalize(artist)
                    if title_norm and artist_norm:
                        or_conditions.append("""(
                            LOWER(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(title, ' ', ''), '-', ''), '''', ''), '.', ''), ',', ''))
                                LIKE LOWER(%s)
                            AND LOWER(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(author, ' ', ''), '-', ''), '''', ''), '.', ''), ',', ''))
                                LIKE LOWER(%s)
                        )""")
                        fuzzy_params.extend([f"%{_escape_like(title_norm)}%", f"%{_escape_like(artist_norm)}%"])
                        fuzzy_lookup_order.append((title_norm, artist_norm))

                if or_conditions:
                    where_clause = ' OR '.join(or_conditions)
                    cur.execute(f"""
                        SELECT item_id, title, author, album
                        FROM public.score
                        WHERE {where_clause}
                        ORDER BY LENGTH(title) + LENGTH(author)
                    """, fuzzy_params)
                    fuzzy_rows = cur.fetchall()

                    for row in fuzzy_rows:
                        if row['item_id'] not in seen_ids:
                            db_title_norm = _normalize(row['title'])
                            db_artist_norm = _normalize(row['author'])
                            for t_norm, a_norm in fuzzy_lookup_order:
                                if t_norm in db_title_norm and a_norm in db_artist_norm:
                                    found_songs.append({
                                        "item_id": row['item_id'],
                                        "title": row['title'],
                                        "artist": row['author'],
                                        "album": row.get('album', '')
                                    })
                                    seen_ids.add(row['item_id'])
                                    fuzzy_lookup_order.remove((t_norm, a_norm))
                                    break

        found_songs = found_songs[:get_songs]

        log_messages.append(f"Found {len(found_songs)} songs in database (from {len(song_list)} AI suggestions)")

        return {"songs": found_songs, "ai_suggestions": len(song_list), "message": "\n".join(log_messages)}
    finally:
        db_conn.close()


def _lyrics_search_sync(description: str, get_songs: int = 200) -> dict:
    """
    Search songs by lyrics query using the E5 embedding + Voyager index.
    This mirrors the frontend "search by text" lyrics feature.
    """
    from tasks.lyrics_manager import search_by_text

    try:
        results = search_by_text(description, limit=get_songs)
        return {"songs": results, "message": f"Lyrics search: '{description}' → {len(results)} songs"}
    except Exception as e:
        return {"error": f"Lyrics search failed: {str(e)}"}
