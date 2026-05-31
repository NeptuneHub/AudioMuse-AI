"""MCP tool implementations.

Synchronous bodies for every MCP tool dispatched by ``tasks.ai.tools``.
Each function performs DB queries and/or AI calls and returns a result dict
shaped like ``{"songs": [...], "message": "..."}``.

This module contains *only* tool bodies and a small shared helper. Prompts
live in ``tasks.ai.prompts``; AI calls go through ``tasks.ai.api``; DB
connections come from ``tasks.mcp_helper.get_db_connection``.

Dependencies on heavy submodules (``tasks.artist_gmm_manager``,
``tasks.clap_text_search``, ``tasks.voyager_manager``, ``tasks.song_alchemy``,
``tasks.ai.api``) are imported lazily inside each tool function. Each of those
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


def _reroute_other_feature_labels(genres, moods, other_features):
    """Move ``OTHER_FEATURE_LABELS`` mistakenly passed as ``genres`` or ``moods`` into
    ``other_features``.

    Small AI models often confuse the 6 CLAP labels (``danceable``, ``aggressive``,
    ``happy``, ``party``, ``relaxed``, ``sad``) with genres or mood-vector tags.
    These labels live in ``score.other_features``; only their canonical
    ``"label:score"`` form survives the substring search there.

    Returns (new_genres, new_moods, new_other_features, log_message_or_None).
    """
    from config import OTHER_FEATURE_LABELS
    other_set = {m.lower() for m in OTHER_FEATURE_LABELS}
    canonical_by_lower = {m.lower(): m for m in OTHER_FEATURE_LABELS}

    new_genres = list(genres or [])
    new_moods = list(moods or [])
    new_other = list(other_features or [])

    rerouted_from_genres = [g for g in new_genres if isinstance(g, str) and g.lower() in other_set]
    rerouted_from_moods = [m for m in new_moods if isinstance(m, str) and m.lower() in other_set]

    if not rerouted_from_genres and not rerouted_from_moods:
        return genres, moods, other_features, None

    new_genres = [g for g in new_genres if not (isinstance(g, str) and g.lower() in other_set)]
    new_moods = [m for m in new_moods if not (isinstance(m, str) and m.lower() in other_set)]

    existing_other_lower = {o.lower() for o in new_other if isinstance(o, str)}
    for label in rerouted_from_genres + rerouted_from_moods:
        canonical = canonical_by_lower[label.lower()]
        if canonical.lower() not in existing_other_lower:
            new_other.append(canonical)
            existing_other_lower.add(canonical.lower())

    parts = []
    if rerouted_from_genres:
        parts.append(f"from genres: {', '.join(rerouted_from_genres)}")
    if rerouted_from_moods:
        parts.append(f"from moods: {', '.join(rerouted_from_moods)}")
    msg = "\u26a0\ufe0f Rerouted to other_features (" + "; ".join(parts) + ")"
    return new_genres, new_moods, new_other, msg


def _reroute_mood_labels_from_genres(genres, moods):
    """Legacy shim: keeps the old (genres, moods) -> (genres, moods, msg) contract
    for callers that haven't migrated to the 3-list signature. Internally now
    routes OTHER_FEATURE_LABELS to the moods list as before (the new function
    handles other_features routing directly).
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


_FUZZY_MATCH_CUTOFF = 75
_FUZZY_CANDIDATE_POOL_LIMIT = 500
_FUZZY_PREFIX_LEN = 3


def _fetch_pool_features(item_ids: List[str]) -> Dict[str, Dict]:
    """Fetch the scoring columns for a set of item_ids via the PK fast-path.

    Returns ``{item_id: {mood_vector, other_features, tempo, energy, year,
    scale, key, rating, author, album}}``. Empty dict when given no ids. One
    indexed query (item_id = ANY) -- no text-column scan, no new index.
    """
    if not item_ids:
        return {}
    db_conn = get_db_connection()
    try:
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                """
                SELECT item_id, mood_vector, other_features, tempo, energy,
                       year, scale, key, rating, author, album
                FROM public.score
                WHERE item_id = ANY(%s)
                """,
                (list(item_ids),),
            )
            rows = cur.fetchall()
        out: Dict[str, Dict] = {}
        for r in rows:
            out[r['item_id']] = {
                'mood_vector': r.get('mood_vector'),
                'other_features': r.get('other_features'),
                'tempo': r.get('tempo'),
                'energy': r.get('energy'),
                'year': r.get('year'),
                'scale': r.get('scale'),
                'key': r.get('key'),
                'rating': r.get('rating'),
                'author': r.get('author'),
                'album': r.get('album'),
            }
        return out
    finally:
        db_conn.close()


def _normalize_for_match(s: Optional[str]) -> str:
    if not s:
        return ""
    return (s.replace(' ', '').replace('-', '').replace('‐', '')
            .replace('/', '').replace("'", '').lower())


def _fuzzy_match_author_title(
    db_conn,
    requested_author: str,
    requested_title: Optional[str] = None,
) -> Optional[Dict]:
    """Last-resort fuzzy DB lookup using rapidfuzz on a narrowed candidate pool.

    When the exact / ILIKE-normalized passes have failed, narrow the search to
    rows whose author OR title shares a short prefix with the requested values
    (cheap SQL filter), then rank candidates with token_set_ratio. Returns
    ``{"item_id", "title", "author", "album", "score", "matched_label"}`` on a
    confident match (score >= _FUZZY_MATCH_CUTOFF), or None.
    """
    from rapidfuzz import fuzz

    if not requested_author and not requested_title:
        return None

    author_prefix = _normalize_for_match(requested_author)[:_FUZZY_PREFIX_LEN]
    title_prefix = _normalize_for_match(requested_title)[:_FUZZY_PREFIX_LEN] if requested_title else ""

    prefix_conditions = []
    prefix_params: List = []
    if author_prefix:
        prefix_conditions.append(
            "REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(LOWER(author), ' ', ''), '-', ''), '‐', ''), '/', ''), '''', '') LIKE %s"
        )
        prefix_params.append(f"{author_prefix}%")
    if title_prefix:
        prefix_conditions.append(
            "REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(LOWER(title), ' ', ''), '-', ''), '‐', ''), '/', ''), '''', '') LIKE %s"
        )
        prefix_params.append(f"{title_prefix}%")
    if not prefix_conditions:
        return None

    where = " OR ".join(prefix_conditions)
    with db_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            f"""
            SELECT item_id, title, author, album
            FROM (SELECT DISTINCT item_id, title, author, album FROM public.score) AS dist
            WHERE {where}
            LIMIT %s
            """,
            prefix_params + [_FUZZY_CANDIDATE_POOL_LIMIT],
        )
        candidates = cur.fetchall()

    if not candidates:
        return None

    if requested_title:
        target = f"{requested_author} {requested_title}".strip().lower()
    else:
        target = (requested_author or "").strip().lower()

    best = None
    best_score = -1.0
    for c in candidates:
        cand_label = (
            f"{c['author']} {c['title']}".strip().lower()
            if requested_title
            else (c['author'] or "").strip().lower()
        )
        score = fuzz.token_set_ratio(target, cand_label)
        if score > best_score:
            best_score = score
            best = c

    if best is None or best_score < _FUZZY_MATCH_CUTOFF:
        return None

    return {
        "item_id": best['item_id'],
        "title": best['title'],
        "author": best['author'],
        "album": best.get('album', ''),
        "score": int(best_score),
        "matched_label": (
            f"{best['author']} - {best['title']}" if requested_title else best['author']
        ),
    }


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

            if not result:
                log_messages.append("ILIKE fallback miss, trying rapidfuzz fallback...")
                fz = _fuzzy_match_author_title(db_conn, artist, requested_title=None)
                if fz:
                    log_messages.append(
                        f"fuzzy matched '{artist}' -> '{fz['matched_label']}' (score {fz['score']})"
                    )
                    result = {'author': fz['author']}

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
    """CLAP audio text search -- PURE similarity, returns a large pool.

    No tempo/energy/genre filtering happens here: the chat planner routes any
    such filter through the ONE shared soft re-rank (``planner._rerank_pool``),
    so this tool just returns the CLAP-similarity-ranked pool. ``tempo_filter`` /
    ``energy_filter`` are accepted for signature compatibility but ignored.
    """
    from tasks.clap_text_search import search_by_text
    from config import CLAP_ENABLED

    log_messages = []
    try:
        if not CLAP_ENABLED:
            log_messages.append("CLAP text search is disabled")
            return {"songs": [], "message": "CLAP text search is not enabled. Please enable CLAP_ENABLED in config."}

        if not description:
            return {"songs": [], "message": "No description provided for text search"}

        limit = int(get_songs) if get_songs else 200
        log_messages.append(f"CLAP text search: '{description}' (pool up to {limit})")

        clap_results = search_by_text(description, limit=limit)
        if not clap_results:
            log_messages.append("No results from CLAP text search")
            return {"songs": [], "message": "\n".join(log_messages)}

        songs = [
            {"item_id": r['item_id'], "title": r['title'], "artist": r['author'], "album": r.get('album', '')}
            for r in clap_results
        ]
        log_messages.append(
            f"CLAP returned {len(songs)} songs (similarity order; any tempo/energy/genre "
            f"filter is applied downstream as a soft re-rank, not a cut)"
        )
        return {"songs": songs, "message": "\n".join(log_messages)}
    except Exception as e:
        log_messages.append(f"Error in text search: {str(e)}")
        log_messages.append(traceback.format_exc())
        return {"songs": [], "message": "\n".join(log_messages)}


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
                log_messages.append("ILIKE fallback miss, trying rapidfuzz fallback...")
                fz = _fuzzy_match_author_title(db_conn, song_artist, song_title)
                if fz:
                    log_messages.append(
                        f"fuzzy matched '{song_artist} - {song_title}' -> '{fz['matched_label']}' (score {fz['score']})"
                    )
                    seed = fz

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
    artist: Optional[str] = None,
    other_features: Optional[List[str]] = None,
    candidate_item_ids: Optional[List[str]] = None,
    voices: Optional[List[str]] = None,
    score_threshold: Optional[float] = None,
) -> Dict:
    get_songs = int(get_songs) if get_songs is not None else 100

    db_conn = get_db_connection()
    log_messages = []

    genres, moods, other_features, reroute_msg = _reroute_other_feature_labels(genres, moods, other_features)
    if reroute_msg:
        log_messages.append(reroute_msg)

    if moods:
        merged_other = list(other_features or [])
        for mood in moods:
            if mood not in merged_other:
                merged_other.append(mood)
        other_features = merged_other
        log_messages.append(
            f"moods {moods} matched against other_features only (never mood_vector)"
        )
        moods = None

    pool_order_index = None
    if candidate_item_ids:
        pool_order_index = {iid: i for i, iid in enumerate(candidate_item_ids)}

    try:
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            conditions = []
            params = []

            if candidate_item_ids:
                conditions.append("item_id = ANY(%s)")
                params.append(list(candidate_item_ids))

            effective_threshold = float(score_threshold) if score_threshold is not None else 0.5
            mood_vector_threshold = effective_threshold
            other_features_threshold = effective_threshold

            has_genre_filter = False
            if genres:
                genre_conditions = []
                for genre in genres:
                    genre_conditions.append(
                        "COALESCE(CAST(NULLIF(SUBSTRING(mood_vector FROM %s), '') AS NUMERIC), 0) >= %s"
                    )
                    params.append(f"(?i)(?:^|,)\\s*{re.escape(genre)}:(\\d+\\.?\\d*)")
                    params.append(mood_vector_threshold)
                conditions.append("(" + " OR ".join(genre_conditions) + ")")
                has_genre_filter = True

            has_voice_filter = False
            if voices:
                voice_conditions = []
                for voice in voices:
                    voice_conditions.append(
                        "COALESCE(CAST(NULLIF(SUBSTRING(mood_vector FROM %s), '') AS NUMERIC), 0) >= %s"
                    )
                    params.append(f"(?i)(?:^|,)\\s*{re.escape(voice)}:(\\d+\\.?\\d*)")
                    params.append(mood_vector_threshold)
                if len(voice_conditions) == 1:
                    conditions.append(voice_conditions[0])
                else:
                    conditions.append("(" + " OR ".join(voice_conditions) + ")")
                has_voice_filter = True

            has_other_filter = False
            other_confidence_threshold = other_features_threshold
            if other_features:
                other_conditions = []
                for of in other_features:
                    other_conditions.append(
                        "COALESCE(CAST(NULLIF(SUBSTRING(other_features FROM %s), '') AS NUMERIC), 0) >= %s"
                    )
                    params.append(f"(?i)(?:^|,)\\s*{re.escape(of)}:(\\d+\\.?\\d*)")
                    params.append(other_confidence_threshold)
                if len(other_conditions) == 1:
                    conditions.append(other_conditions[0])
                else:
                    conditions.append("(" + " OR ".join(other_conditions) + ")")
                has_other_filter = True

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

            order_clause = "ORDER BY RANDOM()" if pool_order_index is None else ""

            if has_genre_filter or has_voice_filter or has_other_filter:
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
                if has_voice_filter:
                    for voice in voices:
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
                        score_params.append(f"(?i)(?:^|,)\\s*{re.escape(voice)}:(\\d+\\.?\\d*)")
                if has_other_filter:
                    for of in other_features:
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
                        score_params.append(f"(?i)(?:^|,)\\s*{re.escape(of)}:(\\d+\\.?\\d*)")

                relevance_expr = " + ".join(score_parts)
                all_params = score_params + params

                inner_order = "ORDER BY relevance_score DESC, RANDOM()" if pool_order_index is None else "ORDER BY relevance_score DESC"
                query = f"""
                    SELECT DISTINCT item_id, title, author, album
                    FROM (
                        SELECT item_id, title, author, album,
                               ({relevance_expr}) AS relevance_score
                        FROM public.score
                        WHERE {where_clause}
                        {inner_order}
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
                        {order_clause}
                    ) AS randomized
                    LIMIT %s
                """
                cur.execute(query, params)

            results = cur.fetchall()

        songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author'], "album": r.get('album', '')} for r in results]

        if pool_order_index is not None:
            songs.sort(key=lambda s: pool_order_index.get(s['item_id'], 10 ** 9))

        filters = []
        if candidate_item_ids:
            filters.append(f"pool_size: {len(candidate_item_ids)}")
        if genres:
            filters.append(f"genres: {', '.join(genres)}")
        if voices:
            filters.append(f"voices: {', '.join(voices)}")
        if other_features:
            filters.append(f"moods: {', '.join(other_features)}")
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


def _lyrics_search_sync(query: str, get_songs: int) -> Dict:
    """Find songs whose lyrics semantically match a free-text query.

    Wraps ``tasks.lyrics_manager.search_by_text`` (the same backend that powers
    the ``/api/lyrics/search/text`` endpoint), maps ``author`` -> ``artist`` to
    match the shape other MCP tools return, and surfaces a clear message when
    lyrics are disabled or the voyager index is not yet loaded.
    """
    from config import LYRICS_ENABLED

    log_messages = []

    if not LYRICS_ENABLED:
        return {"songs": [], "message": "lyrics_search is disabled (set LYRICS_ENABLED=true in config to enable)"}

    text = (query or '').strip()
    if not text:
        return {"songs": [], "message": "lyrics_search requires a non-empty query"}

    try:
        from tasks.lyrics_manager import search_by_text
        log_messages.append(f"Lyrics search: '{text}'")
        results = search_by_text(text, limit=int(get_songs) if get_songs else 200, artist_cap=0, album_cap=0)
        if not results:
            log_messages.append("No lyrics matched (index empty, not loaded, or no semantic match)")
            return {"songs": [], "message": "\n".join(log_messages)}

        songs = [
            {
                "item_id": r['item_id'],
                "title": r.get('title', ''),
                "artist": r.get('author', ''),
                "album": r.get('album', ''),
            }
            for r in results
        ]
        log_messages.append(f"Lyrics search returned {len(songs)} songs")
        return {"songs": songs, "message": "\n".join(log_messages)}
    except Exception as e:
        logger.exception("lyrics_search failed")
        return {"songs": [], "message": f"lyrics_search error: {str(e)[:200]}"}


def _ai_brainstorm_sync(user_request: str, ai_config: Dict, get_songs: int) -> Dict:
    """Use AI to brainstorm songs from world knowledge for any free-form request."""
    from tasks.ai.api import generate_text as _ai_generate_text
    from tasks.ai.prompts import build_ai_brainstorm_prompt

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
