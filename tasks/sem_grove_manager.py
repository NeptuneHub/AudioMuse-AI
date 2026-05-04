"""
Semantic & Groove (SemGrove) Manager

Builds and queries a merged lyrics + audio Voyager index for song-by-song
similarity that respects both lyrical meaning and acoustic genre simultaneously.

Architecture
------------
* Fetches lyrics embeddings (e5-base-v2, 768-dim) from ``lyrics_embedding``
* Fetches audio embeddings (MusicNN, N-dim) from ``embedding``
* For every song present in **both** tables:
    1. L2-normalise each vector to unit length
    2. Whiten per-dimension (divide by empirical std across the library)
    3. Re-normalise to unit length after whitening
    4. Scale by w_L = sqrt(WEIGHT_LYRICS) and w_A = sqrt(WEIGHT_AUDIO)
    5. Concatenate → merged vector of dimension (lyrics_dim + audio_dim)
* Builds a Voyager Cosine index over the merged vectors
* Persists:
    - The index binary in ``lyrics_index_data`` (index_name='sem_grove_index')
    - Whitening stats in the same table  (index_name='sem_grove_whitening')
* At query time the seed song's pre-stored merged vector is fetched directly
  from the index (no re-computation needed).

Cosine similarity of two merged vectors equals:
    cos ≈ W_L² · cos(l₁,l₂) + W_A² · cos(a₁,a₂)
so the weight split is determined by the squared scale factors baked at
build time.  Default: 75 % lyrics / 25 % audio.
"""

import json
import logging
import math
import os
import re
import sys
import tempfile
from typing import Dict, List, Optional

import numpy as np
import psycopg2

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight configuration — read from config (which reads from DB / env).
# Values are the squared scale factors: merged_cos = W²_L·cos(l)+W²_A·cos(a)
# ---------------------------------------------------------------------------
def _get_weights():
    """Read current weights from config (supports hot-reload via setup wizard)."""
    wl = max(0.0, float(getattr(config, 'SEM_GROVE_WEIGHT_LYRICS', 0.74)))
    wa = max(0.0, float(getattr(config, 'SEM_GROVE_WEIGHT_AUDIO',  0.26)))
    return math.sqrt(wl), math.sqrt(wa)

# Module-level defaults used at build time
W_LYRICS, W_AUDIO = _get_weights()

SEM_GROVE_INDEX_NAME     = "sem_grove_index"
SEM_GROVE_WHITENING_NAME = "sem_grove_whitening"

# ---------------------------------------------------------------------------
# In-memory cache (module-level singleton)
# ---------------------------------------------------------------------------
_SEM_GROVE_CACHE: Dict = {
    "index":          None,   # voyager.Index
    "id_map":         None,   # {voyager_int_id: item_id_str}
    "reverse_id_map": None,   # {item_id_str: voyager_int_id}
    "std_lyrics":     None,   # np.ndarray (lyrics_dim,)
    "std_audio":      None,   # np.ndarray (audio_dim,)
    "lyrics_dim":     None,
    "audio_dim":      None,
    "w_lyrics":       W_LYRICS,
    "w_audio":        W_AUDIO,
    "loaded":         False,
    "song_count":     0,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_bytes(data: bytes, part_size: int) -> List[bytes]:
    return [data[i:i + part_size] for i in range(0, len(data), part_size)]


def _make_merged_vector(
    l_vec: np.ndarray,
    a_vec: np.ndarray,
    std_lyrics: np.ndarray,
    std_audio: np.ndarray,
    w_l: float,
    w_a: float,
) -> Optional[np.ndarray]:
    """Whiten, normalise, weight, and concatenate one lyrics+audio pair."""
    try:
        # --- Lyrics half ---
        l = l_vec.astype(np.float32, copy=False)
        n = np.linalg.norm(l)
        if n == 0:
            return None
        l = l / n
        l = l / (std_lyrics + 1e-8)
        n2 = np.linalg.norm(l)
        if n2 < 1e-8:
            return None
        l = l / n2

        # --- Audio half ---
        a = a_vec.astype(np.float32, copy=False)
        n = np.linalg.norm(a)
        if n == 0:
            return None
        a = a / n
        a = a / (std_audio + 1e-8)
        n2 = np.linalg.norm(a)
        if n2 < 1e-8:
            return None
        a = a / n2

        return np.concatenate([w_l * l, w_a * a]).astype(np.float32)
    except Exception as exc:
        logger.debug("_make_merged_vector: %s", exc)
        return None


def _fetch_metadata(item_ids: List[str]) -> Dict[str, Dict]:
    if not item_ids:
        return {}
    from app_helper import get_score_data_by_ids
    try:
        rows = get_score_data_by_ids(item_ids)
        return {
            r["item_id"]: {
                "title":  r.get("title",  "") or "",
                "author": r.get("author", "") or "",
            }
            for r in rows
        }
    except Exception as exc:
        logger.warning("SemGrove metadata fetch failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Build and persist
# ---------------------------------------------------------------------------

def build_and_store_sem_grove_index(db_conn=None) -> bool:
    """
    Build the merged lyrics+audio Voyager index and persist to the DB.

    Rows written to ``lyrics_index_data``:
      * ``sem_grove_whitening`` – JSON whitening stats (no binary payload)
      * ``sem_grove_index``     – Voyager binary + id_map (segmented if large)
    """
    try:
        import voyager  # type: ignore
    except ImportError:
        logger.warning("Voyager unavailable; skipping SemGrove index build.")
        return False

    from app_helper import get_db
    from config import (
        LYRICS_EMBEDDING_DIMENSION,
        EMBEDDING_DIMENSION,
        VOYAGER_M,
        VOYAGER_EF_CONSTRUCTION,
        VOYAGER_MAX_PART_SIZE_MB,
    )

    if db_conn is None:
        db_conn = get_db()

    lyrics_dim    = LYRICS_EMBEDDING_DIMENSION
    audio_dim     = EMBEDDING_DIMENSION
    merged_dim    = lyrics_dim + audio_dim
    max_part_size = VOYAGER_MAX_PART_SIZE_MB * 1024 * 1024

    # Read weights fresh from config at build time so setup-wizard changes
    # are picked up without a server restart.
    W_LYRICS, W_AUDIO = _get_weights()

    try:
        with db_conn.cursor() as cur:
            # ---- Fetch lyrics embeddings ----
            logger.info("SemGrove: fetching lyrics embeddings…")
            cur.execute(
                "SELECT item_id, embedding FROM lyrics_embedding "
                "WHERE embedding IS NOT NULL"
            )
            lyrics_map: Dict[str, np.ndarray] = {}
            for item_id, blob in cur.fetchall():
                if blob is None:
                    continue
                v = np.frombuffer(blob, dtype=np.float32)
                if v.shape[0] == lyrics_dim:
                    lyrics_map[item_id] = v

            if not lyrics_map:
                logger.warning("SemGrove: no lyrics embeddings found; aborting.")
                return False

            # ---- Fetch audio embeddings ----
            logger.info("SemGrove: fetching audio embeddings…")
            cur.execute(
                "SELECT item_id, embedding FROM embedding "
                "WHERE embedding IS NOT NULL"
            )
            audio_map: Dict[str, np.ndarray] = {}
            for item_id, blob in cur.fetchall():
                if blob is None:
                    continue
                v = np.frombuffer(blob, dtype=np.float32)
                if v.shape[0] == audio_dim:
                    audio_map[item_id] = v

            if not audio_map:
                logger.warning("SemGrove: no audio embeddings found; aborting.")
                return False

            # ---- Intersection ----
            common_ids = sorted(set(lyrics_map.keys()) & set(audio_map.keys()))
            if not common_ids:
                logger.warning(
                    "SemGrove: no songs have both lyrics and audio embeddings; aborting."
                )
                return False
            logger.info(
                "SemGrove: %d songs have both embeddings (lyrics=%d, audio=%d).",
                len(common_ids), len(lyrics_map), len(audio_map),
            )

            # ---- Whitening statistics (computed on unit-normed vectors) ----
            logger.info("SemGrove: computing whitening statistics…")
            norm_lyrics = np.vstack([
                lyrics_map[i] / (np.linalg.norm(lyrics_map[i]) + 1e-8)
                for i in common_ids
            ])
            norm_audio = np.vstack([
                audio_map[i] / (np.linalg.norm(audio_map[i]) + 1e-8)
                for i in common_ids
            ])
            std_lyrics = np.std(norm_lyrics, axis=0).astype(np.float32)
            std_audio  = np.std(norm_audio,  axis=0).astype(np.float32)

            # ---- Build merged vectors ----
            logger.info("SemGrove: building %d merged vectors (dim=%d)…", len(common_ids), merged_dim)
            id_map:  Dict[int, str]      = {}
            vectors: List[np.ndarray]    = []
            vid = 0
            for item_id in common_ids:
                mv = _make_merged_vector(
                    lyrics_map[item_id], audio_map[item_id],
                    std_lyrics, std_audio,
                    W_LYRICS, W_AUDIO,
                )
                if mv is None:
                    continue
                vectors.append(mv)
                id_map[vid] = item_id
                vid += 1

            if not vectors:
                logger.warning("SemGrove: no valid merged vectors; aborting.")
                return False

            # ---- Build Voyager index ----
            logger.info("SemGrove: building Voyager index for %d items…", len(vectors))
            builder = voyager.Index(
                space=voyager.Space.Cosine,
                num_dimensions=merged_dim,
                M=VOYAGER_M,
                ef_construction=VOYAGER_EF_CONSTRUCTION,
            )
            builder.add_items(np.vstack(vectors), ids=np.array(list(id_map.keys())))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".voyager") as tmp:
                temp_path = tmp.name
            try:
                builder.save(temp_path)
                with open(temp_path, "rb") as f:
                    index_binary = f.read()
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            if not index_binary:
                logger.error("SemGrove: generated index binary is empty; aborting.")
                return False

            id_map_json = json.dumps(id_map)

            # ---- Persist ----
            # Delete any previous rows
            cur.execute(
                "DELETE FROM lyrics_index_data "
                "WHERE index_name IN (%s, %s) OR index_name LIKE %s ESCAPE '\\'",
                (SEM_GROVE_INDEX_NAME, SEM_GROVE_WHITENING_NAME,
                 r"sem_grove_index\_%\_%"),
            )

            # Whitening stats row (index_data is empty, payload is in id_map_json)
            whitening_json = json.dumps({
                "std_lyrics": std_lyrics.tolist(),
                "std_audio":  std_audio.tolist(),
                "w_lyrics":   W_LYRICS,
                "w_audio":    W_AUDIO,
                "lyrics_dim": lyrics_dim,
                "audio_dim":  audio_dim,
            })
            cur.execute(
                "INSERT INTO lyrics_index_data "
                "(index_name, index_data, id_map_json, embedding_dimension, created_at) "
                "VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP) "
                "ON CONFLICT (index_name) DO UPDATE SET "
                "index_data = EXCLUDED.index_data, "
                "id_map_json = EXCLUDED.id_map_json, "
                "embedding_dimension = EXCLUDED.embedding_dimension, "
                "created_at = EXCLUDED.created_at",
                (SEM_GROVE_WHITENING_NAME, psycopg2.Binary(b""), whitening_json, 0),
            )

            # Index binary (single row or segmented)
            if len(index_binary) <= max_part_size:
                cur.execute(
                    "INSERT INTO lyrics_index_data "
                    "(index_name, index_data, id_map_json, embedding_dimension, created_at) "
                    "VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP) "
                    "ON CONFLICT (index_name) DO UPDATE SET "
                    "index_data = EXCLUDED.index_data, "
                    "id_map_json = EXCLUDED.id_map_json, "
                    "embedding_dimension = EXCLUDED.embedding_dimension, "
                    "created_at = EXCLUDED.created_at",
                    (SEM_GROVE_INDEX_NAME, psycopg2.Binary(index_binary), id_map_json, merged_dim),
                )
                logger.info("SemGrove: stored index as single row.")
            else:
                parts    = _split_bytes(index_binary, max_part_size)
                n_parts  = len(parts)
                insert_q = (
                    "INSERT INTO lyrics_index_data "
                    "(index_name, index_data, id_map_json, embedding_dimension, created_at) "
                    "VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)"
                )
                for idx, part in enumerate(parts, start=1):
                    name        = f"{SEM_GROVE_INDEX_NAME}_{idx}_{n_parts}"
                    part_id_map = id_map_json if idx == 1 else ""
                    cur.execute(insert_q, (name, psycopg2.Binary(part), part_id_map, merged_dim))
                logger.info("SemGrove: stored index in %d segmented rows.", n_parts)

        db_conn.commit()
        logger.info(
            "SemGrove index build complete: %d songs, dim=%d.", len(vectors), merged_dim
        )
        return True

    except Exception as exc:
        logger.error("SemGrove index build failed: %s", exc, exc_info=True)
        try:
            db_conn.rollback()
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _load_sem_grove_index_from_db() -> bool:
    """Load the SemGrove merged Voyager index from the DB into the global cache."""
    try:
        import voyager  # type: ignore
    except ImportError:
        logger.warning("Voyager unavailable; cannot load SemGrove index.")
        return False

    from app_helper import get_db
    from config import VOYAGER_QUERY_EF

    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = 0")

            # ---- Whitening stats ----
            cur.execute(
                "SELECT id_map_json FROM lyrics_index_data WHERE index_name = %s",
                (SEM_GROVE_WHITENING_NAME,),
            )
            row = cur.fetchone()
            if not row:
                logger.info("SemGrove: no whitening stats found; index not built yet.")
                return False

            whitening   = json.loads(row[0])
            std_lyrics  = np.array(whitening["std_lyrics"], dtype=np.float32)
            std_audio   = np.array(whitening["std_audio"],  dtype=np.float32)
            w_lyrics    = float(whitening.get("w_lyrics", W_LYRICS))
            w_audio     = float(whitening.get("w_audio",  W_AUDIO))
            lyrics_dim  = int(whitening["lyrics_dim"])
            audio_dim   = int(whitening["audio_dim"])
            merged_dim  = lyrics_dim + audio_dim

            # ---- Index binary ----
            cur.execute(
                "SELECT index_data, id_map_json, embedding_dimension "
                "FROM lyrics_index_data WHERE index_name = %s",
                (SEM_GROVE_INDEX_NAME,),
            )
            row = cur.fetchone()

            index_stream = None
            id_map_json  = None
            db_dim       = None
            try:
                if row:
                    binary, id_map_json, db_dim = row
                    index_stream = tempfile.TemporaryFile()
                    index_stream.write(binary)
                    index_stream.seek(0)
                else:
                    seg_pattern       = re.compile(r"^sem_grove_index_(\d+)_(\d+)$")
                    parts             = []
                    total_expected    = None
                    id_map_json_cand  = None

                    with conn.cursor(name="sem_grove_index_segments") as seg_cur:
                        seg_cur.itersize = 50
                        seg_cur.execute(
                            "SELECT index_name, index_data, id_map_json, embedding_dimension "
                            "FROM lyrics_index_data WHERE index_name LIKE %s ESCAPE '\\'",
                            (r"sem_grove_index\_%\_%",),
                        )
                        for name, part_data, part_id_map, part_dim in seg_cur:
                            m = seg_pattern.match(name)
                            if not m:
                                continue
                            part_no = int(m.group(1))
                            total   = int(m.group(2))
                            if total_expected is None:
                                total_expected = total
                            elif total_expected != total:
                                logger.error("SemGrove: segment total mismatch.")
                                return False
                            parts.append((part_no, part_data, part_id_map, part_dim))
                            if part_id_map and not id_map_json_cand:
                                id_map_json_cand = part_id_map

                    if total_expected is None or len(parts) != total_expected:
                        logger.info(
                            "SemGrove: no complete index found in DB "
                            "(expected=%s, got=%d).", total_expected, len(parts)
                        )
                        return False

                    parts.sort(key=lambda p: p[0])
                    db_dim       = parts[0][3]
                    id_map_json  = id_map_json_cand
                    index_stream = tempfile.TemporaryFile()
                    for _, part_data, _, _ in parts:
                        index_stream.write(part_data)
                    index_stream.seek(0)

                if index_stream is None or not id_map_json:
                    return False

                if db_dim != merged_dim:
                    logger.error(
                        "SemGrove: dimension mismatch (db=%d, expected=%d).",
                        db_dim, merged_dim,
                    )
                    return False

                loaded_index    = voyager.Index.load(index_stream)
                loaded_index.ef = VOYAGER_QUERY_EF

            finally:
                if index_stream is not None:
                    try:
                        index_stream.close()
                    except Exception:
                        pass

            id_map         = {int(k): v for k, v in json.loads(id_map_json).items()}
            reverse_id_map = {v: k for k, v in id_map.items()}

            if not id_map:
                logger.warning("SemGrove: id_map is empty after load.")
                return False

            _SEM_GROVE_CACHE.update({
                "index":          loaded_index,
                "id_map":         id_map,
                "reverse_id_map": reverse_id_map,
                "std_lyrics":     std_lyrics,
                "std_audio":      std_audio,
                "lyrics_dim":     lyrics_dim,
                "audio_dim":      audio_dim,
                "w_lyrics":       w_lyrics,
                "w_audio":        w_audio,
                "loaded":         True,
                "song_count":     len(id_map),
            })

            logger.info(
                "SemGrove index loaded: %d items, dim=%d.", len(id_map), merged_dim
            )
            return True

    except Exception as exc:
        logger.error("SemGrove index load failed: %s", exc, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Public cache management
# ---------------------------------------------------------------------------

def load_sem_grove_cache_from_db() -> bool:
    """Load the SemGrove index from the DB into the global in-memory cache."""
    ok = _load_sem_grove_index_from_db()
    if not ok:
        _SEM_GROVE_CACHE.update({
            "index":          None,
            "id_map":         None,
            "reverse_id_map": None,
            "std_lyrics":     None,
            "std_audio":      None,
            "loaded":         False,
            "song_count":     0,
        })
    return ok


def refresh_sem_grove_cache() -> bool:
    """Reload the SemGrove index from the DB (hot-reload without restart)."""
    old = _SEM_GROVE_CACHE["song_count"]
    logger.info("SemGrove: refreshing cache (current=%d songs)…", old)
    result = load_sem_grove_cache_from_db()
    logger.info(
        "SemGrove: cache refreshed (%d → %d songs).",
        old, _SEM_GROVE_CACHE["song_count"],
    )
    return result


def is_sem_grove_cache_loaded() -> bool:
    return _SEM_GROVE_CACHE["loaded"]


def get_sem_grove_stats() -> Dict:
    loaded = _SEM_GROVE_CACHE["loaded"]
    idx    = _SEM_GROVE_CACHE.get("index")
    mem_mb = 0
    if loaded and idx is not None:
        mem_mb = round(sys.getsizeof(idx) / (1024 * 1024), 2)
    return {
        "loaded":     loaded,
        "song_count": _SEM_GROVE_CACHE["song_count"],
        "lyrics_dim": _SEM_GROVE_CACHE.get("lyrics_dim"),
        "audio_dim":  _SEM_GROVE_CACHE.get("audio_dim"),
        "w_lyrics":   round(_SEM_GROVE_CACHE["w_lyrics"] ** 2, 2) if loaded else None,
        "w_audio":    round(_SEM_GROVE_CACHE["w_audio"]  ** 2, 2) if loaded else None,
        "memory_mb":  mem_mb,
    }


def get_sem_grove_item_ids() -> set:
    """Return the set of item_ids present in the loaded SemGrove index.

    Returns an empty set if the index is not loaded.
    Used by the autocomplete endpoint to restrict suggestions to songs that
    are actually searchable via the merged index.
    """
    if not _SEM_GROVE_CACHE["loaded"]:
        return set()
    return set(_SEM_GROVE_CACHE["id_map"].values())


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_by_song(seed_item_id: str, limit: int = 50) -> List[Dict]:
    """
    Find songs semantically + acoustically similar to ``seed_item_id``.

    The seed song's pre-stored merged vector is retrieved directly from the
    Voyager index (no re-computation), then used as the query vector.
    Returns a list of ``{item_id, title, author, similarity}`` dicts sorted
    by descending merged-cosine similarity, excluding the seed itself.
    """
    if not _SEM_GROVE_CACHE["loaded"] or _SEM_GROVE_CACHE["index"] is None:
        logger.error("SemGrove index not loaded.")
        return []

    index          = _SEM_GROVE_CACHE["index"]
    id_map         = _SEM_GROVE_CACHE["id_map"]
    reverse_id_map = _SEM_GROVE_CACHE["reverse_id_map"]

    seed_vid = reverse_id_map.get(seed_item_id)
    if seed_vid is None:
        logger.warning("SemGrove: seed '%s' not in index.", seed_item_id)
        return []

    try:
        query_vector = index.get_vector(seed_vid)
    except Exception as exc:
        logger.error("SemGrove: cannot fetch vector for seed '%s': %s", seed_item_id, exc)
        return []

    from config import MAX_SONGS_PER_ARTIST, DUPLICATE_DISTANCE_THRESHOLD_COSINE, DUPLICATE_DISTANCE_CHECK_LOOKBACK
    import numpy as np

    artist_cap    = MAX_SONGS_PER_ARTIST if MAX_SONGS_PER_ARTIST and MAX_SONGS_PER_ARTIST > 0 else 0
    dist_threshold = DUPLICATE_DISTANCE_THRESHOLD_COSINE  # cosine dist < this → near-duplicate
    lookback_n     = DUPLICATE_DISTANCE_CHECK_LOOKBACK if DUPLICATE_DISTANCE_CHECK_LOOKBACK > 0 else 0
    # +1 because the seed itself may appear and will be skipped
    fetch_size   = (limit + max(20, limit * 4) + 1) if (artist_cap or lookback_n) else (limit + 1)
    num_to_query = min(fetch_size, len(index))
    if num_to_query <= 0:
        return []

    try:
        neighbor_ids, distances = index.query(query_vector, k=num_to_query)
    except Exception as exc:
        logger.error("SemGrove: Voyager query failed: %s", exc, exc_info=True)
        return []

    candidate_ids = [
        id_map.get(int(v))
        for v in neighbor_ids
        if id_map.get(int(v)) and id_map.get(int(v)) != seed_item_id
    ]
    metadata_map = _fetch_metadata([seed_item_id] + candidate_ids)

    # Prepend the seed song as the first entry so callers (playlist builders, API
    # consumers) always receive it at position 0.  The frontend hides it.
    seed_meta = metadata_map.get(seed_item_id, {"title": "", "author": ""})
    results: List[Dict] = [{
        "item_id":    seed_item_id,
        "title":      seed_meta.get("title",  "") or "",
        "author":     seed_meta.get("author", "") or "",
        "similarity": 1.0,
        "is_seed":    True,
    }]
    artist_counts:  Dict[str, int]   = {}
    seen_names:     set               = set()   # (title_lower, author_lower) deduplication
    lookback_vecs:  list              = []       # recent kept vectors for distance-based dedup

    for vid, dist in zip(neighbor_ids, distances):
        if len(results) - 1 >= limit:  # -1 to exclude the seed prepended at index 0
            break
        item_id = id_map.get(int(vid))
        if not item_id or item_id == seed_item_id:
            continue
        meta   = metadata_map.get(item_id, {"title": "", "author": ""})
        author = meta.get("author", "") or ""
        title  = meta.get("title",  "") or ""

        # Name-based deduplication (same title+artist = likely duplicate recording)
        name_key = (title.strip().lower(), author.strip().lower())
        if name_key in seen_names:
            continue
        seen_names.add(name_key)

        if artist_cap and author:
            an = author.strip().lower()
            if artist_counts.get(an, 0) >= artist_cap:
                continue
            artist_counts[an] = artist_counts.get(an, 0) + 1

        # Distance-based near-duplicate filter: skip if this song's merged vector is
        # too close (cosine dist < dist_threshold) to any song in the lookback window.
        if lookback_n and dist_threshold > 0:
            try:
                candidate_vec = np.array(index.get_vector(int(vid)), dtype=np.float32)
                norm = np.linalg.norm(candidate_vec)
                if norm > 0:
                    candidate_vec = candidate_vec / norm
                too_close = False
                for lv in lookback_vecs[-lookback_n:]:
                    cosine_dist = float(np.clip(1.0 - np.dot(candidate_vec, lv), 0.0, 2.0))
                    if cosine_dist < dist_threshold:
                        logger.debug(
                            "SemGrove: dropping near-duplicate '%s' by '%s' "
                            "(cosine dist %.4f < threshold %.4f).",
                            title, author, cosine_dist, dist_threshold,
                        )
                        too_close = True
                        break
                if too_close:
                    continue
                lookback_vecs.append(candidate_vec)
            except Exception as _vec_exc:
                logger.debug("SemGrove: could not fetch vector for distance check: %s", _vec_exc)

        results.append({
            "item_id":    item_id,
            "title":      title,
            "author":     author,
            "similarity": max(0.0, 1.0 - float(dist)),
        })

    logger.info("SemGrove search for '%s': %d results.", seed_item_id, len(results))
    return results
