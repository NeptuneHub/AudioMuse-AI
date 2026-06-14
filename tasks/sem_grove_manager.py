"""
Semantic & Groove (SemGrove) Manager

Builds and queries a merged lyrics + audio Voyager index for song-by-song
similarity that respects both lyrical meaning and acoustic genre simultaneously.

Architecture
------------
* Fetches lyrics embeddings (gte-multilingual-base, 768-dim) from ``lyrics_embedding``
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

import gc
import json
import logging
import math
import sys
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
    wl = max(0.0, float(getattr(config, 'SEM_GROVE_WEIGHT_LYRICS', 0.75)))
    wa = max(0.0, float(getattr(config, 'SEM_GROVE_WEIGHT_AUDIO',  0.25)))
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
    from .commons import fetch_track_metadata_map
    return fetch_track_metadata_map(item_ids)


# ---------------------------------------------------------------------------
# Build and persist
# ---------------------------------------------------------------------------

def build_and_store_sem_grove_index(db_conn=None) -> bool:
    """
    Build the merged lyrics+audio Voyager index and persist to the DB.

    Rows written to ``lyrics_index_data``:
      * ``sem_grove_whitening`` – JSON whitening stats (no binary payload)
      * ``sem_grove_index``     – Voyager binary + id_map (segmented if large)

    Memory profile: lyrics and audio embeddings are each streamed into one
    contiguous float32 buffer (no list-of-ndarrays, no vstack). Whitening
    statistics are computed via running sum / sum-of-squares accumulators
    instead of materializing full-library normalized matrices. Merged
    vectors are produced in small batches by an internal generator and fed
    directly to the streaming Voyager index builder -- the full-library
    merged buffer (lyrics_dim + audio_dim per row × N) is never
    materialized. Peak RAM is therefore ``lyrics_buf + audio_buf +
    one merge batch + voyager's own HNSW storage``, with no extra copies.
    """
    try:
        import voyager  # type: ignore  # noqa: F401
    except ImportError:
        logger.warning("Voyager unavailable; skipping SemGrove index build.")
        return False

    from app_helper import get_db
    from config import LYRICS_EMBEDDING_DIMENSION, EMBEDDING_DIMENSION
    from .index_build_helpers import (
        stream_embeddings_to_buffer,
        build_voyager_index_bytes_streaming,
        store_voyager_index_segmented,
        build_id_map,
        EmptyIndexError,
    )

    if db_conn is None:
        db_conn = get_db()

    lyrics_dim = LYRICS_EMBEDDING_DIMENSION
    audio_dim  = EMBEDDING_DIMENSION
    merged_dim = lyrics_dim + audio_dim

    W_LYRICS, W_AUDIO = _get_weights()

    try:
        logger.info("SemGrove: streaming lyrics embeddings…")
        lyrics_buf, lyrics_ids = stream_embeddings_to_buffer(
            table="lyrics_embedding",
            column="embedding",
            dim=lyrics_dim,
            where_clause="embedding IS NOT NULL",
        )
        if lyrics_buf.shape[0] == 0:
            logger.warning("SemGrove: no lyrics embeddings found; aborting.")
            return False

        logger.info("SemGrove: streaming audio embeddings…")
        audio_buf, audio_ids = stream_embeddings_to_buffer(
            table="embedding",
            column="embedding",
            dim=audio_dim,
            where_clause="embedding IS NOT NULL",
        )
        if audio_buf.shape[0] == 0:
            logger.warning("SemGrove: no audio embeddings found; aborting.")
            return False

        lyrics_pos = {item_id: i for i, item_id in enumerate(lyrics_ids)}
        audio_pos  = {item_id: i for i, item_id in enumerate(audio_ids)}
        common_ids = sorted(set(lyrics_ids) & set(audio_ids))
        if not common_ids:
            logger.warning("SemGrove: no songs have both lyrics and audio embeddings; aborting.")
            return False
        logger.info(
            "SemGrove: %d songs have both embeddings (lyrics=%d, audio=%d).",
            len(common_ids), lyrics_buf.shape[0], audio_buf.shape[0],
        )

        logger.info("SemGrove: computing whitening statistics (streaming)…")
        sum_l   = np.zeros(lyrics_dim, dtype=np.float64)
        sumsq_l = np.zeros(lyrics_dim, dtype=np.float64)
        sum_a   = np.zeros(audio_dim,  dtype=np.float64)
        sumsq_a = np.zeros(audio_dim,  dtype=np.float64)
        n_stats = 0
        for item_id in common_ids:
            lv = lyrics_buf[lyrics_pos[item_id]]
            av = audio_buf[audio_pos[item_id]]
            lv_n = lv / (np.linalg.norm(lv) + 1e-8)
            av_n = av / (np.linalg.norm(av) + 1e-8)
            sum_l   += lv_n
            sumsq_l += lv_n * lv_n
            sum_a   += av_n
            sumsq_a += av_n * av_n
            n_stats += 1
        mean_l = sum_l / n_stats
        mean_a = sum_a / n_stats
        var_l  = np.maximum(sumsq_l / n_stats - mean_l * mean_l, 0.0)
        var_a  = np.maximum(sumsq_a / n_stats - mean_a * mean_a, 0.0)
        std_lyrics = np.sqrt(var_l).astype(np.float32)
        std_audio  = np.sqrt(var_a).astype(np.float32)
        del sum_l, sumsq_l, sum_a, sumsq_a, mean_l, mean_a, var_l, var_a
        gc.collect()

        logger.info(
            "SemGrove: building Voyager index for up to %d items (batched merge, dim=%d)…",
            len(common_ids), merged_dim,
        )

        merge_batch_size = 4096

        def _merge_batches():
            """Yield (batch_buf, batch_ids) of merged vectors in chunks.

            Reads ``lyrics_buf`` / ``audio_buf`` by closure; writes one
            small batch buffer at a time and yields it to the streaming
            index builder. Rows that produce ``None`` from
            ``_make_merged_vector`` (zero-norm input) are skipped exactly
            as the previous single-buffer implementation did.
            """
            for start in range(0, len(common_ids), merge_batch_size):
                chunk = common_ids[start:start + merge_batch_size]
                batch_buf = np.empty((len(chunk), merged_dim), dtype=np.float32)
                batch_ids: List[str] = []
                write_idx = 0
                for item_id in chunk:
                    mv = _make_merged_vector(
                        lyrics_buf[lyrics_pos[item_id]],
                        audio_buf[audio_pos[item_id]],
                        std_lyrics, std_audio,
                        W_LYRICS, W_AUDIO,
                    )
                    if mv is None:
                        continue
                    batch_buf[write_idx] = mv
                    batch_ids.append(item_id)
                    write_idx += 1
                if write_idx == 0:
                    continue
                if write_idx < batch_buf.shape[0]:
                    yield batch_buf[:write_idx].copy(), batch_ids
                else:
                    yield batch_buf, batch_ids

        try:
            index_bytes, kept_ids = build_voyager_index_bytes_streaming(
                _merge_batches(), merged_dim, metric="angular",
            )
        except EmptyIndexError as ve:
            logger.warning("SemGrove: no valid merged vectors; aborting: %s", ve)
            return False
        finally:
            lyrics_buf = audio_buf = lyrics_pos = audio_pos = None
            gc.collect()

        if not index_bytes:
            logger.error("SemGrove: generated index binary is empty; aborting.")
            return False

        id_map: Dict[int, str] = build_id_map(kept_ids)
        vid = len(kept_ids)

        whitening_json = json.dumps({
            "std_lyrics": std_lyrics.tolist(),
            "std_audio":  std_audio.tolist(),
            "w_lyrics":   W_LYRICS,
            "w_audio":    W_AUDIO,
            "lyrics_dim": lyrics_dim,
            "audio_dim":  audio_dim,
        })
        with db_conn.cursor() as cur:
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

        store_voyager_index_segmented(
            db_conn,
            target_table="lyrics_index_data",
            index_name=SEM_GROVE_INDEX_NAME,
            index_bytes=index_bytes,
            id_map=id_map,
            embedding_dimension=merged_dim,
        )

        db_conn.commit()
        logger.info("SemGrove index build complete: %d songs, dim=%d.", vid, merged_dim)
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
        import voyager  # type: ignore  # noqa: F401
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

        from .index_build_helpers import load_voyager_index_from_db
        loaded = load_voyager_index_from_db(
            conn, 'lyrics_index_data', SEM_GROVE_INDEX_NAME,
            merged_dim, VOYAGER_QUERY_EF, label='SemGrove',
        )
        if loaded is None:
            return False
        loaded_index, id_map, reverse_id_map = loaded

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
# Vector backend (used by Song Path's Lyrics mode)
# ---------------------------------------------------------------------------

def get_sem_grove_vector_by_id(item_id: str) -> Optional[np.ndarray]:
    """Return the stored merged lyrics+audio vector for ``item_id``, or None."""
    if not _SEM_GROVE_CACHE["loaded"] or _SEM_GROVE_CACHE["index"] is None:
        return None
    vid = _SEM_GROVE_CACHE["reverse_id_map"].get(item_id)
    if vid is None:
        return None
    try:
        return np.asarray(_SEM_GROVE_CACHE["index"].get_vector(vid), dtype=np.float32)
    except Exception as exc:
        logger.debug("SemGrove get_vector failed for '%s': %s", item_id, exc)
        return None


def find_sem_grove_neighbors_by_vector(query_vector, n: int = 100) -> List[Dict]:
    """Nearest neighbours of ``query_vector`` in merged SemGrove space.

    Returns ``[{"item_id": str, "distance": float}, ...]`` mirroring the shape
    that ``voyager_manager.find_nearest_neighbors_by_vector`` returns, so the
    Song Path engine can use it as a drop-in vector backend. Deduplication and
    artist-cap filtering are handled by the path engine itself, so this only
    performs the raw index query.
    """
    if not _SEM_GROVE_CACHE["loaded"] or _SEM_GROVE_CACHE["index"] is None:
        return []
    index  = _SEM_GROVE_CACHE["index"]
    id_map = _SEM_GROVE_CACHE["id_map"]
    num_to_query = min(max(1, int(n)), len(index))
    if num_to_query <= 0:
        return []
    try:
        neighbor_ids, distances = index.query(
            np.asarray(query_vector, dtype=np.float32), k=num_to_query
        )
    except Exception as exc:
        logger.error("SemGrove neighbor query failed: %s", exc, exc_info=True)
        return []
    results: List[Dict] = []
    for vid, dist in zip(neighbor_ids, distances):
        item_id = id_map.get(int(vid))
        if item_id is not None:
            results.append({"item_id": item_id, "distance": float(dist)})
    return results


def find_sem_grove_neighbors_by_id(item_id: str, n: int = 100) -> List[Dict]:
    """Nearest neighbours of a song (by id) in merged SemGrove space."""
    vec = get_sem_grove_vector_by_id(item_id)
    if vec is None:
        return []
    return find_sem_grove_neighbors_by_vector(vec, n=n)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_by_song(seed_item_id: str, limit: int = 50, radius_similarity: bool | None = None) -> List[Dict]:
    """
    Find songs semantically + acoustically similar to ``seed_item_id``.

    The seed song's pre-stored merged vector is retrieved directly from the
    Voyager index (no re-computation), then used as the query vector.
    Returns a list of ``{item_id, title, author, similarity}`` dicts sorted
    by descending merged-cosine similarity, excluding the seed itself.

    When ``radius_similarity`` is True the results are reordered via a
    bucketed greedy radius walk that enforces per-bucket artist limits
    and avoids three consecutive songs from the same artist.
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

    from config import MAX_SONGS_PER_ARTIST, DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS, DUPLICATE_DISTANCE_CHECK_LOOKBACK, SIMILARITY_RADIUS_DEFAULT
    import numpy as np

    # Resolve radius_similarity default
    if radius_similarity is None:
        radius_similarity = SIMILARITY_RADIUS_DEFAULT

    artist_cap    = MAX_SONGS_PER_ARTIST if MAX_SONGS_PER_ARTIST and MAX_SONGS_PER_ARTIST > 0 else 0
    dist_threshold = DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS  # cosine dist < this → near-duplicate
    lookback_n     = DUPLICATE_DISTANCE_CHECK_LOOKBACK if DUPLICATE_DISTANCE_CHECK_LOOKBACK > 0 else 0
    # +1 because the seed itself may appear and will be skipped
    if radius_similarity:
        # Fetch a large pool for the radius walk to have enough candidates to bucket
        fetch_size = limit + max(limit * 5, limit * 15) + 1
    else:
        fetch_size = (limit + max(20, limit * 4) + 1) if (artist_cap or lookback_n) else (limit + 1)
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
    seed_meta = metadata_map.get(seed_item_id, {"title": "", "author": "", "album": ""})
    results: List[Dict] = [{
        "item_id":    seed_item_id,
        "title":      seed_meta.get("title",  "") or "",
        "author":     seed_meta.get("author", "") or "",
        "album":      seed_meta.get("album",  "") or "",
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
        meta   = metadata_map.get(item_id, {"title": "", "author": "", "album": ""})
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
            "album":      meta.get("album", "") or "",
            "similarity": max(0.0, 1.0 - float(dist)),
        })

    logger.info("SemGrove search for '%s': %d results.", seed_item_id, len(results))

    # --- Radius Walk reordering (per-bucket artist limits + triple-adjacent avoidance) ---
    if radius_similarity and len(results) > 1:
        try:
            non_seed = [r for r in results if not r.get("is_seed")]
            if non_seed:
                candidate_data: List[Dict] = []
                for r in non_seed:
                    vid = reverse_id_map.get(r["item_id"])
                    if vid is None:
                        continue
                    try:
                        vec = np.array(index.get_vector(vid), dtype=np.float32)
                        norm_val = np.linalg.norm(vec)
                        if norm_val > 0:
                            vec = vec / norm_val
                        dist_anchor = max(0.0, 1.0 - r.get("similarity", 0.0))
                        candidate_data.append({
                            "item_id":    r["item_id"],
                            "vector":     vec,
                            "dist_anchor": dist_anchor,
                            "title":      r.get("title"),
                            "author":     r.get("author"),
                        })
                    except Exception:
                        continue

                if candidate_data:
                    from .radius_walk_helper import execute_radius_walk

                    def _cosine_dist(v1, v2):
                        try:
                            dot = np.dot(v1, v2)
                            return float(np.clip(1.0 - dot, 0.0, 2.0))
                        except Exception:
                            return float("inf")

                    reordered = execute_radius_walk(
                        candidate_data=candidate_data,
                        n=limit,
                        eliminate_duplicates=True,
                        max_songs_per_artist=MAX_SONGS_PER_ARTIST,
                        get_distance_fn=_cosine_dist,
                    )

                    # Map reordered IDs back to full result dicts
                    reordered_ids = [rd["item_id"] for rd in reordered]
                    non_seed_map  = {r["item_id"]: r for r in non_seed}

                    new_results = [results[0]]  # seed stays at position 0
                    seen_ids = {results[0]["item_id"]}
                    for rid in reordered_ids:
                        if rid in non_seed_map and rid not in seen_ids:
                            new_results.append(non_seed_map[rid])
                            seen_ids.add(rid)
                    # Append any remaining non-seed songs not picked by the walk
                    for r in non_seed:
                        if r["item_id"] not in seen_ids:
                            new_results.append(r)

                    results = new_results
                    logger.info(
                        "SemGrove radius walk: reordered %d results for seed '%s'.",
                        len(results) - 1, seed_item_id,
                    )
        except Exception:
            logger.exception("SemGrove radius walk failed; returning standard order.")

    return results
