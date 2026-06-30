import gc
import logging
import math
import sys
from typing import Dict, List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


def _get_weights():
    wl = max(0.0, float(config.SEM_GROVE_WEIGHT_LYRICS))
    wa = max(0.0, float(config.SEM_GROVE_WEIGHT_AUDIO))
    return math.sqrt(wl), math.sqrt(wa)


W_LYRICS, W_AUDIO = _get_weights()

SEM_GROVE_INDEX_NAME = "sem_grove_index"
SEM_GROVE_WHITENING_NAME = "sem_grove_whitening"

_SEM_GROVE_CACHE: Dict = {
    "index": None,
    "id_map": None,
    "reverse_id_map": None,
    "std_lyrics": None,
    "std_audio": None,
    "lyrics_dim": None,
    "audio_dim": None,
    "w_lyrics": W_LYRICS,
    "w_audio": W_AUDIO,
    "loaded": False,
    "song_count": 0,
}


def _make_merged_vector(
    l_vec: np.ndarray,
    a_vec: np.ndarray,
    std_lyrics: np.ndarray,
    std_audio: np.ndarray,
    w_l: float,
    w_a: float,
) -> Optional[np.ndarray]:
    try:
        lyr = l_vec.astype(np.float32, copy=False)
        n = np.linalg.norm(lyr)
        if n == 0:
            return None
        lyr = lyr / n
        lyr = lyr / (std_lyrics + 1e-8)
        n2 = np.linalg.norm(lyr)
        if n2 < 1e-8:
            return None
        lyr = lyr / n2

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

        return np.concatenate([w_l * lyr, w_a * a]).astype(np.float32)
    except Exception as exc:
        logger.debug("_make_merged_vector: %s", exc)
        return None


def _fetch_metadata(item_ids: List[str]) -> Dict[str, Dict]:
    from .commons import fetch_track_metadata_map

    return fetch_track_metadata_map(item_ids)


def build_and_store_sem_grove_index(db_conn=None) -> bool:
    from app_helper import get_db
    from config import LYRICS_EMBEDDING_DIMENSION, EMBEDDING_DIMENSION
    from .index_build_helpers import stream_embeddings_to_buffer
    from .paged_ivf import build_and_store_paged_ivf

    if db_conn is None:
        db_conn = get_db()

    lyrics_dim = LYRICS_EMBEDDING_DIMENSION
    audio_dim = EMBEDDING_DIMENSION
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
        audio_pos = {item_id: i for i, item_id in enumerate(audio_ids)}
        common_ids = sorted(set(lyrics_ids) & set(audio_ids))
        if not common_ids:
            logger.warning("SemGrove: no songs have both lyrics and audio embeddings; aborting.")
            return False
        logger.info(
            "SemGrove: %d songs have both embeddings (lyrics=%d, audio=%d).",
            len(common_ids),
            lyrics_buf.shape[0],
            audio_buf.shape[0],
        )

        logger.info("SemGrove: computing whitening statistics (streaming)…")
        sum_l = np.zeros(lyrics_dim, dtype=np.float64)
        sumsq_l = np.zeros(lyrics_dim, dtype=np.float64)
        sum_a = np.zeros(audio_dim, dtype=np.float64)
        sumsq_a = np.zeros(audio_dim, dtype=np.float64)
        n_stats = 0
        for item_id in common_ids:
            lv = lyrics_buf[lyrics_pos[item_id]]
            av = audio_buf[audio_pos[item_id]]
            lv_n = lv / (np.linalg.norm(lv) + 1e-8)
            av_n = av / (np.linalg.norm(av) + 1e-8)
            sum_l += lv_n
            sumsq_l += lv_n * lv_n
            sum_a += av_n
            sumsq_a += av_n * av_n
            n_stats += 1
        mean_l = sum_l / n_stats
        mean_a = sum_a / n_stats
        var_l = np.maximum(sumsq_l / n_stats - mean_l * mean_l, 0.0)
        var_a = np.maximum(sumsq_a / n_stats - mean_a * mean_a, 0.0)
        std_lyrics = np.sqrt(var_l).astype(np.float32)
        std_audio = np.sqrt(var_a).astype(np.float32)
        del sum_l, sumsq_l, sum_a, sumsq_a, mean_l, mean_a, var_l, var_a
        gc.collect()

        logger.info(
            "SemGrove: building disk-paged IVF index for up to %d items (dim=%d)...",
            len(common_ids),
            merged_dim,
        )
        merged = np.empty((len(common_ids), merged_dim), dtype=np.float32)
        kept_ids: List[str] = []
        w = 0
        for item_id in common_ids:
            mv = _make_merged_vector(
                lyrics_buf[lyrics_pos[item_id]],
                audio_buf[audio_pos[item_id]],
                std_lyrics,
                std_audio,
                W_LYRICS,
                W_AUDIO,
            )
            if mv is None:
                continue
            merged[w] = mv
            kept_ids.append(item_id)
            w += 1
        lyrics_buf = audio_buf = lyrics_pos = audio_pos = None
        gc.collect()
        if w == 0:
            logger.warning("SemGrove: no valid merged vectors; aborting build.")
            return False
        merged = merged[:w]

        ok = build_and_store_paged_ivf(
            db_conn, SEM_GROVE_INDEX_NAME, merged, kept_ids, merged_dim, "angular"
        )
        if not ok:
            db_conn.rollback()
            return False
        db_conn.commit()
        logger.info("SemGrove IVF index build complete: %d songs, dim=%d.", w, merged_dim)
        return True

    except Exception as exc:
        logger.error("SemGrove index build failed: %s", exc, exc_info=True)
        try:
            db_conn.rollback()
        except Exception:
            pass
        return False


def _load_sem_grove_index_from_db() -> bool:
    from app_helper import get_db
    from config import LYRICS_EMBEDDING_DIMENSION, EMBEDDING_DIMENSION
    from .paged_ivf import load_index_auto

    lyrics_dim = LYRICS_EMBEDDING_DIMENSION
    audio_dim = EMBEDDING_DIMENSION
    merged_dim = lyrics_dim + audio_dim

    try:
        conn = get_db()
        loaded = load_index_auto(
            conn,
            SEM_GROVE_INDEX_NAME,
            merged_dim,
            'angular',
            label='SemGrove',
        )
        if loaded is None:
            logger.info("SemGrove: IVF index not found; not built yet.")
            return False
        loaded_index, id_map, reverse_id_map = loaded

        _SEM_GROVE_CACHE.update(
            {
                "index": loaded_index,
                "id_map": id_map,
                "reverse_id_map": reverse_id_map,
                "std_lyrics": None,
                "std_audio": None,
                "lyrics_dim": lyrics_dim,
                "audio_dim": audio_dim,
                "w_lyrics": W_LYRICS,
                "w_audio": W_AUDIO,
                "loaded": True,
                "song_count": len(id_map),
            }
        )

        logger.info("SemGrove index loaded: %d items, dim=%d.", len(id_map), merged_dim)
        return True

    except Exception as exc:
        logger.error("SemGrove index load failed: %s", exc, exc_info=True)
        return False


def load_sem_grove_cache_from_db() -> bool:
    ok = _load_sem_grove_index_from_db()
    if not ok:
        _SEM_GROVE_CACHE.update(
            {
                "index": None,
                "id_map": None,
                "reverse_id_map": None,
                "std_lyrics": None,
                "std_audio": None,
                "loaded": False,
                "song_count": 0,
            }
        )
    return ok


def refresh_sem_grove_cache() -> bool:
    old = _SEM_GROVE_CACHE["song_count"]
    logger.info("SemGrove: refreshing cache (current=%d songs)…", old)
    result = load_sem_grove_cache_from_db()
    logger.info(
        "SemGrove: cache refreshed (%d -> %d songs).",
        old,
        _SEM_GROVE_CACHE["song_count"],
    )
    return result


def is_sem_grove_cache_loaded() -> bool:
    return _SEM_GROVE_CACHE["loaded"]


def get_sem_grove_stats() -> Dict:
    loaded = _SEM_GROVE_CACHE["loaded"]
    idx = _SEM_GROVE_CACHE.get("index")
    mem_mb = 0
    if loaded and idx is not None:
        mem_mb = round(sys.getsizeof(idx) / (1024 * 1024), 2)
    return {
        "loaded": loaded,
        "song_count": _SEM_GROVE_CACHE["song_count"],
        "lyrics_dim": _SEM_GROVE_CACHE.get("lyrics_dim"),
        "audio_dim": _SEM_GROVE_CACHE.get("audio_dim"),
        "w_lyrics": round(_SEM_GROVE_CACHE["w_lyrics"] ** 2, 2) if loaded else None,
        "w_audio": round(_SEM_GROVE_CACHE["w_audio"] ** 2, 2) if loaded else None,
        "memory_mb": mem_mb,
    }


def get_sem_grove_item_ids() -> set:
    if not _SEM_GROVE_CACHE["loaded"]:
        return set()
    return set(_SEM_GROVE_CACHE["id_map"].values())


def get_sem_grove_vector_by_id(item_id: str) -> Optional[np.ndarray]:
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
    if not _SEM_GROVE_CACHE["loaded"] or _SEM_GROVE_CACHE["index"] is None:
        return []
    index = _SEM_GROVE_CACHE["index"]
    id_map = _SEM_GROVE_CACHE["id_map"]
    from .paged_ivf import begin_query

    begin_query(index)
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
    vec = get_sem_grove_vector_by_id(item_id)
    if vec is None:
        return []
    return find_sem_grove_neighbors_by_vector(vec, n=n)


def _resolve_search_params(limit, radius_similarity):
    from config import (
        MAX_SONGS_PER_ARTIST,
        DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS,
        DUPLICATE_DISTANCE_CHECK_LOOKBACK,
        SIMILARITY_RADIUS_DEFAULT,
    )

    if radius_similarity is None:
        radius_similarity = SIMILARITY_RADIUS_DEFAULT

    artist_cap = MAX_SONGS_PER_ARTIST if MAX_SONGS_PER_ARTIST and MAX_SONGS_PER_ARTIST > 0 else 0
    dist_threshold = DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS
    lookback_n = DUPLICATE_DISTANCE_CHECK_LOOKBACK if DUPLICATE_DISTANCE_CHECK_LOOKBACK > 0 else 0
    if radius_similarity:
        fetch_size = limit + max(limit * 5, limit * 15) + 1
    else:
        fetch_size = (limit + max(20, limit * 4) + 1) if (artist_cap or lookback_n) else (limit + 1)
    return {
        "radius_similarity": radius_similarity,
        "max_songs_per_artist": MAX_SONGS_PER_ARTIST,
        "artist_cap": artist_cap,
        "dist_threshold": dist_threshold,
        "lookback_n": lookback_n,
        "fetch_size": fetch_size,
    }


def _is_near_duplicate(index, vid, lookback_vecs, lookback_n, dist_threshold, title, author):
    import numpy as np

    try:
        candidate_vec = np.array(index.get_vector(int(vid)), dtype=np.float32)
        norm = np.linalg.norm(candidate_vec)
        if norm > 0:
            candidate_vec = candidate_vec / norm
        for lv in lookback_vecs[-lookback_n:]:
            cosine_dist = float(np.clip(1.0 - np.dot(candidate_vec, lv), 0.0, 2.0))
            if cosine_dist < dist_threshold:
                logger.debug(
                    "SemGrove: dropping near-duplicate '%s' by '%s' "
                    "(cosine dist %.4f < threshold %.4f).",
                    title,
                    author,
                    cosine_dist,
                    dist_threshold,
                )
                return True
        lookback_vecs.append(candidate_vec)
    except Exception as _vec_exc:
        logger.debug("SemGrove: could not fetch vector for distance check: %s", _vec_exc)
    return False


def _collect_search_results(
    index,
    id_map,
    seed_item_id,
    neighbor_ids,
    distances,
    metadata_map,
    limit,
    artist_cap,
    dist_threshold,
    lookback_n,
):
    seed_meta = metadata_map.get(seed_item_id, {"title": "", "author": "", "album": ""})
    results: List[Dict] = [
        {
            "item_id": seed_item_id,
            "title": seed_meta.get("title", "") or "",
            "author": seed_meta.get("author", "") or "",
            "album": seed_meta.get("album", "") or "",
            "similarity": 1.0,
            "is_seed": True,
        }
    ]
    artist_counts: Dict[str, int] = {}
    seen_names: set = set()
    lookback_vecs: list = []

    for vid, dist in zip(neighbor_ids, distances):
        if len(results) - 1 >= limit:
            break
        item_id = id_map.get(int(vid))
        if not item_id or item_id == seed_item_id:
            continue
        meta = metadata_map.get(item_id, {"title": "", "author": "", "album": ""})
        author = meta.get("author", "") or ""
        title = meta.get("title", "") or ""

        name_key = (title.strip().lower(), author.strip().lower())
        if name_key in seen_names:
            continue
        seen_names.add(name_key)

        if artist_cap and author:
            an = author.strip().lower()
            if artist_counts.get(an, 0) >= artist_cap:
                continue
            artist_counts[an] = artist_counts.get(an, 0) + 1

        if (
            lookback_n
            and dist_threshold > 0
            and _is_near_duplicate(
                index, vid, lookback_vecs, lookback_n, dist_threshold, title, author
            )
        ):
            continue

        results.append(
            {
                "item_id": item_id,
                "title": title,
                "author": author,
                "album": meta.get("album", "") or "",
                "similarity": max(0.0, 1.0 - float(dist)),
            }
        )

    return results


def _build_radius_candidates(non_seed, index, reverse_id_map):
    import numpy as np

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
            candidate_data.append(
                {
                    "item_id": r["item_id"],
                    "vector": vec,
                    "dist_anchor": dist_anchor,
                    "title": r.get("title"),
                    "author": r.get("author"),
                }
            )
        except Exception:
            continue
    return candidate_data


def _apply_radius_walk(results, index, reverse_id_map, limit, max_songs_per_artist, seed_item_id):
    import numpy as np

    if len(results) <= 1:
        return results
    try:
        non_seed = [r for r in results if not r.get("is_seed")]
        if not non_seed:
            return results

        candidate_data = _build_radius_candidates(non_seed, index, reverse_id_map)
        if not candidate_data:
            return results

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
            max_songs_per_artist=max_songs_per_artist,
            get_distance_fn=_cosine_dist,
        )

        reordered_ids = [rd["item_id"] for rd in reordered]
        non_seed_map = {r["item_id"]: r for r in non_seed}

        new_results = [results[0]]
        seen_ids = {results[0]["item_id"]}
        for rid in reordered_ids:
            if rid in non_seed_map and rid not in seen_ids:
                new_results.append(non_seed_map[rid])
                seen_ids.add(rid)
        for r in non_seed:
            if r["item_id"] not in seen_ids:
                new_results.append(r)

        logger.info(
            "SemGrove radius walk: reordered %d results for seed '%s'.",
            len(new_results) - 1,
            seed_item_id,
        )
        return new_results
    except Exception:
        logger.exception("SemGrove radius walk failed; returning standard order.")
        return results


def search_by_song(
    seed_item_id: str, limit: int = 50, radius_similarity: bool | None = None
) -> List[Dict]:
    if not _SEM_GROVE_CACHE["loaded"] or _SEM_GROVE_CACHE["index"] is None:
        logger.error("SemGrove index not loaded.")
        return []

    index = _SEM_GROVE_CACHE["index"]
    id_map = _SEM_GROVE_CACHE["id_map"]
    reverse_id_map = _SEM_GROVE_CACHE["reverse_id_map"]

    from .paged_ivf import begin_query

    begin_query(index)

    seed_vid = reverse_id_map.get(seed_item_id)
    if seed_vid is None:
        logger.warning("SemGrove: seed '%s' not in index.", seed_item_id)
        return []

    try:
        query_vector = index.get_vector(seed_vid)
    except Exception as exc:
        logger.exception("SemGrove: cannot fetch vector for seed '%s': %s", seed_item_id, exc)
        return []

    params = _resolve_search_params(limit, radius_similarity)
    num_to_query = min(params["fetch_size"], len(index))
    if num_to_query <= 0:
        return []

    try:
        neighbor_ids, distances = index.query(query_vector, k=num_to_query)
    except Exception as exc:
        logger.error("SemGrove: IVF query failed: %s", exc, exc_info=True)
        return []

    candidate_ids = [
        id_map.get(int(v))
        for v in neighbor_ids
        if id_map.get(int(v)) and id_map.get(int(v)) != seed_item_id
    ]
    metadata_map = _fetch_metadata([seed_item_id] + candidate_ids)

    results = _collect_search_results(
        index,
        id_map,
        seed_item_id,
        neighbor_ids,
        distances,
        metadata_map,
        limit,
        params["artist_cap"],
        params["dist_threshold"],
        params["lookback_n"],
    )

    logger.info("SemGrove search for '%s': %d results.", seed_item_id, len(results))

    if params["radius_similarity"]:
        results = _apply_radius_walk(
            results,
            index,
            reverse_id_map,
            limit,
            params["max_songs_per_artist"],
            seed_item_id,
        )

    return results
