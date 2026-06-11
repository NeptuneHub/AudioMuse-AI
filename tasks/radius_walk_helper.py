"""
Shared Radius Walk helper.

Provides a generalized bucketed greedy walk algorithm for reordering
similarity search results. Used by both MusicNN (voyager_manager) and
SemGrove (sem_grove_manager) search paths.

The walk:
1. Groups candidates into distance buckets from the anchor song
2. Walks inside each bucket picking the next song greedily by
   score = 0.7 * dist_prev + 0.3 * dist_anchor
3. Enforces per-bucket artist limits (max 1 per artist per bucket)
4. Enforces global artist cap (max N songs per artist across all buckets)
5. Post-processes to avoid 3 same-artist songs in a row

Callers are responsible for pre-filtering (distance dedup, name dedup,
mood filtering) and for preparing candidate_data in the expected format.
"""

import logging
import math
import os
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable constants (kept small so the module stays self-contained)
# ---------------------------------------------------------------------------
BUCKET_SIZE = 50
INSTRUMENT_BUCKET_SKIPS = os.environ.get("RADIUS_INSTRUMENTATION", "False").lower() == "true"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_string(text: str) -> str:
    """Lowercase and strip whitespace."""
    if not text:
        return ""
    return text.strip().lower()


def _default_distance_fn(v1: np.ndarray, v2: np.ndarray) -> float:
    """Default Euclidean distance between two float32 vectors."""
    try:
        return float(np.linalg.norm(v1 - v2))
    except Exception:
        return float("inf")


# ---------------------------------------------------------------------------
# Internal: single-bucket walk (extracted to reduce cognitive complexity)
# ---------------------------------------------------------------------------

def _walk_single_bucket(
    bucket_index: int,
    start_item_id: Optional[str],
    buckets: List[Dict],
    n: int,
    get_distance_fn: Callable,
    cap_active: bool,
    max_songs_per_artist: Optional[int],
    walk_state: Dict,
) -> None:
    """
    Greedy walk constrained to one bucket's items.

    Parameters
    ----------
    walk_state : dict
        Mutable shared state with keys:
        - ``playlist_ids`` (List[str])
        - ``used_ids`` (set[str])
        - ``selected_vectors`` (Dict[str, np.ndarray])
        - ``artist_counts`` (Dict[str, int])
        - ``artist_bucket_counts`` (Dict[str, int])
    """
    bucket = buckets[bucket_index]
    items = bucket["items"]
    if not items:
        return

    playlist_ids: List[str] = walk_state["playlist_ids"]
    used_ids: set = walk_state["used_ids"]
    selected_vectors: Dict[str, np.ndarray] = walk_state["selected_vectors"]
    artist_counts: Dict[str, int] = walk_state["artist_counts"]
    artist_bucket_counts: Dict[str, int] = walk_state["artist_bucket_counts"]

    cand_ids: List[str] = bucket["ids"]
    cand_vecs = bucket["vecs"]
    cand_anchor = bucket["dist_anchor"]
    remaining = [True] * len(cand_ids)
    bucket_artist_set: set = set()

    def _find_start_index() -> Optional[int]:
        if start_item_id:
            try:
                si = cand_ids.index(start_item_id)
                if remaining[si] and cand_ids[si] not in used_ids:
                    return si
            except ValueError:
                pass
        for i, cid in enumerate(cand_ids):
            if remaining[i] and cid not in used_ids:
                return i
        return None

    def _accept_at_index(i: int) -> bool:
        cid = cand_ids[i]
        if cid in used_ids:
            return False
        if cap_active:
            author = items[i].get("author")
            if author and author in bucket_artist_set:
                return False
            if author:
                if (
                    artist_bucket_counts.get(author, 0) >= 2
                    and artist_counts.get(author, 0) < max_songs_per_artist
                ):
                    return False
                if artist_counts.get(author, 0) >= max_songs_per_artist:
                    return False
        return True

    cur_idx = _find_start_index()
    if cur_idx is None:
        return

    if _accept_at_index(cur_idx):
        remaining[cur_idx] = False
        cid = cand_ids[cur_idx]
        used_ids.add(cid)
        if len(playlist_ids) < n:
            playlist_ids.append(cid)
        try:
            v = np.asarray(items[cur_idx]["vector"], dtype=np.float32)
        except Exception:
            v = np.array(items[cur_idx]["vector"], dtype=np.float32)
        selected_vectors[cid] = v
        if cap_active:
            a = items[cur_idx].get("author")
            if a:
                artist_counts[a] = artist_counts.get(a, 0) + 1
                if a not in bucket_artist_set:
                    bucket_artist_set.add(a)
                    artist_bucket_counts[a] = artist_bucket_counts.get(a, 0) + 1
    else:
        remaining[cur_idx] = False

    # ---- Greedy selection loop ----
    while True:
        if len(playlist_ids) >= n:
            break
        avail_idxs = [
            i for i, r in enumerate(remaining)
            if r and cand_ids[i] not in used_ids
        ]
        if not avail_idxs:
            break

        try:
            cur_vec = selected_vectors[playlist_ids[-1]]
        except Exception:
            break

        best_i: Optional[int] = None
        best_score = float("inf")

        for i in avail_idxs:
            cid = cand_ids[i]
            if cid in used_ids:
                continue

            if cap_active:
                auth = items[i].get("author")
                if auth and auth in bucket_artist_set:
                    if INSTRUMENT_BUCKET_SKIPS:
                        logger.debug(
                            "Bucket %d: skip idx=%d bucket-artist-limit",
                            bucket_index, i,
                        )
                    continue
                if auth:
                    if (
                        artist_bucket_counts.get(auth, 0) >= 2
                        and artist_counts.get(auth, 0) < max_songs_per_artist
                    ):
                        if INSTRUMENT_BUCKET_SKIPS:
                            logger.debug(
                                "Bucket %d: skip idx=%d bucket-count-limit",
                                bucket_index, i,
                            )
                        continue
                    if artist_counts.get(auth, 0) >= max_songs_per_artist:
                        if INSTRUMENT_BUCKET_SKIPS:
                            logger.debug(
                                "Bucket %d: skip idx=%d artist-cap",
                                bucket_index, i,
                            )
                        continue

            try:
                dist_prev = get_distance_fn(cand_vecs[i], cur_vec)
            except Exception:
                dist_prev = float("inf")

            score = 0.7 * dist_prev + 0.3 * float(cand_anchor[i])
            if score < best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break

        # Accept best candidate
        remaining[best_i] = False
        cid = cand_ids[best_i]
        used_ids.add(cid)
        if len(playlist_ids) < n:
            playlist_ids.append(cid)
        try:
            v = np.asarray(items[best_i]["vector"], dtype=np.float32)
        except Exception:
            v = np.array(items[best_i]["vector"], dtype=np.float32)
        selected_vectors[cid] = v
        if cap_active:
            a = items[best_i].get("author")
            if a:
                artist_counts[a] = artist_counts.get(a, 0) + 1
                if a not in bucket_artist_set:
                    bucket_artist_set.add(a)
                    artist_bucket_counts[a] = artist_bucket_counts.get(a, 0) + 1

        if INSTRUMENT_BUCKET_SKIPS:
            logger.debug(
                "Bucket %d: accepted idx=%d",
                bucket_index, best_i,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _swap_out_artist(ids: List[str], i: int, bad_artist: str, id_to_author: Dict[str, Optional[str]]) -> bool:
    """Try to swap ids[i+2] or ids[i+1] with a later item whose artist differs."""
    for target in (i + 2, i + 1):
        for j in range(i + 3, len(ids)):
            if id_to_author.get(ids[j]) != bad_artist:
                ids[target], ids[j] = ids[j], ids[target]
                return True
    return False


def avoid_triple_adjacent(
    ids: List[str],
    id_to_author: Dict[str, Optional[str]],
) -> List[str]:
    """
    Post-process a playlist so no three consecutive songs share the same artist.

    When a triple is found the function tries to swap with a later song
    whose artist differs.  If no swap is possible the triple is left as-is
    and the window advances.
    """
    i = 0
    while i <= len(ids) - 3:
        a1 = id_to_author.get(ids[i])
        if a1 and a1 == id_to_author.get(ids[i + 1]) == id_to_author.get(ids[i + 2]):
            if not _swap_out_artist(ids, i, a1, id_to_author):
                i += 1
            # If swapped, re-evaluate current window (don't increment i)
        else:
            i += 1
    return ids


def execute_radius_walk(
    candidate_data: List[Dict],
    n: int,
    eliminate_duplicates: bool = False,
    max_songs_per_artist: Optional[int] = None,
    get_distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> List[Dict]:
    """
    Execute the bucketed greedy radius walk on pre-filtered candidate data.

    Parameters
    ----------
    candidate_data : list of dict
        Each dict must have:
        - ``item_id`` (str)
        - ``vector`` (np.ndarray, float32)
        - ``dist_anchor`` (float)  — distance from this candidate to the anchor song
        Optional but strongly recommended for artist dedup:
        - ``title`` (str)
        - ``author`` (str)
    n : int
        Number of songs to return.
    eliminate_duplicates : bool
        When True, per-bucket artist limit (1/artist/bucket) and the global
        artist cap (``max_songs_per_artist``) are enforced during the walk.
    max_songs_per_artist : int or None
        Global cap; only meaningful when *eliminate_duplicates* is True.
        A value ≤ 0 or None disables the cap.
    get_distance_fn : callable or None
        ``fn(v1: np.ndarray, v2: np.ndarray) -> float``.
        Defaults to Euclidean (L2) distance.

    Returns
    -------
    list of dict
        Each dict has ``item_id`` (str) and ``distance`` (float, distance to anchor).
        The list is ordered by the walk; length ≤ *n*.
    """
    if get_distance_fn is None:
        get_distance_fn = _default_distance_fn

    if not candidate_data:
        logger.warning("Radius walk: No candidates available. Returning empty list.")
        return []

    # --- Parameters ---
    buckets_to_scan = max(3, int(math.ceil(n / BUCKET_SIZE)))
    logger.info(
        "Radius walk: N=%d, BUCKET_SIZE=%d, BUCKETS_TO_SCAN=%d",
        n, BUCKET_SIZE, buckets_to_scan,
    )

    # --- Sort by distance to anchor and create buckets ---
    candidate_data.sort(key=lambda x: x["dist_anchor"])

    num_buckets = int(math.ceil(len(candidate_data) / BUCKET_SIZE))
    raw_buckets = [
        candidate_data[i * BUCKET_SIZE : (i + 1) * BUCKET_SIZE]
        for i in range(num_buckets)
    ]

    # Determine vector dimension from first candidate
    vec_dim = 0
    if candidate_data:
        vec_dim = candidate_data[0]["vector"].shape[0]

    buckets: List[Dict] = []
    for b in raw_buckets:
        if b:
            vecs = np.vstack([c["vector"] for c in b])
            dist_anchor_arr = np.array([c["dist_anchor"] for c in b], dtype=np.float32)
        else:
            vecs = np.empty((0, vec_dim), dtype=np.float32)
            dist_anchor_arr = np.empty((0,), dtype=np.float32)
        buckets.append({
            "items": b,
            "ids": [c["item_id"] for c in b],
            "vecs": vecs,
            "dist_anchor": dist_anchor_arr,
        })

    logger.info("Radius walk: Created %d buckets.", len(buckets))

    # --- Initialize the walk ---
    playlist_ids: List[str] = []
    used_ids: set = set()

    try:
        first_song = candidate_data[0]
        playlist_ids.append(first_song["item_id"])
        used_ids.add(first_song["item_id"])
        logger.info("Radius walk: Starting walk with first candidate.")
    except IndexError:
        logger.warning("Radius walk: Candidate data empty, cannot start.")
        return []

    # Dict of selected item_id -> vector for distance checks
    selected_vectors: Dict[str, np.ndarray] = {
        playlist_ids[0]: first_song["vector"].astype(np.float32)
    }

    # Artist tracking (only used when eliminate_duplicates is True)
    cap_active = bool(
        eliminate_duplicates
        and max_songs_per_artist is not None
        and max_songs_per_artist > 0
    )
    artist_counts: Dict[str, int] = {}
    artist_bucket_counts: Dict[str, int] = {}
    try:
        fa = first_song.get("author")
        if fa:
            artist_counts[fa] = 1
            artist_bucket_counts[fa] = 1
    except Exception:
        pass

    num_buckets = len(buckets)
    buckets_to_check = min(num_buckets, buckets_to_scan)

    # ------------------------------------------------------------------
    # Internal helper: walk a single bucket (delegates to module-level)
    # ------------------------------------------------------------------
    walk_state = {
        "playlist_ids":        playlist_ids,
        "used_ids":            used_ids,
        "selected_vectors":    selected_vectors,
        "artist_counts":       artist_counts,
        "artist_bucket_counts": artist_bucket_counts,
    }

    # ------------------------------------------------------------------
    # Process buckets sequentially, expanding window if needed
    # ------------------------------------------------------------------
    processed_buckets = 0
    while len(playlist_ids) < n and processed_buckets < num_buckets:
        target = min(num_buckets, buckets_to_check)
        for bi in range(processed_buckets, target):
            start_id: Optional[str] = None
            if bi == 0:
                start_id = playlist_ids[0] if playlist_ids else None
            _walk_single_bucket(
                bucket_index=bi,
                start_item_id=start_id,
                buckets=buckets,
                n=n,
                get_distance_fn=get_distance_fn,
                cap_active=cap_active,
                max_songs_per_artist=max_songs_per_artist,
                walk_state=walk_state,
            )
            processed_buckets += 1
            if len(playlist_ids) >= n:
                break

        if len(playlist_ids) < n and buckets_to_check < num_buckets:
            prev = buckets_to_check
            buckets_to_check = min(num_buckets, max(prev + 1, prev * 2))
            logger.info(
                "Radius walk: expanded bucket window to %d buckets (needed more songs).",
                buckets_to_check,
            )

    logger.info("Radius walk: Walk complete. Collected %d songs.", len(playlist_ids))

    # ---- Post-processing: avoid triple adjacent ----
    id_to_author: Dict[str, Optional[str]] = {
        c["item_id"]: c.get("author") for c in candidate_data
    }
    playlist_ids = avoid_triple_adjacent(playlist_ids, id_to_author)

    # ---- Build final results (trim to n, map distances) ----
    playlist_ids = playlist_ids[:n]
    dist_anchor_map = {c["item_id"]: c["dist_anchor"] for c in candidate_data}

    final_results: List[Dict] = []
    for item_id in playlist_ids:
        dist_anchor = dist_anchor_map.get(item_id)
        if dist_anchor is not None:
            final_results.append({"item_id": item_id, "distance": dist_anchor})
        else:
            # Fallback: look up in candidate_data directly
            for c in candidate_data:
                if c["item_id"] == item_id:
                    final_results.append({
                        "item_id": item_id,
                        "distance": c["dist_anchor"],
                    })
                    break

    return final_results
