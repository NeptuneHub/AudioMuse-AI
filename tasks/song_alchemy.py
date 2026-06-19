import logging
from typing import List, Tuple
import numpy as np

from .ivf_manager import multi_query_ids, find_nearest_neighbors_by_id, get_vector_by_id, _filter_by_distance
from .alchemy_projections import (
    _project_to_2d,
    _project_with_discriminant,
)
from database import get_score_data_by_ids, load_map_projection
import config

logger = logging.getLogger(__name__)


def _get_artist_gmm_vectors_and_weights(artist_identifier: str) -> Tuple[List[np.ndarray], List[float]]:
    """
    Get GMM component centroids and weights for an artist.
    Returns: (list of mean vectors, list of component weights)
    """
    from tasks.artist_gmm_manager import artist_gmm_params, load_artist_index_for_querying, reverse_artist_map
    from app_helper_artist import get_artist_name_by_id
    
    # Ensure artist index is loaded
    if artist_gmm_params is None:
        load_artist_index_for_querying()
    
    if artist_gmm_params is None:
        logger.warning(f"Artist GMM index not available for {artist_identifier}")
        return [], []
    
    # Resolve artist ID to name if needed
    artist_name = artist_identifier
    resolved_name = get_artist_name_by_id(artist_identifier)
    if resolved_name:
        artist_name = resolved_name
    
    gmm = artist_gmm_params.get(artist_name)

    # Fuzzy fallback: normalize away hyphens, en-dashes, spaces, slashes, apostrophes
    # Handles "Blink-182" (hyphen) vs "blink‐182" (en-dash) in GMM index
    if not gmm and reverse_artist_map:
        def _normalize(s: str) -> str:
            return s.lower().replace(' ', '').replace('-', '').replace('\u2010', '').replace('/', '').replace("'", '')

        query_norm = _normalize(artist_name)
        for gmm_artist in reverse_artist_map:
            if _normalize(gmm_artist) == query_norm:
                gmm = artist_gmm_params.get(gmm_artist)
                if gmm:
                    logger.info(f"Fuzzy GMM match: '{artist_name}' → '{gmm_artist}'")
                    artist_name = gmm_artist
                    break

    if not gmm:
        logger.warning(f"No GMM found for artist '{artist_name}'")
        return [], []
    
    means = np.array(gmm['means'])  # Shape: [n_components, embedding_dim]
    weights = np.array(gmm['weights'])  # Shape: [n_components]
    
    # Log info about single-track artists for debugging
    if gmm.get('is_single_track', False):
        logger.info(f"Loaded single-track artist '{artist_name}' with 1 component")
    
    return [means[i] for i in range(len(means))], weights.tolist()


def _get_mood_centroid_vector(item_id: str):
    """Parse 'mood_name:centroid_index' and return the centroid vector as np.ndarray, or None."""
    parts = str(item_id).split(':', 1)
    if len(parts) != 2:
        return None
    mood_name, idx_str = parts[0].strip().lower(), parts[1].strip()
    try:
        cidx = int(idx_str)
        import json as _json
        with open(config.MOOD_CENTROIDS_FILE) as _f:
            _mcdata = _json.load(_f)
        centroids_list = _mcdata.get(mood_name, {}).get('centroids', [])
        if 0 <= cidx < len(centroids_list):
            vec = centroids_list[cidx].get('centroid')
            if vec:
                return np.array(vec, dtype=float)
    except (ValueError, FileNotFoundError) as exc:
        logger.warning(f"Failed to load mood centroid from '{item_id}': {exc}")
    return None


def _get_mood_label(item_id: str) -> str:
    """Return a human-readable label for a mood centroid id like 'happy:3'."""
    parts = str(item_id).split(':', 1)
    if len(parts) != 2:
        return str(item_id)
    mood_name = parts[0].strip()
    return f"{mood_name.capitalize()} #{parts[1].strip()}"


def _get_playlist_components(playlist_id: str) -> Tuple[List[np.ndarray], List[float]]:
    """Represent a media-server playlist as weighted IVF-cell centroids (no re-clustering).

    Every analyzed song already belongs to one IVF coarse cell, and those cells + their
    centroids are held in memory, so we just group the playlist's songs by cell (zero I/O,
    no vector reads, no KMeans). Songs that share a sub-style collapse onto the same cell.
    If the songs hit more than ALCHEMY_PLAYLIST_MAX_CENTROIDS cells, we keep the most
    spread-out centroids. Large playlists are sampled to ALCHEMY_PLAYLIST_MAX_SONGS first.
    Returns (list of centroid vectors, list of weights).
    """
    import random
    from tasks.mediaserver import get_playlist_track_ids
    from .ivf_manager import get_cell_groups_for_items

    track_ids = get_playlist_track_ids(playlist_id)
    if not track_ids:
        logger.warning(f"Playlist '{playlist_id}' returned no tracks")
        return [], []

    total = len(track_ids)
    if total > config.ALCHEMY_PLAYLIST_MAX_SONGS:
        track_ids = random.sample(track_ids, config.ALCHEMY_PLAYLIST_MAX_SONGS)

    groups = get_cell_groups_for_items(track_ids)
    if not groups:
        logger.warning(f"Playlist '{playlist_id}': none of {total} tracks are in the index; no anchor points")
        return [], []

    if len(groups) > config.ALCHEMY_PLAYLIST_MAX_CENTROIDS:
        groups = _select_spread_centroids(groups, config.ALCHEMY_PLAYLIST_MAX_CENTROIDS)

    counts = np.array([count for _, count in groups], dtype=float)
    weights = (counts / counts.sum()).tolist()
    centroids = [np.array(vec, dtype=float) for vec, _ in groups]
    logger.info(f"Playlist '{playlist_id}': {total} tracks -> {len(centroids)} IVF-cell centroids")
    return centroids, weights


def _select_spread_centroids(groups, k):
    """Keep k of the cell centroids, chosen to be as spread out as possible.

    Greedy farthest-first: start from the most-populated cell, then repeatedly add the
    centroid whose nearest already-kept centroid is the farthest away. Cheap (k passes over
    a short list) and avoids picking near-duplicate centroids.
    """
    vecs = [np.array(vec, dtype=float) for vec, _ in groups]
    selected = [0]
    remaining = set(range(1, len(vecs)))
    while len(selected) < k and remaining:
        far_idx, far_dist = None, -1.0
        for i in remaining:
            nearest = min(_metric_distance(vecs[i], vecs[s]) for s in selected)
            if nearest > far_dist:
                far_dist, far_idx = nearest, i
        selected.append(far_idx)
        remaining.discard(far_idx)
    return [groups[i] for i in selected]


def _metric_distance(v_query: np.ndarray, v_cand: np.ndarray) -> float:
    """Distance between two vectors using config.PATH_DISTANCE_METRIC (angular or euclidean)."""
    a = np.asarray(v_query, dtype=float)
    b = np.asarray(v_cand, dtype=float)
    if config.PATH_DISTANCE_METRIC == 'angular':
        a = a / (np.linalg.norm(a) or 1.0)
        b = b / (np.linalg.norm(b) or 1.0)
        cosine = np.clip(np.dot(a, b), -1.0, 1.0)
        return float(np.arccos(cosine) / np.pi)
    return float(np.linalg.norm(a - b))


def _song_anchor_points(item_id) -> List[dict]:
    """One anchor point for a song id, or none if its vector is missing."""
    vec = get_vector_by_id(item_id)
    if vec is None:
        return []
    return [{'vector': np.array(vec, dtype=float), 'weight': 1.0, 'source_type': 'song', 'source_id': item_id, 'comp_idx': 0, 'label': None}]


def _artist_anchor_points(item_id) -> List[dict]:
    """One anchor point per GMM component of an artist."""
    gmm_vecs, gmm_weights = _get_artist_gmm_vectors_and_weights(item_id)
    return [{'vector': np.array(vec, dtype=float), 'weight': float(weight), 'source_type': 'artist', 'source_id': item_id, 'comp_idx': idx, 'label': None}
            for idx, (vec, weight) in enumerate(zip(gmm_vecs, gmm_weights))]


def _anchor_anchor_points(item_id) -> List[dict]:
    """One anchor point for a saved alchemy anchor's centroid."""
    from database import get_alchemy_anchor_by_id
    anchor = get_alchemy_anchor_by_id(item_id)
    if not (anchor and anchor.get('centroid') and isinstance(anchor.get('centroid'), list)):
        return []
    return [{'vector': np.array(anchor['centroid'], dtype=float), 'weight': 1.0, 'source_type': 'anchor', 'source_id': item_id, 'comp_idx': 0, 'label': anchor.get('name', 'Anchor')}]


def _mood_anchor_points(item_id) -> List[dict]:
    """One anchor point for a mood centroid, or none if it cannot be resolved."""
    vec = _get_mood_centroid_vector(item_id)
    if vec is None:
        return []
    return [{'vector': vec, 'weight': 1.0, 'source_type': 'mood', 'source_id': item_id, 'comp_idx': 0, 'label': _get_mood_label(item_id)}]


def _playlist_anchor_points(item_id) -> List[dict]:
    """One anchor point per cluster centroid of a playlist."""
    pl_vecs, pl_weights = _get_playlist_components(item_id)
    return [{'vector': np.array(vec, dtype=float), 'weight': float(weight), 'source_type': 'playlist', 'source_id': item_id, 'comp_idx': idx, 'label': f'Cluster {idx + 1} (w={float(weight):.2f})'}
            for idx, (vec, weight) in enumerate(zip(pl_vecs, pl_weights))]


_ANCHOR_POINT_HANDLERS = {
    'song': _song_anchor_points,
    'artist': _artist_anchor_points,
    'anchor': _anchor_anchor_points,
    'mood': _mood_anchor_points,
    'playlist': _playlist_anchor_points,
}


def _gather_anchor_points(items: List[dict]) -> List[dict]:
    """Expand each add/subtract item into one or more weighted anchor points.

    Returns a list of dicts: {'vector', 'weight', 'source_type', 'source_id', 'comp_idx', 'label'}.
    song/anchor/mood -> 1 point; artist -> GMM components; playlist -> cluster centroids.
    """
    points = []
    for item in items or []:
        item_id = item.get('id')
        if not item_id:
            continue
        handler = _ANCHOR_POINT_HANDLERS.get(item.get('type', 'song').lower())
        if handler:
            points.extend(handler(item_id))
    return points


def _compute_centroid_from_points(points: List[dict]) -> np.ndarray:
    """Weighted mean of anchor-point vectors (weights normalized), or None if there are no points."""
    if not points:
        return None
    vectors_array = np.array([p['vector'] for p in points])
    weights_array = np.array([p['weight'] for p in points], dtype=float)
    total = np.sum(weights_array)
    if total <= 0:
        weights_array = np.ones(len(weights_array)) / len(weights_array)
    else:
        weights_array = weights_array / total
    return np.sum(vectors_array * weights_array[:, np.newaxis], axis=0)


def _compute_centroid_from_items(items: List[dict]) -> np.ndarray:
    """Backward-compatible weighted centroid over the anchor points produced from items."""
    return _compute_centroid_from_points(_gather_anchor_points(items))


def _select_query_points(points: List[dict], max_points: int) -> List[dict]:
    """Cap how many anchor points are used as IVF query vectors, keeping the highest-weight ones."""
    if len(points) <= max_points:
        return points
    return sorted(points, key=lambda p: p['weight'], reverse=True)[:max_points]


def _multi_query_candidates(points: List[dict], n_results: int) -> List[str]:
    """Union the IVF neighbor ids of every (capped) anchor point, in first-seen order.

    Uses the raw, ids-only index search (no DB fetch / no content de-dup) so cost stays at one
    cheap cell search per anchor point; metadata and final selection happen once downstream.
    """
    query_points = _select_query_points(points, config.ALCHEMY_MAX_ANCHOR_POINTS)
    p = len(query_points)
    if p == 0:
        return []
    target = n_results * 3
    if p == 1:
        per_point_n = target
    else:
        per_point_n = max(n_results // 4, (target + p - 1) // p)
    return multi_query_ids([pt['vector'] for pt in query_points], per_point_n)


def song_alchemy(add_items=None, subtract_items=None, add_ids=None, subtract_ids=None, n_results: int = None, subtract_distance: float = None, temperature: float = None) -> dict:
    """Perform Song Alchemy:
    - add_items: list of dicts with 'type' ('song'/'artist') and 'id'
    - subtract_items: list of dicts with 'type' and 'id'
    - add_ids/subtract_ids: legacy support for song IDs only
    - n_results: number of similar songs to fetch (default from config)

    Returns list of song detail dicts (using get_score_data_by_ids mapping)
    """
    from app_helper_artist import get_artist_name_by_id

    if n_results is None:
        n_results = config.ALCHEMY_DEFAULT_N_RESULTS
    n_results = min(n_results, config.ALCHEMY_MAX_N_RESULTS)

    # Support both new (items with type) and legacy (IDs only) API
    if add_items is None and add_ids is not None:
        # Legacy: convert IDs to items
        add_items = [{'type': 'song', 'id': aid} for aid in add_ids]
    if subtract_items is None and subtract_ids is not None:
        subtract_items = [{'type': 'song', 'id': sid} for sid in subtract_ids]
    
    if not add_items or len(add_items) < 1:
        raise ValueError("At least one item must be in the ADD set")

    add_anchor_points = _gather_anchor_points(add_items)
    if not add_anchor_points:
        return {"results": [], "filtered_out": [], "centroid_2d": None}
    sub_anchor_points = _gather_anchor_points(subtract_items) if subtract_items else []

    add_centroid = _compute_centroid_from_points(add_anchor_points)
    subtract_centroid = _compute_centroid_from_points(sub_anchor_points) if sub_anchor_points else None

    # Normalize temperature early so downstream logic (including the special-case
    # branch below) can safely compare/convert it. If the frontend omitted the
    # parameter or provided a non-numeric/null value, fall back to the configured
    # default. This ensures temperature is optional from the API perspective.
    try:
        if temperature is None:
            # Use configured default
            temperature = float(config.ALCHEMY_TEMPERATURE)
        else:
            # Coerce numeric-like strings as well
            temperature = float(temperature)
    except Exception:
        logger.warning(f"Invalid temperature value passed to song_alchemy: {temperature!r}; falling back to config default")
        try:
            temperature = float(config.ALCHEMY_TEMPERATURE)
        except Exception:
            temperature = 1.0

    # Find nearest neighbors using IVF, one query per ADD anchor point, then union them.
    # Special-case: exactly one ADD song with temperature==0 uses the id-based query so results
    # match the "similar song" path exactly.
    if temperature is not None and float(temperature) == 0.0 and add_items and len(add_items) == 1 and add_items[0].get('type') == 'song':
        try:
            neighbors = find_nearest_neighbors_by_id(add_items[0]['id'], n=n_results)
            candidate_ids = [n['item_id'] for n in neighbors]
        except Exception:
            candidate_ids = _multi_query_candidates(add_anchor_points, n_results)
    else:
        candidate_ids = _multi_query_candidates(add_anchor_points, n_results)
    if not candidate_ids:
        return {"results": [], "filtered_out": [], "centroid_2d": None}

    vec_cache: dict = {}

    def _vec(cid):
        """Memoized get_vector_by_id so the subtract-filter and scoring loops below
        each re-read the same candidate vector at most once per request."""
        if cid not in vec_cache:
            vec_cache[cid] = get_vector_by_id(cid)
        return vec_cache[cid]

    # Remove any user-provided ADD or SUBTRACT song items from candidate list
    # Extract song IDs from items
    add_song_ids = [item['id'] for item in add_items if item.get('type') == 'song' and item.get('id')]
    subtract_song_ids = [item['id'] for item in (subtract_items or []) if item.get('type') == 'song' and item.get('id')]
    
    if add_song_ids:
        add_set = set(add_song_ids)
        candidate_ids = [cid for cid in candidate_ids if cid not in add_set]
    if subtract_song_ids:
        sub_set = set(subtract_song_ids)
        candidate_ids = [cid for cid in candidate_ids if cid not in sub_set]

    # If subtract anchor points present, drop any candidate within the threshold of ANY of them.
    filtered_out = []
    filtered = candidate_ids
    if sub_anchor_points:
        filtered = []
        if subtract_distance is None:
            if config.PATH_DISTANCE_METRIC == 'angular':
                threshold = config.ALCHEMY_SUBTRACT_DISTANCE_ANGULAR
            else:
                threshold = config.ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN
        else:
            threshold = subtract_distance

        sub_vecs = [p['vector'] for p in sub_anchor_points]
        for cid in candidate_ids:
            vec = _vec(cid)
            if vec is None:
                continue
            v_sub = np.array(vec, dtype=float)
            min_sub = min(_metric_distance(s, v_sub) for s in sub_vecs)
            if min_sub >= threshold:
                filtered.append(cid)
            else:
                filtered_out.append(cid)

    candidate_ids = filtered

    # Trim to desired n_results (we'll sample probabilistically from these candidates)
    candidate_ids = candidate_ids[: max(n_results * 3, n_results)]

    from app_helper import get_db
    candidate_ids = [r['item_id'] for r in _filter_by_distance([{'item_id': cid} for cid in candidate_ids], get_db())]

    # Compute distance from add_centroid for each candidate (to display in results)
    # Gather vectors for projection
    proj_vectors = []
    proj_ids = []
    playlist_vec_by_marker = {}
    # include add_centroid and subtract_centroid in the matrix so projection aligns
    # include individual add/subtract song vectors first so we can show them explicitly
    add_meta = []
    if add_items:
        # Add songs as individual points
        add_song_items = [item for item in add_items if item.get('type') == 'song']
        if add_song_items:
            add_song_ids = [item['id'] for item in add_song_items]
            add_details = get_score_data_by_ids(add_song_ids)
            add_map = {d['item_id']: d for d in add_details}
            for item in add_song_items:
                aid = item['id']
                vec = get_vector_by_id(aid)
                if vec is not None:
                    proj_vectors.append(np.array(vec, dtype=float))
                    proj_ids.append(f'__add_id__{aid}')
                    add_meta.append({'item_id': aid, 'title': add_map.get(aid, {}).get('title'), 'author': add_map.get(aid, {}).get('author'), 'type': 'song'})

        # Add anchors as individual points
        add_anchor_items = [item for item in add_items if item.get('type') == 'anchor']
        if add_anchor_items:
            from database import get_alchemy_anchor_by_id
            for item in add_anchor_items:
                anchor_id = item['id']
                anchor = get_alchemy_anchor_by_id(anchor_id)
                if anchor and anchor.get('centroid') and isinstance(anchor['centroid'], list):
                    vec = np.array(anchor['centroid'], dtype=float)
                    proj_vectors.append(vec)
                    proj_ids.append(f'__add_anchor__{anchor_id}')
                    add_meta.append({'item_id': anchor_id, 'title': anchor.get('name', 'Anchor'), 'author': '', 'type': 'anchor'})
        
        # Add mood centroids as individual points
        add_mood_items = [item for item in add_items if item.get('type') == 'mood']
        for item in add_mood_items:
            mood_id = item['id']
            vec = _get_mood_centroid_vector(mood_id)
            if vec is not None:
                proj_vectors.append(vec)
                proj_ids.append(f'__add_mood__{mood_id}')
                add_meta.append({'item_id': mood_id, 'title': _get_mood_label(mood_id), 'author': '', 'type': 'mood'})

        # Add artist GMM components - metadata only (projections will be looked up from precomputed cache)
        add_artist_items = [item for item in add_items if item.get('type') == 'artist']
        for item in add_artist_items:
            artist_id = item['id']
            logger.info(f"Processing ADD artist: {artist_id}")
            gmm_vecs, gmm_weights = _get_artist_gmm_vectors_and_weights(artist_id)
            logger.info(f"Retrieved {len(gmm_vecs)} GMM components for artist {artist_id}")
            for comp_idx, (vec, weight) in enumerate(zip(gmm_vecs, gmm_weights)):
                # Store metadata for artist component
                artist_name = artist_id
                resolved = get_artist_name_by_id(artist_id)
                if resolved:
                    artist_name = resolved
                logger.info(f"Added ADD artist component {comp_idx}: {artist_name} (weight={weight:.2f})")
                add_meta.append({
                    'item_id': f'{artist_id}_comp{comp_idx}',
                    'title': f'Component {comp_idx+1} (w={weight:.2f})',
                    'author': artist_name,
                    'is_artist_component': True,
                    'weight': weight
                })

        # Add playlist cluster centroids as individual points (projected locally)
        for p in add_anchor_points:
            if p['source_type'] != 'playlist':
                continue
            marker = f"__add_playlist__{p['source_id']}_c{p['comp_idx']}"
            vec = np.array(p['vector'], dtype=float)
            proj_vectors.append(vec)
            proj_ids.append(marker)
            playlist_vec_by_marker[marker] = vec
            add_meta.append({
                'item_id': f"{p['source_id']}_c{p['comp_idx']}",
                'title': p['label'],
                'author': 'Playlist',
                'is_playlist_component': True,
                'weight': p['weight']
            })

    sub_meta = []
    if subtract_items:
        # Add songs as individual points
        subtract_song_items = [item for item in subtract_items if item.get('type') == 'song']
        if subtract_song_items:
            subtract_song_ids = [item['id'] for item in subtract_song_items]
            sub_details = get_score_data_by_ids(subtract_song_ids)
            sub_map = {d['item_id']: d for d in sub_details}
            for item in subtract_song_items:
                sid = item['id']
                vec = get_vector_by_id(sid)
                if vec is not None:
                    proj_vectors.append(np.array(vec, dtype=float))
                    proj_ids.append(f'__sub_id__{sid}')
                    sub_meta.append({'item_id': sid, 'title': sub_map.get(sid, {}).get('title'), 'author': sub_map.get(sid, {}).get('author'), 'type': 'song'})

        # Add anchors as individual points
        subtract_anchor_items = [item for item in subtract_items if item.get('type') == 'anchor']
        if subtract_anchor_items:
            from database import get_alchemy_anchor_by_id
            for item in subtract_anchor_items:
                anchor_id = item['id']
                anchor = get_alchemy_anchor_by_id(anchor_id)
                if anchor and anchor.get('centroid') and isinstance(anchor['centroid'], list):
                    vec = np.array(anchor['centroid'], dtype=float)
                    proj_vectors.append(vec)
                    proj_ids.append(f'__sub_anchor__{anchor_id}')
                    sub_meta.append({'item_id': anchor_id, 'title': anchor.get('name', 'Anchor'), 'author': '', 'type': 'anchor'})
        
        # Add mood centroids as individual points
        subtract_mood_items = [item for item in subtract_items if item.get('type') == 'mood']
        for item in subtract_mood_items:
            mood_id = item['id']
            vec = _get_mood_centroid_vector(mood_id)
            if vec is not None:
                proj_vectors.append(vec)
                proj_ids.append(f'__sub_mood__{mood_id}')
                sub_meta.append({'item_id': mood_id, 'title': _get_mood_label(mood_id), 'author': '', 'type': 'mood'})

        # Add artist GMM components - metadata only (projections will be looked up from precomputed cache)
        subtract_artist_items = [item for item in subtract_items if item.get('type') == 'artist']
        for item in subtract_artist_items:
            artist_id = item['id']
            logger.info(f"Processing SUBTRACT artist: {artist_id}")
            gmm_vecs, gmm_weights = _get_artist_gmm_vectors_and_weights(artist_id)
            logger.info(f"Retrieved {len(gmm_vecs)} GMM components for artist {artist_id}")
            for comp_idx, (vec, weight) in enumerate(zip(gmm_vecs, gmm_weights)):
                # Store metadata for artist component
                artist_name = artist_id
                resolved = get_artist_name_by_id(artist_id)
                if resolved:
                    artist_name = resolved
                logger.info(f"Added SUBTRACT artist component {comp_idx}: {artist_name} (weight={weight:.2f})")
                sub_meta.append({
                    'item_id': f'{artist_id}_comp{comp_idx}',
                    'title': f'Component {comp_idx+1} (w={weight:.2f})',
                    'author': artist_name,
                    'is_artist_component': True,
                    'weight': weight
                })

        # Add playlist cluster centroids as individual points (projected locally)
        for p in sub_anchor_points:
            if p['source_type'] != 'playlist':
                continue
            marker = f"__sub_playlist__{p['source_id']}_c{p['comp_idx']}"
            vec = np.array(p['vector'], dtype=float)
            proj_vectors.append(vec)
            proj_ids.append(marker)
            playlist_vec_by_marker[marker] = vec
            sub_meta.append({
                'item_id': f"{p['source_id']}_c{p['comp_idx']}",
                'title': p['label'],
                'author': 'Playlist',
                'is_playlist_component': True,
                'weight': p['weight']
            })

    # include centroids as well so they are in the same projection space
    if add_centroid is not None:
        proj_vectors.append(add_centroid)
        proj_ids.append('__add_centroid__')
    if subtract_centroid is not None:
        proj_vectors.append(subtract_centroid)
        proj_ids.append('__subtract_centroid__')

    # keep track of mapping for candidate and filtered_out
    for cid in candidate_ids:
        vec = get_vector_by_id(cid)
        if vec is None:
            continue
        proj_vectors.append(np.array(vec, dtype=float))
        proj_ids.append(cid)
    for fid in filtered_out:
        vec = get_vector_by_id(fid)
        if vec is None:
            continue
        proj_vectors.append(np.array(vec, dtype=float))
        proj_ids.append(fid)

    # Try to use a precomputed 2D projection saved in the DB (same approach as app_map.py).
    # If a precomputed map exists, use its coords for any matching item_ids. Only compute
    # projections locally for the subset of proj_ids that are missing from the precomputed map.
    projection_used = 'none'
    proj_map = {}

    try:
        id_map, precomp_proj = load_map_projection('main_map')
    except Exception:
        id_map, precomp_proj = None, None

    id_to_coord = {}
    if id_map is not None and precomp_proj is not None:
        try:
            # id_map is expected to be a list of item_ids in the same order as rows in precomp_proj
            # Use string keys to be robust (DB item_ids are text)
            for iid, coord in zip(id_map, precomp_proj.tolist()):
                id_to_coord[str(iid)] = (float(coord[0]), float(coord[1]))
        except Exception:
            id_to_coord = {}
    
    # Load precomputed artist component projections
    artist_comp_to_coord = {}
    try:
        from database import ARTIST_PROJECTION_CACHE
        if ARTIST_PROJECTION_CACHE:
            component_map = ARTIST_PROJECTION_CACHE.get('component_map', [])
            projection = ARTIST_PROJECTION_CACHE.get('projection')
            if projection is not None and len(component_map) > 0:
                for idx, comp_info in enumerate(component_map):
                    if idx < len(projection):
                        artist_id = comp_info['artist_id']
                        comp_idx = comp_info['component_idx']
                        key = f"{artist_id}_{comp_idx}"
                        artist_comp_to_coord[key] = (float(projection[idx][0]), float(projection[idx][1]))
                logger.info(f"Loaded {len(artist_comp_to_coord)} precomputed artist component projections")
    except Exception as e:
        logger.warning(f"Failed to load artist projection cache: {e}")

    # Fill proj_map from precomputed projection where possible
    missing_ids = []
    missing_vectors = []
    for pid in proj_ids:
        # markers like '__add_id__{id}' or '__sub_id__{id}' map to underlying item ids
        if isinstance(pid, str) and pid.startswith('__add_id__'):
            item_id = pid.replace('__add_id__', '')
            coord = id_to_coord.get(str(item_id))
            if coord is not None:
                proj_map[pid] = coord
            else:
                missing_ids.append(pid)
        elif isinstance(pid, str) and pid.startswith('__sub_id__'):
            item_id = pid.replace('__sub_id__', '')
            coord = id_to_coord.get(str(item_id))
            if coord is not None:
                proj_map[pid] = coord
            else:
                missing_ids.append(pid)
        elif pid in ('__add_centroid__', '__subtract_centroid__'):
            # compute centroid coords later (from member point coords) if possible
            continue
        else:
            # regular item id
            coord = id_to_coord.get(str(pid))
            if coord is not None:
                proj_map[pid] = coord
            else:
                missing_ids.append(pid)
    
    # Now add artist component projections from precomputed cache
    # Note: Artist components are NOT in proj_ids because we didn't add their vectors
    # We need to manually add them to proj_map
    for m in add_meta:
        if m.get('is_artist_component'):
            # Extract artist_id and component_idx from item_id format: {artist_id}_comp{comp_idx}
            item_id_parts = m['item_id'].split('_comp')
            if len(item_id_parts) == 2:
                artist_id = item_id_parts[0]
                comp_idx = int(item_id_parts[1])
                key = f"{artist_id}_{comp_idx}"
                coord = artist_comp_to_coord.get(key)
                if coord is not None:
                    pid = f"__add_artist_comp__{artist_id}_{comp_idx}"
                    proj_map[pid] = coord
                    logger.debug(f"Added ADD artist component to proj_map: key={key}, pid={pid}, coord={coord}")
                else:
                    logger.warning(f"No precomputed projection for ADD artist component: key={key}, available keys={list(artist_comp_to_coord.keys())[:5]}")
    
    for m in sub_meta:
        if m.get('is_artist_component'):
            # Extract artist_id and component_idx from item_id format: {artist_id}_comp{comp_idx}
            item_id_parts = m['item_id'].split('_comp')
            if len(item_id_parts) == 2:
                artist_id = item_id_parts[0]
                comp_idx = int(item_id_parts[1])
                key = f"{artist_id}_{comp_idx}"
                coord = artist_comp_to_coord.get(key)
                if coord is not None:
                    pid = f"__sub_artist_comp__{artist_id}_{comp_idx}"
                    proj_map[pid] = coord
                    logger.debug(f"Added SUB artist component to proj_map: key={key}, pid={pid}, coord={coord}")
                else:
                    logger.warning(f"No precomputed projection for SUB artist component: key={key}, available keys={list(artist_comp_to_coord.keys())[:5]}")

    # For centroids, attempt to compute centroid coordinates from precomputed member points
    # This function computes weighted centroid from songs + artist component projections
    def _centroid_from_member_coords(items, is_add=True):
        coords = []
        weights = []
        
        # Collect song coordinates
        for item in items:
            if item.get('type') == 'song':
                mid = item['id']
                c = id_to_coord.get(str(mid))
                if c is not None:
                    coords.append(np.array(c, dtype=float))
                    weights.append(1.0)
            elif item.get('type') == 'anchor':
                mid = item['id']
                c = id_to_coord.get(str(mid))
                if c is not None:
                    coords.append(np.array(c, dtype=float))
                    weights.append(1.0)
            elif item.get('type') == 'mood':
                # Mood centroids won't be in id_to_coord; use proj_map instead
                prefix = '__add_mood__' if is_add else '__sub_mood__'
                c = proj_map.get(f"{prefix}{item['id']}")
                if c is not None:
                    coords.append(np.array(c, dtype=float))
                    weights.append(1.0)

        # Collect artist component coordinates (with their GMM weights) from precomputed projections
        for item in items:
            if item.get('type') == 'artist':
                artist_id = item['id']
                gmm_vecs, gmm_weights = _get_artist_gmm_vectors_and_weights(artist_id)
                for comp_idx, weight in enumerate(gmm_weights):
                    # Look up projection coordinate for this artist component from precomputed cache
                    key = f"{artist_id}_{comp_idx}"
                    c = artist_comp_to_coord.get(key)
                    if c is not None:
                        coords.append(np.array(c, dtype=float))
                        weights.append(weight)

        # Collect playlist cluster coordinates (with membership weights) from local projections
        member_points = add_anchor_points if is_add else sub_anchor_points
        prefix = '__add_playlist__' if is_add else '__sub_playlist__'
        for item in items:
            if item.get('type') == 'playlist':
                for p in member_points:
                    if p['source_type'] != 'playlist' or p['source_id'] != item['id']:
                        continue
                    c = proj_map.get(f"{prefix}{p['source_id']}_c{p['comp_idx']}")
                    if c is not None:
                        coords.append(np.array(c, dtype=float))
                        weights.append(p['weight'])

        if not coords:
            return None
        
        # Weighted mean of all coordinates
        coords_array = np.vstack(coords)
        weights_array = np.array(weights)
        weights_array = weights_array / np.sum(weights_array)  # Normalize
        
        weighted_mean = np.sum(coords_array * weights_array[:, np.newaxis], axis=0)
        return (float(weighted_mean[0]), float(weighted_mean[1]))

    # NOTE: We will compute centroid coordinates AFTER all projections are done
    # (see below after proj_map is fully populated)

    # Collect vectors for any proj_ids that are still missing (we will compute only these)
    # Note: Artist components are NOT in this list - they use precomputed projections
    for pid in proj_ids:
        if pid in proj_map:
            continue
        if pid in ('__add_centroid__', '__subtract_centroid__'):
            # if not set from member coords, skip for now
            continue
        
        # Get the actual vector for this projection ID
        vec = None
        
        # resolve underlying item id for add/sub song markers
        if isinstance(pid, str) and pid.startswith('__add_id__'):
            item_id = pid.replace('__add_id__', '')
            vec = get_vector_by_id(item_id)
        elif isinstance(pid, str) and pid.startswith('__sub_id__'):
            item_id = pid.replace('__sub_id__', '')
            vec = get_vector_by_id(item_id)
        elif isinstance(pid, str) and pid.startswith('__add_anchor__'):
            anchor_id = pid.replace('__add_anchor__', '')
            from database import get_alchemy_anchor_by_id
            anchor = get_alchemy_anchor_by_id(anchor_id)
            if anchor and anchor.get('centroid') and isinstance(anchor['centroid'], list):
                vec = np.array(anchor['centroid'], dtype=float)
            else:
                vec = None
        elif isinstance(pid, str) and pid.startswith('__sub_anchor__'):
            anchor_id = pid.replace('__sub_anchor__', '')
            from database import get_alchemy_anchor_by_id
            anchor = get_alchemy_anchor_by_id(anchor_id)
            if anchor and anchor.get('centroid') and isinstance(anchor['centroid'], list):
                vec = np.array(anchor['centroid'], dtype=float)
            else:
                vec = None
        elif isinstance(pid, str) and pid.startswith(('__add_mood__', '__sub_mood__')):
            mood_id = pid.split('__', 3)[-1]  # extract 'happy:3' from '__add_mood__happy:3'
            vec = _get_mood_centroid_vector(mood_id)
        elif isinstance(pid, str) and pid.startswith(('__add_playlist__', '__sub_playlist__')):
            vec = playlist_vec_by_marker.get(pid)
        else:
            # regular item id
            vec = get_vector_by_id(pid)

        if vec is None:
            # can't project without vector; leave missing
            continue
        missing_ids.append(pid)
        missing_vectors.append(np.array(vec, dtype=float))

    # If we have missing vectors, compute projections for them only
    if missing_vectors:
        try:
            # For small sets (< 50 points), skip expensive UMAP and use fast PCA
            # Artist components typically add only 2-10 points per artist
            local_projections = None
            
            # Try discriminant first if we have both add/sub vectors
            if len(missing_vectors) >= 4:  # Need at least 2+2 for discriminant
                try:
                    # Build add/sub vectors from all add/subtract items (songs + artist components)
                    local_add_vecs = []
                    local_sub_vecs = []
                    
                    for pid in missing_ids:
                        idx = missing_ids.index(pid)
                        vec = missing_vectors[idx]
                        if pid.startswith(('__add_id__', '__add_artist_comp__', '__add_playlist__')):
                            local_add_vecs.append(vec)
                        elif pid.startswith(('__sub_id__', '__sub_artist_comp__', '__sub_playlist__')):
                            local_sub_vecs.append(vec)
                    
                    if local_add_vecs and local_sub_vecs and _project_with_discriminant is not None:
                        local_projections = _project_with_discriminant(local_add_vecs, local_sub_vecs, missing_vectors)
                        projection_used = 'discriminant'
                except Exception:
                    local_projections = None

            # For small sets or if discriminant failed, use fast PCA instead of slow UMAP
            if local_projections is None:
                try:
                    # Use PCA for speed (sub-second vs 30 seconds for UMAP)
                    local_projections = _project_to_2d(missing_vectors)
                    projection_used = 'pca'
                except Exception:
                    # fallback zeros
                    local_projections = [(0.0, 0.0) for _ in missing_vectors]

            # Assign local projections back into proj_map in the same order
            for pid, coord in zip(missing_ids, local_projections):
                proj_map[pid] = (float(coord[0]), float(coord[1]))
        except Exception as e:
            logger.warning(f"Failed to compute local projections for missing ids: {e}")

    # Ensure every proj_id has something (fill remaining with zeros)
    for pid in proj_ids:
        if pid not in proj_map:
            proj_map[pid] = (0.0, 0.0)
    
    # NOW compute centroid coordinates from member coordinates (after proj_map is fully populated)
    add_centroid_2d_db = None
    subtract_centroid_2d_db = None
    try:
        if add_items:
            add_centroid_2d_db = _centroid_from_member_coords(add_items, is_add=True)
        if subtract_items:
            subtract_centroid_2d_db = _centroid_from_member_coords(subtract_items, is_add=False)
        if add_centroid_2d_db is not None:
            proj_map['__add_centroid__'] = add_centroid_2d_db
            logger.info(f"ADD centroid 2D computed from members: {add_centroid_2d_db}")
        if subtract_centroid_2d_db is not None:
            proj_map['__subtract_centroid__'] = subtract_centroid_2d_db
            logger.info(f"SUBTRACT centroid 2D computed from members: {subtract_centroid_2d_db}")
    except Exception as e:
        logger.warning(f"Failed to compute centroid from member coords: {e}")

    # Score each candidate by its distance to the NEAREST add anchor point (match-any).
    distances = {}
    add_vecs = [p['vector'] for p in add_anchor_points]
    for cid in candidate_ids:
        vec = _vec(cid)
        if vec is None:
            continue
        v = np.array(vec, dtype=float)
        distances[cid] = min(_metric_distance(a, v) for a in add_vecs)

    # Fetch details
    details = get_score_data_by_ids(candidate_ids)
    details_map = {d['item_id']: d for d in details}

    # Minimal: ensure album/album_artist is present for each result (from score table via get_score_data_by_ids)
    for d in details_map.values():
        if 'album' not in d or not d['album']:
            d['album'] = 'Unknown'
        if 'album_artist' not in d or not d['album_artist']:
            d['album_artist'] = 'Unknown'

    seed_song_ids = [sid for sid in (add_song_ids + subtract_song_ids) if sid]
    seen_signatures = set()
    if seed_song_ids:
        for sd in get_score_data_by_ids(seed_song_ids):
            seen_signatures.add(((sd.get('title') or '').strip().lower(), (sd.get('author') or '').strip().lower()))
    deduped_ids = []
    for cid in candidate_ids:
        d = details_map.get(cid)
        if not d:
            continue
        signature = ((d.get('title') or '').strip().lower(), (d.get('author') or '').strip().lower())
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped_ids.append(cid)
    candidate_ids = deduped_ids

    # Build a list of scored candidates for probabilistic sampling
    scored_candidates = []
    for cid in candidate_ids:
        if cid in details_map and cid in distances:
            scored_candidates.append((cid, distances[cid]))

    # Temperature was already normalized earlier in the function, but double-check here
    if temperature is None:
        try:
            from config import ALCHEMY_TEMPERATURE as _cfg_temp
            temperature = float(_cfg_temp)
        except Exception:
            temperature = 1.0
    
    logger.info(f"Song Alchemy: Using temperature={temperature} for probabilistic sampling of {len(scored_candidates)} candidates")

    # Convert distances into similarity-like scores (smaller distance => higher similarity)
    # We'll negate distances so higher is better
    import math, random

    ids = [c[0] for c in scored_candidates]
    raw_scores = [ -float(c[1]) for c in scored_candidates ]

    ordered = []
    if ids:
        # If temperature is exactly zero, use deterministic selection (best matches first)
        try:
            if temperature is not None and float(temperature) == 0.0:
                ids_sorted = sorted(ids, key=lambda x: distances.get(x, float('inf')))
                for cid in ids_sorted[:n_results]:
                    item = details_map.get(cid, {})
                    item['distance'] = distances.get(cid)
                    item['embedding_2d'] = proj_map.get(cid)
                    # Ensure album/album_artist is present
                    if 'album' not in item or not item['album']:
                        item['album'] = 'Unknown'
                    if 'album_artist' not in item or not item['album_artist']:
                        item['album_artist'] = 'Unknown'
                    ordered.append(item)
            else:
                # Softmax with temperature (temperature may be None or >0)
                # Divide by temperature to get logits (higher temp = flatter distribution)
                temps = [s / temperature for s in raw_scores]
                max_t = max(temps) if temps else 0.0
                exps = [math.exp(t - max_t) for t in temps]
                total = sum(exps)
                if total <= 0:
                    probs = [1.0 / len(exps)] * len(exps)
                else:
                    probs = [e / total for e in exps]

                # Log probability distribution stats to help debug temperature effect
                if probs:
                    max_prob = max(probs)
                    min_prob = min(probs)
                    mean_prob = sum(probs) / len(probs)
                    logger.info(f"Temperature={temperature}: Probability distribution - max={max_prob:.4f}, min={min_prob:.6f}, mean={mean_prob:.4f}, entropy={(- sum(p * math.log(p) if p > 0 else 0 for p in probs)):.3f}")


                # Weighted sampling without replacement to get n_results items (preserve projection/metadata)
                chosen = []
                avail_ids = ids.copy()
                avail_probs = probs.copy()
                k = min(n_results, len(avail_ids))
                for _ in range(k):
                    # Normalize
                    s = sum(avail_probs)
                    if s <= 0:
                        idx = random.randrange(len(avail_ids))
                    else:
                        r = random.random() * s
                        acc = 0.0
                        idx = 0
                        for j, p in enumerate(avail_probs):
                            acc += p
                            if r <= acc:
                                idx = j
                                break
                    chosen_id = avail_ids.pop(idx)
                    avail_probs.pop(idx)
                    chosen.append(chosen_id)

                # Build ordered results from chosen ids in the order selected
                for cid in chosen:
                    item = details_map.get(cid, {})
                    item['distance'] = distances.get(cid)
                    item['embedding_2d'] = proj_map.get(cid)
                    # Ensure album/album_artist is present
                    if 'album' not in item or not item['album']:
                        item['album'] = 'Unknown'
                    if 'album_artist' not in item or not item['album_artist']:
                        item['album_artist'] = 'Unknown'
                    ordered.append(item)
        except Exception as e:
            # Fallback deterministic ordering by best match
            logger.warning(f"Sampling failed, falling back to deterministic selection: {e}")
            ids_sorted = sorted(ids, key=lambda x: distances.get(x, float('inf')))
            for i in ids_sorted[:n_results]:
                item = details_map.get(i, {})
                item['distance'] = distances.get(i)
                item['embedding_2d'] = proj_map.get(i)
                # Ensure album/album_artist is present
                if 'album' not in item or not item['album']:
                    item['album'] = 'Unknown'
                if 'album_artist' not in item or not item['album_artist']:
                    item['album_artist'] = 'Unknown'
                ordered.append(item)

    # Prepare filtered_out details
    filtered_details = []
    if filtered_out:
        details_f = get_score_data_by_ids(filtered_out)
        details_f_map = {d['item_id']: d for d in details_f}
        for fid in filtered_out:
            if fid in details_f_map:
                fd = details_f_map[fid]
                fd['embedding_2d'] = proj_map.get(fid)
                # Ensure album/album_artist is present
                if 'album' not in fd or not fd['album']:
                    fd['album'] = 'Unknown'
                if 'album_artist' not in fd or not fd['album_artist']:
                    fd['album_artist'] = 'Unknown'
                filtered_details.append(fd)

    # Centroid projections
    centroid_2d = proj_map.get('__add_centroid__')
    subtract_centroid_2d = proj_map.get('__subtract_centroid__')

    # Attach 2D coords to add/sub selected items (songs and artist components)
    add_points = []
    for m in add_meta:
        if m.get('is_artist_component'):
            pid = f"__add_artist_comp__{m['item_id'].rsplit('_comp', 1)[0]}_{m['item_id'].split('_comp')[1]}"
            logger.debug(f"Looking for ADD artist component: item_id={m['item_id']}, pid={pid}, found={pid in proj_map}")
        elif m.get('is_playlist_component'):
            pid = f"__add_playlist__{m['item_id']}"
        elif m.get('type') == 'anchor':
            pid = f"__add_anchor__{m['item_id']}"
        elif m.get('type') == 'mood':
            pid = f"__add_mood__{m['item_id']}"
        else:
            pid = f"__add_id__{m['item_id']}"
        coord = proj_map.get(pid)
        add_points.append({**m, 'embedding_2d': coord})

    sub_points = []
    for m in sub_meta:
        if m.get('is_artist_component'):
            pid = f"__sub_artist_comp__{m['item_id'].rsplit('_comp', 1)[0]}_{m['item_id'].split('_comp')[1]}"
            logger.debug(f"Looking for SUB artist component: item_id={m['item_id']}, pid={pid}, found={pid in proj_map}")
        elif m.get('is_playlist_component'):
            pid = f"__sub_playlist__{m['item_id']}"
        elif m.get('type') == 'anchor':
            pid = f"__sub_anchor__{m['item_id']}"
        elif m.get('type') == 'mood':
            pid = f"__sub_mood__{m['item_id']}"
        else:
            pid = f"__sub_id__{m['item_id']}"
        coord = proj_map.get(pid)
        sub_points.append({**m, 'embedding_2d': coord})
    
    logger.info(f"Returning {len(add_points)} add_points and {len(sub_points)} sub_points")
    logger.info(f"add_points artist components: {sum(1 for p in add_points if p.get('is_artist_component'))}")
    logger.info(f"sub_points artist components: {sum(1 for p in sub_points if p.get('is_artist_component'))}")

    return {
        'results': ordered,
        'filtered_out': filtered_details,
        'centroid_2d': centroid_2d,
        'add_centroid_2d': centroid_2d,
        'subtract_centroid_2d': subtract_centroid_2d,
        'add_centroid_vector': add_centroid.tolist() if add_centroid is not None else None,
        'subtract_centroid_vector': subtract_centroid.tolist() if subtract_centroid is not None else None,
        'add_points': add_points,
        'sub_points': sub_points,
        'projection': projection_used,
    }

