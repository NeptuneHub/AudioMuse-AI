import logging
from typing import List, Tuple
import numpy as np

from .voyager_manager import find_nearest_neighbors_by_vector, get_vector_by_id
from app_helper import get_score_data_by_ids
import config

logger = logging.getLogger(__name__)


def _compute_centroid_from_ids(ids: List[str]) -> np.ndarray:
    """Fetch vectors by id and compute their centroid (mean)."""
    vectors = []
    for item_id in ids:
        vec = get_vector_by_id(item_id)
        if vec is not None:
            vectors.append(np.array(vec, dtype=float))
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def song_alchemy(add_ids: List[str], subtract_ids: List[str], n_results: int = None, subtract_distance: float = None) -> List[dict]:
    """Perform Song Alchemy:
    - add_ids: items to include in positive centroid
    - subtract_ids: items to include in negative centroid
    - n_results: number of similar songs to fetch (default from config)

    Returns list of song detail dicts (using get_score_data_by_ids mapping)
    """
    if n_results is None:
        n_results = config.ALCHEMY_DEFAULT_N_RESULTS
    n_results = min(n_results, config.ALCHEMY_MAX_N_RESULTS)

    # Allow one or more songs in the ADD set — a single-song centroid is valid
    if not add_ids or len(add_ids) < 1:
        raise ValueError("At least one song must be in the ADD set")
    if len(add_ids) > 10 or len(subtract_ids) > 10:
        raise ValueError("Max 10 songs per ADD or SUBTRACT set")

    add_centroid = _compute_centroid_from_ids(add_ids)
    if add_centroid is None:
        return []

    subtract_centroid = None
    if subtract_ids:
        subtract_centroid = _compute_centroid_from_ids(subtract_ids)

    # Find nearest neighbors to add_centroid using Voyager
    neighbors = find_nearest_neighbors_by_vector(add_centroid, n=n_results * 3)
    if not neighbors:
        return []

    # neighbors is a list of dicts with item_id and score; keep candidate ids
    candidate_ids = [n['item_id'] for n in neighbors]

    # If subtract centroid present, filter candidates by distance
    if subtract_centroid is not None:
        # Compute distances and keep those farther than threshold
        filtered = []
        # Use provided override or default from config depending on metric
        if subtract_distance is None:
            if config.PATH_DISTANCE_METRIC == 'angular':
                threshold = config.ALCHEMY_SUBTRACT_DISTANCE_ANGULAR
            else:
                threshold = config.ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN
        else:
            threshold = subtract_distance
        for cid in candidate_ids:
            vec = get_vector_by_id(cid)
            if vec is None:
                continue
            # Use same metric as PATH: angular => cosine-derived; else euclidean
            if config.PATH_DISTANCE_METRIC == 'angular':
                # angular distance as in path_manager: arccos(cosine)/pi
                v1 = add_centroid / np.linalg.norm(add_centroid) if np.linalg.norm(add_centroid) > 0 else add_centroid
                v2 = np.array(vec, dtype=float) / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else np.array(vec, dtype=float)
                cosine = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angular_distance = np.arccos(cosine) / np.pi
                keep = angular_distance >= threshold
            else:
                # Euclidean
                dist = np.linalg.norm(add_centroid - np.array(vec, dtype=float))
                keep = dist >= threshold

            if keep:
                filtered.append(cid)

        candidate_ids = filtered

    # Trim to desired n_results
    candidate_ids = candidate_ids[:n_results]

    # Fetch details
    details = get_score_data_by_ids(candidate_ids)
    # Preserve order in candidate_ids
    details_map = {d['item_id']: d for d in details}
    ordered = [details_map[i] for i in candidate_ids if i in details_map]

    return ordered
