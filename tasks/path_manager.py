# tasks/path_manager.py
import logging
import numpy as np
import random
import psycopg2 # Import psycopg2 to catch specific errors

# Imports from the project
from .voyager_manager import get_vector_by_id, find_nearest_neighbors_by_vector, find_nearest_neighbors_by_id
from config import (
    PATH_AVG_JUMP_SAMPLE_SIZE, PATH_CANDIDATES_PER_STEP, PATH_DEFAULT_LENGTH,
    PATH_DISTANCE_METRIC, VOYAGER_METRIC, PATH_LCORE_MULTIPLIER,
    DUPLICATE_DISTANCE_THRESHOLD_COSINE, DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN,
    DUPLICATE_DISTANCE_CHECK_LOOKBACK
)

logger = logging.getLogger(__name__)


def get_euclidean_distance(v1, v2):
    """Calculates the Euclidean distance between two vectors."""
    if v1 is not None and v2 is not None:
        return np.linalg.norm(v1 - v2)
    return float('inf')


def get_angular_distance(v1, v2):
    """Calculates the angular distance (derived from cosine similarity) between two vectors."""
    if v1 is not None and v2 is not None and np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
        # Normalize vectors to unit length
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        # Calculate cosine similarity, clipping to handle potential floating point inaccuracies
        cosine_similarity = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        # Angular distance is derived from the angle: arccos(similarity) / pi
        return np.arccos(cosine_similarity) / np.pi
    return float('inf')


def get_distance(v1, v2):
    """Calculates distance based on the configured metric."""
    if PATH_DISTANCE_METRIC == 'angular':
        return get_angular_distance(v1, v2)
    else: # Default to euclidean
        return get_euclidean_distance(v1, v2)


def interpolate_centroids(v1, v2, num, metric="euclidean"):
    """
    Generate interpolated centroid vectors between v1 and v2
    based on the chosen metric: 'euclidean' or 'angular'.
    """
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    
    if metric == "angular":
        # Normalize to unit vectors
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        
        # Compute the angle between them
        dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        theta = np.arccos(dot)
        
        # If vectors are almost identical, fallback to linear
        if np.isclose(theta, 0):
            return np.linspace(v1, v2, num=num)
        
        # Spherical linear interpolation (SLERP)
        t_vals = np.linspace(0, 1, num)
        centroids = []
        for t in t_vals:
            s1 = np.sin((1 - t) * theta) / np.sin(theta)
            s2 = np.sin(t * theta) / np.sin(theta)
            centroids.append(s1 * v1_u + s2 * v2_u)
        return np.array(centroids)
    
    else:
        # Default: Euclidean interpolation
        return np.linspace(v1, v2, num=num)


def _create_path_from_ids(path_ids):
    """Helper to fetch song details for a list of IDs and format the final path."""
    # Local import to prevent circular dependency
    from app_helper import get_tracks_by_ids
    if not path_ids:
        return []
    
    seen = set()
    unique_path_ids = [x for x in path_ids if not (x in seen or seen.add(x))]

    path_details = get_tracks_by_ids(unique_path_ids)
    details_map = {d['item_id']: d for d in path_details}
    
    ordered_path_details = [details_map[song_id] for song_id in unique_path_ids if song_id in details_map]
    return ordered_path_details


def _calculate_local_average_jump_distance(start_item_id, end_item_id, sample_size=PATH_AVG_JUMP_SAMPLE_SIZE):
    """
    Calculates the average distance by creating a chain of neighbors and measuring the
    distance between each step in the chain.
    """
    logger.info(f"Calculating chained average jump distance ({PATH_DISTANCE_METRIC}) for neighbors of {start_item_id} and {end_item_id}.")
    
    distances = []
    
    for item_id in [start_item_id, end_item_id]:
        try:
            neighbors = find_nearest_neighbors_by_id(item_id, n=sample_size, radius_similarity=None)
            if not neighbors:
                continue

            source_vector = get_vector_by_id(item_id)
            if source_vector is None:
                continue

            neighbor_vectors = [get_vector_by_id(n['item_id']) for n in neighbors]
            valid_neighbor_vectors = [v for v in neighbor_vectors if v is not None]
            vector_chain = [source_vector] + valid_neighbor_vectors

            for i in range(len(vector_chain) - 1):
                dist = get_distance(vector_chain[i], vector_chain[i+1])
                distances.append(dist)

        except Exception as e:
            logger.warning(f"Could not process neighbors for song {item_id} during chained jump calculation: {e}")

    if not distances:
        logger.error("No valid chained distances could be calculated from start/end songs.")
        return 0.1 # Return a sensible fallback default

    avg_dist = np.mean(distances)
    logger.info(f"Calculated chained average jump distance: {avg_dist:.4f} from {len(distances)} steps.")
    return avg_dist


def _normalize_signature(artist, title):
    """Creates a standardized, case-insensitive signature for a song."""
    artist_norm = (artist or "").strip().lower()
    title_norm = (title or "").strip().lower()
    return (artist_norm, title_norm)


def _find_best_unique_song(centroid_vec, used_song_ids, used_signatures, path_songs_details_so_far=None):
    """
    --- OPTIMIZED ---
    Finds the best song for a centroid that is not already used by ID, signature,
    or too close in distance to recently added songs in the path.
    The lookback window is configurable via DUPLICATE_DISTANCE_CHECK_LOOKBACK.
    """
    # Local import to prevent circular dependency
    from app_helper import get_score_data_by_ids

    threshold = DUPLICATE_DISTANCE_THRESHOLD_COSINE if PATH_DISTANCE_METRIC == 'angular' else DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN
    metric_name = 'Angular' if PATH_DISTANCE_METRIC == 'angular' else 'Euclidean'

    try:
        candidates_voyager = find_nearest_neighbors_by_vector(centroid_vec, n=PATH_CANDIDATES_PER_STEP * 3)
    except Exception as e:
        logger.error(f"Error finding neighbors for a centroid: {e}")
        return None

    if not candidates_voyager:
        return None

    candidate_ids = [c['item_id'] for c in candidates_voyager]
    candidate_details = get_score_data_by_ids(candidate_ids)
    details_map = {d['item_id']: d for d in candidate_details}

    for candidate in candidates_voyager:
        candidate_id = candidate['item_id']
        details = details_map.get(candidate_id)

        if not details:
            continue

        signature = _normalize_signature(details.get('author'), details.get('title'))

        if candidate_id in used_song_ids or signature in used_signatures:
            logger.info(f"Filtering song (NAME/ID FILTER): '{details.get('title')}' by '{details.get('author')}' as it is already in the path.")
            continue

        candidate_vector = get_vector_by_id(candidate_id)
        if candidate_vector is None:
            continue
        
        is_too_close = False
        # Only perform the distance check if lookback is enabled ( > 0)
        if DUPLICATE_DISTANCE_CHECK_LOOKBACK > 0 and path_songs_details_so_far:
            # Check against the last N songs added to the path for performance
            for prev_song_details in path_songs_details_so_far[-DUPLICATE_DISTANCE_CHECK_LOOKBACK:]:
                if 'vector' in prev_song_details:
                    distance_from_prev = get_distance(candidate_vector, prev_song_details['vector'])
                    if distance_from_prev < threshold:
                        logger.info(
                            f"Filtering song (DISTANCE FILTER) with {metric_name} distance: '{details.get('title')}' by '{details.get('author')}' "
                            f"due to direct distance of {distance_from_prev:.4f} from "
                            f"'{prev_song_details['title']}' by '{prev_song_details['author']}' (Threshold: {threshold})."
                        )
                        is_too_close = True
                        break # No need to check other previous songs
        
        if is_too_close:
            continue

        return {
            "item_id": candidate_id,
            "signature": signature,
            "vector": candidate_vector,
            "title": details.get('title'),
            "author": details.get('author')
        }

    return None


def find_path_between_songs(start_item_id, end_item_id, Lreq=PATH_DEFAULT_LENGTH):
    """
    Finds an adaptive path between two songs, ensuring exact length and no duplicate artist/title pairs.
    """
    from app_helper import get_db, get_score_data_by_ids, get_tracks_by_ids
    logger.info(f"Starting adaptive path generation from {start_item_id} to {end_item_id} with requested length {Lreq}.")

    start_vector = get_vector_by_id(start_item_id)
    end_vector = get_vector_by_id(end_item_id)
    start_details_list = get_score_data_by_ids([start_item_id])
    end_details_list = get_score_data_by_ids([end_item_id])

    if not all([start_vector is not None, end_vector is not None, start_details_list, end_details_list]):
        logger.error("Could not retrieve vectors or details for start or end song.")
        return None, 0.0
        
    start_details = start_details_list[0]
    end_details = end_details_list[0]

    used_song_ids = {start_item_id, end_item_id}
    used_signatures = {
        _normalize_signature(start_details.get('author'), start_details.get('title')),
        _normalize_signature(end_details.get('author'), end_details.get('title'))
    }

    delta_avg = _calculate_local_average_jump_distance(start_item_id, end_item_id)
    if delta_avg is None or delta_avg == 0:
        logger.error("Average jump distance (delta_avg) is not available or zero.")
        return None, 0.0

    D = get_distance(start_vector, end_vector)
    Lmax = int(np.floor(D / delta_avg)) + 1 if delta_avg > 0 else Lreq
    Lcore = min(Lreq, Lmax)
    Lcore = max(2, Lcore)
    Lcore *= PATH_LCORE_MULTIPLIER

    logger.info(f"Direct distance (D): {D:.4f}, Local avg jump (δ_avg): {delta_avg:.4f}")
    logger.info(f"Requested steps (L_req): {Lreq}, Max realistic (L_max): {Lmax}, Using core steps (L_core): {Lcore}")
    
    if Lcore < 2:
        return _create_path_from_ids([start_item_id, end_item_id]), D

    backbone_centroids = interpolate_centroids(start_vector, end_vector, num=Lcore, metric=PATH_DISTANCE_METRIC)
    
    path_songs_details = [{**start_details, 'vector': start_vector}]

    for i in range(1, Lcore - 1):
        centroid_vec = backbone_centroids[i]
        best_song = _find_best_unique_song(centroid_vec, used_song_ids, used_signatures, path_songs_details_so_far=path_songs_details)
        
        if best_song:
            path_songs_details.append(best_song)
            used_song_ids.add(best_song['item_id'])
            used_signatures.add(best_song['signature'])
        else:
            logger.warning(f"Could not find a unique song for backbone step {i+1}.")

    path_songs_details.append({**end_details, 'vector': end_vector})
    path_ids = [song['item_id'] for song in path_songs_details]

    if Lreq > len(path_ids):
        logger.info(f"Backbone path has {len(path_ids)} songs. Filling {Lreq - len(path_ids)} more to meet request.")
        
        final_path_details = []
        songs_to_add = Lreq - len(path_ids)
        num_segments = len(path_ids) - 1

        base_songs_per_segment = songs_to_add // num_segments if num_segments > 0 else 0
        remainder = songs_to_add % num_segments if num_segments > 0 else 0

        for i in range(num_segments):
            start_segment_id = path_ids[i]
            end_segment_id = path_ids[i+1]
            
            start_segment_details_list = get_tracks_by_ids([start_segment_id])
            if not start_segment_details_list: continue
            
            current_path_segment_details = [start_segment_details_list[0]]
            final_path_details.append(start_segment_details_list[0])


            num_to_insert = base_songs_per_segment + (1 if i < remainder else 0)
            if num_to_insert > 0:
                start_segment_vec = get_vector_by_id(start_segment_id)
                end_segment_vec = get_vector_by_id(end_segment_id)

                if start_segment_vec is not None and end_segment_vec is not None:
                    intermediate_centroids = interpolate_centroids(
                        start_segment_vec, end_segment_vec, num=num_to_insert + 2, metric=PATH_DISTANCE_METRIC
                    )[1:-1]
                    
                    for centroid_vec in intermediate_centroids:
                        # Pass the details of songs added in this segment for accurate filtering
                        best_song = _find_best_unique_song(centroid_vec, used_song_ids, used_signatures, path_songs_details_so_far=current_path_segment_details)
                        if best_song:
                            final_path_details.append(get_tracks_by_ids([best_song['item_id']])[0])
                            current_path_segment_details.append(best_song)
                            used_song_ids.add(best_song['item_id'])
                            used_signatures.add(best_song['signature'])
                        else:
                            logger.warning(f"Could not find a unique filler song for segment {i}.")
        
        final_path_details.append(get_tracks_by_ids([path_ids[-1]])[0])
        path_ids = [song['item_id'] for song in final_path_details]


    final_path_details = _create_path_from_ids(path_ids)
    
    total_path_distance = 0.0
    if len(final_path_details) > 1:
        path_vectors = [get_vector_by_id(song['item_id']) for song in final_path_details]
        for i in range(len(path_vectors) - 1):
            v1 = path_vectors[i]
            v2 = path_vectors[i+1]
            if v1 is not None and v2 is not None:
                total_path_distance += get_distance(v1, v2)

    return final_path_details, total_path_distance
