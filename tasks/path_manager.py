# tasks/path_manager.py
import logging
import numpy as np
import random
import psycopg2 # Import psycopg2 to catch specific errors
import os
from concurrent.futures import ThreadPoolExecutor

# Imports from the project
from .voyager_manager import get_vector_by_id, find_nearest_neighbors_by_vector, find_nearest_neighbors_by_id
from config import (
    PATH_AVG_JUMP_SAMPLE_SIZE, PATH_CANDIDATES_PER_STEP, PATH_DEFAULT_LENGTH,
    PATH_DISTANCE_METRIC, VOYAGER_METRIC, PATH_LCORE_MULTIPLIER,
    PATH_FIX_SIZE,
    DUPLICATE_DISTANCE_THRESHOLD_COSINE, DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN,
    DUPLICATE_DISTANCE_CHECK_LOOKBACK
)
# Also import per-artist cap
from config import MAX_SONGS_PER_ARTIST

logger = logging.getLogger(__name__)

# small shared thread pool in case of parallel vector fetch needs
_PATH_THREAD_POOL = ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 1) - 1), thread_name_prefix="path")


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
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # Handle zero vectors
        if norm_v1 == 0 or norm_v2 == 0:
            logger.warning("Cannot perform angular interpolation with a zero vector. Falling back to linear.")
            return np.linspace(v1, v2, num=num)

        v1_u = v1 / norm_v1
        v2_u = v2 / norm_v2
        
        # Compute the angle between them
        dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        theta = np.arccos(dot)
        
        # If vectors are almost identical, fallback to linear
        if np.isclose(theta, 0) or np.isnan(theta):
            return np.linspace(v1, v2, num=num)
        
        # Spherical linear interpolation (SLERP)
        t_vals = np.linspace(0, 1, num)
        centroids = []
        sin_theta = np.sin(theta)
        
        # If sin_theta is close to 0 (vectors are collinear), use linear
        if np.isclose(sin_theta, 0):
            return np.linspace(v1, v2, num=num)
            
        for t in t_vals:
            s1 = np.sin((1 - t) * theta) / sin_theta
            s2 = np.sin(t * theta) / sin_theta
            # Interpolate magnitude linearly
            magnitude = (1 - t) * norm_v1 + t * norm_v2
            centroids.append((s1 * v1_u + s2 * v2_u) * magnitude)
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

    # Ensure album and album_artist fields are present in each song dict
    for song in details_map.values():
        # Try to get album from song dict, fallback to 'Unknown Album' if not found or empty
        album = song.get('album')
        if not album:
            # Try to get from other possible keys (e.g. 'album_name')
            album = song.get('album_name')
        song['album'] = album if album else 'Unknown'
        album_artist = song.get('album_artist')
        song['album_artist'] = album_artist if album_artist else 'Unknown'

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


def _find_best_songs_for_job(centroid_vec, used_song_ids, used_signatures, path_songs_details_so_far,
                             k_search=10, num_to_find=1, artist_counts=None):
    """
    Finds a specific number (`num_to_find`) of best songs for a centroid
    by searching k_search neighbors.
    Returns a list of found songs.
    If the list length is less than num_to_find, returns [] and rolls back internal state.
    """
    # Local import to prevent circular dependency
    from app_helper import get_score_data_by_ids

    threshold = DUPLICATE_DISTANCE_THRESHOLD_COSINE if PATH_DISTANCE_METRIC == 'angular' else DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN
    metric_name = 'Angular' if PATH_DISTANCE_METRIC == 'angular' else 'Euclidean'

    found_songs = []

    try:
        candidates_voyager = find_nearest_neighbors_by_vector(centroid_vec, n=k_search)
    except Exception as e:
        logger.error(f"Error finding neighbors for a centroid with k={k_search}: {e}")
        return [] # Failed to search

    if not candidates_voyager:
        logger.warning(f"No candidates found for centroid with k={k_search}.")
        return []

    candidate_ids = [c['item_id'] for c in candidates_voyager]
    candidate_details = get_score_data_by_ids(candidate_ids)
    details_map = {d['item_id']: d for d in candidate_details}

    # Scan in order (nearest first) and collect acceptable candidates
    for candidate in candidates_voyager:
        if len(found_songs) >= num_to_find:
            break # We found all the songs we needed for this job

        candidate_id = candidate['item_id']

        # Check if this is a duplicate by ID
        if candidate_id in used_song_ids:
            continue # Skip if already used by ID

        details = details_map.get(candidate_id)
        if not details:
            continue # Skip if no details found

        # Check if this is a duplicate by Artist/Title signature
        signature = _normalize_signature(details.get('author'), details.get('title'))
        if signature in used_signatures:
            logger.debug(f"Filtering song (NAME/ID FILTER): '{details.get('title')}' by '{details.get('author')}' as it is already in the path.")
            continue

        # Enforce global per-artist cap if configured. Treat MAX_SONGS_PER_ARTIST <= 0 as DISABLED.
        author_norm = (details.get('author') or '').strip().lower()
        if artist_counts is not None and MAX_SONGS_PER_ARTIST is not None and MAX_SONGS_PER_ARTIST > 0:
            if artist_counts.get(author_norm, 0) >= MAX_SONGS_PER_ARTIST:
                logger.debug(f"Filtering song (ARTIST CAP) '{details.get('title')}' by '{details.get('author')}' because artist cap {MAX_SONGS_PER_ARTIST} reached.")
                continue

        candidate_vector = get_vector_by_id(candidate_id)
        if candidate_vector is None:
            continue # Skip if vector is missing

        is_too_close = False

        # Check against songs *already added to the main path*
        if DUPLICATE_DISTANCE_CHECK_LOOKBACK > 0 and path_songs_details_so_far:
            for prev_song_details in path_songs_details_so_far[-DUPLICATE_DISTANCE_CHECK_LOOKBACK:]:
                if 'vector' in prev_song_details:
                    distance_from_prev = get_distance(candidate_vector, prev_song_details['vector'])
                    if distance_from_prev < threshold:
                        logger.debug(
                            f"Filtering song (DISTANCE FILTER) with {metric_name} distance: '{details.get('title')}' by '{details.get('author')}' "
                            f"due to direct distance of {distance_from_prev:.4f} from "
                            f"'{prev_song_details['title']}' by '{prev_song_details['author']}' (Threshold: {threshold})."
                        )
                        is_too_close = True
                        break
            if is_too_close:
                continue

        # Also check against songs found *within this same job*
        if DUPLICATE_DISTANCE_CHECK_LOOKBACK > 0 and found_songs:
            for prev_song_details in found_songs[-DUPLICATE_DISTANCE_CHECK_LOOKBACK:]:
                if 'vector' in prev_song_details:
                    distance_from_prev = get_distance(candidate_vector, prev_song_details['vector'])
                    if distance_from_prev < threshold:
                        logger.debug(
                            f"Filtering song (INTERNAL JOB DISTANCE FILTER) with {metric_name} distance: '{details.get('title')}' by '{details.get('author')}' "
                            f"due to direct distance of {distance_from_prev:.4f} from "
                            f"'{prev_song_details['title']}' by '{prev_song_details['author']}' (Threshold: {threshold})."
                        )
                        is_too_close = True
                        break
            if is_too_close:
                continue

        # If we get here, the song is acceptable
        found_songs.append({
            "item_id": candidate_id,
            "signature": signature,
            "vector": candidate_vector,
            "title": details.get('title'),
            "author": details.get('author')
        })

        # IMPORTANT: Add to used_song_ids and used_signatures immediately
        # so they are not picked again by this same job.
        used_song_ids.add(candidate_id)
        used_signatures.add(signature)
        # Increment artist count
        if artist_counts is not None:
            artist_counts[author_norm] = artist_counts.get(author_norm, 0) + 1


    if len(found_songs) < num_to_find:
        # We failed to find enough songs.
        # We must "roll back" the IDs/signatures we added *within this job*
        # so that a future, more powerful merged job can find them.
        logger.warning(f"Found only {len(found_songs)} of {num_to_find} songs for centroid (k={k_search}). Rolling back adds.")
        for song in found_songs:
            # Add a safety check to prevent the KeyError
            if song['item_id'] in used_song_ids:
                used_song_ids.remove(song['item_id'])
            if song['signature'] in used_signatures:
                used_signatures.remove(song['signature'])
            # Decrement artist count for rolled-back songs
            if artist_counts is not None:
                auth = (song.get('author') or '').strip().lower()
                if auth in artist_counts:
                    artist_counts[auth] = max(0, artist_counts.get(auth, 0) - 1)
        
        # Return an empty list to signal complete failure of this job
        return []
        
    # Success! Return the full list.
    logger.info(f"Successfully found {len(found_songs)} of {num_to_find} songs for centroid (k={k_search}).")
    # We already added them to used_ids/signatures inside the loop.
    return found_songs


def find_path_between_songs(start_item_id, end_item_id, Lreq=PATH_DEFAULT_LENGTH, path_fix_size=PATH_FIX_SIZE):
    """
    Finds a path between two songs using linear interpolation of centroids
    and a centroid-merging strategy on failure, ensuring exact path length.
    
    The final path length *will be* Lreq, unless Lreq < 2 or a
    final merge fails catastrophically.
    """
    # Local import to prevent circular dependency
    from app_helper import get_db, get_score_data_by_ids, get_tracks_by_ids
    logger.info(f"Starting centroid path generation (with merge logic) from {start_item_id} to {end_item_id} with requested length {Lreq}.")

    if Lreq < 2:
        logger.warning(f"Requested path length {Lreq} is less than 2. Returning just start and end songs if different.")
        if start_item_id == end_item_id:
             path_details = _create_path_from_ids([start_item_id])
             return path_details, 0.0
        path_details = _create_path_from_ids([start_item_id, end_item_id])
        
        # Calculate distance for the 2-song path
        total_path_distance = 0.0
        v1 = get_vector_by_id(start_item_id)
        v2 = get_vector_by_id(end_item_id)
        if v1 is not None and v2 is not None:
            total_path_distance = get_distance(v1, v2)
        return path_details, total_path_distance

    start_vector = get_vector_by_id(start_item_id)
    end_vector = get_vector_by_id(end_item_id)
    start_details_list = get_score_data_by_ids([start_item_id])
    end_details_list = get_score_data_by_ids([end_item_id])

    if not all([start_vector is not None, end_vector is not None, start_details_list, end_details_list]):
        logger.error("Could not retrieve vectors or details for start or end song.")
        return None, 0.0
        
    start_details = start_details_list[0]
    end_details = end_details_list[0]

    # Initialize trackers with *both* start and end songs to avoid duplication
    used_song_ids = {start_item_id, end_item_id}
    used_signatures = {
        _normalize_signature(start_details.get('author'), start_details.get('title')),
        _normalize_signature(end_details.get('author'), end_details.get('title'))
    }

    # Track per-artist counts so we can enforce MAX_SONGS_PER_ARTIST across the path
    artist_counts = {}
    start_author = (start_details.get('author') or '').strip().lower()
    end_author = (end_details.get('author') or '').strip().lower()
    if start_author:
        artist_counts[start_author] = artist_counts.get(start_author, 0) + 1
    if end_author:
        artist_counts[end_author] = artist_counts.get(end_author, 0) + 1

    # This list holds the start song + all *found* intermediate songs
    path_songs_details = [{**start_details, 'vector': start_vector}]
    
    num_intermediate = Lreq - 2
    k_base = 10
    k_max = 1000 # Max search neighbors
    
    if num_intermediate > 0:
        logger.info(f"Attempting to find {num_intermediate} intermediate songs for a total path of {Lreq}.")
        # Generate Lreq total centroids (start, ...intermediate..., end)
        all_centroids = interpolate_centroids(start_vector, end_vector, num=Lreq, metric=PATH_DISTANCE_METRIC)
        
        # We only want the intermediate ones
        intermediate_centroids = all_centroids[1:-1] 

        # Heuristic to choose initial number of centroid jobs:
        # - Sample PATH_CANDIDATES_PER_STEP neighbors near start and end
        # - Compute intersection/union counts and pick a representative count
        # - initial_count = max(1, min(num_intermediate, representative // 2))
        # This avoids creating many centroids when the local neighborhood is small.
        try:
            sample_n = max(10, int(PATH_CANDIDATES_PER_STEP))
        except Exception:
            sample_n = 50

        try:
            start_neighbors = find_nearest_neighbors_by_id(start_item_id, n=sample_n) or []
            end_neighbors = find_nearest_neighbors_by_id(end_item_id, n=sample_n) or []
            start_ids = {n['item_id'] for n in start_neighbors}
            end_ids = {n['item_id'] for n in end_neighbors}
            intersection_size = len(start_ids & end_ids)
            union_size = len(start_ids | end_ids)
        except Exception as e:
            logger.debug(f"Heuristic neighbor sampling failed: {e}")
            intersection_size = 0
            union_size = 0

        representative = intersection_size if intersection_size > 0 else union_size
        initial_count = int(max(1, min(num_intermediate, representative // 2 if representative > 0 else num_intermediate)))
        logger.info(f"Centroid heuristic: sampled {sample_n} neighbors each side -> intersection={intersection_size}, union={union_size}. Using initial centroid count {initial_count} (requested intermediate {num_intermediate}).")

        # Prepare jobs variable; if path_fix_size is False we won't populate it and
        # will skip the merge/retry logic below.
        jobs = []

        # If path_fix_size is False, skip the centroid-merging approach and do
        # a single-pass per-centroid nearest-neighbor pick. This may result in
        # a shorter path if some centroids produce no acceptable candidate.
        if not path_fix_size:
            logger.info("PATH_FIX_SIZE disabled: using single-pass centroid picks (no merging). Path may be shorter than requested.")
            for centroid in intermediate_centroids:
                found = _find_best_songs_for_job(
                    centroid,
                    used_song_ids,
                    used_signatures,
                    path_songs_details,
                    k_search=k_base,
                    num_to_find=1,
                    artist_counts=artist_counts
                )
                if found and len(found) > 0:
                    path_songs_details.extend(found)
            # Skip merging logic entirely
        else:
            # Group original intermediate centroids into `initial_count` buckets and create jobs per bucket.
            jobs = []
            if initial_count >= num_intermediate:
                # One job per original centroid
                for idx in range(num_intermediate):
                    jobs.append({'vector': intermediate_centroids[idx], 'k': k_base, 'original_indices': [idx], 'num_to_find': 1})
            else:
                # Bucket centroids evenly
                group_size = float(num_intermediate) / float(initial_count)
                for j in range(initial_count):
                    start_idx = int(round(j * group_size))
                    end_idx = int(round((j + 1) * group_size)) - 1
                    if end_idx < start_idx:
                        end_idx = start_idx
                    # Clamp to bounds
                    start_idx = max(0, min(start_idx, num_intermediate - 1))
                    end_idx = max(0, min(end_idx, num_intermediate - 1))
                    indices = list(range(start_idx, end_idx + 1))
                    # Average vectors for the bucket to produce a representative centroid
                    bucket_vecs = [intermediate_centroids[idx] for idx in indices]
                    bucket_mid = np.mean(bucket_vecs, axis=0)
                    # Determine how many songs to find for this bucket (sum of originals)
                    num_to_find = len(indices)
                    # Scale k proportionally so total search budget roughly preserved
                    scaled_k = min(k_max, max(k_base, int(k_base * (num_intermediate / float(initial_count)))))
                    jobs.append({'vector': bucket_mid, 'k': scaled_k, 'original_indices': indices, 'num_to_find': num_to_find})
        
        # Process the jobs with the merge logic (only if we actually created jobs)
        if path_fix_size and jobs:
            i = 0
            while i < len(jobs):
                job = jobs[i]

                # Pass the *current* path details for distance checking
                found_songs = _find_best_songs_for_job(
                    job['vector'],
                    used_song_ids,
                    used_signatures,
                    path_songs_details, # Pass the list of songs found so far
                    k_search=job['k'],
                    num_to_find=job['num_to_find'],
                    artist_counts=artist_counts
                )

                num_found = len(found_songs)
                num_needed = job['num_to_find']

                if num_found == num_needed:
                    # Success! Add songs to path and move to the next job
                    path_songs_details.extend(found_songs)
                    # Note: used_song_ids was already updated inside _find_best_songs_for_job
                    i += 1
                else:
                    # Failure. num_found is 0 because _find_best_songs_for_job rolled back.
                    num_missing = num_needed # Since we found 0
                    logger.warning(f"Job {i} (k={job['k']}, needed {num_needed}) failed. Merging with next job.")

                    if i + 1 >= len(jobs):
                        # This was the *last* job in the list. It failed.
                        logger.error(f"CRITICAL: Last centroid job failed (k={job['k']}) and cannot merge. Path will be short.")
                        i += 1 # End the loop
                    else:
                        # Merge job[i] and job[i+1]
                        job_a = jobs[i]
                        job_b = jobs.pop(i + 1) # Remove job_b from list

                        # We need the *original* vectors to calculate the new midpoint
                        idx_a_start = job_a['original_indices'][0]
                        idx_b_end = job_b['original_indices'][-1]

                        vec_a_orig = intermediate_centroids[idx_a_start]
                        vec_b_orig = intermediate_centroids[idx_b_end]

                        # Create new merged vector (midpoint of the two *original* vectors)
                        merged_vector = interpolate_centroids(vec_a_orig, vec_b_orig, num=3, metric=PATH_DISTANCE_METRIC)[1]

                        # Sum k, cap at k_max
                        merged_k = min(job_a['k'] + job_b['k'], k_max)

                        # Sum how many songs we *still* need to find
                        still_need_to_find = num_missing + job_b['num_to_find']

                        merged_indices = job_a['original_indices'] + job_b['original_indices']

                        # Update job[i] in-place with new merged values
                        job_a['vector'] = merged_vector
                        job_a['k'] = merged_k
                        job_a['original_indices'] = merged_indices
                        job_a['num_to_find'] = still_need_to_find

                        logger.info(f"Retrying merged job at index {i} (k={merged_k}, need to find {still_need_to_find} more songs, represents {len(merged_indices)} original centroids)")
                        # Do not increment i. We re-try the *current* job (which is now the merged job)
    
    # Add the end song (it was already in used_song_ids)
    path_songs_details.append({**end_details, 'vector': end_vector})
    
    # Get final list of IDs
    path_ids = [song['item_id'] for song in path_songs_details]
    
    final_path_details = _create_path_from_ids(path_ids)
    
    # Recalculate total distance
    total_path_distance = 0.0
    if len(final_path_details) > 1:
        # We must re-fetch vectors because _create_path_from_ids doesn't return them
        path_vectors = [get_vector_by_id(song['item_id']) for song in final_path_details]
        for i in range(len(path_vectors) - 1):
            v1 = path_vectors[i]
            v2 = path_vectors[i+1]
            if v1 is not None and v2 is not None:
                total_path_distance += get_distance(v1, v2)

    # Log if the final length is not what was requested
    if len(final_path_details) != Lreq:
         logger.warning(f"Final path length is {len(final_path_details)}, but {Lreq} was requested. This can happen if the last job fails to merge and find all songs.")
    else:
        logger.info(f"Successfully generated path with exact requested length of {Lreq}.")

    return final_path_details, total_path_distance

