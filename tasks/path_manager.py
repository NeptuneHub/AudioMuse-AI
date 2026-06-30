import logging
import numpy as np

from .ivf_manager import get_vector_by_id, find_nearest_neighbors_by_vector, find_nearest_neighbors_by_id
from config import PATH_CANDIDATES_PER_STEP, PATH_DEFAULT_LENGTH, PATH_DISTANCE_METRIC, PATH_FIX_SIZE, DUPLICATE_DISTANCE_THRESHOLD_COSINE, DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN, DUPLICATE_DISTANCE_CHECK_LOOKBACK
from config import MAX_SONGS_PER_ARTIST

logger = logging.getLogger(__name__)


def get_euclidean_distance(v1, v2):
    if v1 is not None and v2 is not None:
        return np.linalg.norm(v1 - v2)
    return float('inf')


def get_angular_distance(v1, v2):
    if v1 is not None and v2 is not None and np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        cosine_similarity = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        return np.arccos(cosine_similarity) / np.pi
    return float('inf')


def get_distance(v1, v2, metric=None):
    if metric is None:
        metric = PATH_DISTANCE_METRIC
    if metric == 'angular':
        return get_angular_distance(v1, v2)
    else:
        return get_euclidean_distance(v1, v2)


def interpolate_centroids(v1, v2, num, metric="euclidean"):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    if metric == "angular":
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            logger.warning("Cannot perform angular interpolation with a zero vector. Falling back to linear.")
            return np.linspace(v1, v2, num=num)

        v1_u = v1 / norm_v1
        v2_u = v2 / norm_v2

        dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        theta = np.arccos(dot)

        if np.isclose(theta, 0) or np.isnan(theta):
            return np.linspace(v1, v2, num=num)

        t_vals = np.linspace(0, 1, num)
        centroids = []
        sin_theta = np.sin(theta)

        if np.isclose(sin_theta, 0):
            return np.linspace(v1, v2, num=num)

        for t in t_vals:
            s1 = np.sin((1 - t) * theta) / sin_theta
            s2 = np.sin(t * theta) / sin_theta
            magnitude = (1 - t) * norm_v1 + t * norm_v2
            centroids.append((s1 * v1_u + s2 * v2_u) * magnitude)
        return np.array(centroids)

    else:
        return np.linspace(v1, v2, num=num)


def _create_path_from_ids(path_ids):
    from app_helper import get_tracks_by_ids
    if not path_ids:
        return []

    seen = set()
    unique_path_ids = [x for x in path_ids if not (x in seen or seen.add(x))]

    path_details = get_tracks_by_ids(unique_path_ids)
    details_map = {d['item_id']: d for d in path_details}

    for song in details_map.values():
        album = song.get('album')
        if not album:
            album = song.get('album_name')
        song['album'] = album if album else 'Unknown'
        album_artist = song.get('album_artist')
        song['album_artist'] = album_artist if album_artist else 'Unknown'

    ordered_path_details = [details_map[song_id] for song_id in unique_path_ids if song_id in details_map]
    return ordered_path_details


def _normalize_signature(artist, title):
    artist_norm = (artist or "").strip().lower()
    title_norm = (title or "").strip().lower()
    return (artist_norm, title_norm)


def _find_best_songs_for_job(centroid_vec, used_song_ids, used_signatures, path_songs_details_so_far,
                             k_search=10, num_to_find=1, artist_counts=None,
                             get_vector_fn=get_vector_by_id, neighbors_fn=find_nearest_neighbors_by_vector,
                             metric=None, dup_threshold_cosine=None):
    from app_helper import get_score_data_by_ids

    if metric is None:
        metric = PATH_DISTANCE_METRIC
    if metric == 'angular':
        threshold = dup_threshold_cosine if dup_threshold_cosine is not None else DUPLICATE_DISTANCE_THRESHOLD_COSINE
    else:
        threshold = DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN
    metric_name = 'Angular' if metric == 'angular' else 'Euclidean'

    found_songs = []

    try:
        candidates_ivf = neighbors_fn(centroid_vec, n=k_search)
    except Exception as e:
        logger.exception(f"Error finding neighbors for a centroid with k={k_search}: {e}")
        return []

    if not candidates_ivf:
        logger.warning(f"No candidates found for centroid with k={k_search}.")
        return []

    candidate_ids = [c['item_id'] for c in candidates_ivf]
    candidate_details = get_score_data_by_ids(candidate_ids)
    details_map = {d['item_id']: d for d in candidate_details}

    for candidate in candidates_ivf:
        if len(found_songs) >= num_to_find:
            break

        candidate_id = candidate['item_id']

        if candidate_id in used_song_ids:
            continue

        details = details_map.get(candidate_id)
        if not details:
            continue

        signature = _normalize_signature(details.get('author'), details.get('title'))
        if signature in used_signatures:
            logger.debug(f"Filtering song (NAME/ID FILTER): '{details.get('title')}' by '{details.get('author')}' as it is already in the path.")
            continue

        author_norm = (details.get('author') or '').strip().lower()
        if artist_counts is not None and MAX_SONGS_PER_ARTIST is not None and MAX_SONGS_PER_ARTIST > 0:
            if artist_counts.get(author_norm, 0) >= MAX_SONGS_PER_ARTIST:
                logger.debug(f"Filtering song (ARTIST CAP) '{details.get('title')}' by '{details.get('author')}' because artist cap {MAX_SONGS_PER_ARTIST} reached.")
                continue

        candidate_vector = get_vector_fn(candidate_id)
        if candidate_vector is None:
            continue

        is_too_close = False

        if DUPLICATE_DISTANCE_CHECK_LOOKBACK > 0 and path_songs_details_so_far:
            for prev_song_details in path_songs_details_so_far[-DUPLICATE_DISTANCE_CHECK_LOOKBACK:]:
                if 'vector' in prev_song_details:
                    distance_from_prev = get_distance(candidate_vector, prev_song_details['vector'], metric=metric)
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

        if DUPLICATE_DISTANCE_CHECK_LOOKBACK > 0 and found_songs:
            for prev_song_details in found_songs[-DUPLICATE_DISTANCE_CHECK_LOOKBACK:]:
                if 'vector' in prev_song_details:
                    distance_from_prev = get_distance(candidate_vector, prev_song_details['vector'], metric=metric)
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

        found_songs.append({
            "item_id": candidate_id,
            "signature": signature,
            "vector": candidate_vector,
            "title": details.get('title'),
            "author": details.get('author')
        })

        used_song_ids.add(candidate_id)
        used_signatures.add(signature)
        if artist_counts is not None:
            artist_counts[author_norm] = artist_counts.get(author_norm, 0) + 1


    if len(found_songs) < num_to_find:
        logger.warning(f"Found only {len(found_songs)} of {num_to_find} songs for centroid (k={k_search}). Rolling back adds.")
        for song in found_songs:
            if song['item_id'] in used_song_ids:
                used_song_ids.remove(song['item_id'])
            if song['signature'] in used_signatures:
                used_signatures.remove(song['signature'])
            if artist_counts is not None:
                auth = (song.get('author') or '').strip().lower()
                if auth in artist_counts:
                    artist_counts[auth] = max(0, artist_counts.get(auth, 0) - 1)

        return []

    logger.info(f"Successfully found {len(found_songs)} of {num_to_find} songs for centroid (k={k_search}).")
    return found_songs


def find_path_between_songs(start_item_id, end_item_id, Lreq=PATH_DEFAULT_LENGTH, path_fix_size=PATH_FIX_SIZE,
                            get_vector_fn=get_vector_by_id, neighbors_fn=find_nearest_neighbors_by_vector,
                            neighbors_by_id_fn=find_nearest_neighbors_by_id, metric=None, dup_threshold_cosine=None):
    from app_helper import get_score_data_by_ids
    logger.info(f"Starting centroid path generation (with merge logic) from {start_item_id} to {end_item_id} with requested length {Lreq}.")

    if metric is None:
        metric = PATH_DISTANCE_METRIC

    if Lreq < 2:
        logger.warning(f"Requested path length {Lreq} is less than 2. Returning just start and end songs if different.")
        if start_item_id == end_item_id:
             path_details = _create_path_from_ids([start_item_id])
             return path_details, 0.0
        path_details = _create_path_from_ids([start_item_id, end_item_id])

        total_path_distance = 0.0
        v1 = get_vector_fn(start_item_id)
        v2 = get_vector_fn(end_item_id)
        if v1 is not None and v2 is not None:
            total_path_distance = get_distance(v1, v2, metric=metric)
        return path_details, total_path_distance

    start_vector = get_vector_fn(start_item_id)
    end_vector = get_vector_fn(end_item_id)
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

    artist_counts = {}
    start_author = (start_details.get('author') or '').strip().lower()
    end_author = (end_details.get('author') or '').strip().lower()
    if start_author:
        artist_counts[start_author] = artist_counts.get(start_author, 0) + 1
    if end_author:
        artist_counts[end_author] = artist_counts.get(end_author, 0) + 1

    path_songs_details = [{**start_details, 'vector': start_vector}]

    num_intermediate = Lreq - 2
    k_base = 10
    k_max = 1000

    if num_intermediate > 0:
        logger.info(f"Attempting to find {num_intermediate} intermediate songs for a total path of {Lreq}.")
        all_centroids = interpolate_centroids(start_vector, end_vector, num=Lreq, metric=metric)

        intermediate_centroids = all_centroids[1:-1]

        try:
            sample_n = max(10, int(PATH_CANDIDATES_PER_STEP))
        except Exception:
            sample_n = 50

        try:
            start_neighbors = neighbors_by_id_fn(start_item_id, n=sample_n) or []
            end_neighbors = neighbors_by_id_fn(end_item_id, n=sample_n) or []
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

        jobs = []

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
                    artist_counts=artist_counts,
                    get_vector_fn=get_vector_fn,
                    neighbors_fn=neighbors_fn,
                    metric=metric,
                    dup_threshold_cosine=dup_threshold_cosine
                )
                if found and len(found) > 0:
                    path_songs_details.extend(found)
        else:
            jobs = []
            if initial_count >= num_intermediate:
                for idx in range(num_intermediate):
                    jobs.append({'vector': intermediate_centroids[idx], 'k': k_base, 'original_indices': [idx], 'num_to_find': 1})
            else:
                group_size = float(num_intermediate) / float(initial_count)
                for j in range(initial_count):
                    start_idx = int(round(j * group_size))
                    end_idx = int(round((j + 1) * group_size)) - 1
                    if end_idx < start_idx:
                        end_idx = start_idx
                    start_idx = max(0, min(start_idx, num_intermediate - 1))
                    end_idx = max(0, min(end_idx, num_intermediate - 1))
                    indices = list(range(start_idx, end_idx + 1))
                    bucket_vecs = [intermediate_centroids[idx] for idx in indices]
                    bucket_mid = np.mean(bucket_vecs, axis=0)
                    num_to_find = len(indices)
                    scaled_k = min(k_max, max(k_base, int(k_base * (num_intermediate / float(initial_count)))))
                    jobs.append({'vector': bucket_mid, 'k': scaled_k, 'original_indices': indices, 'num_to_find': num_to_find})

        if path_fix_size and jobs:
            i = 0
            while i < len(jobs):
                job = jobs[i]

                found_songs = _find_best_songs_for_job(
                    job['vector'],
                    used_song_ids,
                    used_signatures,
                    path_songs_details,
                    k_search=job['k'],
                    num_to_find=job['num_to_find'],
                    artist_counts=artist_counts,
                    get_vector_fn=get_vector_fn,
                    neighbors_fn=neighbors_fn,
                    metric=metric,
                    dup_threshold_cosine=dup_threshold_cosine
                )

                num_found = len(found_songs)
                num_needed = job['num_to_find']

                if num_found == num_needed:
                    path_songs_details.extend(found_songs)
                    i += 1
                else:
                    num_missing = num_needed
                    logger.warning(f"Job {i} (k={job['k']}, needed {num_needed}) failed. Merging with next job.")

                    if i + 1 >= len(jobs):
                        logger.error(f"CRITICAL: Last centroid job failed (k={job['k']}) and cannot merge. Path will be short.")
                        i += 1
                    else:
                        job_a = jobs[i]
                        job_b = jobs.pop(i + 1)

                        idx_a_start = job_a['original_indices'][0]
                        idx_b_end = job_b['original_indices'][-1]

                        vec_a_orig = intermediate_centroids[idx_a_start]
                        vec_b_orig = intermediate_centroids[idx_b_end]

                        merged_vector = interpolate_centroids(vec_a_orig, vec_b_orig, num=3, metric=metric)[1]

                        merged_k = min(job_a['k'] + job_b['k'], k_max)

                        still_need_to_find = num_missing + job_b['num_to_find']

                        merged_indices = job_a['original_indices'] + job_b['original_indices']

                        job_a['vector'] = merged_vector
                        job_a['k'] = merged_k
                        job_a['original_indices'] = merged_indices
                        job_a['num_to_find'] = still_need_to_find

                        logger.info(f"Retrying merged job at index {i} (k={merged_k}, need to find {still_need_to_find} more songs, represents {len(merged_indices)} original centroids)")

    path_songs_details.append({**end_details, 'vector': end_vector})

    path_ids = [song['item_id'] for song in path_songs_details]

    final_path_details = _create_path_from_ids(path_ids)

    total_path_distance = 0.0
    if len(final_path_details) > 1:
        path_vectors = [get_vector_fn(song['item_id']) for song in final_path_details]
        for i in range(len(path_vectors) - 1):
            v1 = path_vectors[i]
            v2 = path_vectors[i+1]
            if v1 is not None and v2 is not None:
                total_path_distance += get_distance(v1, v2, metric=metric)

    if len(final_path_details) != Lreq:
         logger.warning(f"Final path length is {len(final_path_details)}, but {Lreq} was requested. This can happen if the last job fails to merge and find all songs.")
    else:
        logger.info(f"Successfully generated path with exact requested length of {Lreq}.")

    return final_path_details, total_path_distance

