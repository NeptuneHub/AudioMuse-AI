# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Post-processing filters applied to a clustering result before it becomes playlists.

Cleans up the best clustering result chosen by tasks.clustering: it strips
duplicate and too-close tracks, drops tiny playlists, and selects a diverse
top-N. The distance and regex work that used to live in clustering.py was moved
here so the orchestrator stays focused on the search.

Main Features:
* apply_distance_filtering_direct / apply_title_artist_deduplication: remove
  near-duplicate vectors and same title/artist repeats within each playlist.
* apply_minimum_size_filter_to_clustering_result: drop playlists below a size floor.
* select_diverse_playlists_with_genre_coverage: select a strict Top-N: two
  maximally separated playlists for each of the three most represented genres,
  then four mutually different non-top genres by max-min centroid distance.
"""

import logging
import numpy as np
import re
import secrets
from scipy.spatial.distance import cdist
from psycopg2.extras import DictCursor


logger = logging.getLogger(__name__)


def get_vectors_from_database(item_ids: list, db_conn):
    vectors_map = {}

    with db_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("SELECT item_id, embedding FROM embedding WHERE item_id = ANY(%s)", (item_ids,))
        rows = cur.fetchall()

        for row in rows:
            if row['embedding']:
                try:
                    vector = np.frombuffer(row['embedding'], dtype=np.float32)
                    vectors_map[row['item_id']] = vector
                except Exception as e:
                    logger.warning(f"Failed to decode embedding for {row['item_id']}: {e}")

    return vectors_map


def apply_distance_filtering_direct(song_results: list, db_conn, log_prefix=""):
    from config import (
        DUPLICATE_DISTANCE_CHECK_LOOKBACK,
        DUPLICATE_DISTANCE_THRESHOLD_COSINE,
        DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN,
        IVF_METRIC,
    )

    if DUPLICATE_DISTANCE_CHECK_LOOKBACK <= 0:
        return song_results

    if not song_results:
        return []

    item_ids = [s['item_id'] for s in song_results]
    vectors_map = get_vectors_from_database(item_ids, db_conn)

    logger.debug(
        f"{log_prefix}Vector availability: {len(vectors_map)}/{len(item_ids)} songs have embedding vectors"
    )
    if len(vectors_map) < len(item_ids):
        missing_vectors = len(item_ids) - len(vectors_map)
        logger.debug(
            f"{log_prefix}WARNING: {missing_vectors} songs missing embedding vectors, they will be kept without distance checking"
        )

    if not vectors_map:
        logger.info(
            f"{log_prefix}No embedding vectors found, falling back to title/artist deduplication"
        )
        return apply_title_artist_deduplication(song_results, db_conn, log_prefix)

    details_map = {}
    with db_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("SELECT item_id, title, author FROM score WHERE item_id = ANY(%s)", (item_ids,))
        rows = cur.fetchall()
        for row in rows:
            details_map[row['item_id']] = {'title': row['title'], 'author': row['author']}

    threshold = (
        DUPLICATE_DISTANCE_THRESHOLD_COSINE
        if IVF_METRIC == 'angular'
        else DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN
    )
    metric_name = 'Angular' if IVF_METRIC == 'angular' else 'Euclidean'

    filtered_songs = []
    distance_filtered_count = 0

    logger.debug(
        f"{log_prefix}Starting distance filtering with threshold {threshold:.4f} ({metric_name}), lookback window: {DUPLICATE_DISTANCE_CHECK_LOOKBACK}"
    )

    total_comparisons = 0
    distances_calculated = []

    for current_song in song_results:
        current_vector = vectors_map.get(current_song['item_id'])
        if current_vector is None:
            logger.debug(f"{log_prefix}No vector found for {current_song['item_id']}, keeping song")
            filtered_songs.append(current_song)
            continue

        is_too_close = False
        min_distance = float('inf')
        closest_song = None

        lookback_window = filtered_songs[-DUPLICATE_DISTANCE_CHECK_LOOKBACK:]
        for recent_song in lookback_window:
            recent_vector = vectors_map.get(recent_song['item_id'])
            if recent_vector is None:
                continue

            total_comparisons += 1

            if IVF_METRIC == 'angular':
                if np.linalg.norm(current_vector) > 0 and np.linalg.norm(recent_vector) > 0:
                    v1_u = current_vector / np.linalg.norm(current_vector)
                    v2_u = recent_vector / np.linalg.norm(recent_vector)
                    cosine_similarity = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
                    direct_dist = np.arccos(cosine_similarity) / np.pi
                else:
                    direct_dist = float('inf')
            else:
                direct_dist = np.linalg.norm(current_vector - recent_vector)

            if direct_dist != float('inf'):
                distances_calculated.append(direct_dist)

            if direct_dist < min_distance:
                min_distance = direct_dist
                closest_song = recent_song

            if direct_dist < threshold:
                current_details = details_map.get(
                    current_song['item_id'], {'title': 'N/A', 'author': 'N/A'}
                )
                recent_details = details_map.get(
                    recent_song['item_id'], {'title': 'N/A', 'author': 'N/A'}
                )
                logger.info(
                    f"{log_prefix}FILTERED OUT: '{current_details['title']}' by '{current_details['author']}' "
                    f"({metric_name} distance {direct_dist:.4f} < {threshold:.4f}) too close to "
                    f"'{recent_details['title']}' by '{recent_details['author']}'"
                )
                is_too_close = True
                distance_filtered_count += 1
                break

        if not is_too_close:
            filtered_songs.append(current_song)
            if len(filtered_songs) <= 5 or len(filtered_songs) % 10 == 0:
                current_details = details_map.get(
                    current_song['item_id'], {'title': 'N/A', 'author': 'N/A'}
                )
                if closest_song and min_distance != float('inf'):
                    closest_details = details_map.get(
                        closest_song['item_id'], {'title': 'N/A', 'author': 'N/A'}
                    )
                    logger.debug(
                        f"{log_prefix}KEPT: '{current_details['title']}' by '{current_details['author']}' "
                        f"(min distance {min_distance:.4f} to '{closest_details['title']}' by '{closest_details['author']}')"
                    )
                else:
                    logger.debug(
                        f"{log_prefix}KEPT: '{current_details['title']}' by '{current_details['author']}' (first song or no close songs)"
                    )

    if distances_calculated:
        min_dist = min(distances_calculated)
        max_dist = max(distances_calculated)
        avg_dist = sum(distances_calculated) / len(distances_calculated)
        distances_below_threshold = [d for d in distances_calculated if d < threshold]
        logger.debug(
            f"{log_prefix}Distance statistics: {total_comparisons} comparisons, min={min_dist:.4f}, max={max_dist:.4f}, avg={avg_dist:.4f}, threshold={threshold:.4f}"
        )
        logger.debug(
            f"{log_prefix}Distances below threshold: {len(distances_below_threshold)} out of {len(distances_calculated)} ({len(distances_below_threshold) / len(distances_calculated) * 100:.1f}%)"
        )
        if distances_below_threshold and distance_filtered_count == 0:
            logger.warning(
                f"{log_prefix}WARNING: Found {len(distances_below_threshold)} distances below threshold but filtered 0 songs - possible logic error!"
            )
    else:
        logger.debug(
            f"{log_prefix}No valid distance calculations performed (no vectors or no comparisons)"
        )

    logger.info(
        f"{log_prefix}Distance filtering complete: {len(song_results)} -> {len(filtered_songs)} songs (removed {distance_filtered_count} duplicates)"
    )
    return filtered_songs


def apply_title_artist_deduplication(song_results: list, db_conn, log_prefix=""):
    if not song_results:
        return []

    item_ids = [s['item_id'] for s in song_results]
    details_map = {}

    with db_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("SELECT item_id, title, author FROM score WHERE item_id = ANY(%s)", (item_ids,))
        rows = cur.fetchall()
        for row in rows:
            details_map[row['item_id']] = {'title': row['title'], 'author': row['author']}

    seen_combinations = set()
    filtered_songs = []
    title_filtered_count = 0

    for song in song_results:
        song_details = details_map.get(song['item_id'])
        if not song_details:
            logger.debug(f"{log_prefix}No details found for {song['item_id']}, skipping")
            continue

        title_raw = song_details['title'] if song_details['title'] else ""
        artist_raw = song_details['author'] if song_details['author'] else ""

        title_clean = title_raw.lower().strip()
        title_clean = re.sub(
            r'\s*\((?=[^)]*(?:remaster|explicit|clean|radio|edit|version|mix))[^)]*\)',
            '',
            title_clean,
            flags=re.IGNORECASE,
        )
        title_clean = re.sub(
            r'\s*\[(?=[^\]]*(?:remaster|explicit|clean|radio|edit|version|mix))[^\]]*\]',
            '',
            title_clean,
            flags=re.IGNORECASE,
        )
        title_clean = re.sub(
            r'\s*-\s*(?:remaster|explicit|clean|radio|edit|version|mix).*',
            '',
            title_clean,
            flags=re.IGNORECASE,
        )
        title_clean = title_clean.strip()

        artist_clean = artist_raw.lower().strip()
        combination = (title_clean, artist_clean)

        if combination not in seen_combinations:
            seen_combinations.add(combination)
            filtered_songs.append(song)
            if len(filtered_songs) <= 5:
                if title_clean != title_raw.lower().strip():
                    logger.debug(
                        f"{log_prefix}KEPT (cleaned): '{title_raw}' -> '{title_clean}' by '{artist_raw}'"
                    )
                else:
                    logger.debug(
                        f"{log_prefix}KEPT: '{song_details['title']}' by '{song_details['author']}'"
                    )
        else:
            title_filtered_count += 1
            logger.info(
                f"{log_prefix}REMOVED duplicate: '{title_raw}' by '{artist_raw}' (normalized to '{title_clean}' by '{artist_clean}')"
            )

    logger.info(
        f"{log_prefix}Title/artist deduplication: {len(song_results)} -> {len(filtered_songs)} songs (removed {title_filtered_count} duplicates)"
    )
    return filtered_songs


def _dedupe_single_playlist(playlist_name, songs_list, db_conn, log_prefix):
    songs_sorted_by_title = sorted(
        songs_list, key=lambda song: song[1].lower() if song[1] else ""
    )
    logger.info(
        f"{log_prefix}SORTED {len(songs_sorted_by_title)} songs BY TITLE in playlist '{playlist_name}'"
    )
    logger.info(
        f"{log_prefix}SORTED ORDER - First 5 titles: {[song[1] for song in songs_sorted_by_title[:5]]}"
    )

    song_results = [
        {"item_id": item_id} for item_id, title, author in songs_sorted_by_title
    ]

    logger.debug(
        f"{log_prefix}Filtering playlist '{playlist_name}' with {len(song_results)} songs"
    )
    logger.debug(
        f"{log_prefix}Applying combined duplicate filtering for playlist '{playlist_name}' on SORTED songs"
    )

    temp_filtered = apply_title_artist_deduplication(
        song_results, db_conn, log_prefix + "[TitleArtist] "
    )
    filtered_song_results = apply_distance_filtering_direct(
        temp_filtered, db_conn, log_prefix + "[Distance] "
    )

    filtered_item_ids = {s["item_id"] for s in filtered_song_results}
    filtered_songs = [
        song for song in songs_sorted_by_title if song[0] in filtered_item_ids
    ]

    logger.debug(
        f"{log_prefix}Filtering complete, now have {len(filtered_songs)} songs in ALPHABETICAL order"
    )
    return filtered_songs


def _restrict_result_metadata(new_result, best_result, kept_names):
    for key in (
        "playlist_centroids",
        "playlist_to_centroid_vector_map",
        "playlist_primary_genres",
    ):
        if key in best_result:
            new_result[key] = {
                name: value
                for name, value in best_result[key].items()
                if name in kept_names
            }


def apply_duplicate_filtering_to_clustering_result(best_result, log_prefix=""):
    try:
        from app_helper import get_db

        if not best_result or not best_result.get("named_playlists"):
            logger.warning(
                f"{log_prefix}No playlists found in best_result, skipping duplicate filtering"
            )
            return best_result

        logger.info(f"{log_prefix}Applying duplicate filtering to clustering playlists...")

        db_conn = get_db()
        original_playlists = best_result["named_playlists"]
        filtered_playlists = {}
        total_songs_before = 0
        total_songs_after = 0

        logger.info(
            f"{log_prefix}Processing {len(original_playlists)} playlists for duplicate filtering"
        )

        logger.info(
            f"{log_prefix}Using database-based vector distance filtering for duplicate detection"
        )

        for playlist_name, songs_list in original_playlists.items():
            total_songs_before += len(songs_list)

            if not songs_list:
                logger.debug(f"{log_prefix}Skipping empty playlist '{playlist_name}'")
                filtered_playlists[playlist_name] = songs_list
                continue

            try:
                filtered_songs = _dedupe_single_playlist(
                    playlist_name, songs_list, db_conn, log_prefix
                )
                filtered_playlists[playlist_name] = filtered_songs
                total_songs_after += len(filtered_songs)

                if len(filtered_songs) != len(songs_list):
                    logger.info(
                        f"{log_prefix}Playlist '{playlist_name}': filtered {len(songs_list)} -> {len(filtered_songs)} songs"
                    )
                else:
                    logger.debug(
                        f"{log_prefix}Playlist '{playlist_name}': no songs filtered ({len(songs_list)} songs)"
                    )

            except Exception:
                logger.exception(
                    f"{log_prefix}Error filtering playlist '{playlist_name}'. Keeping original playlist."
                )
                filtered_playlists[playlist_name] = songs_list
                total_songs_after += len(songs_list)

        new_result = best_result.copy()
        new_result["named_playlists"] = filtered_playlists
        _restrict_result_metadata(new_result, best_result, filtered_playlists)

        logger.info(
            f"{log_prefix}Duplicate filtering complete: {total_songs_before} -> {total_songs_after} songs total across {len(filtered_playlists)} playlists"
        )

        return new_result

    except Exception:
        logger.exception(
            f"{log_prefix}Critical error in duplicate filtering. Returning original result."
        )
        return best_result


def apply_minimum_size_filter_to_clustering_result(best_result, min_size=20, log_prefix=""):
    try:
        if not best_result or not best_result.get("named_playlists"):
            logger.warning(
                f"{log_prefix}No playlists found in best_result, skipping minimum size filtering"
            )
            return best_result

        logger.info(
            f"{log_prefix}Applying minimum size filter (>= {min_size} songs) to clustering playlists..."
        )

        original_playlists = best_result["named_playlists"]
        large_playlists = {}
        removed_count = 0

        logger.info(
            f"{log_prefix}Processing {len(original_playlists)} playlists for minimum size filtering"
        )

        for playlist_name, songs_list in original_playlists.items():
            if len(songs_list) >= min_size:
                large_playlists[playlist_name] = songs_list
                logger.debug(
                    f"{log_prefix}Keeping playlist '{playlist_name}' with {len(songs_list)} songs"
                )
            else:
                removed_count += 1
                logger.info(
                    f"{log_prefix}Removed playlist '{playlist_name}' with {len(songs_list)} songs (< {min_size})"
                )

        new_result = best_result.copy()
        new_result["named_playlists"] = large_playlists

        if "playlist_centroids" in best_result:
            new_result["playlist_centroids"] = {
                name: centroids
                for name, centroids in best_result["playlist_centroids"].items()
                if name in large_playlists
            }

        if "playlist_to_centroid_vector_map" in best_result:
            new_result["playlist_to_centroid_vector_map"] = {
                name: vector_map
                for name, vector_map in best_result["playlist_to_centroid_vector_map"].items()
                if name in large_playlists
            }

        if "playlist_primary_genres" in best_result:
            new_result["playlist_primary_genres"] = {
                name: genre
                for name, genre in best_result["playlist_primary_genres"].items()
                if name in large_playlists
            }

        logger.info(
            f"{log_prefix}Minimum size filtering complete: kept {len(large_playlists)} playlists, removed {removed_count} small playlists"
        )

        if len(large_playlists) == 0:
            logger.warning(
                f"{log_prefix}WARNING: All playlists were removed by minimum size filter! Original had {len(original_playlists)} playlists."
            )

        return new_result

    except Exception:
        logger.exception(
            f"{log_prefix}Critical error in minimum size filtering. Returning original result."
        )
        return best_result


def _farthest_pair(candidate_indices, available_vectors):
    candidate_indices = list(candidate_indices)
    if len(candidate_indices) <= 2:
        return candidate_indices

    distances = cdist(
        available_vectors[candidate_indices],
        available_vectors[candidate_indices],
        'euclidean',
    )
    upper_rows, upper_cols = np.triu_indices(len(candidate_indices), k=1)
    pair_distances = distances[upper_rows, upper_cols]
    maximum = np.max(pair_distances)
    tied_pairs = [
        (candidate_indices[row], candidate_indices[col])
        for row, col, distance in zip(upper_rows, upper_cols, pair_distances)
        if np.isclose(distance, maximum)
    ]
    return list(secrets.choice(tied_pairs))


def _maximin_index(
    candidate_indices,
    selected_indices,
    available_vectors,
    available_names,
    original_playlists,
):
    candidate_indices = list(candidate_indices)
    if not candidate_indices:
        return None
    if not selected_indices:
        return max(
            candidate_indices,
            key=lambda index: len(original_playlists.get(available_names[index], [])),
        )

    distances = cdist(
        available_vectors[candidate_indices],
        available_vectors[selected_indices],
        'euclidean',
    )
    minimum_distances = np.min(distances, axis=1)
    maximum = np.max(minimum_distances)
    tied = [
        candidate_index
        for candidate_index, distance in zip(candidate_indices, minimum_distances)
        if np.isclose(distance, maximum)
    ]
    return max(
        tied,
        key=lambda index: len(original_playlists.get(available_names[index], [])),
    )


def _assemble_top_n_result(
    best_result,
    selected_names,
    playlist_to_vector,
    original_playlists,
    original_centroids,
    playlist_primary_genres,
):
    filtered_playlists = {
        name: original_playlists[name] for name in selected_names if name in original_playlists
    }
    filtered_centroids = {
        name: original_centroids[name] for name in selected_names if name in original_centroids
    }
    filtered_vector_map = {
        name: playlist_to_vector[name] for name in selected_names if name in playlist_to_vector
    }
    filtered_primary_genres = {
        name: playlist_primary_genres[name]
        for name in selected_names
        if name in playlist_primary_genres
    }

    new_result = best_result.copy()
    new_result["named_playlists"] = filtered_playlists
    new_result["playlist_centroids"] = filtered_centroids
    new_result["playlist_to_centroid_vector_map"] = filtered_vector_map
    if playlist_primary_genres:
        new_result["playlist_primary_genres"] = filtered_primary_genres
    return new_result


def _rank_top_primary_genres(
    available_names,
    playlist_primary_genres,
    primary_genre_counts,
):
    represented = {
        playlist_primary_genres.get(name)
        for name in available_names
        if playlist_primary_genres.get(name) not in (None, '__other__')
    }
    counts = primary_genre_counts or {
        genre: sum(
            playlist_primary_genres.get(name) == genre
            for name in available_names
        )
        for genre in represented
    }
    return [
        genre
        for genre, _count in sorted(
            (
                (genre, int(counts.get(genre, 0)))
                for genre in represented
            ),
            key=lambda item: (-item[1], str(item[0]).casefold()),
        )[:3]
    ]


def select_diverse_playlists_with_genre_coverage(
    best_result,
    limit,
    primary_genre_counts=None,
):
    playlist_to_vector = best_result.get("playlist_to_centroid_vector_map", {})
    original_playlists = best_result.get("named_playlists", {})
    original_centroids = best_result.get("playlist_centroids", {})
    playlist_primary_genres = best_result.get("playlist_primary_genres", {})

    limit = max(0, int(limit))
    if not playlist_to_vector or limit <= 0:
        logger.info(
            "Skipping Top-N selection: limit=%s, available playlists=%s. "
            "Returning original set.",
            limit,
            len(playlist_to_vector),
        )
        return best_result

    logger.info(
        "Starting 6+4 diverse selection from %d candidates with limit=%d.",
        len(playlist_to_vector),
        limit,
    )

    available_names = list(playlist_to_vector.keys())
    available_vectors = np.array(list(playlist_to_vector.values()))

    logger.info(
        f"Selecting from all {len(available_names)} available playlists (size filtering already applied)."
    )

    if available_vectors.shape[0] <= limit:
        return best_result

    top_genres = _rank_top_primary_genres(
        available_names,
        playlist_primary_genres,
        primary_genre_counts,
    )
    selected_indices = []

    for genre in top_genres:
        genre_indices = [
            index
            for index, name in enumerate(available_names)
            if playlist_primary_genres.get(name) == genre
        ]
        for index in _farthest_pair(genre_indices, available_vectors):
            if len(selected_indices) >= limit:
                break
            selected_indices.append(index)

    selected_other_genres = set()
    for _slot in range(min(4, max(0, limit - len(selected_indices)))):
        candidates = [
            index
            for index, name in enumerate(available_names)
            if index not in selected_indices
            and playlist_primary_genres.get(name) not in set(top_genres)
            and playlist_primary_genres.get(name) not in selected_other_genres
            and playlist_primary_genres.get(name) not in (None, '__other__')
        ]
        chosen = _maximin_index(
            candidates,
            selected_indices,
            available_vectors,
            available_names,
            original_playlists,
        )
        if chosen is None:
            break
        selected_indices.append(chosen)
        selected_other_genres.add(
            playlist_primary_genres.get(available_names[chosen])
        )

    while len(selected_indices) < min(limit, len(available_names)):
        chosen = _maximin_index(
            (
                index
                for index in range(len(available_names))
                if index not in selected_indices
            ),
            selected_indices,
            available_vectors,
            available_names,
            original_playlists,
        )
        if chosen is None:
            break
        selected_indices.append(chosen)

    selected_names = [available_names[i] for i in selected_indices]
    new_result = _assemble_top_n_result(
        best_result,
        selected_names,
        playlist_to_vector,
        original_playlists,
        original_centroids,
        playlist_primary_genres,
    )

    logger.info(
        "Selected %d/%d playlists with top genres %s and non-top genres %s: %s",
        len(selected_names),
        limit,
        top_genres,
        sorted(selected_other_genres),
        selected_names,
    )

    return new_result


def select_top_n_diverse_playlists(best_result, n):
    return select_diverse_playlists_with_genre_coverage(best_result, n)
