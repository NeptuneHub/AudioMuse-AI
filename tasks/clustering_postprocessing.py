# tasks/clustering_postprocessing.py

"""
Clustering post-processing functions for duplicate filtering, size filtering, and playlist selection.

This module contains functions that were moved from clustering.py and clustering_helper.py 
to improve code organization by separating post-processing logic from core clustering algorithms.

Functions:
- get_vectors_from_database: Fetch embedding vectors directly from database
- apply_distance_filtering_direct: Distance-based duplicate filtering using database vectors
- apply_title_artist_deduplication: Title/artist duplicate filtering fallback
- apply_duplicate_filtering_to_clustering_result: Apply duplicate filtering to clustering playlists
- apply_minimum_size_filter_to_clustering_result: Filter out small playlists
- select_top_n_diverse_playlists: Select most diverse playlists from clustering results
"""

import logging
import numpy as np
import time
import random
import re
from collections import defaultdict
from scipy.spatial.distance import cdist
from psycopg2.extras import DictCursor

from config import (OTHER_FEATURE_LABELS, MOOD_LABELS, MAX_DISTANCE, MAX_SONGS_PER_ARTIST)

logger = logging.getLogger(__name__)


def get_vectors_from_database(item_ids: list, db_conn):
    """
    Fetches embedding vectors directly from the database for distance calculation.
    This bypasses the need for the Voyager index to be loaded.
    
    Args:
        item_ids: List of item IDs to fetch vectors for
        db_conn: Database connection
    
    Returns:
        Dictionary mapping item_id to numpy array vector
    """
    vectors_map = {}
    
    with db_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("SELECT item_id, embedding FROM embedding WHERE item_id = ANY(%s)", (item_ids,))
        rows = cur.fetchall()
        
        for row in rows:
            if row['embedding']:
                try:
                    # Convert bytea to numpy array
                    vector = np.frombuffer(row['embedding'], dtype=np.float32)
                    vectors_map[row['item_id']] = vector
                except Exception as e:
                    logger.warning(f"Failed to decode embedding for {row['item_id']}: {e}")
                    
    return vectors_map


def apply_distance_filtering_direct(song_results: list, db_conn, log_prefix=""):
    """
    Applies distance-based duplicate filtering by fetching vectors directly from the database.
    This works without requiring the Voyager index to be loaded.
    
    Args:
        song_results: List of dictionaries with 'item_id' keys
        db_conn: Database connection
        log_prefix: Optional prefix for logging messages
    
    Returns:
        Filtered list of song dictionaries
    """
    from config import (DUPLICATE_DISTANCE_CHECK_LOOKBACK, 
                       DUPLICATE_DISTANCE_THRESHOLD_COSINE, 
                       DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN, 
                       VOYAGER_METRIC)
    
    if DUPLICATE_DISTANCE_CHECK_LOOKBACK <= 0:
        return song_results

    if not song_results:
        return []
    
    # Fetch vectors directly from database
    item_ids = [s['item_id'] for s in song_results]
    vectors_map = get_vectors_from_database(item_ids, db_conn)
    
    # *** DIAGNOSTIC: Log vector availability ***
    logger.debug(f"{log_prefix}Vector availability: {len(vectors_map)}/{len(item_ids)} songs have embedding vectors")
    if len(vectors_map) < len(item_ids):
        missing_vectors = len(item_ids) - len(vectors_map)
        logger.debug(f"{log_prefix}WARNING: {missing_vectors} songs missing embedding vectors, they will be kept without distance checking")
    
    # If no vectors are available, fall back to title/artist matching
    if not vectors_map:
        logger.info(f"{log_prefix}No embedding vectors found, falling back to title/artist deduplication")
        return apply_title_artist_deduplication(song_results, db_conn, log_prefix)
    
    # Fetch song details for logging
    details_map = {}
    with db_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("SELECT item_id, title, author FROM score WHERE item_id = ANY(%s)", (item_ids,))
        rows = cur.fetchall()
        for row in rows:
            details_map[row['item_id']] = {'title': row['title'], 'author': row['author']}
    
    # Use the same thresholds as voyager_manager
    threshold = DUPLICATE_DISTANCE_THRESHOLD_COSINE if VOYAGER_METRIC == 'angular' else DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN
    metric_name = 'Angular' if VOYAGER_METRIC == 'angular' else 'Euclidean'
    
    filtered_songs = []
    distance_filtered_count = 0
    
    logger.debug(f"{log_prefix}Starting distance filtering with threshold {threshold:.4f} ({metric_name}), lookback window: {DUPLICATE_DISTANCE_CHECK_LOOKBACK}")
    
    # *** DIAGNOSTIC: Track some statistics ***
    total_comparisons = 0
    distances_calculated = []
    
    for current_song in song_results:
        current_vector = vectors_map.get(current_song['item_id'])
        if current_vector is None:
            # Keep songs without vectors (shouldn't happen in clustering)
            logger.debug(f"{log_prefix}No vector found for {current_song['item_id']}, keeping song")
            filtered_songs.append(current_song)
            continue
        
        is_too_close = False
        min_distance = float('inf')
        closest_song = None
        
        # Check against the last N songs in the filtered list
        lookback_window = filtered_songs[-DUPLICATE_DISTANCE_CHECK_LOOKBACK:]
        for recent_song in lookback_window:
            recent_vector = vectors_map.get(recent_song['item_id'])
            if recent_vector is None:
                continue
            
            # *** DIAGNOSTIC: Count comparisons ***
            total_comparisons += 1
            
            # Calculate direct distance using the same functions as voyager_manager
            if VOYAGER_METRIC == 'angular':
                # Angular distance calculation
                if np.linalg.norm(current_vector) > 0 and np.linalg.norm(recent_vector) > 0:
                    v1_u = current_vector / np.linalg.norm(current_vector)
                    v2_u = recent_vector / np.linalg.norm(recent_vector)
                    cosine_similarity = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
                    direct_dist = np.arccos(cosine_similarity) / np.pi
                else:
                    direct_dist = float('inf')
            else:
                # Euclidean distance calculation
                direct_dist = np.linalg.norm(current_vector - recent_vector)
            
            # *** DIAGNOSTIC: Track distance values ***
            if direct_dist != float('inf'):
                distances_calculated.append(direct_dist)
            
            # Track minimum distance for debugging
            if direct_dist < min_distance:
                min_distance = direct_dist
                closest_song = recent_song
            
            if direct_dist < threshold:
                current_details = details_map.get(current_song['item_id'], {'title': 'N/A', 'author': 'N/A'})
                recent_details = details_map.get(recent_song['item_id'], {'title': 'N/A', 'author': 'N/A'})
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
            # Log some examples of kept songs for debugging
            if len(filtered_songs) <= 5 or len(filtered_songs) % 10 == 0:
                current_details = details_map.get(current_song['item_id'], {'title': 'N/A', 'author': 'N/A'})
                if closest_song and min_distance != float('inf'):
                    closest_details = details_map.get(closest_song['item_id'], {'title': 'N/A', 'author': 'N/A'})
                    logger.debug(
                        f"{log_prefix}KEPT: '{current_details['title']}' by '{current_details['author']}' "
                        f"(min distance {min_distance:.4f} to '{closest_details['title']}' by '{closest_details['author']}')"
                    )
                else:
                    logger.debug(f"{log_prefix}KEPT: '{current_details['title']}' by '{current_details['author']}' (first song or no close songs)")
    
    # *** DIAGNOSTIC: Log statistics ***
    if distances_calculated:
        min_dist = min(distances_calculated)
        max_dist = max(distances_calculated) 
        avg_dist = sum(distances_calculated) / len(distances_calculated)
        distances_below_threshold = [d for d in distances_calculated if d < threshold]
        logger.debug(f"{log_prefix}Distance statistics: {total_comparisons} comparisons, min={min_dist:.4f}, max={max_dist:.4f}, avg={avg_dist:.4f}, threshold={threshold:.4f}")
        logger.debug(f"{log_prefix}Distances below threshold: {len(distances_below_threshold)} out of {len(distances_calculated)} ({len(distances_below_threshold)/len(distances_calculated)*100:.1f}%)")
        if distances_below_threshold and distance_filtered_count == 0:
            logger.warning(f"{log_prefix}WARNING: Found {len(distances_below_threshold)} distances below threshold but filtered 0 songs - possible logic error!")
    else:
        logger.debug(f"{log_prefix}No valid distance calculations performed (no vectors or no comparisons)")
    
    logger.info(f"{log_prefix}Distance filtering complete: {len(song_results)} -> {len(filtered_songs)} songs (removed {distance_filtered_count} duplicates)")
    return filtered_songs


def apply_title_artist_deduplication(song_results: list, db_conn, log_prefix=""):
    """
    Fallback duplicate detection using title/artist matching when vectors are not available.
    Removes exact title/artist duplicates from the song list.
    
    Args:
        song_results: List of dictionaries with 'item_id' keys
        db_conn: Database connection
        log_prefix: Optional prefix for logging messages
    
    Returns:
        Filtered list of song dictionaries
    """
    if not song_results:
        return []
    
    # Fetch song details
    item_ids = [s['item_id'] for s in song_results]
    details_map = {}
    
    with db_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("SELECT item_id, title, author FROM score WHERE item_id = ANY(%s)", (item_ids,))
        rows = cur.fetchall()
        for row in rows:
            details_map[row['item_id']] = {'title': row['title'], 'author': row['author']}
    
    # Track seen title/artist combinations
    seen_combinations = set()
    filtered_songs = []
    title_filtered_count = 0
    
    for song in song_results:
        song_details = details_map.get(song['item_id'])
        if not song_details:
            logger.debug(f"{log_prefix}No details found for {song['item_id']}, skipping")
            continue
        
        # Normalize title and artist for comparison
        # Remove common variations like (Remastered), [Explicit], etc.
        title_raw = song_details['title'] if song_details['title'] else ""
        artist_raw = song_details['author'] if song_details['author'] else ""
        
        # Clean up title - remove common suffixes that indicate same song
        title_clean = title_raw.lower().strip()
        # Remove common patterns
        title_clean = re.sub(r'\s*\(.*?(remaster|explicit|clean|radio|edit|version|mix)\).*?$', '', title_clean, flags=re.IGNORECASE)
        title_clean = re.sub(r'\s*\[.*?(remaster|explicit|clean|radio|edit|version|mix)\].*?$', '', title_clean, flags=re.IGNORECASE)
        title_clean = re.sub(r'\s*-\s*(remaster|explicit|clean|radio|edit|version|mix).*?$', '', title_clean, flags=re.IGNORECASE)
        title_clean = title_clean.strip()
        
        artist_clean = artist_raw.lower().strip()
        combination = (title_clean, artist_clean)
        
        if combination not in seen_combinations:
            seen_combinations.add(combination)
            filtered_songs.append(song)
            # Log some examples for debugging
            if len(filtered_songs) <= 5:
                if title_clean != title_raw.lower().strip():
                    logger.debug(f"{log_prefix}KEPT (cleaned): '{title_raw}' -> '{title_clean}' by '{artist_raw}'")
                else:
                    logger.debug(f"{log_prefix}KEPT: '{song_details['title']}' by '{song_details['author']}'")
        else:
            title_filtered_count += 1
            logger.info(f"{log_prefix}REMOVED duplicate: '{title_raw}' by '{artist_raw}' (normalized to '{title_clean}' by '{artist_clean}')")
    
    logger.info(f"{log_prefix}Title/artist deduplication: {len(song_results)} -> {len(filtered_songs)} songs (removed {title_filtered_count} duplicates)")
    return filtered_songs


def apply_duplicate_filtering_to_clustering_result(best_result, log_prefix=""):
    """
    Applies duplicate filtering to clustering playlists using the same logic as voyager_manager.
    Removes songs that are too close in vector distance within each playlist.
    
    This function addresses the issue where clustering can produce playlists with very similar
    songs that are close in the embedding/feature space. It uses the same distance thresholds
    and filtering logic as the voyager_manager to ensure consistent duplicate detection across
    the system, avoiding expensive recalculations by reusing proven filtering algorithms.
    
    Args:
        best_result: The clustering result dictionary with named_playlists
        log_prefix: Optional prefix for logging messages
    
    Returns:
        Modified clustering result with duplicate filtering applied
    """
    try:
        from app_helper import get_db
        
        if not best_result or not best_result.get("named_playlists"):
            logger.warning(f"{log_prefix}No playlists found in best_result, skipping duplicate filtering")
            return best_result
        
        logger.info(f"{log_prefix}Applying duplicate filtering to clustering playlists...")
        
        db_conn = get_db()
        original_playlists = best_result["named_playlists"]
        filtered_playlists = {}
        total_songs_before = 0
        total_songs_after = 0
        
        logger.info(f"{log_prefix}Processing {len(original_playlists)} playlists for duplicate filtering")
        
        # Use database-based vector distance filtering (no need for Voyager index)
        logger.info(f"{log_prefix}Using database-based vector distance filtering for duplicate detection")
        
        for playlist_name, songs_list in original_playlists.items():
            total_songs_before += len(songs_list)
            
            if not songs_list:
                logger.debug(f"{log_prefix}Skipping empty playlist '{playlist_name}'")
                filtered_playlists[playlist_name] = songs_list
                continue
            
            try:
                # *** CLUSTERING-SPECIFIC: Sort songs alphabetically by title to group similar songs together ***
                # This ensures that songs with similar titles (like remixes, different versions) are adjacent,
                # allowing the distance filter with lookback=1 to catch duplicates effectively
                songs_sorted_by_title = sorted(songs_list, key=lambda song: song[1].lower() if song[1] else "")
                logger.info(f"{log_prefix}SORTED {len(songs_sorted_by_title)} songs BY TITLE in playlist '{playlist_name}'")
                logger.info(f"{log_prefix}SORTED ORDER - First 5 titles: {[song[1] for song in songs_sorted_by_title[:5]]}")
                
                # Convert songs list to the format expected by _filter_by_distance
                # Songs are tuples of (item_id, title, author), convert to list of dicts
                song_results = [{"item_id": item_id} for item_id, title, author in songs_sorted_by_title]
                
                logger.debug(f"{log_prefix}Filtering playlist '{playlist_name}' with {len(song_results)} songs")
                
                # *** CRITICAL: Apply filtering on ALPHABETICALLY SORTED songs ***
                # This is essential because distance filtering with lookback=1 only works when similar songs are adjacent
                logger.debug(f"{log_prefix}Applying combined duplicate filtering for playlist '{playlist_name}' on SORTED songs")
                
                # Step 1: Remove exact title/artist duplicates first (on sorted songs)
                temp_filtered = apply_title_artist_deduplication(song_results, db_conn, log_prefix + "[TitleArtist] ")
                
                # Step 2: Apply distance filtering to remaining songs (similar titles are adjacent due to sorting)
                filtered_song_results = apply_distance_filtering_direct(temp_filtered, db_conn, log_prefix + "[Distance] ")
                
                # Convert back to original format and maintain original song details
                filtered_item_ids = {s["item_id"] for s in filtered_song_results}
                # Preserve the sorted order by maintaining the sorted list structure (STILL SORTED at this point)
                filtered_songs = [song for song in songs_sorted_by_title if song[0] in filtered_item_ids]
                
                logger.debug(f"{log_prefix}Filtering complete, now have {len(filtered_songs)} songs in ALPHABETICAL order")
                
                # *** NOW SHUFFLE: Distance filtering is complete, safe to randomize order for user experience ***
                # IMPORTANT: Filtering MUST happen on sorted songs, shuffling MUST happen after filtering
                
                # *** COMPLETELY NEW SHUFFLE ALGORITHM - FISHER-YATES MANUAL IMPLEMENTATION ***
                shuffled_songs = filtered_songs.copy()
                n = len(shuffled_songs)
                
                if n > 1:
                    # Manual Fisher-Yates shuffle - guaranteed to work differently than random.shuffle
                    # Use current time in microseconds as additional randomness
                    current_time_seed = int(time.time() * 1000000) % 1000000
                    
                    for i in range(n - 1, 0, -1):
                        # Generate random index using multiple sources of randomness
                        j = (random.randint(0, i) + current_time_seed + i) % (i + 1)
                        # Swap elements
                        shuffled_songs[i], shuffled_songs[j] = shuffled_songs[j], shuffled_songs[i]
                        current_time_seed = (current_time_seed * 1103515245 + 12345) % (2**31)  # Linear congruential generator
                    
                    # Verify the shuffle actually worked
                    original_titles = [song[1] for song in filtered_songs]
                    shuffled_titles = [song[1] for song in shuffled_songs]
                    
                    # Check if ANY position changed (not just first song)
                    positions_changed = sum(1 for i in range(len(original_titles)) if original_titles[i] != shuffled_titles[i])
                    
                    logger.info(f"{log_prefix}FISHER-YATES SHUFFLED '{playlist_name}': {n} songs, {positions_changed}/{n} positions changed")
                    logger.info(f"{log_prefix}BEFORE SHUFFLE: First 5 titles = {original_titles[:5]}")
                    logger.info(f"{log_prefix}AFTER SHUFFLE:  First 5 titles = {shuffled_titles[:5]}")
                    logger.info(f"{log_prefix}BEFORE SHUFFLE: Last 3 titles = {original_titles[-3:]}")
                    logger.info(f"{log_prefix}AFTER SHUFFLE:  Last 3 titles = {shuffled_titles[-3:]}")
                    
                    if positions_changed == 0:
                        logger.error(f"{log_prefix}SHUFFLE COMPLETELY FAILED FOR '{playlist_name}' - FORCE REVERSING ORDER")
                        shuffled_songs = list(reversed(shuffled_songs))
                        logger.info(f"{log_prefix}FORCE REVERSED: First 5 titles = {[song[1] for song in shuffled_songs[:5]]}")
                else:
                    logger.info(f"{log_prefix}Playlist '{playlist_name}' has only {n} songs - no shuffling needed")
                
                filtered_playlists[playlist_name] = shuffled_songs
                total_songs_after += len(shuffled_songs)
                
                if len(filtered_songs) != len(songs_list):
                    logger.info(f"{log_prefix}Playlist '{playlist_name}': filtered {len(songs_list)} -> {len(filtered_songs)} songs")
                else:
                    logger.debug(f"{log_prefix}Playlist '{playlist_name}': no songs filtered ({len(songs_list)} songs)")
                    
            except Exception as e:
                logger.error(f"{log_prefix}Error filtering playlist '{playlist_name}': {e}. Keeping original playlist but shuffling it.", exc_info=True)
                # Keep original playlist if filtering fails, but still shuffle it
                shuffled_original = songs_list.copy()
                random.shuffle(shuffled_original)
                logger.info(f"{log_prefix}SHUFFLED original playlist '{playlist_name}' as fallback: {len(shuffled_original)} songs")
                filtered_playlists[playlist_name] = shuffled_original
                total_songs_after += len(shuffled_original)
        
        # Create new result with filtered playlists
        new_result = best_result.copy()
        new_result["named_playlists"] = filtered_playlists
        
        # Update other related data structures to match the filtered playlists
        if "playlist_centroids" in best_result:
            # Remove centroids for playlists that no longer exist (shouldn't happen, but safety check)
            new_result["playlist_centroids"] = {
                name: centroids for name, centroids in best_result["playlist_centroids"].items() 
                if name in filtered_playlists
            }
        
        if "playlist_to_centroid_vector_map" in best_result:
            new_result["playlist_to_centroid_vector_map"] = {
                name: vector_map for name, vector_map in best_result["playlist_to_centroid_vector_map"].items()
                if name in filtered_playlists
            }
        
        logger.info(f"{log_prefix}Duplicate filtering complete: {total_songs_before} -> {total_songs_after} songs total across {len(filtered_playlists)} playlists")
        
        return new_result
        
    except Exception as e:
        logger.error(f"{log_prefix}Critical error in duplicate filtering: {e}. Returning original result.", exc_info=True)
        return best_result


def apply_minimum_size_filter_to_clustering_result(best_result, min_size=20, log_prefix=""):
    """
    Applies minimum size filtering to clustering playlists.
    Removes playlists that have fewer than min_size songs.
    
    Args:
        best_result: The clustering result dictionary with named_playlists
        min_size: Minimum number of songs required for a playlist
        log_prefix: Optional prefix for logging messages
    
    Returns:
        Modified clustering result with small playlists filtered out
    """
    try:
        if not best_result or not best_result.get("named_playlists"):
            logger.warning(f"{log_prefix}No playlists found in best_result, skipping minimum size filtering")
            return best_result
        
        logger.info(f"{log_prefix}Applying minimum size filter (>= {min_size} songs) to clustering playlists...")
        
        original_playlists = best_result["named_playlists"]
        large_playlists = {}
        removed_count = 0
        
        logger.info(f"{log_prefix}Processing {len(original_playlists)} playlists for minimum size filtering")
        
        for playlist_name, songs_list in original_playlists.items():
            if len(songs_list) >= min_size:
                large_playlists[playlist_name] = songs_list
                logger.debug(f"{log_prefix}Keeping playlist '{playlist_name}' with {len(songs_list)} songs")
            else:
                removed_count += 1
                logger.info(f"{log_prefix}Removed playlist '{playlist_name}' with {len(songs_list)} songs (< {min_size})")
        
        # Create new result with large playlists only
        new_result = best_result.copy()
        new_result["named_playlists"] = large_playlists
        
        # Update other related data structures to match the filtered playlists
        if "playlist_centroids" in best_result:
            new_result["playlist_centroids"] = {
                name: centroids for name, centroids in best_result["playlist_centroids"].items() 
                if name in large_playlists
            }
        
        if "playlist_to_centroid_vector_map" in best_result:
            new_result["playlist_to_centroid_vector_map"] = {
                name: vector_map for name, vector_map in best_result["playlist_to_centroid_vector_map"].items()
                if name in large_playlists
            }
        
        logger.info(f"{log_prefix}Minimum size filtering complete: kept {len(large_playlists)} playlists, removed {removed_count} small playlists")
        
        if len(large_playlists) == 0:
            logger.warning(f"{log_prefix}WARNING: All playlists were removed by minimum size filter! Original had {len(original_playlists)} playlists.")
        
        return new_result
        
    except Exception as e:
        logger.error(f"{log_prefix}Critical error in minimum size filtering: {e}. Returning original result.", exc_info=True)
        return best_result


def select_top_n_diverse_playlists(best_result, n):
    """
    Selects the N most diverse playlists from a clustering result by weighting
    both distance (diversity) and size (usefulness).
    """
    playlist_to_vector = best_result.get("playlist_to_centroid_vector_map", {})
    original_playlists = best_result.get("named_playlists", {})
    original_centroids = best_result.get("playlist_centroids", {})

    if not playlist_to_vector or n <= 0 or n >= len(playlist_to_vector):
        logger.info(f"Skipping Top-N selection: N={n}, available playlists={len(playlist_to_vector)}. Returning original set.")
        return best_result

    logger.info(f"Starting selection of Top {n} diverse playlists from {len(playlist_to_vector)} candidates.")

    # Since we've already applied minimum size filtering before this step,
    # all remaining playlists should meet the size requirement. Use all available playlists.
    available_names = list(playlist_to_vector.keys())
    available_vectors = np.array(list(playlist_to_vector.values()))
    
    logger.info(f"Selecting from all {len(available_names)} available playlists (size filtering already applied).")

    if available_vectors.shape[0] <= n:
        return best_result

    selected_indices = []
    
    # 1. Start with the largest playlist to anchor the selection
    playlist_sizes = [len(original_playlists.get(name, [])) for name in available_names]
    first_idx = np.argmax(playlist_sizes)
    selected_indices.append(first_idx)

    # Create a boolean mask for available items
    is_available = np.ones(len(available_names), dtype=bool)
    is_available[first_idx] = False
    
    # 2. Iteratively select the playlist with the best combined score of distance and size
    for _ in range(n - 1):
        if not np.any(is_available):
            break # No more playlists to select

        selected_vectors = available_vectors[selected_indices]
        remaining_vectors = available_vectors[is_available]
        
        # --- Calculate Diversity Score (Distance) ---
        dist_matrix = cdist(remaining_vectors, selected_vectors, 'euclidean')
        min_distances = np.min(dist_matrix, axis=1)
        
        # --- Calculate Size Score ---
        original_indices_available = np.where(is_available)[0]
        sizes_available = np.array([len(original_playlists.get(available_names[i], [])) for i in original_indices_available])
        # Use log1p for a smooth curve with diminishing returns for size
        size_scores = np.log1p(sizes_available)

        # --- Normalize and Combine Scores ---
        # Normalize both scores to a 0-1 range to make them comparable
        max_dist = np.max(min_distances)
        normalized_dist_scores = min_distances / max_dist if max_dist > 0 else np.zeros_like(min_distances)

        max_size_score = np.max(size_scores)
        normalized_size_scores = size_scores / max_size_score if max_size_score > 0 else np.zeros_like(size_scores)
        
        # Combine the scores (equal weighting)
        # TEST USING * INSTEAD OF +
        combined_scores = normalized_dist_scores * normalized_size_scores
        
        # Find the playlist that has the maximum combined score
        best_candidate_local_idx = np.argmax(combined_scores)
        
        # Convert this local index back to the original full list index
        best_original_idx = original_indices_available[best_candidate_local_idx]
        
        # Add to selected and mark as unavailable
        selected_indices.append(best_original_idx)
        is_available[best_original_idx] = False

    # 3. Build the new, filtered result
    selected_names = [available_names[i] for i in selected_indices]
    
    filtered_playlists = {name: original_playlists[name] for name in selected_names if name in original_playlists}
    filtered_centroids = {name: original_centroids[name] for name in selected_names if name in original_centroids}
    filtered_vector_map = {name: playlist_to_vector[name] for name in selected_names if name in playlist_to_vector}

    new_result = best_result.copy()
    new_result["named_playlists"] = filtered_playlists
    new_result["playlist_centroids"] = filtered_centroids
    new_result["playlist_to_centroid_vector_map"] = filtered_vector_map
    
    logger.info(f"Selected {len(selected_names)} diverse playlists: {selected_names}")

    return new_result