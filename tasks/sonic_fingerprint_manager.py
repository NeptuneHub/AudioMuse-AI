import logging
import numpy as np
from datetime import datetime, timezone

from config import (
    SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM,
    SONIC_FINGERPRINT_NEIGHBORS,
    SONIC_FINGERPRINT_TOP_N_SONGS,
)
from .mediaserver import get_top_played_songs, get_last_played_time
from .voyager_manager import find_nearest_neighbors_by_vector


logger = logging.getLogger(__name__)


def _album_key(track_details):
    album = (track_details or {}).get('album')
    if not album:
        return None
    album = str(album).strip().lower()
    return album if album and album != 'unknown' else None


def _apply_album_cap(results, details_by_id, max_songs_per_album):
    if max_songs_per_album is None or max_songs_per_album <= 0:
        return results

    album_counts = {}
    capped_results = []
    for song in results:
        item_id = str(song.get('item_id'))
        key = _album_key(details_by_id.get(item_id))
        if key is None:
            capped_results.append(song)
            continue

        current_count = album_counts.get(key, 0)
        if current_count < max_songs_per_album:
            capped_results.append(song)
            album_counts[key] = current_count + 1
        else:
            logger.debug(
                "Skipping song %s because album cap %s was reached for '%s'.",
                item_id,
                max_songs_per_album,
                key,
            )

    return capped_results

def generate_sonic_fingerprint(num_neighbors=None, user_creds=None):
    """
    Generates a 'sonic fingerprint' by averaging the embeddings of the most played songs,
    weighted by recency, and then finds similar songs to this fingerprint.
    The final list will be truncated to the requested number of results.

    Args:
        num_neighbors (int, optional): The total number of desired tracks in the final playlist.
                                       If None, defaults to SONIC_FINGERPRINT_NEIGHBORS from config.
        user_creds (dict, optional): A dictionary containing user-specific credentials.
                                     For Jellyfin: {'user_id': '...', 'token': '...'}
                                     For Navidrome: {'user': '...', 'password': '...'}
    """
    from app_helper import get_tracks_by_ids
    logger.info("Generating sonic fingerprint...")

    # Determine the total desired size for the final playlist
    total_desired_size = num_neighbors if num_neighbors is not None else SONIC_FINGERPRINT_NEIGHBORS
    logger.info(f"Targeting a total playlist size of {total_desired_size}.")

    # 1. Get top N played songs from the media server, passing user credentials
    top_songs = get_top_played_songs(limit=SONIC_FINGERPRINT_TOP_N_SONGS, user_creds=user_creds)
    if not top_songs:
        logger.warning("No top played songs found. Cannot generate sonic fingerprint.")
        return []

    top_song_ids = [str(song['Id']) for song in top_songs]  # Convert to strings for consistency
    logger.info(f"Found {len(top_song_ids)} top played songs to create fingerprint from.")
    logger.debug(f"Top played song IDs: {top_song_ids[:5]}...")  # Log first 5 IDs for debugging

    # 2. Get embeddings for these songs from our DB
    track_details = get_tracks_by_ids(top_song_ids)
    track_details_by_id = {str(track['item_id']): track for track in track_details}
    logger.info(f"Retrieved embeddings for {len(track_details)} out of {len(top_song_ids)} songs from database.")
    if track_details:
        logger.debug(f"Sample track details - item_ids: {[t['item_id'] for t in track_details[:3]]}")
    else:
        logger.warning("No track details found in database for any of the top played songs!")
    if not track_details:
        logger.warning("Could not retrieve embeddings for any of the top songs.")
        return []

    # This map will only contain songs that have a valid embedding
    embeddings_map = {track['item_id']: track['embedding_vector'] for track in track_details if 'embedding_vector' in track and track['embedding_vector'].size > 0}
    logger.info(f"Found valid embeddings for {len(embeddings_map)} songs out of {len(track_details)} track details.")
    
    if not embeddings_map:
        logger.error("No songs have valid embeddings in the database!")
        logger.debug(f"Track details sample: {track_details[:2] if track_details else 'None'}")
        return []
    
    weighted_vectors = []
    total_weight = 0

    # 3. Calculate weighted average of embeddings
    for song_id in top_song_ids:
        if song_id not in embeddings_map:
            logger.debug(f"Skipping song {song_id} as it has no embedding in the database.")
            continue

        embedding_vector = embeddings_map[song_id]
        
        # Pass user credentials to get last played time
        last_played_str = get_last_played_time(song_id, user_creds=user_creds)
        
        weight = 1.0
        days_since_played = "N/A"
        if last_played_str:
            try:
                if '.' in last_played_str and last_played_str.endswith('Z'):
                    dot_index = last_played_str.rfind('.')
                    z_index = last_played_str.rfind('Z')
                    if z_index > dot_index and (z_index - dot_index - 1) > 6:
                        last_played_str = last_played_str[:dot_index+7] + 'Z'

                last_played_dt = datetime.fromisoformat(last_played_str.replace('Z', '+00:00'))
                days_since_played = (datetime.now(timezone.utc) - last_played_dt).days
                
                half_life = 30.0
                decay_rate = -np.log(0.5) / half_life
                weight = np.exp(-decay_rate * max(0, days_since_played))

            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse date '{last_played_str}' for song {song_id}. Using lower weight. Error: {e}")
                weight = 0.5
        else:
            weight = 0.25

        weighted_vectors.append(embedding_vector * weight)
        total_weight += weight
        logger.info(f"Song {song_id}, days since played: {days_since_played}, weight: {weight:.4f}")

    if not weighted_vectors:
        logger.error("No valid embeddings with weights could be calculated. Cannot generate fingerprint.")
        return []

    average_vector = np.sum(weighted_vectors, axis=0) / total_weight
    logger.info(f"Calculated average vector (sonic fingerprint) from {len(weighted_vectors)} songs.")

    # 4. Use Voyager to find similar songs
    # Get the IDs of the songs that actually contributed to the fingerprint
    contributing_seed_ids = list(embeddings_map.keys())
    num_seed_songs = len(contributing_seed_ids)

    album_cap = SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM

    # Calculate how many new neighbors to find to reach the total desired size
    neighbors_to_find = total_desired_size - num_seed_songs

    if neighbors_to_find <= 0 and (album_cap is None or album_cap <= 0):
        logger.info(f"The number of seed songs ({num_seed_songs}) is >= the desired playlist size ({total_desired_size}). Returning only seed songs, truncated to the desired size.")
        final_results = [{'item_id': song_id, 'distance': 0.0} for song_id in contributing_seed_ids[:total_desired_size]]
        return final_results

    try:
        query_neighbors = total_desired_size if album_cap and album_cap > 0 else neighbors_to_find
        logger.info(f"Searching for {query_neighbors} new neighbors to supplement the {num_seed_songs} seed songs.")
        similar_songs_from_voyager = find_nearest_neighbors_by_vector(
            query_vector=average_vector,
            n=query_neighbors,
            eliminate_duplicates=True
        )
        logger.info(f"Found {len(similar_songs_from_voyager)} similar songs for the sonic fingerprint.")

        candidate_ids = [str(song['item_id']) for song in similar_songs_from_voyager]
        candidate_details = get_tracks_by_ids(candidate_ids) if album_cap and album_cap > 0 else []
        details_by_id = {
            **track_details_by_id,
            **{str(track['item_id']): track for track in candidate_details},
        }

        # --- Combine seed songs and similar songs ---
        final_song_ids = set()
        combined_results = []

        # 1. Add the seed songs first
        for song_id in contributing_seed_ids:
            if song_id not in final_song_ids:
                combined_results.append({'item_id': song_id, 'distance': 0.0})
                final_song_ids.add(song_id)
        
        logger.info(f"Added {len(final_song_ids)} seed songs to the results.")

        # 2. Add the similar songs found by Voyager, skipping duplicates, until the desired size is reached
        for song in similar_songs_from_voyager:
            if (album_cap is None or album_cap <= 0) and len(combined_results) >= total_desired_size:
                break
            if song['item_id'] not in final_song_ids:
                combined_results.append(song)
                final_song_ids.add(song['item_id'])
                
        combined_results = _apply_album_cap(combined_results, details_by_id, album_cap)
        combined_results = combined_results[:total_desired_size]
        logger.info(f"Total unique songs in final fingerprint playlist: {len(combined_results)}")

        return combined_results

    except Exception as e:
        logger.error(f"Error finding neighbors for sonic fingerprint: {e}", exc_info=True)
        return []
