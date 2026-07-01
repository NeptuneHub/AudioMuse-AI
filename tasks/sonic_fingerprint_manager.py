# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Build a user's sonic fingerprint playlist from their listening history.

Derives a personalised playlist by expanding the user's most-played songs into
similar tracks, using the media server's play counts and the similarity index.

Main Features:
* Fetches the top played songs, loads their embeddings, and grows the set toward
  a target size via nearest-neighbour lookups around each seed.
* Falls back gracefully to an empty result when no play history or embeddings are
  available, and can scope play counts to per-user credentials.
"""

import logging
import numpy as np
from datetime import datetime, timezone

from config import SONIC_FINGERPRINT_TOP_N_SONGS, SONIC_FINGERPRINT_NEIGHBORS
from .mediaserver import get_top_played_songs, get_last_played_time
from .ivf_manager import find_nearest_neighbors_by_vector


logger = logging.getLogger(__name__)


def generate_sonic_fingerprint(num_neighbors=None, user_creds=None):
    from app_helper import get_tracks_by_ids

    logger.info("Generating sonic fingerprint...")

    total_desired_size = num_neighbors if num_neighbors is not None else SONIC_FINGERPRINT_NEIGHBORS
    logger.info(f"Targeting a total playlist size of {total_desired_size}.")

    top_songs = get_top_played_songs(limit=SONIC_FINGERPRINT_TOP_N_SONGS, user_creds=user_creds)
    if not top_songs:
        logger.warning("No top played songs found. Cannot generate sonic fingerprint.")
        return []

    top_song_ids = [str(song['Id']) for song in top_songs]
    logger.info(f"Found {len(top_song_ids)} top played songs to create fingerprint from.")
    logger.debug(f"Top played song IDs: {top_song_ids[:5]}...")

    track_details = get_tracks_by_ids(top_song_ids)
    logger.info(
        f"Retrieved embeddings for {len(track_details)} out of {len(top_song_ids)} songs from database."
    )
    if track_details:
        logger.debug(
            f"Sample track details - item_ids: {[t['item_id'] for t in track_details[:3]]}"
        )
    else:
        logger.warning("No track details found in database for any of the top played songs!")
    if not track_details:
        logger.warning("Could not retrieve embeddings for any of the top songs.")
        return []

    embeddings_map = {
        track['item_id']: track['embedding_vector']
        for track in track_details
        if 'embedding_vector' in track and track['embedding_vector'].size > 0
    }
    logger.info(
        f"Found valid embeddings for {len(embeddings_map)} songs out of {len(track_details)} track details."
    )

    if not embeddings_map:
        logger.error("No songs have valid embeddings in the database!")
        logger.debug(f"Track details sample: {track_details[:2] if track_details else 'None'}")
        return []

    weighted_vectors = []
    total_weight = 0

    for song_id in top_song_ids:
        if song_id not in embeddings_map:
            logger.debug(f"Skipping song {song_id} as it has no embedding in the database.")
            continue

        embedding_vector = embeddings_map[song_id]

        last_played_str = get_last_played_time(song_id, user_creds=user_creds)

        weight = 1.0
        days_since_played = "N/A"
        if last_played_str:
            try:
                if '.' in last_played_str and last_played_str.endswith('Z'):
                    dot_index = last_played_str.rfind('.')
                    z_index = last_played_str.rfind('Z')
                    if z_index > dot_index and (z_index - dot_index - 1) > 6:
                        last_played_str = last_played_str[: dot_index + 7] + 'Z'

                last_played_dt = datetime.fromisoformat(last_played_str.replace('Z', '+00:00'))
                days_since_played = (datetime.now(timezone.utc) - last_played_dt).days

                half_life = 30.0
                decay_rate = -np.log(0.5) / half_life
                weight = np.exp(-decay_rate * max(0, days_since_played))

            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not parse date '{last_played_str}' for song {song_id}. Using lower weight. Error: {e}"
                )
                weight = 0.5
        else:
            weight = 0.25

        weighted_vectors.append(embedding_vector * weight)
        total_weight += weight
        logger.info(f"Song {song_id}, days since played: {days_since_played}, weight: {weight:.4f}")

    if not weighted_vectors:
        logger.error(
            "No valid embeddings with weights could be calculated. Cannot generate fingerprint."
        )
        return []

    average_vector = np.sum(weighted_vectors, axis=0) / total_weight
    logger.info(
        f"Calculated average vector (sonic fingerprint) from {len(weighted_vectors)} songs."
    )

    contributing_seed_ids = list(embeddings_map.keys())
    num_seed_songs = len(contributing_seed_ids)

    neighbors_to_find = total_desired_size - num_seed_songs

    if neighbors_to_find <= 0:
        logger.info(
            f"The number of seed songs ({num_seed_songs}) is >= the desired playlist size ({total_desired_size}). Returning only seed songs, truncated to the desired size."
        )
        final_results = [
            {'item_id': song_id, 'distance': 0.0}
            for song_id in contributing_seed_ids[:total_desired_size]
        ]
        return final_results

    try:
        logger.info(
            f"Searching for {neighbors_to_find} new neighbors to supplement the {num_seed_songs} seed songs."
        )
        similar_songs_from_ivf = find_nearest_neighbors_by_vector(
            query_vector=average_vector, n=neighbors_to_find, eliminate_duplicates=True
        )
        logger.info(f"Found {len(similar_songs_from_ivf)} similar songs for the sonic fingerprint.")

        final_song_ids = set()
        combined_results = []

        for song_id in contributing_seed_ids:
            if song_id not in final_song_ids:
                combined_results.append({'item_id': song_id, 'distance': 0.0})
                final_song_ids.add(song_id)

        logger.info(f"Added {len(final_song_ids)} seed songs to the results.")

        for song in similar_songs_from_ivf:
            if len(combined_results) >= total_desired_size:
                break
            if song['item_id'] not in final_song_ids:
                combined_results.append(song)
                final_song_ids.add(song['item_id'])

        logger.info(f"Total unique songs in final fingerprint playlist: {len(combined_results)}")

        return combined_results

    except Exception as e:
        logger.error(f"Error finding neighbors for sonic fingerprint: {e}", exc_info=True)
        return []
