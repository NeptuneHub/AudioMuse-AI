# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared result-shaping helpers for the similarity search managers.

Holds the dedup-by-content, per-artist cap, and capped-result assembly used by
the IVF, lyrics, CLAP text and hyperbolic search paths, so each manager calls
one implementation instead of reaching into another manager's private helper.

Main Features:
* is_same_song: case/whitespace-insensitive title + author equality check.
* dedup_by_content: drop duplicate tracks (same normalized title + author)
  that can appear twice after duplicate recordings were merged into one row.
* apply_artist_cap: cap the number of tracks per artist at MAX_SONGS_PER_ARTIST.
* build_capped_results: walk IVF neighbours and assemble the final capped list.
"""

import logging
from typing import Dict, List

import config

logger = logging.getLogger(__name__)


def is_same_song(title1, artist1, title2, artist2):
    def _norm(value):
        return (value or "").strip().lower()

    return _norm(title1) == _norm(title2) and _norm(artist1) == _norm(artist2)


def dedup_by_content(songs, item_details):
    unique_songs = []
    added_songs_details = []
    for song in songs:
        current_details = item_details.get(song['item_id'])
        if not current_details:
            continue

        is_duplicate = any(
            is_same_song(
                current_details['title'], current_details['author'], added['title'], added['author']
            )
            for added in added_songs_details
        )

        if not is_duplicate:
            unique_songs.append(song)
            added_songs_details.append(current_details)
    return unique_songs


def apply_artist_cap(songs, author_resolver, warn_missing=False):
    if config.MAX_SONGS_PER_ARTIST is None or config.MAX_SONGS_PER_ARTIST <= 0:
        return songs

    artist_counts = {}
    capped = []
    for song in songs:
        author = author_resolver(song)
        if not author:
            if warn_missing:
                logger.warning(
                    f"Could not find author for item_id {song['item_id']} during artist deduplication. Skipping."
                )
            continue

        current_count = artist_counts.get(author, 0)
        if current_count < config.MAX_SONGS_PER_ARTIST:
            capped.append(song)
            artist_counts[author] = current_count + 1
    return capped


def build_capped_results(
    ivf_index, id_map, metadata_map, neighbor_ids, distances, limit, artist_cap
) -> List[Dict]:
    results: List[Dict] = []
    artist_counts: Dict[str, int] = {}
    seen: set = set()
    for vid, dist in zip(neighbor_ids, distances):
        if len(results) >= limit:
            break
        item_id = id_map.get(int(vid))
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        meta = metadata_map.get(item_id, {'title': '', 'author': '', 'album': ''})
        author = meta.get('author', '') or ''
        if artist_cap and author:
            an = author.strip().lower()
            if artist_counts.get(an, 0) >= artist_cap:
                continue
            artist_counts[an] = artist_counts.get(an, 0) + 1
        similarity = ivf_index.distance_to_similarity(dist)
        results.append(
            {
                'item_id': item_id,
                'title': meta.get('title', ''),
                'author': author,
                'album': meta.get('album', ''),
                'similarity': similarity,
            }
        )
    return results
