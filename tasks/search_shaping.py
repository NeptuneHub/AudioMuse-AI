# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared result-shaping helpers for the similarity search managers.

Holds the dedup-by-content, per-artist cap, near-duplicate window and capped-result
assembly used by the IVF, lyrics, CLAP text and hyperbolic search paths, so each
manager calls one implementation instead of reaching into another manager's private
helper or growing its own copy of the same rule.

Main Features:
* is_same_song: case/whitespace-insensitive title + author equality check.
* name_key_for: the dedup key, or None when the track has no title to compare. An
  author on its own is not an identity, so untitled tracks are never folded onto
  each other and one failed metadata lookup cannot collapse a whole result page.
* dedup_by_content: drop duplicate tracks (same normalized title + author)
  that can appear twice after duplicate recordings were merged into one row.
  Keyed through name_key_for, so it is one set lookup per track rather than a
  scan of everything kept so far, and untitled tracks are never folded together.
* apply_artist_cap: cap the number of tracks per artist at MAX_SONGS_PER_ARTIST.
  A track with no resolvable author is EXEMPT from the cap, never dropped: there
  is no artist to count it against, and deleting it made an untagged track
  unreachable through every path that shapes results this way. build_capped_results
  and the journey picker apply the same rule.
* NearDuplicateWindow: the single definition of the rolling lookback window, taking
  the distance function from the caller so the cosine indexes and the Poincare ones
  share it. Only ACCEPTED vectors enter the window.
* overfetch_size: the one over-fetch formula, so every filtered search asks the index
  for the same headroom and none of them return short.
* read_unit_vectors: one batched get_vectors call per query rather than one cell
  lookup per candidate, and one loud log if the index cannot serve it.
* build_capped_results: walk IVF neighbours and assemble the final capped list. Every
  rejection test runs before any counter is written, so a candidate dropped by the
  distance filter never consumes an artist slot or claims a name.
"""

import logging
from typing import Dict, List

import numpy as np

import config

logger = logging.getLogger(__name__)


def is_same_song(title1, artist1, title2, artist2):
    def _norm(value):
        return (value or "").strip().lower()

    return _norm(title1) == _norm(title2) and _norm(artist1) == _norm(artist2)


def dedup_by_content(songs, item_details):
    unique_songs = []
    seen_keys = set()
    for song in songs:
        current_details = item_details.get(song['item_id'])
        if not current_details:
            continue

        key = name_key_for(current_details.get('title'), current_details.get('author'))
        if key is not None:
            if key in seen_keys:
                continue
            seen_keys.add(key)
        unique_songs.append(song)
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
                    f"Could not find author for item_id {song['item_id']} during artist "
                    "deduplication. Keeping it and exempting it from the per-artist cap."
                )
            capped.append(song)
            continue

        current_count = artist_counts.get(author, 0)
        if current_count < config.MAX_SONGS_PER_ARTIST:
            capped.append(song)
            artist_counts[author] = current_count + 1
    return capped


def to_unit_vector(vec):
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return arr
    return arr / norm


def cosine_distance(unit_a, unit_b):
    return float(np.clip(1.0 - float(np.dot(unit_a, unit_b)), 0.0, 2.0))


def overfetch_size(limit):
    return int(limit) + max(20, int(limit) * 4) + 1


def read_unit_vectors(ivf_index, vec_ids):
    ids = [int(v) for v in vec_ids]
    if not ids:
        return {}
    try:
        raw = ivf_index.get_vectors(ids)
    except Exception:
        logger.exception(
            "Duplicate distance filter DISABLED for this query: could not batch-read %d vectors "
            "back from the index",
            len(ids),
        )
        return None
    return {int(vid): to_unit_vector(vec) for vid, vec in raw.items() if vec is not None}


class NearDuplicateWindow:
    """The one definition of "is this vector a near duplicate of what we just kept".

    Holds the rolling window of the last `lookback` ACCEPTED vectors, in the order
    the caller accepted them, and answers against a caller-supplied distance
    function so the cosine indexes and the Poincare ones share this logic instead
    of each carrying their own copy of the same rule.
    """

    def __init__(self, threshold, lookback, distance_fn):
        self.threshold = float(threshold or 0.0)
        self.lookback = int(lookback or 0)
        self.distance_fn = distance_fn
        self.window: List = []

    @property
    def active(self):
        return self.threshold > 0.0 and self.lookback > 0

    def is_duplicate(self, vector):
        if not self.active or vector is None or not self.window:
            return False
        return any(
            self.distance_fn(vector, previous) < self.threshold
            for previous in self.window[-self.lookback:]
        )

    def remember(self, vector):
        if self.active and vector is not None:
            self.window.append(vector)


def cosine_duplicate_window(threshold, lookback):
    return NearDuplicateWindow(threshold, lookback, cosine_distance)


def name_key_for(title, author):
    clean_title = (title or '').strip()
    if not clean_title:
        return None
    return (clean_title.lower(), (author or '').strip().lower())


def build_capped_results(
    ivf_index,
    id_map,
    metadata_map,
    neighbor_ids,
    distances,
    limit,
    artist_cap,
    dedup_names=False,
    dup_threshold=0.0,
    lookback=0,
) -> List[Dict]:
    results: List[Dict] = []
    artist_counts: Dict[str, int] = {}
    seen: set = set()
    seen_names: set = set()
    window = cosine_duplicate_window(dup_threshold, lookback)

    unit_vectors = {}
    if window.active:
        unit_vectors = read_unit_vectors(ivf_index, neighbor_ids)
        if unit_vectors is None:
            unit_vectors = {}
            window = cosine_duplicate_window(0.0, 0)

    for vid, dist in zip(neighbor_ids, distances):
        if len(results) >= limit:
            break
        item_id = id_map.get(int(vid))
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        meta = metadata_map.get(item_id, {'title': '', 'author': '', 'album': ''})
        author = meta.get('author', '') or ''
        title = meta.get('title', '') or ''

        name_key = name_key_for(title, author) if dedup_names else None
        if name_key is not None and name_key in seen_names:
            logger.info("Found duplicate (NAME FILTER): '%s' by '%s'.", title, author)
            continue

        artist_key = author.strip().lower() if (artist_cap and author.strip()) else None
        if artist_key is not None and artist_counts.get(artist_key, 0) >= artist_cap:
            continue

        candidate_unit = unit_vectors.get(int(vid)) if window.active else None
        if window.is_duplicate(candidate_unit):
            logger.info(
                "Found duplicate (DISTANCE FILTER): '%s' by '%s' within %.4f.",
                title,
                author,
                window.threshold,
            )
            continue

        if name_key is not None:
            seen_names.add(name_key)
        if artist_key is not None:
            artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1
        window.remember(candidate_unit)

        results.append(
            {
                'item_id': item_id,
                'title': meta.get('title', ''),
                'author': author,
                'album': meta.get('album', ''),
                'similarity': ivf_index.distance_to_similarity(dist),
            }
        )
    return results
