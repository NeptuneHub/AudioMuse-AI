# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared task-layer helpers: track metadata lookup and feature scoring.

Small utilities reused across the task modules to avoid duplication. Fetches
per-track metadata maps and computes the scalar feature score used to rank and
describe songs, normalizing tempo/energy against the configured min/max bounds.

Main Features:
* fetch_track_metadata_map: batch title/author/album lookup keyed by item id.
* score_vector: derive a track's scored feature values from its mood and other
  feature labels, clamped to the configured tempo and energy ranges.
"""

import logging

from config import TEMPO_MAX_BPM, TEMPO_MIN_BPM, ENERGY_MAX, ENERGY_MIN

logger = logging.getLogger(__name__)


def fetch_track_metadata_map(item_ids):
    metadata_map = {}
    if not item_ids:
        return metadata_map
    from database import get_score_data_by_ids

    try:
        for row in get_score_data_by_ids(item_ids):
            metadata_map[row['item_id']] = {
                'title': row.get('title', '') or '',
                'author': row.get('author', '') or '',
                'album': row.get('album', '') or '',
            }
    except Exception as e:
        logger.warning(f"Failed to fetch track metadata: {e}")
    return metadata_map


_LABEL_INDEX_CACHE = {}


def _label_index_map(labels):
    key = id(labels)
    cached = _LABEL_INDEX_CACHE.get(key)
    if cached is not None and cached[0] is labels and cached[1] == len(labels):
        return cached[2]
    index_map = {}
    for position, label in enumerate(labels):
        index_map.setdefault(label, position)
    if len(_LABEL_INDEX_CACHE) >= 16:
        _LABEL_INDEX_CACHE.clear()
    _LABEL_INDEX_CACHE[key] = (labels, len(labels), index_map)
    return index_map


def _fill_label_scores(text, index_map, vector, base):
    if not text:
        return
    for pair in text.split(","):
        label, separator, score_str = pair.partition(":")
        if not separator:
            continue
        position = index_map.get(label)
        if position is None:
            continue
        try:
            vector[base + position] = float(score_str)
        except ValueError:
            continue


def score_vector(row, mood_labels_list, other_feature_labels_list):
    tempo = float(row['tempo']) if row['tempo'] is not None else 0.0
    energy = float(row['energy']) if row['energy'] is not None else 0.0

    tempo_range = TEMPO_MAX_BPM - TEMPO_MIN_BPM
    tempo_norm = (tempo - TEMPO_MIN_BPM) / tempo_range if tempo_range > 0 else 0.0
    tempo_norm = min(max(tempo_norm, 0.0), 1.0)

    energy_range = ENERGY_MAX - ENERGY_MIN
    energy_norm = (energy - ENERGY_MIN) / energy_range if energy_range > 0 else 0.0
    energy_norm = min(max(energy_norm, 0.0), 1.0)

    mood_count = len(mood_labels_list)
    full_vector = [0.0] * (2 + mood_count + len(other_feature_labels_list))
    full_vector[0] = tempo_norm
    full_vector[1] = energy_norm

    _fill_label_scores(
        row['mood_vector'] or "", _label_index_map(mood_labels_list), full_vector, 2
    )
    _fill_label_scores(
        row.get('other_features', ""),
        _label_index_map(other_feature_labels_list),
        full_vector,
        2 + mood_count,
    )
    return full_vector
