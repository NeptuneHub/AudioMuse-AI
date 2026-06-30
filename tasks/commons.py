
import logging

import numpy as np

from config import (
    TEMPO_MAX_BPM, TEMPO_MIN_BPM,
    ENERGY_MAX, ENERGY_MIN
)

logger = logging.getLogger(__name__)


def fetch_track_metadata_map(item_ids):
    metadata_map = {}
    if not item_ids:
        return metadata_map
    from app_helper import get_score_data_by_ids
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

def score_vector(row, mood_labels_list, other_feature_labels_list):
    tempo = float(row['tempo']) if row['tempo'] is not None else 0.0
    energy = float(row['energy']) if row['energy'] is not None else 0.0
    mood_str = row['mood_vector'] or ""

    tempo_range = TEMPO_MAX_BPM - TEMPO_MIN_BPM
    tempo_norm = (tempo - TEMPO_MIN_BPM) / tempo_range if tempo_range > 0 else 0.0
    tempo_norm = np.clip(tempo_norm, 0.0, 1.0)

    energy_range = ENERGY_MAX - ENERGY_MIN
    energy_norm = (energy - ENERGY_MIN) / energy_range if energy_range > 0 else 0.0
    energy_norm = np.clip(energy_norm, 0.0, 1.0)

    tempo_val = tempo_norm
    energy_val = energy_norm

    mood_scores_for_vector = np.zeros(len(mood_labels_list))
    if mood_str:
        for pair in mood_str.split(","):
            if ":" not in pair:
                continue
            label, score_str = pair.split(":")
            if label in mood_labels_list:
                try:
                    mood_scores_for_vector[mood_labels_list.index(label)] = float(score_str)
                except ValueError:
                    continue

    other_feature_scores_for_vector = np.zeros(len(other_feature_labels_list))
    other_features_str = row.get('other_features', "")
    if other_features_str:
        for pair in other_features_str.split(","):
            if ":" not in pair: continue
            label, score_str = pair.split(":")
            if label in other_feature_labels_list:
                try: other_feature_scores_for_vector[other_feature_labels_list.index(label)] = float(score_str)
                except ValueError: continue
    full_vector = [tempo_val, energy_val] + list(mood_scores_for_vector) + list(other_feature_scores_for_vector)
    return full_vector
