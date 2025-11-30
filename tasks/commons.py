# tasks/commons.py

import numpy as np

# Import necessary constants from config
from config import (
    TEMPO_MAX_BPM, TEMPO_MIN_BPM,
    ENERGY_MAX, ENERGY_MIN
)

def score_vector(row, mood_labels_list, other_feature_labels_list): # other_feature_labels_list is now passed
    """Converts a database row into a numerical feature vector for clustering."""
    # Extract features from the database row
    tempo = float(row['tempo']) if row['tempo'] is not None else 0.0
    energy = float(row['energy']) if row['energy'] is not None else 0.0 # Get energy
    mood_str = row['mood_vector'] or ""
    
    # Normalize tempo to 0-1 range
    tempo_range = TEMPO_MAX_BPM - TEMPO_MIN_BPM
    tempo_norm = (tempo - TEMPO_MIN_BPM) / tempo_range if tempo_range > 0 else 0.0
    tempo_norm = np.clip(tempo_norm, 0.0, 1.0)

    # Normalize energy to 0-1 range
    energy_range = ENERGY_MAX - ENERGY_MIN
    energy_norm = (energy - ENERGY_MIN) / energy_range if energy_range > 0 else 0.0
    energy_norm = np.clip(energy_norm, 0.0, 1.0)

    tempo_val = tempo_norm
    energy_val = energy_norm

    # Build label-to-index lookup dictionaries for O(1) access instead of O(n) list.index() calls
    mood_label_to_index = {label: idx for idx, label in enumerate(mood_labels_list)}
    other_label_to_index = {label: idx for idx, label in enumerate(other_feature_labels_list)}

    mood_scores_for_vector = np.zeros(len(mood_labels_list)) # Initialize vector for all moods
    if mood_str:
        for pair in mood_str.split(","):
            if ":" not in pair:
                continue
            label, score_str = pair.split(":")
            idx = mood_label_to_index.get(label)
            if idx is not None:
                try:
                    mood_scores_for_vector[idx] = float(score_str) # Populate all mood scores
                except ValueError:
                    continue

    other_feature_scores_for_vector = np.zeros(len(other_feature_labels_list))
    other_features_str = row.get('other_features', "")
    if other_features_str:
        for pair in other_features_str.split(","):
            if ":" not in pair: continue
            label, score_str = pair.split(":")
            idx = other_label_to_index.get(label)
            if idx is not None:
                try: other_feature_scores_for_vector[idx] = float(score_str)
                except ValueError: continue
    full_vector = [tempo_val, energy_val] + list(mood_scores_for_vector) + list(other_feature_scores_for_vector)
    return full_vector