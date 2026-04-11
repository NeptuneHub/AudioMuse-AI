# tasks/commons.py

import numpy as np

# Import necessary constants from config
import config

def score_vector(row, mood_labels_list, other_feature_labels_list): # other_feature_labels_list is now passed
    """Converts a database row into a numerical feature vector for clustering."""
    # Extract features from the database row
    tempo = float(row['tempo']) if row['tempo'] is not None else 0.0
    energy = float(row['energy']) if row['energy'] is not None else 0.0 # Get energy
    mood_str = row['mood_vector'] or ""
    
    # Normalize tempo to 0-1 range
    tempo_range = config.TEMPO_MAX_BPM - config.TEMPO_MIN_BPM
    tempo_norm = (tempo - config.TEMPO_MIN_BPM) / tempo_range if tempo_range > 0 else 0.0
    tempo_norm = np.clip(tempo_norm, 0.0, 1.0)

    # Normalize energy to 0-1 range
    energy_range = config.ENERGY_MAX - config.ENERGY_MIN
    energy_norm = (energy - config.ENERGY_MIN) / energy_range if energy_range > 0 else 0.0
    energy_norm = np.clip(energy_norm, 0.0, 1.0)

    tempo_val = tempo_norm
    energy_val = energy_norm

    mood_scores_for_vector = np.zeros(len(mood_labels_list)) # Initialize vector for all moods
    if mood_str:
        for pair in mood_str.split(","):
            if ":" not in pair:
                continue
            label, score_str = pair.split(":")
            if label in mood_labels_list:
                try:
                    mood_scores_for_vector[mood_labels_list.index(label)] = float(score_str) # Populate all mood scores
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