"""Unit tests for tasks/commons.py"""
import pytest
import numpy as np
from tasks.commons import score_vector


class TestScoreVector:
    """Tests for the score_vector function"""

    def test_basic_score_vector(self):
        """Test score_vector with basic inputs"""
        row = {
            'tempo': 120.0,
            'energy': 0.5,
            'mood_vector': "happy:0.8,sad:0.2",
            'other_features': "danceability:0.7"
        }
        mood_labels = ["happy", "sad", "energetic"]
        other_labels = ["danceability", "aggressive"]

        result = score_vector(row, mood_labels, other_labels)

        # Should return a list with tempo, energy, moods, and other features
        assert isinstance(result, list)
        assert len(result) == 2 + len(mood_labels) + len(other_labels)

    def test_score_vector_with_none_values(self):
        """Test score_vector handles None values gracefully"""
        row = {
            'tempo': None,
            'energy': None,
            'mood_vector': None,
            'other_features': ""
        }
        mood_labels = ["happy", "sad"]
        other_labels = ["danceability"]

        result = score_vector(row, mood_labels, other_labels)

        # Should still return valid vector with zeros
        assert len(result) == 2 + len(mood_labels) + len(other_labels)
        # Tempo and energy should be 0 when None
        assert result[0] == 0.0  # tempo
        assert result[1] == 0.0  # energy

    def test_score_vector_normalization(self):
        """Test that tempo and energy are normalized to 0-1 range"""
        row = {
            'tempo': 120.0,
            'energy': 0.5,
            'mood_vector': "",
            'other_features': ""
        }
        mood_labels = []
        other_labels = []

        result = score_vector(row, mood_labels, other_labels)

        # Tempo and energy should be between 0 and 1
        assert 0.0 <= result[0] <= 1.0
        assert 0.0 <= result[1] <= 1.0

    def test_score_vector_mood_parsing(self):
        """Test mood vector parsing"""
        row = {
            'tempo': 100.0,
            'energy': 0.3,
            'mood_vector': "happy:0.9,sad:0.1,energetic:0.5",
            'other_features': ""
        }
        mood_labels = ["happy", "sad", "energetic"]
        other_labels = []

        result = score_vector(row, mood_labels, other_labels)

        # Check mood scores are in the result (after tempo and energy)
        mood_scores = result[2:2+len(mood_labels)]
        assert mood_scores[0] == 0.9  # happy
        assert mood_scores[1] == 0.1  # sad
        assert mood_scores[2] == 0.5  # energetic

    def test_score_vector_ignores_unknown_moods(self):
        """Test that unknown mood labels are ignored"""
        row = {
            'tempo': 100.0,
            'energy': 0.3,
            'mood_vector': "happy:0.8,unknown_mood:0.5",
            'other_features': ""
        }
        mood_labels = ["happy", "sad"]
        other_labels = []

        result = score_vector(row, mood_labels, other_labels)

        # Should only include happy, sad should be 0
        mood_scores = result[2:2+len(mood_labels)]
        assert mood_scores[0] == 0.8  # happy
        assert mood_scores[1] == 0.0  # sad (not in input)

    def test_score_vector_invalid_mood_format(self):
        """Test handling of malformed mood strings"""
        row = {
            'tempo': 100.0,
            'energy': 0.3,
            'mood_vector': "happy:not_a_number,sad:0.5,invalid_no_colon",
            'other_features': ""
        }
        mood_labels = ["happy", "sad"]
        other_labels = []

        result = score_vector(row, mood_labels, other_labels)

        # Should skip invalid entries
        mood_scores = result[2:2+len(mood_labels)]
        assert mood_scores[0] == 0.0  # happy (invalid number)
        assert mood_scores[1] == 0.5  # sad (valid)
