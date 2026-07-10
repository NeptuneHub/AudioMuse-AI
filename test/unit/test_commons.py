# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Feature-vector construction in tasks.commons.score_vector.

Covers turning a score row (tempo, energy, mood and other-feature strings) into
the fixed-length numeric vector used for clustering and similarity.

Main Features:
* Vector length is 2 plus the mood and other-label counts, in label order
* None tempo/energy become 0.0 and values normalize into 0..1
* Mood parsing ignores unknown labels and skips malformed "label:value" entries
"""

from tasks.commons import score_vector


class TestScoreVector:
    def test_basic_score_vector(self):
        row = {
            'tempo': 120.0,
            'energy': 0.5,
            'mood_vector': "happy:0.8,sad:0.2",
            'other_features': "danceability:0.7",
        }
        mood_labels = ["happy", "sad", "energetic"]
        other_labels = ["danceability", "aggressive"]

        result = score_vector(row, mood_labels, other_labels)

        assert isinstance(result, list)
        assert len(result) == 2 + len(mood_labels) + len(other_labels)

    def test_score_vector_with_none_values(self):
        row = {'tempo': None, 'energy': None, 'mood_vector': None, 'other_features': ""}
        mood_labels = ["happy", "sad"]
        other_labels = ["danceability"]

        result = score_vector(row, mood_labels, other_labels)

        assert len(result) == 2 + len(mood_labels) + len(other_labels)
        assert result[0] == 0.0
        assert result[1] == 0.0

    def test_score_vector_normalization(self):
        row = {'tempo': 120.0, 'energy': 0.5, 'mood_vector': "", 'other_features': ""}
        mood_labels = []
        other_labels = []

        result = score_vector(row, mood_labels, other_labels)

        assert 0.0 <= result[0] <= 1.0
        assert 0.0 <= result[1] <= 1.0

    def test_score_vector_mood_parsing(self):
        row = {
            'tempo': 100.0,
            'energy': 0.3,
            'mood_vector': "happy:0.9,sad:0.1,energetic:0.5",
            'other_features': "",
        }
        mood_labels = ["happy", "sad", "energetic"]
        other_labels = []

        result = score_vector(row, mood_labels, other_labels)

        mood_scores = result[2 : 2 + len(mood_labels)]
        assert mood_scores[0] == 0.9
        assert mood_scores[1] == 0.1
        assert mood_scores[2] == 0.5

    def test_score_vector_ignores_unknown_moods(self):
        row = {
            'tempo': 100.0,
            'energy': 0.3,
            'mood_vector': "happy:0.8,unknown_mood:0.5",
            'other_features': "",
        }
        mood_labels = ["happy", "sad"]
        other_labels = []

        result = score_vector(row, mood_labels, other_labels)

        mood_scores = result[2 : 2 + len(mood_labels)]
        assert mood_scores[0] == 0.8
        assert mood_scores[1] == 0.0

    def test_score_vector_invalid_mood_format(self):
        row = {
            'tempo': 100.0,
            'energy': 0.3,
            'mood_vector': "happy:not_a_number,sad:0.5,invalid_no_colon",
            'other_features': "",
        }
        mood_labels = ["happy", "sad"]
        other_labels = []

        result = score_vector(row, mood_labels, other_labels)

        mood_scores = result[2 : 2 + len(mood_labels)]
        assert mood_scores[0] == 0.0
        assert mood_scores[1] == 0.5
