# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Coordinate and mood helpers for the song map in app_map.

Covers _pick_top_mood, _round_coord, and _sample_items used to build the
2D projection payload sent to the map view.

Main Features:
* _pick_top_mood returns the highest-scoring label or "unknown" on bad input
* _round_coord rounds to three decimals and zeroes out malformed coordinates
* _sample_items samples a deterministic fraction, returning a fresh list
"""

from app_map import _pick_top_mood, _round_coord, _sample_items


class TestPickTopMood:
    def test_returns_highest_scoring_label(self):
        assert _pick_top_mood('happy:0.8,sad:0.2') == 'happy'

    def test_empty_string_returns_unknown(self):
        assert _pick_top_mood('') == 'unknown'

    def test_none_returns_unknown(self):
        assert _pick_top_mood(None) == 'unknown'

    def test_no_colon_parts_returns_unknown(self):
        assert _pick_top_mood('justalabel') == 'unknown'

    def test_unparseable_score_treated_as_zero(self):
        assert _pick_top_mood('happy:abc,sad:0.2') == 'sad'

    def test_single_unparseable_score_still_returns_label(self):
        assert _pick_top_mood('happy:abc') == 'happy'


class TestRoundCoord:
    def test_rounds_to_three_decimals(self):
        assert _round_coord([1.23456789, 2.98765432]) == [1.235, 2.988]

    def test_non_numeric_entries_return_zeros(self):
        assert _round_coord(['a', 'b']) == [0.0, 0.0]

    def test_none_returns_zeros(self):
        assert _round_coord(None) == [0.0, 0.0]

    def test_too_short_returns_zeros(self):
        assert _round_coord([1.0]) == [0.0, 0.0]


class TestSampleItems:
    def test_deterministic_for_same_input(self):
        items = list(range(40))
        assert _sample_items(items, 0.5) == _sample_items(items, 0.5)

    def test_fraction_075_of_100_returns_75(self):
        items = list(range(100))
        assert len(_sample_items(items, 0.75)) == 75

    def test_empty_list_returns_empty(self):
        assert _sample_items([], 0.5) == []

    def test_fraction_one_returns_all_items(self):
        items = list(range(10))
        result = _sample_items(items, 1.0)
        assert result == items
        assert result is not items
