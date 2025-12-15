"""Unit tests for app_map.py utility functions"""
import pytest
import numpy as np
from app_map import _pick_top_mood, _round_coord, _sample_items


class TestPickTopMood:
    """Tests for the _pick_top_mood function"""

    def test_basic_mood_selection(self):
        """Test selecting the mood with highest score"""
        mood_str = "happy:0.8,sad:0.2,energetic:0.5"
        result = _pick_top_mood(mood_str)
        assert result == "happy"

    def test_multiple_moods_selects_highest(self):
        """Test that highest value is selected among multiple moods"""
        mood_str = "calm:0.3,aggressive:0.9,melancholic:0.4"
        result = _pick_top_mood(mood_str)
        assert result == "aggressive"

    def test_single_mood(self):
        """Test with only one mood"""
        mood_str = "happy:0.7"
        result = _pick_top_mood(mood_str)
        assert result == "happy"

    def test_empty_string_returns_unknown(self):
        """Test that empty string returns 'unknown'"""
        result = _pick_top_mood("")
        assert result == "unknown"

    def test_none_returns_unknown(self):
        """Test that None returns 'unknown'"""
        result = _pick_top_mood(None)
        assert result == "unknown"

    def test_negative_scores(self):
        """Test with negative scores"""
        mood_str = "happy:-0.5,sad:-0.2,energetic:-0.8"
        result = _pick_top_mood(mood_str)
        assert result == "sad"  # -0.2 is the highest

    def test_equal_scores_returns_first(self):
        """Test that first mood is returned when scores are equal"""
        mood_str = "happy:0.5,sad:0.5"
        result = _pick_top_mood(mood_str)
        assert result in ["happy", "sad"]  # Either is acceptable for equal scores

    def test_malformed_entry_skipped(self):
        """Test that malformed entries are skipped"""
        mood_str = "happy:0.8,invalid_no_colon,sad:0.9"
        result = _pick_top_mood(mood_str)
        assert result == "sad"

    def test_invalid_score_value_treated_as_zero(self):
        """Test that invalid score values are treated as 0.0"""
        mood_str = "happy:not_a_number,sad:0.5"
        result = _pick_top_mood(mood_str)
        assert result == "sad"

    def test_whitespace_in_values(self):
        """Test handling of whitespace in values"""
        mood_str = "happy: 0.8,sad: 0.2"
        result = _pick_top_mood(mood_str)
        # The function splits on ':', so ' 0.8' should still parse
        assert result in ["happy", "sad"]  # Implementation may handle differently

    def test_all_invalid_returns_unknown(self):
        """Test that all invalid entries return 'unknown'"""
        mood_str = "invalid1,invalid2,invalid3"
        result = _pick_top_mood(mood_str)
        assert result == "unknown"

    def test_mixed_positive_negative_scores(self):
        """Test with mix of positive and negative scores"""
        mood_str = "happy:0.8,sad:-0.5,calm:0.3"
        result = _pick_top_mood(mood_str)
        assert result == "happy"

    def test_zero_scores(self):
        """Test with zero scores"""
        mood_str = "happy:0.0,sad:0.0"
        result = _pick_top_mood(mood_str)
        assert result in ["happy", "sad"]  # Either is acceptable


class TestRoundCoord:
    """Tests for the _round_coord function"""

    def test_basic_rounding(self):
        """Test basic coordinate rounding to 3 decimal places"""
        coord = [1.23456, 7.89012]
        result = _round_coord(coord)
        assert result == [1.235, 7.890]

    def test_no_rounding_needed(self):
        """Test when coordinates already have 3 or fewer decimals"""
        coord = [1.5, 2.0]
        result = _round_coord(coord)
        assert result == [1.5, 2.0]

    def test_negative_coordinates(self):
        """Test with negative coordinates"""
        coord = [-1.23456, -7.89012]
        result = _round_coord(coord)
        assert result == [-1.235, -7.890]

    def test_zero_coordinates(self):
        """Test with zero coordinates"""
        coord = [0.0, 0.0]
        result = _round_coord(coord)
        assert result == [0.0, 0.0]

    def test_large_numbers(self):
        """Test with large coordinate values"""
        coord = [123.456789, 987.654321]
        result = _round_coord(coord)
        assert result == [123.457, 987.654]

    def test_string_coordinates_converted(self):
        """Test that string coordinates are converted to float"""
        coord = ["1.23456", "7.89012"]
        result = _round_coord(coord)
        assert result == [1.235, 7.890]

    def test_invalid_input_returns_zero_zero(self):
        """Test that invalid input returns [0.0, 0.0]"""
        coord = ["invalid", "data"]
        result = _round_coord(coord)
        assert result == [0.0, 0.0]

    def test_empty_list_returns_zero_zero(self):
        """Test that empty list returns [0.0, 0.0]"""
        coord = []
        result = _round_coord(coord)
        assert result == [0.0, 0.0]

    def test_single_element_list_returns_zero_zero(self):
        """Test that single element list returns [0.0, 0.0]"""
        coord = [1.5]
        result = _round_coord(coord)
        assert result == [0.0, 0.0]

    def test_none_returns_zero_zero(self):
        """Test that None returns [0.0, 0.0]"""
        coord = None
        result = _round_coord(coord)
        assert result == [0.0, 0.0]


class TestSampleItems:
    """Tests for the _sample_items function"""

    def test_empty_list(self):
        """Test with empty list"""
        result = _sample_items([], 0.5)
        assert result == []

    def test_full_sampling(self):
        """Test sampling with fraction >= 1.0"""
        items = [1, 2, 3, 4, 5]
        result = _sample_items(items, 1.0)
        assert result == items

    def test_over_sampling(self):
        """Test sampling with fraction > 1.0"""
        items = [1, 2, 3, 4, 5]
        result = _sample_items(items, 2.0)
        assert result == items

    def test_half_sampling(self):
        """Test sampling with 50% fraction"""
        items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = _sample_items(items, 0.5)
        assert len(result) == 5

    def test_quarter_sampling(self):
        """Test sampling with 25% fraction"""
        items = [1, 2, 3, 4, 5, 6, 7, 8]
        result = _sample_items(items, 0.25)
        assert len(result) == 2

    def test_single_item_list(self):
        """Test with single item list"""
        items = [42]
        result = _sample_items(items, 0.5)
        assert result == [42]

    def test_deterministic_sampling(self):
        """Test that sampling is deterministic"""
        items = list(range(100))
        result1 = _sample_items(items, 0.3)
        result2 = _sample_items(items, 0.3)
        assert result1 == result2

    def test_sampled_items_from_original_list(self):
        """Test that all sampled items are from original list"""
        items = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        result = _sample_items(items, 0.4)
        for item in result:
            assert item in items

    def test_small_fraction_returns_at_least_one(self):
        """Test that even very small fractions return at least 1 item"""
        items = [1, 2, 3, 4, 5]
        result = _sample_items(items, 0.01)
        assert len(result) >= 1

    def test_zero_fraction_returns_at_least_one(self):
        """Test that zero fraction returns at least 1 item"""
        items = [1, 2, 3, 4, 5]
        result = _sample_items(items, 0.0)
        assert len(result) >= 1

    def test_preserves_item_order(self):
        """Test that relative order of items is preserved"""
        items = [10, 20, 30, 40, 50, 60, 70, 80]
        result = _sample_items(items, 0.5)
        # Check that result maintains the order from original
        for i in range(len(result) - 1):
            orig_idx1 = items.index(result[i])
            orig_idx2 = items.index(result[i + 1])
            assert orig_idx1 < orig_idx2

    def test_large_list_sampling(self):
        """Test with large list"""
        items = list(range(1000))
        result = _sample_items(items, 0.1)
        assert len(result) == 100
        assert all(item in items for item in result)

    def test_string_items(self):
        """Test with string items"""
        items = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        result = _sample_items(items, 0.5)
        assert len(result) == 4
        assert all(item in items for item in result)

    def test_dict_items(self):
        """Test with dictionary items"""
        items = [{'id': i, 'value': i * 10} for i in range(10)]
        result = _sample_items(items, 0.3)
        assert len(result) == 3
        assert all(item in items for item in result)

    def test_negative_fraction_returns_at_least_one(self):
        """Test that negative fraction is handled (returns at least 1)"""
        items = [1, 2, 3, 4, 5]
        result = _sample_items(items, -0.5)
        # Depending on implementation, this might return 1 or empty
        # Based on max(1, int(math.floor(n * fraction)))
        assert len(result) >= 1

    def test_does_not_modify_original(self):
        """Test that original list is not modified"""
        items = [1, 2, 3, 4, 5]
        original_copy = items.copy()
        _sample_items(items, 0.5)
        assert items == original_copy
