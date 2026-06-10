import pytest

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
