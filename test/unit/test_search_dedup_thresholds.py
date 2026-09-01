# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the per-space duplicate filtering added to the search paths.

The lyrics, axis and hyperbolic searches each rank in their own geometry, so
each carries its own threshold rather than sharing the audio one. These tests
pin the behaviour that a shared number cannot express: that the axis default
of 0.0 really disables the distance pass, that the hyperbolic threshold is
read in arccosh units where a cosine-sized value would never fire, and that
the name filter is applied everywhere regardless of the distance setting.

Main Features:
* build_capped_results drops a second track with the same title and author
* build_capped_results drops a vector-space near duplicate inside the lookback
* A zero threshold or a zero lookback leaves every neighbour in place
* The distance pass never consults the index when it is disabled
* The lyrics text search passes its own threshold, the axis search passes its
  own, and the two are read from config rather than hard coded
* The hyperbolic result filter measures in arccosh units, so a cosine-sized
  0.01 keeps tracks that 0.30 removes
* The journey drops a near-duplicate pick instead of shortening by a hole
"""

from unittest.mock import patch

import numpy as np
import pytest

import config
from tasks.search_shaping import build_capped_results, is_near_duplicate_vector


class _FakeIndex:
    def __init__(self, vectors):
        self.vectors = vectors
        self.get_vector_calls = 0

    def get_vector(self, vid):
        self.get_vector_calls += 1
        return self.vectors[int(vid)]

    def distance_to_similarity(self, dist):
        return 1.0 - float(dist)


def _index(vectors):
    return _FakeIndex(vectors)


def _meta(rows):
    return {item_id: {'title': t, 'author': a, 'album': ''} for item_id, t, a in rows}


class TestNameFilterOnTheLyricsStyleResultBuilder:
    def test_a_second_track_with_the_same_title_and_author_is_dropped(self):
        idx = _index({0: [1.0, 0.0], 1: [0.0, 1.0]})
        results = build_capped_results(
            idx, {0: 'a', 1: 'b'},
            _meta([('a', 'Song', 'Artist'), ('b', 'song', ' artist ')]),
            [0, 1], [0.1, 0.2], 10, 0, dedup_names=True,
        )

        assert [r['item_id'] for r in results] == ['a']

    def test_without_the_flag_the_same_title_and_author_is_still_returned(self):
        idx = _index({0: [1.0, 0.0], 1: [0.0, 1.0]})
        results = build_capped_results(
            idx, {0: 'a', 1: 'b'},
            _meta([('a', 'Song', 'Artist'), ('b', 'Song', 'Artist')]),
            [0, 1], [0.1, 0.2], 10, 0,
        )

        assert [r['item_id'] for r in results] == ['a', 'b']


class TestDistanceFilterOnTheLyricsStyleResultBuilder:
    def test_a_near_duplicate_vector_inside_the_lookback_is_dropped(self):
        idx = _index({0: [1.0, 0.0], 1: [1.0, 0.001], 2: [0.0, 1.0]})
        results = build_capped_results(
            idx, {0: 'a', 1: 'b', 2: 'c'},
            _meta([('a', 'One', 'X'), ('b', 'Two', 'Y'), ('c', 'Three', 'Z')]),
            [0, 1, 2], [0.1, 0.2, 0.3], 10, 0,
            dedup_names=True, dup_threshold=0.05, lookback=1,
        )

        assert [r['item_id'] for r in results] == ['a', 'c']

    def test_a_zero_threshold_keeps_every_neighbour(self):
        idx = _index({0: [1.0, 0.0], 1: [1.0, 0.0], 2: [0.0, 1.0]})
        results = build_capped_results(
            idx, {0: 'a', 1: 'b', 2: 'c'},
            _meta([('a', 'One', 'X'), ('b', 'Two', 'Y'), ('c', 'Three', 'Z')]),
            [0, 1, 2], [0.1, 0.2, 0.3], 10, 0,
            dedup_names=True, dup_threshold=0.0, lookback=1,
        )

        assert [r['item_id'] for r in results] == ['a', 'b', 'c']

    def test_a_zero_lookback_keeps_every_neighbour(self):
        idx = _index({0: [1.0, 0.0], 1: [1.0, 0.0]})
        results = build_capped_results(
            idx, {0: 'a', 1: 'b'},
            _meta([('a', 'One', 'X'), ('b', 'Two', 'Y')]),
            [0, 1], [0.1, 0.2], 10, 0,
            dedup_names=True, dup_threshold=0.05, lookback=0,
        )

        assert [r['item_id'] for r in results] == ['a', 'b']

    def test_a_disabled_distance_pass_never_reads_a_vector_back_from_the_index(self):
        idx = _index({0: [1.0, 0.0], 1: [1.0, 0.0]})
        build_capped_results(
            idx, {0: 'a', 1: 'b'},
            _meta([('a', 'One', 'X'), ('b', 'Two', 'Y')]),
            [0, 1], [0.1, 0.2], 10, 0,
            dedup_names=True, dup_threshold=0.0, lookback=1,
        )

        assert idx.get_vector_calls == 0

    def test_a_vector_the_index_cannot_return_does_not_drop_the_track(self):
        class _Broken(_FakeIndex):
            def get_vector(self, vid):
                raise RuntimeError('cell missing')

        idx = _Broken({})
        results = build_capped_results(
            idx, {0: 'a'}, _meta([('a', 'One', 'X')]),
            [0], [0.1], 10, 0, dedup_names=True, dup_threshold=0.05, lookback=1,
        )

        assert [r['item_id'] for r in results] == ['a']


class TestTheAxisDefaultReallyDisablesTheDistancePass:
    def test_the_shipped_axis_threshold_is_zero(self):
        assert config.DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS_AXIS == 0.0

    def test_a_zero_threshold_is_never_a_near_duplicate_however_close(self):
        a = np.array([1.0, 0.0], dtype=np.float32)

        assert is_near_duplicate_vector(a, [a], 0.0) is False


class TestTheHyperbolicThresholdIsReadInArccoshUnits:
    def _rows(self):
        return {
            'a': (np.array([0.10, 0.0], dtype=np.float32), 0.10),
            'b': (np.array([0.11, 0.0], dtype=np.float32), 0.11),
            'c': (np.array([0.0, 0.60], dtype=np.float32), 0.60),
        }

    def _run(self, threshold):
        from tasks import hyperbolic_manager

        results = [{'item_id': i} for i in ('a', 'b', 'c')]
        with patch.object(hyperbolic_manager, '_fetch_poincare_rows', return_value=self._rows()), \
                patch.object(config, 'DUPLICATE_DISTANCE_THRESHOLD_HYPERBOLIC', threshold), \
                patch.object(config, 'DUPLICATE_DISTANCE_CHECK_LOOKBACK', 1):
            return [r['item_id'] for r in
                    hyperbolic_manager._filter_hyperbolic_near_duplicates(results)]

    def test_a_cosine_sized_threshold_removes_nothing_in_this_space(self):
        assert self._run(0.01) == ['a', 'b', 'c']

    def test_the_shipped_threshold_removes_the_near_duplicate(self):
        assert self._run(0.30) == ['a', 'c']

    def test_the_shipped_default_is_far_larger_than_the_cosine_defaults(self):
        assert config.DUPLICATE_DISTANCE_THRESHOLD_HYPERBOLIC > \
            10 * config.DUPLICATE_DISTANCE_THRESHOLD_COSINE

    def test_a_track_with_no_projection_row_is_kept_rather_than_dropped(self):
        from tasks import hyperbolic_manager

        results = [{'item_id': 'a'}, {'item_id': 'missing'}]
        with patch.object(hyperbolic_manager, '_fetch_poincare_rows', return_value=self._rows()), \
                patch.object(config, 'DUPLICATE_DISTANCE_THRESHOLD_HYPERBOLIC', 0.30), \
                patch.object(config, 'DUPLICATE_DISTANCE_CHECK_LOOKBACK', 1):
            kept = hyperbolic_manager._filter_hyperbolic_near_duplicates(results)

        assert [r['item_id'] for r in kept] == ['a', 'missing']


class TestTheJourneySkipsANearDuplicatePickInsteadOfLeavingAHole:
    def test_the_step_takes_the_next_candidate_when_the_closest_is_a_near_duplicate(self):
        from tasks import hyperbolic_journey_manager as jm

        candidate_ids = ['near', 'far']
        candidate_vecs = np.array([[0.101, 0.0], [0.0, 0.60]], dtype=np.float32)
        details = {
            'near': {'item_id': 'near', 'title': 'Near', 'author': 'A'},
            'far': {'item_id': 'far', 'title': 'Far', 'author': 'B'},
        }
        interior = np.array([[0.102, 0.0]], dtype=np.float32)
        seed_vecs = [np.array([0.10, 0.0], dtype=np.float32)]

        with patch.object(config, 'DUPLICATE_DISTANCE_THRESHOLD_HYPERBOLIC', 0.30), \
                patch.object(config, 'DUPLICATE_DISTANCE_CHECK_LOOKBACK', 1), \
                patch.object(config, 'MAX_SONGS_PER_ARTIST', 3):
            picks = jm._pick_steps(
                interior, candidate_ids, candidate_vecs, details, [], seed_vecs
            )

        assert [p['item_id'] for p in picks] == ['far']

    def test_without_seed_vectors_the_closest_candidate_is_still_taken(self):
        from tasks import hyperbolic_journey_manager as jm

        candidate_ids = ['near', 'far']
        candidate_vecs = np.array([[0.101, 0.0], [0.0, 0.60]], dtype=np.float32)
        details = {
            'near': {'item_id': 'near', 'title': 'Near', 'author': 'A'},
            'far': {'item_id': 'far', 'title': 'Far', 'author': 'B'},
        }
        interior = np.array([[0.102, 0.0]], dtype=np.float32)

        with patch.object(config, 'DUPLICATE_DISTANCE_THRESHOLD_HYPERBOLIC', 0.30), \
                patch.object(config, 'DUPLICATE_DISTANCE_CHECK_LOOKBACK', 1), \
                patch.object(config, 'MAX_SONGS_PER_ARTIST', 3):
            picks = jm._pick_steps(interior, candidate_ids, candidate_vecs, details, [], None)

        assert [p['item_id'] for p in picks] == ['near']


class TestEachSearchPathPassesItsOwnThreshold:
    @pytest.mark.parametrize('name,expected', [
        ('DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS_TEXT', 0.05),
        ('DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS_AXIS', 0.0),
        ('DUPLICATE_DISTANCE_THRESHOLD_HYPERBOLIC', 0.30),
    ])
    def test_the_measured_default_is_the_shipped_default(self, name, expected):
        assert getattr(config, name) == pytest.approx(expected)

    @pytest.mark.parametrize('name', [
        'DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS_TEXT',
        'DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS_AXIS',
        'DUPLICATE_DISTANCE_THRESHOLD_HYPERBOLIC',
    ])
    def test_the_operator_can_reach_it_from_the_setup_wizard(self, name):
        import app_setup

        assert app_setup.should_show_advanced(name) is True
        assert name not in config.SETUP_BOOTSTRAP_EXCLUDED_KEYS

    def test_the_lyrics_text_search_reads_the_text_threshold_not_the_axis_one(self):
        import inspect

        from tasks import lyrics_manager

        source = inspect.getsource(lyrics_manager.search_by_text)
        assert 'DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS_TEXT' in source
        assert 'DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS_AXIS' not in source

    def test_the_axis_search_reads_the_axis_threshold_not_the_text_one(self):
        import inspect

        from tasks import lyrics_manager

        source = inspect.getsource(lyrics_manager.search_by_axes)
        assert 'DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS_AXIS' in source
        assert 'DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS_TEXT' not in source
