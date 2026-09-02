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
* CLAP text search asks for both filters and reuses the audio threshold, the
  space it measures like
* The hyperbolic result filter measures in arccosh units, so a cosine-sized
  0.01 keeps tracks that 0.30 removes
* The hyperbolic filter is handed the projections its caller already fetched
  rather than reading them back from the database a second time
* The journey drops a near-duplicate pick instead of shortening by a hole
* The journey window follows walk order: seeded with the START song, and the
  destination compared against the final pick
* A candidate rejected at one step stays available at every later step
* A distance-rejected candidate consumes neither an artist slot nor a name
* Untitled tracks by one artist are not folded onto each other
* Vectors are read back in one batch, and an index that cannot batch degrades to
  keeping every result rather than dropping them
* All three artist caps agree: a track with no author is EXEMPT from the cap,
  never dropped, in apply_artist_cap, build_capped_results and the journey
* The other dedup paths carry the same fixes: dedup_by_content and the journey
  never fold untitled tracks together, and SemGrove rejects on distance before a
  candidate claims its name or an artist slot
"""

from unittest.mock import patch

import numpy as np
import pytest

import config
from tasks.search_shaping import build_capped_results, cosine_duplicate_window


class _FakeIndex:
    def __init__(self, vectors):
        self.vectors = vectors
        self.get_vector_calls = 0
        self.get_vectors_calls = 0

    def get_vector(self, vid):
        self.get_vector_calls += 1
        return self.vectors[int(vid)]

    def get_vectors(self, vids):
        self.get_vectors_calls += 1
        return {int(v): self.vectors[int(v)] for v in vids if int(v) in self.vectors}

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

    def test_untitled_tracks_by_the_same_artist_are_not_collapsed_onto_the_first(self):
        idx = _index({0: [1.0, 0.0], 1: [0.0, 1.0], 2: [0.5, 0.5]})
        results = build_capped_results(
            idx, {0: 'a', 1: 'b', 2: 'c'},
            _meta([('a', '', 'Aphex Twin'), ('b', '', 'Aphex Twin'), ('c', '', 'Aphex Twin')]),
            [0, 1, 2], [0.1, 0.2, 0.3], 10, 0, dedup_names=True,
        )

        assert [r['item_id'] for r in results] == ['a', 'b', 'c']

    def test_tracks_with_no_metadata_at_all_are_not_collapsed_onto_the_first(self):
        idx = _index({0: [1.0, 0.0], 1: [0.0, 1.0], 2: [0.5, 0.5]})
        results = build_capped_results(
            idx, {0: 'a', 1: 'b', 2: 'c'},
            _meta([('a', '', ''), ('b', '', ''), ('c', '', '')]),
            [0, 1, 2], [0.1, 0.2, 0.3], 10, 0, dedup_names=True,
        )

        assert [r['item_id'] for r in results] == ['a', 'b', 'c']

    def test_a_missing_author_alone_still_dedups_on_the_title(self):
        idx = _index({0: [1.0, 0.0], 1: [0.0, 1.0]})
        results = build_capped_results(
            idx, {0: 'a', 1: 'b'},
            _meta([('a', 'Song', ''), ('b', 'song', '')]),
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
        assert idx.get_vectors_calls == 0

    def test_the_vectors_are_read_in_one_batch_not_one_call_per_candidate(self):
        idx = _index({i: [1.0, float(i)] for i in range(6)})
        build_capped_results(
            idx, {i: 'id%d' % i for i in range(6)},
            _meta([('id%d' % i, 'T%d' % i, 'A%d' % i) for i in range(6)]),
            list(range(6)), [0.1] * 6, 10, 0,
            dedup_names=True, dup_threshold=0.05, lookback=1,
        )

        assert idx.get_vectors_calls == 1
        assert idx.get_vector_calls == 0

    def test_an_index_that_cannot_batch_keeps_every_result_instead_of_dropping_them(self):
        class _NoBatch(_FakeIndex):
            def get_vectors(self, vids):
                raise AttributeError('no get_vectors on this index')

        idx = _NoBatch({0: [1.0, 0.0], 1: [1.0, 0.0]})
        results = build_capped_results(
            idx, {0: 'a', 1: 'b'},
            _meta([('a', 'One', 'X'), ('b', 'Two', 'Y')]),
            [0, 1], [0.1, 0.2], 10, 0,
            dedup_names=True, dup_threshold=0.05, lookback=1,
        )

        assert [r['item_id'] for r in results] == ['a', 'b']

    def test_a_distance_rejected_candidate_does_not_consume_an_artist_slot(self):
        idx = _index({
            0: [1.0, 0.0],
            1: [1.0, 0.0005],
            2: [0.0, 1.0],
            3: [1.0, 1.0],
        })
        results = build_capped_results(
            idx, {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
            _meta([('a', 'One', 'X'), ('b', 'Two', 'X'),
                   ('c', 'Three', 'X'), ('d', 'Four', 'X')]),
            [0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4], 10, 3,
            dedup_names=True, dup_threshold=0.05, lookback=1,
        )

        assert [r['item_id'] for r in results] == ['a', 'c', 'd']

    def test_a_distance_rejected_candidate_does_not_claim_its_name(self):
        idx = _index({0: [1.0, 0.0], 1: [1.0, 0.0005], 2: [0.0, 1.0]})
        results = build_capped_results(
            idx, {0: 'a', 1: 'b', 2: 'c'},
            _meta([('a', 'One', 'X'), ('b', 'Repeat', 'Y'), ('c', 'Repeat', 'Y')]),
            [0, 1, 2], [0.1, 0.2, 0.3], 10, 0,
            dedup_names=True, dup_threshold=0.05, lookback=1,
        )

        assert [r['item_id'] for r in results] == ['a', 'c']

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

    def test_a_zero_threshold_window_is_inactive_and_never_matches(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        window = cosine_duplicate_window(0.0, 1)
        window.remember(a)

        assert window.active is False
        assert window.is_duplicate(a) is False


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
        with patch.object(config, 'DUPLICATE_DISTANCE_THRESHOLD_HYPERBOLIC', threshold), \
                patch.object(config, 'DUPLICATE_DISTANCE_CHECK_LOOKBACK', 1):
            return [r['item_id'] for r in
                    hyperbolic_manager._filter_hyperbolic_near_duplicates(
                        results, self._rows())]

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
        with patch.object(config, 'DUPLICATE_DISTANCE_THRESHOLD_HYPERBOLIC', 0.30), \
                patch.object(config, 'DUPLICATE_DISTANCE_CHECK_LOOKBACK', 1):
            kept = hyperbolic_manager._filter_hyperbolic_near_duplicates(
                results, self._rows())

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

    def test_the_window_is_seeded_with_the_start_song_not_the_destination(self):
        from tasks import hyperbolic_journey_manager as jm

        candidate_ids = ['near_start', 'far']
        candidate_vecs = np.array([[0.101, 0.0], [0.0, 0.60]], dtype=np.float32)
        details = {
            'near_start': {'item_id': 'near_start', 'title': 'Near Start', 'author': 'A'},
            'far': {'item_id': 'far', 'title': 'Far', 'author': 'B'},
        }
        interior = np.array([[0.102, 0.0]], dtype=np.float32)
        start_vec = np.array([0.10, 0.0], dtype=np.float32)
        end_vec = np.array([0.0, 0.85], dtype=np.float32)

        with patch.object(config, 'DUPLICATE_DISTANCE_THRESHOLD_HYPERBOLIC', 0.30), \
                patch.object(config, 'DUPLICATE_DISTANCE_CHECK_LOOKBACK', 1), \
                patch.object(config, 'MAX_SONGS_PER_ARTIST', 3):
            picks = jm._pick_steps(
                interior, candidate_ids, candidate_vecs, details, [], [start_vec, end_vec]
            )

        assert [p['item_id'] for p in picks] == ['far']

    def test_a_candidate_rejected_at_one_step_is_still_available_at_a_later_step(self):
        from tasks import hyperbolic_journey_manager as jm

        candidate_ids = ['shadow', 'other']
        candidate_vecs = np.array([[0.101, 0.0], [0.0, 0.60]], dtype=np.float32)
        details = {
            'shadow': {'item_id': 'shadow', 'title': 'Shadow', 'author': 'A'},
            'other': {'item_id': 'other', 'title': 'Other', 'author': 'B'},
        }
        interior = np.array([[0.102, 0.0], [0.0, 0.61]], dtype=np.float32)
        start_vec = np.array([0.10, 0.0], dtype=np.float32)

        with patch.object(config, 'DUPLICATE_DISTANCE_THRESHOLD_HYPERBOLIC', 0.30), \
                patch.object(config, 'DUPLICATE_DISTANCE_CHECK_LOOKBACK', 1), \
                patch.object(config, 'MAX_SONGS_PER_ARTIST', 3):
            picks = jm._pick_steps(
                interior, candidate_ids, candidate_vecs, details, [], [start_vec]
            )

        assert [p['item_id'] for p in picks] == ['other', 'shadow']

    def test_a_final_pick_that_shadows_the_destination_is_dropped(self):
        from tasks import hyperbolic_journey_manager as jm

        picks = [{'item_id': 'twin', 'column': 0}]
        candidate_vecs = np.array([[0.0, 0.601]], dtype=np.float32)
        end_vec = np.array([0.0, 0.60], dtype=np.float32)

        with patch.object(config, 'DUPLICATE_DISTANCE_THRESHOLD_HYPERBOLIC', 0.30), \
                patch.object(config, 'DUPLICATE_DISTANCE_CHECK_LOOKBACK', 1):
            from tasks.hyperbolic_manager import hyperbolic_duplicate_window

            window = hyperbolic_duplicate_window()
            kept = jm._drop_last_pick_if_it_shadows_the_destination(
                picks, candidate_vecs, end_vec, window
            )

        assert kept == []

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


class TestClapTextSearchGetsTheSameTreatment:
    def test_the_clap_threshold_matches_the_audio_one_it_was_measured_against(self):
        assert config.DUPLICATE_DISTANCE_THRESHOLD_COSINE_CLAP == pytest.approx(
            config.DUPLICATE_DISTANCE_THRESHOLD_COSINE
        )

    def test_the_clap_query_asks_for_both_the_name_and_the_distance_filter(self):
        import inspect

        from tasks import clap_text_search

        source = inspect.getsource(clap_text_search._query_clap_index)
        assert 'dedup_names=True' in source
        assert 'DUPLICATE_DISTANCE_THRESHOLD_COSINE_CLAP' in source
        assert 'DUPLICATE_DISTANCE_CHECK_LOOKBACK' in source

    def test_every_filtered_search_shares_one_over_fetch_formula(self):
        import inspect

        from tasks import clap_text_search, lyrics_manager
        from tasks.search_shaping import overfetch_size

        for source in (
            inspect.getsource(clap_text_search.search_by_text),
            inspect.getsource(lyrics_manager.search_by_text),
            inspect.getsource(lyrics_manager.search_by_axes),
        ):
            assert 'overfetch_size(limit)' in source
            assert 'limit * 4' not in source

        assert overfetch_size(50) == 251
        assert overfetch_size(1) == 22


class TestTheOtherDedupPathsCarryTheSameFixes:
    def test_dedup_by_content_does_not_fold_untitled_tracks_together(self):
        from tasks.search_shaping import dedup_by_content

        songs = [{'item_id': 'a'}, {'item_id': 'b'}, {'item_id': 'c'}]
        details = {
            'a': {'item_id': 'a', 'title': '', 'author': 'Aphex Twin'},
            'b': {'item_id': 'b', 'title': '', 'author': 'Aphex Twin'},
            'c': {'item_id': 'c', 'title': '', 'author': ''},
        }

        assert [s['item_id'] for s in dedup_by_content(songs, details)] == ['a', 'b', 'c']

    def test_dedup_by_content_still_drops_a_real_repeat(self):
        from tasks.search_shaping import dedup_by_content

        songs = [{'item_id': 'a'}, {'item_id': 'b'}]
        details = {
            'a': {'item_id': 'a', 'title': 'Song', 'author': 'Artist'},
            'b': {'item_id': 'b', 'title': ' song ', 'author': 'ARTIST'},
        }

        assert [s['item_id'] for s in dedup_by_content(songs, details)] == ['a']

    def test_semgrove_runs_every_rejection_test_before_writing_a_counter(self):
        import inspect

        from tasks import sem_grove_manager

        source = inspect.getsource(sem_grove_manager._collect_search_results)
        name_claim = source.index('seen_names.add(name_key)')
        distance_check = source.index('window.is_duplicate(candidate_unit)')
        assert distance_check < name_claim, (
            'SemGrove must reject on distance before a candidate claims its name '
            'or an artist slot'
        )

    def test_semgrove_uses_the_shared_window_and_name_key(self):
        import inspect

        from tasks import sem_grove_manager

        source = inspect.getsource(sem_grove_manager._collect_search_results)
        assert 'cosine_duplicate_window' in source
        assert 'name_key_for' in source
        assert 'read_unit_vectors' in source

    def test_the_journey_does_not_fold_untitled_candidates_onto_each_other(self):
        from tasks import hyperbolic_journey_manager as jm

        candidate_ids = ['x', 'y']
        candidate_vecs = np.array([[0.0, 0.60], [0.60, 0.0]], dtype=np.float32)
        details = {
            'x': {'item_id': 'x', 'title': '', 'author': 'A'},
            'y': {'item_id': 'y', 'title': '', 'author': 'B'},
        }
        interior = np.array([[0.0, 0.61], [0.61, 0.0]], dtype=np.float32)

        with patch.object(config, 'DUPLICATE_DISTANCE_THRESHOLD_HYPERBOLIC', 0.30), \
                patch.object(config, 'DUPLICATE_DISTANCE_CHECK_LOOKBACK', 1), \
                patch.object(config, 'MAX_SONGS_PER_ARTIST', 3):
            picks = jm._pick_steps(interior, candidate_ids, candidate_vecs, details, [], None)

        assert sorted(p['item_id'] for p in picks) == ['x', 'y']


class TestEveryArtistCapAgreesOnTheNoAuthorCase:
    def test_apply_artist_cap_exempts_an_untagged_track_instead_of_dropping_it(self):
        from tasks.search_shaping import apply_artist_cap

        songs = [{'item_id': 'a'}, {'item_id': 'b'}, {'item_id': 'c'}]
        authors = {'a': 'Artist', 'b': '', 'c': None}

        with patch.object(config, 'MAX_SONGS_PER_ARTIST', 3):
            kept = apply_artist_cap(songs, lambda s: authors[s['item_id']])

        assert [s['item_id'] for s in kept] == ['a', 'b', 'c']

    def test_apply_artist_cap_still_caps_a_named_artist(self):
        from tasks.search_shaping import apply_artist_cap

        songs = [{'item_id': str(i)} for i in range(5)]

        with patch.object(config, 'MAX_SONGS_PER_ARTIST', 3):
            kept = apply_artist_cap(songs, lambda s: 'Artist')

        assert [s['item_id'] for s in kept] == ['0', '1', '2']

    def test_untagged_tracks_are_not_counted_against_each_other(self):
        from tasks.search_shaping import apply_artist_cap

        songs = [{'item_id': str(i)} for i in range(6)]

        with patch.object(config, 'MAX_SONGS_PER_ARTIST', 3):
            kept = apply_artist_cap(songs, lambda s: '')

        assert len(kept) == 6

    def test_build_capped_results_exempts_an_untagged_track_too(self):
        idx = _index({i: [1.0, float(i)] for i in range(5)})
        results = build_capped_results(
            idx, {i: 'id%d' % i for i in range(5)},
            _meta([('id%d' % i, 'T%d' % i, '') for i in range(5)]),
            list(range(5)), [0.1] * 5, 10, 3,
        )

        assert len(results) == 5

    def test_the_journey_exempts_an_untagged_candidate_instead_of_rejecting_it(self):
        from tasks import hyperbolic_journey_manager as jm

        candidate_ids = ['p', 'q']
        candidate_vecs = np.array([[0.0, 0.60], [0.60, 0.0]], dtype=np.float32)
        details = {
            'p': {'item_id': 'p', 'title': 'P', 'author': ''},
            'q': {'item_id': 'q', 'title': 'Q', 'author': None},
        }
        interior = np.array([[0.0, 0.61], [0.61, 0.0]], dtype=np.float32)

        with patch.object(config, 'DUPLICATE_DISTANCE_THRESHOLD_HYPERBOLIC', 0.30), \
                patch.object(config, 'DUPLICATE_DISTANCE_CHECK_LOOKBACK', 1), \
                patch.object(config, 'MAX_SONGS_PER_ARTIST', 3):
            picks = jm._pick_steps(interior, candidate_ids, candidate_vecs, details, [], None)

        assert sorted(p['item_id'] for p in picks) == ['p', 'q']

    def test_no_cap_implementation_drops_a_track_for_having_no_author(self):
        import inspect

        from tasks import hyperbolic_journey_manager, search_shaping

        cap_source = inspect.getsource(search_shaping.apply_artist_cap)
        assert 'capped.append(song)\n            continue' in cap_source, (
            'apply_artist_cap must keep an untagged track, not skip it'
        )

        chooser = inspect.getsource(hyperbolic_journey_manager._choose_candidate)
        assert 'not author or' not in chooser, (
            'the journey must exempt an untagged candidate from the cap, not reject it'
        )


class TestEachSearchPathPassesItsOwnThreshold:
    @pytest.mark.parametrize('name,expected', [
        ('DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS_TEXT', 0.05),
        ('DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS_AXIS', 0.0),
        ('DUPLICATE_DISTANCE_THRESHOLD_HYPERBOLIC', 0.30),
        ('DUPLICATE_DISTANCE_THRESHOLD_COSINE_CLAP', 0.01),
    ])
    def test_the_measured_default_is_the_shipped_default(self, name, expected):
        assert getattr(config, name) == pytest.approx(expected)

    @pytest.mark.parametrize('name', [
        'DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS_TEXT',
        'DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS_AXIS',
        'DUPLICATE_DISTANCE_THRESHOLD_HYPERBOLIC',
        'DUPLICATE_DISTANCE_THRESHOLD_COSINE_CLAP',
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
