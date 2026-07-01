# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Radius-walk playlist generation with artist-adjacency constraints.

Covers the radius_walk helpers that build a walk-ordered playlist from candidate
neighbours while capping per-artist repeats and avoiding runs of one artist.

Main Features:
* avoid_triple_adjacent swaps a third same-artist track with a later different one
* execute_radius_walk returns dicts in walk order and honors empty input
* max-songs-per-artist of one prevents author repeats
* Duplicate title/author pairs are capped to one unless dedup is disabled
"""

import numpy as np

from tasks.radius_walk_helper import avoid_triple_adjacent, execute_radius_walk


def _euclid(v1, v2):
    return float(np.linalg.norm(v1 - v2))


def _assert_no_triple(ids, id_to_author):
    for i in range(len(ids) - 2):
        a1 = id_to_author.get(ids[i])
        a2 = id_to_author.get(ids[i + 1])
        a3 = id_to_author.get(ids[i + 2])
        assert not (a1 and a1 == a2 == a3)


class TestAvoidTripleAdjacent:
    def test_triple_swapped_with_later_different_artist(self):
        ids = ['a1', 'a2', 'a3', 'b1']
        id_to_author = {'a1': 'A', 'a2': 'A', 'a3': 'A', 'b1': 'B'}
        original = ids.copy()

        result = avoid_triple_adjacent(ids, id_to_author)

        assert result is ids
        assert sorted(result) == sorted(original)
        _assert_no_triple(result, id_to_author)

    def test_all_same_artist_unchanged(self):
        ids = ['a1', 'a2', 'a3', 'a4']
        id_to_author = {'a1': 'A', 'a2': 'A', 'a3': 'A', 'a4': 'A'}

        result = avoid_triple_adjacent(ids, id_to_author)

        assert result is ids
        assert result == ['a1', 'a2', 'a3', 'a4']

    def test_no_triple_order_identical(self):
        ids = ['a1', 'b1', 'a2', 'b2']
        id_to_author = {'a1': 'A', 'b1': 'B', 'a2': 'A', 'b2': 'B'}

        result = avoid_triple_adjacent(ids, id_to_author)

        assert result is ids
        assert result == ['a1', 'b1', 'a2', 'b2']


def _make_candidates():
    candidates = []
    for i in range(12):
        candidates.append(
            {
                'item_id': f's{i:02d}',
                'vector': np.array([float(i), 0.0, 0.0, 0.0], dtype=np.float32),
                'dist_anchor': round(i * 0.1, 1),
                'title': f'Title {i}',
                'author': f'art{i % 6}',
            }
        )
    return candidates


def _make_candidates_with_duplicate_pair():
    candidates = []
    for i in range(10):
        candidates.append(
            {
                'item_id': f's{i:02d}',
                'vector': np.array([float(i), 0.0, 0.0, 0.0], dtype=np.float32),
                'dist_anchor': round(i * 0.1, 1),
                'title': f'Track {i}',
                'author': f'solo{i}',
            }
        )
    for j, item_id in enumerate(['dup1', 'dup2']):
        candidates.append(
            {
                'item_id': item_id,
                'vector': np.array([float(10 + j), 0.0, 0.0, 0.0], dtype=np.float32),
                'dist_anchor': round((10 + j) * 0.1, 1),
                'title': 'Same Song',
                'author': 'dupart',
            }
        )
    return candidates


class TestExecuteRadiusWalk:
    def test_empty_candidates_returns_empty_list(self):
        assert execute_radius_walk([], 5, get_distance_fn=_euclid) == []

    def test_returns_dicts_walk_order_and_length(self):
        candidates = _make_candidates()
        dist_map = {c['item_id']: c['dist_anchor'] for c in candidates}

        result = execute_radius_walk(candidates, 5, get_distance_fn=_euclid)

        assert isinstance(result, list)
        assert len(result) == 5
        for entry in result:
            assert isinstance(entry, dict)
            assert set(entry.keys()) == {'item_id', 'distance'}
            assert entry['distance'] == dist_map[entry['item_id']]
        assert [e['item_id'] for e in result] == ['s00', 's01', 's02', 's03', 's04']

    def test_max_songs_per_artist_one_no_author_repeats(self):
        candidates = _make_candidates()
        id_to_author = {c['item_id']: c['author'] for c in candidates}

        result = execute_radius_walk(
            candidates,
            10,
            eliminate_duplicates=True,
            max_songs_per_artist=1,
            get_distance_fn=_euclid,
        )

        assert len(result) <= 10
        assert len(result) == 6
        authors = [id_to_author[e['item_id']] for e in result]
        assert len(authors) == len(set(authors))

    def test_duplicate_title_author_pair_capped_to_one(self):
        candidates = _make_candidates_with_duplicate_pair()

        result = execute_radius_walk(
            candidates,
            12,
            eliminate_duplicates=True,
            max_songs_per_artist=1,
            get_distance_fn=_euclid,
        )

        result_ids = {e['item_id'] for e in result}
        assert len(result_ids & {'dup1', 'dup2'}) == 1
        assert len(result) == 11

    def test_eliminate_duplicates_without_cap_keeps_both_duplicates(self):
        candidates = _make_candidates_with_duplicate_pair()

        result = execute_radius_walk(
            candidates,
            12,
            eliminate_duplicates=True,
            max_songs_per_artist=None,
            get_distance_fn=_euclid,
        )

        result_ids = {e['item_id'] for e in result}
        assert {'dup1', 'dup2'} <= result_ids
        assert len(result) == 12
