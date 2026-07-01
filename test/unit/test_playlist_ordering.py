# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Sonic ordering of playlist tracks by key, tempo and energy distance.

Covers the playlist_ordering helpers that reorder tracks for a smooth flow,
loaded in isolation via the conftest module importer.

Main Features:
* Key distance is scale-aware, case-insensitive and neutral for missing keys
* Composite distance combines tempo/energy with missing values treated as zero
* order_playlist leaves too-short inputs and empty input unchanged
"""

from test.unit.conftest import _import_module


def _load_playlist_ordering():
    return _import_module('tasks.playlist_ordering', 'tasks/playlist_ordering.py')


class TestKeyDistance:
    def test_identical_keys_same_scale(self):
        mod = _load_playlist_ordering()
        dist = mod._key_distance('C', 'major', 'C', 'major')
        assert dist == 0.0

    def test_adjacent_keys_without_scale_bonus(self):
        mod = _load_playlist_ordering()
        dist = mod._key_distance('C', None, 'G', None)
        assert abs(dist - 1 / 6) < 0.01

    def test_missing_key_returns_neutral(self):
        mod = _load_playlist_ordering()
        dist = mod._key_distance(None, None, 'C', None)
        assert dist == 0.5

    def test_unknown_key_returns_neutral(self):
        mod = _load_playlist_ordering()
        dist = mod._key_distance('C', None, 'XYZ', None)
        assert dist == 0.5

    def test_case_insensitive(self):
        mod = _load_playlist_ordering()
        dist1 = mod._key_distance('C', None, 'G', None)
        dist2 = mod._key_distance('c', None, 'g', None)
        assert abs(dist1 - dist2) < 0.01


class TestCompositeDistance:
    def test_identical_songs(self):
        mod = _load_playlist_ordering()
        song = {'tempo': 120, 'energy': 0.08, 'key': 'C', 'scale': 'major'}
        dist = mod._composite_distance(song, song)
        assert dist == 0.0

    def test_tempo_difference(self):
        mod = _load_playlist_ordering()
        song1 = {'tempo': 80, 'energy': 0.05, 'key': 'C', 'scale': 'major'}
        song2 = {'tempo': 160, 'energy': 0.05, 'key': 'C', 'scale': 'major'}
        dist = mod._composite_distance(song1, song2)
        assert abs(dist - 0.35) < 0.01

    def test_energy_capped_at_one(self):
        mod = _load_playlist_ordering()
        song1 = {'tempo': 100, 'energy': 0.01, 'key': 'C', 'scale': None}
        song2 = {'tempo': 100, 'energy': 0.15, 'key': 'C', 'scale': None}
        dist = mod._composite_distance(song1, song2)
        assert abs(dist - 0.35) < 0.01

    def test_missing_values_as_zero(self):
        mod = _load_playlist_ordering()
        song1 = {'tempo': None, 'energy': None, 'key': 'C', 'scale': None}
        song2 = {'tempo': 100, 'energy': 0.10, 'key': 'C', 'scale': None}
        dist = mod._composite_distance(song1, song2)
        assert dist > 0


class TestOrderPlaylist:
    def test_single_song_unchanged(self):
        mod = _load_playlist_ordering()
        result = mod.order_playlist(['only_id'])
        assert result == ['only_id']

    def test_two_songs_unchanged(self):
        mod = _load_playlist_ordering()
        result = mod.order_playlist(['id1', 'id2'])
        assert result == ['id1', 'id2']

    def test_empty_input(self):
        mod = _load_playlist_ordering()
        result = mod.order_playlist([])
        assert result == []

    def test_minimum_songs_no_ordering(self):
        _load_playlist_ordering()
