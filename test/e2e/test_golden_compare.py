# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Contract of the golden comparison itself, checked without booting anything.

The recorded answers are compared on machines that never produced them, so
the comparison has to tell a regression from the last-digit drift a different
CPU puts into whatever is derived from analyzing real audio. These cases pin
both sides: what must pass (a swap between two near-tied songs, a song
entering at the tied end of a list, a score off by a few thousandths, a mood
label entering at the bottom of the top five) and what must keep failing (a
different nearest song, a swap across a real gap, a changed title, a changed
count, a missing key).

Main Features:
* ranked lists are compared as rankings, keyed by song, tolerant of near ties
* numbers are compared within a hundredth, integers exactly
* tag maps tolerate a label entering or leaving at their lowest score
* every genuine change is still reported with its path
"""

import copy

import pytest

from test.e2e.golden import Resolver, differences, normalize

pytestmark = pytest.mark.e2e


def _row(name, similarity, **extra):
    row = {'item_id': f'track:{name}', 'title': name, 'similarity': similarity}
    row.update(extra)
    return row


RECORDED = [
    _row('Missing Person', 0.654),
    _row('What Im sayin', 0.558),
    _row('Un week-end avec papy', 0.552),
    _row('Pizza', 0.470),
    _row('It Came Upon A Midnight Clear', 0.401),
    _row('Skyscrapers', 0.100),
]


def test_identical_answers_have_no_differences():
    assert differences(RECORDED, copy.deepcopy(RECORDED)) == []


def test_a_swap_between_near_tied_songs_is_not_a_change():
    actual = copy.deepcopy(RECORDED)
    actual[1], actual[2] = actual[2], actual[1]
    actual[0]['similarity'] = 0.658
    assert differences(RECORDED, actual) == []


def test_a_song_entering_at_the_tied_end_of_the_list_is_not_a_change():
    actual = copy.deepcopy(RECORDED)
    actual[-1] = _row('song 47', 0.100)
    assert differences(RECORDED, actual) == []


def test_a_different_nearest_song_is_a_change():
    actual = copy.deepcopy(RECORDED)
    actual[0], actual[3] = actual[3], actual[0]
    lines = differences(RECORDED, actual)
    assert lines
    assert any('must rank before' in line for line in lines), lines


def test_a_song_replaced_in_the_middle_is_a_change():
    actual = copy.deepcopy(RECORDED)
    actual[3] = _row('song 12', 0.470)
    lines = differences(RECORDED, actual)
    assert any('is missing' in line for line in lines), lines
    assert any('is unexpected' in line for line in lines), lines


def test_a_changed_field_of_a_song_is_a_change():
    actual = copy.deepcopy(RECORDED)
    actual[2]['title'] = 'Another title'
    lines = differences(RECORDED, actual)
    assert lines == ["$[track:Un week-end avec papy].title: expected \"Un week-end avec papy\", got \"Another title\""], lines


def test_numbers_within_a_hundredth_pass_and_integers_are_exact():
    assert differences({'angle': 0.38, 'tempo': 144.231}, {'angle': 0.381, 'tempo': 144.3}) == []
    assert differences({'angle': 0.38}, {'angle': 0.42}) == ['$.angle: expected 0.38, got 0.42']
    assert differences({'total_songs': 310}, {'total_songs': 309}) == ['$.total_songs: expected 310, got 309']


def test_tag_maps_tolerate_a_label_at_their_lowest_score_only():
    recorded = {'mood_vector': {'jazz': 0.627, 'rock': 0.515, 'ambient': 0.514}}
    drifted = {'mood_vector': {'jazz': 0.629, 'rock': 0.516, 'folk': 0.513}}
    assert differences(recorded, drifted) == []
    changed = {'mood_vector': {'metal': 0.627, 'rock': 0.515, 'ambient': 0.514}}
    assert differences(recorded, changed)


def test_a_top_label_may_flip_only_between_tied_scores():
    recorded = {'top_mood': 'happy', 'other_features': {'happy': 0.63, 'relaxed': 0.63, 'sad': 0.41}}
    tied = {'top_mood': 'relaxed', 'other_features': {'happy': 0.629, 'relaxed': 0.631, 'sad': 0.41}}
    assert differences(recorded, tied) == []
    wrong = {'top_mood': 'sad', 'other_features': {'happy': 0.63, 'relaxed': 0.63, 'sad': 0.41}}
    assert differences(recorded, wrong) == ['$.top_mood: expected "happy", got "sad"']


def test_the_wider_tolerance_is_only_for_what_asks_for_it():
    recorded = [_row('Missing Person', 0.654), _row('What Im sayin', 0.558), _row('Un week-end avec papy', 0.552)]
    drifted = [_row('Missing Person', 0.658), _row('Un week-end avec papy', 0.553), _row('What Im sayin', 0.541)]
    assert differences(recorded, drifted) == ['$[track:What Im sayin].similarity: expected 0.558, got 0.541']
    assert differences(recorded, drifted, tolerance=0.03) == []


def test_missing_and_unexpected_keys_are_reported():
    lines = differences({'a': 1, 'b': 2}, {'a': 1, 'c': 3})
    assert lines == ['$.b: missing, expected 2', '$.c: unexpected 3'], lines


def test_normalize_drops_what_no_other_machine_can_reproduce():
    payload = {
        'fp': '3d3b15a0cad82244', 'umap_x': 0.6, 'embedding_2d': [0.1, 0.2], 'updated_at': '2026-09-19 10:00:00',
        'file_path': '/home/runner/work/x/test/e2e/library/A/01.mp3', 'title': 'Aria',
        'mood_vector': 'jazz:0.627,rock:0.515', 'files': ['/mnt/c/repo/test/e2e/library/A - x/01 - y.mp3'],
    }
    assert normalize(payload, Resolver()) == {
        'title': 'Aria', 'mood_vector': {'jazz': 0.627, 'rock': 0.515}, 'files': ['<library>/A - x/01 - y.mp3'],
    }
