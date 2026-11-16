# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Demo manager: the demo window picked from a song's neural fingerprint.

Main Features:
* demo_window lands on a part that comes back three times (a chorus) where the
  window in the middle of the song would not, holds exactly the asked length
  and stays inside the song
* digital silence (rows that do not change) never wins, although its rows are
  the most alike of all: the same song with the rule off picks the silence
* a song no longer than the asked length plays whole, an empty one has no
  demo, and a long track is pooled to MAX_ROWS rows and still finds its chorus
* demo_sections reads every asked song once and answers in the order asked,
  None for a song without a fingerprint
"""

import numpy as np
import pytest

from tasks import demo_manager as dm

_CHORUS_AT = (30.0, 110.0, 170.0)
_CHORUS_SECONDS = 20.0


def _seconds_to_row(seconds):
    return int(round(seconds / 0.5))


def _chorus_song(rng, rows=400):
    vectors = rng.standard_normal((rows, 128)).astype(np.float32)
    chorus = rng.standard_normal((_seconds_to_row(_CHORUS_SECONDS), 128)).astype(np.float32)
    for start in _CHORUS_AT:
        row = _seconds_to_row(start)
        vectors[row:row + chorus.shape[0]] = chorus + 0.3 * rng.standard_normal(chorus.shape)
    return vectors


def _overlap(window, start, end):
    return max(0.0, min(window['end_seconds'], end) - max(window['start_seconds'], start))


def _chorus_overlap(window):
    return max(_overlap(window, start, start + _CHORUS_SECONDS) for start in _CHORUS_AT)


def test_the_demo_is_a_part_that_comes_back_where_the_middle_of_the_song_is_not():
    window = dm.demo_window(_chorus_song(np.random.default_rng(3)), 30)
    duration = 399 * 0.5 + 1.0
    assert window['duration_seconds'] == duration
    assert window['end_seconds'] - window['start_seconds'] == pytest.approx(30.0)
    assert 0.0 <= window['start_seconds'] and window['end_seconds'] <= duration
    assert _chorus_overlap(window) >= 15.0
    middle = (duration - 30.0) / 2.0
    assert _chorus_overlap({'start_seconds': middle, 'end_seconds': middle + 30.0}) < 15.0


def test_a_longer_demo_holds_the_asked_length_and_a_chorus():
    window = dm.demo_window(_chorus_song(np.random.default_rng(4)), 60)
    assert window['end_seconds'] - window['start_seconds'] == pytest.approx(60.0)
    assert _chorus_overlap(window) >= _CHORUS_SECONDS - 1.0


def test_digital_silence_never_wins_although_its_rows_are_the_most_alike(monkeypatch):
    rng = np.random.default_rng(5)
    song = rng.standard_normal((400, 128)).astype(np.float32)
    song[150:251] = rng.standard_normal(128).astype(np.float32)
    window = dm.demo_window(song, 30)
    assert _overlap(window, 75.0, 126.0) <= 2.0
    monkeypatch.setattr(dm, 'SILENCE_COSINE', 1.01)
    assert _overlap(dm.demo_window(song, 30), 75.0, 126.0) >= 25.0


def test_a_short_song_plays_whole_and_an_empty_one_has_no_demo():
    rng = np.random.default_rng(6)
    assert dm.demo_window(rng.standard_normal((50, 128)).astype(np.float32), 30) == {
        'start_seconds': 0.0, 'end_seconds': 25.5, 'duration_seconds': 25.5,
    }
    assert dm.demo_window(np.zeros((0, 128), dtype=np.float32), 30) is None
    window = dm.demo_window(rng.standard_normal((61, 128)).astype(np.float32), 30)
    assert window['end_seconds'] - window['start_seconds'] == pytest.approx(30.0)
    assert 0.0 <= window['start_seconds'] and window['end_seconds'] <= 31.0


def test_a_long_track_is_pooled_and_still_finds_its_chorus(monkeypatch):
    monkeypatch.setattr(dm, 'MAX_ROWS', 100)
    song = _chorus_song(np.random.default_rng(7))
    rows, stride = dm._pooled(song)
    assert stride == 4 and rows.shape == (100, 128)
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-5)
    window = dm.demo_window(song, 30)
    assert window['end_seconds'] - window['start_seconds'] == pytest.approx(30.0)
    assert _chorus_overlap(window) >= 15.0


def test_demo_sections_reads_each_song_once_and_answers_in_the_order_asked(monkeypatch):
    rng = np.random.default_rng(8)
    stored = {'a': _chorus_song(rng), 'b': rng.standard_normal((50, 128)).astype(np.float32)}
    asked = []

    def codes(conn, ids):
        asked.append(list(ids))
        for item_id in ids:
            yield item_id, stored.get(item_id)

    monkeypatch.setattr(dm, 'get_db', lambda: 'conn')
    monkeypatch.setattr(dm, 'iter_codes', codes)
    monkeypatch.setattr(dm.neural_fingerprint, 'decode_codes', lambda rows: rows)
    sections = dm.demo_sections(['b', 'ghost', 'a', 'b'], 30)
    assert asked == [['b', 'ghost', 'a']]
    assert sections[0] == sections[3] == {'start_seconds': 0.0, 'end_seconds': 25.5, 'duration_seconds': 25.5}
    assert sections[1] is None
    assert _chorus_overlap(sections[2]) >= 15.0
