# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Near-duplicate distance filter of the audio IVF manager.

Pins tasks.ivf_manager._filter_by_distance, the filter Similar Song, Sonic
Fingerprint, Album Creation and Song Alchemy run on their neighbour lists, to the
per-pair loop it replaced: same kept songs, same order, same log lines.

Main Features:
* Identity against the per-pair reference for the angular, euclidean and dot
  metrics, both lookbacks, the single-window path (50 or fewer songs) and the
  batched path, with missing vectors, a zero vector and exact copies planted.
  A logged distance may differ from the reference in its 4th decimal: the
  matrix product and the per-pair np.dot may round the last float32 bit apart
  (BLAS kernels vary by CPU), so it is compared to within that rounding
* The filter measures a whole window with one matrix product instead of one
  Python distance call per pair
"""

import logging
import re

import numpy as np
import pytest

from tasks import ivf_manager as im


def _reference_filter(songs, details_map, lookback, distance):
    threshold = (
        im.DUPLICATE_DISTANCE_THRESHOLD_COSINE
        if im.IVF_METRIC == 'angular'
        else im.DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN
    )
    metric_name = 'Angular' if im.IVF_METRIC == 'angular' else 'Euclidean'
    messages = []

    def too_close(current, vector, window):
        for recent in window:
            recent_vector = im._get_cached_vector(recent['item_id'])
            if recent_vector is None:
                continue
            direct_dist = distance(vector, recent_vector)
            if direct_dist < threshold:
                current_details = details_map[current['item_id']]
                recent_details = details_map[recent['item_id']]
                messages.append(
                    f"Filtering song (DISTANCE FILTER) with {metric_name} distance: '{current_details['title']}' by '{current_details['author']}' "
                    f"due to direct distance of {direct_dist:.4f} from "
                    f"'{recent_details['title']}' by '{recent_details['author']}' (Threshold: {threshold})."
                )
                return True
        return False

    kept = []
    if len(songs) <= im.BATCH_SIZE_VECTOR_OPS:
        for song in songs:
            vector = im._get_cached_vector(song['item_id'])
            if vector is not None and not too_close(song, vector, kept[-lookback:]):
                kept.append(song)
    else:
        for start in range(0, len(songs), im.BATCH_SIZE_VECTOR_OPS):
            base = kept[-lookback:]
            batch = []
            for song in songs[start:start + im.BATCH_SIZE_VECTOR_OPS]:
                vector = im._get_cached_vector(song['item_id'])
                if vector is not None and not too_close(song, vector, base + batch):
                    batch.append(song)
            kept.extend(batch)
    return [song['item_id'] for song in kept], messages


_DISTANCE_IN_MESSAGE = re.compile(r"direct distance of (-?\d+\.\d+)")


def _split_distance(message):
    match = _DISTANCE_IN_MESSAGE.search(message)
    return _DISTANCE_IN_MESSAGE.sub("direct distance of <d>", message), float(match.group(1))


_NOISE = {'angular': (0.05, 0.3), 'euclidean': (0.005, 0.07), 'dot': (0.05, 0.3)}


def _planted_library(metric, total, seed):
    rng = np.random.default_rng(seed)
    low, high = _NOISE[metric]
    vectors = {}
    previous = []
    for position in range(total):
        item_id = f'song-{position}'
        roll = rng.random()
        if position == 3:
            vector = np.zeros(16, dtype=np.float32)
        elif roll < 0.05:
            continue
        elif previous and roll < 0.12:
            vector = previous[-min(len(previous), int(rng.integers(1, 4)))].copy()
        elif previous and roll < 0.45:
            noise = rng.uniform(low, high) * rng.standard_normal(16)
            vector = (previous[-1] + noise).astype(np.float32)
        else:
            vector = rng.standard_normal(16).astype(np.float32)
        vectors[item_id] = vector
        previous.append(vector)
    songs = [{'item_id': f'song-{position}', 'distance': float(position)} for position in range(total)]
    details = {s['item_id']: {'title': f"Song {s['item_id']}", 'author': 'Artist A'} for s in songs}
    return songs, vectors, details


@pytest.fixture
def filter_setup(monkeypatch):
    monkeypatch.setattr(im, 'ivf_index', None)
    monkeypatch.setattr(im, 'reverse_id_map', None)
    monkeypatch.setattr(im, 'DUPLICATE_DISTANCE_THRESHOLD_COSINE', 0.01)
    monkeypatch.setattr(im, 'DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN', 0.15)
    monkeypatch.setattr(im, 'BATCH_SIZE_VECTOR_OPS', 50)

    def configure(metric, lookback, total, seed):
        songs, vectors, details = _planted_library(metric, total, seed)
        monkeypatch.setattr(im, 'IVF_METRIC', metric)
        monkeypatch.setattr(im, 'DUPLICATE_DISTANCE_CHECK_LOOKBACK', lookback)
        monkeypatch.setattr(im, '_fetch_details_map', lambda conn, ids, columns: details)
        im._prime_request_f32(vectors)
        return songs, details

    yield configure
    im._clear_request_f32()


@pytest.mark.parametrize('metric', ['angular', 'euclidean', 'dot'])
@pytest.mark.parametrize('lookback', [1, 3])
@pytest.mark.parametrize('total', [37, 50, 51, 263])
def test_the_matrix_filter_keeps_exactly_what_the_per_pair_loop_kept(filter_setup, caplog, metric, lookback, total):
    songs, details = filter_setup(metric, lookback, total, seed=total * 7 + lookback)
    expected_ids, expected_messages = _reference_filter(songs, details, lookback, im.get_direct_distance)

    with caplog.at_level(logging.INFO, logger=im.logger.name):
        kept = im._filter_by_distance(songs, db_conn=None)

    messages = [r.getMessage() for r in caplog.records if 'DISTANCE FILTER' in r.getMessage()]
    assert [song['item_id'] for song in kept] == expected_ids
    assert [_split_distance(m)[0] for m in messages] == [
        _split_distance(m)[0] for m in expected_messages
    ]
    assert [_split_distance(m)[1] for m in messages] == pytest.approx(
        [_split_distance(m)[1] for m in expected_messages], abs=1.5e-4
    )
    assert all(any(song is original for original in songs) for song in kept)
    assert 0 < len(expected_messages) and len(expected_ids) < total


def test_the_filter_measures_windows_as_matrices_not_one_python_call_per_pair(filter_setup, monkeypatch):
    songs, details = filter_setup('angular', 1, 263, seed=11)
    expected_ids, _ = _reference_filter(songs, details, 1, im.get_direct_distance)

    def per_pair_call(*_args):
        raise AssertionError('per-pair distance call')

    for name in ('get_direct_distance', '_get_direct_cosine_distance',
                 '_get_direct_euclidean_distance', '_get_direct_dot_distance'):
        monkeypatch.setattr(im, name, per_pair_call)

    kept = im._filter_by_distance(songs, db_conn=None)

    assert [song['item_id'] for song in kept] == expected_ids
