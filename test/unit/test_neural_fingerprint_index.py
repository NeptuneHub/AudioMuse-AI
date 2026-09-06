# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Neural fingerprint index: pack of codes, cells and the vote-and-align search.

Main Features:
* a pack built from stored blobs keeps every 32-byte code row in track order,
  one cell list per centroid, reopens from disk with the same ids and lengths,
  and a legacy int8 blob lands in it as the same codes
* a noisy slice of one track is found at rank one, at its offset in seconds,
  with a high score and flagged identified; the runner-up is not flagged
* a clip of random vectors matches nothing confidently, and a clip shorter
  than two segments is refused before any scan
* a pack written in the previous layout counts as stale and is rebuilt
* the pack is released by unload and the status reports the state
"""

import numpy as np
import pytest

import config
from tasks import ivf_quant
from tasks import neural_fingerprint as nf
from tasks import neural_fingerprint_index as nfi


def _unit(rng, shape):
    vectors = rng.standard_normal(shape).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _legacy_blob(vectors):
    body = np.clip(np.rint(vectors * ivf_quant.I8_SCALE), -127, 127).astype(np.int8)
    return nf._LEGACY_HEADER.pack(nf.LEGACY_MAGIC, nf.DIM, vectors.shape[0], ivf_quant.DTYPE_I8) + body.tobytes()


@pytest.fixture
def library(monkeypatch, tmp_path):
    rng = np.random.default_rng(4)
    tracks = {f'fp_{i:04d}': _unit(rng, (60 + i % 7, nf.DIM)) for i in range(40)}
    book = nf.train_codebook(np.concatenate(list(tracks.values())), iterations=8)
    np.savez(tmp_path / 'pq.npz', codebook=book)
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_CODEBOOK_PATH', str(tmp_path / 'pq.npz'))
    for key in ('codebook', 'codebook_id', 'codebook_bias'):
        monkeypatch.setitem(nf._STATE, key, None)
    paths = {
        'codes': str(tmp_path / 'rows.u8'), 'cells': str(tmp_path / 'cells.i32'), 'meta': str(tmp_path / 'meta.npz'),
    }
    blobs = (
        (item_id, _legacy_blob(v) if i % 5 == 0 else nf.encode_blob(v)) for i, (item_id, v) in enumerate(tracks.items())
    )
    nfi.build_pack_from_rows(blobs, paths, len(tracks))
    nfi.unload()
    nfi._open_pack(paths)
    monkeypatch.setattr(nfi, 'ensure_loaded', lambda: True)
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_MODEL_PATH', __file__)
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_NPROBE', 4)
    yield tracks, rng, paths
    nfi.unload()


def _serve(monkeypatch, vectors):
    monkeypatch.setattr(nfi, 'fingerprint_audio', lambda audio, sr, hop: vectors)


def test_the_pack_keeps_track_order_and_one_list_per_cell(library):
    tracks, _rng, paths = library
    assert [str(i) for i in nfi._STATE['ids']] == sorted(tracks)
    lengths = [int(n) for n in nfi._STATE['lengths']]
    assert lengths == [tracks[k].shape[0] for k in sorted(tracks)]
    codes = nfi._STATE['codes']
    assert codes.shape == (sum(lengths), nf.CODE_BYTES)
    assert codes.dtype == np.uint8
    starts = nfi._STATE['starts']
    assert np.array_equal(codes[starts[1]:starts[1] + lengths[1]], nf.encode_codes(tracks['fp_0001']))
    legacy_codes = codes[starts[0]:starts[0] + lengths[0]]
    assert np.mean(legacy_codes == nf.encode_codes(tracks['fp_0000'])) > 0.9
    bounds = nfi._STATE['cell_bounds']
    assert bounds[0] == 0
    assert bounds[-1] == sum(lengths)
    assert nfi._STATE['centroids'].shape[1] == nf.DIM
    assert np.array_equal(np.sort(np.asarray(nfi._STATE['cell_rows'])), np.arange(sum(lengths)))
    assert nfi._cached_count(paths) == 40
    status = nfi.get_status()
    assert status['loaded'] is True
    assert status['tracks'] == 40


def test_a_noisy_slice_is_found_at_its_offset_and_flagged(monkeypatch, library):
    tracks, rng, _paths = library
    source = tracks['fp_0017']
    start = 21
    query = source[start:start + 30] + 0.03 * rng.standard_normal((30, nf.DIM)).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    _serve(monkeypatch, query)
    rows = nfi.identify(np.zeros(8000 * 16, dtype=np.float32), 8000, 10)
    assert rows[0]['item_id'] == 'fp_0017'
    assert rows[0]['offset_seconds'] == pytest.approx(start * nf.HOP_SECONDS, abs=0.01)
    assert rows[0]['score'] > 0.7
    assert rows[0]['identified'] is True
    assert rows[0]['lead'] > 0.4
    assert all(row['identified'] is False and row['lead'] is None for row in rows[1:])
    assert len(rows) == 10


def test_random_vectors_match_nothing_confidently_and_a_short_clip_is_refused(monkeypatch, library):
    _tracks, rng, _paths = library
    _serve(monkeypatch, _unit(rng, (30, nf.DIM)))
    rows = nfi.identify(np.zeros(8000 * 16, dtype=np.float32), 8000, 5)
    assert rows
    assert not any(row['identified'] for row in rows)
    assert rows[0]['score'] < config.NEURAL_FINGERPRINT_MIN_SCORE
    _serve(monkeypatch, _unit(rng, (1, nf.DIM)))
    with pytest.raises(ValueError, match='too short'):
        nfi.identify(np.zeros(8000, dtype=np.float32), 8000, 5)


def test_a_pack_in_the_previous_layout_is_stale(library):
    _tracks, _rng, paths = library
    with np.load(paths['meta'], allow_pickle=False) as meta:
        old = {key: meta[key] for key in meta.files if key != 'format'}
    np.savez(paths['meta'], **old)
    assert nfi._cached_count(paths) is None


def test_unload_releases_the_pack_and_a_missing_model_is_a_runtime_error(monkeypatch, library):
    assert nfi.unload() is True
    assert nfi.get_status()['loaded'] is False
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_MODEL_PATH', '/nowhere/model.onnx')
    with pytest.raises(RuntimeError, match='not available'):
        nfi.identify(np.zeros(8000 * 4, dtype=np.float32), 8000, 5)
