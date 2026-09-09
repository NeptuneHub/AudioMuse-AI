# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Neural fingerprint index: quantizer, cells, directory, on-demand paging and the vote-and-align search.

Main Features:
* a cell round-trips through its blob and a part writes one row per non-empty
  cell holding exactly that cell's rows
* the two-level quantizer covers the asked cells with a coarse level of about
  sqrt(cells) groups, assigns rows to their nearest cell within the nearest
  group, and stays single-level under the small-library threshold
* label_tracks splits the rows into parts at track boundaries under the row
  budget and hands each part its codes
* the directory round-trips with int8 centroids, ids, lengths, cell sizes and
  the stride; a directory of an older layout is reported, never a crash
* the full-build decision names its reason: no index, layout, codebook or
  stride change, too many tracks gone, the library grown past the retrain
  factor, unbalanced cells; otherwise the build appends
* the builder commits its own transaction and rolls back on failure
* a loaded index reads only the cells a query probes, keeps them in a RAM
  cache that answers the next query without the database, is bounded by
  NEURAL_FINGERPRINT_CACHE_MB, is dropped on demand and forgets the previous
  build on a swap
* a noisy slice of one track is found at rank one at its offset and flagged;
  a track appended in a later part is found the same way; random vectors
  match nothing confidently; a clip shorter than two segments is refused;
  excluded ids leave the vote so a song does not find itself; a track whose
  blob is gone can no longer come back
* a clear match is decided by the first pass of segments and the rest is
  skipped; an unclear one votes with every segment
* a stride of two indexes every other row and still finds the track at its
  real offset
* a request scoped to a server votes only over that server's tracks through
  the shared availability mask, cached per server and build and dropped by
  invalidate_availability_cache; the mask is skipped only for a lone default
  server whose ids are all legacy
* the loaded build offers the song picker a database-side filter, none when
  unloaded; the startup load is blocking and returns the track count, 0 when
  the feature is off, the model is missing, nothing is stored or the stored
  directory is unusable
"""

import numpy as np
import pytest

import config
from tasks import neural_fingerprint as nf
from tasks import neural_fingerprint_index as nfi

_REAL_ENSURE_LOADED = nfi.ensure_loaded


def _unit(rng, shape):
    vectors = rng.standard_normal(shape).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _quantizer(codes, n_cells=16):
    sample = nfi.training_sample(iter(codes.items()), 4000, np.random.default_rng(0))
    return nfi.train_quantizer(sample, n_cells)


def _drain(generator):
    items = []
    while True:
        try:
            items.append(next(generator))
        except StopIteration as done:
            return items, done.value


def _build_parts(codes, quantizer, first_part=0, first_track=0, part_rows=nfi._PART_ROWS):
    ids, lengths, cell_sizes, rows = [], [], np.zeros(quantizer.n_cells, dtype=np.int64), []
    part, track = first_part, first_track
    for chunk_ids, chunk_lengths, labels, chunk_codes in nfi.label_tracks(iter(codes.items()), quantizer, part_rows):
        tracks, offsets = nfi.track_offset_arrays(track, chunk_lengths)
        cell_rows, counts = _drain(nfi.part_cells(part, labels, chunk_codes, tracks, offsets, quantizer.n_cells))
        rows.extend(cell_rows)
        cell_sizes += counts
        ids.extend(chunk_ids)
        lengths.append(chunk_lengths)
        part += 1
        track += len(chunk_ids)
    return ids, np.concatenate(lengths), cell_sizes, part, rows


def _directory(build_id, quantizer, ids, lengths, cell_sizes, parts, trained_tracks):
    blob = nfi.pack_directory(build_id, nf.codebook()[1], quantizer, ids, lengths, cell_sizes, parts, trained_tracks)
    return nfi.unpack_directory(blob)


def _serve_rows(monkeypatch, store, fetched=None):
    def read_rows(cell_ids):
        wanted = set(int(cell) for cell in cell_ids)
        if fetched is not None:
            fetched.append(sorted(wanted))
        return [
            (name, cell, blob) for (name, cell), blob in store.items()
            if cell in wanted and name.startswith(nfi._CELL_NAMESPACE)
        ]

    monkeypatch.setattr(nfi, '_read_cell_rows', read_rows)


@pytest.fixture
def codebook(monkeypatch, tmp_path):
    rng = np.random.default_rng(4)
    tracks = {f'fp_{i:04d}': _unit(rng, (60 + i % 7, nf.DIM)) for i in range(40)}
    book = nf.train_codebook(np.concatenate(list(tracks.values())), iterations=8)
    np.savez(tmp_path / 'pq.npz', codebook=book)
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_CODEBOOK_PATH', str(tmp_path / 'pq.npz'))
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_MODEL_PATH', __file__)
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_NPROBE', 4)
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_INDEX_STRIDE', 1)
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_QUERY_THREADS', 2)
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_CACHE_MB', 64)
    for key in ('codebook', 'codebook_id', 'codebook_bias'):
        monkeypatch.setitem(nf._STATE, key, None)
    return tracks, rng


@pytest.fixture
def library(monkeypatch, codebook):
    tracks, rng = codebook
    codes = {item_id: nf.encode_codes(v) for item_id, v in tracks.items()}
    quantizer = _quantizer(codes)
    ids, lengths, cell_sizes, parts, rows = _build_parts(codes, quantizer)
    directory = _directory('build-one', quantizer, ids, lengths, cell_sizes, parts, len(ids))
    store = {(name, cell): blob for name, cell, blob in rows}
    _serve_rows(monkeypatch, store)
    monkeypatch.setattr(nfi, '_candidate_codes', lambda wanted: {i: codes[i] for i in wanted if i in codes})
    nfi.unload()
    nfi._swap_pack(nfi._pack_from(directory))
    monkeypatch.setattr(nfi, 'ensure_loaded', lambda: True)
    yield tracks, rng, codes, quantizer, directory, store
    nfi.unload()


def _serve(monkeypatch, vectors):
    monkeypatch.setattr(nfi, 'fingerprint_audio', lambda audio, sr, hop: vectors)


def test_a_cell_round_trips_and_a_part_writes_one_row_per_non_empty_cell(codebook):
    tracks, rng = codebook
    codes = nf.encode_codes(_unit(rng, (500, nf.DIM)))
    labels = rng.integers(0, 6, size=500).astype(np.int32)
    labels[labels == 3] = 4
    tracks_of = rng.integers(0, 9, size=500).astype(np.uint32)
    offsets = rng.integers(0, 400, size=500).astype(np.uint16)
    blob = nfi.pack_cell(codes[:7], tracks_of[:7], offsets[:7], np.linspace(0.8, 1.2, 7, dtype=np.float32))
    back_codes, back_tracks, back_offsets, back_norms = nfi.unpack_cell(blob)
    assert np.array_equal(back_codes, codes[:7])
    assert np.array_equal(back_tracks, tracks_of[:7])
    assert np.array_equal(back_offsets, offsets[:7])
    assert np.allclose(back_norms.astype(np.float32), np.linspace(0.8, 1.2, 7), atol=1e-3)
    with pytest.raises(ValueError):
        nfi.unpack_cell(blob[:-1])
    rows, counts = _drain(nfi.part_cells(3, labels, codes, tracks_of, offsets, 6))
    assert np.array_equal(counts, np.bincount(labels, minlength=6))
    assert [cell for _name, cell, _blob in rows] == sorted(set(labels.tolist()))
    assert all(name == 'neural_fingerprint_index/p3' for name, _cell, _blob in rows)
    seen = 0
    for _name, cell, cell_blob in rows:
        cell_codes, cell_tracks, cell_offsets, _norms = nfi.unpack_cell(cell_blob)
        picked = np.flatnonzero(labels == cell)
        assert np.array_equal(cell_codes, codes[picked])
        assert np.array_equal(cell_tracks, tracks_of[picked])
        assert np.array_equal(cell_offsets, offsets[picked])
        seen += picked.size
    assert seen == 500


def test_a_cell_slice_over_the_stored_value_cap_is_split_over_rows_the_reader_joins(monkeypatch, library):
    tracks, _rng, codes, quantizer, directory, store = library
    monkeypatch.setattr(config, 'IVF_MAX_PART_SIZE_MB', 0)
    monkeypatch.setattr(nfi, '_rows_per_cell_row', lambda: 5)
    ids, lengths, cell_sizes, parts, rows = _build_parts(codes, quantizer)
    names = {name for name, _cell, _blob in rows}
    assert 'neural_fingerprint_index/p0' in names
    assert 'neural_fingerprint_index/p0.1' in names
    for cell in range(quantizer.n_cells):
        pieces = [nfi.unpack_cell(blob)[0].shape[0] for name, c, blob in rows if c == cell]
        assert sum(pieces) == cell_sizes[cell]
        assert all(piece <= 5 for piece in pieces)
    split_store = {(name, cell): blob for name, cell, blob in rows}
    _serve_rows(monkeypatch, split_store)
    nfi.drop_cell_cache()
    cells = nfi._cells_for(nfi._STATE['pack'], list(range(quantizer.n_cells)))
    assert sum(arrays[0].shape[0] for arrays in cells.values()) == int(cell_sizes.sum())
    assert nfi.identify_vectors(tracks['fp_0017'][10:40], 5)[0]['item_id'] == 'fp_0017'


def test_the_two_level_quantizer_covers_the_cells_and_assigns_within_the_nearest_group(codebook):
    tracks, rng = codebook
    sample = _unit(rng, (6000, nf.DIM))
    quantizer = nfi.train_quantizer(sample, 100)
    assert quantizer.coarse.shape == (10, nf.DIM)
    assert 90 <= quantizer.n_cells <= 100
    assert quantizer.offsets[0] == 0
    assert quantizer.offsets[-1] == quantizer.n_cells
    assert (np.diff(quantizer.offsets) >= 1).all()
    codes = nf.encode_codes(sample[:2000])
    labels = nfi.assign_cells(codes, quantizer)
    assert labels.min() >= 0
    assert labels.max() < quantizer.n_cells
    decoded = nf.decode_codes(codes)
    groups = np.argmax(decoded @ quantizer.coarse.T, axis=1)
    assert np.array_equal(np.searchsorted(quantizer.offsets, labels, side='right') - 1, groups)
    exact = np.argmax(decoded @ quantizer.cells.T, axis=1)
    assert np.mean(exact == labels) > 0.6
    small = nfi.train_quantizer(sample, 16)
    assert small.coarse.shape[0] == 1
    assert small.n_cells == 16
    assert np.array_equal(nfi.assign_cells(codes, small), np.argmax(decoded @ small.cells.T, axis=1))


def test_label_tracks_splits_parts_at_track_boundaries_under_the_row_budget(library):
    tracks, _rng, codes, quantizer, *_rest = library
    chunks = list(nfi.label_tracks(iter(codes.items()), quantizer, part_rows=200))
    assert len(chunks) > 1
    assert [i for ids, _lengths, _labels, _codes in chunks for i in ids] == sorted(tracks)
    for ids, lengths, labels, chunk_codes in chunks:
        assert int(lengths.sum()) <= 200
        assert labels.size == int(lengths.sum())
        assert chunk_codes.shape == (int(lengths.sum()), nf.CODE_BYTES)
        assert lengths.tolist() == [tracks[i].shape[0] for i in ids]
        assert np.array_equal(chunk_codes, np.concatenate([codes[i] for i in ids]))
    whole = np.concatenate([labels for _ids, _lengths, labels, _codes in chunks])
    assert np.array_equal(whole, next(iter(nfi.label_tracks(iter(codes.items()), quantizer)))[2])


def test_the_directory_round_trips_with_int8_centroids_and_an_older_layout_is_reported(codebook):
    import io

    tracks, rng = codebook
    quantizer = nfi.train_quantizer(_unit(rng, (300, nf.DIM)), 7)
    directory = _directory('b1', quantizer, ['a', 'b'], np.array([3, 4]), np.arange(7), 2, 9)
    assert directory['build_id'] == 'b1'
    assert directory['ids'] == ['a', 'b']
    assert directory['lengths'].tolist() == [3, 4]
    assert directory['parts'] == 2
    assert directory['trained_tracks'] == 9
    assert directory['stride'] == 1
    assert directory['codebook_id'] == nf.codebook()[1]
    assert np.einsum('ij,ij->i', directory['centroids'], quantizer.cells).min() > 0.999
    assert np.array_equal(directory['quantizer'].offsets, quantizer.offsets)
    buffer = io.BytesIO()
    np.savez(buffer, format=np.int64(nfi.FORMAT - 1), build_id=np.asarray('old-build'), centroids=np.zeros((4, nf.DIM), dtype=np.int8))
    old = nfi.unpack_directory(buffer.getvalue())
    assert old == {'format': nfi.FORMAT - 1, 'build_id': 'old-build'}
    assert 'layout' in nfi.needs_full_build(old, ['a'], nf.codebook()[1])


def test_the_full_build_decision_names_its_reason(monkeypatch, codebook):
    tracks, rng = codebook
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_RETRAIN_GROWTH', 4.0)
    ids = [f'fp_{i:04d}' for i in range(10)]
    quantizer = nfi.train_quantizer(_unit(rng, (300, nf.DIM)), 8)
    directory = _directory('b', quantizer, ids, np.full(10, 60), np.full(8, 100), 1, 10)
    book_id = nf.codebook()[1]
    assert nfi.needs_full_build(None, ids, book_id) == 'no index yet'
    assert 'codebook' in nfi.needs_full_build(directory, ids, book_id ^ 1)
    assert nfi.needs_full_build(directory, ids + ['fp_new'], book_id) is None
    assert 'gone' in nfi.needs_full_build(directory, ids[:5], book_id)
    assert 'grew' in nfi.needs_full_build(directory, ids + [f'x{i}' for i in range(30)], book_id)
    sizes = np.full(100, 10)
    sizes[0] = 2000
    unbalanced = _directory('b', nfi.train_quantizer(_unit(rng, (400, nf.DIM)), 100), ids, np.full(10, 60), sizes, 1, 10)
    assert 'largest cell' in nfi.needs_full_build(unbalanced, ids, book_id)
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_INDEX_STRIDE', 2)
    assert 'stride' in nfi.needs_full_build(directory, ids, book_id)


def test_the_builder_commits_its_own_transaction_and_rolls_back_on_failure(monkeypatch, codebook):
    class Conn:
        def __init__(self):
            self.commits = 0
            self.rollbacks = 0

        def commit(self):
            self.commits += 1

        def rollback(self):
            self.rollbacks += 1

    built = []
    monkeypatch.setattr(nfi, '_fingerprinted_tracks', lambda conn: ['fp_0001'])
    monkeypatch.setattr(nfi, '_load_directory', lambda conn: None)
    monkeypatch.setattr(nfi, '_full_build', lambda conn, ids, book_id, reason: built.append(reason))
    conn = Conn()
    assert nfi.build_and_store_neural_fingerprint_index(conn) is True
    assert built == ['no index yet']
    assert (conn.commits, conn.rollbacks) == (1, 0)

    def boom(conn, ids, book_id, reason):
        raise RuntimeError('disk full')

    monkeypatch.setattr(nfi, '_full_build', boom)
    failing = Conn()
    with pytest.raises(RuntimeError, match='disk full'):
        nfi.build_and_store_neural_fingerprint_index(failing)
    assert (failing.commits, failing.rollbacks) == (0, 1)


def test_the_loaded_index_reads_probed_cells_on_demand_and_caches_them(monkeypatch, library):
    tracks, _rng, codes, quantizer, directory, store = library
    fetched = []
    _serve_rows(monkeypatch, store, fetched)
    nfi.drop_cell_cache()
    query = tracks['fp_0017'][10:40]
    rows = nfi.identify_vectors(query, 5)
    assert rows[0]['item_id'] == 'fp_0017'
    assert fetched
    assert all(cells for cells in fetched)
    assert set(cell for cells in fetched for cell in cells) < set(range(quantizer.n_cells))
    status = nfi.get_status()
    assert status['loaded'] is True
    assert status['tracks'] == 40
    assert status['cells'] == quantizer.n_cells
    assert status['parts'] == 1
    assert status['cached_cells'] == len(set(cell for cells in fetched for cell in cells))
    assert status['cache_mb'] > 0
    fetched.clear()
    assert nfi.identify_vectors(query, 5)[0]['item_id'] == 'fp_0017'
    assert fetched == []
    nfi.drop_cell_cache()
    assert nfi.get_status()['cached_cells'] == 0
    assert nfi.get_status()['cache_mb'] == 0
    nfi.identify_vectors(query, 5)
    assert fetched
    nfi._swap_pack(nfi._pack_from(dict(directory, build_id='build-two')))
    assert nfi.get_status()['cached_cells'] == 0


def test_the_cell_cache_is_bounded_by_the_configured_megabytes(monkeypatch, library):
    tracks, *_rest = library
    query = tracks['fp_0017'][10:40]
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_CACHE_MB', 0)
    nfi.drop_cell_cache()
    assert nfi.identify_vectors(query, 5)[0]['item_id'] == 'fp_0017'
    assert nfi.get_status()['cached_cells'] == 0
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_CACHE_MB', 64)
    nfi.identify_vectors(query, 5)
    assert nfi.get_status()['cached_cells'] > 0
    assert nfi.get_status()['cache_mb'] <= 64


def test_a_noisy_slice_is_found_at_its_offset_and_flagged(monkeypatch, library):
    tracks, rng, *_rest = library
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


def test_an_appended_build_adds_parts_and_finds_the_new_track(monkeypatch, library):
    tracks, rng, codes, quantizer, directory, store = library
    new_tracks = {f'fp_{i:04d}': _unit(rng, (50, nf.DIM)) for i in range(40, 44)}
    new_codes = {item_id: nf.encode_codes(v) for item_id, v in new_tracks.items()}
    ids, lengths, cell_sizes, parts, rows = _build_parts(new_codes, quantizer, first_part=directory['parts'], first_track=len(directory['ids']))
    assert parts == 2
    assert all(name == 'neural_fingerprint_index/p1' for name, _cell, _blob in rows)
    store.update({(name, cell): blob for name, cell, blob in rows})
    second = _directory(
        'build-two', quantizer, directory['ids'] + ids, np.concatenate([directory['lengths'], lengths]),
        directory['cell_sizes'] + cell_sizes, parts, directory['trained_tracks'],
    )
    every = {**codes, **new_codes}
    monkeypatch.setattr(nfi, '_candidate_codes', lambda wanted: {i: every[i] for i in wanted})
    nfi._swap_pack(nfi._pack_from(second))
    pack = nfi._STATE['pack']
    assert pack.ids.size == 44
    assert pack.parts == 2
    query = new_tracks['fp_0042'][5:35] + 0.03 * rng.standard_normal((30, nf.DIM)).astype(np.float32)
    _serve(monkeypatch, query / np.linalg.norm(query, axis=1, keepdims=True))
    rows = nfi.identify(np.zeros(8000 * 16, dtype=np.float32), 8000, 5)
    assert rows[0]['item_id'] == 'fp_0042'
    assert rows[0]['offset_seconds'] == pytest.approx(5 * nf.HOP_SECONDS, abs=0.01)
    assert rows[0]['identified'] is True
    assert nfi.identify_vectors(tracks['fp_0017'][10:40], 5)[0]['item_id'] == 'fp_0017'


def test_identify_vectors_can_leave_the_source_track_out_and_a_track_without_a_blob_never_returns(monkeypatch, library):
    tracks, _rng, codes, *_rest = library
    query = tracks['fp_0017'][10:40]
    with_self = nfi.identify_vectors(query, 5)
    assert with_self[0]['item_id'] == 'fp_0017'
    without = nfi.identify_vectors(query, 5, exclude_ids=('fp_0017',))
    assert without
    assert 'fp_0017' not in [row['item_id'] for row in without]
    assert without[0]['score'] < with_self[0]['score']
    with pytest.raises(ValueError, match='too short'):
        nfi.identify_vectors(query[:1], 5)
    monkeypatch.setattr(nfi, '_candidate_codes', lambda wanted: {i: codes[i] for i in wanted if i != 'fp_0017'})
    assert 'fp_0017' not in [row['item_id'] for row in nfi.identify_vectors(query, 5)]


def test_a_clear_match_is_decided_by_the_first_pass_and_an_unclear_one_votes_with_every_segment(monkeypatch, library):
    tracks, rng, *_rest = library
    scored = []
    original = nfi._segment_votes

    def counting(pack, book, query_vectors, qi, probed, cells, allowed):
        scored.append(qi)
        return original(pack, book, query_vectors, qi, probed, cells, allowed)

    monkeypatch.setattr(nfi, '_segment_votes', counting)
    query = tracks['fp_0017'][10:40]
    rows = nfi.identify_vectors(query, 5)
    assert rows[0]['item_id'] == 'fp_0017'
    assert sorted(scored) == list(range(0, 30, nfi._FIRST_PASS_EVERY))
    scored.clear()
    rows = nfi.identify_vectors(_unit(rng, (30, nf.DIM)), 5)
    assert sorted(scored) == list(range(30))
    assert not any(row['identified'] for row in rows)


def test_a_stride_of_two_indexes_every_other_row_and_still_finds_the_track(monkeypatch, codebook):
    tracks, rng = codebook
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_INDEX_STRIDE', 2)
    codes = {item_id: nf.encode_codes(v) for item_id, v in tracks.items()}
    quantizer = _quantizer(codes)
    ids, lengths, cell_sizes, parts, rows = _build_parts(codes, quantizer)
    assert lengths.tolist() == [(tracks[i].shape[0] + 1) // 2 for i in ids]
    directory = _directory('build-stride', quantizer, ids, lengths, cell_sizes, parts, len(ids))
    assert directory['stride'] == 2
    _serve_rows(monkeypatch, {(name, cell): blob for name, cell, blob in rows})
    monkeypatch.setattr(nfi, '_candidate_codes', lambda wanted: {i: codes[i] for i in wanted})
    monkeypatch.setattr(nfi, 'ensure_loaded', lambda: True)
    nfi.unload()
    nfi._swap_pack(nfi._pack_from(directory))
    try:
        assert nfi._STATE['pack'].stride == 2
        for start in (20, 21):
            found = nfi.identify_vectors(tracks['fp_0017'][start:start + 30], 5)
            assert found[0]['item_id'] == 'fp_0017'
            assert found[0]['offset_seconds'] == pytest.approx(start * nf.HOP_SECONDS, abs=0.01)
            assert found[0]['identified'] is True
    finally:
        nfi.unload()


def test_a_server_scope_votes_only_over_that_servers_tracks_and_the_mask_is_cached_per_server(monkeypatch, library):
    tracks, _rng, *_rest = library
    builds = []

    def mask(server_id, item_ids, conn_factory):
        builds.append(server_id)
        return np.array([server_id == 'with' or item_id != 'fp_0017' for item_id in item_ids], dtype=np.bool_)

    monkeypatch.setattr(nfi, 'build_availability_mask', mask)
    monkeypatch.setattr(nfi, '_mask_unneeded', lambda pack, server_id: False)
    nfi.invalidate_availability_cache()
    query = tracks['fp_0017'][10:40]
    monkeypatch.setattr(nfi, 'active_availability_scope', lambda: 'without')
    rows = nfi.identify_vectors(query, 5)
    assert rows
    assert 'fp_0017' not in [row['item_id'] for row in rows]
    nfi.identify_vectors(query, 5)
    assert builds == ['without']
    monkeypatch.setattr(nfi, 'active_availability_scope', lambda: 'with')
    assert nfi.identify_vectors(query, 5)[0]['item_id'] == 'fp_0017'
    assert 'fp_0017' not in [row['item_id'] for row in nfi.identify_vectors(query, 5, exclude_ids=('fp_0017',))]
    assert builds == ['without', 'with']
    nfi.invalidate_availability_cache('with')
    nfi.identify_vectors(query, 5)
    assert builds == ['without', 'with', 'with']
    monkeypatch.setattr(nfi, 'active_availability_scope', lambda: None)
    nfi.identify_vectors(query, 5)
    assert builds == ['without', 'with', 'with']
    nfi.invalidate_availability_cache()
    assert len(nfi._AVAILABILITY) == 0


def test_a_disabled_feature_builds_loads_and_reloads_nothing(monkeypatch):
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_ENABLED', False)
    assert nfi.build_and_store_neural_fingerprint_index(None) is False
    assert nfi.load_at_startup() == 0
    assert nfi.reload_from_db() is False


def test_the_loaded_build_offers_a_database_side_picker_filter_and_none_when_unloaded(library):
    where = nfi.picker_where()
    assert where is not None
    assert 'neural_fingerprint IS NOT NULL' in where[0]
    assert 'score.item_id' in where[0]
    assert where[1] == ()
    nfi.unload()
    assert nfi.picker_where() is None


def test_the_mask_is_skipped_only_for_a_lone_default_server_over_legacy_ids(monkeypatch, library):
    from tasks.mediaserver import registry

    monkeypatch.setattr(registry, 'get_default_server_id', lambda: 'main')
    monkeypatch.setattr(registry, 'has_secondary_servers', lambda: False)
    pack = nfi._STATE['pack']
    nfi._CANONICAL.clear()
    assert nfi._has_canonical_ids(pack) is True
    assert nfi._mask_unneeded(pack, 'main') is False
    legacy = pack._replace(ids=np.array(['legacy-1', 'legacy-2']), build_id='legacy-build')
    assert nfi._has_canonical_ids(legacy) is False
    assert nfi._mask_unneeded(legacy, 'main') is True
    assert nfi._mask_unneeded(legacy, 'other') is False
    monkeypatch.setattr(registry, 'has_secondary_servers', lambda: True)
    assert nfi._mask_unneeded(legacy, 'main') is False
    nfi._CANONICAL.clear()


def test_random_vectors_match_nothing_confidently_and_a_short_clip_is_refused(monkeypatch, library):
    tracks, rng, *_rest = library
    _serve(monkeypatch, _unit(rng, (24, nf.DIM)))
    rows = nfi.identify(np.zeros(8000 * 13, dtype=np.float32), 8000, 5)
    assert not any(row['identified'] for row in rows)
    assert rows[0]['score'] < config.NEURAL_FINGERPRINT_MIN_SCORE
    _serve(monkeypatch, _unit(rng, (1, nf.DIM)))
    with pytest.raises(ValueError, match='too short'):
        nfi.identify(np.zeros(8000, dtype=np.float32), 8000, 5)


def test_startup_load_is_blocking_returns_the_track_count_and_reports_an_unusable_directory(monkeypatch, library):
    import database

    tracks, _rng, _codes, _quantizer, directory, _store = library

    class Conn:
        def close(self):
            pass

    monkeypatch.setattr(database, 'connect_raw', lambda **kw: Conn())
    monkeypatch.setattr(nfi, 'ensure_loaded', _REAL_ENSURE_LOADED)
    nfi.unload()
    monkeypatch.setattr(nfi, '_stored_build_id', lambda conn: None)
    assert nfi.load_at_startup() == 0
    assert nfi.is_loaded() is False
    monkeypatch.setattr(nfi, '_stored_build_id', lambda conn: 'build-one')
    monkeypatch.setattr(nfi, '_load_directory', lambda conn: directory)
    assert nfi.load_at_startup() == 40
    assert nfi.is_loaded() is True
    assert nfi.load_at_startup() == 40
    nfi.unload()
    monkeypatch.setattr(nfi, '_load_directory', lambda conn: {'format': nfi.FORMAT - 1, 'build_id': 'build-one'})
    assert nfi.load_at_startup() == 0
    assert 'older version' in nfi.get_status()['error']
    with pytest.raises(nfi.IndexUnavailable, match='older version'):
        nfi.ensure_loaded()
    assert nfi.reload_from_db() is False
    monkeypatch.setattr(nfi, '_load_directory', lambda conn: directory)
    assert nfi.reload_from_db() is True
    assert nfi._STATE['pack'].build_id == 'build-one'
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_MODEL_PATH', '/nowhere/model.onnx')
    assert nfi.load_at_startup() == 0


def test_unload_releases_the_index_and_a_missing_model_is_unavailable(monkeypatch, library):
    assert nfi.unload() is True
    status = nfi.get_status()
    assert status['loaded'] is False
    assert status['tracks'] == 0
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_MODEL_PATH', '/nowhere/model.onnx')
    with pytest.raises(nfi.IndexUnavailable, match='not available'):
        nfi.identify(np.zeros(8000 * 4, dtype=np.float32), 8000, 5)
