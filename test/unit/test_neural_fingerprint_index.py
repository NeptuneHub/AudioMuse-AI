# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Neural fingerprint index: parts, directory, local pack and the vote-and-align search.

Main Features:
* a part round-trips through its blob and merging parts gives the same cell
  lists as one part over all rows
* the directory round-trips with int8 centroids, ids, lengths and cell sizes
* the full-build decision names its reason: no index, layout or codebook
  change, too many tracks gone, the library grown past the retrain factor,
  unbalanced cells; otherwise the build appends
* a local pack written from a directory and its parts stores the codes by
  cell, with the row id, the track and the inverse norm beside every slab
  position and the slab position of every row, so the track order is
  recoverable; a second build under the same centroids appends new tracks by
  copying every cell's previous run and fetching only the new blobs
* the builder commits its own transaction and rolls back on failure, so a
  web process never sees a half-written build
* the loaded build offers the song picker a database-side filter, none when
  unloaded
* a noisy slice of one track is found at rank one at its offset and flagged;
  a track appended later is found the same way; random vectors match nothing
  confidently; a clip shorter than two segments is refused; excluded ids
  leave the vote so a song does not find itself
* a request scoped to a server votes only over that server's tracks through
  the shared availability mask, cached per server and build and dropped by
  invalidate_availability_cache; the mask is skipped only for a lone default
  server whose ids are all legacy
* the pack is released by unload and the status reports the state
* a track whose fingerprint row is gone since the build is written as dead
  rows, never voted, left out of the count, and stays dead when the next
  pack reuses the previous cells as a prefix
* the startup load hands the sync to a background thread and returns None
  when the feature is off, the model is missing or no build is stored
"""

import numpy as np
import pytest

import config
from tasks import neural_fingerprint as nf
from tasks import neural_fingerprint_index as nfi


def _unit(rng, shape):
    vectors = rng.standard_normal(shape).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _build(tracks, centroids=None, previous=None, previous_paths=None, paths=None, trained_tracks=None):
    rng = np.random.default_rng(0)
    codes = {item_id: nf.encode_codes(v) for item_id, v in tracks.items()}
    if centroids is None:
        sample = nfi.training_sample(iter(codes.items()), 4000, rng)
        centroids = nfi.train_centroids(sample, 16)
    ids, lengths, labels = nfi.label_tracks(iter(codes.items()), centroids)
    return ids, lengths, labels, centroids, codes


@pytest.fixture
def codebook(monkeypatch, tmp_path):
    rng = np.random.default_rng(4)
    tracks = {f'fp_{i:04d}': _unit(rng, (60 + i % 7, nf.DIM)) for i in range(40)}
    book = nf.train_codebook(np.concatenate(list(tracks.values())), iterations=8)
    np.savez(tmp_path / 'pq.npz', codebook=book)
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_CODEBOOK_PATH', str(tmp_path / 'pq.npz'))
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_MODEL_PATH', __file__)
    monkeypatch.setattr(config, 'IVF_DISK_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_NPROBE', 4)
    for key in ('codebook', 'codebook_id', 'codebook_bias'):
        monkeypatch.setitem(nf._STATE, key, None)
    return tracks, rng


def _directory(build_id, ids, lengths, centroids, counts, parts, trained_tracks):
    blob = nfi.pack_directory(build_id, nf.codebook()[1], centroids, ids, lengths, counts, parts, trained_tracks)
    return nfi.unpack_directory(blob)


@pytest.fixture
def library(monkeypatch, codebook):
    tracks, rng = codebook
    ids, lengths, labels, centroids, codes = _build(tracks)
    part_blob, counts = nfi.pack_part(labels, centroids.shape[0], 0)
    directory = _directory('build-one', ids, lengths, centroids, counts, 1, len(ids))
    paths = nfi._paths('build-one')
    nfi.write_local_pack(paths, directory, [nfi.unpack_part(part_blob)], lambda wanted: ((i, codes[i]) for i in wanted))
    nfi.unload()
    nfi._open_pack(paths)
    monkeypatch.setattr(nfi, 'ensure_loaded', lambda: True)
    yield tracks, rng, codes, centroids, directory, paths
    nfi.unload()


def _serve(monkeypatch, vectors):
    monkeypatch.setattr(nfi, 'fingerprint_audio', lambda audio, sr, hop: vectors)


def test_a_part_round_trips_and_merging_parts_equals_one_part(codebook):
    labels = np.random.default_rng(1).integers(0, 5, size=1000).astype(np.int32)
    blob, counts = nfi.pack_part(labels, 5, 0)
    part = nfi.unpack_part(blob)
    assert part['n_rows'] == 1000
    assert np.array_equal(np.diff(part['bounds']), counts)
    assert np.array_equal(np.sort(part['order']), np.arange(1000))
    assert all(labels[part['order'][part['bounds'][c]:part['bounds'][c + 1]]].tolist() == [c] * int(counts[c]) for c in range(5))
    first, _ = nfi.pack_part(labels[:600], 5, 0)
    second, _ = nfi.pack_part(labels[600:], 5, 600)
    merged_rows, merged_bounds = nfi.merge_parts([nfi.unpack_part(first), nfi.unpack_part(second)], 5)
    whole_rows, whole_bounds = nfi.merge_parts([part], 5)
    assert np.array_equal(merged_bounds, whole_bounds)
    for c in range(5):
        assert set(merged_rows[merged_bounds[c]:merged_bounds[c + 1]].tolist()) == set(whole_rows[whole_bounds[c]:whole_bounds[c + 1]].tolist())
    with pytest.raises(ValueError):
        nfi.unpack_part(b'XXXX' + blob[4:])


def test_the_directory_round_trips_with_int8_centroids(codebook):
    tracks, rng = codebook
    centroids = _unit(rng, (7, nf.DIM))
    directory = _directory('b1', ['a', 'b'], np.array([3, 4]), centroids, np.arange(7), 2, 9)
    assert directory['build_id'] == 'b1'
    assert directory['ids'] == ['a', 'b']
    assert directory['lengths'].tolist() == [3, 4]
    assert directory['parts'] == 2
    assert directory['trained_tracks'] == 9
    assert directory['codebook_id'] == nf.codebook()[1]
    assert np.einsum('ij,ij->i', directory['centroids'], centroids).min() > 0.999


def test_the_full_build_decision_names_its_reason(monkeypatch, codebook):
    tracks, rng = codebook
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_RETRAIN_GROWTH', 4.0)
    ids = [f'fp_{i:04d}' for i in range(10)]
    counts = np.full(8, 100)
    directory = _directory('b', ids, np.full(10, 60), _unit(rng, (8, nf.DIM)), counts, 1, 10)
    book_id = nf.codebook()[1]
    assert nfi.needs_full_build(None, ids, book_id) == 'no index yet'
    assert 'codebook' in nfi.needs_full_build(directory, ids, book_id ^ 1)
    assert nfi.needs_full_build(directory, ids + ['fp_new'], book_id) is None
    assert 'gone' in nfi.needs_full_build(directory, ids[:5], book_id)
    assert 'grew' in nfi.needs_full_build(directory, ids + [f'x{i}' for i in range(30)], book_id)
    sizes = np.full(100, 10)
    sizes[0] = 2000
    unbalanced = _directory('b', ids, np.full(10, 60), _unit(rng, (100, nf.DIM)), sizes, 1, 10)
    assert 'largest cell' in nfi.needs_full_build(unbalanced, ids, book_id)
    stale = dict(directory, format=nfi.FORMAT - 1)
    assert 'layout' in nfi.needs_full_build(stale, ids, book_id)


def test_the_pack_stores_the_codes_by_cell_with_the_track_order_recoverable(library):
    tracks, _rng, codes, centroids, directory, paths = library
    pack = nfi._STATE['pack']
    assert [str(i) for i in pack.ids] == sorted(tracks)
    lengths = [int(n) for n in pack.lengths]
    assert lengths == [tracks[k].shape[0] for k in sorted(tracks)]
    n_rows = sum(lengths)
    assert pack.slab.shape == (n_rows, nf.CODE_BYTES)
    rows = np.asarray(pack.row)
    assert np.array_equal(np.sort(rows), np.arange(n_rows))
    assert np.array_equal(rows[np.asarray(pack.pos)], np.arange(n_rows))
    start = int(pack.starts[1])
    assert np.array_equal(pack.slab[np.asarray(pack.pos[start:start + lengths[1]])], codes['fp_0001'])
    bounds = pack.cell_bounds
    assert bounds[0] == 0
    assert bounds[-1] == n_rows
    _ids, _lengths, labels = nfi.label_tracks(iter(codes.items()), centroids)
    by_position = labels[rows]
    assert (np.diff(by_position) >= 0).all()
    assert np.array_equal(np.bincount(by_position, minlength=bounds.size - 1), np.diff(bounds))
    assert np.array_equal(np.asarray(pack.track), np.searchsorted(pack.starts, rows, side='right') - 1)
    book = nf.codebook()[0]
    raw = book[np.arange(nf.PQ_SUBSPACES)[None, :], np.asarray(pack.slab[:7])].reshape(7, nf.DIM)
    assert np.allclose(np.asarray(pack.norm[:7]).astype(np.float32), 1.0 / np.linalg.norm(raw, axis=1), rtol=2e-3)
    assert pack.build_id == 'build-one'
    assert nfi._local_builds() == ['build-one']
    status = nfi.get_status()
    assert status['loaded'] is True
    assert status['tracks'] == 40


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


def test_an_appended_build_reuses_the_previous_codes_and_finds_the_new_track(monkeypatch, library):
    tracks, rng, codes, centroids, directory, paths = library
    new_tracks = {f'fp_{i:04d}': _unit(rng, (50, nf.DIM)) for i in range(40, 44)}
    new_codes = {item_id: nf.encode_codes(v) for item_id, v in new_tracks.items()}
    ids, lengths, labels = nfi.label_tracks(iter(new_codes.items()), centroids)
    part_blob, counts = nfi.pack_part(labels, centroids.shape[0], int(directory['lengths'].sum()))
    second = _directory(
        'build-two', directory['ids'] + ids, np.concatenate([directory['lengths'], lengths]), centroids,
        directory['cell_sizes'] + counts, 2, directory['trained_tracks'],
    )
    fetched = []

    def codes_iter(wanted):
        fetched.extend(wanted)
        return ((i, new_codes[i]) for i in wanted)

    new_paths = nfi._paths('build-two')
    first_part = nfi.unpack_part(nfi.pack_part(nfi.label_tracks(iter(codes.items()), centroids)[2], centroids.shape[0], 0)[0])
    nfi.write_local_pack(new_paths, second, [first_part, nfi.unpack_part(part_blob)], codes_iter, paths, nfi._local_meta('build-one'))
    assert fetched == ids
    nfi._open_pack(new_paths)
    assert nfi._STATE['pack'].ids.size == 44
    query = new_tracks['fp_0042'][5:35] + 0.03 * rng.standard_normal((30, nf.DIM)).astype(np.float32)
    _serve(monkeypatch, query / np.linalg.norm(query, axis=1, keepdims=True))
    rows = nfi.identify(np.zeros(8000 * 16, dtype=np.float32), 8000, 5)
    assert rows[0]['item_id'] == 'fp_0042'
    assert rows[0]['offset_seconds'] == pytest.approx(5 * nf.HOP_SECONDS, abs=0.01)
    assert rows[0]['identified'] is True
    nfi._prune_local('build-two')
    assert nfi._local_builds() == ['build-two']


def test_identify_vectors_can_leave_the_source_track_out(library):
    tracks, _rng, *_rest = library
    query = tracks['fp_0017'][10:40]
    with_self = nfi.identify_vectors(query, 5)
    assert with_self[0]['item_id'] == 'fp_0017'
    without = nfi.identify_vectors(query, 5, exclude_ids=('fp_0017',))
    assert without
    assert 'fp_0017' not in [row['item_id'] for row in without]
    assert without[0]['score'] < with_self[0]['score']
    with pytest.raises(ValueError, match='too short'):
        nfi.identify_vectors(query[:1], 5)


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
    assert nfi._AVAILABILITY_CACHE == {}


def test_a_disabled_feature_builds_loads_and_reloads_nothing(monkeypatch):
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_ENABLED', False)
    assert nfi.build_and_store_neural_fingerprint_index(None) is False
    assert nfi.load_at_startup() is None
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
    _tracks, rng, *_rest = library
    _serve(monkeypatch, _unit(rng, (30, nf.DIM)))
    rows = nfi.identify(np.zeros(8000 * 16, dtype=np.float32), 8000, 5)
    assert rows
    assert not any(row['identified'] for row in rows)
    assert rows[0]['score'] < config.NEURAL_FINGERPRINT_MIN_SCORE
    _serve(monkeypatch, _unit(rng, (1, nf.DIM)))
    with pytest.raises(ValueError, match='too short'):
        nfi.identify(np.zeros(8000, dtype=np.float32), 8000, 5)


def test_a_track_gone_since_the_build_is_written_dead_masked_and_uncounted(library):
    tracks, _rng, codes, centroids, directory, paths = library
    gone = 'fp_0017'
    gone_index = sorted(tracks).index(gone)
    part = nfi.unpack_part(nfi.pack_part(nfi.label_tracks(iter(codes.items()), centroids)[2], centroids.shape[0], 0)[0])
    second = dict(directory, build_id='build-two')
    second_paths = nfi._paths('build-two')
    nfi.write_local_pack(
        second_paths, second, [part], lambda wanted: ((i, None if i == gone else codes[i]) for i in wanted)
    )
    nfi._open_pack(second_paths)
    assert nfi._STATE['pack'].dead.tolist() == [gone_index]
    assert nfi.get_status()['tracks'] == 39
    rows = nfi.identify_vectors(tracks[gone][10:40], 5)
    assert rows
    assert gone not in [row['item_id'] for row in rows]
    fetched = []

    def codes_iter(wanted):
        fetched.extend(wanted)
        return ((i, codes[i]) for i in wanted)

    third_paths = nfi._paths('build-three')
    nfi.write_local_pack(
        third_paths, dict(second, build_id='build-three'), [part], codes_iter, second_paths, nfi._local_meta('build-two')
    )
    assert fetched == []
    nfi._open_pack(third_paths)
    assert nfi._STATE['pack'].dead.tolist() == [gone_index]
    assert gone not in [row['item_id'] for row in nfi.identify_vectors(tracks[gone][10:40], 5)]


def test_startup_load_maps_the_stored_build_in_the_background_and_says_when_there_is_none(monkeypatch, library):
    import database

    class Conn:
        def close(self):
            pass

    monkeypatch.setattr(database, 'connect_raw', lambda **kw: Conn())
    monkeypatch.setattr(nfi, '_stored_build_id', lambda conn: None)
    assert nfi.load_at_startup() is None
    monkeypatch.setattr(nfi, '_stored_build_id', lambda conn: 'build-one')
    assert nfi.load_at_startup() is None
    nfi.unload()
    monkeypatch.setattr(nfi, 'ensure_loaded', lambda: nfi._open_pack(nfi._paths('build-one')) or True)
    thread = nfi.load_at_startup()
    thread.join(10)
    assert not thread.is_alive()
    assert nfi.get_status()['tracks'] == 40
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_MODEL_PATH', '/nowhere/model.onnx')
    assert nfi.load_at_startup() is None


def test_unload_releases_the_pack_and_a_missing_model_is_a_runtime_error(monkeypatch, library):
    assert nfi.unload() is True
    status = nfi.get_status()
    assert status['loaded'] is False
    assert status['synced'] is True
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_MODEL_PATH', '/nowhere/model.onnx')
    with pytest.raises(RuntimeError, match='not available'):
        nfi.identify(np.zeros(8000 * 4, dtype=np.float32), 8000, 5)
