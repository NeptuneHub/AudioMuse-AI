# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The neural fingerprint index lifecycle against a real PostgreSQL.

The worker builds the index from the stored blobs into ivf_dir, the web
process syncs its local pack from the table and searches it, a later build
appends the new tracks as one more part and the web process picks it up by
reusing the codes it already has, and a library grown past the retrain factor
gets its centroids rebuilt from scratch.

Main Features:
* build -> ivf_dir holds the directory and the build id, ivf_cell the cells of
  part 0, committed by the builder itself so another connection sees them at once
* ensure_loaded reads the directory and a slice of a track is identified at its
  offset from cells paged out of the table
* new tracks -> the next build appends part 1, keeps the build history, and
  reload_from_db swaps the served directory to the new build
* a library grown fourfold -> the next build retrains and goes back to one part
"""

import os
import sys

import numpy as np
import pytest

try:
    import psycopg2
except Exception:  # pragma: no cover
    psycopg2 = None

pytestmark = pytest.mark.integration

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_SCHEMA = (
    "CREATE TABLE embedding (item_id TEXT PRIMARY KEY, embedding BYTEA, neural_fingerprint BYTEA)",
    "CREATE TABLE ivf_dir (name VARCHAR(255) PRIMARY KEY, blob_data BYTEA NOT NULL, "
    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE ivf_cell (index_name VARCHAR(255) NOT NULL, cell_id INTEGER NOT NULL, "
    "cell_data BYTEA NOT NULL, PRIMARY KEY (index_name, cell_id))",
)


def _unit(rng, shape):
    vectors = rng.standard_normal(shape).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _insert(conn, nf, tracks):
    with conn.cursor() as cur:
        for item_id, vectors in tracks.items():
            cur.execute(
                'INSERT INTO embedding (item_id, neural_fingerprint) VALUES (%s, %s)',
                (item_id, psycopg2.Binary(nf.encode_blob(vectors))),
            )
    conn.commit()


def _names(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT name FROM ivf_dir ORDER BY name")
        return [row[0] for row in cur.fetchall()]


def _names_seen_by_a_fresh_connection(dsn):
    other = psycopg2.connect(dsn)
    try:
        return _names(other)
    finally:
        other.close()


def _cell_parts(dsn):
    other = psycopg2.connect(dsn)
    try:
        with other.cursor() as cur:
            cur.execute("SELECT DISTINCT index_name FROM ivf_cell WHERE index_name LIKE 'neural%%' ORDER BY index_name")
            return [row[0] for row in cur.fetchall()]
    finally:
        other.close()


def test_worker_build_web_sync_append_and_retrain(shared_pg_dsn, monkeypatch, tmp_path):
    if psycopg2 is None:
        pytest.skip('psycopg2 not importable')
    import config
    import database
    from tasks import neural_fingerprint as nf
    from tasks import neural_fingerprint_index as nfi

    rng = np.random.default_rng(11)
    tracks = {f'fp_{i:04d}': _unit(rng, (40 + i % 5, nf.DIM)) for i in range(30)}
    book = nf.train_codebook(np.concatenate(list(tracks.values())), iterations=8)
    np.savez(tmp_path / 'pq.npz', codebook=book)
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_ENABLED', True)
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_CODEBOOK_PATH', str(tmp_path / 'pq.npz'))
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_MODEL_PATH', __file__)
    monkeypatch.setattr(config, 'IVF_DISK_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_NPROBE', 4)
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_RETRAIN_GROWTH', 4.0)
    for key in ('codebook', 'codebook_id', 'codebook_bias'):
        monkeypatch.setitem(nf._STATE, key, None)
    monkeypatch.setattr(database, 'connect_raw', lambda **kw: psycopg2.connect(shared_pg_dsn))
    monkeypatch.setattr(nfi, 'fingerprint_audio', lambda audio, sr, hop: nfi._TEST_QUERY)
    nfi.unload()

    conn = psycopg2.connect(shared_pg_dsn)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute('DROP TABLE IF EXISTS ivf_dir')
            cur.execute('DROP TABLE IF EXISTS ivf_cell')
            cur.execute('DROP TABLE IF EXISTS embedding CASCADE')
            for statement in _SCHEMA:
                cur.execute(statement)
        conn.commit()
        _insert(conn, nf, tracks)

        assert nfi.build_and_store_neural_fingerprint_index(conn) is True
        names = _names_seen_by_a_fresh_connection(shared_pg_dsn)
        assert 'neural_fingerprint_index__ivf_dir' in names
        assert 'neural_fingerprint_index__build' in names
        assert _cell_parts(shared_pg_dsn) == ['neural_fingerprint_index/p0']
        first = nfi._load_directory(conn)
        assert first['parts'] == 1
        assert first['ids'] == sorted(tracks)
        assert first['trained_tracks'] == 30

        assert nfi.ensure_loaded() is True
        assert nfi._STATE['pack'].build_id == first['build_id']
        query = tracks['fp_0012'][7:30] + 0.03 * rng.standard_normal((23, nf.DIM)).astype(np.float32)
        nfi._TEST_QUERY = query / np.linalg.norm(query, axis=1, keepdims=True)
        rows = nfi.identify(np.zeros(8000 * 12, dtype=np.float32), 8000, 5)
        assert rows[0]['item_id'] == 'fp_0012'
        assert rows[0]['offset_seconds'] == pytest.approx(7 * nf.HOP_SECONDS, abs=0.01)
        assert rows[0]['identified'] is True

        added = {f'fp_{i:04d}': _unit(rng, (35, nf.DIM)) for i in range(30, 36)}
        _insert(conn, nf, added)
        assert nfi.build_and_store_neural_fingerprint_index(conn) is True
        second = nfi._load_directory(conn)
        assert second['parts'] == 2
        assert second['ids'] == sorted(tracks) + sorted(added)
        assert second['build_id'] != first['build_id']
        assert _cell_parts(shared_pg_dsn) == ['neural_fingerprint_index/p0', 'neural_fingerprint_index/p1']
        assert nfi.reload_from_db() is True
        assert nfi._STATE['pack'].build_id == second['build_id']
        assert nfi._STATE['pack'].ids.size == 36
        assert nfi._STATE['pack'].parts == 2
        query = added['fp_0033'][3:25] + 0.03 * rng.standard_normal((22, nf.DIM)).astype(np.float32)
        nfi._TEST_QUERY = query / np.linalg.norm(query, axis=1, keepdims=True)
        rows = nfi.identify(np.zeros(8000 * 12, dtype=np.float32), 8000, 5)
        assert rows[0]['item_id'] == 'fp_0033'
        assert rows[0]['identified'] is True
        assert nfi.get_status()['cached_cells'] > 0

        grown = {f'fp_{i:04d}': _unit(rng, (30, nf.DIM)) for i in range(36, 130)}
        _insert(conn, nf, grown)
        assert nfi.build_and_store_neural_fingerprint_index(conn) is True
        third = nfi._load_directory(conn)
        assert third['parts'] == 1
        assert third['trained_tracks'] == 130
        assert _cell_parts(shared_pg_dsn) == ['neural_fingerprint_index/p0']
        assert nfi.reload_from_db() is True
        assert nfi._STATE['pack'].ids.size == 130
        assert nfi.get_status()['cached_cells'] == 0
    finally:
        nfi.unload()
        conn.close()
