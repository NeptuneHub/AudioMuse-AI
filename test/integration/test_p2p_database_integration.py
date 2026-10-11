# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The output provenance stamps of database.py against a real Postgres.

The unit tests pin the statements; this module proves their SQL does what the
P2P code relies on: every save stamps its own output key in embedding.output_origin
in the same transaction as the data, a peer-supplied result carries the peer's
version and id, and a later local save replaces that stamp with a local one.
Every table lives in a private schema dropped afterwards, so the shared session
database is left as it was found.

Main Features:
* each local save stamps its output key with config.APP_VERSION; a neural save
  on a missing row writes nothing
* origin=(version, peer) on the CLAP, lyrics and neural saves stamps
  {v, p} for that key and keeps the other keys
* a later local save of that output replaces the peer stamp with a local one
* a stamp that fails rolls the data of that save back with it
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
except Exception:
    psycopg2 = None

pytestmark = pytest.mark.integration

_SCHEMA = 'p2p_database_test'
_DDL = (
    "CREATE TABLE score (item_id TEXT PRIMARY KEY, title TEXT, author TEXT, album TEXT, "
    "album_artist TEXT, tempo REAL, key TEXT, scale TEXT, mood_vector TEXT, energy REAL, "
    "other_features TEXT, year INTEGER, rating INTEGER, duration DOUBLE PRECISION)",
    "CREATE TABLE embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) ON DELETE CASCADE, "
    "embedding BYTEA, poincare_embedding BYTEA, hyperbolic_radius DOUBLE PRECISION, "
    "neural_fingerprint BYTEA, output_origin JSONB)",
    "CREATE TABLE clap_embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) ON DELETE CASCADE, "
    "embedding BYTEA)",
    "CREATE TABLE lyrics_embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) ON DELETE CASCADE, "
    "embedding BYTEA, axis_vector BYTEA, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
)
PEER = '12D3KooWPeerAlphaTestOnly'


def _connect(dsn):
    return psycopg2.connect(dsn, options=f'-c search_path={_SCHEMA}')


@pytest.fixture
def p2p_db(shared_pg_dsn, monkeypatch):
    if psycopg2 is None:
        pytest.skip('psycopg2 not importable')
    import database

    admin = psycopg2.connect(shared_pg_dsn)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f'DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE')
        cur.execute(f'CREATE SCHEMA {_SCHEMA}')
    conn = _connect(shared_pg_dsn)
    with conn.cursor() as cur:
        for ddl in _DDL:
            cur.execute(ddl)
    conn.commit()
    monkeypatch.setattr(database, 'get_db', lambda: conn)
    monkeypatch.setattr(database, '_persist_hyperbolic_inline', lambda item_id, vector, cur: None)
    yield conn
    conn.rollback()
    conn.close()
    with admin.cursor() as cur:
        cur.execute(f'DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE')
    admin.close()


def _vector(dim, seed):
    vector = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _analysed(item_id, seed=0):
    import database

    database.save_track_analysis_and_embedding(
        item_id, 'Title', 'Author', 120.0, 'C', 'major', {'happy': 0.5}, _vector(200, seed),
        energy=0.1, other_features='danceable:0.40', duration=200.0,
    )


def _origin(conn, item_id):
    with conn.cursor() as cur:
        cur.execute('SELECT output_origin FROM embedding WHERE item_id = %s', (item_id,))
        row = cur.fetchone()
    conn.commit()
    return row[0] if row else None


def _scalar(conn, query, params):
    with conn.cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()
    conn.commit()
    return row[0] if row else None


def test_local_saves_stamp_their_own_key_with_the_running_version(p2p_db):
    import config
    import database

    conn = p2p_db
    local = {'v': config.APP_VERSION}
    _analysed('item-1')
    assert _origin(conn, 'item-1') == {'m': local}
    database.save_clap_embedding('item-1', _vector(512, 1))
    database.save_lyrics_embedding('item-1', _vector(768, 2), np.full(27, 0.2, dtype=np.float32))
    assert database.save_neural_fingerprint('item-1', b'NFP2' + bytes(20)) is True
    assert _origin(conn, 'item-1') == {'m': local, 'c': local, 'l': local, 'n': local}

    assert database.save_neural_fingerprint('never-analysed', b'NFP2' + bytes(20)) is False
    assert _scalar(conn, 'SELECT count(*) FROM embedding WHERE item_id = %s', ('never-analysed',)) == 0


def test_peer_saves_stamp_the_peer_version_and_id_and_keep_the_other_keys(p2p_db):
    import config
    import database

    conn = p2p_db
    _analysed('item-2')
    clap = _vector(512, 3)
    lyrics = _vector(768, 4)
    axis = np.linspace(0, 1, 27).astype(np.float32)
    blob = b'NFP2' + bytes(range(20))
    database.save_clap_embedding('item-2', clap, origin=('v3.1.0', PEER))
    database.save_lyrics_embedding('item-2', lyrics, axis, origin=('v3.0.0', PEER))
    assert database.save_neural_fingerprint('item-2', blob, origin=('v3.2.0', PEER)) is True

    assert _origin(conn, 'item-2') == {
        'm': {'v': config.APP_VERSION},
        'c': {'v': 'v3.1.0', 'p': PEER},
        'l': {'v': 'v3.0.0', 'p': PEER},
        'n': {'v': 'v3.2.0', 'p': PEER},
    }
    assert bytes(_scalar(conn, 'SELECT embedding FROM clap_embedding WHERE item_id = %s', ('item-2',))) == clap.tobytes()
    assert bytes(_scalar(conn, 'SELECT embedding FROM lyrics_embedding WHERE item_id = %s', ('item-2',))) == (
        lyrics.tobytes()
    )
    assert bytes(_scalar(conn, 'SELECT axis_vector FROM lyrics_embedding WHERE item_id = %s', ('item-2',))) == (
        axis.tobytes()
    )
    assert bytes(_scalar(conn, 'SELECT neural_fingerprint FROM embedding WHERE item_id = %s', ('item-2',))) == blob


def test_a_local_save_replaces_the_peer_stamp_of_that_output_only(p2p_db):
    import config
    import database

    conn = p2p_db
    _analysed('item-3')
    database.save_clap_embedding('item-3', _vector(512, 5), origin=('v3.1.0', PEER))
    database.save_neural_fingerprint('item-3', b'NFP2' + bytes(8), origin=('v3.2.0', PEER))
    database.save_clap_embedding('item-3', _vector(512, 6))
    assert _origin(conn, 'item-3') == {
        'm': {'v': config.APP_VERSION},
        'c': {'v': config.APP_VERSION},
        'n': {'v': 'v3.2.0', 'p': PEER},
    }


def test_a_peer_neural_fingerprint_on_a_missing_row_writes_nothing(p2p_db):
    import database

    conn = p2p_db
    assert database.save_neural_fingerprint('never-analysed', b'NFP2' + bytes(8), origin=('v3.2.0', PEER)) is False
    assert _scalar(conn, 'SELECT count(*) FROM embedding WHERE item_id = %s', ('never-analysed',)) == 0


def test_a_stamp_that_fails_rolls_the_data_back(p2p_db):
    import database

    conn = p2p_db
    _analysed('item-4')
    with pytest.raises(ValueError):
        database.save_clap_embedding('item-4', _vector(512, 7), origin=('v3.1.0', 'bad\x00peer'))
    assert _scalar(conn, 'SELECT count(*) FROM clap_embedding WHERE item_id = %s', ('item-4',)) == 0
    assert 'c' not in _origin(conn, 'item-4')
