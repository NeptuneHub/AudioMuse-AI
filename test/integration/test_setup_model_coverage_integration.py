# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The setup wizard's per-model coverage bands against a real Postgres database.

Runs the exact queries the wizard runs on every open (two counts and one
directory-header substring per index) over real tables and real stored IVF
directories, for a fresh installation and for a configured library, and
proves that a broken schema hides the bars without poisoning the request
connection.

Main Features:
* A fresh database (every table empty) reports band 0 for every model
* A configured library reports the bands from real counts and real headers,
  the lyrics denominator counting only rows that hold a vector
* A missing table hides every bar and leaves the connection usable afterwards
* Everything lives in a private schema on the shared instance, so the other
  modules' tables (and their foreign keys onto score) are never touched
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

_DDL = (
    "CREATE TABLE IF NOT EXISTS score (item_id TEXT PRIMARY KEY, title TEXT, author TEXT, album TEXT, "
    "album_artist TEXT, tempo REAL, key TEXT, scale TEXT, mood_vector TEXT)",
    "CREATE TABLE IF NOT EXISTS lyrics_embedding (item_id TEXT PRIMARY KEY, embedding BYTEA, axis_vector BYTEA, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE)",
    "CREATE TABLE IF NOT EXISTS ivf_dir (name VARCHAR(255) PRIMARY KEY, blob_data BYTEA NOT NULL, "
    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE IF NOT EXISTS ivf_cell (index_name VARCHAR(255) NOT NULL, cell_id INTEGER NOT NULL, "
    "cell_data BYTEA NOT NULL, PRIMARY KEY (index_name, cell_id))",
)


_SCHEMA = 'wizard_coverage_test'


@pytest.fixture
def wizard_db(shared_pg_dsn):
    admin = psycopg2.connect(shared_pg_dsn)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS %s CASCADE" % _SCHEMA)
        cur.execute("CREATE SCHEMA %s" % _SCHEMA)
        cur.execute("SET search_path TO %s" % _SCHEMA)
        for ddl in _DDL:
            cur.execute(ddl)
    request_conn = psycopg2.connect(shared_pg_dsn)
    with request_conn.cursor() as cur:
        cur.execute("SET search_path TO %s" % _SCHEMA)
    request_conn.commit()
    yield admin, request_conn
    request_conn.close()
    with admin.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS %s CASCADE" % _SCHEMA)
    admin.close()


def _store_directory(conn, index_name, item_ids):
    from tasks import paged_ivf

    dim = 4
    centroids = np.zeros((1, dim), dtype=np.float32)
    cells = [(0, np.arange(min(3, len(item_ids)), dtype=np.int32), np.zeros((min(3, len(item_ids)), dim), dtype=np.float32))]
    paged_ivf.store_paged_ivf(
        conn, index_name, centroids, np.zeros(len(item_ids), dtype=np.uint32), list(item_ids), cells, dim, "angular",
    )


def _levels_through(request_conn, monkeypatch, neural_tracks):
    import app_setup
    import database
    from tasks import neural_fingerprint_index as nfi

    monkeypatch.setattr(database, 'get_db', lambda: request_conn)
    monkeypatch.setattr(nfi, 'indexed_track_count', lambda: neural_tracks)
    return app_setup.model_coverage_levels()


def test_a_fresh_database_reports_band_zero_for_every_model(wizard_db, monkeypatch):
    _admin, request_conn = wizard_db

    assert _levels_through(request_conn, monkeypatch, None) == {
        'musicnn': 0, 'clap': 0, 'lyrics': 0, 'neural-fingerprint': 0,
    }


def test_a_configured_library_reports_the_bands_from_real_counts_and_headers(wizard_db, monkeypatch):
    import config

    admin, request_conn = wizard_db
    ids = ['song-%04d' % n for n in range(1000)]
    with admin.cursor() as cur:
        cur.executemany("INSERT INTO score (item_id, title) VALUES (%s, %s)", [(i, 'title ' + i) for i in ids])
        cur.executemany(
            "INSERT INTO lyrics_embedding (item_id, embedding) VALUES (%s, %s)",
            [(i, psycopg2.Binary(b'\x00' * 16)) for i in ids[:400]],
        )
        cur.executemany("INSERT INTO lyrics_embedding (item_id, embedding) VALUES (%s, NULL)", [(i,) for i in ids[400:500]])
    _store_directory(admin, config.INDEX_NAME, ids[:990])
    _store_directory(admin, 'clap_index', ids[:450])
    _store_directory(admin, 'lyrics_index', ids[:380])

    assert _levels_through(request_conn, monkeypatch, 700) == {
        'musicnn': 5, 'clap': 2, 'lyrics': 5, 'neural-fingerprint': 3,
    }


def test_a_missing_table_hides_every_bar_and_leaves_the_connection_usable(wizard_db, monkeypatch):
    admin, request_conn = wizard_db
    with admin.cursor() as cur:
        cur.execute("DROP TABLE score CASCADE")

    assert _levels_through(request_conn, monkeypatch, 700) == {}
    with request_conn.cursor() as cur:
        cur.execute("SELECT 1")
        assert cur.fetchone()[0] == 1
