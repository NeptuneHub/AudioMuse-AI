# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""IVF detail-batch fetch tests against a real Postgres database.

Seeds a large score table and exercises tasks.ivf_manager._fetch_details_map
to confirm its multi-batch id fetching returns complete and correct rows.

Main Features:
* Full multi-batch fetch is complete and matches seeded rows.
* Spot-checked and partial id subsets spanning batch boundaries.
"""

import importlib.util
import os
import sys
import tempfile

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
except Exception:  # pragma: no cover - psycopg2 is in test/requirements.txt
    psycopg2 = None


_SCORE_DDL = (
    "CREATE TABLE score (item_id TEXT PRIMARY KEY, title TEXT, author TEXT, "
    "album TEXT, album_artist TEXT, tempo REAL, key TEXT, scale TEXT, "
    "mood_vector TEXT, energy REAL, other_features TEXT, year INTEGER, "
    "rating INTEGER, file_path TEXT)"
)

_ROW_COUNT = 250


def _load_ivf_manager():
    mod_name = 'tasks.ivf_manager'
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    mod_path = os.path.join(_REPO_ROOT, 'tasks', 'ivf_manager.py')
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='session')
def pg_dsn():
    if psycopg2 is None:
        pytest.skip("psycopg2 not importable")
    dsn = os.environ.get('AUDIOMUSE_TEST_DATABASE_URL')
    if dsn:
        try:
            psycopg2.connect(dsn).close()
        except Exception as e:
            pytest.skip(f"AUDIOMUSE_TEST_DATABASE_URL not reachable: {e}")
        yield dsn
        return
    try:
        import pgserver
    except Exception:
        pytest.skip(
            "No test database. Set AUDIOMUSE_TEST_DATABASE_URL to a disposable "
            "DB, or `pip install pgserver` for an ephemeral local instance."
        )
    data_dir = tempfile.mkdtemp(prefix='audiomuse_pg_')
    server = pgserver.get_server(data_dir)
    try:
        yield server.get_uri()
    finally:
        server.cleanup()


@pytest.fixture
def batch_db(pg_dsn):
    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS embedding")
        cur.execute("DROP TABLE IF EXISTS score CASCADE")
        cur.execute(_SCORE_DDL)
        rows = [(f"item-{i:04d}", f"Title {i}", f"Author {i}") for i in range(_ROW_COUNT)]
        cur.executemany(
            "INSERT INTO score (item_id, title, author) VALUES (%s, %s, %s)",
            rows,
        )
    yield conn
    conn.close()


@pytest.mark.integration
class TestFetchDetailsMapRealDb:
    def test_multi_batch_fetch_is_complete_and_correct(self, batch_db):
        ivf = _load_ivf_manager()
        all_ids = [f"item-{i:04d}" for i in range(_ROW_COUNT)]

        assert _ROW_COUNT > ivf.BATCH_SIZE_DB_OPS
        n_batches = (_ROW_COUNT + ivf.BATCH_SIZE_DB_OPS - 1) // ivf.BATCH_SIZE_DB_OPS
        assert n_batches >= 3

        details = ivf._fetch_details_map(batch_db, all_ids, ivf.SCORE_DETAIL_COLUMNS)

        assert isinstance(details, dict)
        assert len(details) == _ROW_COUNT
        assert set(details.keys()) == set(all_ids)
        for i, item_id in enumerate(all_ids):
            entry = details[item_id]
            assert entry['title'] == f"Title {i}"
            assert entry['author'] == f"Author {i}"

    def test_spot_checked_ids(self, batch_db):
        ivf = _load_ivf_manager()
        all_ids = [f"item-{i:04d}" for i in range(_ROW_COUNT)]
        details = ivf._fetch_details_map(batch_db, all_ids, ivf.SCORE_DETAIL_COLUMNS)

        assert details["item-0000"] == {"title": "Title 0", "author": "Author 0"}
        assert details["item-0100"] == {"title": "Title 100", "author": "Author 100"}
        assert details["item-0249"] == {"title": "Title 249", "author": "Author 249"}

    def test_partial_id_subset_spanning_batches(self, batch_db):
        ivf = _load_ivf_manager()
        requested = [f"item-{i:04d}" for i in range(150)] + ["missing-a", "missing-b"]
        details = ivf._fetch_details_map(batch_db, requested, ivf.SCORE_DETAIL_COLUMNS)

        assert len(details) == 150
        assert "missing-a" not in details
        assert "missing-b" not in details
        assert details["item-0149"]["author"] == "Author 149"
