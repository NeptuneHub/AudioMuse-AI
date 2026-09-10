# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Segmented-blob LIKE queries against a Postgres with standard_conforming_strings off.

Issue #901: a server configured with standard_conforming_strings = off parsed
the plain-string backslash ESCAPE clause as an unterminated string and every
index store, load and size check failed. The helpers now write the clause as
an E-prefixed string literal, which reads as one backslash in both string
modes, so the same patterns keep matching only the real segment rows.

Main Features:
* The fixture connection really runs with standard_conforming_strings off
* store, load, length and completeness of a segmented ivf_dir blob work
* re-storing a name deletes only its own segments, a look-alike row survives
* the segmented index-table store works the same way
* the shared old-scheme signature-id predicate keeps its escaped underscore, so
  a provider id that merely starts with fp is not taken for a fingerprint id
"""

import json
import os
import sys

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
except Exception:
    psycopg2 = None

pytestmark = pytest.mark.integration

_DIR_DDL = (
    "CREATE TABLE IF NOT EXISTS ivf_dir (name VARCHAR(255) PRIMARY KEY, "
    "blob_data BYTEA NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
)
_INDEX_DDL = (
    "CREATE TABLE IF NOT EXISTS escape_index (index_name VARCHAR(255) PRIMARY KEY, "
    "index_data BYTEA, id_map_json TEXT, embedding_dimension INTEGER, "
    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
)
_SCORE_DDL = "CREATE TABLE IF NOT EXISTS score (item_id TEXT PRIMARY KEY)"
_NAME = "clap_index__ivf_dir"
_LOOKALIKE = "clap_index__ivf_dirX1_2"
_TWO_PARTS = 1024 * 1024 + 1024


@pytest.fixture
def off_db(shared_pg_dsn):
    conn = psycopg2.connect(shared_pg_dsn, options="-c standard_conforming_strings=off")
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS ivf_dir")
        cur.execute("DROP TABLE IF EXISTS escape_index")
        cur.execute("DROP TABLE IF EXISTS score")
        cur.execute(_DIR_DDL)
        cur.execute(_INDEX_DDL)
        cur.execute(_SCORE_DDL)
    yield conn
    conn.close()


def _names(conn, table, column):
    with conn.cursor() as cur:
        cur.execute(f"SELECT {column} FROM {table} ORDER BY 1")
        return [row[0] for row in cur.fetchall()]


def test_fixture_connection_runs_with_standard_conforming_strings_off(off_db):
    with off_db.cursor() as cur:
        cur.execute("SHOW standard_conforming_strings")
        assert cur.fetchone()[0] == "off"


def test_segmented_blob_store_load_length_and_complete_work_with_strings_off(off_db):
    from tasks.index_build_helpers import (
        load_segmented_blob,
        segmented_blob_complete,
        segmented_blob_length,
        store_segmented_blob,
    )

    blob = bytes(range(256)) * (_TWO_PARTS // 256)
    store_segmented_blob(off_db, "ivf_dir", _NAME, blob, max_part_size_mb=1)

    assert _names(off_db, "ivf_dir", "name") == [f"{_NAME}_1_2", f"{_NAME}_2_2"]
    assert load_segmented_blob(off_db, "ivf_dir", _NAME) == blob
    assert segmented_blob_length(off_db, "ivf_dir", _NAME) == len(blob)
    assert segmented_blob_complete(off_db, "ivf_dir", _NAME) is True


def test_restoring_a_name_deletes_only_its_own_segments_with_strings_off(off_db):
    from tasks.index_build_helpers import load_segmented_blob, store_segmented_blob

    first = b"\x01" * _TWO_PARTS
    second = b"\x02" * _TWO_PARTS
    store_segmented_blob(off_db, "ivf_dir", _NAME, first, max_part_size_mb=1)
    with off_db.cursor() as cur:
        cur.execute("INSERT INTO ivf_dir (name, blob_data) VALUES (%s, %s)", (_LOOKALIKE, b"x"))

    store_segmented_blob(off_db, "ivf_dir", _NAME, second, max_part_size_mb=1)

    assert set(_names(off_db, "ivf_dir", "name")) == {f"{_NAME}_1_2", f"{_NAME}_2_2", _LOOKALIKE}
    assert load_segmented_blob(off_db, "ivf_dir", _NAME) == second


def test_segmented_index_store_replaces_its_own_segments_with_strings_off(off_db):
    from tasks.index_build_helpers import store_ivf_index_segmented

    id_map = {str(i): f"item{i}" for i in range(10)}
    store_ivf_index_segmented(off_db, "escape_index", _NAME, b"\x03" * _TWO_PARTS, id_map, 8, max_part_size_mb=1)
    store_ivf_index_segmented(off_db, "escape_index", _NAME, b"\x04" * _TWO_PARTS, id_map, 8, max_part_size_mb=1)

    assert _names(off_db, "escape_index", "index_name") == [f"{_NAME}_1_2", f"{_NAME}_2_2"]
    with off_db.cursor() as cur:
        cur.execute("SELECT index_data, id_map_json FROM escape_index WHERE index_name = %s", (f"{_NAME}_1_2",))
        data, id_map_json = cur.fetchone()
    assert bytes(data)[:1] == b"\x04"
    assert json.loads(id_map_json) == id_map


def test_old_scheme_signature_predicate_rejects_provider_id_starting_with_fp_with_strings_off(off_db):
    from tasks.simhash import CANONICAL_ID_LEN, CURRENT_ID_HEAD, signature_id_sql

    old_head = CURRENT_ID_HEAD[:-1] + str(int(CURRENT_ID_HEAD[-1]) - 1)
    tail = "0" * (CANONICAL_ID_LEN - len(old_head))
    old_scheme_id = old_head + tail
    provider_id = "fpX" + old_head[3:] + tail
    with off_db.cursor() as cur:
        cur.execute("INSERT INTO score (item_id) VALUES (%s), (%s)", (old_scheme_id, provider_id))

    sql, params = signature_id_sql()
    with off_db.cursor() as cur:
        cur.execute(f"SELECT item_id FROM score WHERE {sql}", params)
        assert [row[0] for row in cur.fetchall()] == [old_scheme_id]
