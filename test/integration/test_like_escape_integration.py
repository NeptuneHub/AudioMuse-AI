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
* every table lives in a private schema that is dropped again afterwards, so
  the shared session database is left as it was found
* the old-scheme signature predicate, the canonicalize scheme predicates, the
  catalogue canonical-id probe, the fingerprint index load, the analysis
  legacy work scan and the two partial-index predicates init_db creates on
  score all keep their escaped underscore, so a provider id that merely starts
  with fp is never taken for a fingerprint id
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
_SCORE_DDL = (
    "CREATE TABLE IF NOT EXISTS score (item_id TEXT PRIMARY KEY, duration DOUBLE PRECISION, "
    "tempo REAL, energy REAL, key TEXT, scale TEXT, mood_vector TEXT, other_features TEXT)"
)
_EMBEDDING_DDL = "CREATE TABLE IF NOT EXISTS embedding (item_id TEXT PRIMARY KEY)"
_MAP_DDL = "CREATE TABLE IF NOT EXISTS track_server_map (item_id TEXT, match_tier TEXT)"
_SCHEMA = "like_escape_test"
_NAME = "clap_index__ivf_dir"
_LOOKALIKE = "clap_index__ivf_dirX1_2"
_TWO_PARTS = 1024 * 1024 + 1024


@pytest.fixture
def off_db(shared_pg_dsn):
    admin = psycopg2.connect(shared_pg_dsn)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
        cur.execute(f"CREATE SCHEMA {_SCHEMA}")
    conn = psycopg2.connect(
        shared_pg_dsn, options=f"-c search_path={_SCHEMA} -c standard_conforming_strings=off"
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        for ddl in (_DIR_DDL, _INDEX_DDL, _SCORE_DDL, _EMBEDDING_DDL, _MAP_DDL):
            cur.execute(ddl)
    yield conn
    conn.close()
    with admin.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
    admin.close()


def _ids():
    from tasks.simhash import CANONICAL_ID_LEN, CURRENT_ID_HEAD

    tail = "0" * (CANONICAL_ID_LEN - len(CURRENT_ID_HEAD))
    return CURRENT_ID_HEAD + tail, "fpX" + CURRENT_ID_HEAD[3:] + tail, "fp_0" + tail


def _insert_scores(conn, ids):
    with conn.cursor() as cur:
        for item_id in ids:
            cur.execute(
                "INSERT INTO score (item_id, duration, tempo, energy, key, scale, mood_vector, other_features) "
                "VALUES (%s, 1.0, 120, 0.5, 'C', 'major', 'm', 'o')",
                (item_id,),
            )


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


def test_tables_live_in_the_private_schema_not_in_public(off_db):
    with off_db.cursor() as cur:
        cur.execute("SELECT table_schema FROM information_schema.tables WHERE table_name = 'score'")
        assert [row[0] for row in cur.fetchall()] == [_SCHEMA]


def test_canonicalize_scheme_predicates_treat_a_provider_id_starting_with_fp_as_legacy_with_strings_off(off_db):
    from tasks import fingerprint_canonicalize as fc
    from tasks.simhash import CANONICAL_ID_LEN

    current, decoy, unsignable = _ids()
    _insert_scores(off_db, [current, decoy, unsignable, "abc"])
    with off_db.cursor() as cur:
        cur.execute(f"SELECT s.item_id FROM score s WHERE {fc._CURRENT_SCHEME_SQL} ORDER BY 1", (CANONICAL_ID_LEN,))
        assert [row[0] for row in cur.fetchall()] == [current]
        cur.execute(f"SELECT s.item_id FROM score s WHERE {fc._LEGACY_ROW_SQL} ORDER BY 1", (CANONICAL_ID_LEN,))
        assert [row[0] for row in cur.fetchall()] == ["abc", decoy]


def test_catalogue_canonical_id_probe_ignores_a_provider_id_starting_with_fp_with_strings_off(off_db, monkeypatch):
    import app_helper

    current, decoy, _unsignable = _ids()
    monkeypatch.setattr(app_helper, "get_db", lambda: off_db)
    _insert_scores(off_db, [current])
    assert app_helper.probe_catalogue_canonical_ids() is True
    with off_db.cursor() as cur:
        cur.execute("DELETE FROM score")
    _insert_scores(off_db, [decoy])
    assert app_helper.probe_catalogue_canonical_ids() is False


def test_fingerprint_index_load_registers_only_real_signature_ids_with_strings_off(off_db, monkeypatch):
    from tasks import simhash
    from tasks.analysis import helper

    class _Resolver:
        def __init__(self, **_kw):
            self.ids = []

        def register(self, item_id, duration=None):
            self.ids.append(item_id)

    current, decoy, unsignable = _ids()
    _insert_scores(off_db, [current, decoy, unsignable])
    monkeypatch.setattr(helper, "get_db", lambda: off_db)
    monkeypatch.setattr(simhash, "CatalogResolver", _Resolver)
    monkeypatch.setitem(helper._fingerprint_index_cache, "resolver", None)
    assert helper.load_fingerprint_index().ids == [current]


def test_analysis_legacy_work_scan_keeps_a_provider_id_starting_with_fp_with_strings_off(off_db):
    from tasks.analysis.helper import _work_sql

    current, decoy, _unsignable = _ids()
    _insert_scores(off_db, [current, decoy, "abc"])
    with off_db.cursor() as cur:
        for item_id in (current, decoy, "abc"):
            cur.execute("INSERT INTO embedding (item_id) VALUES (%s)", (item_id,))
        _mapped_sql, legacy_sql = _work_sql(False, False)
        cur.execute(legacy_sql + " AND s.item_id > %s ORDER BY s.item_id LIMIT %s", ("", 100))
        assert [row[0] for row in cur.fetchall()] == ["abc", decoy]


def _stored_index_predicate_rows(cur, index_name):
    cur.execute(
        "SELECT pg_get_expr(i.indpred, i.indrelid) FROM pg_index i "
        "JOIN pg_class c ON c.oid = i.indexrelid JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = %s AND c.relname = %s",
        (_SCHEMA, index_name),
    )
    predicate = cur.fetchone()[0]
    cur.execute(f"SELECT item_id FROM score WHERE {predicate} ORDER BY 1")
    return [row[0] for row in cur.fetchall()]


def test_score_partial_index_predicates_keep_the_escaped_underscore_with_strings_off(off_db):
    import database
    from tasks.simhash import CANONICAL_ID_LEN, CURRENT_ID_HEAD

    current, decoy, unsignable = _ids()
    old_head = CURRENT_ID_HEAD[:-1] + str(int(CURRENT_ID_HEAD[-1]) - 1)
    old_scheme = old_head + "0" * (CANONICAL_ID_LEN - len(old_head))
    _insert_scores(off_db, [current, decoy, unsignable, old_scheme, "abc"])
    with off_db.cursor() as cur:
        cur.execute(database._SCORE_LEGACY_ID_INDEX_SQL)
        cur.execute(database._score_old_scheme_index_sql())
        assert _stored_index_predicate_rows(cur, "idx_score_legacy_item_id") == ["abc", decoy]
        assert _stored_index_predicate_rows(cur, "idx_score_old_scheme") == [old_scheme]
