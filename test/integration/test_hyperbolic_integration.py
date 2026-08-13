# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Hyperbolic Explorer integration tests against a real Postgres database.

Seeds the score and embedding tables (including the poincare columns),
exercises projection persistence, the backfill job, and the similarity and
tree engines against the live database, and drives the Flask endpoint routes
through their request handling.

Main Features:
* set_hyperbolic_projection round-trips a projected vector and radius
* backfill_hyperbolic_columns fills the new columns for every non-NULL embedding
* hyperbolic_similar re-ranks candidates by exact Poincare distance
* build_hyperbolic_tree_cache materializes root / mood / genre / track nodes
  from real rows; build_hyperbolic_tree then reads them back from the cache
* The tree cache persists as segmented BYTEA rows in ivf_dir (50 MB chunks,
  like every other index) and reassembles on load, so analysis writes it and
  Flask reads it back at startup
* Endpoint routes answer 200 / 400 with the documented payload shapes
"""

import base64
import importlib.util
import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from flask import Flask

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
except Exception:  # pragma: no cover - psycopg2 is in test/requirements.txt
    psycopg2 = None

import config

_DIM = 200

_SCORE_DDL = (
    "CREATE TABLE score (item_id TEXT PRIMARY KEY, title TEXT, author TEXT, "
    "album TEXT, album_artist TEXT, tempo REAL, key TEXT, scale TEXT, "
    "mood_vector TEXT, energy REAL, other_features TEXT, year INTEGER, "
    "rating INTEGER, file_path TEXT)"
)
_EMBEDDING_DDL = (
    "CREATE TABLE embedding (item_id TEXT PRIMARY KEY, embedding BYTEA, "
    "poincare_embedding BYTEA, hyperbolic_radius DOUBLE PRECISION, "
    "FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE)"
)
_APP_CONFIG_DDL = (
    "CREATE TABLE app_config (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
)
_IVF_DIR_DDL = (
    "CREATE TABLE ivf_dir (name VARCHAR(255) PRIMARY KEY, blob_data BYTEA NOT NULL, "
    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
)


@pytest.fixture(scope='session')
def pg_dsn():
    if psycopg2 is None:
        pytest.skip("psycopg2 not importable")
    dsn = os.environ.get('AUDIOMUSE_TEST_DATABASE_URL')
    if dsn:
        try:
            psycopg2.connect(dsn).close()
        except Exception as e:
            pytest.fail(f"AUDIOMUSE_TEST_DATABASE_URL is set but not reachable, refusing to skip: {e}")
        yield dsn
        return
    try:
        import pgserver
    except Exception:
        pytest.skip(
            "No test database. Set AUDIOMUSE_TEST_DATABASE_URL to a disposable "
            "DB, or `pip install pgserver` for an ephemeral local instance."
        )
    data_dir = tempfile.mkdtemp(prefix='audiomuse_hyper_pg_')
    server = pgserver.get_server(data_dir)
    try:
        yield server.get_uri()
    finally:
        server.cleanup()


@pytest.fixture
def hyper_db(pg_dsn):
    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS embedding")
        cur.execute("DROP TABLE IF EXISTS score CASCADE")
        cur.execute("DROP TABLE IF EXISTS app_config")
        cur.execute("DROP TABLE IF EXISTS ivf_dir")
        cur.execute(_SCORE_DDL)
        cur.execute(_EMBEDDING_DDL)
        cur.execute(_APP_CONFIG_DDL)
        cur.execute(_IVF_DIR_DDL)
        rows = []
        embeddings = []
        for i in range(20):
            iid = f"item-{i:03d}"
            rows.append((iid, f"Title {i}", f"Author {i}"))
            base = 0.5 + 0.2 * i
            vec = np.full(_DIM, 0.01, dtype=np.float32)
            vec[0] = base
            embeddings.append((iid, vec))
        cur.executemany(
            "INSERT INTO score (item_id, title, author) VALUES (%s, %s, %s)",
            rows,
        )
        cur.executemany(
            "INSERT INTO embedding (item_id, embedding) VALUES (%s, %s)",
            [(iid, psycopg2.Binary(vec.tobytes())) for iid, vec in embeddings],
        )
    yield conn
    conn.close()


@pytest.fixture
def _point_get_db_to_test(monkeypatch, hyper_db, pg_dsn):
    # NOTE: patch with the raw pg_dsn, never hyper_db.dsn - psycopg2's
    # conn.dsn REDACTS the password to 'xxx', so the streaming side
    # connection (_open_side_connection) would fail auth on any server that
    # actually checks passwords (CI's postgres service does).
    monkeypatch.setattr("app_helper.get_db", lambda: hyper_db)
    monkeypatch.setattr("database.get_db", lambda: hyper_db)
    # tasks.mediaserver.registry binds get_db by value at import time, and the
    # tree build now imports the registry (per-server tree targets), so point
    # its captured reference at the live test connection too. Without this the
    # registry keeps talking to a previous test's closed hyper_db connection.
    monkeypatch.setattr("tasks.mediaserver.registry.get_db", lambda: hyper_db)
    monkeypatch.setattr(config, "DATABASE_URL", pg_dsn)
    yield hyper_db


@pytest.fixture(autouse=True)
def _fixed_scale(monkeypatch):
    monkeypatch.setattr(config, "HYPERBOLIC_RADIUS_SCALE", 1.0)


def _load_hyperbolic_manager():
    mod_name = 'tasks.hyperbolic_manager'
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    mod_path = os.path.join(_REPO_ROOT, 'tasks', 'hyperbolic_manager.py')
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_app_hyperbolic():
    mod_name = 'app_hyperbolic'
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_REPO_ROOT, 'app_hyperbolic.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _seed_poincare(conn):
    from tasks.hyperbolic_geometry import poincare_radius, project_to_poincare

    with conn.cursor() as cur:
        cur.execute("SELECT item_id, embedding FROM embedding ORDER BY item_id")
        rows = cur.fetchall()
    for item_id, blob in rows:
        vec = np.frombuffer(bytes(blob), dtype=np.float32)
        proj = project_to_poincare(vec, 1.0)
        radius = float(poincare_radius(vec, 1.0))
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE embedding SET poincare_embedding = %s, hyperbolic_radius = %s "
                "WHERE item_id = %s",
                (psycopg2.Binary(proj.tobytes()), radius, item_id),
            )


def _get_radius(conn, item_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT hyperbolic_radius FROM embedding WHERE item_id = %s", (item_id,)
        )
        row = cur.fetchone()
        return float(row[0]) if row else None


@pytest.mark.integration
class TestProjectionPersistence:
    def test_set_hyperbolic_projection_roundtrip(self, _point_get_db_to_test):
        conn = _point_get_db_to_test
        from database import set_hyperbolic_projection

        proj = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        set_hyperbolic_projection("item-000", proj, 0.5)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT poincare_embedding, hyperbolic_radius FROM embedding WHERE item_id = %s",
                ("item-000",),
            )
            blob, radius = cur.fetchone()
        np.testing.assert_allclose(np.frombuffer(bytes(blob), dtype=np.float32), proj, rtol=1e-6)
        assert radius == 0.5

    def test_backfill_populates_poincare_columns(self, _point_get_db_to_test):
        conn = _point_get_db_to_test
        hm = _load_hyperbolic_manager()
        hm.reset_hyperbolic_scale_cache()
        try:
            total = hm.backfill_hyperbolic_columns()
        finally:
            hm.reset_hyperbolic_scale_cache()
        assert total == 20
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM embedding WHERE poincare_embedding IS NOT NULL "
                "AND hyperbolic_radius IS NOT NULL"
            )
            assert cur.fetchone()[0] == 20
        radius = _get_radius(conn, "item-010")
        assert 0.0 < radius < 1.0


@pytest.mark.integration
class TestSimilarityEngine:
    def test_hyperbolic_similar_reranks_candidates(self, _point_get_db_to_test):
        conn = _point_get_db_to_test
        _seed_poincare(conn)
        hm = _load_hyperbolic_manager()
        target = "item-010"
        candidates = [{"item_id": f"item-{i:03d}"} for i in range(20) if f"item-{i:03d}" != target]

        with patch("tasks.ivf_manager.find_nearest_neighbors_by_id", return_value=candidates):
            results = hm.hyperbolic_similar(target, mode="similar", limit=5)

        assert len(results) == 5
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)
        for r in results:
            assert r["item_id"] != target
            assert 0.0 < r["hyperbolic_radius"] < 1.0

    def test_hyperbolic_similar_roots_and_niche_split_by_radius(
        self, _point_get_db_to_test, monkeypatch
    ):
        conn = _point_get_db_to_test
        _seed_poincare(conn)
        hm = _load_hyperbolic_manager()
        target = "item-010"
        target_radius = _get_radius(conn, target)
        monkeypatch.setattr(config, "HYPERBOLIC_RADIAL_SPREAD", 0.15)

        roots = hm.hyperbolic_similar(target, mode="roots", limit=50)
        niche = hm.hyperbolic_similar(target, mode="niche", limit=50)

        assert roots
        assert niche
        assert all(r["hyperbolic_radius"] < target_radius for r in roots)
        assert all(r["hyperbolic_radius"] > target_radius for r in niche)
        # The radius window must move the modes visibly away from the seed's
        # radius band, not just a hair inward / outward.
        roots_hi = target_radius * (1.0 - 0.15)
        niche_lo = target_radius + (1.0 - target_radius) * 0.15
        assert all(r["hyperbolic_radius"] < roots_hi for r in roots)
        assert all(r["hyperbolic_radius"] > niche_lo for r in niche)

    def test_get_poincare_radius_returns_seed_radius(self, _point_get_db_to_test):
        conn = _point_get_db_to_test
        _seed_poincare(conn)
        hm = _load_hyperbolic_manager()
        expected = _get_radius(conn, "item-010")
        assert hm.get_poincare_radius("item-010") == expected

    def test_get_poincare_radius_none_when_unavailable(self, _point_get_db_to_test):
        hm = _load_hyperbolic_manager()
        # Unknown id and empty id must not raise, and an id whose hyperbolic
        # columns are still NULL must yield None too.
        assert hm.get_poincare_radius("item-999") is None
        assert hm.get_poincare_radius("") is None
        assert hm.get_poincare_radius(None) is None


@pytest.mark.integration
class TestTreeEngine:
    def test_tree_root_and_band_from_real_rows(self, _point_get_db_to_test):
        conn = _point_get_db_to_test
        _seed_poincare(conn)
        hm = _load_hyperbolic_manager()

        hm.reset_hyperbolic_tree_cache()
        try:
            track_count = hm.build_hyperbolic_tree_cache()
            node, flat = hm.build_hyperbolic_tree(None)
            assert node["id"] == "root"
            assert node["type"] == "folder"
            assert node["items"]
            assert all(i["type"] == "folder" for i in node["items"])
            # Root is never a leaf, so its own flat_ids stays empty; the total
            # track count is reported separately (build_hyperbolic_tree_cache's
            # return value), not derived from aggregating descendant lists.
            assert track_count == 20
            assert flat == []

            band = node["items"][0]
            band_node, band_flat = hm.build_hyperbolic_tree(band["id"])
            assert band_node["id"] == band["id"]
            assert band_node["children_count"] == len(band_node["items"])
            if band_node.get("leaf"):
                assert len(band_flat) == len(band_node["items"])
            else:
                assert all(i["type"] == "folder" for i in band_node["items"])
                assert len(band_flat) == band_node["summary"]["track_count"]
        finally:
            hm.reset_hyperbolic_tree_cache()

    def test_tree_cache_round_trips_through_real_postgres(self, _point_get_db_to_test):
        conn = _point_get_db_to_test
        _seed_poincare(conn)
        hm = _load_hyperbolic_manager()

        hm.reset_hyperbolic_tree_cache()
        try:
            built_count = hm.build_hyperbolic_tree_cache()
            built_node_ids = set(hm._TREE_CACHE["nodes"].keys())
            built_root = hm._TREE_CACHE["nodes"]["root"]

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT blob_data FROM ivf_dir WHERE name = %s",
                    (hm._TREE_CACHE_BLOB_NAME,),
                )
                single = cur.fetchone()
                cur.execute(
                    "SELECT name FROM ivf_dir WHERE name LIKE %s ESCAPE '\\'",
                    (hm._TREE_CACHE_BLOB_NAME.replace("_", r"\_") + r"\_%\_%",),
                )
                parts = cur.fetchall()
            assert single or parts

            hm.reset_hyperbolic_tree_cache()
            loaded_count = hm.load_hyperbolic_tree_cache()
            assert loaded_count == built_count == 20
            assert set(hm._TREE_CACHE["nodes"].keys()) == built_node_ids
            assert hm._TREE_CACHE["nodes"]["root"] == built_root
        finally:
            hm.reset_hyperbolic_tree_cache()

    def test_load_tree_cache_warms_per_server_blobs(self, _point_get_db_to_test):
        # A secondary server's tree blob must actually be DISCOVERED and loaded
        # at startup, not silently skipped. The scan LIKE pattern escapes the
        # prefix's underscores but the trailing % must stay a WILDCARD - a
        # literal-escaped % would match nothing and every server except the
        # default would silently fall back to the union tree (clusters labeled
        # with the union's track counts that only held that server's few tracks
        # when opened).
        conn = _point_get_db_to_test
        hm = _load_hyperbolic_manager()
        blob_name = f"{hm._TREE_CACHE_BLOB_NAME}__sec"

        payload = {
            "n_bands": 1,
            "nodes": {"root": {"id": "root", "type": "folder", "name": "Root",
                               "children_count": 0, "items": []}},
            "flat_ids": {},
            "track_count": 7,
        }
        default_payload = {
            "n_bands": 0,
            "nodes": {"root": {"id": "root", "type": "folder", "name": "Root",
                               "children_count": 0, "items": []}},
            "flat_ids": {},
            "track_count": 3,
        }
        hm.reset_hyperbolic_tree_cache()
        try:
            # The load path requires the default blob to exist (as it always
            # does after an analysis run) before it scans for per-server blobs.
            hm._persist_tree_cache_blob(default_payload, name=hm._TREE_CACHE_BLOB_NAME)
            hm._persist_tree_cache_blob(payload, name=blob_name)
            assert hm._scan_tree_cache_blob_names() == [blob_name]

            hm.reset_hyperbolic_tree_cache()
            hm.load_hyperbolic_tree_cache()
            sec_tree = hm._TREE_CACHE["servers"].get("sec")
            assert sec_tree is not None
            assert sec_tree["track_count"] == 7
            assert hm.tree_for_server("sec")["track_count"] == 7
        finally:
            hm.reset_hyperbolic_tree_cache()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM ivf_dir WHERE name IN (%s, %s)",
                            (hm._TREE_CACHE_BLOB_NAME, blob_name))
            conn.commit()

    def test_tree_cache_persists_segmented_and_reassembles(
        self, _point_get_db_to_test, monkeypatch
    ):
        # Force 1 MB parts so a payload that gzips above that threshold must
        # be split into multiple "name_i_n" rows, then reassembled on load.
        monkeypatch.setattr(config, "IVF_MAX_PART_SIZE_MB", 1)
        conn = _point_get_db_to_test
        hm = _load_hyperbolic_manager()
        payload = {
            "nodes": {"root": {"id": "root", "type": "folder", "name": "Root", "items": []}},
            "flat_ids": [],
            "track_count": 0,
            "data": base64.b64encode(os.urandom(1_400_000)).decode("ascii"),
        }
        hm.reset_hyperbolic_tree_cache()
        try:
            hm._persist_tree_cache_blob(payload)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM ivf_dir WHERE name LIKE %s ESCAPE '\\'",
                    (hm._TREE_CACHE_BLOB_NAME.replace("_", r"\_") + r"\_%\_%",),
                )
                part_count = cur.fetchone()[0]
            assert part_count >= 2
            assert hm._load_tree_cache_blob() == {**payload, "version": hm._TREE_CACHE_VERSION}
        finally:
            hm.reset_hyperbolic_tree_cache()
            hm._persist_tree_cache_blob(None)

    def test_tree_cache_build_survives_a_corrupted_row(self, _point_get_db_to_test):
        conn = _point_get_db_to_test
        _seed_poincare(conn)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE embedding SET hyperbolic_radius = 'NaN'::float8 "
                "WHERE item_id = %s",
                ("item-005",),
            )
        hm = _load_hyperbolic_manager()

        hm.reset_hyperbolic_tree_cache()
        try:
            track_count = hm.build_hyperbolic_tree_cache()
            node, flat = hm.build_hyperbolic_tree(None)
        finally:
            hm.reset_hyperbolic_tree_cache()

        assert track_count == 19
        assert "item-005" not in flat
        assert node["items"]

    def test_load_projected_mood_centroids_reads_the_real_file(self, _point_get_db_to_test):
        hm = _load_hyperbolic_manager()
        centroids = hm._load_projected_mood_centroids()
        assert centroids
        for c in centroids:
            assert c["vec"].shape == (_DIM,)
            assert np.linalg.norm(c["vec"]) < 1.0
            assert c["tags"]

    def test_load_projected_genre_subgenres_reads_the_real_file(self, _point_get_db_to_test):
        hm = _load_hyperbolic_manager()
        genre_subgenres = hm._load_projected_genre_subgenres()
        assert genre_subgenres
        for genre, info in genre_subgenres.items():
            assert info["vec"].shape == (_DIM,)
            assert np.linalg.norm(info["vec"]) < 1.0
            assert len(info["subgenres"]) >= 2
            for s in info["subgenres"]:
                assert s["vec"].shape == (_DIM,)
                assert s["name"]

    def test_tree_cluster_names_blend_tags_from_real_centroids(self, _point_get_db_to_test, monkeypatch):
        conn = _point_get_db_to_test
        _seed_poincare(conn)
        monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 2)
        # This test catalog is tiny (20 tracks) purely to exercise the naming
        # path; the per-server pruning floor would otherwise hide every cluster.
        monkeypatch.setattr(config, "HYPERBOLIC_MIN_CLUSTER_SIZE", 1)
        hm = _load_hyperbolic_manager()
        hm.reset_hyperbolic_tree_cache()
        try:
            # The mood fallback path (no genre_subgenre.json data) is the one
            # whose clusters are named by blending the nearest mood-centroid
            # tags; with the genre file usable the clusters sit under a genre
            # path and are prefix-named instead.
            with patch.object(hm, "_load_projected_genre_subgenres", return_value={}):
                hm.build_hyperbolic_tree_cache()
            nodes = hm._TREE_CACHE["nodes"]
            cluster_names = [n["name"] for node_id, n in nodes.items() if ".c" in node_id]
        finally:
            hm.reset_hyperbolic_tree_cache()
        assert cluster_names
        for name in cluster_names:
            assert " / " in name
            assert name.split(" (")[0].count(" / ") == 1

    def test_tree_build_is_per_server_with_two_configured_servers(
        self, _point_get_db_to_test, monkeypatch
    ):
        """The tree is built PER SERVER at analysis time, not as one union.

        With a default server and a secondary server configured in
        track_server_map, build_hyperbolic_tree_cache must produce two trees:
        the default tree from the default server's tracks and the secondary
        tree from only that server's tracks (no cross-server leakage), each
        persisted under its own blob name, and each resolvable through
        tree_for_server for the request path.
        """
        conn = _point_get_db_to_test
        from tasks.mediaserver import registry

        _seed_poincare(conn)

        with conn.cursor() as cur:
            # Other integration files (run earlier in the shared test DB) may
            # leave a music_servers table behind with an older schema plus a
            # chromaprint FK into it; DROP CASCADE + fresh CREATE guarantees
            # this test's own schema instead of silently no-opping against theirs.
            cur.execute("DROP TABLE IF EXISTS track_server_map, music_servers CASCADE")
            cur.execute(
                "CREATE TABLE music_servers ("
                "server_id TEXT PRIMARY KEY, name TEXT NOT NULL, "
                "server_type TEXT NOT NULL, creds JSONB NOT NULL DEFAULT '{}'::jsonb, "
                "music_libraries TEXT NOT NULL DEFAULT '', is_default BOOLEAN NOT NULL DEFAULT FALSE)"
            )
            cur.execute(
                "CREATE TABLE track_server_map ("
                "item_id TEXT NOT NULL, server_id TEXT NOT NULL, "
                "provider_track_id TEXT NOT NULL, match_tier TEXT, "
                "PRIMARY KEY (item_id, server_id))"
            )
            cur.executemany(
                "INSERT INTO music_servers (server_id, name, server_type, is_default) "
                "VALUES (%s, %s, %s, %s)",
                [
                    ("def", "Home", "jellyfin", True),
                    ("sec", "Plex", "plex", False),
                ],
            )
            # 10 tracks only on the default server, 10 only on Plex. Each
            # server's tree must be built from only that server's tracks.
            cur.executemany(
                "INSERT INTO track_server_map (item_id, server_id, provider_track_id, match_tier) "
                "VALUES (%s, %s, %s, %s)",
                [(f"item-{i:03d}", "def", f"home-{i:03d}", "fingerprint") for i in range(10)]
                + [(f"item-{i:03d}", "sec", f"plex-{i:03d}", "fingerprint") for i in range(10, 20)],
            )
        registry.invalidate_server_cache()

        hm = _load_hyperbolic_manager()
        hm.reset_hyperbolic_tree_cache()
        persisted = {}

        def _fake_persist(payload, name=None):
            persisted[name] = payload

        try:
            with patch.object(hm, "_persist_tree_cache_blob", side_effect=_fake_persist):
                default_count = hm.build_hyperbolic_tree_cache()

            # The default server keeps the legacy whole-catalogue semantic
            # (all non-fp_ tracks are available on it), so its tree spans the
            # full catalogue; the secondary server is strictly scoped to its
            # own mapped tracks.
            assert default_count == 20
            default_tree = hm._TREE_CACHE["servers"][hm._DEFAULT_SERVER_KEY]
            sec_tree = hm._TREE_CACHE["servers"]["sec"]
            assert default_tree["track_count"] == 20
            assert sec_tree["track_count"] == 10
            # No cross-server leakage: the secondary tree's leaf ids are
            # exactly the tracks mapped to that server.
            def _leaf_ids(tree):
                ids = set()
                for node in tree["nodes"].values():
                    if node.get("leaf"):
                        ids.update(tree["flat_ids"].get(node["id"]) or [])
                return ids

            assert _leaf_ids(default_tree) == {f"item-{i:03d}" for i in range(20)}
            assert _leaf_ids(sec_tree) == {f"item-{i:03d}" for i in range(10, 20)}
            # Persisted under distinct per-server blob names.
            assert set(persisted) == {hm._TREE_CACHE_BLOB_NAME, f"{hm._TREE_CACHE_BLOB_NAME}__sec"}
            # The request path resolves the right tree per server.
            assert hm.tree_for_server(None)["track_count"] == 20
            assert hm.tree_for_server("sec")["track_count"] == 10
            assert hm.tree_for_server("def")["track_count"] == 20
        finally:
            hm.reset_hyperbolic_tree_cache()
            registry.invalidate_server_cache()
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS track_server_map, music_servers CASCADE")


@pytest.mark.integration
class TestEndpoints:
    def _client(self):
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(_import_app_hyperbolic().hyperbolic_bp)
        return app.test_client()

    def test_similar_endpoint_missing_item_id_400(self, _point_get_db_to_test):
        client = self._client()
        response = client.post("/api/hyperbolic/similar", json={})
        assert response.status_code == 400

    def test_similar_endpoint_invalid_mode_400(self, _point_get_db_to_test):
        client = self._client()
        response = client.post(
            "/api/hyperbolic/similar", json={"item_id": "item-000", "mode": "bogus"}
        )
        assert response.status_code == 400

    def test_similar_endpoint_non_numeric_radial_spread_400(self, _point_get_db_to_test):
        client = self._client()
        response = client.post(
            "/api/hyperbolic/similar",
            json={"item_id": "item-000", "mode": "roots", "radial_spread": "not-a-number"},
        )
        assert response.status_code == 400

    def test_similar_endpoint_out_of_range_radial_spread_400(self, _point_get_db_to_test):
        client = self._client()
        response = client.post(
            "/api/hyperbolic/similar",
            json={"item_id": "item-000", "mode": "roots", "radial_spread": 1.5},
        )
        assert response.status_code == 400

    def test_similar_endpoint_passes_caller_radial_spread_to_manager(
        self, _point_get_db_to_test, monkeypatch
    ):
        conn = _point_get_db_to_test
        import app_server_context
        from app_hyperbolic import hyperbolic_bp

        _seed_poincare(conn)
        monkeypatch.setattr(app_server_context, "resolve_input_item_id", lambda raw, data=None: raw)
        monkeypatch.setattr(
            app_server_context, "scope_results",
            lambda rows, requested_n=None, id_key="item_id": rows,
        )
        monkeypatch.setattr(
            app_server_context, "translate_ids_for_request",
            lambda item_ids: {str(i): str(i) for i in (item_ids or [])},
        )

        with patch("tasks.hyperbolic_manager.hyperbolic_similar", return_value=[]) as sim:
            app = Flask(__name__)
            app.config["TESTING"] = True
            app.register_blueprint(hyperbolic_bp)
            response = app.test_client().post(
                "/api/hyperbolic/similar",
                json={"item_id": "item-000", "mode": "roots", "radial_spread": 0.42},
            )

        assert response.status_code == 200
        assert response.get_json()["radial_spread"] == pytest.approx(0.42)
        sim.assert_called_once()
        assert sim.call_args.kwargs["radial_spread"] == pytest.approx(0.42)

    def test_similar_endpoint_defaults_radial_spread_to_config(
        self, _point_get_db_to_test, monkeypatch
    ):
        conn = _point_get_db_to_test
        import app_server_context
        from app_hyperbolic import hyperbolic_bp

        _seed_poincare(conn)
        monkeypatch.setattr(config, "HYPERBOLIC_RADIAL_SPREAD", 0.33)
        monkeypatch.setattr(app_server_context, "resolve_input_item_id", lambda raw, data=None: raw)
        monkeypatch.setattr(
            app_server_context, "scope_results",
            lambda rows, requested_n=None, id_key="item_id": rows,
        )
        monkeypatch.setattr(
            app_server_context, "translate_ids_for_request",
            lambda item_ids: {str(i): str(i) for i in (item_ids or [])},
        )

        with patch("tasks.hyperbolic_manager.hyperbolic_similar", return_value=[]) as sim:
            app = Flask(__name__)
            app.config["TESTING"] = True
            app.register_blueprint(hyperbolic_bp)
            response = app.test_client().post(
                "/api/hyperbolic/similar",
                json={"item_id": "item-000", "mode": "niche"},
            )

        assert response.status_code == 200
        assert response.get_json()["radial_spread"] == pytest.approx(0.33)
        assert sim.call_args.kwargs["radial_spread"] == pytest.approx(0.33)

    def test_similar_endpoint_returns_200(self, _point_get_db_to_test, monkeypatch):
        conn = _point_get_db_to_test
        import app_server_context
        from app_hyperbolic import hyperbolic_bp

        _seed_poincare(conn)
        monkeypatch.setattr(app_server_context, "resolve_input_item_id", lambda raw, data=None: raw)
        monkeypatch.setattr(
            app_server_context, "scope_results",
            lambda rows, requested_n=None, id_key="item_id": rows,
        )
        monkeypatch.setattr(
            app_server_context, "translate_ids_for_request",
            lambda item_ids: {str(i): str(i) for i in (item_ids or [])},
        )

        canned = [
            {"item_id": "item-001", "distance": 0.25, "hyperbolic_radius": 0.55},
            {"item_id": "item-002", "distance": 0.4, "hyperbolic_radius": 0.6},
        ]
        with patch("tasks.hyperbolic_manager.hyperbolic_similar", return_value=canned) as sim:
            app = Flask(__name__)
            app.config["TESTING"] = True
            app.register_blueprint(hyperbolic_bp)
            response = app.test_client().post(
                "/api/hyperbolic/similar",
                json={"item_id": "item-000", "mode": "niche", "limit": 2},
            )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["mode"] == "niche"
        assert payload["count"] == 2
        assert payload["results"][0]["item_id"] == "item-001"
        assert payload["seed_item_id"] == "item-000"
        assert payload["seed_radius"] == _get_radius(conn, "item-000")
        sim.assert_called_once()

    def test_similar_endpoint_attaches_title_author_even_when_ids_are_translated(
        self, _point_get_db_to_test, monkeypatch
    ):
        import app_server_context
        from app_hyperbolic import hyperbolic_bp

        monkeypatch.setattr(app_server_context, "resolve_input_item_id", lambda raw, data=None: raw)

        def _fake_scope_results(rows, requested_n=None, id_key="item_id"):
            for r in rows:
                r[id_key] = "provider-" + r[id_key]
            return rows

        monkeypatch.setattr(app_server_context, "scope_results", _fake_scope_results)
        monkeypatch.setattr(
            app_server_context, "translate_ids_for_request",
            lambda item_ids: {str(i): str(i) for i in (item_ids or [])},
        )

        canned = [
            {"item_id": "item-001", "distance": 0.25, "hyperbolic_radius": 0.55},
            {"item_id": "item-002", "distance": 0.4, "hyperbolic_radius": 0.6},
        ]
        with patch("tasks.hyperbolic_manager.hyperbolic_similar", return_value=canned):
            app = Flask(__name__)
            app.config["TESTING"] = True
            app.register_blueprint(hyperbolic_bp)
            response = app.test_client().post(
                "/api/hyperbolic/similar",
                json={"item_id": "item-000", "mode": "similar", "limit": 2},
            )

        assert response.status_code == 200
        results = response.get_json()["results"]
        by_id = {r["item_id"]: r for r in results}
        assert by_id["provider-item-001"]["title"] == "Title 1"
        assert by_id["provider-item-001"]["author"] == "Author 1"
        assert by_id["provider-item-002"]["title"] == "Title 2"
        assert by_id["provider-item-002"]["author"] == "Author 2"

    def test_similar_endpoint_scopes_results_to_the_selected_server(
        self, _point_get_db_to_test, monkeypatch
    ):
        """End-to-end per-server scoping of the similar search.

        Uses the REAL scope_results / resolve_input_item_id / translate_ids with
        two configured servers in track_server_map: a track that only exists on
        the default server must NOT leak into a Plex-scoped response, and every
        surviving id must be Plex's own provider id.
        """
        conn = _point_get_db_to_test
        from app_hyperbolic import hyperbolic_bp
        from tasks.mediaserver import registry

        _seed_poincare(conn)

        with conn.cursor() as cur:
            # Other integration files (run earlier in the shared test DB) may
            # leave a music_servers table behind with an older schema plus a
            # chromaprint FK into it; DROP CASCADE + fresh CREATE guarantees
            # this test's own schema instead of silently no-opping against theirs.
            cur.execute("DROP TABLE IF EXISTS track_server_map, music_servers CASCADE")
            cur.execute(
                "CREATE TABLE music_servers ("
                "server_id TEXT PRIMARY KEY, name TEXT NOT NULL, "
                "server_type TEXT NOT NULL, creds JSONB NOT NULL DEFAULT '{}'::jsonb, "
                "music_libraries TEXT NOT NULL DEFAULT '', is_default BOOLEAN NOT NULL DEFAULT FALSE)"
            )
            cur.execute(
                "CREATE TABLE track_server_map ("
                "item_id TEXT NOT NULL, server_id TEXT NOT NULL, "
                "provider_track_id TEXT NOT NULL, match_tier TEXT, "
                "PRIMARY KEY (item_id, server_id))"
            )
            cur.executemany(
                "INSERT INTO music_servers (server_id, name, server_type, is_default) "
                "VALUES (%s, %s, %s, %s)",
                [
                    ("def", "Home", "jellyfin", True),
                    ("sec", "Plex", "plex", False),
                ],
            )
            # item-001 and item-002 exist on BOTH servers; item-003 only on the
            # default server; item-004 only on Plex.
            cur.executemany(
                "INSERT INTO track_server_map (item_id, server_id, provider_track_id, match_tier) "
                "VALUES (%s, %s, %s, %s)",
                [
                    ("item-001", "def", "home-001", "fingerprint"),
                    ("item-001", "sec", "plex-001", "fingerprint"),
                    ("item-002", "def", "home-002", "fingerprint"),
                    ("item-002", "sec", "plex-002", "fingerprint"),
                    ("item-003", "def", "home-003", "fingerprint"),
                    ("item-004", "sec", "plex-004", "fingerprint"),
                ],
            )
        registry.invalidate_server_cache()

        canned = [
            {"item_id": "item-001", "distance": 0.25, "hyperbolic_radius": 0.55},
            {"item_id": "item-002", "distance": 0.4, "hyperbolic_radius": 0.6},
            {"item_id": "item-003", "distance": 0.5, "hyperbolic_radius": 0.65},
            {"item_id": "item-004", "distance": 0.6, "hyperbolic_radius": 0.7},
        ]
        try:
            with patch("tasks.hyperbolic_manager.hyperbolic_similar", return_value=canned):
                app = Flask(__name__)
                app.config["TESTING"] = True
                app.register_blueprint(hyperbolic_bp)
                response = app.test_client().post(
                    "/api/hyperbolic/similar?server=Plex",
                    json={"item_id": "plex-001", "mode": "similar", "limit": 10},
                )
        finally:
            registry.invalidate_server_cache()
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS track_server_map, music_servers CASCADE")

        assert response.status_code == 200
        payload = response.get_json()
        result_ids = [r["item_id"] for r in payload["results"]]
        # Plex only: item-003 (default-only) dropped, ids are Plex provider ids.
        assert result_ids == ["plex-001", "plex-002", "plex-004"]
        assert not any(str(i).startswith(("item-", "home-", "fp_")) for i in result_ids)
        assert payload["seed_item_id"] == "plex-001"

    def test_tree_endpoint_returns_directory_json(self, _point_get_db_to_test, monkeypatch):
        import app_server_context
        from app_hyperbolic import hyperbolic_bp
        from tasks.hyperbolic_manager import _TREE_CACHE

        monkeypatch.setattr(
            app_server_context,
            "translate_ids_for_request",
            lambda ids: {i: i for i in ids},
        )
        node = {
            "id": "root",
            "name": "Hyperbolic Explorer",
            "type": "folder",
            "children_count": 1,
            "items": [{"id": "item-001", "name": "T - A", "type": "track", "children_count": 0, "items": []}],
        }
        # The API scopes the tree to the request's server by walking the shared
        # cache, so the test must populate the cache's nodes/flat_ids the way a
        # real build does.
        saved = dict(_TREE_CACHE)
        try:
            _TREE_CACHE["nodes"] = {"root": node}
            _TREE_CACHE["flat_ids"] = {"root": ["item-001"]}
            with patch("tasks.hyperbolic_manager.build_hyperbolic_tree",
                       return_value=(node, ["item-001"])) as tree:
                app = Flask(__name__)
                app.config["TESTING"] = True
                app.register_blueprint(hyperbolic_bp)
                response = app.test_client().get("/api/hyperbolic/tree")
        finally:
            _TREE_CACHE.clear()
            _TREE_CACHE.update(saved)

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["node"]["id"] == "root"
        assert payload["node"]["items"][0]["type"] == "track"
        tree.assert_called_once()

    def test_tree_endpoint_unknown_node_400(self, _point_get_db_to_test, monkeypatch):
        import app_server_context
        from app_hyperbolic import hyperbolic_bp

        monkeypatch.setattr(
            app_server_context,
            "translate_ids_for_request",
            lambda ids: {i: i for i in ids},
        )

        def _boom(node_id, server_id=None):
            raise ValueError("Unknown tree node id: nope")

        with patch("tasks.hyperbolic_manager.build_hyperbolic_tree", side_effect=_boom):
            app = Flask(__name__)
            app.config["TESTING"] = True
            app.register_blueprint(hyperbolic_bp)
            response = app.test_client().get("/api/hyperbolic/tree?node_id=nope")

        assert response.status_code == 400

    def test_tree_endpoint_keeps_non_leaf_bands_in_the_root(self, _point_get_db_to_test, monkeypatch):
        import app_server_context
        from app_hyperbolic import hyperbolic_bp
        from tasks.hyperbolic_manager import _TREE_CACHE

        monkeypatch.setattr(
            app_server_context,
            "translate_ids_for_request",
            lambda ids: {i: i for i in ids},
        )
        root = {
            "id": "root",
            "name": "Hyperbolic Explorer",
            "type": "folder",
            "children_count": 2,
            "summary": {"track_count": 200},
            "items": [
                {
                    "id": "b0",
                    "name": "Band 1",
                    "type": "folder",
                    "leaf": False,
                    "children_count": 2,
                    "summary": {"track_count": 120},
                    "items": [
                        {"id": "b0.c0", "name": "Cluster 1", "type": "folder",
                         "children_count": 60, "summary": {"track_count": 60}, "items": []},
                        {"id": "b0.c1", "name": "Cluster 2", "type": "folder",
                         "children_count": 60, "summary": {"track_count": 60}, "items": []},
                    ],
                },
                {
                    "id": "b1",
                    "name": "Band 2",
                    "type": "folder",
                    "leaf": True,
                    "children_count": 2,
                    "summary": {"track_count": 2},
                    "items": [
                        {"id": "item-001", "name": "T - A", "type": "track", "children_count": 0, "items": []},
                        {"id": "item-002", "name": "T2 - A2", "type": "track", "children_count": 0, "items": []},
                    ],
                },
            ],
        }
        # Populate the shared cache the way a real build does so the per-server
        # scope walk can find every track under the non-leaf band b0.
        cached_b0 = dict(root["items"][0])
        cached_b0["items"] = [
            {"id": "b0.c0", "name": "Cluster 1", "type": "folder", "children_count": 60,
             "summary": {"track_count": 60}, "items": []},
            {"id": "b0.c1", "name": "Cluster 2", "type": "folder", "children_count": 60,
             "summary": {"track_count": 60}, "items": []},
        ]
        cached_b1 = dict(root["items"][1])
        cached_b1["items"] = [
            {"id": "item-001", "name": "T - A", "type": "track", "children_count": 0, "items": []},
            {"id": "item-002", "name": "T2 - A2", "type": "track", "children_count": 0, "items": []},
        ]
        saved = dict(_TREE_CACHE)
        try:
            _TREE_CACHE["nodes"] = {
                "root": root,
                "b0": cached_b0,
                "b0.c0": {"id": "b0.c0", "type": "folder", "items": []},
                "b0.c1": {"id": "b0.c1", "type": "folder", "items": []},
                "b1": cached_b1,
            }
            _TREE_CACHE["flat_ids"] = {
                "b0.c0": ["item-001"],
                "b0.c1": ["item-002"],
                "b1": ["item-001", "item-002"],
            }
            with patch("tasks.hyperbolic_manager.build_hyperbolic_tree",
                       return_value=(root, ["item-001", "item-002"])):
                app = Flask(__name__)
                app.config["TESTING"] = True
                app.register_blueprint(hyperbolic_bp)
                response = app.test_client().get("/api/hyperbolic/tree")
        finally:
            _TREE_CACHE.clear()
            _TREE_CACHE.update(saved)

        assert response.status_code == 200
        node = response.get_json()["node"]
        assert node["id"] == "root"
        assert node["children_count"] == 2
        assert [c["id"] for c in node["items"]] == ["b0", "b1"]
        b0 = node["items"][0]
        assert b0["leaf"] is False
        assert [c["id"] for c in b0["items"]] == ["b0.c0", "b0.c1"]
        assert [c["items"] for c in b0["items"]] == [[], []]
        b1 = node["items"][1]
        assert b1["leaf"] is True
        assert [t["id"] for t in b1["items"]] == ["item-001", "item-002"]

    def test_cache_status_reflects_a_backfill_and_rebuild(self, _point_get_db_to_test):
        hm = _load_hyperbolic_manager()
        hm.reset_hyperbolic_tree_cache()

        from app_hyperbolic import hyperbolic_bp

        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(hyperbolic_bp)
        try:
            empty_status = app.test_client().get("/api/hyperbolic/cache_status")
            assert empty_status.get_json()["ok"] is False

            backfilled = hm.backfill_hyperbolic_columns()
            track_count = hm.build_hyperbolic_tree_cache()
            assert backfilled == 20
            assert track_count == 20

            status = app.test_client().get("/api/hyperbolic/cache_status")
            assert status.get_json()["ok"] is True
            assert status.get_json()["track_count"] == 20
        finally:
            hm.reset_hyperbolic_tree_cache()
