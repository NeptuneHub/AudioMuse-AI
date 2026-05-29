# tests/unit/test_artist_metadata_codec.py
"""
Unit tests for the artist metadata storage helpers in
``tasks/index_build_helpers.py``:

- ``pack_artist_metadata`` / ``unpack_artist_metadata`` (binary codec)
- ``store_segmented_blob`` / ``load_segmented_blob`` (segmented BYTEA storage)

The codec replaces the previous JSON-of-floats storage of artist GMM
parameters. The segmented-blob helpers decouple metadata storage from the
Voyager index storage so a single PG column value cannot exceed
``MaxAllocSize`` (1 GB) at any realistic library size.

The helper module is loaded via importlib so this file does not pull in
``tasks/__init__.py`` (which imports librosa). Same pattern the other
tasks unit tests use.
"""

import importlib.util
import os
import sys
import types

import numpy as np
import pytest
from unittest.mock import MagicMock


def _load_helpers():
    """Load tasks.index_build_helpers without triggering tasks/__init__.py."""
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    )
    tasks_dir = os.path.join(repo_root, 'tasks')
    if 'tasks' not in sys.modules:
        stub = types.ModuleType('tasks')
        stub.__path__ = [tasks_dir]
        sys.modules['tasks'] = stub

    mod_path = os.path.join(tasks_dir, 'index_build_helpers.py')
    mod_name = 'tasks.index_build_helpers'
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name]


_helpers = _load_helpers()


def _make_synthetic_artists(n_artists, n_features=8, rng_seed=42):
    """Return (artist_map, artist_gmms) with random GMM params.

    Mix of few-songs (1-4 components) and multi-song (5+ components) artists
    to exercise both code paths in ``pack_artist_metadata``.
    """
    rng = np.random.default_rng(rng_seed)
    artist_map = {}
    artist_gmms = {}
    for i in range(n_artists):
        name = f"Artist {i}"
        artist_map[i] = name

        if i % 3 == 0:
            n_components = 1 + (i % 3)
            is_few_songs = True
        else:
            n_components = 5
            is_few_songs = False

        means = rng.standard_normal((n_components, n_features)).astype(np.float32)
        weights = rng.uniform(0.0, 1.0, size=n_components).astype(np.float32)
        weights = weights / weights.sum()

        artist_gmms[name] = {
            'means': means.tolist(),
            'weights': weights.tolist(),
            'n_components': n_components,
            'n_features': n_features,
            'n_tracks': n_components * 2,
            'is_few_songs': is_few_songs,
            'tracks_hash': f"{i:032x}",
        }
    return artist_map, artist_gmms


class TestPackUnpackArtistMetadata:

    def test_round_trip_preserves_artist_map(self):
        artist_map, artist_gmms = _make_synthetic_artists(5)
        blob = _helpers.pack_artist_metadata(artist_map, artist_gmms)
        loaded_map, _ = _helpers.unpack_artist_metadata(blob)
        assert loaded_map == artist_map

    def test_round_trip_preserves_gmm_params(self):
        artist_map, artist_gmms = _make_synthetic_artists(7)
        blob = _helpers.pack_artist_metadata(artist_map, artist_gmms)
        _, loaded_gmms = _helpers.unpack_artist_metadata(blob)
        assert set(loaded_gmms.keys()) == set(artist_gmms.keys())
        for name, original in artist_gmms.items():
            got = loaded_gmms[name]
            assert got['n_components']  == original['n_components']
            assert got['n_features']    == original['n_features']
            assert got['n_tracks']      == original['n_tracks']
            assert got['is_few_songs']  == original['is_few_songs']
            assert got['tracks_hash']   == original['tracks_hash']
            np.testing.assert_allclose(
                np.asarray(got['means'],   dtype=np.float32),
                np.asarray(original['means'], dtype=np.float32),
                atol=0,
            )
            np.testing.assert_allclose(
                np.asarray(got['weights'], dtype=np.float32),
                np.asarray(original['weights'], dtype=np.float32),
                atol=0,
            )

    def test_covariances_and_covariance_type_never_appear_in_output(self):
        """Even if the input still carries them (mid-migration), the codec
        must not propagate them. covariances has zero live readers."""
        artist_map, artist_gmms = _make_synthetic_artists(3)
        for gmm in artist_gmms.values():
            gmm['covariances'] = [[0.01] * gmm['n_features']] * gmm['n_components']
            gmm['covariance_type'] = 'diag'

        blob = _helpers.pack_artist_metadata(artist_map, artist_gmms)
        _, loaded_gmms = _helpers.unpack_artist_metadata(blob)
        for got in loaded_gmms.values():
            assert 'covariances' not in got
            assert 'covariance_type' not in got

    def test_empty_input_yields_valid_header(self):
        blob = _helpers.pack_artist_metadata({}, {})
        assert len(blob) >= _helpers._ARTIST_META_HEADER_SIZE
        assert blob[:4] == _helpers._ARTIST_META_MAGIC
        loaded_map, loaded_gmms = _helpers.unpack_artist_metadata(blob)
        assert loaded_map == {}
        assert loaded_gmms == {}

    def test_unicode_artist_names_round_trip(self):
        artist_map = {0: "Sigur Rós", 1: "東京事変", 2: "Mötley Crüe"}
        artist_gmms = {
            name: {
                'means':       [[0.1, 0.2, 0.3]],
                'weights':     [1.0],
                'n_components': 1,
                'n_features':   3,
                'n_tracks':     1,
                'is_few_songs': True,
                'tracks_hash':  f"{i:032x}",
            }
            for i, name in artist_map.items()
        }
        blob = _helpers.pack_artist_metadata(artist_map, artist_gmms)
        loaded_map, loaded_gmms = _helpers.unpack_artist_metadata(blob)
        assert loaded_map == artist_map
        assert set(loaded_gmms.keys()) == set(artist_gmms.keys())

    def test_unpack_rejects_bad_magic(self):
        blob = b"XXXX" + b"\x00" * 100
        with pytest.raises(ValueError, match="magic"):
            _helpers.unpack_artist_metadata(blob)

    def test_unpack_rejects_truncated_blob(self):
        with pytest.raises(ValueError, match="too short"):
            _helpers.unpack_artist_metadata(b"only-a-few-bytes")

    def test_means_shape_mismatch_raises_on_pack(self):
        artist_map = {0: "X"}
        artist_gmms = {
            "X": {
                'means':       [[1.0, 2.0]],
                'weights':     [1.0],
                'n_components': 1,
                'n_features':   3,
                'n_tracks':     1,
                'is_few_songs': True,
                'tracks_hash':  "0" * 32,
            }
        }
        with pytest.raises(ValueError, match="means shape"):
            _helpers.pack_artist_metadata(artist_map, artist_gmms)


class TestStoreLoadSegmentedBlob:
    """Mock psycopg2 cursor to verify the segmented-blob storage pattern."""

    def _captured_conn(self):
        captured = []
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)

        def execute_side(sql, params=None):
            captured.append((sql, params))

        mock_cur.execute.side_effect = execute_side
        mock_cur.fetchone.return_value = None
        mock_cur.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        return mock_conn, mock_cur, captured

    def test_store_rejects_empty_blob(self):
        mock_conn, _, captured = self._captured_conn()
        with pytest.raises(ValueError, match="empty"):
            _helpers.store_segmented_blob(
                mock_conn,
                target_table="artist_metadata_data",
                name="artist_metadata",
                blob=b"",
            )
        assert captured == []

    def test_store_rejects_invalid_identifiers(self):
        mock_conn, _, _ = self._captured_conn()
        with pytest.raises(ValueError):
            _helpers.store_segmented_blob(
                mock_conn,
                target_table="bad;name",
                name="artist_metadata",
                blob=b"x",
            )
        with pytest.raises(ValueError):
            _helpers.store_segmented_blob(
                mock_conn,
                target_table="artist_metadata_data",
                name="bad name",
                blob=b"x",
            )

    def test_store_single_row_path(self):
        mock_conn, _, captured = self._captured_conn()
        _helpers.store_segmented_blob(
            mock_conn,
            target_table="artist_metadata_data",
            name="artist_metadata",
            blob=b"small-payload",
            max_part_size_mb=100,
        )
        assert len(captured) == 2
        delete_sql, delete_params = captured[0]
        assert "DELETE FROM artist_metadata_data" in delete_sql
        assert delete_params[0] == "artist_metadata"
        assert delete_params[1] == r"artist\_metadata\_%\_%"

        insert_sql, insert_params = captured[1]
        assert "INSERT INTO artist_metadata_data" in insert_sql
        assert "ON CONFLICT" in insert_sql
        assert insert_params[0] == "artist_metadata"

    def test_store_segmented_path(self):
        mock_conn, _, captured = self._captured_conn()
        payload = b"Y" * (3 * 1024 * 1024 + 11)
        _helpers.store_segmented_blob(
            mock_conn,
            target_table="artist_metadata_data",
            name="artist_metadata",
            blob=payload,
            max_part_size_mb=1,
        )
        assert "DELETE FROM artist_metadata_data" in captured[0][0]
        inserts = captured[1:]
        assert len(inserts) >= 3
        n_parts = len(inserts)
        for idx, (sql, params) in enumerate(inserts, start=1):
            assert "INSERT INTO artist_metadata_data" in sql
            assert "ON CONFLICT" not in sql
            assert params[0] == f"artist_metadata_{idx}_{n_parts}"

    def test_load_returns_none_when_no_rows(self):
        mock_conn, mock_cur, _ = self._captured_conn()
        mock_cur.fetchone.return_value = None
        mock_cur.fetchall.return_value = []
        result = _helpers.load_segmented_blob(mock_conn, "artist_metadata_data", "artist_metadata")
        assert result is None

    def test_load_returns_single_row_payload(self):
        mock_conn, mock_cur, _ = self._captured_conn()
        mock_cur.fetchone.return_value = (b"hello-world",)
        result = _helpers.load_segmented_blob(mock_conn, "artist_metadata_data", "artist_metadata")
        assert result == b"hello-world"

    def test_load_reassembles_segments_in_order(self):
        mock_conn, mock_cur, _ = self._captured_conn()
        mock_cur.fetchone.return_value = None
        mock_cur.fetchall.return_value = [
            ("artist_metadata_3_3", b"-third"),
            ("artist_metadata_1_3", b"first"),
            ("artist_metadata_2_3", b"-second"),
        ]
        result = _helpers.load_segmented_blob(mock_conn, "artist_metadata_data", "artist_metadata")
        assert result == b"first-second-third"

    def test_load_raises_on_incomplete_segments(self):
        mock_conn, mock_cur, _ = self._captured_conn()
        mock_cur.fetchone.return_value = None
        mock_cur.fetchall.return_value = [
            ("artist_metadata_1_3", b"a"),
            ("artist_metadata_3_3", b"c"),
        ]
        with pytest.raises(ValueError, match="Incomplete"):
            _helpers.load_segmented_blob(mock_conn, "artist_metadata_data", "artist_metadata")

    def test_load_raises_on_segment_total_mismatch(self):
        mock_conn, mock_cur, _ = self._captured_conn()
        mock_cur.fetchone.return_value = None
        mock_cur.fetchall.return_value = [
            ("artist_metadata_1_2", b"a"),
            ("artist_metadata_2_3", b"b"),
        ]
        with pytest.raises(ValueError, match="mismatch"):
            _helpers.load_segmented_blob(mock_conn, "artist_metadata_data", "artist_metadata")
