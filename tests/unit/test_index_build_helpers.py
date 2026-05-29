# tests/unit/test_index_build_helpers.py
"""
Unit tests for tasks/index_build_helpers.py

Covers the centralized helpers used by every Voyager index builder:
- stream_embeddings_to_buffer: side-connection streaming into a pre-allocated
  numpy buffer; identifier validation; NULL/wrong-dim skipping; buffer
  growth when the COUNT hint under-estimates due to concurrent writes.
- build_voyager_index_bytes: rejects empty/wrong-shape buffers, coerces
  non-float32 input, round-trips through voyager.Index.load.
- store_voyager_index_segmented: single-row vs segmented persistence,
  identifier validation, empty-bytes guard.
- build_id_map / _split_bytes / _resolve_voyager_space / _validate_sql_identifier.

The helper module is loaded via importlib so this file does not pull in
tasks/__init__.py (which imports librosa).
"""

import importlib.util
import os
import sys
import types

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _load_helpers():
    """Load tasks.index_build_helpers without going through tasks/__init__.py."""
    if 'tasks' not in sys.modules:
        stub = types.ModuleType('tasks')
        stub.__path__ = []
        sys.modules['tasks'] = stub

    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    )
    mod_path = os.path.join(repo_root, 'tasks', 'index_build_helpers.py')
    mod_name = 'tasks.index_build_helpers'
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name]


_helpers = _load_helpers()


class TestValidateSqlIdentifier:
    """Strict identifier guard for table / column / cursor / index names."""

    def test_accepts_bare_identifiers(self):
        for ok in ("embedding", "lyrics_embedding", "axis_vector",
                   "music_library", "_underscore_start", "a", "A1_b2"):
            _helpers._validate_sql_identifier(ok, "table")

    def test_rejects_bad_inputs(self):
        bad_values = [
            "",
            "1starts_with_digit",
            "has-dash",
            "has space",
            "has.dot",
            "drop;drop",
            "quote'tail",
            'double"quote',
            "backtick`tail",
            "newline\nthere",
            "tab\tin",
            None,
            123,
            ["list"],
            {"dict": 1},
        ]
        for bad in bad_values:
            with pytest.raises(ValueError):
                _helpers._validate_sql_identifier(bad, "table")


class TestBuildIdMap:
    def test_empty(self):
        assert _helpers.build_id_map([]) == {}

    def test_preserves_order_and_keys(self):
        ids = ["song-c", "song-a", "song-b"]
        assert _helpers.build_id_map(ids) == {0: "song-c", 1: "song-a", 2: "song-b"}

    def test_accepts_iterable_not_just_list(self):
        gen = (f"id-{i}" for i in range(3))
        assert _helpers.build_id_map(gen) == {0: "id-0", 1: "id-1", 2: "id-2"}


class TestSplitBytes:
    def test_exact_multiple(self):
        assert _helpers._split_bytes(b"abcdef", 2) == [b"ab", b"cd", b"ef"]

    def test_with_remainder(self):
        assert _helpers._split_bytes(b"abcdefg", 3) == [b"abc", b"def", b"g"]

    def test_empty(self):
        assert _helpers._split_bytes(b"", 4) == []

    def test_part_larger_than_input(self):
        assert _helpers._split_bytes(b"abc", 100) == [b"abc"]


class TestResolveVoyagerSpace:
    def test_known_metrics(self):
        try:
            import voyager
        except ImportError:
            pytest.skip("voyager not installed")
        assert _helpers._resolve_voyager_space("angular") == voyager.Space.Cosine
        assert _helpers._resolve_voyager_space("ANGULAR") == voyager.Space.Cosine
        assert _helpers._resolve_voyager_space("euclidean") == voyager.Space.Euclidean
        assert _helpers._resolve_voyager_space("dot") == voyager.Space.InnerProduct

    def test_unknown_defaults_to_cosine(self):
        try:
            import voyager
        except ImportError:
            pytest.skip("voyager not installed")
        assert _helpers._resolve_voyager_space("nonsense") == voyager.Space.Cosine

    def test_none_defaults_to_angular(self):
        try:
            import voyager
        except ImportError:
            pytest.skip("voyager not installed")
        assert _helpers._resolve_voyager_space(None) == voyager.Space.Cosine


class TestBuildVoyagerIndexBytes:
    def test_rejects_empty_buffer(self):
        try:
            import voyager  # noqa: F401
        except ImportError:
            pytest.skip("voyager not installed")
        with pytest.raises(ValueError, match="empty"):
            _helpers.build_voyager_index_bytes(
                np.empty((0, 8), dtype=np.float32), 8,
            )

    def test_rejects_dim_mismatch(self):
        try:
            import voyager  # noqa: F401
        except ImportError:
            pytest.skip("voyager not installed")
        with pytest.raises(ValueError, match="dim"):
            _helpers.build_voyager_index_bytes(
                np.zeros((3, 7), dtype=np.float32), 8,
            )

    def test_rejects_one_dim_buffer(self):
        try:
            import voyager  # noqa: F401
        except ImportError:
            pytest.skip("voyager not installed")
        with pytest.raises(ValueError, match="2-D"):
            _helpers.build_voyager_index_bytes(
                np.zeros(8, dtype=np.float32), 8,
            )

    def test_round_trip_load(self):
        try:
            import voyager
        except ImportError:
            pytest.skip("voyager not installed")

        rng = np.random.default_rng(42)
        buf = rng.standard_normal((10, 16)).astype(np.float32)

        index_bytes = _helpers.build_voyager_index_bytes(buf, 16, metric="angular")
        assert isinstance(index_bytes, (bytes, bytearray)) and len(index_bytes) > 0

        import io
        loaded = voyager.Index.load(io.BytesIO(index_bytes))
        assert len(loaded) == 10
        neighbour_ids, _ = loaded.query(buf[3], k=1)
        assert int(neighbour_ids[0]) == 3

    def test_coerces_non_float32_input(self):
        try:
            import voyager  # noqa: F401
        except ImportError:
            pytest.skip("voyager not installed")
        buf64 = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float64)
        out = _helpers.build_voyager_index_bytes(buf64, 8, metric="angular")
        assert len(out) > 0


class TestStoreVoyagerIndexSegmented:
    def _mock_conn(self, captured):
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)

        def execute_side(sql, params=None):
            captured.append((sql, params))

        mock_cur.execute.side_effect = execute_side
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        return mock_conn, mock_cur

    def test_single_row_path_emits_delete_then_upsert(self):
        captured = []
        mock_conn, _ = self._mock_conn(captured)
        _helpers.store_voyager_index_segmented(
            mock_conn,
            target_table="clap_index_data",
            index_name="clap_index",
            index_bytes=b"small-payload",
            id_map={0: "a", 1: "b"},
            embedding_dimension=512,
            max_part_size_mb=100,
        )
        assert len(captured) == 2
        first_sql, first_params = captured[0]
        assert "DELETE FROM clap_index_data" in first_sql
        assert first_params[0] == "clap_index"
        assert first_params[1] == r"clap\_index\_%\_%"

        second_sql, second_params = captured[1]
        assert "INSERT INTO clap_index_data" in second_sql
        assert "ON CONFLICT" in second_sql
        assert second_params[0] == "clap_index"
        assert second_params[3] == 512

    def test_segmented_path_writes_n_parts_with_id_map_only_on_first(self):
        captured = []
        mock_conn, _ = self._mock_conn(captured)
        payload = b"X" * (3 * 1024 * 1024 + 17)
        _helpers.store_voyager_index_segmented(
            mock_conn,
            target_table="voyager_index_data",
            index_name="music_library",
            index_bytes=payload,
            id_map={0: "a", 1: "b"},
            embedding_dimension=256,
            max_part_size_mb=1,
        )

        assert "DELETE FROM voyager_index_data" in captured[0][0]
        inserts = captured[1:]
        assert len(inserts) >= 3
        n_parts = len(inserts)
        for idx, (sql, params) in enumerate(inserts, start=1):
            assert "INSERT INTO voyager_index_data" in sql
            assert params[0] == f"music_library_{idx}_{n_parts}"
            assert params[3] == 256
            if idx == 1:
                assert params[2] != ""
            else:
                assert params[2] == ""

    def test_rejects_empty_bytes(self):
        captured = []
        mock_conn, _ = self._mock_conn(captured)
        with pytest.raises(ValueError, match="empty"):
            _helpers.store_voyager_index_segmented(
                mock_conn,
                target_table="lyrics_index_data",
                index_name="lyrics_index",
                index_bytes=b"",
                id_map={},
                embedding_dimension=768,
            )
        assert captured == []

    def test_rejects_invalid_identifiers(self):
        captured = []
        mock_conn, _ = self._mock_conn(captured)
        with pytest.raises(ValueError):
            _helpers.store_voyager_index_segmented(
                mock_conn,
                target_table="lyrics_index_data; DROP TABLE x",
                index_name="lyrics_index",
                index_bytes=b"x",
                id_map={},
                embedding_dimension=768,
            )
        with pytest.raises(ValueError):
            _helpers.store_voyager_index_segmented(
                mock_conn,
                target_table="lyrics_index_data",
                index_name="weird name",
                index_bytes=b"x",
                id_map={},
                embedding_dimension=768,
            )


class TestStreamEmbeddingsToBuffer:
    """Mock psycopg2.connect to drive the streaming code without a real DB."""

    def _fake_conn(self, count_value, rows):
        """Return a MagicMock that mimics psycopg2 for one stream call.

        - cursor() with no args yields a context-manager cursor whose
          ``fetchone`` returns (count_value,) for the COUNT(*) query.
        - cursor(name=...) yields a context-manager iterable cursor that
          yields the supplied (item_id, blob) tuples.
        """
        count_cur = MagicMock()
        count_cur.__enter__ = MagicMock(return_value=count_cur)
        count_cur.__exit__ = MagicMock(return_value=False)
        count_cur.fetchone = MagicMock(return_value=(count_value,))

        stream_cur = MagicMock()
        stream_cur.__enter__ = MagicMock(return_value=stream_cur)
        stream_cur.__exit__ = MagicMock(return_value=False)
        stream_cur.itersize = 0
        stream_cur.execute = MagicMock()
        stream_cur.__iter__ = MagicMock(return_value=iter(rows))

        conn = MagicMock()

        def cursor_factory(*args, **kwargs):
            if "name" in kwargs:
                return stream_cur
            return count_cur

        conn.cursor.side_effect = cursor_factory
        conn.set_session = MagicMock()
        conn.close = MagicMock()
        return conn

    def test_validates_table(self):
        with pytest.raises(ValueError):
            _helpers.stream_embeddings_to_buffer("bad table", "embedding", dim=8)

    def test_validates_column(self):
        with pytest.raises(ValueError):
            _helpers.stream_embeddings_to_buffer("embedding", "bad-col", dim=8)

    def test_validates_dim(self):
        with pytest.raises(ValueError):
            _helpers.stream_embeddings_to_buffer("embedding", "embedding", dim=0)
        with pytest.raises(ValueError):
            _helpers.stream_embeddings_to_buffer("embedding", "embedding", dim=-3)

    def test_returns_empty_when_source_empty(self):
        conn = self._fake_conn(count_value=0, rows=[])
        with patch.object(_helpers.psycopg2, "connect", return_value=conn):
            buf, ids = _helpers.stream_embeddings_to_buffer(
                table="embedding", column="embedding", dim=8,
            )
        assert buf.shape == (0, 8)
        assert ids == []
        conn.close.assert_called_once()

    def test_fills_buffer_correctly(self):
        rng = np.random.default_rng(123)
        vectors = [rng.standard_normal(8).astype(np.float32) for _ in range(4)]
        rows = [(f"id-{i}", v.tobytes()) for i, v in enumerate(vectors)]
        conn = self._fake_conn(count_value=len(rows), rows=rows)
        with patch.object(_helpers.psycopg2, "connect", return_value=conn):
            buf, ids = _helpers.stream_embeddings_to_buffer(
                table="embedding", column="embedding", dim=8,
            )
        assert buf.shape == (4, 8)
        assert buf.dtype == np.float32
        assert ids == ["id-0", "id-1", "id-2", "id-3"]
        for i, expected in enumerate(vectors):
            np.testing.assert_array_equal(buf[i], expected)

    def test_skips_null_and_wrong_dim_rows(self):
        good = np.ones(8, dtype=np.float32)
        wrong_dim = np.ones(7, dtype=np.float32)
        rows = [
            ("a", good.tobytes()),
            ("b", None),
            ("c", wrong_dim.tobytes()),
            ("d", good.tobytes()),
        ]
        conn = self._fake_conn(count_value=4, rows=rows)
        with patch.object(_helpers.psycopg2, "connect", return_value=conn):
            buf, ids = _helpers.stream_embeddings_to_buffer(
                table="embedding", column="embedding", dim=8,
            )
        assert ids == ["a", "d"]
        assert buf.shape == (2, 8)

    def test_grows_buffer_when_select_yields_more_than_count_hint(self):
        rng = np.random.default_rng(7)
        vectors = [rng.standard_normal(4).astype(np.float32) for _ in range(6)]
        rows = [(f"id-{i}", v.tobytes()) for i, v in enumerate(vectors)]
        conn = self._fake_conn(count_value=3, rows=rows)
        with patch.object(_helpers.psycopg2, "connect", return_value=conn):
            buf, ids = _helpers.stream_embeddings_to_buffer(
                table="embedding", column="embedding", dim=4,
            )
        assert buf.shape == (6, 4)
        assert ids == [f"id-{i}" for i in range(6)]
        for i, expected in enumerate(vectors):
            np.testing.assert_array_equal(buf[i], expected)

    def test_closes_side_connection_on_iteration_failure(self):
        """Subclassing MagicMock + ``def __iter__`` does NOT actually
        override iteration -- MagicMock's metaclass re-binds dunder methods
        at instance construction. The documented way to force ``iter(m)``
        to raise is ``m.__iter__.side_effect = ExceptionInstance``."""
        boom_cur = MagicMock()
        boom_cur.__enter__ = MagicMock(return_value=boom_cur)
        boom_cur.__exit__ = MagicMock(return_value=False)
        boom_cur.itersize = 0
        boom_cur.execute = MagicMock()
        boom_cur.__iter__ = MagicMock(side_effect=RuntimeError("simulated stream failure"))

        count_cur = MagicMock()
        count_cur.__enter__ = MagicMock(return_value=count_cur)
        count_cur.__exit__ = MagicMock(return_value=False)
        count_cur.fetchone = MagicMock(return_value=(5,))

        conn = MagicMock()

        def cursor_factory(*args, **kwargs):
            return boom_cur if "name" in kwargs else count_cur

        conn.cursor.side_effect = cursor_factory
        conn.set_session = MagicMock()
        conn.close = MagicMock()

        with patch.object(_helpers.psycopg2, "connect", return_value=conn):
            with pytest.raises(RuntimeError, match="simulated"):
                _helpers.stream_embeddings_to_buffer(
                    table="embedding", column="embedding", dim=4,
                )
        conn.close.assert_called_once()

    def test_side_session_is_read_only_and_transactional(self):
        """The side connection must be readonly AND must NOT be autocommit.

        autocommit=True + a named cursor in psycopg2 does NOT give snapshot
        consistency: fetches can see different snapshots, defeating the
        "concurrent writes can't corrupt this read" guarantee. Stay in the
        default transactional mode so the named cursor inherits the
        implicit BEGIN's snapshot for its entire lifetime.
        """
        conn = self._fake_conn(count_value=0, rows=[])
        with patch.object(_helpers.psycopg2, "connect", return_value=conn):
            _helpers.stream_embeddings_to_buffer(
                table="embedding", column="embedding", dim=4,
            )
        conn.set_session.assert_called_once()
        _, kwargs = conn.set_session.call_args
        assert kwargs.get("readonly") is True
        assert kwargs.get("autocommit") in (None, False), (
            "side connection must NOT be autocommit -- snapshot consistency "
            "of the named cursor depends on the default transactional mode"
        )


class TestIterEmbeddingBatches:
    """Tests for the batched-streaming generator.

    Same snapshot-safe pattern as TestStreamEmbeddingsToBuffer but the
    generator does NOT issue a separate COUNT(*); only the named cursor is
    used. The mock here only needs to support the streaming cursor.
    """

    def _fake_conn(self, rows):
        stream_cur = MagicMock()
        stream_cur.__enter__ = MagicMock(return_value=stream_cur)
        stream_cur.__exit__ = MagicMock(return_value=False)
        stream_cur.itersize = 0
        stream_cur.execute = MagicMock()
        stream_cur.__iter__ = MagicMock(return_value=iter(rows))

        conn = MagicMock()
        conn.cursor = MagicMock(return_value=stream_cur)
        conn.set_session = MagicMock()
        conn.close = MagicMock()
        return conn

    def test_validates_table(self):
        with pytest.raises(ValueError):
            list(_helpers.iter_embedding_batches("bad table", "embedding", dim=8))

    def test_validates_column(self):
        with pytest.raises(ValueError):
            list(_helpers.iter_embedding_batches("embedding", "bad-col", dim=8))

    def test_validates_dim_and_batch_size(self):
        with pytest.raises(ValueError):
            list(_helpers.iter_embedding_batches("embedding", "embedding", dim=0))
        with pytest.raises(ValueError):
            list(_helpers.iter_embedding_batches("embedding", "embedding", dim=8, batch_size=0))
        with pytest.raises(ValueError):
            list(_helpers.iter_embedding_batches("embedding", "embedding", dim=8, batch_size=-5))

    def test_empty_source_yields_no_batches(self):
        conn = self._fake_conn(rows=[])
        with patch.object(_helpers.psycopg2, "connect", return_value=conn):
            batches = list(_helpers.iter_embedding_batches(
                table="embedding", column="embedding", dim=8,
            ))
        assert batches == []
        conn.close.assert_called_once()

    def test_yields_exactly_one_partial_batch_when_smaller_than_batch_size(self):
        rng = np.random.default_rng(1)
        vecs = [rng.standard_normal(4).astype(np.float32) for _ in range(3)]
        rows = [(f"id-{i}", v.tobytes()) for i, v in enumerate(vecs)]
        conn = self._fake_conn(rows=rows)
        with patch.object(_helpers.psycopg2, "connect", return_value=conn):
            batches = list(_helpers.iter_embedding_batches(
                table="embedding", column="embedding", dim=4, batch_size=10,
            ))
        assert len(batches) == 1
        buf, ids = batches[0]
        assert buf.shape == (3, 4)
        assert buf.dtype == np.float32
        assert ids == ["id-0", "id-1", "id-2"]
        for i, expected in enumerate(vecs):
            np.testing.assert_array_equal(buf[i], expected)

    def test_yields_multiple_full_batches_and_one_partial(self):
        rng = np.random.default_rng(2)
        vecs = [rng.standard_normal(4).astype(np.float32) for _ in range(7)]
        rows = [(f"id-{i}", v.tobytes()) for i, v in enumerate(vecs)]
        conn = self._fake_conn(rows=rows)
        with patch.object(_helpers.psycopg2, "connect", return_value=conn):
            batches = list(_helpers.iter_embedding_batches(
                table="embedding", column="embedding", dim=4, batch_size=3,
            ))
        assert len(batches) == 3
        assert [b[0].shape[0] for b in batches] == [3, 3, 1]
        flat_ids = [i for _, ids in batches for i in ids]
        assert flat_ids == [f"id-{i}" for i in range(7)]
        flat_vecs = np.vstack([b[0] for b in batches])
        for i, expected in enumerate(vecs):
            np.testing.assert_array_equal(flat_vecs[i], expected)

    def test_skips_null_and_wrong_dim_across_batch_boundaries(self):
        good = np.ones(4, dtype=np.float32)
        bad_dim = np.ones(3, dtype=np.float32)
        rows = [
            ("a", good.tobytes()),
            ("b", None),
            ("c", bad_dim.tobytes()),
            ("d", good.tobytes()),
            ("e", good.tobytes()),
            ("f", None),
            ("g", good.tobytes()),
        ]
        conn = self._fake_conn(rows=rows)
        with patch.object(_helpers.psycopg2, "connect", return_value=conn):
            batches = list(_helpers.iter_embedding_batches(
                table="embedding", column="embedding", dim=4, batch_size=2,
            ))
        flat_ids = [i for _, ids in batches for i in ids]
        assert flat_ids == ["a", "d", "e", "g"]
        total_rows = sum(b[0].shape[0] for b in batches)
        assert total_rows == 4

    def test_each_batch_is_a_fresh_buffer_not_a_view(self):
        """If batches shared memory, mutating one would corrupt the next."""
        rng = np.random.default_rng(3)
        vecs = [rng.standard_normal(4).astype(np.float32) for _ in range(4)]
        rows = [(f"id-{i}", v.tobytes()) for i, v in enumerate(vecs)]
        conn = self._fake_conn(rows=rows)
        with patch.object(_helpers.psycopg2, "connect", return_value=conn):
            batches = list(_helpers.iter_embedding_batches(
                table="embedding", column="embedding", dim=4, batch_size=2,
            ))
        assert len(batches) == 2
        batches[0][0].fill(99.0)
        np.testing.assert_array_equal(batches[1][0][0], vecs[2])
        np.testing.assert_array_equal(batches[1][0][1], vecs[3])

    def test_closes_connection_on_early_consumer_exit(self):
        rng = np.random.default_rng(4)
        rows = [(f"id-{i}", rng.standard_normal(4).astype(np.float32).tobytes()) for i in range(10)]
        conn = self._fake_conn(rows=rows)
        with patch.object(_helpers.psycopg2, "connect", return_value=conn):
            gen = _helpers.iter_embedding_batches(
                table="embedding", column="embedding", dim=4, batch_size=2,
            )
            next(gen)
            gen.close()
        conn.close.assert_called_once()

    def test_uses_readonly_non_autocommit_side_session(self):
        conn = self._fake_conn(rows=[])
        with patch.object(_helpers.psycopg2, "connect", return_value=conn):
            list(_helpers.iter_embedding_batches(
                table="embedding", column="embedding", dim=4,
            ))
        conn.set_session.assert_called_once()
        _, kwargs = conn.set_session.call_args
        assert kwargs.get("readonly") is True
        assert kwargs.get("autocommit") in (None, False)


class TestBuildVoyagerIndexBytesStreaming:
    """Tests for the incremental index builder."""

    def test_validates_dim(self):
        try:
            import voyager  # noqa: F401
        except ImportError:
            pytest.skip("voyager not installed")
        with pytest.raises(ValueError):
            _helpers.build_voyager_index_bytes_streaming(iter([]), dim=0)

    def test_rejects_empty_generator(self):
        try:
            import voyager  # noqa: F401
        except ImportError:
            pytest.skip("voyager not installed")
        with pytest.raises(ValueError, match="no items"):
            _helpers.build_voyager_index_bytes_streaming(iter([]), dim=8)

    def test_rejects_batch_with_wrong_dim(self):
        try:
            import voyager  # noqa: F401
        except ImportError:
            pytest.skip("voyager not installed")
        bad_batch = (np.zeros((2, 7), dtype=np.float32), ["a", "b"])
        with pytest.raises(ValueError, match="batch dim"):
            _helpers.build_voyager_index_bytes_streaming(iter([bad_batch]), dim=8)

    def test_rejects_batch_with_mismatched_ids_length(self):
        try:
            import voyager  # noqa: F401
        except ImportError:
            pytest.skip("voyager not installed")
        bad = (np.zeros((3, 4), dtype=np.float32), ["only-two", "ids"])
        with pytest.raises(ValueError, match="batch_ids len"):
            _helpers.build_voyager_index_bytes_streaming(iter([bad]), dim=4)

    def test_skips_empty_batches_silently(self):
        try:
            import voyager
        except ImportError:
            pytest.skip("voyager not installed")
        rng = np.random.default_rng(5)
        b1 = (rng.standard_normal((3, 4)).astype(np.float32), ["a", "b", "c"])
        empty = (np.empty((0, 4), dtype=np.float32), [])
        b2 = (rng.standard_normal((2, 4)).astype(np.float32), ["d", "e"])
        index_bytes, ids = _helpers.build_voyager_index_bytes_streaming(
            iter([b1, empty, b2]), dim=4, metric="angular",
        )
        assert ids == ["a", "b", "c", "d", "e"]
        assert len(index_bytes) > 0

    def test_round_trip_across_multiple_batches(self):
        try:
            import voyager
        except ImportError:
            pytest.skip("voyager not installed")
        import io as _io

        rng = np.random.default_rng(6)
        all_vecs = rng.standard_normal((12, 8)).astype(np.float32)
        batches = [
            (all_vecs[0:5].copy(), [f"id-{i}" for i in range(0, 5)]),
            (all_vecs[5:10].copy(), [f"id-{i}" for i in range(5, 10)]),
            (all_vecs[10:12].copy(), [f"id-{i}" for i in range(10, 12)]),
        ]
        index_bytes, ids = _helpers.build_voyager_index_bytes_streaming(
            iter(batches), dim=8, metric="angular",
        )
        assert ids == [f"id-{i}" for i in range(12)]

        loaded = voyager.Index.load(_io.BytesIO(index_bytes))
        assert len(loaded) == 12
        neighbour_ids, _ = loaded.query(all_vecs[7], k=1)
        assert int(neighbour_ids[0]) == 7

    def test_dense_ids_are_assigned_consecutively_across_batches(self):
        try:
            import voyager
        except ImportError:
            pytest.skip("voyager not installed")
        import io as _io

        rng = np.random.default_rng(7)
        b1_vecs = rng.standard_normal((3, 4)).astype(np.float32)
        b2_vecs = rng.standard_normal((2, 4)).astype(np.float32)
        batches = [
            (b1_vecs, ["x", "y", "z"]),
            (b2_vecs, ["w", "v"]),
        ]
        index_bytes, ids = _helpers.build_voyager_index_bytes_streaming(
            iter(batches), dim=4, metric="angular",
        )
        loaded = voyager.Index.load(_io.BytesIO(index_bytes))
        for voyager_id in range(5):
            assert voyager_id in loaded
        assert ids == ["x", "y", "z", "w", "v"]
