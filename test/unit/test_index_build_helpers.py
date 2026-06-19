# test/unit/test_index_build_helpers.py
"""
Unit tests for tasks/index_build_helpers.py

Covers the centralized helpers used by every IVF index builder:
- stream_embeddings_to_buffer: side-connection streaming into a pre-allocated
  numpy buffer; identifier validation; NULL/wrong-dim skipping; buffer
  growth when the COUNT hint under-estimates due to concurrent writes.
- store_ivf_index_segmented: single-row vs segmented persistence,
  identifier validation, empty-bytes guard.
- build_id_map / _split_bytes / _resolve_ivf_space / _validate_sql_identifier.

The helper module is loaded via importlib so this file does not pull in
tasks/__init__.py (which imports librosa).
"""

import importlib.util
import json
import os
import re
import sys
import types

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _load_helpers():
    """Load tasks.index_build_helpers without going through tasks/__init__.py."""
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


class TestSplitText:
    def test_empty_returns_single_empty(self):
        assert _helpers._split_text("", 16) == [""]

    def test_small_returns_single(self):
        assert _helpers._split_text("hello", 64) == ["hello"]

    def test_large_splits_and_roundtrips(self):
        s = "x" * 1000
        parts = _helpers._split_text(s, 100)
        assert len(parts) > 1
        assert "".join(parts) == s

    def test_each_fragment_within_byte_bound(self):
        s = "é" * 1000
        max_bytes = 100
        parts = _helpers._split_text(s, max_bytes)
        assert "".join(parts) == s
        for p in parts:
            assert len(p.encode("utf-8")) <= max_bytes


class TestReassembleSegmentedIdMap:
    def test_single_fragment_legacy_layout(self):
        frags = [(1, '{"0": "a", "1": "b"}'), (2, ""), (3, "")]
        assert _helpers.reassemble_segmented_id_map(frags) == '{"0": "a", "1": "b"}'

    def test_multi_fragment_concatenates_in_part_order(self):
        full = '{"0": "aaaa", "1": "bbbb"}'
        frags = [(2, full[10:]), (1, full[:10])]
        assert _helpers.reassemble_segmented_id_map(frags) == full

    def test_handles_none_fragments(self):
        frags = [(1, '{"0": "a"}'), (2, None)]
        assert _helpers.reassemble_segmented_id_map(frags) == '{"0": "a"}'


class TestRewriteSegmentedIdMap:
    """rewrite_segmented_id_map reassembles, rewrites, then re-splits in place."""

    class _FakeCursor:
        """Minimal psycopg2-cursor stand-in backed by an ``{index_name: json}`` dict."""

        def __init__(self, store):
            self.store = store
            self._result = []

        def execute(self, sql, params=None):
            s = " ".join(sql.split())
            if s.startswith("SELECT id_map_json FROM") and "WHERE index_name = %s" in s:
                key = params[0]
                self._result = [(self.store[key],)] if key in self.store else []
            elif s.startswith("SELECT index_name, id_map_json FROM") and "LIKE" in s:
                self._result = [
                    (k, v) for k, v in self.store.items()
                    if re.match(r".*_\d+_\d+$", k)
                ]
            elif s.startswith("UPDATE") and "SET id_map_json = %s WHERE index_name = %s" in s:
                self.store[params[1]] = params[0]
                self._result = []
            else:
                raise AssertionError(f"unexpected SQL: {s}")

        def fetchone(self):
            return self._result[0] if self._result else None

        def fetchall(self):
            return list(self._result)

    @staticmethod
    def _dict_rewriter(mapping):
        def _fn(js):
            d = json.loads(js)
            return json.dumps({k: mapping[v] for k, v in d.items() if v in mapping})
        return _fn

    def test_single_row_rewrite(self):
        store = {"ivf_main": json.dumps({"0": "a", "1": "b"})}
        cur = self._FakeCursor(store)
        changed = _helpers.rewrite_segmented_id_map(
            cur, "voyager_index_data", "ivf_main",
            self._dict_rewriter({"a": "A", "b": "B"}), max_part_size_mb=50,
        )
        assert changed is True
        assert json.loads(store["ivf_main"]) == {"0": "A", "1": "B"}

    def test_segmented_reassemble_rewrite_resplit(self):
        full = json.dumps({str(i): c for i, c in enumerate("abcdef")})
        step = max(1, -(-len(full) // 3))
        frags = [full[i:i + step] for i in range(0, len(full), step)]
        while len(frags) < 3:
            frags.append("")
        store = {f"ivf_main_{k}_3": frags[k - 1] for k in range(1, 4)}
        assert "".join(store[f"ivf_main_{k}_3"] for k in range(1, 4)) == full
        assert sum(1 for f in frags if f) >= 2, "fixture must actually be segmented"

        cur = self._FakeCursor(store)
        mapping = {c: c.upper() for c in "abcdef"}
        changed = _helpers.rewrite_segmented_id_map(
            cur, "voyager_index_data", "ivf_main",
            self._dict_rewriter(mapping), max_part_size_mb=50,
        )
        assert changed is True

        keys = [k for k in store if re.match(r".*_\d+_\d+$", k)]
        assert len(keys) == 3, "row count must be preserved"
        reassembled = _helpers.reassemble_segmented_id_map(
            (int(k.split("_")[-2]), store[k]) for k in keys
        )
        assert json.loads(reassembled) == {str(i): c.upper() for i, c in enumerate("abcdef")}

    def test_no_op_when_rewrite_returns_unchanged(self):
        full = json.dumps({"0": "a"})
        store = {"ivf_main": full}
        cur = self._FakeCursor(store)
        changed = _helpers.rewrite_segmented_id_map(
            cur, "voyager_index_data", "ivf_main",
            lambda js: js, max_part_size_mb=50,
        )
        assert changed is False
        assert store["ivf_main"] == full

    def test_raises_when_rewrite_needs_more_rows_than_exist(self):
        store = {"ivf_main_1_1": json.dumps({"0": "a", "1": "b"})}
        cur = self._FakeCursor(store)
        with pytest.raises(ValueError, match="rebuild"):
            _helpers.rewrite_segmented_id_map(
                cur, "voyager_index_data", "ivf_main",
                self._dict_rewriter({"a": "AAAA", "b": "BBBB"}),
                max_part_size_mb=0,
            )


class TestStoreIVFIndexSegmented:
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
        _helpers.store_ivf_index_segmented(
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
        _helpers.store_ivf_index_segmented(
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
            _helpers.store_ivf_index_segmented(
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
            _helpers.store_ivf_index_segmented(
                mock_conn,
                target_table="lyrics_index_data; DROP TABLE x",
                index_name="lyrics_index",
                index_bytes=b"x",
                id_map={},
                embedding_dimension=768,
            )
        with pytest.raises(ValueError):
            _helpers.store_ivf_index_segmented(
                mock_conn,
                target_table="lyrics_index_data",
                index_name="weird name",
                index_bytes=b"x",
                id_map={},
                embedding_dimension=768,
            )

    def test_large_id_map_is_split_across_parts_and_reassembles(self):
        captured = []
        mock_conn, _ = self._mock_conn(captured)
        id_map = {i: "v" * 40 for i in range(40000)}
        id_map_json = json.dumps(id_map)
        max_part_size = 1 * 1024 * 1024
        assert len(id_map_json.encode("utf-8")) > max_part_size

        _helpers.store_ivf_index_segmented(
            mock_conn,
            target_table="map_projection_data",
            index_name="main_map",
            index_bytes=b"x" * 10,
            id_map=id_map,
            embedding_dimension=2,
            max_part_size_mb=1,
            binary_column="projection_data",
        )

        assert "DELETE FROM map_projection_data" in captured[0][0]
        inserts = captured[1:]
        assert len(inserts) >= 2

        frags = []
        for sql, params in inserts:
            assert "INSERT INTO map_projection_data" in sql
            assert "projection_data" in sql and "id_map_json" in sql
            m = re.match(r"^main_map_(\d+)_(\d+)$", params[0])
            assert m is not None
            assert len(params[2].encode("utf-8")) <= max_part_size
            frags.append((int(m.group(1)), params[2]))

        assert _helpers.reassemble_segmented_id_map(frags) == id_map_json

    def test_binary_column_defaults_to_index_data(self):
        captured = []
        mock_conn, _ = self._mock_conn(captured)
        _helpers.store_ivf_index_segmented(
            mock_conn,
            target_table="voyager_index_data",
            index_name="music_library",
            index_bytes=b"small",
            id_map={0: "a"},
            embedding_dimension=256,
            max_part_size_mb=100,
        )
        assert "(index_name, index_data, id_map_json" in captured[1][0]


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

