"""
Unit tests for tasks/ivf_manager.py batched fetch helpers.

Covers:
- _fetch_in_batches: empty input, single batch, multi-batch merge, and the
  sequential-execution regression guard (batches must run on the calling thread,
  not fanned out across a thread pool that would share one psycopg2 connection).
- _fetch_details_map: builds an id->details map from a mock cursor.
- SCORE_DETAIL_COLUMNS constant value.
"""

import os
import sys
import threading

from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import make_dict_row, make_mock_connection


# =============================================================================
# SCORE_DETAIL_COLUMNS CONSTANT
# =============================================================================

class TestScoreDetailColumns:

    def test_constant_value(self):
        from tasks.ivf_manager import SCORE_DETAIL_COLUMNS

        assert SCORE_DETAIL_COLUMNS == 'title, author'


# =============================================================================
# _fetch_in_batches
# =============================================================================

class TestFetchInBatches:

    def test_empty_item_ids_returns_empty_dict_no_call(self):
        """Empty item_ids should return an empty dict and never call fetch fn."""
        from tasks.ivf_manager import _fetch_in_batches

        fetch_fn = MagicMock(return_value={'unexpected': 1})

        result = _fetch_in_batches([], fetch_fn)

        assert result == {}
        fetch_fn.assert_not_called()

    def test_single_batch_calls_fetch_once(self):
        """<= BATCH_SIZE_DB_OPS ids -> one call, result passed through."""
        from tasks.ivf_manager import _fetch_in_batches, BATCH_SIZE_DB_OPS

        ids = [f'id-{i}' for i in range(BATCH_SIZE_DB_OPS)]
        expected = {i: {'v': i} for i in ids}
        fetch_fn = MagicMock(return_value=expected)

        result = _fetch_in_batches(ids, fetch_fn)

        assert result == expected
        assert fetch_fn.call_count == 1
        # The single call received the full list of ids.
        called_batch = fetch_fn.call_args[0][0]
        assert called_batch == ids

    def test_multiple_batches_split_and_merge(self):
        """250 ids with batch size 100 -> 3 chunks, merged into one dict."""
        from tasks.ivf_manager import _fetch_in_batches, BATCH_SIZE_DB_OPS

        assert BATCH_SIZE_DB_OPS == 100
        n = 250
        ids = [f'id-{i}' for i in range(n)]

        batches_seen = []

        def fetch_fn(batch):
            batches_seen.append(list(batch))
            return {item: {'idx': item} for item in batch}

        result = _fetch_in_batches(ids, fetch_fn)

        # ceil(250 / 100) == 3 chunks.
        assert len(batches_seen) == 3
        assert [len(b) for b in batches_seen] == [100, 100, 50]

        # Every id present exactly once in the merged result.
        assert len(result) == n
        for item in ids:
            assert result[item] == {'idx': item}

        # Chunks are contiguous and cover the whole input in order.
        rejoined = [item for batch in batches_seen for item in batch]
        assert rejoined == ids

    def test_batches_run_sequentially_on_calling_thread(self):
        """REGRESSION GUARD: batches must run on the calling thread, not a pool.

        The old code fanned batches out via a ThreadPoolExecutor; because the
        batches share one psycopg2 connection (not concurrency-safe) the fix
        made execution strictly sequential on the caller's thread.
        """
        from tasks.ivf_manager import _fetch_in_batches

        n = 250
        ids = [f'id-{i}' for i in range(n)]
        test_ident = threading.get_ident()
        idents_seen = []

        def fetch_fn(batch):
            idents_seen.append(threading.get_ident())
            return {item: 1 for item in batch}

        _fetch_in_batches(ids, fetch_fn)

        # More than one batch ran (so a pool would have shown other idents).
        assert len(idents_seen) == 3
        assert all(ident == test_ident for ident in idents_seen)

    def test_later_batch_keys_override_earlier(self):
        """merged.update semantics: identical keys in a later batch win."""
        from tasks.ivf_manager import _fetch_in_batches

        ids = [f'id-{i}' for i in range(150)]

        def fetch_fn(batch):
            # Every batch claims the same shared key; later call must win.
            out = {item: 'own' for item in batch}
            out['shared'] = batch[0]
            return out

        result = _fetch_in_batches(ids, fetch_fn)

        # Second (final) batch starts at id-100.
        assert result['shared'] == 'id-100'


# =============================================================================
# _fetch_details_map
# =============================================================================

class TestFetchDetailsMap:

    def _make_conn(self, rows):
        cursor = MagicMock()
        cursor.fetchall.return_value = rows
        cursor.__enter__.return_value = cursor
        cursor.__exit__.return_value = False
        conn = make_mock_connection(cursor)
        # cursor() is used as a context manager: with conn.cursor(...) as cur
        conn.cursor.return_value.__enter__.return_value = cursor
        conn.cursor.return_value.__exit__.return_value = False
        return conn, cursor

    def test_builds_id_to_details_map(self):
        from tasks.ivf_manager import _fetch_details_map

        rows = [
            make_dict_row({'item_id': 'a', 'title': 'Song A', 'author': 'Artist A'}),
            make_dict_row({'item_id': 'b', 'title': 'Song B', 'author': 'Artist B'}),
        ]
        conn, cursor = self._make_conn(rows)

        result = _fetch_details_map(conn, ['a', 'b'], 'title, author')

        assert result == {
            'a': {'title': 'Song A', 'author': 'Artist A'},
            'b': {'title': 'Song B', 'author': 'Artist B'},
        }
        cursor.execute.assert_called_once()
        query, params = cursor.execute.call_args[0]
        assert 'SELECT item_id, title, author FROM score' in query
        assert 'item_id = ANY(%s)' in query
        assert params == (['a', 'b'],)

    def test_respects_passed_column_list(self):
        """Only the columns named in the list should appear in each detail dict."""
        from tasks.ivf_manager import _fetch_details_map

        rows = [
            make_dict_row({'item_id': 'x', 'title': 'T', 'author': 'X'}),
        ]
        conn, cursor = self._make_conn(rows)

        result = _fetch_details_map(conn, ['x'], 'title')

        assert result == {'x': {'title': 'T'}}
        assert 'author' not in result['x']

    def test_empty_item_ids_no_query(self):
        """No ids -> no batch, so the cursor is never opened or executed."""
        from tasks.ivf_manager import _fetch_details_map

        conn, cursor = self._make_conn([])

        result = _fetch_details_map(conn, [], 'title, author')

        assert result == {}
        cursor.execute.assert_not_called()
