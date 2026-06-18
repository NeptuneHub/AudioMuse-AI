"""Unit tests for the disk-paged IVF format and the byte-bounded cell cache.

These exercise the pure-Python pieces (no database): the directory and cell
binary round-trips and the LRU cache's eviction invariant.
"""
import os
import sys
import threading
import time

import numpy as np
import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tasks.paged_ivf import (
    pack_directory,
    unpack_directory,
    pack_cell,
    unpack_cell,
    _CellLruCache,
    _GlobalCellCache,
)


def test_directory_round_trip():
    dim = 16
    nlist = 5
    n_items = 23
    centroids = np.random.randn(nlist, dim).astype(np.float32)
    id2cell = np.random.randint(0, nlist, size=n_items).astype(np.uint32)
    item_ids = [f"item-{i}-é" for i in range(n_items)]

    blob = pack_directory(centroids, id2cell, item_ids, dim, "angular")
    c2, id2cell2, ids2, dim2, metric2, normalized2 = unpack_directory(blob)

    assert dim2 == dim
    assert metric2 == "angular"
    assert normalized2 is False
    assert ids2 == item_ids
    np.testing.assert_array_equal(id2cell2, id2cell)
    np.testing.assert_allclose(c2, centroids, rtol=0, atol=0)


def test_directory_normalized_flag_round_trip():
    dim = 6
    nlist = 3
    n_items = 4
    centroids = np.random.randn(nlist, dim).astype(np.float32)
    id2cell = np.zeros(n_items, dtype=np.uint32)
    item_ids = [f"id-{i}" for i in range(n_items)]

    blob = pack_directory(centroids, id2cell, item_ids, dim, "angular", normalized=True)
    _c, _i, _ids, _dim, metric, normalized = unpack_directory(blob)
    assert metric == "angular"
    assert normalized is True

    blob_default = pack_directory(centroids, id2cell, item_ids, dim, "angular")
    *_rest, normalized_default = unpack_directory(blob_default)
    assert normalized_default is False


def test_cell_round_trip():
    dim = 8
    n = 7
    ids = np.array([3, 1, 9, 4, 2, 8, 0], dtype=np.int32)
    vecs = np.random.randn(n, dim).astype(np.float32)
    blob = pack_cell(ids, vecs)
    ids2, vecs2 = unpack_cell(blob, dim)
    np.testing.assert_array_equal(ids2, ids)
    np.testing.assert_allclose(vecs2, vecs, rtol=0, atol=0)


def test_cell_cache_byte_bound_holds():
    dim = 10
    record = 4 + dim * 4
    cap = 5 * record
    cache = _CellLruCache(record, cap)

    for cell_id in range(50):
        n = 3
        ids = np.arange(cell_id * 10, cell_id * 10 + n, dtype=np.int32)
        vecs = np.random.randn(n, dim).astype(np.float32)
        cache.add_cell(cell_id, ids, vecs)
        assert cache.resident_bytes() <= cap

    assert cache.resident_bytes() <= cap


def test_cell_cache_vector_lookup_and_eviction():
    dim = 4
    record = 4 + dim * 4
    cache = _CellLruCache(record, 100 * record)
    ids = np.array([10, 11, 12], dtype=np.int32)
    vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    cache.add_cell(7, ids, vecs)
    got = cache.vector_for(11)
    assert got is not None
    np.testing.assert_array_equal(got, vecs[1])
    assert cache.vector_for(999) is None


def _mk_cell(cell_id, n, dim):
    ids = np.arange(cell_id * 100, cell_id * 100 + n, dtype=np.int32)
    vecs = np.random.randn(n, dim).astype(np.float32)
    return ids, vecs


def test_global_cache_byte_bound_across_indexes():
    dim = 10
    one = 4 + dim * 4
    cap = 5 * 3 * one
    cache = _GlobalCellCache(cap)

    for cell_id in range(60):
        for index_name in ("idx_a", "idx_b"):
            ids, vecs = _mk_cell(cell_id, 3, dim)
            cache.put_cell(index_name, cell_id, ids, vecs)
            assert cache.resident_bytes() <= cap

    assert cache.resident_bytes() <= cap


def test_global_cache_invalidate_only_target_index():
    dim = 4
    cache = _GlobalCellCache(10_000_000)
    for cell_id in range(5):
        ids, vecs = _mk_cell(cell_id, 3, dim)
        cache.put_cell("keep", cell_id, ids, vecs)
        ids, vecs = _mk_cell(cell_id, 3, dim)
        cache.put_cell("drop", cell_id, ids, vecs)

    bytes_before = cache.resident_bytes()
    assert bytes_before > 0
    cache.invalidate_index("drop")

    assert cache.get_cell("drop", 0) is None
    assert cache.get_cell("keep", 0) is not None
    assert cache.resident_bytes() == bytes_before // 2


def test_global_cache_disabled_is_noop():
    cache = _GlobalCellCache(0)
    ids, vecs = _mk_cell(1, 3, 4)
    cache.put_cell("x", 1, ids, vecs)
    assert cache.get_cell("x", 1) is None
    assert cache.resident_bytes() == 0
    assert cache.enabled is False


def test_global_cache_thread_safe_invariant():
    dim = 8
    one = 4 + dim * 4
    cap = 50 * 4 * one
    cache = _GlobalCellCache(cap)
    errors = []

    def worker(index_name):
        try:
            for cell_id in range(200):
                ids, vecs = _mk_cell(cell_id, 4, dim)
                cache.put_cell(index_name, cell_id, ids, vecs)
                cache.get_cell(index_name, cell_id % 50)
                assert cache.resident_bytes() <= cap
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(f"idx{t}",)) for t in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread errors: {errors}"
    assert cache.resident_bytes() <= cap


def test_global_cache_idle_drop():
    cache = _GlobalCellCache(10_000_000, idle_seconds=1)
    ids, vecs = _mk_cell(1, 3, 4)
    cache.put_cell("idle", 1, ids, vecs)
    assert cache.resident_bytes() > 0

    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline and cache.resident_bytes() > 0:
        time.sleep(0.1)

    assert cache.resident_bytes() == 0, "idle cache should have been dropped"
    assert cache.get_cell("idle", 1) is None


def test_global_cache_no_idle_drop_when_disabled():
    cache = _GlobalCellCache(10_000_000, idle_seconds=0)
    ids, vecs = _mk_cell(1, 3, 4)
    cache.put_cell("keep", 1, ids, vecs)
    time.sleep(0.3)
    assert cache.resident_bytes() > 0
    assert cache._timer_thread is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
