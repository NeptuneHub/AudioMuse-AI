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
    _vec_in_cell,
    _bounded_cell_groups,
    _split_cells_over_cap,
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
    entry = cache.get_cell(7)
    assert entry is not None
    got = _vec_in_cell(entry[0], entry[1], 11)
    assert got is not None
    np.testing.assert_array_equal(got, vecs[1])
    assert _vec_in_cell(entry[0], entry[1], 999) is None
    assert cache.get_cell(999) is None


def test_cell_groups_groups_items_by_cell_in_memory():
    from tasks.paged_ivf import PagedIvfIndex

    idx = object.__new__(PagedIvfIndex)
    idx._n_items = 5
    idx._num_cells = 3
    idx._id2cell = np.array([0, 0, 1, 2, 1], dtype=np.uint32)
    idx._centroids = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)

    groups = idx.cell_groups([0, 1, 2, 3, 4])

    assert sorted(count for _, count in groups) == [1, 2, 2]
    assert groups[0][1] == 2

    out_of_range = idx.cell_groups([3, 99, -1])
    assert len(out_of_range) == 1
    assert out_of_range[0][1] == 1
    np.testing.assert_array_equal(out_of_range[0][0], np.array([2.0, 0.0], dtype=np.float32))


def test_bounded_cell_groups_keeps_small_cell_whole():
    members = np.arange(50, dtype=np.int32)
    vecs = np.random.randn(50, 8).astype(np.float32)
    base = vecs.mean(axis=0)
    groups = _bounded_cell_groups(members, vecs, base, 100)
    assert len(groups) == 1
    np.testing.assert_array_equal(groups[0][0], members)
    np.testing.assert_array_equal(groups[0][1], base)


def test_bounded_cell_groups_splits_identical_vectors_under_cap():
    n = 5000
    dim = 16
    members = np.arange(n, dtype=np.int32)
    vecs = np.ones((n, dim), dtype=np.float32)
    max_records = 500
    groups = _bounded_cell_groups(members, vecs, vecs[0], max_records)

    assert all(g.shape[0] <= max_records for g, _ in groups)
    all_idx = np.concatenate([g for g, _ in groups])
    np.testing.assert_array_equal(np.sort(all_idx), members)
    assert all_idx.shape[0] == n


def test_bounded_cell_groups_splits_distinct_vectors_under_cap():
    rng = np.random.default_rng(0)
    n = 5000
    dim = 16
    members = np.arange(n, dtype=np.int32)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    max_records = 500
    groups = _bounded_cell_groups(members, vecs, vecs.mean(axis=0), max_records)

    assert all(g.shape[0] <= max_records for g, _ in groups)
    all_idx = np.concatenate([g for g, _ in groups])
    np.testing.assert_array_equal(np.sort(all_idx), members)


def test_split_cells_over_cap_noop_when_under():
    dim = 8
    record = 4 + dim * 4
    centroids = np.random.randn(2, dim).astype(np.float32)
    id2cell = np.array([0, 0, 1], dtype=np.uint32)
    cells = [
        (0, np.array([0, 1], dtype=np.int32), np.random.randn(2, dim).astype(np.float32)),
        (1, np.array([2], dtype=np.int32), np.random.randn(1, dim).astype(np.float32)),
    ]
    out_c, out_id2cell, out_cells = _split_cells_over_cap(centroids, id2cell, cells, dim, 100 * record)
    assert out_c.shape[0] == 2
    assert len(out_cells) == 2
    np.testing.assert_array_equal(out_id2cell, id2cell)


def test_split_cells_over_cap_splits_and_stays_under_cap():
    dim = 8
    record = 4 + dim * 4
    cap_records = 1000
    cap_bytes = cap_records * record
    n = 3500
    ids = np.arange(n, dtype=np.int32)
    vecs = np.random.randn(n, dim).astype(np.float32)
    centroids = np.random.randn(1, dim).astype(np.float32)
    id2cell = np.zeros(n, dtype=np.uint32)

    out_c, out_id2cell, out_cells = _split_cells_over_cap(centroids, id2cell, [(0, ids, vecs)], dim, cap_bytes)

    assert all(c.shape[0] <= cap_records for _cid, c, _v in out_cells)
    assert all(c.shape[0] * record <= cap_bytes for _cid, c, _v in out_cells)
    assert out_c.shape[0] == len(out_cells)

    seen = np.concatenate([c for _cid, c, _v in out_cells])
    np.testing.assert_array_equal(np.sort(seen), ids)
    for cid, c, _v in out_cells:
        for i in c:
            assert int(out_id2cell[int(i)]) == cid


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


import mmap as _mmap_mod


def _vmrss_kb():
    """Resident set size in KB from /proc/self/status, or None off Linux."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


@pytest.mark.skipif(
    not hasattr(_mmap_mod, "MADV_DONTNEED") or _vmrss_kb() is None,
    reason="MADV_DONTNEED / VmRSS only available on Linux",
)
def test_drop_resident_mmap_pages_reduces_rss(tmp_path):
    """MADV_DONTNEED on a live index's mapping must actually shrink process RSS.

    This is the real fix for idle RAM not returning to baseline: the disk-cache
    mmap pages faulted in during queries are file-backed (malloc_trim cannot free
    them), so the idle watcher advises them away. We map a real file, fault every
    page in, then assert the drop is reflected in VmRSS.
    """
    import tasks.paged_ivf as pv

    size = 96 * 1024 * 1024
    path = tmp_path / "cells.bin"
    with open(path, "wb") as f:
        block = (b"\xab" * (4 * 1024 * 1024))
        for _ in range(size // len(block)):
            f.write(block)

    mm = np.memmap(str(path), dtype=np.uint8, mode="r")
    _ = int(mm.sum(dtype=np.int64))  # touch every page -> resident

    class _Stub:
        pass

    stub = _Stub()
    stub._mmap = mm
    pv._LIVE_INDEXES.add(stub)
    try:
        rss_before = _vmrss_kb()
        dropped = pv._drop_resident_mmap_pages()
        rss_after = _vmrss_kb()
    finally:
        try:
            pv._LIVE_INDEXES.discard(stub)
        except Exception:
            pass

    assert dropped >= 1
    assert (rss_before - rss_after) > 30_000, (
        f"expected >30MB RSS drop, got {rss_before - rss_after} KB"
    )


def test_drop_pages_windows_routing(tmp_path, monkeypatch):
    """On Windows the drop routes through VirtualUnlock; ERROR_NOT_LOCKED counts as success."""
    import tasks.paged_ivf as pv

    path = tmp_path / "wcells.bin"
    with open(path, "wb") as f:
        f.write(b"\xcd" * (2 * 1024 * 1024))

    def make_stub():
        mm = np.memmap(str(path), dtype=np.uint8, mode="r")
        stub = type("S", (), {})()
        stub._mmap = mm
        return stub, mm

    monkeypatch.setattr(pv.platform, "system", lambda: "Windows")

    # FALSE + ERROR_NOT_LOCKED(158) is the documented success path for unlocked pages.
    stub, mm = make_stub()
    calls = []

    def unlock_not_locked(addr, size):
        calls.append((addr, size))
        return 0

    monkeypatch.setattr(pv, "_win_virtual_unlock", lambda: (unlock_not_locked, lambda: pv._WIN_ERROR_NOT_LOCKED))
    assert pv._drop_resident_mmap_pages([stub]) == 1
    assert calls == [(int(mm.ctypes.data), int(mm.nbytes))]

    # A genuine failure (other error code, FALSE) must not be counted as dropped.
    stub2, _mm2 = make_stub()
    monkeypatch.setattr(pv, "_win_virtual_unlock", lambda: ((lambda addr, size: 0), lambda: 5))
    assert pv._drop_resident_mmap_pages([stub2]) == 0


def test_mmap_idle_worker_drops_pages_and_runs_callbacks(monkeypatch):
    """An idle disk-cache working set must fire the watcher: drop + idle callbacks."""
    import tasks.paged_ivf as pv

    monkeypatch.setattr(pv.config, "IVF_DISK_CACHE_IDLE_SECONDS", 1)
    fired = {"n": 0}

    def cb():
        fired["n"] += 1

    pv.register_idle_callback(cb)
    try:
        pv._note_mmap_activity()
        assert pv._MMAP_IDLE_THREAD is not None

        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline and fired["n"] == 0:
            time.sleep(0.1)

        assert fired["n"] >= 1, "idle watcher should have run idle callbacks"
        assert pv._MMAP_IDLE_THREAD is None, "watcher should exit after firing"
    finally:
        pv.unregister_idle_callback(cb)


def test_note_mmap_activity_noop_when_disabled(monkeypatch):
    import tasks.paged_ivf as pv

    monkeypatch.setattr(pv.config, "IVF_DISK_CACHE_IDLE_SECONDS", 0)
    monkeypatch.setattr(pv, "_MMAP_IDLE_THREAD", None)
    pv._note_mmap_activity()
    assert pv._MMAP_IDLE_THREAD is None


def test_idle_watcher_drops_only_the_idle_index(monkeypatch):
    """A continuously-queried index must not keep an idle index's pages resident."""
    import tasks.paged_ivf as pv

    monkeypatch.setattr(pv.config, "IVF_DISK_CACHE_IDLE_SECONDS", 2)

    captured = []

    def fake_drop(indexes=None):
        batch = list(indexes) if indexes is not None else None
        captured.append(batch)
        return len(batch) if batch else 0

    monkeypatch.setattr(pv, "_drop_resident_mmap_pages", fake_drop)

    class _Stub:
        def __init__(self):
            self._mmap = object()
            self._mmap_pages_dropped = False
            self._last_mmap_access = time.monotonic()

    hot = _Stub()
    idle = _Stub()
    idle._last_mmap_access = time.monotonic() - 100  # already well past the window
    pv._LIVE_INDEXES.add(hot)
    pv._LIVE_INDEXES.add(idle)
    try:
        pv._note_mmap_activity(hot)  # hot is fresh; starts the watcher
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not captured:
            time.sleep(0.1)
        assert captured, "watcher never dropped anything"
        assert captured[0] == [idle], "first drop must target only the idle index"
        assert idle._mmap_pages_dropped is True
    finally:
        pv._LIVE_INDEXES.discard(hot)
        pv._LIVE_INDEXES.discard(idle)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
