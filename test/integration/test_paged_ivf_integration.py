"""Real-Postgres integration tests for the disk-paged IVF backend.

Builds a paged IVF index from synthetic clustered vectors into a live database,
reloads it through the production load path, and proves the four properties the
design rests on:

  * format round-trip: every get_vector returns the exact stored float32 vector;
  * recall: IVF top-k matches a brute-force ground truth within tolerance;
  * the hard RAM bound: the per-request cell cache never exceeds its byte cap,
    even when get_vectors touches ids spread across many cells;
  * get_max_distance is exact.

Database selection mirrors test_app_endpoints_integration.py:
  * AUDIOMUSE_TEST_DATABASE_URL, or
  * an ephemeral instance via the optional pgserver package, or
  * the module is skipped.

Run locally:
    pip install pgserver
    pytest test/integration/test_paged_ivf_integration.py -m integration -s -v --tb=short
"""
import os
import sys
import tempfile

import numpy as np
import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
except Exception:
    psycopg2 = None

pytestmark = pytest.mark.integration

_DIR_DDL = "CREATE TABLE IF NOT EXISTS ivf_dir (name VARCHAR(255) PRIMARY KEY, blob_data BYTEA NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
_CELL_DDL = "CREATE TABLE IF NOT EXISTS ivf_cell (index_name VARCHAR(255) NOT NULL, cell_id INTEGER NOT NULL, cell_data BYTEA NOT NULL, PRIMARY KEY (index_name, cell_id))"


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
        pytest.skip("No test database. Set AUDIOMUSE_TEST_DATABASE_URL or pip install pgserver.")
    data_dir = tempfile.mkdtemp(prefix='audiomuse_ivf_pg_')
    server = pgserver.get_server(data_dir)
    try:
        yield server.get_uri()
    finally:
        server.cleanup()


@pytest.fixture
def ivf_db(pg_dsn):
    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS ivf_cell")
        cur.execute("DROP TABLE IF EXISTS ivf_dir")
        cur.execute(_DIR_DDL)
        cur.execute(_CELL_DDL)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def _ivf_disk_cache(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "IVF_DISK_CACHE_DIR", str(tmp_path / "ivf_cache"))
    monkeypatch.setattr(config, "IVF_DISK_CACHE_ENABLED", True)
    yield


def _make_clustered(n, dim, n_clusters, spread, seed):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    assign = rng.integers(0, n_clusters, n)
    x = centers[assign] + spread * rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


def _brute_force_topk(x, qi, k):
    q = x[qi]
    sims = x @ q
    sims[qi] = -1e9
    top = np.argpartition(-sims, k)[:k]
    return set(int(t) for t in top)


class _CountingCursor:
    def __init__(self, cur, counter):
        self._cur = cur
        self._counter = counter

    def execute(self, sql, params=None):
        if "ivf_cell" in sql and sql.lstrip().upper().startswith("SELECT"):
            self._counter[0] += 1
        return self._cur.execute(sql, params)

    def __getattr__(self, name):
        return getattr(self._cur, name)

    def __enter__(self):
        self._cur.__enter__()
        return self

    def __exit__(self, *exc):
        return self._cur.__exit__(*exc)


class _CountingConn:
    """Wraps a real connection and counts SELECTs against ivf_cell."""

    def __init__(self, conn, counter):
        self._conn = conn
        self._counter = counter

    def cursor(self, *args, **kwargs):
        return _CountingCursor(self._conn.cursor(*args, **kwargs), self._counter)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_ivf_build_load_query_recall_and_ram_bound(ivf_db, monkeypatch):
    import config
    # This test asserts the EXACT float32 round-trip and exact distances, so pin
    # storage to f32 (the no-quantization path). i8/f16 behaviour is covered by
    # test_ivf_i8_storage_recall_and_approx_roundtrip.
    monkeypatch.setattr(config, "IVF_STORAGE_DTYPE", "f32")
    from tasks import paged_ivf

    n, dim = 6000, 64
    x = _make_clustered(n, dim, n_clusters=120, spread=0.35, seed=7)
    item_ids = [f"track-{i}" for i in range(n)]

    ok = paged_ivf.build_and_store_paged_ivf(ivf_db, "audio_test", x, item_ids, dim, "angular")
    assert ok

    loaded = paged_ivf.load_paged_ivf_index(
        ivf_db, "audio_test", dim, "angular", conn_factory=lambda: ivf_db, label="audio_test",
    )
    assert loaded is not None
    index, id_map, reverse_id_map = loaded
    assert index.num_elements == n
    assert len(id_map) == n

    from tasks.paged_ivf import _normalize_rows
    x_stored = _normalize_rows(x)
    index.begin_request()
    rng = np.random.default_rng(1)
    for vid in rng.choice(n, 40, replace=False):
        v = index.get_vector(int(vid))
        assert v is not None
        np.testing.assert_allclose(v, x_stored[int(vid)], rtol=0, atol=0)

    queries = rng.choice(n, 60, replace=False)
    hit = tot = 0
    for qi in queries:
        index.begin_request()
        gt = _brute_force_topk(x, int(qi), 10)
        ids, dists = index.query(x[int(qi)], k=11)
        got = {int(i) for i in ids if int(i) != int(qi)}
        got = set(list(got)[:10])
        hit += len(gt & got)
        tot += 10
    recall = hit / tot
    assert recall >= 0.90, f"IVF recall too low: {recall}"

    cap = index._cache_bytes
    index.begin_request()
    all_ids = list(range(n))
    index.get_vectors(all_ids)
    assert index._cache().resident_bytes() <= cap


def test_ivf_i8_storage_recall_and_approx_roundtrip(ivf_db, monkeypatch):
    """int8 storage: cells are 1 byte/dim, recall holds, get_vector is approx-unit."""
    import config
    from tasks import ivf_quant as quant
    monkeypatch.setattr(config, "IVF_STORAGE_DTYPE", "i8")
    from tasks import paged_ivf

    n, dim = 6000, 64
    x = _make_clustered(n, dim, n_clusters=120, spread=0.35, seed=7)
    item_ids = [f"i8-{i}" for i in range(n)]
    assert paged_ivf.build_and_store_paged_ivf(ivf_db, "i8_test", x, item_ids, dim, "angular")

    # The stored directory must advertise int8 and cells must be 1 byte/dim.
    from tasks.paged_ivf import unpack_directory, IVF_DIR_TABLE
    from tasks.index_build_helpers import load_segmented_blob
    blob = load_segmented_blob(ivf_db, IVF_DIR_TABLE, "i8_test__ivf_dir")
    *_rest, storage_dtype = unpack_directory(bytes(blob))
    assert storage_dtype == quant.DTYPE_I8
    with ivf_db.cursor() as cur:
        cur.execute("SELECT cell_data FROM ivf_cell WHERE index_name=%s AND octet_length(cell_data) > 0 LIMIT 1", ("i8_test",))
        cell = bytes(cur.fetchone()[0])
    assert len(cell) % (4 + dim) == 0, "i8 cell record must be 4-byte id + 1 byte per dim"

    loaded = paged_ivf.load_paged_ivf_index(
        ivf_db, "i8_test", dim, "angular", conn_factory=lambda: ivf_db, label="i8_test")
    index = loaded[0]
    assert index._storage_dtype == quant.DTYPE_I8
    assert index._mmap is not None, "disk mmap should be active for the i8 index"

    from tasks.paged_ivf import _normalize_rows
    x_unit = _normalize_rows(x)
    index.begin_request()
    rng = np.random.default_rng(1)
    for vid in rng.choice(n, 30, replace=False):
        v = index.get_vector(int(vid))
        assert v is not None
        # dequantized i8 recovers the unit vector within the quantization step.
        np.testing.assert_allclose(v, x_unit[int(vid)], atol=2.0 / 127.0)

    queries = rng.choice(n, 60, replace=False)
    hit = tot = 0
    for qi in queries:
        index.begin_request()
        gt = _brute_force_topk(x, int(qi), 10)
        ids, _d = index.query(x[int(qi)], k=11)
        got = {int(i) for i in ids if int(i) != int(qi)}
        got = set(list(got)[:10])
        hit += len(gt & got)
        tot += 10
    recall = hit / tot
    assert recall >= 0.88, f"i8 IVF recall too low: {recall}"


def test_ivf_stale_storage_dtype_loads_as_none_so_it_rebuilds(ivf_db, monkeypatch):
    """An index stored in a precision the current config no longer builds must load as
    None, so the normal build path regenerates it (no mixed f32/i8 cells)."""
    import config
    monkeypatch.setattr(config, "IVF_STORAGE_DTYPE", "f32")
    from tasks import paged_ivf

    n, dim = 1500, 32
    x = _make_clustered(n, dim, n_clusters=40, spread=0.4, seed=11)
    item_ids = [f"stale-{i}" for i in range(n)]
    assert paged_ivf.build_and_store_paged_ivf(ivf_db, "stale_test", x, item_ids, dim, "angular")

    # Same config the index was built with -> loads normally.
    assert paged_ivf.load_paged_ivf_index(
        ivf_db, "stale_test", dim, "angular", conn_factory=lambda: ivf_db, label="stale_test") is not None

    # Flip the configured precision: the stored f32 index is now stale and must not load.
    monkeypatch.setattr(config, "IVF_STORAGE_DTYPE", "i8")
    assert paged_ivf.load_paged_ivf_index(
        ivf_db, "stale_test", dim, "angular", conn_factory=lambda: ivf_db, label="stale_test") is None


def test_ivf_cross_request_cell_reuse(ivf_db, monkeypatch):
    import config
    monkeypatch.setattr(config, "IVF_DISK_CACHE_ENABLED", False)
    from tasks import paged_ivf

    gcache = paged_ivf.get_global_cell_cache()
    if not gcache.enabled:
        pytest.skip("global cell cache disabled by config (IVF_GLOBAL_CACHE_MB=0)")

    n, dim = 4000, 32
    x = _make_clustered(n, dim, n_clusters=80, spread=0.4, seed=5)
    item_ids = [f"r{i}" for i in range(n)]
    assert paged_ivf.build_and_store_paged_ivf(ivf_db, "reuse_test", x, item_ids, dim, "angular")

    counter = [0]
    counting = _CountingConn(ivf_db, counter)
    loaded = paged_ivf.load_paged_ivf_index(
        ivf_db, "reuse_test", dim, "angular", conn_factory=lambda: counting, label="reuse_test",
    )
    index = loaded[0]

    qi = 123
    index.begin_request()
    index.query(x[qi], k=10)
    first = counter[0]
    assert first >= 1, "first query should read cells from Postgres"

    index.begin_request()
    index.query(x[qi], k=10)
    second = counter[0] - first
    assert second == 0, f"second identical query hit DB {second} times; expected 0 (served from L2)"


def test_ivf_max_distance_uses_l2_when_preloaded(ivf_db, monkeypatch):
    import config
    monkeypatch.setattr(config, "IVF_DISK_CACHE_ENABLED", False)
    monkeypatch.setattr(config, "IVF_STORAGE_DTYPE", "f32")  # exact 1e-4 distance check
    from tasks import paged_ivf

    gcache = paged_ivf.get_global_cell_cache()
    if not gcache.enabled:
        pytest.skip("global cell cache disabled by config (IVF_GLOBAL_CACHE_MB=0)")
    n, dim = 1500, 32
    x = _make_clustered(n, dim, n_clusters=30, spread=0.5, seed=13)
    item_ids = [f"mxl2-{i}" for i in range(n)]
    assert paged_ivf.build_and_store_paged_ivf(ivf_db, "mxl2_test", x, item_ids, dim, "angular")

    counter = [0]
    counting = _CountingConn(ivf_db, counter)
    loaded = paged_ivf.load_paged_ivf_index(
        ivf_db, "mxl2_test", dim, "angular", conn_factory=lambda: counting, label="mxl2_test",
    )
    index = loaded[0]

    index.preload_all(ivf_db)
    index.begin_request()
    counter[0] = 0
    got, far_id = index.get_max_distance(42, nprobe=0)
    assert counter[0] == 0, f"max_distance hit DB {counter[0]} times after preload; expected 0 (served from L2)"

    q = x[42]
    sims = x @ q
    sims[42] = 1e9
    expected = float(1.0 - np.clip(sims.min(), -1.0, 1.0))
    assert abs(got - expected) < 1e-4, f"max distance mismatch got={got} expected={expected}"


def test_ivf_global_cache_invalidated_on_rebuild_and_reload(ivf_db):
    from tasks import paged_ivf

    gcache = paged_ivf.get_global_cell_cache()
    if not gcache.enabled:
        pytest.skip("global cell cache disabled by config (IVF_GLOBAL_CACHE_MB=0)")

    n, dim = 1500, 16
    x = _make_clustered(n, dim, n_clusters=30, spread=0.4, seed=9)
    item_ids = [f"v{i}" for i in range(n)]
    assert paged_ivf.build_and_store_paged_ivf(ivf_db, "inval_test", x, item_ids, dim, "angular")

    sentinel_id = 10 ** 8
    sids = np.array([0, 1, 2], dtype=np.int32)
    svecs = np.zeros((3, dim), dtype=np.float32)

    gcache.put_cell("inval_test", sentinel_id, sids, svecs)
    assert gcache.get_cell("inval_test", sentinel_id) is not None
    paged_ivf.load_paged_ivf_index(ivf_db, "inval_test", dim, "angular", conn_factory=lambda: ivf_db, label="inval_test")
    assert gcache.get_cell("inval_test", sentinel_id) is None, "load_paged_ivf_index must invalidate L2 for the index"

    gcache.put_cell("inval_test", sentinel_id, sids, svecs)
    assert gcache.get_cell("inval_test", sentinel_id) is not None
    assert paged_ivf.build_and_store_paged_ivf(ivf_db, "inval_test", x, item_ids, dim, "angular")
    assert gcache.get_cell("inval_test", sentinel_id) is None, "store_paged_ivf must invalidate L2 for the index"


def test_ivf_disk_mmap_created_and_no_postgres_on_query(ivf_db):
    import glob as _glob
    import config
    from tasks import paged_ivf

    n, dim = 3000, 32
    x = _make_clustered(n, dim, n_clusters=60, spread=0.4, seed=23)
    item_ids = [f"dm-{i}" for i in range(n)]
    assert paged_ivf.build_and_store_paged_ivf(ivf_db, "diskmm_test", x, item_ids, dim, "angular")

    counter = [0]
    counting = _CountingConn(ivf_db, counter)
    loaded = paged_ivf.load_paged_ivf_index(
        ivf_db, "diskmm_test", dim, "angular", conn_factory=lambda: counting, label="diskmm_test",
    )
    index = loaded[0]
    assert index._mmap is not None, "disk mmap should be active by default"

    files = _glob.glob(os.path.join(config.IVF_DISK_CACHE_DIR, "diskmm_test.*.amivf"))
    assert len(files) == 1, f"expected one cell file, got {files}"

    index.begin_request()
    counter[0] = 0
    rng = np.random.default_rng(2)
    hit = tot = 0
    for qi in rng.choice(n, 30, replace=False):
        gt = _brute_force_topk(x, int(qi), 10)
        ids, _d = index.query(x[int(qi)], k=11)
        got = {int(i) for i in ids if int(i) != int(qi)}
        got = set(list(got)[:10])
        hit += len(gt & got)
        tot += 10
    assert hit / tot >= 0.90, f"recall too low via mmap: {hit/tot}"
    assert counter[0] == 0, f"query hit Postgres {counter[0]} times; expected 0 (served from mmap)"


def test_ivf_disk_mmap_reuse_and_prune(ivf_db):
    import glob as _glob
    import config
    from tasks import paged_ivf

    n, dim = 1500, 16
    x = _make_clustered(n, dim, n_clusters=30, spread=0.4, seed=24)
    item_ids = [f"rp-{i}" for i in range(n)]
    assert paged_ivf.build_and_store_paged_ivf(ivf_db, "reuse_disk", x, item_ids, dim, "angular")

    paged_ivf.load_paged_ivf_index(ivf_db, "reuse_disk", dim, "angular", conn_factory=lambda: ivf_db, label="reuse_disk")
    files1 = _glob.glob(os.path.join(config.IVF_DISK_CACHE_DIR, "reuse_disk.*.amivf"))
    assert len(files1) == 1
    paged_ivf.load_paged_ivf_index(ivf_db, "reuse_disk", dim, "angular", conn_factory=lambda: ivf_db, label="reuse_disk")
    files2 = _glob.glob(os.path.join(config.IVF_DISK_CACHE_DIR, "reuse_disk.*.amivf"))
    assert files2 == files1, "unchanged index must reuse the same cell file (same dir hash)"

    x2 = _make_clustered(n, dim, n_clusters=30, spread=0.4, seed=25)
    assert paged_ivf.build_and_store_paged_ivf(ivf_db, "reuse_disk", x2, item_ids, dim, "angular")
    paged_ivf.load_paged_ivf_index(ivf_db, "reuse_disk", dim, "angular", conn_factory=lambda: ivf_db, label="reuse_disk")
    files3 = _glob.glob(os.path.join(config.IVF_DISK_CACHE_DIR, "reuse_disk.*.amivf"))
    assert len(files3) == 1 and files3[0] != files1[0], "rebuild must create a new file and prune the old one"


def test_ivf_disk_cache_disabled_falls_back_to_postgres(ivf_db, monkeypatch):
    import config
    monkeypatch.setattr(config, "IVF_DISK_CACHE_ENABLED", False)
    from tasks import paged_ivf

    n, dim = 1500, 16
    x = _make_clustered(n, dim, n_clusters=30, spread=0.4, seed=26)
    item_ids = [f"fb-{i}" for i in range(n)]
    assert paged_ivf.build_and_store_paged_ivf(ivf_db, "fb_disk", x, item_ids, dim, "angular")

    counter = [0]
    counting = _CountingConn(ivf_db, counter)
    loaded = paged_ivf.load_paged_ivf_index(
        ivf_db, "fb_disk", dim, "angular", conn_factory=lambda: counting, label="fb_disk",
    )
    index = loaded[0]
    assert index._mmap is None, "mmap must be disabled"
    index.begin_request()
    counter[0] = 0
    index.query(x[10], k=11)
    assert counter[0] >= 1, "with disk cache disabled, query must read cells from Postgres"


def test_ivf_get_max_distance_exact(ivf_db, monkeypatch):
    import config
    monkeypatch.setattr(config, "IVF_STORAGE_DTYPE", "f32")  # exact 1e-4 distance + exact far_id
    from tasks import paged_ivf

    n, dim = 1500, 32
    x = _make_clustered(n, dim, n_clusters=30, spread=0.5, seed=11)
    item_ids = [f"t{i}" for i in range(n)]
    assert paged_ivf.build_and_store_paged_ivf(ivf_db, "mx_test", x, item_ids, dim, "angular")

    loaded = paged_ivf.load_paged_ivf_index(
        ivf_db, "mx_test", dim, "angular", conn_factory=lambda: ivf_db, label="mx_test",
    )
    index = loaded[0]
    index.begin_request()

    target = 42
    got, far_id = index.get_max_distance(target, nprobe=0)
    q = x[target]
    sims = x @ q
    sims[target] = 1e9
    expected = float(1.0 - np.clip(sims.min(), -1.0, 1.0))
    expected_far = int(np.argmin(sims))
    assert abs(got - expected) < 1e-4, f"max distance mismatch got={got} expected={expected}"
    assert far_id == expected_far, f"farthest id mismatch got={far_id} expected={expected_far}"

    approx, approx_far = index.get_max_distance(target)
    assert abs(approx - expected) < 1e-4, f"approx max distance off: got={approx} expected={expected}"


def test_ivf_max_distance_approx_reads_fewer_cells(ivf_db, monkeypatch):
    import config
    monkeypatch.setattr(config, "IVF_DISK_CACHE_ENABLED", False)
    from tasks import paged_ivf

    n, dim = 4000, 32
    x = _make_clustered(n, dim, n_clusters=60, spread=0.5, seed=17)
    item_ids = [f"mxf-{i}" for i in range(n)]
    assert paged_ivf.build_and_store_paged_ivf(ivf_db, "mxf_test", x, item_ids, dim, "angular")

    counter = [0]
    counting = _CountingConn(ivf_db, counter)
    index = paged_ivf.load_paged_ivf_index(
        ivf_db, "mxf_test", dim, "angular", conn_factory=lambda: counting, label="mxf_test")[0]

    paged_ivf.invalidate_global_cell_cache("mxf_test")
    index.begin_request()
    counter[0] = 0
    index.get_max_distance(7, nprobe=0)
    exact_reads = counter[0]

    paged_ivf.invalidate_global_cell_cache("mxf_test")
    index.begin_request()
    counter[0] = 0
    index.get_max_distance(7, nprobe=64)
    approx_reads = counter[0]

    assert approx_reads < exact_reads, f"approx round-trips {approx_reads} not < exact {exact_reads}"


def test_ivf_euclidean_metric(ivf_db):
    from tasks import paged_ivf

    n, dim = 2000, 24
    rng = np.random.default_rng(3)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    item_ids = [f"e{i}" for i in range(n)]
    assert paged_ivf.build_and_store_paged_ivf(ivf_db, "eu_test", x, item_ids, dim, "euclidean")

    loaded = paged_ivf.load_paged_ivf_index(
        ivf_db, "eu_test", dim, "euclidean", conn_factory=lambda: ivf_db, label="eu_test",
    )
    index = loaded[0]
    index.begin_request()
    qi = 100
    ids, dists = index.query(x[qi], k=5)
    assert ids
    diffs = x - x[qi][None, :]
    d = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
    d[qi] = 1e9
    true_nn = int(np.argmin(d))
    assert true_nn in set(int(i) for i in ids)


def test_ivf_directory_is_segmented_under_cap(ivf_db):
    from tasks import paged_ivf

    part_mb = 1
    part_bytes = part_mb * 1024 * 1024
    dim = 8
    n_items = 400000
    item_ids = [f"id{i}" for i in range(n_items)]
    id2cell = np.zeros(n_items, dtype=np.uint32)
    centroids = np.random.randn(1, dim).astype(np.float32)
    cells = [(0, np.arange(3, dtype=np.int32), np.random.randn(3, dim).astype(np.float32))]

    paged_ivf.store_paged_ivf(ivf_db, "captest", centroids, id2cell, item_ids, cells, dim, "angular", max_part_size_mb=part_mb)

    with ivf_db.cursor() as cur:
        cur.execute("SELECT count(*), max(octet_length(blob_data)) FROM ivf_dir WHERE name LIKE %s ESCAPE '\\'",
                    ("captest\\_\\_ivf\\_dir%",))
        n_parts, max_blob = cur.fetchone()
        cur.execute("SELECT max(octet_length(cell_data)) FROM ivf_cell WHERE index_name = %s", ("captest",))
        max_cell = cur.fetchone()[0]

    assert n_parts >= 2, f"directory should be segmented into multiple parts, got {n_parts}"
    assert max_blob <= part_bytes, f"a directory part is {max_blob} > cap {part_bytes}"
    assert max_cell <= part_bytes, f"a cell is {max_cell} > cap {part_bytes}"


def test_ivf_oversized_cell_is_split_not_rejected(ivf_db, monkeypatch):
    import config
    # store_paged_ivf writes f32 cells here, so pin storage to f32 for a consistent load.
    monkeypatch.setattr(config, "IVF_STORAGE_DTYPE", "f32")
    from tasks import paged_ivf

    dim = 8
    part_mb = 1
    cap = part_mb * 1024 * 1024
    n = 40000
    ids = np.arange(n, dtype=np.int32)
    vecs = np.random.randn(n, dim).astype(np.float32)
    cells = [(0, ids, vecs)]
    centroids = np.random.randn(1, dim).astype(np.float32)
    item_ids = [f"i{i}" for i in range(n)]
    id2cell = np.zeros(n, dtype=np.uint32)

    paged_ivf.store_paged_ivf(ivf_db, "bigcell", centroids, id2cell, item_ids, cells, dim, "angular", max_part_size_mb=part_mb)

    with ivf_db.cursor() as cur:
        cur.execute("SELECT count(*), max(octet_length(cell_data)) FROM ivf_cell WHERE index_name = %s", ("bigcell",))
        n_rows, max_cell = cur.fetchone()
    assert n_rows >= 2, f"oversized cell should split into multiple rows, got {n_rows}"
    assert max_cell <= cap, f"a cell is {max_cell} > cap {cap}"

    loaded = paged_ivf.load_paged_ivf_index(
        ivf_db, "bigcell", dim, "angular", conn_factory=lambda: ivf_db, label="bigcell",
    )
    assert loaded is not None
    index = loaded[0]
    assert index.num_elements == n
    index.begin_request()
    got = index.get_vectors([0, n - 1])
    assert 0 in got and (n - 1) in got


def test_ivf_real_build_all_rows_under_default_cap(ivf_db):
    import config
    from tasks import paged_ivf

    n, dim = 6000, 64
    x = _make_clustered(n, dim, n_clusters=120, spread=0.35, seed=21)
    item_ids = [f"s{i}" for i in range(n)]
    assert paged_ivf.build_and_store_paged_ivf(ivf_db, "capreal", x, item_ids, dim, "angular")

    cap = config.IVF_MAX_PART_SIZE_MB * 1024 * 1024
    with ivf_db.cursor() as cur:
        cur.execute("SELECT max(octet_length(cell_data)) FROM ivf_cell WHERE index_name = %s", ("capreal",))
        max_cell = cur.fetchone()[0]
        cur.execute("SELECT max(octet_length(blob_data)) FROM ivf_dir WHERE name LIKE %s ESCAPE '\\'",
                    ("capreal\\_\\_ivf\\_dir%",))
        max_blob = cur.fetchone()[0]
    assert max_cell is not None and max_cell <= cap
    assert max_blob is not None and max_blob <= cap


def test_ivf_build_splits_identical_vectors_under_cap(ivf_db, monkeypatch):
    import config
    from tasks import paged_ivf

    monkeypatch.setattr(config, "IVF_MAX_CELL_MB", 1)
    monkeypatch.setattr(config, "IVF_MAX_PART_SIZE_MB", 1)
    monkeypatch.setattr(config, "IVF_STORAGE_DTYPE", "f32")  # asserts exact get_vectors round-trip

    dim = 64
    n_dupes, n_rest = 8000, 4000
    rng = np.random.default_rng(7)
    shared = rng.standard_normal((1, dim)).astype(np.float32)
    dupes = np.repeat(shared, n_dupes, axis=0)
    rest = rng.standard_normal((n_rest, dim)).astype(np.float32)
    x = np.vstack([dupes, rest]).astype(np.float32)
    item_ids = [f"d{i}" for i in range(n_dupes + n_rest)]

    assert paged_ivf.build_and_store_paged_ivf(ivf_db, "dupes_test", x, item_ids, dim, "angular")

    cap = config.IVF_MAX_PART_SIZE_MB * 1024 * 1024
    with ivf_db.cursor() as cur:
        cur.execute("SELECT max(octet_length(cell_data)) FROM ivf_cell WHERE index_name = %s", ("dupes_test",))
        max_cell = cur.fetchone()[0]
    assert max_cell is not None and max_cell <= cap, f"a cell is {max_cell} > cap {cap}"

    loaded = paged_ivf.load_paged_ivf_index(
        ivf_db, "dupes_test", dim, "angular", conn_factory=lambda: ivf_db, label="dupes_test",
    )
    assert loaded is not None
    index = loaded[0]
    assert index.num_elements == n_dupes + n_rest

    from tasks.paged_ivf import _normalize_rows
    x_norm = _normalize_rows(x)
    index.begin_request()
    got = index.get_vectors([0, n_dupes - 1, n_dupes + 100])
    np.testing.assert_allclose(got[0], x_norm[0], atol=0)
    np.testing.assert_allclose(got[n_dupes - 1], x_norm[n_dupes - 1], atol=0)
    np.testing.assert_allclose(got[n_dupes + 100], x_norm[n_dupes + 100], atol=0)

    index.begin_request()
    probe_id = n_dupes + 100
    ids, _dists = index.query(x[probe_id], k=10)
    assert probe_id in [int(i) for i in ids]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-m", "integration", "-v", "-s"]))
