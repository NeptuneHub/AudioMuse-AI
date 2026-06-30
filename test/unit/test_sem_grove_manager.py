import sys
import os
import importlib.util
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


def _load_sem_grove():
    import types

    if 'tasks' not in sys.modules:
        stub = types.ModuleType('tasks')
        stub.__path__ = []
        sys.modules['tasks'] = stub

    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    )

    helper_path = os.path.join(repo_root, 'tasks', 'index_build_helpers.py')
    helper_name = 'tasks.index_build_helpers'
    if helper_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(helper_name, helper_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[helper_name] = mod
        spec.loader.exec_module(mod)

    mod_path = os.path.join(repo_root, 'tasks', 'sem_grove_manager.py')
    mod_name = 'tasks.sem_grove_manager'
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name]


_sgm = _load_sem_grove()


class TestMakeMergedVector:
    def _std(self, dim):
        return np.ones(dim, dtype=np.float32)

    def test_returns_float32_array_of_correct_shape(self):
        from tasks.sem_grove_manager import _make_merged_vector

        lyr = np.random.randn(8).astype(np.float32)
        a = np.random.randn(4).astype(np.float32)
        mv = _make_merged_vector(lyr, a, self._std(8), self._std(4), 1.0, 1.0)

        assert mv is not None
        assert mv.dtype == np.float32
        assert mv.shape == (12,)

    def test_weights_scale_halves(self):
        from tasks.sem_grove_manager import _make_merged_vector

        lyr = np.ones(4, dtype=np.float32)
        a = np.ones(4, dtype=np.float32)
        mv = _make_merged_vector(lyr, a, self._std(4), self._std(4), 1.0, 0.0)

        assert mv is not None
        assert np.any(mv[:4] != 0)
        np.testing.assert_array_equal(mv[4:], 0.0)

    def test_zero_lyrics_vector_returns_none(self):
        from tasks.sem_grove_manager import _make_merged_vector

        lyr = np.zeros(4, dtype=np.float32)
        a = np.ones(4, dtype=np.float32)
        assert _make_merged_vector(lyr, a, self._std(4), self._std(4), 1.0, 1.0) is None

    def test_zero_audio_vector_returns_none(self):
        from tasks.sem_grove_manager import _make_merged_vector

        lyr = np.ones(4, dtype=np.float32)
        a = np.zeros(4, dtype=np.float32)
        assert _make_merged_vector(lyr, a, self._std(4), self._std(4), 1.0, 1.0) is None

    def test_all_zero_whitening_std_still_returns_vector(self):
        from tasks.sem_grove_manager import _make_merged_vector

        lyr = np.ones(4, dtype=np.float32)
        a = np.ones(4, dtype=np.float32)
        std_zero = np.zeros(4, dtype=np.float32)
        mv = _make_merged_vector(lyr, a, std_zero, std_zero, 1.0, 1.0)
        assert mv is not None
        assert np.all(np.isfinite(mv))

    def test_cosine_of_identical_songs_is_one(self):
        from tasks.sem_grove_manager import _make_merged_vector

        rng = np.random.default_rng(0)
        lyr = rng.standard_normal(8).astype(np.float32)
        a = rng.standard_normal(4).astype(np.float32)
        std_l = np.abs(rng.standard_normal(8)).astype(np.float32) + 0.1
        std_a = np.abs(rng.standard_normal(4)).astype(np.float32) + 0.1

        mv1 = _make_merged_vector(lyr.copy(), a.copy(), std_l, std_a, 1.0, 1.0)
        mv2 = _make_merged_vector(lyr.copy(), a.copy(), std_l, std_a, 1.0, 1.0)

        assert mv1 is not None and mv2 is not None
        n1 = np.linalg.norm(mv1)
        n2 = np.linalg.norm(mv2)
        cos_sim = float(np.dot(mv1 / n1, mv2 / n2))
        assert abs(cos_sim - 1.0) < 1e-5


class TestCacheHelpers:
    def _patch_cache(self, loaded, id_map=None):
        fake_cache = {
            "loaded": loaded,
            "id_map": id_map,
            "song_count": len(id_map) if id_map else 0,
        }
        return patch("tasks.sem_grove_manager._SEM_GROVE_CACHE", fake_cache)

    def test_get_item_ids_returns_empty_set_when_not_loaded(self):
        from tasks.sem_grove_manager import get_sem_grove_item_ids

        with self._patch_cache(loaded=False):
            result = get_sem_grove_item_ids()
        assert result == set()

    def test_get_item_ids_returns_values_when_loaded(self):
        from tasks.sem_grove_manager import get_sem_grove_item_ids

        id_map = {0: "song-A", 1: "song-B", 2: "song-C"}
        with self._patch_cache(loaded=True, id_map=id_map):
            result = get_sem_grove_item_ids()
        assert result == {"song-A", "song-B", "song-C"}

    def test_is_loaded_false_when_cache_empty(self):
        from tasks.sem_grove_manager import is_sem_grove_cache_loaded

        with self._patch_cache(loaded=False):
            assert is_sem_grove_cache_loaded() is False

    def test_is_loaded_true_when_cache_filled(self):
        from tasks.sem_grove_manager import is_sem_grove_cache_loaded

        with self._patch_cache(loaded=True, id_map={0: "x"}):
            assert is_sem_grove_cache_loaded() is True


class TestSearchBySong:
    def _make_fake_index(self, n_songs, dim=12):
        rng = np.random.default_rng(42)
        vecs = []
        for _ in range(n_songs):
            v = rng.standard_normal(dim).astype(np.float32)
            v /= np.linalg.norm(v)
            vecs.append(v)

        mock_idx = MagicMock()
        mock_idx.__len__ = MagicMock(return_value=n_songs)

        mock_idx.get_vector.side_effect = lambda vid: vecs[vid]

        def fake_query(qvec, k):
            scores = [float(np.dot(qvec, v)) for v in vecs]
            ranked = sorted(range(n_songs), key=lambda i: -scores[i])[:k]
            dists = [1.0 - scores[i] for i in ranked]
            return np.array(ranked), np.array(dists, dtype=np.float32)

        mock_idx.query.side_effect = fake_query
        return mock_idx, vecs

    def _build_cache(self, n_songs, dim=12):
        idx, vecs = self._make_fake_index(n_songs, dim)
        id_map = {i: f"song-{i}" for i in range(n_songs)}
        rev_map = {v: k for k, v in id_map.items()}
        return {
            "index": idx,
            "id_map": id_map,
            "reverse_id_map": rev_map,
            "loaded": True,
            "song_count": n_songs,
        }, vecs

    def _fake_fetch_metadata(self, item_ids):
        return {iid: {"title": f"Title {iid}", "author": f"Artist {iid}"} for iid in item_ids}

    def test_returns_empty_when_not_loaded(self):
        from tasks.sem_grove_manager import search_by_song

        with patch("tasks.sem_grove_manager._SEM_GROVE_CACHE", {"loaded": False, "index": None}):
            assert search_by_song("any-id") == []

    def test_returns_empty_when_seed_not_in_index(self):
        from tasks.sem_grove_manager import search_by_song

        cache, _ = self._build_cache(5)
        with patch("tasks.sem_grove_manager._SEM_GROVE_CACHE", cache):
            assert search_by_song("unknown-id") == []

    def test_seed_is_first_with_is_seed_flag(self):
        from tasks.sem_grove_manager import search_by_song

        n = 10
        cache, _ = self._build_cache(n)
        seed = "song-0"

        with (
            patch("tasks.sem_grove_manager._SEM_GROVE_CACHE", cache),
            patch("tasks.sem_grove_manager._fetch_metadata", side_effect=self._fake_fetch_metadata),
            patch("config.MAX_SONGS_PER_ARTIST", 0),
            patch("config.DUPLICATE_DISTANCE_THRESHOLD_COSINE", 0.0),
            patch("config.DUPLICATE_DISTANCE_CHECK_LOOKBACK", 0),
        ):
            results = search_by_song(seed, limit=5)

        assert results, "search_by_song returned an empty list"
        assert results[0]["item_id"] == seed
        assert results[0]["is_seed"] is True
        assert results[0]["similarity"] == 1.0

    def test_limit_excludes_seed_from_count(self):
        from tasks.sem_grove_manager import search_by_song

        n = 20
        cache, _ = self._build_cache(n)
        seed = "song-3"
        limit = 5

        with (
            patch("tasks.sem_grove_manager._SEM_GROVE_CACHE", cache),
            patch("tasks.sem_grove_manager._fetch_metadata", side_effect=self._fake_fetch_metadata),
            patch("config.MAX_SONGS_PER_ARTIST", 0),
            patch("config.DUPLICATE_DISTANCE_THRESHOLD_COSINE", 0.0),
            patch("config.DUPLICATE_DISTANCE_CHECK_LOOKBACK", 0),
        ):
            results = search_by_song(seed, limit=limit)

        non_seed = [r for r in results if not r.get("is_seed")]
        assert len(non_seed) == limit

    def test_seed_never_appears_as_neighbour(self):
        from tasks.sem_grove_manager import search_by_song

        n = 10
        cache, _ = self._build_cache(n)
        seed = "song-2"

        with (
            patch("tasks.sem_grove_manager._SEM_GROVE_CACHE", cache),
            patch("tasks.sem_grove_manager._fetch_metadata", side_effect=self._fake_fetch_metadata),
            patch("config.MAX_SONGS_PER_ARTIST", 0),
            patch("config.DUPLICATE_DISTANCE_THRESHOLD_COSINE", 0.0),
            patch("config.DUPLICATE_DISTANCE_CHECK_LOOKBACK", 0),
        ):
            results = search_by_song(seed, limit=8)

        neighbour_ids = [r["item_id"] for r in results if not r.get("is_seed")]
        assert seed not in neighbour_ids

    def test_artist_cap_respected(self):
        from tasks.sem_grove_manager import search_by_song

        n = 20
        cache, _ = self._build_cache(n)
        seed = "song-0"

        def same_artist_fetch(item_ids):
            return {iid: {"title": f"Title {iid}", "author": "Same Artist"} for iid in item_ids}

        with (
            patch("tasks.sem_grove_manager._SEM_GROVE_CACHE", cache),
            patch("tasks.sem_grove_manager._fetch_metadata", side_effect=same_artist_fetch),
            patch("config.MAX_SONGS_PER_ARTIST", 1),
            patch("config.DUPLICATE_DISTANCE_THRESHOLD_COSINE", 0.0),
            patch("config.DUPLICATE_DISTANCE_CHECK_LOOKBACK", 0),
        ):
            results = search_by_song(seed, limit=10)

        neighbours = [r for r in results if not r.get("is_seed")]
        assert len(neighbours) <= 1

    def test_name_deduplication_removes_same_title_artist(self):
        from tasks.sem_grove_manager import search_by_song

        n = 10
        cache, _ = self._build_cache(n)
        seed = "song-0"

        def dedup_fetch(item_ids):
            result = {}
            for iid in item_ids:
                if iid in ("song-1", "song-2"):
                    result[iid] = {"title": "Dup Title", "author": "Dup Artist"}
                else:
                    result[iid] = {"title": f"Title {iid}", "author": f"Artist {iid}"}
            return result

        with (
            patch("tasks.sem_grove_manager._SEM_GROVE_CACHE", cache),
            patch("tasks.sem_grove_manager._fetch_metadata", side_effect=dedup_fetch),
            patch("config.MAX_SONGS_PER_ARTIST", 0),
            patch("config.DUPLICATE_DISTANCE_THRESHOLD_COSINE", 0.0),
            patch("config.DUPLICATE_DISTANCE_CHECK_LOOKBACK", 0),
        ):
            results = search_by_song(seed, limit=8)

        titles_authors = [(r["title"], r["author"]) for r in results if not r.get("is_seed")]
        assert titles_authors.count(("Dup Title", "Dup Artist")) <= 1


class TestSemGroveRoundTrip:
    pytest.importorskip("ivf", reason="ivf package required for round-trip test")

    def _make_db_mock(self, n_songs, lyrics_dim=16, audio_dim=8):
        rng = np.random.default_rng(7)

        lyrics_rows = []
        audio_rows = []
        for i in range(n_songs):
            lid = f"song-{i}"
            lv = rng.standard_normal(lyrics_dim).astype(np.float32).tobytes()
            av = rng.standard_normal(audio_dim).astype(np.float32).tobytes()
            lyrics_rows.append((lid, lv))
            audio_rows.append((lid, av))

        return lyrics_rows, audio_rows, lyrics_dim, audio_dim

    def _make_cursor_mock(self, lyrics_rows, audio_rows):
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)

        call_count = {"n": 0}

        def fetchall_side():
            call_count["n"] += 1
            if call_count["n"] == 1:
                return lyrics_rows
            return audio_rows

        mock_cur.fetchall.side_effect = fetchall_side

        mock_cur.fetchone.return_value = None
        return mock_cur

    def test_build_and_search_produces_seed_first(self):
        import io

        try:
            import ivf  # noqa: F401
        except ImportError:
            pytest.skip("ivf not installed")

        from tasks.sem_grove_manager import (
            build_and_store_sem_grove_index,
            search_by_song,
            _SEM_GROVE_CACHE,
        )

        n = 15
        lyrics_dim = 16
        audio_dim = 8
        lyrics_rows, audio_rows, ld, ad = self._make_db_mock(n, lyrics_dim, audio_dim)

        stored: dict = {}

        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchone.return_value = None

        def execute_side(sql, params=None):
            if params and "INSERT" in sql.upper() and len(params) >= 4:
                name, data, idmap, dim = params[0], params[1], params[2], params[3]
                stored[name] = (bytes(data) if data else b"", idmap, int(dim))

        mock_cur.execute.side_effect = execute_side

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_conn.commit.return_value = None

        def fake_stream(table, column, dim, where_clause=None, **kwargs):
            if table == "lyrics_embedding":
                rows = lyrics_rows
            elif table == "embedding":
                rows = audio_rows
            else:
                raise AssertionError(f"unexpected stream table: {table!r}")
            ids = [r[0] for r in rows]
            buf = np.empty((len(rows), dim), dtype=np.float32)
            for i, (_, blob) in enumerate(rows):
                buf[i] = np.frombuffer(blob, dtype=np.float32)
            return buf, ids

        import types as _t

        _ah_stub = _t.ModuleType('app_helper')
        _ah_stub.get_db = MagicMock(return_value=mock_conn)
        _ah_stub.get_score_data_by_ids = lambda item_ids: []

        with (
            patch.dict(sys.modules, {'app_helper': _ah_stub}),
            patch("config.LYRICS_EMBEDDING_DIMENSION", lyrics_dim, create=True),
            patch("config.EMBEDDING_DIMENSION", audio_dim, create=True),
            patch("tasks.index_build_helpers.stream_embeddings_to_buffer", side_effect=fake_stream),
        ):
            from config import IVF_MAX_PART_SIZE_MB  # noqa: F401

            ok = build_and_store_sem_grove_index(db_conn=mock_conn)

        if not ok or not stored:
            pytest.skip(
                "build_and_store_sem_grove_index did not store anything (likely missing config)"
            )

        whitening_row = stored.get("sem_grove_whitening")
        index_row = stored.get("sem_grove_index")

        if not whitening_row or not index_row:
            pytest.skip("Expected whitening and index rows not captured")

        whitening_json = whitening_row[1]
        index_binary = index_row[0]
        index_idmap = index_row[1]

        def fake_load():
            import json as _json
            import ivf as _ivf
            from config import IVF_QUERY_EF

            whitening = _json.loads(whitening_json)
            std_lyrics = np.array(whitening["std_lyrics"], dtype=np.float32)
            std_audio = np.array(whitening["std_audio"], dtype=np.float32)
            w_l = float(whitening["w_lyrics"])
            w_a = float(whitening["w_audio"])
            ld_ = int(whitening["lyrics_dim"])
            ad_ = int(whitening["audio_dim"])

            stream = io.BytesIO(index_binary)
            loaded = _ivf.Index.load(stream)
            loaded.ef = IVF_QUERY_EF

            id_map_ = {int(k): v for k, v in _json.loads(index_idmap).items()}
            reverse_id_map = {v: k for k, v in id_map_.items()}

            _SEM_GROVE_CACHE.update(
                {
                    "index": loaded,
                    "id_map": id_map_,
                    "reverse_id_map": reverse_id_map,
                    "std_lyrics": std_lyrics,
                    "std_audio": std_audio,
                    "lyrics_dim": ld_,
                    "audio_dim": ad_,
                    "w_lyrics": w_l,
                    "w_audio": w_a,
                    "loaded": True,
                    "song_count": len(id_map_),
                }
            )
            return True

        fake_load()

        seed_id = "song-0"

        def fake_fetch_meta(item_ids):
            return {iid: {"title": f"Title {iid}", "author": f"Artist {iid}"} for iid in item_ids}

        with (
            patch("tasks.sem_grove_manager._fetch_metadata", side_effect=fake_fetch_meta),
            patch("config.MAX_SONGS_PER_ARTIST", 0),
            patch("config.DUPLICATE_DISTANCE_THRESHOLD_COSINE", 0.0),
            patch("config.DUPLICATE_DISTANCE_CHECK_LOOKBACK", 0),
        ):
            results = search_by_song(seed_id, limit=5)

        assert results, "search_by_song returned nothing after round-trip build+load"
        assert results[0]["item_id"] == seed_id
        assert results[0].get("is_seed") is True
        non_seed = [r for r in results if not r.get("is_seed")]
        assert len(non_seed) <= 5
        for r in non_seed:
            assert r["item_id"] != seed_id
