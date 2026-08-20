# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Semantic-grove index that fuses lyrics and audio vectors for song search.

Covers the sem-grove manager's merged-vector construction, in-memory cache
helpers and song-seeded neighbour search, plus a build/load/search round trip
driven through the real disk-paged IVF layer against an in-memory stand-in for
the ivf_dir and ivf_cell tables.

Main Features:
* make_merged_vector returns a scaled float32 vector or None for zero inputs
* Cache helpers report loaded state and item ids only once filled
* search_by_song puts the seed first, excludes it from the limit and caps per artist
* Same title/artist neighbours are de-duplicated
* build_and_store_sem_grove_index persists one IVF directory plus cells covering
  every song, and load_sem_grove_cache_from_db reads them back so a search over
  the reloaded index is seed-first
"""

import re
import numpy as np
from unittest.mock import MagicMock, patch

from tasks.paged_ivf import unpack_cell, unpack_directory


def _blob_bytes(value):
    return bytes(getattr(value, "adapted", value))


def _segment_base(like_pattern):
    base = like_pattern.replace("\\", "")
    return base[: -len("_%_%")] if base.endswith("_%_%") else base


def _segment_pattern(like_pattern):
    return re.compile(r"^%s_\d+_\d+$" % re.escape(_segment_base(like_pattern)))


class _FakeIvfCursor:
    def __init__(self, db):
        self._db = db
        self._rows = []
        self.connection = db

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def close(self):
        return None

    def mogrify(self, template, args):
        self._db.mogrified.append(tuple(args))
        return ("<<%d>>" % (len(self._db.mogrified) - 1)).encode("ascii")

    def execute(self, sql, params=None):
        text = sql.decode("utf-8") if isinstance(sql, (bytes, bytearray)) else sql
        self._rows = self._db.run(" ".join(text.split()), params)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _FakeIvfDb:
    encoding = "UTF8"

    def __init__(self):
        self.cells = {}
        self.blobs = {}
        self.mogrified = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, *args, **kwargs):
        return _FakeIvfCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def run(self, sql, params):
        if sql.startswith("DELETE FROM ivf_cell"):
            for key in [k for k in self.cells if k[0] == params[0]]:
                del self.cells[key]
            return []
        if sql.startswith("INSERT INTO ivf_cell"):
            for token in re.findall(r"<<(\d+)>>", sql):
                name, cell_id, blob = self.mogrified[int(token)]
                self.cells[(name, int(cell_id))] = _blob_bytes(blob)
            return []
        if sql.startswith("DELETE FROM ivf_dir"):
            pattern = _segment_pattern(params[1])
            for name in [n for n in self.blobs if n == params[0] or pattern.match(n)]:
                del self.blobs[name]
            return []
        if sql.startswith("INSERT INTO ivf_dir"):
            self.blobs[params[0]] = _blob_bytes(params[1])
            return []
        if sql.startswith("SELECT blob_data FROM ivf_dir"):
            blob = self.blobs.get(params[0])
            return [(blob,)] if blob is not None else []
        if sql.startswith("SELECT name, blob_data FROM ivf_dir"):
            pattern = _segment_pattern(params[0])
            return [(n, b) for n, b in self.blobs.items() if pattern.match(n)]
        if sql.startswith("SELECT cell_id, octet_length(cell_data) FROM ivf_cell"):
            return sorted(
                (cell_id, len(blob))
                for (name, cell_id), blob in self.cells.items()
                if name == params[0]
            )
        if sql.startswith("SELECT cell_id, cell_data FROM ivf_cell"):
            wanted = {int(c) for c in params[1]}
            return [
                (cell_id, blob)
                for (name, cell_id), blob in self.cells.items()
                if name == params[0] and cell_id in wanted
            ]
        raise AssertionError("unexpected SQL against the fake IVF database: " + sql)


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
            patch("config.DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS", 0.0),
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
            patch("config.DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS", 0.0),
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
            patch("config.DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS", 0.0),
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
            patch("config.DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS", 0.0),
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
            patch("config.DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS", 0.0),
            patch("config.DUPLICATE_DISTANCE_CHECK_LOOKBACK", 0),
        ):
            results = search_by_song(seed, limit=8)

        titles_authors = [(r["title"], r["author"]) for r in results if not r.get("is_seed")]
        assert titles_authors.count(("Dup Title", "Dup Artist")) <= 1


class TestSemGroveRoundTrip:
    LYRICS_DIM = 16
    AUDIO_DIM = 8
    MERGED_DIM = 24
    SONG_COUNT = 15
    INDEX_NAME = "sem_grove_index"
    DIRECTORY_NAME = "sem_grove_index__ivf_dir"

    def _build_into_fake_db(self):
        from tasks.sem_grove_manager import build_and_store_sem_grove_index

        rng = np.random.default_rng(7)
        item_ids = [f"song-{i}" for i in range(self.SONG_COUNT)]
        lyrics = rng.standard_normal((self.SONG_COUNT, self.LYRICS_DIM)).astype(np.float32)
        audio = rng.standard_normal((self.SONG_COUNT, self.AUDIO_DIM)).astype(np.float32)

        def fake_stream(table, column, dim, where_clause=None, **kwargs):
            if table == "lyrics_embedding":
                return lyrics.copy(), list(item_ids)
            if table == "embedding":
                return audio.copy(), list(item_ids)
            raise AssertionError(f"unexpected stream table: {table!r}")

        db = _FakeIvfDb()
        with (
            patch("config.LYRICS_EMBEDDING_DIMENSION", self.LYRICS_DIM),
            patch("config.EMBEDDING_DIMENSION", self.AUDIO_DIM),
            patch("config.IVF_DISK_CACHE_ENABLED", False),
            patch(
                "tasks.index_build_helpers.stream_embeddings_to_buffer",
                side_effect=fake_stream,
            ),
        ):
            ok = build_and_store_sem_grove_index(db_conn=db)
        return ok, db, item_ids

    def test_build_persists_one_ivf_directory_and_cells_covering_every_song(self):
        ok, db, item_ids = self._build_into_fake_db()

        assert ok is True
        assert db.commits == 1
        assert db.rollbacks == 0
        assert set(db.blobs) == {self.DIRECTORY_NAME}

        centroids, id2cell, stored_ids, dim, metric, normalized, storage_dtype = unpack_directory(
            db.blobs[self.DIRECTORY_NAME]
        )
        assert stored_ids == sorted(item_ids)
        assert dim == self.MERGED_DIM
        assert metric == "angular"
        assert normalized is True
        assert centroids.shape[1] == self.MERGED_DIM
        assert id2cell.shape == (self.SONG_COUNT,)

        assert {name for name, _cell_id in db.cells} == {self.INDEX_NAME}
        vector_ids = []
        for blob in db.cells.values():
            ids, vecs = unpack_cell(blob, dim, storage_dtype)
            assert vecs.shape == (ids.shape[0], self.MERGED_DIM)
            vector_ids.extend(int(i) for i in ids)
        assert sorted(vector_ids) == list(range(self.SONG_COUNT))

    def test_stored_index_reloads_through_the_real_cache_loader(self):
        import tasks.sem_grove_manager as sgm
        from tasks.sem_grove_manager import (
            get_sem_grove_item_ids,
            is_sem_grove_cache_loaded,
            load_sem_grove_cache_from_db,
        )

        ok, db, item_ids = self._build_into_fake_db()
        assert ok is True

        with (
            patch("database.get_db", return_value=db),
            patch.dict(sgm._SEM_GROVE_CACHE, {}, clear=False),
            patch("config.LYRICS_EMBEDDING_DIMENSION", self.LYRICS_DIM),
            patch("config.EMBEDDING_DIMENSION", self.AUDIO_DIM),
            patch("config.IVF_DISK_CACHE_ENABLED", False),
        ):
            loaded = load_sem_grove_cache_from_db()

            assert loaded is True
            assert is_sem_grove_cache_loaded() is True
            assert get_sem_grove_item_ids() == set(item_ids)
            assert sgm._SEM_GROVE_CACHE["song_count"] == self.SONG_COUNT
            assert sgm._SEM_GROVE_CACHE["lyrics_dim"] == self.LYRICS_DIM
            assert sgm._SEM_GROVE_CACHE["audio_dim"] == self.AUDIO_DIM
            assert len(sgm._SEM_GROVE_CACHE["index"]) == self.SONG_COUNT

    def test_missing_directory_row_leaves_the_cache_unloaded(self):
        import tasks.sem_grove_manager as sgm
        from tasks.sem_grove_manager import get_sem_grove_item_ids, load_sem_grove_cache_from_db

        db = _FakeIvfDb()

        with (
            patch("database.get_db", return_value=db),
            patch.dict(sgm._SEM_GROVE_CACHE, {}, clear=False),
            patch("config.LYRICS_EMBEDDING_DIMENSION", self.LYRICS_DIM),
            patch("config.EMBEDDING_DIMENSION", self.AUDIO_DIM),
            patch("config.IVF_DISK_CACHE_ENABLED", False),
        ):
            assert load_sem_grove_cache_from_db() is False
            assert sgm._SEM_GROVE_CACHE["loaded"] is False
            assert sgm._SEM_GROVE_CACHE["song_count"] == 0
            assert sgm._SEM_GROVE_CACHE["index"] is None
            assert get_sem_grove_item_ids() == set()

    def test_search_over_the_reloaded_index_is_seed_first_then_five_neighbours(self):
        import tasks.sem_grove_manager as sgm
        from tasks.sem_grove_manager import load_sem_grove_cache_from_db, search_by_song

        ok, db, item_ids = self._build_into_fake_db()
        assert ok is True

        def fake_fetch_metadata(ids):
            return {
                iid: {"title": f"Title {iid}", "author": f"Artist {iid}", "album": ""}
                for iid in ids
            }

        seed = "song-0"
        with (
            patch("database.get_db", return_value=db),
            patch.dict(sgm._SEM_GROVE_CACHE, {}, clear=False),
            patch("config.LYRICS_EMBEDDING_DIMENSION", self.LYRICS_DIM),
            patch("config.EMBEDDING_DIMENSION", self.AUDIO_DIM),
            patch("config.IVF_DISK_CACHE_ENABLED", False),
            patch("config.MAX_SONGS_PER_ARTIST", 0),
            patch("config.DUPLICATE_DISTANCE_CHECK_LOOKBACK", 0),
            patch("tasks.sem_grove_manager._fetch_metadata", side_effect=fake_fetch_metadata),
        ):
            assert load_sem_grove_cache_from_db() is True
            results = search_by_song(seed, limit=5, radius_similarity=False)

        assert results[0]["item_id"] == seed
        assert results[0]["is_seed"] is True
        assert results[0]["similarity"] == 1.0
        assert results[0]["title"] == "Title song-0"

        neighbours = [r["item_id"] for r in results[1:]]
        assert len(neighbours) == 5
        assert len(set(neighbours)) == 5
        assert seed not in neighbours
        assert set(neighbours) <= set(item_ids)
        assert all(r.get("is_seed") is None for r in results[1:])
        assert [r["similarity"] for r in results[1:]] == sorted(
            (r["similarity"] for r in results[1:]), reverse=True
        )
