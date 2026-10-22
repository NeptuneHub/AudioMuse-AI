# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""CLAP text-to-audio search over the cached IVF index in clap_text_search.

Covers cache stats/state reporting, search_by_text ranking and search_by_song
against a dummy IVF index, using cosine similarity of the query vector.

Main Features:
* get_cache_stats and is_clap_cache_loaded reflect loaded/unloaded state
* Results are ranked by descending similarity, respect the limit, and carry the
  required item_id/title/author/similarity fields
* Edge cases: CLAP disabled, cache unloaded, failed text embedding, zero limit
  all return an empty list
* search_by_song seeds the index with the seed's stored embedding row and
  follows the MusiCNN similar-song rules: the seed and any copy of it (same
  name or identical vector) are never returned, the seed uses no artist slot,
  and the artist limit follows eliminate_duplicates. A starved pool is read
  deeper once, an exhausted one is not, the index snapshot survives a refresh,
  and a missing seed or an unloaded index raise distinct errors
* The radius walk (default SIMILARITY_RADIUS_DEFAULT) walks a MusiCNN-sized
  deduplicated pool through the shared helper with the artist limit, never
  sees the seed or its copies, and falls back to the plain order if it fails
* The song picker filter offers only songs with a DCLAP embedding
"""

import numpy as np
import pytest
from unittest.mock import patch


class DummyIVFIndex:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def begin_request(self):
        pass

    def distance_to_similarity(self, distance):
        return 1.0 - float(distance)

    def get_vector(self, vec_id):
        return self.embeddings[int(vec_id)]

    def query(self, query_vector: np.ndarray, k: int):
        similarities = self.embeddings @ query_vector
        order = np.argsort(similarities)[::-1]
        top = order[:k]
        distances = (1.0 - similarities[top]).astype(np.float32)
        return list(top), distances


def setup_dummy_clap_index_cache(_CLAP_INDEX_CACHE, embeddings, item_ids):
    _CLAP_INDEX_CACHE['loaded'] = True
    _CLAP_INDEX_CACHE['index'] = DummyIVFIndex(embeddings)
    _CLAP_INDEX_CACHE['id_map'] = {i: item_ids[i] for i in range(len(item_ids))}
    _CLAP_INDEX_CACHE['reverse_id_map'] = {item_ids[i]: i for i in range(len(item_ids))}


def teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE):
    _CLAP_INDEX_CACHE['loaded'] = False
    _CLAP_INDEX_CACHE['index'] = None
    _CLAP_INDEX_CACHE['id_map'] = None
    _CLAP_INDEX_CACHE['reverse_id_map'] = None


class TestCacheStatsCalculation:
    def test_get_cache_stats_when_not_loaded(self):
        from tasks.clap_text_search import get_cache_stats, _CLAP_CACHE, _CLAP_INDEX_CACHE

        _CLAP_CACHE['loaded'] = False
        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)

        stats = get_cache_stats()

        assert stats['loaded'] is False
        assert stats['song_count'] == 0
        assert stats['embedding_dimension'] == 0
        assert stats['memory_mb'] == 0

    def test_get_cache_stats_with_loaded_cache(self):
        from tasks.clap_text_search import get_cache_stats, _CLAP_CACHE, _CLAP_INDEX_CACHE

        _CLAP_CACHE['loaded'] = True
        embeddings = np.random.rand(100, 512).astype(np.float32)
        item_ids = [f'song{i}' for i in range(100)]
        setup_dummy_clap_index_cache(_CLAP_INDEX_CACHE, embeddings, item_ids)

        stats = get_cache_stats()

        assert stats['loaded'] is True
        assert stats['song_count'] == 100
        assert stats['embedding_dimension'] == 512
        assert stats['memory_mb'] > 0

        _CLAP_CACHE['loaded'] = False
        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)

    def test_get_cache_stats_memory_calculation_accuracy(self):
        from tasks.clap_text_search import get_cache_stats, _CLAP_CACHE, _CLAP_INDEX_CACHE

        _CLAP_CACHE['loaded'] = True
        embeddings = np.random.rand(10, 512).astype(np.float32)
        item_ids = [f'song{i}' for i in range(10)]
        setup_dummy_clap_index_cache(_CLAP_INDEX_CACHE, embeddings, item_ids)

        stats = get_cache_stats()

        assert 0.0 < stats['memory_mb'] < 1.0

        _CLAP_CACHE['loaded'] = False
        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)


class TestCacheStateChecking:
    def test_is_clap_cache_loaded_when_true(self):
        from tasks.clap_text_search import is_clap_cache_loaded, _CLAP_CACHE

        _CLAP_CACHE['loaded'] = True

        assert is_clap_cache_loaded() is True

        _CLAP_CACHE['loaded'] = False

    def test_is_clap_cache_loaded_when_false(self):
        from tasks.clap_text_search import is_clap_cache_loaded, _CLAP_CACHE

        _CLAP_CACHE['loaded'] = False

        assert is_clap_cache_loaded() is False


class TestSimilarityCalculationLogic:
    @patch('config.CLAP_ENABLED', True)
    @patch('tasks.clap_analyzer.get_text_embedding')
    def test_search_similarity_ranking_order(self, mock_get_embedding):
        from tasks.clap_text_search import search_by_text, _CLAP_CACHE, _CLAP_INDEX_CACHE

        _CLAP_CACHE['loaded'] = True
        _CLAP_CACHE['embeddings'] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 0.0, 1.0],
                [0.6, 0.8, 0.0],
            ],
            dtype=np.float32,
        )

        for i in range(len(_CLAP_CACHE['embeddings'])):
            norm = np.linalg.norm(_CLAP_CACHE['embeddings'][i])
            if norm > 0:
                _CLAP_CACHE['embeddings'][i] = _CLAP_CACHE['embeddings'][i] / norm

        _CLAP_CACHE['metadata'] = [
            {'item_id': f'song{i}', 'title': f'Song {i}', 'author': f'Artist {i}'} for i in range(5)
        ]
        _CLAP_CACHE['item_ids'] = [f'song{i}' for i in range(5)]
        setup_dummy_clap_index_cache(
            _CLAP_INDEX_CACHE, _CLAP_CACHE['embeddings'], _CLAP_CACHE['item_ids']
        )

        query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        mock_get_embedding.return_value = query_embedding

        results = search_by_text("test query", limit=5)

        assert len(results) == 5
        for i in range(len(results) - 1):
            assert results[i]['similarity'] >= results[i + 1]['similarity']

        assert results[0]['item_id'] == 'song0'

        _CLAP_CACHE['loaded'] = False
        _CLAP_CACHE['embeddings'] = None
        _CLAP_CACHE['metadata'] = None
        _CLAP_CACHE['item_ids'] = None

    @patch('config.CLAP_ENABLED', True)
    @patch('tasks.clap_analyzer.get_text_embedding')
    def test_search_respects_limit_parameter(self, mock_get_embedding):
        from tasks.clap_text_search import search_by_text, _CLAP_CACHE, _CLAP_INDEX_CACHE

        _CLAP_CACHE['loaded'] = True
        _CLAP_CACHE['embeddings'] = np.random.rand(20, 512).astype(np.float32)

        for i in range(len(_CLAP_CACHE['embeddings'])):
            _CLAP_CACHE['embeddings'][i] /= np.linalg.norm(_CLAP_CACHE['embeddings'][i])

        _CLAP_CACHE['metadata'] = [
            {'item_id': f'song{i}', 'title': f'Song {i}', 'author': f'Artist {i}'}
            for i in range(20)
        ]
        _CLAP_CACHE['item_ids'] = [f'song{i}' for i in range(20)]
        setup_dummy_clap_index_cache(
            _CLAP_INDEX_CACHE, _CLAP_CACHE['embeddings'], _CLAP_CACHE['item_ids']
        )

        query_embedding = np.random.rand(512).astype(np.float32)
        query_embedding /= np.linalg.norm(query_embedding)
        mock_get_embedding.return_value = query_embedding

        results = search_by_text("test query", limit=5)

        assert len(results) == 5

        results = search_by_text("test query", limit=15)

        assert len(results) == 15

        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)
        _CLAP_CACHE['loaded'] = False
        _CLAP_CACHE['embeddings'] = None
        _CLAP_CACHE['metadata'] = None
        _CLAP_CACHE['item_ids'] = None

    @patch('config.CLAP_ENABLED', True)
    @patch('tasks.clap_analyzer.get_text_embedding')
    def test_search_limit_larger_than_cache(self, mock_get_embedding):
        from tasks.clap_text_search import search_by_text, _CLAP_CACHE, _CLAP_INDEX_CACHE

        _CLAP_CACHE['loaded'] = True
        _CLAP_CACHE['embeddings'] = np.random.rand(5, 512).astype(np.float32)

        for i in range(len(_CLAP_CACHE['embeddings'])):
            _CLAP_CACHE['embeddings'][i] /= np.linalg.norm(_CLAP_CACHE['embeddings'][i])

        _CLAP_CACHE['metadata'] = [
            {'item_id': f'song{i}', 'title': f'Song {i}', 'author': f'Artist {i}'} for i in range(5)
        ]
        _CLAP_CACHE['item_ids'] = [f'song{i}' for i in range(5)]
        setup_dummy_clap_index_cache(
            _CLAP_INDEX_CACHE, _CLAP_CACHE['embeddings'], _CLAP_CACHE['item_ids']
        )

        query_embedding = np.random.rand(512).astype(np.float32)
        query_embedding /= np.linalg.norm(query_embedding)
        mock_get_embedding.return_value = query_embedding

        results = search_by_text("test query", limit=100)

        assert len(results) == 5

        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)
        _CLAP_CACHE['loaded'] = False
        _CLAP_CACHE['embeddings'] = None
        _CLAP_CACHE['metadata'] = None
        _CLAP_CACHE['item_ids'] = None


class TestSearchResultStructure:
    @patch('config.CLAP_ENABLED', True)
    @patch('tasks.clap_analyzer.get_text_embedding')
    def test_search_returns_one_result_per_song_with_item_id_title_author_and_similarity(
        self, mock_get_embedding
    ):
        from tasks.clap_text_search import search_by_text, _CLAP_CACHE, _CLAP_INDEX_CACHE

        _CLAP_CACHE['loaded'] = True
        embeddings = np.random.rand(3, 512).astype(np.float32)

        for i in range(len(embeddings)):
            embeddings[i] /= np.linalg.norm(embeddings[i])

        item_ids = ['song1', 'song2', 'song3']
        setup_dummy_clap_index_cache(_CLAP_INDEX_CACHE, embeddings, item_ids)

        query_embedding = np.random.rand(512).astype(np.float32)
        query_embedding /= np.linalg.norm(query_embedding)
        mock_get_embedding.return_value = query_embedding

        with patch('database.get_score_data_by_ids') as mock_get_score_data:
            mock_get_score_data.return_value = [
                {
                    'item_id': 'song1',
                    'title': 'Test Song',
                    'author': 'Test Artist',
                    'album': 'Test Album',
                },
                {
                    'item_id': 'song2',
                    'title': 'Another Song',
                    'author': 'Another Artist',
                    'album': 'Another Album',
                },
                {
                    'item_id': 'song3',
                    'title': 'Third Song',
                    'author': 'Third Artist',
                    'album': 'Third Album',
                },
            ]
            results = search_by_text("test query", limit=3)

        assert len(results) == 3

        by_id = {result['item_id']: result for result in results}
        assert sorted(by_id) == ['song1', 'song2', 'song3']

        assert by_id['song1']['title'] == 'Test Song'
        assert by_id['song1']['author'] == 'Test Artist'
        assert by_id['song1']['album'] == 'Test Album'
        assert by_id['song2']['title'] == 'Another Song'
        assert by_id['song2']['author'] == 'Another Artist'
        assert by_id['song2']['album'] == 'Another Album'
        assert by_id['song3']['title'] == 'Third Song'
        assert by_id['song3']['author'] == 'Third Artist'
        assert by_id['song3']['album'] == 'Third Album'

        for result in results:
            assert isinstance(result['similarity'], float)
            assert -1.0 <= result['similarity'] <= 1.0

        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)
        _CLAP_CACHE['loaded'] = False

    @patch('config.CLAP_ENABLED', True)
    @patch('tasks.clap_analyzer.get_text_embedding')
    def test_search_preserves_metadata_accurately(self, mock_get_embedding):
        from tasks.clap_text_search import search_by_text, _CLAP_CACHE, _CLAP_INDEX_CACHE

        _CLAP_CACHE['loaded'] = True
        _CLAP_CACHE['embeddings'] = np.random.rand(2, 512).astype(np.float32)

        for i in range(len(_CLAP_CACHE['embeddings'])):
            _CLAP_CACHE['embeddings'][i] /= np.linalg.norm(_CLAP_CACHE['embeddings'][i])

        _CLAP_CACHE['metadata'] = [
            {'item_id': 'abc123', 'title': 'Song 1', 'author': 'Artist A'},
            {'item_id': 'xyz789', 'title': 'Song 2', 'author': 'Artist B'},
        ]
        _CLAP_CACHE['item_ids'] = ['abc123', 'xyz789']
        setup_dummy_clap_index_cache(
            _CLAP_INDEX_CACHE, _CLAP_CACHE['embeddings'], _CLAP_CACHE['item_ids']
        )

        query_embedding = np.random.rand(512).astype(np.float32)
        query_embedding /= np.linalg.norm(query_embedding)
        mock_get_embedding.return_value = query_embedding

        with patch('database.get_score_data_by_ids') as mock_get_score_data:
            mock_get_score_data.return_value = [
                {'item_id': 'abc123', 'title': 'Song 1', 'author': 'Artist A'},
                {'item_id': 'xyz789', 'title': 'Song 2', 'author': 'Artist B'},
            ]
            results = search_by_text("rock classics", limit=2)

        song1 = next((r for r in results if r['item_id'] == 'abc123'), None)
        song2 = next((r for r in results if r['item_id'] == 'xyz789'), None)

        assert song1 is not None
        assert song1['title'] == 'Song 1'
        assert song1['author'] == 'Artist A'

        assert song2 is not None
        assert song2['title'] == 'Song 2'
        assert song2['author'] == 'Artist B'

        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)
        _CLAP_CACHE['loaded'] = False
        _CLAP_CACHE['embeddings'] = None
        _CLAP_CACHE['metadata'] = None
        _CLAP_CACHE['item_ids'] = None


class TestSearchEdgeCases:
    @patch('config.CLAP_ENABLED', False)
    @patch('tasks.clap_analyzer.get_text_embedding')
    def test_search_returns_empty_when_disabled_even_with_a_loaded_matching_index(
        self, mock_get_embedding
    ):
        from tasks.clap_text_search import search_by_text, _CLAP_INDEX_CACHE

        embeddings = np.eye(3, dtype=np.float32)
        setup_dummy_clap_index_cache(_CLAP_INDEX_CACHE, embeddings, ['song0', 'song1', 'song2'])
        mock_get_embedding.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        try:
            assert search_by_text("any query", limit=10) == []
            mock_get_embedding.assert_not_called()

            with patch('config.CLAP_ENABLED', True):
                enabled_results = search_by_text("any query", limit=10)

            assert len(enabled_results) == 3
            assert enabled_results[0]['item_id'] == 'song0'
        finally:
            teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)

    @patch('config.CLAP_ENABLED', True)
    @patch('tasks.clap_analyzer.get_text_embedding')
    def test_search_returns_empty_when_the_index_cache_is_not_loaded(self, mock_get_embedding):
        from tasks.clap_text_search import search_by_text, _CLAP_CACHE, _CLAP_INDEX_CACHE

        _CLAP_CACHE['loaded'] = True
        _CLAP_INDEX_CACHE['loaded'] = False
        _CLAP_INDEX_CACHE['index'] = None
        mock_get_embedding.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        results = search_by_text("test query", limit=10)

        assert results == []
        mock_get_embedding.assert_not_called()

        _CLAP_CACHE['loaded'] = False

    @patch('config.CLAP_ENABLED', True)
    @patch('tasks.clap_analyzer.get_text_embedding')
    def test_search_returns_empty_without_querying_the_index_when_text_embedding_fails(
        self, mock_get_embedding
    ):
        from tasks.clap_text_search import search_by_text, _CLAP_CACHE, _CLAP_INDEX_CACHE

        _CLAP_CACHE['loaded'] = True
        embeddings = np.random.rand(5, 512).astype(np.float32)
        setup_dummy_clap_index_cache(
            _CLAP_INDEX_CACHE, embeddings, [f'song{i}' for i in range(5)]
        )

        mock_get_embedding.return_value = None

        index = _CLAP_INDEX_CACHE['index']
        with patch.object(index, 'query', wraps=index.query) as query_spy, patch.object(
            index, 'begin_request', wraps=index.begin_request
        ) as begin_request_spy:
            results = search_by_text("test query", limit=10)

        assert results == []
        query_spy.assert_not_called()
        begin_request_spy.assert_not_called()

        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)
        _CLAP_CACHE['loaded'] = False

    @patch('config.CLAP_ENABLED', True)
    @patch('tasks.clap_analyzer.get_text_embedding')
    def test_search_with_zero_limit(self, mock_get_embedding):
        from tasks.clap_text_search import search_by_text, _CLAP_CACHE, _CLAP_INDEX_CACHE

        _CLAP_CACHE['loaded'] = True
        _CLAP_CACHE['embeddings'] = np.random.rand(10, 512).astype(np.float32)

        for i in range(len(_CLAP_CACHE['embeddings'])):
            _CLAP_CACHE['embeddings'][i] /= np.linalg.norm(_CLAP_CACHE['embeddings'][i])

        _CLAP_CACHE['metadata'] = [
            {'item_id': f'song{i}', 'title': f'Song {i}', 'author': f'Artist {i}'}
            for i in range(10)
        ]
        _CLAP_CACHE['item_ids'] = [f'song{i}' for i in range(10)]
        setup_dummy_clap_index_cache(
            _CLAP_INDEX_CACHE, _CLAP_CACHE['embeddings'], _CLAP_CACHE['item_ids']
        )

        query_embedding = np.random.rand(512).astype(np.float32)
        query_embedding /= np.linalg.norm(query_embedding)
        mock_get_embedding.return_value = query_embedding

        results = search_by_text("test query", limit=0)

        assert results == []

        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)
        _CLAP_CACHE['loaded'] = False
        _CLAP_CACHE['embeddings'] = None
        _CLAP_CACHE['metadata'] = None
        _CLAP_CACHE['item_ids'] = None


def _fan_embeddings(n, dim=512):
    out = np.zeros((n, dim), dtype=np.float32)
    out[0, 0] = 1.0
    for i, closeness in enumerate(np.linspace(0.95, 0.3, n - 1), start=1):
        out[i, 0] = closeness
        out[i, i] = np.sqrt(1.0 - closeness * closeness)
    return out


class _SongIndex(DummyIVFIndex):
    def __init__(self, embeddings, probe_limit=None):
        super().__init__(embeddings)
        self.probe_limit = probe_limit
        self.query_ks = []

    def get_vectors(self, vec_ids):
        return {int(v): self.embeddings[int(v)] for v in vec_ids}

    def query(self, query_vector, k):
        self.query_ks.append(k)
        if self.probe_limit is not None:
            k = min(k, self.probe_limit)
        return super().query(query_vector, k)


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.row = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        self.conn.seen.append(params[0])
        blob = self.conn.store.get(params[0])
        self.row = None if blob is None else (memoryview(blob),)

    def fetchone(self):
        return self.row


class _FakeConn:
    def __init__(self, store):
        self.store = store
        self.seen = []

    def cursor(self, *args, **kwargs):
        return _FakeCursor(self)


@pytest.fixture
def song_index(monkeypatch):
    import database
    from tasks import clap_text_search

    def install(embeddings, authors=None, titles=None, probe_limit=None):
        item_ids = [f'song{i}' for i in range(len(embeddings))]
        setup_dummy_clap_index_cache(clap_text_search._CLAP_INDEX_CACHE, embeddings, item_ids)
        index = _SongIndex(embeddings, probe_limit)
        clap_text_search._CLAP_INDEX_CACHE['index'] = index
        conn = _FakeConn({iid: embeddings[i].tobytes() for i, iid in enumerate(item_ids)})
        monkeypatch.setattr(database, 'get_db', lambda: conn)
        meta = {
            iid: {
                'title': titles[i] if titles else f'Song {i}',
                'author': authors[i] if authors else f'Artist {i}',
                'album': '',
            }
            for i, iid in enumerate(item_ids)
        }
        monkeypatch.setattr(
            clap_text_search, '_fetch_clap_metadata', lambda ids: {i: meta[i] for i in ids if i in meta}
        )
        return conn, index

    monkeypatch.setattr('config.CLAP_ENABLED', True)
    monkeypatch.setattr('config.MAX_SONGS_PER_ARTIST', 3)
    monkeypatch.setattr('config.SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT', True)
    monkeypatch.setattr('config.SIMILARITY_RADIUS_DEFAULT', False)
    monkeypatch.setattr('config.DUPLICATE_DISTANCE_THRESHOLD_COSINE_CLAP', 0.01)
    monkeypatch.setattr('config.DUPLICATE_DISTANCE_CHECK_LOOKBACK', 1)
    yield install
    teardown_dummy_clap_index_cache(clap_text_search._CLAP_INDEX_CACHE)


class TestSearchBySong:
    def test_the_seed_is_never_returned_and_neighbours_come_closest_first(self, song_index):
        from tasks.clap_text_search import search_by_song

        song_index(_fan_embeddings(8))
        results = search_by_song('song0', limit=5)

        assert [r['item_id'] for r in results] == ['song1', 'song2', 'song3', 'song4', 'song5']
        similarities = [r['similarity'] for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_the_seed_vector_is_its_stored_clap_embedding_row(self, song_index):
        from tasks.clap_text_search import search_by_song

        conn, _ = song_index(_fan_embeddings(7))
        results = search_by_song('song3', limit=2)

        assert conn.seen == ['song3']
        assert [r['item_id'] for r in results][0] == 'song0'
        assert 'song3' not in [r['item_id'] for r in results]

    def test_the_seed_uses_no_artist_slot_and_a_shortfall_reads_a_deeper_pool(self, song_index):
        from tasks.clap_text_search import search_by_song

        authors = ['Artist A'] * 40 + [f'Artist {i}' for i in range(40, 60)]
        _, index = song_index(_fan_embeddings(60), authors)
        results = search_by_song('song0', limit=5)

        assert len(results) == 5
        assert 'song0' not in [r['item_id'] for r in results]
        assert sum(1 for r in results if r['author'] == 'Artist A') == 3, (
            'the seed is by Artist A too, but like the MusiCNN path it must leave all 3 slots to other songs'
        )
        assert index.query_ks == [26, 60], 'the first pool of 26 holds one artist only, so one deeper pool is read'

    def test_an_exhausted_pool_is_not_queried_a_second_time(self, song_index):
        from tasks.clap_text_search import search_by_song

        _, index = song_index(_fan_embeddings(40), ['Artist A'] * 40, probe_limit=10)
        results = search_by_song('song0', limit=5)

        assert len(results) == 3
        assert index.query_ks == [26], 'the probed cells returned fewer rows than asked, a bigger k returns the same rows'

    def test_with_the_artist_limit_off_one_artist_can_fill_the_list(self, song_index):
        from tasks.clap_text_search import search_by_song

        song_index(_fan_embeddings(12), ['Artist A'] * 12)
        results = search_by_song('song0', limit=5, eliminate_duplicates=False)

        assert [r['item_id'] for r in results] == ['song1', 'song2', 'song3', 'song4', 'song5']

    @pytest.mark.parametrize('default, expected', [(True, 3), (False, 5)])
    def test_the_artist_limit_defaults_to_the_musicnn_setting(self, song_index, monkeypatch, default, expected):
        from tasks.clap_text_search import search_by_song

        monkeypatch.setattr('config.SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT', default)
        song_index(_fan_embeddings(12), ['Artist A'] * 12)

        assert len(search_by_song('song0', limit=5)) == expected

    def test_a_copy_of_the_seed_with_the_same_title_and_artist_is_not_returned(self, song_index):
        from tasks.clap_text_search import search_by_song

        titles = [f'Song {i}' for i in range(8)]
        authors = [f'Artist {i}' for i in range(8)]
        titles[2], authors[2] = ' song 0 ', 'ARTIST 0'
        song_index(_fan_embeddings(8), authors, titles)
        results = search_by_song('song0', limit=5)

        assert 'song2' not in [r['item_id'] for r in results]
        assert len(results) == 5

    def test_a_near_identical_copy_of_the_seed_is_not_returned(self, song_index):
        from tasks.clap_text_search import search_by_song

        embeddings = _fan_embeddings(8)
        embeddings[4] = embeddings[0]
        song_index(embeddings)
        results = search_by_song('song0', limit=5)

        assert 'song4' not in [r['item_id'] for r in results], 'an identical vector under another id is the seed again'
        assert len(results) == 5

    def test_the_search_keeps_its_index_snapshot_when_a_refresh_unloads_it(self, song_index, monkeypatch):
        from tasks import clap_text_search

        song_index(_fan_embeddings(8))
        original = clap_text_search._seed_clap_vector

        def read_then_unload(item_id):
            vector = original(item_id)
            clap_text_search._CLAP_INDEX_CACHE['index'] = None
            clap_text_search._CLAP_INDEX_CACHE['loaded'] = False
            return vector

        monkeypatch.setattr(clap_text_search, '_seed_clap_vector', read_then_unload)
        assert len(clap_text_search.search_by_song('song0', limit=5)) == 5

    def test_a_song_without_a_clap_embedding_is_reported_as_not_found(self, song_index):
        from tasks.clap_text_search import SongSeedNotFound, search_by_song

        song_index(_fan_embeddings(4))
        with pytest.raises(SongSeedNotFound):
            search_by_song('not-analysed', limit=3)

    @pytest.mark.parametrize(
        'blob',
        [bytes(16), np.zeros(512, dtype=np.float32).tobytes(), np.full(512, np.nan, dtype=np.float32).tobytes()],
        ids=['wrong-length', 'all-zero', 'not-finite'],
    )
    def test_a_corrupt_stored_embedding_is_reported_as_not_found(self, song_index, blob):
        from tasks.clap_text_search import SongSeedNotFound, search_by_song

        conn, _ = song_index(_fan_embeddings(4))
        conn.store['song0'] = blob
        with pytest.raises(SongSeedNotFound):
            search_by_song('song0', limit=3)

    def test_an_unloaded_index_is_reported_as_unavailable(self, song_index):
        from tasks.clap_text_search import ClapIndexUnavailable, search_by_song, _CLAP_INDEX_CACHE

        song_index(_fan_embeddings(4))
        _CLAP_INDEX_CACHE['loaded'] = False
        with pytest.raises(ClapIndexUnavailable):
            search_by_song('song0', limit=3)

    def test_disabled_or_zero_limit_returns_nothing(self, song_index, monkeypatch):
        from tasks.clap_text_search import search_by_song

        song_index(_fan_embeddings(4))
        assert search_by_song('song0', limit=0) == []

        monkeypatch.setattr('config.CLAP_ENABLED', False)
        assert search_by_song('song0', limit=3) == []


@pytest.fixture
def walk_spy(monkeypatch):
    from tasks import clap_text_search

    calls = []
    real = clap_text_search.execute_radius_walk

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(clap_text_search, 'execute_radius_walk', spy)
    return calls


class TestSearchBySongRadiusWalk:
    def test_the_radius_walk_picks_the_songs_from_a_musicnn_sized_pool(self, song_index, walk_spy):
        from tasks.clap_text_search import search_by_song

        _, index = song_index(_fan_embeddings(60))
        results = search_by_song('song0', limit=5, radius_similarity=True)

        assert len(results) == 5
        assert 'song0' not in [r['item_id'] for r in results]
        assert index.query_ks == [5 + max(20, 15) + 1], 'the pool is limit + max(20, 3 * limit) + 1, as MusiCNN asks'
        assert len(walk_spy) == 1
        assert walk_spy[0]['n'] == 5
        assert walk_spy[0]['eliminate_duplicates'] is True
        assert walk_spy[0]['max_songs_per_artist'] == 3
        assert 'song0' not in [c['item_id'] for c in walk_spy[0]['candidate_data']], 'the seed never enters the walk'

    def test_the_radius_walk_keeps_the_artist_limit(self, song_index, walk_spy):
        from tasks.clap_text_search import search_by_song

        authors = ['Artist A'] * 40 + [f'Artist {i}' for i in range(40, 60)]
        song_index(_fan_embeddings(60), authors)
        results = search_by_song('song0', limit=10, radius_similarity=True)

        assert sum(1 for r in results if r['author'] == 'Artist A') <= 3
        assert 'song0' not in [r['item_id'] for r in results]

    def test_with_the_artist_limit_off_the_walk_gets_no_cap(self, song_index, walk_spy):
        from tasks.clap_text_search import search_by_song

        song_index(_fan_embeddings(20), ['Artist A'] * 20)
        results = search_by_song('song0', limit=5, radius_similarity=True, eliminate_duplicates=False)

        assert walk_spy[0]['eliminate_duplicates'] is False
        assert len(results) == 5

    def test_a_copy_of_the_seed_never_reaches_the_walk(self, song_index, walk_spy):
        from tasks.clap_text_search import search_by_song

        embeddings = _fan_embeddings(12)
        embeddings[4] = embeddings[0]
        titles = [f'Song {i}' for i in range(12)]
        authors = [f'Artist {i}' for i in range(12)]
        titles[6], authors[6] = 'Song 0', 'Artist 0'
        song_index(embeddings, authors, titles)
        search_by_song('song0', limit=5, radius_similarity=True)

        walked = [c['item_id'] for c in walk_spy[0]['candidate_data']]
        assert 'song4' not in walked and 'song6' not in walked

    @pytest.mark.parametrize('default, walks', [(True, 1), (False, 0)])
    def test_the_radius_walk_defaults_to_the_musicnn_setting(self, song_index, walk_spy, monkeypatch, default, walks):
        from tasks.clap_text_search import search_by_song

        monkeypatch.setattr('config.SIMILARITY_RADIUS_DEFAULT', default)
        song_index(_fan_embeddings(12))
        assert len(search_by_song('song0', limit=5)) == 5
        assert len(walk_spy) == walks

    def test_a_failing_walk_answers_with_the_plain_order(self, song_index, monkeypatch):
        from tasks import clap_text_search

        song_index(_fan_embeddings(12))
        plain = clap_text_search.search_by_song('song0', limit=5, radius_similarity=False)

        def broken(**kwargs):
            raise RuntimeError('walk failed')

        monkeypatch.setattr(clap_text_search, 'execute_radius_walk', broken)
        assert clap_text_search.search_by_song('song0', limit=5, radius_similarity=True) == plain


class TestSongPickerFilter:
    def test_the_picker_offers_only_songs_with_a_clap_embedding_while_the_index_is_loaded(self, monkeypatch):
        from tasks import clap_text_search

        monkeypatch.setitem(clap_text_search._CLAP_CACHE, 'loaded', True)
        assert clap_text_search.picker_where() == clap_text_search.PICKER_WHERE
        assert 'clap_embedding' in clap_text_search.PICKER_WHERE[0]

        monkeypatch.setitem(clap_text_search._CLAP_CACHE, 'loaded', False)
        assert clap_text_search.picker_where() is None
