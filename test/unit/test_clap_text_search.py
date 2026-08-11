# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""CLAP text-to-audio search over the cached IVF index in clap_text_search.

Covers cache stats/state reporting and search_by_text ranking against a dummy
IVF index, using cosine similarity of the text embedding.

Main Features:
* get_cache_stats and is_clap_cache_loaded reflect loaded/unloaded state
* Results are ranked by descending similarity, respect the limit, and carry the
  required item_id/title/author/similarity fields
* Edge cases: CLAP disabled, cache unloaded, failed text embedding, zero limit
  all return an empty list
"""

import numpy as np
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
                [0.9, 0.1, 0.0],
                [0.0, 0.0, 1.0],
                [0.8, 0.2, 0.0],
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

        with patch('app_helper.get_score_data_by_ids') as mock_get_score_data:
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
            {'item_id': 'abc123', 'title': 'Bohemian Rhapsody', 'author': 'Queen'},
            {'item_id': 'xyz789', 'title': 'Stairway to Heaven', 'author': 'Led Zeppelin'},
        ]
        _CLAP_CACHE['item_ids'] = ['abc123', 'xyz789']
        setup_dummy_clap_index_cache(
            _CLAP_INDEX_CACHE, _CLAP_CACHE['embeddings'], _CLAP_CACHE['item_ids']
        )

        query_embedding = np.random.rand(512).astype(np.float32)
        query_embedding /= np.linalg.norm(query_embedding)
        mock_get_embedding.return_value = query_embedding

        with patch('app_helper.get_score_data_by_ids') as mock_get_score_data:
            mock_get_score_data.return_value = [
                {'item_id': 'abc123', 'title': 'Bohemian Rhapsody', 'author': 'Queen'},
                {'item_id': 'xyz789', 'title': 'Stairway to Heaven', 'author': 'Led Zeppelin'},
            ]
            results = search_by_text("rock classics", limit=2)

        song1 = next((r for r in results if r['item_id'] == 'abc123'), None)
        song2 = next((r for r in results if r['item_id'] == 'xyz789'), None)

        assert song1 is not None
        assert song1['title'] == 'Bohemian Rhapsody'
        assert song1['author'] == 'Queen'

        assert song2 is not None
        assert song2['title'] == 'Stairway to Heaven'
        assert song2['author'] == 'Led Zeppelin'

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
