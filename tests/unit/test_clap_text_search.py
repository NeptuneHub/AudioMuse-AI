"""Unit tests for CLAP text search functionality

Tests cover core search logic, cache management, similarity calculations,
and result ranking with minimal external dependencies.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class DummyVoyagerIndex:
    """Simple fake Voyager index for unit tests."""
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def query(self, query_vector: np.ndarray, k: int):
        similarities = self.embeddings @ query_vector
        order = np.argsort(similarities)[::-1]
        top = order[:k]
        distances = (1.0 - similarities[top]).astype(np.float32)
        return list(top), distances


def setup_dummy_clap_index_cache(_CLAP_INDEX_CACHE, embeddings, item_ids):
    _CLAP_INDEX_CACHE['loaded'] = True
    _CLAP_INDEX_CACHE['index'] = DummyVoyagerIndex(embeddings)
    _CLAP_INDEX_CACHE['id_map'] = {i: item_ids[i] for i in range(len(item_ids))}
    _CLAP_INDEX_CACHE['reverse_id_map'] = {item_ids[i]: i for i in range(len(item_ids))}


def teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE):
    _CLAP_INDEX_CACHE['loaded'] = False
    _CLAP_INDEX_CACHE['index'] = None
    _CLAP_INDEX_CACHE['id_map'] = None
    _CLAP_INDEX_CACHE['reverse_id_map'] = None


class TestCacheStatsCalculation:
    """Tests for CLAP cache statistics calculation"""

    def test_get_cache_stats_when_not_loaded(self):
        """Test cache stats return empty structure when cache not loaded"""
        from tasks.clap_text_search import get_cache_stats, _CLAP_CACHE, _CLAP_INDEX_CACHE

        # Ensure cache is not loaded
        _CLAP_CACHE['loaded'] = False
        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)

        stats = get_cache_stats()

        assert stats['loaded'] is False
        assert stats['song_count'] == 0
        assert stats['embedding_dimension'] == 0
        assert stats['memory_mb'] == 0

    def test_get_cache_stats_with_loaded_cache(self):
        """Test cache stats calculation with loaded data"""
        from tasks.clap_text_search import get_cache_stats, _CLAP_CACHE, _CLAP_INDEX_CACHE

        # Simulate loaded CLAP index with 100 songs
        _CLAP_CACHE['loaded'] = True
        embeddings = np.random.rand(100, 512).astype(np.float32)
        item_ids = [f'song{i}' for i in range(100)]
        setup_dummy_clap_index_cache(_CLAP_INDEX_CACHE, embeddings, item_ids)

        stats = get_cache_stats()

        assert stats['loaded'] is True
        assert stats['song_count'] == 100
        assert stats['embedding_dimension'] == 512
        assert stats['memory_mb'] > 0  # Should calculate memory size

        # Cleanup
        _CLAP_CACHE['loaded'] = False
        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)

    def test_get_cache_stats_memory_calculation_accuracy(self):
        """Test that memory calculation is reasonable"""
        from tasks.clap_text_search import get_cache_stats, _CLAP_CACHE, _CLAP_INDEX_CACHE

        # Simulate loaded CLAP index with 10 songs
        _CLAP_CACHE['loaded'] = True
        embeddings = np.random.rand(10, 512).astype(np.float32)
        item_ids = [f'song{i}' for i in range(10)]
        setup_dummy_clap_index_cache(_CLAP_INDEX_CACHE, embeddings, item_ids)

        stats = get_cache_stats()

        # Should report a small non-zero memory size for the in-memory index
        assert 0.0 < stats['memory_mb'] < 1.0

        # Cleanup
        _CLAP_CACHE['loaded'] = False
        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)


class TestCacheStateChecking:
    """Tests for checking CLAP cache state"""

    def test_is_clap_cache_loaded_when_true(self):
        """Test cache loaded check returns True when loaded"""
        from tasks.clap_text_search import is_clap_cache_loaded, _CLAP_CACHE
        
        _CLAP_CACHE['loaded'] = True
        
        assert is_clap_cache_loaded() is True
        
        # Cleanup
        _CLAP_CACHE['loaded'] = False

    def test_is_clap_cache_loaded_when_false(self):
        """Test cache loaded check returns False when not loaded"""
        from tasks.clap_text_search import is_clap_cache_loaded, _CLAP_CACHE
        
        _CLAP_CACHE['loaded'] = False
        
        assert is_clap_cache_loaded() is False


class TestSimilarityCalculationLogic:
    """Tests for similarity calculation and ranking logic"""

    @patch('config.CLAP_ENABLED', True)
    @patch('tasks.clap_analyzer.get_text_embedding')
    def test_search_similarity_ranking_order(self, mock_get_embedding):
        """Test that search results are ranked by similarity (highest first)"""
        from tasks.clap_text_search import search_by_text, _CLAP_CACHE, _CLAP_INDEX_CACHE
        
        # Setup cache with known embeddings
        _CLAP_CACHE['loaded'] = True
        # Create 5 embeddings with varying similarity to query
        _CLAP_CACHE['embeddings'] = np.array([
            [1.0, 0.0, 0.0],  # Will have high similarity to query
            [0.0, 1.0, 0.0],  # Low similarity
            [0.9, 0.1, 0.0],  # High similarity
            [0.0, 0.0, 1.0],  # Low similarity
            [0.8, 0.2, 0.0],  # Medium-high similarity
        ], dtype=np.float32)
        
        # Normalize embeddings (required for cosine similarity)
        for i in range(len(_CLAP_CACHE['embeddings'])):
            norm = np.linalg.norm(_CLAP_CACHE['embeddings'][i])
            if norm > 0:
                _CLAP_CACHE['embeddings'][i] = _CLAP_CACHE['embeddings'][i] / norm
        
        _CLAP_CACHE['metadata'] = [
            {'item_id': f'song{i}', 'title': f'Song {i}', 'author': f'Artist {i}'}
            for i in range(5)
        ]
        _CLAP_CACHE['item_ids'] = [f'song{i}' for i in range(5)]
        setup_dummy_clap_index_cache(_CLAP_INDEX_CACHE, _CLAP_CACHE['embeddings'], _CLAP_CACHE['item_ids'])
        
        # Query embedding similar to first embedding
        query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        mock_get_embedding.return_value = query_embedding
        
        results = search_by_text("test query", limit=5)
        
        # Check that results are ordered by decreasing similarity
        assert len(results) == 5
        for i in range(len(results) - 1):
            assert results[i]['similarity'] >= results[i + 1]['similarity']
        
        # First result should be song0 (highest similarity)
        assert results[0]['item_id'] == 'song0'
        
        # Cleanup
        _CLAP_CACHE['loaded'] = False
        _CLAP_CACHE['embeddings'] = None
        _CLAP_CACHE['metadata'] = None
        _CLAP_CACHE['item_ids'] = None

    @patch('config.CLAP_ENABLED', True)
    @patch('tasks.clap_analyzer.get_text_embedding')
    def test_search_respects_limit_parameter(self, mock_get_embedding):
        """Test that search returns at most 'limit' results"""
        from tasks.clap_text_search import search_by_text, _CLAP_CACHE, _CLAP_INDEX_CACHE
        
        # Setup cache with 20 songs
        _CLAP_CACHE['loaded'] = True
        _CLAP_CACHE['embeddings'] = np.random.rand(20, 512).astype(np.float32)
        
        # Normalize
        for i in range(len(_CLAP_CACHE['embeddings'])):
            _CLAP_CACHE['embeddings'][i] /= np.linalg.norm(_CLAP_CACHE['embeddings'][i])
        
        _CLAP_CACHE['metadata'] = [
            {'item_id': f'song{i}', 'title': f'Song {i}', 'author': f'Artist {i}'}
            for i in range(20)
        ]
        _CLAP_CACHE['item_ids'] = [f'song{i}' for i in range(20)]
        setup_dummy_clap_index_cache(_CLAP_INDEX_CACHE, _CLAP_CACHE['embeddings'], _CLAP_CACHE['item_ids'])
        
        query_embedding = np.random.rand(512).astype(np.float32)
        query_embedding /= np.linalg.norm(query_embedding)
        mock_get_embedding.return_value = query_embedding
        
        # Request only 5 results
        results = search_by_text("test query", limit=5)
        
        assert len(results) == 5
        
        # Request 15 results
        results = search_by_text("test query", limit=15)
        
        assert len(results) == 15
        
        # Cleanup
        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)
        _CLAP_CACHE['loaded'] = False
        _CLAP_CACHE['embeddings'] = None
        _CLAP_CACHE['metadata'] = None
        _CLAP_CACHE['item_ids'] = None

    @patch('config.CLAP_ENABLED', True)
    @patch('tasks.clap_analyzer.get_text_embedding')
    def test_search_limit_larger_than_cache(self, mock_get_embedding):
        """Test search when limit exceeds available songs"""
        from tasks.clap_text_search import search_by_text, _CLAP_CACHE, _CLAP_INDEX_CACHE
        
        # Setup cache with only 5 songs
        _CLAP_CACHE['loaded'] = True
        _CLAP_CACHE['embeddings'] = np.random.rand(5, 512).astype(np.float32)
        
        for i in range(len(_CLAP_CACHE['embeddings'])):
            _CLAP_CACHE['embeddings'][i] /= np.linalg.norm(_CLAP_CACHE['embeddings'][i])
        
        _CLAP_CACHE['metadata'] = [
            {'item_id': f'song{i}', 'title': f'Song {i}', 'author': f'Artist {i}'}
            for i in range(5)
        ]
        _CLAP_CACHE['item_ids'] = [f'song{i}' for i in range(5)]
        setup_dummy_clap_index_cache(_CLAP_INDEX_CACHE, _CLAP_CACHE['embeddings'], _CLAP_CACHE['item_ids'])
        
        query_embedding = np.random.rand(512).astype(np.float32)
        query_embedding /= np.linalg.norm(query_embedding)
        mock_get_embedding.return_value = query_embedding
        
        # Request 100 results when only 5 exist
        results = search_by_text("test query", limit=100)
        
        # Should return all 5 available songs
        assert len(results) == 5
        
        # Cleanup
        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)
        _CLAP_CACHE['loaded'] = False
        _CLAP_CACHE['embeddings'] = None
        _CLAP_CACHE['metadata'] = None
        _CLAP_CACHE['item_ids'] = None


class TestSearchResultStructure:
    """Tests for search result data structure"""

    @patch('config.CLAP_ENABLED', True)
    @patch('tasks.clap_analyzer.get_text_embedding')
    def test_search_result_contains_required_fields(self, mock_get_embedding):
        """Test that each result contains all required fields"""
        from tasks.clap_text_search import search_by_text, _CLAP_CACHE, _CLAP_INDEX_CACHE
        
        _CLAP_CACHE['loaded'] = True
        _CLAP_CACHE['embeddings'] = np.random.rand(3, 512).astype(np.float32)
        
        for i in range(len(_CLAP_CACHE['embeddings'])):
            _CLAP_CACHE['embeddings'][i] /= np.linalg.norm(_CLAP_CACHE['embeddings'][i])
        
        _CLAP_CACHE['metadata'] = [
            {'item_id': 'song1', 'title': 'Test Song', 'author': 'Test Artist'},
            {'item_id': 'song2', 'title': 'Another Song', 'author': 'Another Artist'},
            {'item_id': 'song3', 'title': 'Third Song', 'author': 'Third Artist'},
        ]
        _CLAP_CACHE['item_ids'] = ['song1', 'song2', 'song3']
        setup_dummy_clap_index_cache(_CLAP_INDEX_CACHE, _CLAP_CACHE['embeddings'], _CLAP_CACHE['item_ids'])
        
        query_embedding = np.random.rand(512).astype(np.float32)
        query_embedding /= np.linalg.norm(query_embedding)
        mock_get_embedding.return_value = query_embedding
        
        results = search_by_text("test query", limit=3)
        
        # Each result should have all required fields
        for result in results:
            assert 'item_id' in result
            assert 'title' in result
            assert 'author' in result
            assert 'similarity' in result
            
            # Check types
            assert isinstance(result['item_id'], str)
            assert isinstance(result['title'], str)
            assert isinstance(result['author'], str)
            assert isinstance(result['similarity'], float)
            
            # Similarity should be between -1 and 1 (cosine similarity range)
            assert -1.0 <= result['similarity'] <= 1.0
        
        # Cleanup
        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)
        _CLAP_CACHE['loaded'] = False
        _CLAP_CACHE['embeddings'] = None
        _CLAP_CACHE['metadata'] = None
        _CLAP_CACHE['item_ids'] = None

    @patch('config.CLAP_ENABLED', True)
    @patch('tasks.clap_analyzer.get_text_embedding')
    def test_search_preserves_metadata_accurately(self, mock_get_embedding):
        """Test that search results preserve original metadata"""
        from tasks.clap_text_search import search_by_text, _CLAP_CACHE, _CLAP_INDEX_CACHE
        
        _CLAP_CACHE['loaded'] = True
        _CLAP_CACHE['embeddings'] = np.random.rand(2, 512).astype(np.float32)
        
        for i in range(len(_CLAP_CACHE['embeddings'])):
            _CLAP_CACHE['embeddings'][i] /= np.linalg.norm(_CLAP_CACHE['embeddings'][i])
        
        # Specific metadata to verify preservation
        _CLAP_CACHE['metadata'] = [
            {'item_id': 'abc123', 'title': 'Bohemian Rhapsody', 'author': 'Queen'},
            {'item_id': 'xyz789', 'title': 'Stairway to Heaven', 'author': 'Led Zeppelin'},
        ]
        _CLAP_CACHE['item_ids'] = ['abc123', 'xyz789']
        setup_dummy_clap_index_cache(_CLAP_INDEX_CACHE, _CLAP_CACHE['embeddings'], _CLAP_CACHE['item_ids'])
        
        query_embedding = np.random.rand(512).astype(np.float32)
        query_embedding /= np.linalg.norm(query_embedding)
        mock_get_embedding.return_value = query_embedding

        with patch('app_helper.get_score_data_by_ids') as mock_get_score_data:
            mock_get_score_data.return_value = [
                {'item_id': 'abc123', 'title': 'Bohemian Rhapsody', 'author': 'Queen'},
                {'item_id': 'xyz789', 'title': 'Stairway to Heaven', 'author': 'Led Zeppelin'},
            ]
            results = search_by_text("rock classics", limit=2)

        # Find each song in results
        song1 = next((r for r in results if r['item_id'] == 'abc123'), None)
        song2 = next((r for r in results if r['item_id'] == 'xyz789'), None)
        
        assert song1 is not None
        assert song1['title'] == 'Bohemian Rhapsody'
        assert song1['author'] == 'Queen'
        
        assert song2 is not None
        assert song2['title'] == 'Stairway to Heaven'
        assert song2['author'] == 'Led Zeppelin'
        
        # Cleanup
        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)
        _CLAP_CACHE['loaded'] = False
        _CLAP_CACHE['embeddings'] = None
        _CLAP_CACHE['metadata'] = None
        _CLAP_CACHE['item_ids'] = None


class TestSearchEdgeCases:
    """Tests for edge cases and error handling"""

    @patch('config.CLAP_ENABLED', False)
    def test_search_returns_empty_when_disabled(self):
        """Test that search returns empty list when CLAP is disabled"""
        from tasks.clap_text_search import search_by_text
        
        results = search_by_text("any query", limit=10)
        
        assert results == []

    @patch('config.CLAP_ENABLED', True)
    def test_search_returns_empty_when_cache_not_loaded(self):
        """Test that search returns empty list when cache is not loaded"""
        from tasks.clap_text_search import search_by_text, _CLAP_CACHE
        
        _CLAP_CACHE['loaded'] = False
        
        results = search_by_text("test query", limit=10)
        
        assert results == []

    @patch('config.CLAP_ENABLED', True)
    @patch('tasks.clap_analyzer.get_text_embedding')
    def test_search_handles_failed_text_embedding(self, mock_get_embedding):
        """Test that search handles text embedding failure gracefully"""
        from tasks.clap_text_search import search_by_text, _CLAP_CACHE, _CLAP_INDEX_CACHE
        
        _CLAP_CACHE['loaded'] = True
        _CLAP_CACHE['embeddings'] = np.random.rand(5, 512).astype(np.float32)
        _CLAP_CACHE['metadata'] = [{'item_id': f'song{i}', 'title': f'Song {i}', 'author': f'Artist {i}'} for i in range(5)]
        _CLAP_CACHE['item_ids'] = [f'song{i}' for i in range(5)]
        
        # Simulate embedding failure
        mock_get_embedding.return_value = None
        setup_dummy_clap_index_cache(_CLAP_INDEX_CACHE, _CLAP_CACHE['embeddings'], _CLAP_CACHE['item_ids'])
        
        results = search_by_text("test query", limit=10)
        
        # Should return empty list on embedding failure
        assert results == []
        
        # Cleanup
        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)
        _CLAP_CACHE['loaded'] = False
        _CLAP_CACHE['embeddings'] = None
        _CLAP_CACHE['metadata'] = None

    @patch('config.CLAP_ENABLED', True)
    @patch('tasks.clap_analyzer.get_text_embedding')
    def test_search_with_zero_limit(self, mock_get_embedding):
        """Test search with limit of 0"""
        from tasks.clap_text_search import search_by_text, _CLAP_CACHE, _CLAP_INDEX_CACHE
        
        _CLAP_CACHE['loaded'] = True
        _CLAP_CACHE['embeddings'] = np.random.rand(10, 512).astype(np.float32)
        
        for i in range(len(_CLAP_CACHE['embeddings'])):
            _CLAP_CACHE['embeddings'][i] /= np.linalg.norm(_CLAP_CACHE['embeddings'][i])
        
        _CLAP_CACHE['metadata'] = [{'item_id': f'song{i}', 'title': f'Song {i}', 'author': f'Artist {i}'} for i in range(10)]
        _CLAP_CACHE['item_ids'] = [f'song{i}' for i in range(10)]
        setup_dummy_clap_index_cache(_CLAP_INDEX_CACHE, _CLAP_CACHE['embeddings'], _CLAP_CACHE['item_ids'])
        
        query_embedding = np.random.rand(512).astype(np.float32)
        query_embedding /= np.linalg.norm(query_embedding)
        mock_get_embedding.return_value = query_embedding
        
        results = search_by_text("test query", limit=0)
        
        # Should return empty list
        assert results == []
        
        # Cleanup
        teardown_dummy_clap_index_cache(_CLAP_INDEX_CACHE)
        _CLAP_CACHE['loaded'] = False
        _CLAP_CACHE['embeddings'] = None
        _CLAP_CACHE['metadata'] = None
        _CLAP_CACHE['item_ids'] = None


class TestNumpyVectorizedOperations:
    """Tests for NumPy vectorized operations correctness"""

    def test_cosine_similarity_calculation(self):
        """Test that cosine similarity via dot product is correct"""
        # This tests the core similarity calculation logic
        # embeddings @ query_embedding should equal cosine similarity
        
        # Create normalized vectors
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])
        embedding3 = np.array([0.7071, 0.7071, 0.0])  # 45 degrees from embedding1
        
        embeddings = np.vstack([embedding1, embedding2, embedding3])
        query = np.array([1.0, 0.0, 0.0])
        
        # Calculate similarities
        similarities = embeddings @ query
        
        # Check results
        assert abs(similarities[0] - 1.0) < 0.0001  # Parallel vectors = 1.0
        assert abs(similarities[1] - 0.0) < 0.0001  # Orthogonal vectors = 0.0
        assert abs(similarities[2] - 0.7071) < 0.01  # 45 degrees ≈ 0.707

    def test_argsort_returns_descending_order(self):
        """Test that np.argsort with [::-1] gives descending order"""
        similarities = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
        
        # Get indices in descending order
        top_indices = np.argsort(similarities)[::-1]
        
        # Check order: should be [1, 3, 4, 2, 0] (0.9, 0.7, 0.5, 0.3, 0.1)
        assert top_indices[0] == 1  # 0.9
        assert top_indices[1] == 3  # 0.7
        assert top_indices[2] == 4  # 0.5
        assert top_indices[3] == 2  # 0.3
        assert top_indices[4] == 0  # 0.1
        
        # Verify values are in descending order
        sorted_values = similarities[top_indices]
        for i in range(len(sorted_values) - 1):
            assert sorted_values[i] >= sorted_values[i + 1]
