"""Tests for playlist curator deduplication logic."""
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Pre-register mock for tasks.voyager_manager to avoid tasks/__init__.py import chain
if 'tasks' not in sys.modules:
    sys.modules['tasks'] = MagicMock()
if 'tasks.voyager_manager' not in sys.modules:
    _mock_voyager = MagicMock()
    sys.modules['tasks.voyager_manager'] = _mock_voyager


def _make_vector(seed, dim=200):
    """Create a deterministic vector from a seed."""
    rng = np.random.RandomState(seed)
    return rng.randn(dim).astype(np.float32)


def _make_near_duplicate(base_vec, noise=0.001):
    """Create a vector very close to base_vec (cosine distance < 0.01)."""
    rng = np.random.RandomState(42)
    return (base_vec + rng.randn(*base_vec.shape) * noise).astype(np.float32)


class TestFindDuplicateGroups:
    """Tests for _find_duplicate_groups()."""

    @patch('app_playlist_curator.get_score_data_by_ids')
    @patch('app_playlist_curator.get_vector_by_id')
    def test_finds_near_duplicates(self, mock_get_vec, mock_get_score):
        from app_playlist_curator import _find_duplicate_groups

        base = _make_vector(1)
        dup = _make_near_duplicate(base)
        different = _make_vector(99)

        mock_get_vec.side_effect = lambda tid: {
            1: base, 2: dup, 3: different
        }.get(tid)

        mock_get_score.return_value = [
            {'track_id': 1, 'item_id': '1', 'title': 'Song A', 'author': 'Artist',
             'album': 'Album 1', 'album_artist': 'Artist', 'year': 2020, 'rating': 4},
            {'track_id': 2, 'item_id': '2', 'title': 'Song A', 'author': 'Artist',
             'album': 'Album 2', 'album_artist': None, 'year': None, 'rating': None},
            {'track_id': 3, 'item_id': '3', 'title': 'Different', 'author': 'Other',
             'album': 'Album 3', 'album_artist': 'Other', 'year': 2021, 'rating': 5},
        ]

        result = _find_duplicate_groups([1, 2, 3], threshold=0.05)

        assert len(result['groups']) == 1
        group = result['groups'][0]
        assert len(group['tracks']) == 2
        # Track 1 should be keeper (has rating + metadata)
        assert group['tracks'][0]['item_id'] == '1'
        assert result['total_groups'] == 1
        assert result['total_duplicate_tracks'] == 2

    @patch('app_playlist_curator.get_score_data_by_ids')
    @patch('app_playlist_curator.get_vector_by_id')
    def test_no_duplicates(self, mock_get_vec, mock_get_score):
        from app_playlist_curator import _find_duplicate_groups

        mock_get_vec.side_effect = lambda tid: _make_vector(tid * 100)
        mock_get_score.return_value = [
            {'track_id': 1, 'item_id': '1', 'title': 'A', 'author': 'X',
             'album': 'A1', 'album_artist': 'X', 'year': 2020, 'rating': 3},
            {'track_id': 2, 'item_id': '2', 'title': 'B', 'author': 'Y',
             'album': 'B1', 'album_artist': 'Y', 'year': 2021, 'rating': 4},
        ]

        result = _find_duplicate_groups([1, 2], threshold=0.05)
        assert len(result['groups']) == 0
        assert result['total_groups'] == 0
        assert result['total_duplicate_tracks'] == 0

    @patch('app_playlist_curator.get_score_data_by_ids')
    @patch('app_playlist_curator.get_vector_by_id')
    def test_tracks_without_embeddings_skipped(self, mock_get_vec, mock_get_score):
        from app_playlist_curator import _find_duplicate_groups

        mock_get_vec.side_effect = lambda tid: None  # no embeddings
        mock_get_score.return_value = []

        result = _find_duplicate_groups([1, 2, 3], threshold=0.05)
        assert len(result['groups']) == 0

    @patch('app_playlist_curator.get_score_data_by_ids')
    @patch('app_playlist_curator.get_vector_by_id')
    def test_keeper_scoring_prefers_rated_with_metadata(self, mock_get_vec, mock_get_score):
        from app_playlist_curator import _find_duplicate_groups

        base = _make_vector(1)

        mock_get_vec.side_effect = lambda tid: {
            1: base,
            2: _make_near_duplicate(base, noise=0.0005),
            3: _make_near_duplicate(base, noise=0.0008),
        }.get(tid)

        mock_get_score.return_value = [
            {'track_id': 1, 'item_id': '1', 'title': 'Song', 'author': 'A',
             'album': None, 'album_artist': None, 'year': None, 'rating': None},
            {'track_id': 2, 'item_id': '2', 'title': 'Song', 'author': 'A',
             'album': 'Best Of', 'album_artist': 'A', 'year': 2020, 'rating': 5},
            {'track_id': 3, 'item_id': '3', 'title': 'Song', 'author': 'A',
             'album': 'Hits', 'album_artist': 'A', 'year': 2019, 'rating': 3},
        ]

        result = _find_duplicate_groups([1, 2, 3], threshold=0.05)
        assert len(result['groups']) == 1
        # Track 2 should be keeper: rating=5, full metadata
        assert result['groups'][0]['tracks'][0]['item_id'] == '2'


@pytest.fixture
def curator_client():
    """Create a minimal Flask app with only the curator blueprint for testing routes."""
    from flask import Flask
    from app_playlist_curator import playlist_curator_bp
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(playlist_curator_bp)
    with app.test_client() as client:
        yield client


class TestFindDuplicatesRoute:
    """Tests for POST /api/curator/find_duplicates."""

    @patch('app_playlist_curator._find_duplicate_groups')
    def test_returns_groups(self, mock_find, curator_client):
        mock_find.return_value = {
            "groups": [{"tracks": [{"item_id": "1", "score": 5.0}, {"item_id": "2", "score": 2.0}]}],
            "total_groups": 1,
            "total_duplicate_tracks": 2
        }

        resp = curator_client.post('/api/curator/find_duplicates',
                                   json={"track_ids": [1, 2, 3], "threshold": 0.05},
                                   content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['total_groups'] == 1

    def test_rejects_too_many_tracks(self, curator_client):
        resp = curator_client.post('/api/curator/find_duplicates',
                                   json={"track_ids": list(range(2001)), "threshold": 0.05},
                                   content_type='application/json')
        assert resp.status_code == 400

    @patch('app_playlist_curator._find_duplicate_groups')
    def test_clamps_threshold(self, mock_find, curator_client):
        mock_find.return_value = {"groups": [], "total_groups": 0, "total_duplicate_tracks": 0}
        resp = curator_client.post('/api/curator/find_duplicates',
                                   json={"track_ids": [1, 2], "threshold": 999},
                                   content_type='application/json')
        assert resp.status_code == 200
        assert mock_find.call_args[1]['threshold'] <= 0.3


class TestFindDuplicateGroupsEdgeCases:

    @patch('app_playlist_curator.get_score_data_by_ids')
    @patch('app_playlist_curator.get_vector_by_id')
    def test_single_track_returns_no_groups(self, mock_get_vec, mock_get_score):
        from app_playlist_curator import _find_duplicate_groups
        mock_get_vec.return_value = _make_vector(1)
        result = _find_duplicate_groups([1], threshold=0.05)
        assert len(result['groups']) == 0

    @patch('app_playlist_curator.get_score_data_by_ids')
    @patch('app_playlist_curator.get_vector_by_id')
    def test_empty_list_returns_no_groups(self, mock_get_vec, mock_get_score):
        from app_playlist_curator import _find_duplicate_groups
        result = _find_duplicate_groups([], threshold=0.05)
        assert len(result['groups']) == 0

    @patch('app_playlist_curator.get_score_data_by_ids')
    @patch('app_playlist_curator.get_vector_by_id')
    def test_strict_threshold_separates_remasters(self, mock_get_vec, mock_get_score):
        from app_playlist_curator import _find_duplicate_groups

        base = _make_vector(1)
        remaster = _make_near_duplicate(base, noise=0.25)

        mock_get_vec.side_effect = lambda tid: {1: base, 2: remaster}.get(tid)
        mock_get_score.return_value = [
            {'track_id': 1, 'item_id': '1', 'title': 'A', 'author': 'X',
             'album': 'Orig', 'album_artist': 'X', 'year': 2000, 'rating': 3},
            {'track_id': 2, 'item_id': '2', 'title': 'A Remaster', 'author': 'X',
             'album': 'Remaster', 'album_artist': 'X', 'year': 2020, 'rating': 4},
        ]

        strict = _find_duplicate_groups([1, 2], threshold=0.02)
        assert len(strict['groups']) == 0

        loose = _find_duplicate_groups([1, 2], threshold=0.15)
        assert len(loose['groups']) == 1
