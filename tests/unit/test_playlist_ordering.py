"""
Tests for tasks/playlist_ordering.py — Greedy nearest-neighbor playlist ordering.

Tests verify:
- Composite distance calculation (tempo, energy, key weighting)
- Circle of Fifths key distance computation
- Greedy nearest-neighbor algorithm
- Energy arc reshaping
- Handling of songs missing from database
"""
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock

from tests.conftest import _import_module, make_dict_row, make_mock_connection


def _load_playlist_ordering():
    """Load playlist_ordering module via importlib to bypass tasks/__init__.py."""
    # Pre-register mock for tasks.mcp_server so order_playlist's lazy import
    # doesn't trigger the heavy tasks/__init__.py chain
    if 'tasks.mcp_server' not in sys.modules:
        sys.modules['tasks.mcp_server'] = MagicMock()
    return _import_module('tasks.playlist_ordering', 'tasks/playlist_ordering.py')


class TestKeyDistance:
    """Test _key_distance() function — Circle of Fifths distance."""

    def test_identical_keys_same_scale(self):
        """Same key, same scale → distance = 0."""
        mod = _load_playlist_ordering()
        dist = mod._key_distance('C', 'major', 'C', 'major')
        assert dist == 0.0

    def test_adjacent_keys_without_scale_bonus(self):
        """C→G is 1 step / 6 max ≈ 0.167."""
        mod = _load_playlist_ordering()
        dist = mod._key_distance('C', None, 'G', None)
        assert abs(dist - 1/6) < 0.01

    def test_missing_key_returns_neutral(self):
        """Missing key → return 0.5 (neutral)."""
        mod = _load_playlist_ordering()
        dist = mod._key_distance(None, None, 'C', None)
        assert dist == 0.5

    def test_unknown_key_returns_neutral(self):
        """Unknown key → return 0.5 (neutral)."""
        mod = _load_playlist_ordering()
        dist = mod._key_distance('C', None, 'XYZ', None)
        assert dist == 0.5

    def test_case_insensitive(self):
        """Keys are uppercased → 'c' should match 'C'."""
        mod = _load_playlist_ordering()
        dist1 = mod._key_distance('C', None, 'G', None)
        dist2 = mod._key_distance('c', None, 'g', None)
        assert abs(dist1 - dist2) < 0.01


class TestCompositeDistance:
    """Test _composite_distance() function — Weighted combination."""

    def test_identical_songs(self):
        """Same song data → distance = 0."""
        mod = _load_playlist_ordering()
        song = {'tempo': 120, 'energy': 0.08, 'key': 'C', 'scale': 'major'}
        dist = mod._composite_distance(song, song)
        assert dist == 0.0

    def test_tempo_difference(self):
        """Different tempos → distance reflects tempo weight (0.35)."""
        mod = _load_playlist_ordering()
        song1 = {'tempo': 80, 'energy': 0.05, 'key': 'C', 'scale': 'major'}
        song2 = {'tempo': 160, 'energy': 0.05, 'key': 'C', 'scale': 'major'}
        # Tempo diff: |160-80|/80 = 1.0, capped at 1.0
        # Dist: 0.35*1.0 = 0.35
        dist = mod._composite_distance(song1, song2)
        assert abs(dist - 0.35) < 0.01

    def test_energy_capped_at_one(self):
        """Large energy diff > 0.14 → capped at 1.0."""
        mod = _load_playlist_ordering()
        song1 = {'tempo': 100, 'energy': 0.01, 'key': 'C', 'scale': None}
        song2 = {'tempo': 100, 'energy': 0.15, 'key': 'C', 'scale': None}
        dist = mod._composite_distance(song1, song2)
        assert abs(dist - 0.35) < 0.01  # energy weight = 0.35

    def test_missing_values_as_zero(self):
        """Missing tempo/energy → treated as 0."""
        mod = _load_playlist_ordering()
        song1 = {'tempo': None, 'energy': None, 'key': 'C', 'scale': None}
        song2 = {'tempo': 100, 'energy': 0.10, 'key': 'C', 'scale': None}
        dist = mod._composite_distance(song1, song2)
        assert dist > 0


def _make_db_rows(song_data):
    """Build mock DictRow list from {track_id: {tempo, energy, key, scale}} dict."""
    rows = []
    for tid, data in song_data.items():
        row = dict(data)
        row['track_id'] = tid
        rows.append(make_dict_row(row))
    return rows


class TestOrderPlaylist:
    """Test order_playlist() function — Main greedy algorithm."""

    def test_single_song_unchanged(self):
        """Single song → return unchanged (no DB call needed)."""
        mod = _load_playlist_ordering()
        result = mod.order_playlist([1])
        assert result == [1]

    def test_two_songs_unchanged(self):
        """Two songs → no reordering (len <= 2, no DB call)."""
        mod = _load_playlist_ordering()
        result = mod.order_playlist([1, 2])
        assert result == [1, 2]

    def test_empty_input(self):
        """Empty input → empty output."""
        mod = _load_playlist_ordering()
        result = mod.order_playlist([])
        assert result == []

    def test_all_input_songs_in_output(self):
        """All input track_ids appear in output, none lost or duplicated."""
        mod = _load_playlist_ordering()
        song_data = {
            i: {'tempo': 80 + i * 10, 'energy': 0.02 + i * 0.02, 'key': 'C', 'scale': 'major'}
            for i in range(10)
        }
        ids = list(song_data.keys())
        rows = _make_db_rows(song_data)
        cur = MagicMock()
        cur.fetchall.return_value = rows
        conn = make_mock_connection(cur)
        sys.modules['tasks.mcp_server'].get_db_connection = Mock(return_value=conn)
        result = mod.order_playlist(ids)
        assert set(result) == set(ids) and len(result) == len(ids)

    def test_unorderable_songs_appended_at_end(self):
        """Songs not in DB are appended at the end."""
        mod = _load_playlist_ordering()
        song_data = {
            i: {'tempo': 100 + i * 10, 'energy': 0.05 + i * 0.02, 'key': 'C', 'scale': 'major'}
            for i in range(3)
        }
        ids = [0, 1, 2, 999]
        rows = _make_db_rows(song_data)
        cur = MagicMock()
        cur.fetchall.return_value = rows
        conn = make_mock_connection(cur)
        sys.modules['tasks.mcp_server'].get_db_connection = Mock(return_value=conn)
        result = mod.order_playlist(ids)
        assert result[-1] == 999
        assert set(result) == set(ids)

    def test_no_db_rows_returns_original_order(self):
        """When DB returns no rows, return input unchanged."""
        mod = _load_playlist_ordering()
        ids = [10, 20, 30]
        cur = MagicMock()
        cur.fetchall.return_value = []
        conn = make_mock_connection(cur)
        sys.modules['tasks.mcp_server'].get_db_connection = Mock(return_value=conn)
        assert mod.order_playlist(ids) == ids

    def test_only_two_orderable_returns_original(self):
        """When only 2 songs have DB data, return input unchanged."""
        mod = _load_playlist_ordering()
        song_data = {
            10: {'tempo': 100, 'energy': 0.05, 'key': 'C', 'scale': 'major'},
            20: {'tempo': 110, 'energy': 0.06, 'key': 'G', 'scale': 'major'},
        }
        ids = [10, 20, 999]
        rows = _make_db_rows(song_data)
        cur = MagicMock()
        cur.fetchall.return_value = rows
        conn = make_mock_connection(cur)
        sys.modules['tasks.mcp_server'].get_db_connection = Mock(return_value=conn)
        assert mod.order_playlist(ids) == ids
