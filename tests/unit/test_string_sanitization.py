"""
Unit tests for string sanitization in app_helper.py
Tests the enhanced string sanitization to prevent database errors.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np


class TestSaveTrackStringSanitization:
    """Test string sanitization in save_track_analysis_and_embedding."""
    
    @patch('app_helper.get_db')
    def test_sanitize_removes_nul_bytes(self, mock_get_db):
        """Test that NUL bytes are removed from all string fields."""
        from app_helper import save_track_analysis_and_embedding
        
        # Setup mock database
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_get_db.return_value = mock_conn
        
        # Test data with NUL bytes
        item_id = "test_id"
        title = "Song\x00Title"
        author = "Artist\x00Name"
        key = "C\x00"
        scale = "major\x00"
        other_features = "feature1:0.5\x00,feature2:0.8"
        moods = {"happy": 0.8, "energetic": 0.6}
        embedding = np.array([0.1, 0.2, 0.3])
        
        # Call the function
        save_track_analysis_and_embedding(
            item_id, title, author, 120.0, key, scale, 
            moods, embedding, energy=0.5, other_features=other_features
        )
        
        # Verify that NUL bytes were removed from the database call
        call_args = mock_cur.execute.call_args_list[0]
        values = call_args[0][1]  # Second argument contains the values tuple
        
        assert "\x00" not in values[1]  # title
        assert "\x00" not in values[2]  # author
        assert "\x00" not in values[4]  # key
        assert "\x00" not in values[5]  # scale
        assert "\x00" not in values[8]  # other_features
        
        # Verify the sanitized values
        assert values[1] == "SongTitle"
        assert values[2] == "ArtistName"
    
    @patch('app_helper.get_db')
    def test_sanitize_removes_control_characters(self, mock_get_db):
        """Test that control characters are removed."""
        from app_helper import save_track_analysis_and_embedding
        
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_get_db.return_value = mock_conn
        
        # Test data with control characters
        title = "Song\x01\x02\x03Title"
        author = "Artist\x1fName"
        
        save_track_analysis_and_embedding(
            "test_id", title, author, 120.0, "C", "major",
            {"happy": 0.5}, np.array([0.1, 0.2])
        )
        
        call_args = mock_cur.execute.call_args_list[0]
        values = call_args[0][1]
        
        assert values[1] == "SongTitle"
        assert values[2] == "ArtistName"
    
    @patch('app_helper.get_db')
    def test_sanitize_handles_none_values(self, mock_get_db):
        """Test that None values are handled correctly."""
        from app_helper import save_track_analysis_and_embedding
        
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_get_db.return_value = mock_conn
        
        save_track_analysis_and_embedding(
            "test_id", None, None, 120.0, None, None,
            {"happy": 0.5}, np.array([0.1, 0.2]), 
            energy=None, other_features=None
        )
        
        call_args = mock_cur.execute.call_args_list[0]
        values = call_args[0][1]
        
        # None values should remain None
        assert values[1] is None  # title
        assert values[2] is None  # author
        assert values[4] is None  # key
        assert values[5] is None  # scale
        assert values[8] is None  # other_features
    
    @patch('app_helper.get_db')
    def test_sanitize_truncates_long_strings(self, mock_get_db):
        """Test that overly long strings are truncated."""
        from app_helper import save_track_analysis_and_embedding
        
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_get_db.return_value = mock_conn
        
        # Create strings longer than the limits
        long_title = "A" * 600  # Should be truncated to 500
        long_author = "B" * 300  # Should be truncated to 200
        long_other = "C" * 2500  # Should be truncated to 2000
        
        save_track_analysis_and_embedding(
            "test_id", long_title, long_author, 120.0, "C", "major",
            {"happy": 0.5}, np.array([0.1, 0.2]), 
            other_features=long_other
        )
        
        call_args = mock_cur.execute.call_args_list[0]
        values = call_args[0][1]
        
        # Check truncation
        assert len(values[1]) == 500  # title
        assert len(values[2]) == 200  # author
        assert len(values[8]) == 2000  # other_features
    
    @patch('app_helper.get_db')
    def test_sanitize_strips_whitespace(self, mock_get_db):
        """Test that leading/trailing whitespace is stripped."""
        from app_helper import save_track_analysis_and_embedding
        
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_get_db.return_value = mock_conn
        
        save_track_analysis_and_embedding(
            "test_id", "  Song Title  ", "  Artist Name  ", 
            120.0, "  C  ", "  major  ",
            {"happy": 0.5}, np.array([0.1, 0.2])
        )
        
        call_args = mock_cur.execute.call_args_list[0]
        values = call_args[0][1]
        
        assert values[1] == "Song Title"
        assert values[2] == "Artist Name"
        assert values[4] == "C"
        assert values[5] == "major"
    
    @patch('app_helper.get_db')
    def test_sanitize_preserves_unicode(self, mock_get_db):
        """Test that Unicode characters are preserved."""
        from app_helper import save_track_analysis_and_embedding
        
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_get_db.return_value = mock_conn
        
        title = "歌曲 - Song 世界"
        author = "艺术家 Артист"
        
        save_track_analysis_and_embedding(
            "test_id", title, author, 120.0, "C", "major",
            {"happy": 0.5}, np.array([0.1, 0.2])
        )
        
        call_args = mock_cur.execute.call_args_list[0]
        values = call_args[0][1]
        
        # Unicode should be preserved
        assert "歌曲" in values[1]
        assert "世界" in values[1]
        assert "艺术家" in values[2]
        assert "Артист" in values[2]


class TestArtistMappingSanitization:
    """Test string sanitization in upsert_artist_mapping."""
    
    @patch('app_helper_artist.get_db')
    def test_artist_sanitize_removes_nul_bytes(self, mock_get_db):
        """Test that NUL bytes are removed from artist name."""
        from app_helper_artist import upsert_artist_mapping
        
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_get_db.return_value = mock_conn
        
        artist_name = "Artist\x00Name"
        artist_id = "artist_123"
        
        upsert_artist_mapping(artist_name, artist_id)
        
        # Verify NUL byte was removed
        call_args = mock_cur.execute.call_args
        values = call_args[0][1]
        
        assert "\x00" not in values[0]  # artist_name
        assert values[0] == "ArtistName"
    
    @patch('app_helper_artist.get_db')
    def test_artist_sanitize_truncates_long_names(self, mock_get_db):
        """Test that long artist names are truncated."""
        from app_helper_artist import upsert_artist_mapping
        
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_get_db.return_value = mock_conn
        
        long_name = "A" * 600
        artist_id = "artist_123"
        
        upsert_artist_mapping(long_name, artist_id)
        
        call_args = mock_cur.execute.call_args
        values = call_args[0][1]
        
        assert len(values[0]) == 500  # Should be truncated to 500
    
    @patch('app_helper_artist.get_db')
    def test_artist_sanitize_handles_empty_inputs(self, mock_get_db):
        """Test that empty inputs are handled correctly."""
        from app_helper_artist import upsert_artist_mapping
        
        mock_get_db.return_value = MagicMock()
        
        # Should return early without database call
        upsert_artist_mapping(None, "artist_123")
        upsert_artist_mapping("Artist", None)
        upsert_artist_mapping("", "artist_123")
        upsert_artist_mapping("Artist", "")
        
        # Database should not be called
        assert not mock_get_db.called
    
    @patch('app_helper_artist.get_db')
    def test_artist_id_truncation(self, mock_get_db):
        """Test that long artist IDs are truncated."""
        from app_helper_artist import upsert_artist_mapping
        
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_get_db.return_value = mock_conn
        
        artist_name = "Artist"
        long_id = "id_" + "X" * 300
        
        upsert_artist_mapping(artist_name, long_id)
        
        call_args = mock_cur.execute.call_args
        values = call_args[0][1]
        
        assert len(values[1]) == 200  # Should be truncated to 200
