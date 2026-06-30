
from unittest.mock import Mock, MagicMock, patch
import numpy as np


class TestSaveTrackStringSanitization:

    @patch('database.get_db')
    def test_sanitize_removes_nul_bytes(self, mock_get_db):
        from app_helper import save_track_analysis_and_embedding

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_get_db.return_value = mock_conn

        item_id = "test_id"
        title = "Song\x00Title"
        author = "Artist\x00Name"
        album = "Album\x00Name"
        key = "C\x00"
        scale = "major\x00"
        other_features = "feature1:0.5\x00,feature2:0.8"
        moods = {"happy": 0.8, "energetic": 0.6}
        embedding = np.array([0.1, 0.2, 0.3])

        save_track_analysis_and_embedding(
            item_id, title, author, 120.0, key, scale,
            moods, embedding, energy=0.5, other_features=other_features, album=album
        )

        call_args = mock_cur.execute.call_args_list[0]
        values = call_args[0][1]

        assert "\x00" not in values[1]
        assert "\x00" not in values[2]
        assert "\x00" not in values[4]
        assert "\x00" not in values[5]
        assert "\x00" not in values[8]
        assert "\x00" not in values[9]

        assert values[1] == "SongTitle"
        assert values[2] == "ArtistName"
        assert values[9] == "AlbumName"

    @patch('database.get_db')
    def test_sanitize_removes_control_characters(self, mock_get_db):
        from app_helper import save_track_analysis_and_embedding

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_get_db.return_value = mock_conn

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

    @patch('database.get_db')
    def test_sanitize_handles_none_values(self, mock_get_db):
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

        assert values[1] is None
        assert values[2] is None
        assert values[4] is None
        assert values[5] is None
        assert values[8] is None

    @patch('database.get_db')
    def test_sanitize_truncates_long_strings(self, mock_get_db):
        from app_helper import save_track_analysis_and_embedding

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_get_db.return_value = mock_conn

        long_title = "A" * 600
        long_author = "B" * 300
        long_other = "C" * 2500

        save_track_analysis_and_embedding(
            "test_id", long_title, long_author, 120.0, "C", "major",
            {"happy": 0.5}, np.array([0.1, 0.2]),
            other_features=long_other
        )

        call_args = mock_cur.execute.call_args_list[0]
        values = call_args[0][1]

        assert len(values[1]) == 500
        assert len(values[2]) == 200
        assert len(values[8]) == 2000

    @patch('database.get_db')
    def test_sanitize_strips_whitespace(self, mock_get_db):
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

    @patch('database.get_db')
    def test_sanitize_preserves_unicode(self, mock_get_db):
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

        assert "歌曲" in values[1]
        assert "世界" in values[1]
        assert "艺术家" in values[2]
        assert "Артист" in values[2]


class TestArtistMappingSanitization:

    @patch('app_helper_artist.get_db')
    def test_artist_sanitize_removes_nul_bytes(self, mock_get_db):
        from app_helper_artist import upsert_artist_mapping

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_get_db.return_value = mock_conn

        artist_name = "Artist\x00Name"
        artist_id = "artist_123"

        upsert_artist_mapping(artist_name, artist_id)

        call_args = mock_cur.execute.call_args
        values = call_args[0][1]

        assert "\x00" not in values[0]
        assert values[0] == "ArtistName"

    @patch('app_helper_artist.get_db')
    def test_artist_sanitize_truncates_long_names(self, mock_get_db):
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

        assert len(values[0]) == 500

    @patch('app_helper_artist.get_db')
    def test_artist_sanitize_handles_empty_inputs(self, mock_get_db):
        from app_helper_artist import upsert_artist_mapping

        mock_get_db.return_value = MagicMock()

        upsert_artist_mapping(None, "artist_123")
        upsert_artist_mapping("Artist", None)
        upsert_artist_mapping("", "artist_123")
        upsert_artist_mapping("Artist", "")

        assert not mock_get_db.called

    @patch('app_helper_artist.get_db')
    def test_artist_id_truncation(self, mock_get_db):
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

        assert len(values[1]) == 200
