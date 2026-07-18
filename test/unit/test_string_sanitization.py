# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""String sanitization applied when persisting track and artist rows.

Verifies that the save paths clean caller-supplied strings before writing to the
database, patching get_db to capture the values actually bound to the query.

Main Features:
* NUL bytes and control characters are stripped from saved track strings
* None values are handled and over-long strings are truncated
* Whitespace is trimmed while unicode is preserved
* Artist mapping names/ids are sanitized and truncated the same way
"""

from unittest.mock import MagicMock, patch
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
            item_id,
            title,
            author,
            120.0,
            key,
            scale,
            moods,
            embedding,
            energy=0.5,
            other_features=other_features,
            album=album,
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
            "test_id", title, author, 120.0, "C", "major", {"happy": 0.5}, np.array([0.1, 0.2])
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
            "test_id",
            None,
            None,
            120.0,
            None,
            None,
            {"happy": 0.5},
            np.array([0.1, 0.2]),
            energy=None,
            other_features=None,
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
            "test_id",
            long_title,
            long_author,
            120.0,
            "C",
            "major",
            {"happy": 0.5},
            np.array([0.1, 0.2]),
            other_features=long_other,
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
            "test_id",
            "  Song Title  ",
            "  Artist Name  ",
            120.0,
            "  C  ",
            "  major  ",
            {"happy": 0.5},
            np.array([0.1, 0.2]),
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
            "test_id", title, author, 120.0, "C", "major", {"happy": 0.5}, np.array([0.1, 0.2])
        )

        call_args = mock_cur.execute.call_args_list[0]
        values = call_args[0][1]

        assert "歌曲" in values[1]
        assert "世界" in values[1]
        assert "艺术家" in values[2]
        assert "Артист" in values[2]
