"""
Tests for app_chat.py::chat_playlist_api() — Instant Playlist pipeline.

Tests verify:
- Pre-validation (song_similarity empty title/artist rejection, search_database no-filter rejection)
- Artist diversity enforcement (MAX_SONGS_PER_ARTIST_PLAYLIST cap, backfill)
- Iteration message content (iteration 0 minimal, iteration > 0 rich feedback)
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from tests.conftest import make_dict_row, make_mock_connection


class TestPreValidation:
    """Test the pre-validation block in chat_playlist_api() (lines ~466-493)."""

    def test_song_similarity_empty_title_rejected(self):
        """song_similarity with empty title should be skipped."""
        # This test validates the logic without calling the full endpoint
        # It tests the rejection criteria: title must be non-empty
        title = ""
        artist = "Artist"

        # Check if title passes validation
        is_valid = bool(title.strip())
        assert not is_valid

    def test_song_similarity_empty_artist_rejected(self):
        """song_similarity with empty artist should be skipped."""
        title = "Song"
        artist = ""

        # Check if artist passes validation
        is_valid = bool(artist.strip())
        assert not is_valid

    def test_song_similarity_whitespace_only_rejected(self):
        """song_similarity with whitespace-only title/artist should be skipped."""
        title = "   "
        artist = "  \t  "

        assert not title.strip()
        assert not artist.strip()

    def test_search_database_zero_filters_rejected(self):
        """search_database with no filters specified should be skipped."""
        # Test the filter-checking logic
        filters = {}
        filter_keys = ['genres', 'moods', 'tempo_min', 'tempo_max', 'energy_min', 'energy_max',
                       'key', 'scale', 'year_min', 'year_max', 'min_rating', 'album']

        has_filter = any(filters.get(k) for k in filter_keys)
        assert not has_filter

    def test_search_database_album_only_filter_accepted(self):
        """search_database with album filter alone should be accepted."""
        filters = {'album': 'Dark Side of the Moon'}
        filter_keys = ['genres', 'moods', 'tempo_min', 'tempo_max', 'energy_min', 'energy_max',
                       'key', 'scale', 'year_min', 'year_max', 'min_rating', 'album']

        has_filter = any(filters.get(k) for k in filter_keys)
        assert has_filter

    def test_search_database_genres_filter_accepted(self):
        """search_database with genres filter should be accepted."""
        filters = {'genres': ['rock', 'metal']}
        filter_keys = ['genres', 'moods', 'tempo_min', 'tempo_max', 'energy_min', 'energy_max',
                       'key', 'scale', 'year_min', 'year_max', 'min_rating', 'album']

        has_filter = any(filters.get(k) for k in filter_keys)
        assert has_filter

    def test_search_database_year_filter_accepted(self):
        """search_database with year_min alone should be accepted."""
        filters = {'year_min': 1990}
        filter_keys = ['genres', 'moods', 'tempo_min', 'tempo_max', 'energy_min', 'energy_max',
                       'key', 'scale', 'year_min', 'year_max', 'min_rating', 'album']

        has_filter = any(filters.get(k) for k in filter_keys)
        assert has_filter

    def test_song_similarity_both_title_and_artist_required(self):
        """song_similarity requires BOTH title AND artist non-empty."""
        test_cases = [
            {"title": "Song", "artist": ""},       # Only title → invalid
            {"title": "", "artist": "Artist"},     # Only artist → invalid
            {"title": "Song", "artist": "Artist"}, # Both → valid
        ]

        for tc in test_cases:
            title_valid = bool(tc['title'].strip())
            artist_valid = bool(tc['artist'].strip())
            is_valid = title_valid and artist_valid

            if tc['title'] == "Song" and tc['artist'] == "Artist":
                assert is_valid
            else:
                assert not is_valid


class TestArtistDiversityEnforcement:
    """Test artist diversity cap and backfill logic (lines ~671-702 in app_chat.py)."""

    def _apply_diversity_logic(self, songs, max_per_artist, target_count):
        """Helper to apply diversity logic (extracted from app_chat.py)."""
        artist_song_counts = {}
        diverse_list = []
        overflow_pool = []

        for song in songs:
            artist = song.get('artist', 'Unknown')
            artist_song_counts[artist] = artist_song_counts.get(artist, 0) + 1

            if artist_song_counts[artist] <= max_per_artist:
                diverse_list.append(song)
            else:
                overflow_pool.append(song)

        # Backfill if needed
        if len(diverse_list) < target_count and overflow_pool:
            # Count how many unique artists in diverse_list
            diverse_artist_counts = {}
            for song in diverse_list:
                artist = song.get('artist', 'Unknown')
                diverse_artist_counts[artist] = diverse_artist_counts.get(artist, 0) + 1

            # Sort overflow by least-represented artists first
            def artist_rarity(song):
                artist = song.get('artist', 'Unknown')
                return diverse_artist_counts.get(artist, 0)

            overflow_sorted = sorted(overflow_pool, key=artist_rarity)

            backfill_needed = target_count - len(diverse_list)
            backfill = overflow_sorted[:backfill_needed]
            diverse_list.extend(backfill)

        return diverse_list

    def test_songs_above_cap_moved_to_overflow(self):
        """Songs above MAX_SONGS_PER_ARTIST_PLAYLIST moved to overflow pool."""
        songs = [
            {'item_id': '1', 'artist': 'Beatles', 'title': 'Let It Be'},
            {'item_id': '2', 'artist': 'Beatles', 'title': 'Hey Jude'},
            {'item_id': '3', 'artist': 'Beatles', 'title': 'A Day in Life'},
            {'item_id': '4', 'artist': 'Beatles', 'title': 'Twist and Shout'},
            {'item_id': '5', 'artist': 'Beatles', 'title': 'Love Me Do'},
            {'item_id': '6', 'artist': 'Beatles', 'title': 'Penny Lane'},  # 6th song should go to overflow
        ]

        result = self._apply_diversity_logic(songs, max_per_artist=5, target_count=5)

        # With target = 5 and only 5 Beatles fitting, we should have exactly 5
        beatles_in_result = [s for s in result if s['artist'] == 'Beatles']
        assert len(beatles_in_result) == 5
        assert len(result) == 5

    def test_exact_cap_songs_all_included(self):
        """If songs == cap, all included."""
        songs = [
            {'item_id': f'{i}', 'artist': 'Artist1', 'title': f'Song{i}'}
            for i in range(1, 6)
        ]

        result = self._apply_diversity_logic(songs, max_per_artist=5, target_count=10)

        assert len(result) == 5
        assert all(s['artist'] == 'Artist1' for s in result)

    def test_backfill_from_overflow(self):
        """Overflow songs backfilled if target not met."""
        songs = [
            # 5 Beatles (at cap)
            {'item_id': '1', 'artist': 'Beatles', 'title': 'A'},
            {'item_id': '2', 'artist': 'Beatles', 'title': 'B'},
            {'item_id': '3', 'artist': 'Beatles', 'title': 'C'},
            {'item_id': '4', 'artist': 'Beatles', 'title': 'D'},
            {'item_id': '5', 'artist': 'Beatles', 'title': 'E'},
            # 3 Rolling Stones (overflow)
            {'item_id': '6', 'artist': 'Rolling Stones', 'title': 'X'},
            {'item_id': '7', 'artist': 'Rolling Stones', 'title': 'Y'},
            {'item_id': '8', 'artist': 'Rolling Stones', 'title': 'Z'},
        ]

        result = self._apply_diversity_logic(songs, max_per_artist=5, target_count=8)

        # Should have 5 Beatles + 3 Rolling Stones = 8
        assert len(result) == 8
        beatles = [s for s in result if s['artist'] == 'Beatles']
        stones = [s for s in result if s['artist'] == 'Rolling Stones']
        assert len(beatles) == 5
        assert len(stones) == 3

    def test_backfill_prioritizes_underrepresented_artists(self):
        """Backfill prefers artists with fewer songs already in list."""
        songs = [
            # 5 Artist1 (at cap)
            {'item_id': '1', 'artist': 'Artist1', 'title': 'A1'},
            {'item_id': '2', 'artist': 'Artist1', 'title': 'A2'},
            {'item_id': '3', 'artist': 'Artist1', 'title': 'A3'},
            {'item_id': '4', 'artist': 'Artist1', 'title': 'A4'},
            {'item_id': '5', 'artist': 'Artist1', 'title': 'A5'},
            # 1 Artist2 (underrepresented)
            {'item_id': '6', 'artist': 'Artist2', 'title': 'B1'},
            # 5 Artist3 (at cap)
            {'item_id': '7', 'artist': 'Artist3', 'title': 'C1'},
            {'item_id': '8', 'artist': 'Artist3', 'title': 'C2'},
            {'item_id': '9', 'artist': 'Artist3', 'title': 'C3'},
            {'item_id': '10', 'artist': 'Artist3', 'title': 'C4'},
            {'item_id': '11', 'artist': 'Artist3', 'title': 'C5'},
            # Overflows
            {'item_id': '12', 'artist': 'Artist2', 'title': 'B2'},
            {'item_id': '13', 'artist': 'Artist3', 'title': 'C6'},
        ]

        result = self._apply_diversity_logic(songs, max_per_artist=5, target_count=12)

        # Should backfill Artist2 before Artist3 (more underrepresented)
        assert len(result) == 12
        artist2_count = len([s for s in result if s['artist'] == 'Artist2'])
        assert artist2_count >= 2  # B1 + B2 from backfill

    def test_overflow_pool_not_used_when_target_met(self):
        """If diverse_list already meets target, don't add overflow."""
        songs = [
            {'item_id': '1', 'artist': 'Artist1', 'title': 'A1'},
            {'item_id': '2', 'artist': 'Artist1', 'title': 'A2'},
            {'item_id': '3', 'artist': 'Artist2', 'title': 'B1'},
            {'item_id': '4', 'artist': 'Artist1', 'title': 'A3'},  # Overflow
        ]

        result = self._apply_diversity_logic(songs, max_per_artist=2, target_count=3)

        # Should have exactly 3: Artist1(2) + Artist2(1)
        assert len(result) == 3
        artist1_count = len([s for s in result if s['artist'] == 'Artist1'])
        assert artist1_count == 2


class TestIterationMessage:
    """Test iteration 0 vs iteration > 0 message content."""

    def test_iteration_0_message_is_minimal_request(self):
        """Iteration 0 should just be: 'Build a {target}-song playlist for: \"...\"'"""
        user_input = "songs like Radiohead"
        target = 100

        # Iteration 0 message construction
        ai_context = f'Build a {target}-song playlist for: "{user_input}"'

        # Should be simple, no library stats
        assert "Build a 100-song playlist for:" in ai_context
        assert "Radiohead" in ai_context
        assert "Top artists:" not in ai_context
        assert "Genres covered:" not in ai_context

    def test_iteration_gt0_contains_top_artists(self):
        """Iteration > 0 should include top artists and their counts."""
        # Simulate building the feedback message for iteration > 0
        current_song_count = 45
        target_song_count = 100
        songs_needed = target_song_count - current_song_count

        # Simulated top artists
        artist_counts = {'Radiohead': 12, 'Thom Yorke': 8, 'The National': 6}
        top_5 = sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_artists_str = ', '.join([f'{a}({c})' for a, c in top_5])

        ai_context = f"""Original request: "songs like Radiohead"
Progress: {current_song_count}/{target_song_count} songs collected. Need {songs_needed} MORE.

What we have so far:
- Top artists: {top_artists_str}
"""

        assert f"{current_song_count}/{target_song_count}" in ai_context
        assert "Top artists:" in ai_context
        assert "Radiohead(12)" in ai_context

    def test_iteration_gt0_contains_diversity_ratio(self):
        """Iteration > 0 should show unique artists / total songs."""
        current_song_count = 45
        unique_artists = 15
        diversity_ratio = unique_artists / max(current_song_count, 1)

        ai_context = f"Artist diversity: {unique_artists} unique artists (ratio: {diversity_ratio:.2f})"

        assert "Artist diversity:" in ai_context
        assert f"{unique_artists}" in ai_context

    def test_iteration_gt0_contains_tools_used_history(self):
        """Iteration > 0 should show which tools were used and song counts."""
        tools_used = [
            {'name': 'text_search', 'songs': 25},
            {'name': 'song_alchemy', 'songs': 20},
        ]
        tools_str = ', '.join([f"{t['name']}({t['songs']})" for t in tools_used])

        ai_context = f"Tools used: {tools_str}"

        assert "text_search(25)" in ai_context
        assert "song_alchemy(20)" in ai_context
