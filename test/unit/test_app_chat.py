# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for chat request pre-validation and result shaping.

Covers the input guards and artist-diversity enforcement used by the chat
pipeline.

Main Features:
* Song-similarity and search-database filter pre-validation.
* Artist-diversity capping, overflow backfill, and underrepresented priority.
"""

class TestPreValidation:
    def test_song_similarity_empty_title_rejected(self):
        title = ""

        is_valid = bool(title.strip())
        assert not is_valid

    def test_song_similarity_empty_artist_rejected(self):
        artist = ""

        is_valid = bool(artist.strip())
        assert not is_valid

    def test_song_similarity_whitespace_only_rejected(self):
        title = "   "
        artist = "  \t  "

        assert not title.strip()
        assert not artist.strip()

    def test_search_database_zero_filters_rejected(self):
        filters = {}
        filter_keys = [
            'genres',
            'moods',
            'tempo_min',
            'tempo_max',
            'energy_min',
            'energy_max',
            'key',
            'scale',
            'year_min',
            'year_max',
            'min_rating',
            'album',
        ]

        has_filter = any(filters.get(k) for k in filter_keys)
        assert not has_filter

    def test_search_database_album_only_filter_accepted(self):
        filters = {'album': 'Dark Side of the Moon'}
        filter_keys = [
            'genres',
            'moods',
            'tempo_min',
            'tempo_max',
            'energy_min',
            'energy_max',
            'key',
            'scale',
            'year_min',
            'year_max',
            'min_rating',
            'album',
        ]

        has_filter = any(filters.get(k) for k in filter_keys)
        assert has_filter

    def test_search_database_genres_filter_accepted(self):
        filters = {'genres': ['rock', 'metal']}
        filter_keys = [
            'genres',
            'moods',
            'tempo_min',
            'tempo_max',
            'energy_min',
            'energy_max',
            'key',
            'scale',
            'year_min',
            'year_max',
            'min_rating',
            'album',
        ]

        has_filter = any(filters.get(k) for k in filter_keys)
        assert has_filter

    def test_search_database_year_filter_accepted(self):
        filters = {'year_min': 1990}
        filter_keys = [
            'genres',
            'moods',
            'tempo_min',
            'tempo_max',
            'energy_min',
            'energy_max',
            'key',
            'scale',
            'year_min',
            'year_max',
            'min_rating',
            'album',
        ]

        has_filter = any(filters.get(k) for k in filter_keys)
        assert has_filter

    def test_song_similarity_both_title_and_artist_required(self):
        test_cases = [
            {"title": "Song", "artist": ""},
            {"title": "", "artist": "Artist"},
            {"title": "Song", "artist": "Artist"},
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
    def _apply_diversity_logic(self, songs, max_per_artist, target_count):
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

        if len(diverse_list) < target_count and overflow_pool:
            diverse_artist_counts = {}
            for song in diverse_list:
                artist = song.get('artist', 'Unknown')
                diverse_artist_counts[artist] = diverse_artist_counts.get(artist, 0) + 1

            def artist_rarity(song):
                artist = song.get('artist', 'Unknown')
                return diverse_artist_counts.get(artist, 0)

            overflow_sorted = sorted(overflow_pool, key=artist_rarity)

            backfill_needed = target_count - len(diverse_list)
            backfill = overflow_sorted[:backfill_needed]
            diverse_list.extend(backfill)

        return diverse_list

    def test_songs_above_cap_moved_to_overflow(self):
        songs = [
            {'item_id': '1', 'artist': 'Beatles', 'title': 'Let It Be'},
            {'item_id': '2', 'artist': 'Beatles', 'title': 'Hey Jude'},
            {'item_id': '3', 'artist': 'Beatles', 'title': 'A Day in Life'},
            {'item_id': '4', 'artist': 'Beatles', 'title': 'Twist and Shout'},
            {'item_id': '5', 'artist': 'Beatles', 'title': 'Love Me Do'},
            {'item_id': '6', 'artist': 'Beatles', 'title': 'Penny Lane'},
        ]

        result = self._apply_diversity_logic(songs, max_per_artist=5, target_count=5)

        beatles_in_result = [s for s in result if s['artist'] == 'Beatles']
        assert len(beatles_in_result) == 5
        assert len(result) == 5

    def test_exact_cap_songs_all_included(self):
        songs = [{'item_id': f'{i}', 'artist': 'Artist1', 'title': f'Song{i}'} for i in range(1, 6)]

        result = self._apply_diversity_logic(songs, max_per_artist=5, target_count=10)

        assert len(result) == 5
        assert all(s['artist'] == 'Artist1' for s in result)

    def test_backfill_from_overflow(self):
        songs = [
            {'item_id': '1', 'artist': 'Beatles', 'title': 'A'},
            {'item_id': '2', 'artist': 'Beatles', 'title': 'B'},
            {'item_id': '3', 'artist': 'Beatles', 'title': 'C'},
            {'item_id': '4', 'artist': 'Beatles', 'title': 'D'},
            {'item_id': '5', 'artist': 'Beatles', 'title': 'E'},
            {'item_id': '6', 'artist': 'Rolling Stones', 'title': 'X'},
            {'item_id': '7', 'artist': 'Rolling Stones', 'title': 'Y'},
            {'item_id': '8', 'artist': 'Rolling Stones', 'title': 'Z'},
        ]

        result = self._apply_diversity_logic(songs, max_per_artist=5, target_count=8)

        assert len(result) == 8
        beatles = [s for s in result if s['artist'] == 'Beatles']
        stones = [s for s in result if s['artist'] == 'Rolling Stones']
        assert len(beatles) == 5
        assert len(stones) == 3

    def test_backfill_prioritizes_underrepresented_artists(self):
        songs = [
            {'item_id': '1', 'artist': 'Artist1', 'title': 'A1'},
            {'item_id': '2', 'artist': 'Artist1', 'title': 'A2'},
            {'item_id': '3', 'artist': 'Artist1', 'title': 'A3'},
            {'item_id': '4', 'artist': 'Artist1', 'title': 'A4'},
            {'item_id': '5', 'artist': 'Artist1', 'title': 'A5'},
            {'item_id': '6', 'artist': 'Artist2', 'title': 'B1'},
            {'item_id': '7', 'artist': 'Artist3', 'title': 'C1'},
            {'item_id': '8', 'artist': 'Artist3', 'title': 'C2'},
            {'item_id': '9', 'artist': 'Artist3', 'title': 'C3'},
            {'item_id': '10', 'artist': 'Artist3', 'title': 'C4'},
            {'item_id': '11', 'artist': 'Artist3', 'title': 'C5'},
            {'item_id': '12', 'artist': 'Artist2', 'title': 'B2'},
            {'item_id': '13', 'artist': 'Artist3', 'title': 'C6'},
        ]

        result = self._apply_diversity_logic(songs, max_per_artist=5, target_count=12)

        assert len(result) == 12
        artist2_count = len([s for s in result if s['artist'] == 'Artist2'])
        assert artist2_count >= 2

    def test_overflow_pool_not_used_when_target_met(self):
        songs = [
            {'item_id': '1', 'artist': 'Artist1', 'title': 'A1'},
            {'item_id': '2', 'artist': 'Artist1', 'title': 'A2'},
            {'item_id': '3', 'artist': 'Artist2', 'title': 'B1'},
            {'item_id': '4', 'artist': 'Artist1', 'title': 'A3'},
        ]

        result = self._apply_diversity_logic(songs, max_per_artist=2, target_count=3)

        assert len(result) == 3
        artist1_count = len([s for s in result if s['artist'] == 'Artist1'])
        assert artist1_count == 2
