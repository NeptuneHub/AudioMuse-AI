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
* Seed-search song-seed validation and search_database filter detection.
* Artist-diversity capping and progressive cap relaxation from the overflow.
"""

import pytest

import app_chat
import config
from tasks.ai import tools


TRUTHY_SEARCH_FILTERS = {
    'genres': ['rock'],
    'moods': ['sad'],
    'tempo_min': 90,
    'tempo_max': 140,
    'energy_min': 0.2,
    'energy_max': 0.9,
    'key': 'C',
    'scale': 'minor',
    'year_min': 1990,
    'year_max': 1999,
    'min_rating': 4,
    'album': 'Dark Side of the Moon',
    'other_features': ['party'],
    'candidate_item_ids': ['abc123'],
    'voices': ['female vocalists'],
    'instrumental': True,
    'exclude_artists': ['Nickelback'],
    'exclude_genres': ['Hip-Hop'],
}


def _song(item_id, artist):
    return {'item_id': item_id, 'artist': artist, 'title': f'{artist} {item_id}'}


def _run_pipeline_with_pool(monkeypatch, songs):
    import tasks.ai.planner as planner
    import tasks.mcp_helper as mcp_helper

    def _fake_plan(**kwargs):
        yield from ()
        return {
            'songs': list(songs),
            'song_sources': {s['item_id']: 0 for s in songs},
            'tools_used_history': [],
            'plan_notes': [],
            'executed_query_str': 'stub-query',
            'filter_applied': True,
        }

    monkeypatch.setattr(planner, 'plan_and_execute_once', _fake_plan)
    monkeypatch.setattr(mcp_helper, 'get_library_context', lambda: {'total_songs': 0})
    monkeypatch.setattr(
        app_chat.app_server_context,
        'scope_results',
        lambda rows, _server, **kwargs: list(rows),
    )

    log_messages = []
    response, status = app_chat._drain_pipeline(
        app_chat._run_chat_pipeline(
            {'userInput': 'build me a playlist', 'ai_provider': 'OLLAMA'},
            log_messages,
        )
    )
    assert status == 200
    return response


class TestSeedSearchSongSeedValidation:
    @pytest.mark.parametrize(
        'title,artist',
        [
            ('', 'Artist'),
            ('Song', ''),
            ('   ', 'Artist'),
            ('Song', '  \t  '),
            ('', ''),
        ],
    )
    def test_song_seed_with_blank_title_or_artist_is_skipped_before_the_similarity_call(
        self, monkeypatch, title, artist
    ):
        calls = []
        monkeypatch.setattr(
            tools,
            '_song_similarity_api_sync',
            lambda *args: calls.append(args) or {'songs': [], 'message': ''},
        )

        result = tools._dispatch_seed_search(
            {'seeds': [{'type': 'song', 'title': title, 'artist': artist}]}, {}
        )

        assert calls == []
        assert result['songs'] == []
        assert 'seed_search: skipping malformed song seed' in result['message']

    def test_complete_song_seed_reaches_the_similarity_call_stripped_of_whitespace(
        self, monkeypatch
    ):
        calls = []

        def _fake_similarity(seed_title, seed_artist, limit):
            calls.append((seed_title, seed_artist, limit))
            return {'songs': [{'item_id': 's1'}], 'message': 'ok'}

        monkeypatch.setattr(tools, '_song_similarity_api_sync', _fake_similarity)

        result = tools._dispatch_seed_search(
            {
                'seeds': [{'type': 'song', 'title': ' Song ', 'artist': ' Artist '}],
                'get_songs': 60,
            },
            {},
        )

        assert calls == [('Song', 'Artist', 60)]
        assert result['songs'] == [{'item_id': 's1'}]


class TestSearchDatabaseFilterDetection:
    def test_other_filter_key_set_is_exactly_the_one_this_module_pins(self):
        assert set(TRUTHY_SEARCH_FILTERS) == set(tools._SEARCH_OTHER_FILTER_KEYS)

    @pytest.mark.parametrize('key', sorted(TRUTHY_SEARCH_FILTERS))
    def test_each_other_filter_key_alone_counts_as_a_filter(self, key):
        assert tools._has_other_search_filters({key: TRUTHY_SEARCH_FILTERS[key]}) is True

    @pytest.mark.parametrize(
        'tool_args',
        [
            {},
            {'artist': 'Nas'},
            {'get_songs': 200},
            {'genres': [], 'moods': None, 'album': ''},
        ],
    )
    def test_artist_only_and_empty_values_do_not_count_as_a_filter(self, tool_args):
        assert tools._has_other_search_filters(tool_args) is False

    def test_search_database_with_zero_filters_still_runs_the_query_once(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            tools,
            '_database_genre_query_sync',
            lambda *args, **kwargs: calls.append(kwargs.get('fuzzy_match'))
            or {'songs': [], 'message': 'no-op'},
        )

        result = tools._dispatch_search_database({})

        assert calls == [False]
        assert result == {'songs': [], 'message': 'no-op'}

    @pytest.mark.parametrize(
        'extra_filter,expected_fuzzy_calls',
        [
            ({}, ['Nas']),
            ({'genres': ['rock']}, []),
            ({'voices': ['female vocalists']}, []),
            ({'exclude_genres': ['Hip-Hop']}, []),
        ],
    )
    def test_fuzzy_artist_fallback_runs_only_when_no_other_filter_narrows_the_search(
        self, monkeypatch, extra_filter, expected_fuzzy_calls
    ):
        query_calls = []
        fuzzy_calls = []

        class _FakeConn:
            def close(self):
                pass

        monkeypatch.setattr(
            tools,
            '_database_genre_query_sync',
            lambda *args, **kwargs: query_calls.append(kwargs.get('fuzzy_match'))
            or {'songs': [], 'message': 'empty'},
        )
        monkeypatch.setattr(tools, '_get_db_connection', _FakeConn)
        monkeypatch.setattr(
            tools,
            '_fuzzy_match_author_title',
            lambda conn, name: fuzzy_calls.append(name) or None,
        )

        tool_args = {'artist': 'Nas'}
        tool_args.update(extra_filter)
        tools._dispatch_search_database(tool_args)

        assert query_calls == [False, True]
        assert fuzzy_calls == expected_fuzzy_calls


class TestArtistDiversityEnforcement:
    @pytest.mark.parametrize('cap', [3, 5])
    def test_final_playlist_holds_at_most_the_configured_songs_per_artist(
        self, monkeypatch, cap
    ):
        monkeypatch.setattr(config, 'MAX_SONGS_PER_ARTIST_PLAYLIST', cap)
        songs = [_song(f'b{i}', 'Beatles') for i in range(20)]
        songs += [_song(f'u{i}', f'Solo{i}') for i in range(180)]

        response = _run_pipeline_with_pool(monkeypatch, songs)

        results = response['query_results']
        assert len(results) == 100
        assert [s['item_id'] for s in results if s['artist'] == 'Beatles'] == [
            f'b{i}' for i in range(cap)
        ]
        assert f'removed {20 - cap} excess songs from pool (max {cap}/artist)' in response['message']

    def test_overflow_songs_are_dropped_when_the_capped_pool_already_fills_the_target(
        self, monkeypatch
    ):
        monkeypatch.setattr(config, 'MAX_SONGS_PER_ARTIST_PLAYLIST', 5)
        songs = [_song(f'a{a}s{i}', f'Artist{a}') for a in range(20) for i in range(10)]

        response = _run_pipeline_with_pool(monkeypatch, songs)

        results = response['query_results']
        assert [s['item_id'] for s in results] == [
            f'a{a}s{i}' for a in range(20) for i in range(5)
        ]
        assert 'Progressive cap relaxation' not in response['message']

    def test_pool_short_of_target_relaxes_the_cap_until_the_overflow_is_exhausted(
        self, monkeypatch
    ):
        monkeypatch.setattr(config, 'MAX_SONGS_PER_ARTIST_PLAYLIST', 5)
        songs = [_song(f'a{i}', 'ArtistA') for i in range(30)]
        songs += [_song(f'b{i}', 'ArtistB') for i in range(30)]

        response = _run_pipeline_with_pool(monkeypatch, songs)

        results = response['query_results']
        assert len(results) == 60
        assert len([s for s in results if s['artist'] == 'ArtistA']) == 30
        assert len([s for s in results if s['artist'] == 'ArtistB']) == 30
        assert (
            'Progressive cap relaxation: 5 -> 30/artist to reach 60 songs' in response['message']
        )

    def test_cap_relaxation_admits_one_song_per_artist_per_level_so_small_overflows_go_first(
        self, monkeypatch
    ):
        monkeypatch.setattr(config, 'MAX_SONGS_PER_ARTIST_PLAYLIST', 5)
        songs = [_song(f'a{i}', 'ArtistA') for i in range(20)]
        songs += [_song(f'b{i}', 'ArtistB') for i in range(6)]
        songs += [_song(f'u{i}', f'Solo{i}') for i in range(80)]

        response = _run_pipeline_with_pool(monkeypatch, songs)

        results = response['query_results']
        assert len(results) == 100
        assert [s['item_id'] for s in results if s['artist'] == 'ArtistB'] == [
            f'b{i}' for i in range(6)
        ]
        assert [s['item_id'] for s in results if s['artist'] == 'ArtistA'] == [
            f'a{i}' for i in range(14)
        ]
        assert (
            'Progressive cap relaxation: 5 -> 14/artist to reach 100 songs' in response['message']
        )
