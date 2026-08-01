# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""K7 media-server backend adapter unit tests.

Exercises the K7 provider helpers with mocked HTTP calls only so playlist
creation, library filtering and top-played fallbacks stay accurate without a
live K7 server.
"""

from unittest.mock import Mock, patch

K7_URL = 'http://k7:7080'
K7_API_KEY = 'test-api-key'


def _resp(payload, status=200):
    r = Mock()
    r.status_code = status
    r.raise_for_status = Mock()
    r.json.return_value = payload
    return r


def _k7_config():
    return patch.multiple(
        'tasks.mediaserver.k7.config',
        K7_URL=K7_URL,
        K7_API_KEY=K7_API_KEY,
        MUSIC_LIBRARIES='',
    )


class TestK7Playlists:
    @patch('tasks.mediaserver.k7._add_items_to_playlist')
    @patch('tasks.mediaserver.k7.requests.post')
    def test_create_playlist_returns_id(self, post, add_items):
        from tasks.mediaserver import k7

        playlist_id = '11111111-1111-4111-8111-111111111111'
        post.return_value = _resp(playlist_id)
        with _k7_config():
            result = k7.create_playlist('Radio Mix', ['track-1', 'track-2'])

        assert result == playlist_id
        body = post.call_args.kwargs.get('json') or post.call_args[1].get('json')
        assert body['title'] == 'Radio Mix'
        add_items.assert_called_once_with(playlist_id, ['track-1', 'track-2'], None)

    @patch('tasks.mediaserver.k7._add_items_to_playlist')
    @patch('tasks.mediaserver.k7.requests.post')
    def test_create_instant_playlist_appends_suffix(self, post, _add_items):
        from tasks.mediaserver import k7

        playlist_id = '22222222-2222-4222-8222-222222222222'
        post.return_value = _resp(playlist_id)
        with _k7_config():
            result = k7.create_instant_playlist('Mood', ['track-1'])

        assert result == playlist_id
        body = post.call_args.kwargs.get('json') or post.call_args[1].get('json')
        assert body['title'] == 'Mood_instant'


class TestK7Libraries:
    def test_library_filter_is_case_insensitive(self):
        from tasks.mediaserver import k7

        with _k7_config(), patch.object(
            k7.config, 'MUSIC_LIBRARIES', 'Music'
        ), patch.object(
            k7, 'list_libraries', return_value=[{'id': 'lib-1', 'name': 'music'}]
        ):
            assert k7._get_target_library_ids() == {'lib-1'}


class TestK7TopPlayed:
    @patch('tasks.mediaserver.k7.requests.get')
    def test_includes_zero_play_tracks(self, get):
        from tasks.mediaserver import k7

        payload = {
            'items': [
                {
                    'id': 'track-1',
                    'title': 'Fresh Track',
                    'artistName': 'Artist',
                    'userState': {'playCount': 0},
                }
            ]
        }
        get.side_effect = [_resp([], status=404), _resp(payload)]
        with _k7_config():
            tracks = k7.get_top_played_songs(10)

        assert len(tracks) == 1
        assert tracks[0]['Id'] == 'track-1'
        assert tracks[0]['PlayCount'] == 0
