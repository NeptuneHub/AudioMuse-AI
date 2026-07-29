# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Ampache backend behaviour, focused on the shared provider contract.

The dispatcher and its callers assume every backend behaves the same way in
places the Ampache API itself has no opinion about: the instant-playlist naming
suffix, refusing to hand a failed HTTP response to the downloader, and the
track dict shape the analysis pipeline reads.

Main Features:
* Instant playlists get the _instant suffix the other five backends append
* A streamed download that returns an HTTP error yields no file
* Handshake caches its token, re-handshakes once on a lapsed session, and
  falls back from the password hash to an API key
* _map_song exposes the keys analysis and provider_probe read
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_token_cache():
    from tasks.mediaserver import ampache

    ampache._token_cache.clear()
    yield
    ampache._token_cache.clear()


@pytest.fixture
def creds():
    return {'url': 'http://ampache.test', 'user': 'amp', 'password': 'secret'}


def _json_response(payload, status_ok=True):
    response = MagicMock()
    response.json.return_value = payload
    if status_ok:
        response.raise_for_status.return_value = None
    else:
        response.raise_for_status.side_effect = Exception('404 Not Found')
    return response


class TestInstantPlaylistNaming:
    def test_create_instant_playlist_appends_the_instant_suffix(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'create_playlist', return_value='7') as created:
            result = ampache.create_instant_playlist('My Mix', ['1', '2'], user_creds=creds)

        assert created.call_args[0][0] == 'My Mix_instant'
        assert result == '7'

    def test_create_instant_playlist_strips_before_appending_the_suffix(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'create_playlist', return_value='7') as created:
            ampache.create_instant_playlist('  Spaced  ', ['1'], user_creds=creds)

        assert created.call_args[0][0] == 'Spaced_instant'


class TestDispatcherArity:
    def test_create_or_replace_playlist_accepts_the_user_creds_the_dispatcher_passes(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'get_playlist_by_name', return_value=None), \
             patch.object(ampache, 'create_playlist', return_value='9') as created:
            result = ampache.create_or_replace_playlist('Nightly', ['1'], creds)

        assert created.call_args[0][0] == 'Nightly'
        assert result == '9'

    def test_create_or_replace_playlist_deletes_the_existing_playlist_first(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'get_playlist_by_name', return_value={'Id': '3'}), \
             patch.object(ampache, 'delete_playlist') as deleted, \
             patch.object(ampache, 'create_playlist', return_value='9'):
            ampache.create_or_replace_playlist('Nightly', ['1'], creds)

        deleted.assert_called_once_with('3')


class TestStreamedDownload:
    def test_a_streamed_download_that_errors_writes_no_file(self, tmp_path):
        from tasks.mediaserver import ampache

        handshake = _json_response({'auth': 'tok'})
        stream = _json_response({'error': {'errorCode': '4704'}}, status_ok=False)
        stream.iter_content.return_value = [b'{"error":"Require: 100"}']
        stream.__enter__.return_value = stream
        stream.__exit__.return_value = False

        with patch.object(ampache.config, 'AMPACHE_URL', 'http://ampache.test'), \
             patch.object(ampache.config, 'AMPACHE_USER', 'amp'), \
             patch.object(ampache.config, 'AMPACHE_PASSWORD', 'secret'), \
             patch.object(ampache, 'requests') as http:
            http.get.side_effect = [handshake, stream]
            path = ampache.download_track(str(tmp_path), {'Id': '1', 'suffix': 'mp3'})

        assert path is None
        assert list(tmp_path.iterdir()) == []

    def test_a_successful_stream_is_written_with_the_format_extension(self, tmp_path):
        from tasks.mediaserver import ampache

        handshake = _json_response({'auth': 'tok'})
        stream = _json_response({}, status_ok=True)
        stream.iter_content.return_value = [b'audio-bytes']
        stream.__enter__.return_value = stream
        stream.__exit__.return_value = False

        with patch.object(ampache.config, 'AMPACHE_URL', 'http://ampache.test'), \
             patch.object(ampache.config, 'AMPACHE_USER', 'amp'), \
             patch.object(ampache.config, 'AMPACHE_PASSWORD', 'secret'), \
             patch.object(ampache, 'requests') as http:
            http.get.side_effect = [handshake, stream]
            path = ampache.download_track(str(tmp_path), {'Id': '1', 'suffix': 'mp3'})

        assert path is not None
        assert path.endswith('1.mp3')
        assert (tmp_path / '1.mp3').read_bytes() == b'audio-bytes'


class TestHandshake:
    def test_a_successful_handshake_is_cached_and_not_repeated(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'auth': 'tok'}),
                _json_response({'song': []}),
                _json_response({'song': []}),
            ]
            ampache._request('songs', user_creds=creds)
            ampache._request('songs', user_creds=creds)

        assert http.get.call_count == 3

    def test_an_expired_session_triggers_exactly_one_rehandshake(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'auth': 'tok'}),
                _json_response({'error': {'errorCode': '4701'}}),
                _json_response({'auth': 'tok2'}),
                _json_response({'song': [{'id': 1}]}),
            ]
            body, err = ampache._request_ex('songs', user_creds=creds)

        assert err is None
        assert body == {'song': [{'id': 1}]}

    def test_the_password_hash_falls_back_to_an_api_key(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'error': {'message': 'bad password'}}),
                _json_response({'auth': 'key-session'}),
            ]
            url, token, err = ampache._token(user_creds=creds)

        assert err is None
        assert token == 'key-session'
        assert http.get.call_args_list[1].kwargs['params']['auth'] == 'secret'

    def test_a_missing_url_or_password_is_reported_as_a_config_error(self):
        from tasks.mediaserver import ampache

        with patch.object(ampache.config, 'AMPACHE_URL', ''), \
             patch.object(ampache.config, 'AMPACHE_PASSWORD', ''):
            url, token, err = ampache._token(user_creds={'url': '', 'user': '', 'password': ''})

        assert token is None
        assert err['kind'] == 'config'


class TestPlayStats:
    def test_top_played_asks_for_most_played_not_highest_rated(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'auth': 'tok'}),
                _json_response({'song': [{'id': 1, 'title': 'S'}]}),
            ]
            songs = ampache.get_top_played_songs(10, creds)

        params = http.get.call_args_list[1].kwargs['params']
        assert params['filter'] == 'frequent'
        assert params['type'] == 'song'
        assert [s['Id'] for s in songs] == ['1']

    def test_last_played_time_is_none_because_ampache_exposes_no_such_field(self, creds):
        from tasks.mediaserver import ampache

        assert ampache.get_last_played_time('1', creds) is None


class TestTrackMapping:
    def test_map_song_exposes_the_keys_analysis_and_the_probe_read(self):
        from tasks.mediaserver import ampache

        mapped = ampache._map_song({
            'id': 12,
            'title': 'Song',
            'artist': {'id': 3, 'name': 'Artist'},
            'albumartist': {'id': 4, 'name': 'Album Artist'},
            'album': {'id': 5, 'name': 'Album'},
            'filename': '/music/song.flac',
            'time': 210,
            'year': 1999,
            'format': 'flac',
        })

        assert mapped['Id'] == '12'
        assert mapped['Name'] == 'Song'
        assert mapped['AlbumArtist'] == 'Album Artist'
        assert mapped['ArtistId'] == '3'
        assert mapped['Album'] == 'Album'
        assert mapped['Path'] == '/music/song.flac'
        assert mapped['FilePath'] == '/music/song.flac'
        assert mapped['DurationSeconds'] == 210
        assert mapped['suffix'] == 'flac'

    def test_map_song_falls_back_to_the_track_artist_when_there_is_no_album_artist(self):
        from tasks.mediaserver import ampache

        mapped = ampache._map_song({'id': 1, 'title': 'S', 'artist': {'id': 2, 'name': 'Only'}})

        assert mapped['AlbumArtist'] == 'Only'

    def test_map_song_survives_a_row_with_no_artist_objects(self):
        from tasks.mediaserver import ampache

        mapped = ampache._map_song({'id': 1})

        assert mapped['Id'] == '1'
        assert mapped['Name'] == 'Unknown'
        assert mapped['AlbumArtist'] == 'Unknown'
        assert mapped['ArtistId'] is None


class TestSecretRedaction:
    @pytest.mark.parametrize('secret_param', ['auth', 'passphrase', 'password'])
    def test_query_string_secrets_are_redacted_from_log_text(self, secret_param):
        from tasks.mediaserver import ampache

        redacted = ampache._redact_ampache_secrets(
            f"http://amp.test/server/json.server.php?action=songs&{secret_param}=supersecret&limit=1"
        )

        assert 'supersecret' not in redacted
        assert '[REDACTED]' in redacted
