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
suffix, the library and playlist dict shapes both the UI and the dispatcher
read, refusing to hand a failed HTTP response to the downloader, and the track
dict shape the analysis pipeline reads.

Main Features:
* Instant playlists get the _instant suffix the other five backends append
* Playlist creation returns the {'Id','Name'} dict callers dereference
* A streamed download that errors, or that carries an Ampache JSON error under
  HTTP 200, yields no file
* list_libraries returns the lowercase id/name both JavaScript consumers read
* Handshake caches its token per credential SET, re-handshakes once on a lapsed
  session, and never sends a real password as a plaintext API key
* The library filter is pushed into the query and still enforced locally
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


@pytest.fixture
def configured():
    from tasks.mediaserver import ampache

    with patch.object(ampache.config, 'AMPACHE_URL', 'http://ampache.test'), \
         patch.object(ampache.config, 'AMPACHE_USER', 'amp'), \
         patch.object(ampache.config, 'AMPACHE_PASSWORD', 'secret'):
        yield


def _json_response(payload, status_ok=True, content_type='application/json'):
    response = MagicMock()
    response.json.return_value = payload
    response.headers = {'Content-Type': content_type}
    if status_ok:
        response.raise_for_status.return_value = None
    else:
        response.raise_for_status.side_effect = Exception('404 Not Found')
    return response


def _audio_response(chunks=(b'audio-bytes',)):
    response = _json_response({}, content_type='audio/mpeg')
    response.iter_content.return_value = list(chunks)
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


class TestInstantPlaylistNaming:
    def test_create_instant_playlist_appends_the_instant_suffix(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'create_playlist', return_value={'Id': '7'}) as created:
            result = ampache.create_instant_playlist('My Mix', ['1', '2'], user_creds=creds)

        assert created.call_args[0][0] == 'My Mix_instant'
        assert result == {'Id': '7'}

    def test_create_instant_playlist_strips_before_appending_the_suffix(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'create_playlist', return_value={'Id': '7'}) as created:
            ampache.create_instant_playlist('  Spaced  ', ['1'], user_creds=creds)

        assert created.call_args[0][0] == 'Spaced_instant'


class TestPlaylistReturnShape:
    def test_create_playlist_returns_the_id_dict_callers_dereference(self, configured):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'auth': 'tok'}),
                _json_response({'playlist': {'id': 42}}),
                _json_response({'success': 'added'}),
            ]
            result = ampache.create_playlist('Nightly', ['1'])

        assert result == {'Id': '42', 'Name': 'Nightly'}
        assert result.get('Id') == '42'

    def test_a_playlist_that_received_none_of_its_tracks_is_not_reported_created(self, configured):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'auth': 'tok'}),
                _json_response({'playlist': {'id': 42}}),
                _json_response({'error': {'errorCode': '4710', 'errorMessage': 'nope'}}),
                _json_response({'error': {'errorCode': '4710', 'errorMessage': 'nope'}}),
            ]
            result = ampache.create_playlist('Nightly', ['1', '2'])

        assert result is None

    def test_a_partially_filled_playlist_is_still_returned(self, configured):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'auth': 'tok'}),
                _json_response({'playlist': {'id': 42}}),
                _json_response({'success': 'added'}),
                _json_response({'error': {'errorCode': '4710', 'errorMessage': 'nope'}}),
            ]
            result = ampache.create_playlist('Nightly', ['1', '2'])

        assert result == {'Id': '42', 'Name': 'Nightly'}


class TestDispatcherArity:
    def test_create_or_replace_playlist_accepts_the_user_creds_the_dispatcher_passes(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'get_playlist_by_name', return_value=None), \
             patch.object(ampache, 'create_playlist', return_value={'Id': '9'}) as created:
            result = ampache.create_or_replace_playlist('Nightly', ['1'], creds)

        assert created.call_args[0][0] == 'Nightly'
        assert result == {'Id': '9'}

    def test_create_or_replace_playlist_deletes_the_existing_playlist_first(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'get_playlist_by_name', return_value={'Id': '3'}), \
             patch.object(ampache, 'delete_playlist', return_value=True) as deleted, \
             patch.object(ampache, 'create_playlist', return_value={'Id': '9'}):
            ampache.create_or_replace_playlist('Nightly', ['1'], creds)

        deleted.assert_called_once_with('3')

    def test_a_failed_delete_aborts_instead_of_creating_a_duplicate_name(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'get_playlist_by_name', return_value={'Id': '3'}), \
             patch.object(ampache, 'delete_playlist', return_value=False), \
             patch.object(ampache, 'create_playlist') as created:
            result = ampache.create_or_replace_playlist('Nightly', ['1'], creds)

        assert result is None
        created.assert_not_called()

    def test_get_playlist_by_name_looks_the_playlist_up_with_the_callers_creds(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_request') as request:
            request.return_value = {'playlist': [{'id': 5, 'name': 'Nightly'}]}
            found = ampache.get_playlist_by_name('Nightly', user_creds=creds)

        assert found['Id'] == '5'
        assert request.call_args.kwargs['user_creds'] == creds


class TestLibraryListing:
    def test_list_libraries_returns_the_lowercase_id_and_name_the_ui_reads(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_request') as request:
            request.return_value = {'catalog': [{'id': 2, 'name': 'Main'}]}
            libraries = ampache.list_libraries(user_creds=creds)

        assert libraries == [{'id': '2', 'name': 'Main'}]

    def test_a_nameless_catalog_still_gets_a_usable_label(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_request') as request:
            request.return_value = {'catalog': [{'id': 9}]}
            libraries = ampache.list_libraries(user_creds=creds)

        assert libraries == [{'id': '9', 'name': 'Catalog 9'}]


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

    def test_a_json_error_served_as_http_200_is_not_written_as_audio(self, tmp_path):
        from tasks.mediaserver import ampache

        handshake = _json_response({'auth': 'tok'})
        stream = _json_response({'error': {'errorCode': '4742', 'errorMessage': 'ACL'}})
        stream.iter_content.return_value = [b'{"error":{"errorCode":"4742"}}']
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

    def test_a_lapsed_session_on_a_download_rehandshakes_and_retries(self, tmp_path):
        from tasks.mediaserver import ampache

        expired = _json_response({'error': {'errorCode': '4701'}})
        expired.iter_content.return_value = [b'{"error":{"errorCode":"4701"}}']
        expired.__enter__.return_value = expired
        expired.__exit__.return_value = False

        with patch.object(ampache.config, 'AMPACHE_URL', 'http://ampache.test'), \
             patch.object(ampache.config, 'AMPACHE_USER', 'amp'), \
             patch.object(ampache.config, 'AMPACHE_PASSWORD', 'secret'), \
             patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'auth': 'tok'}),
                expired,
                _json_response({'auth': 'tok2'}),
                _audio_response(),
            ]
            path = ampache.download_track(str(tmp_path), {'Id': '1', 'suffix': 'mp3'})

        assert path is not None
        assert (tmp_path / '1.mp3').read_bytes() == b'audio-bytes'

    def test_a_successful_stream_is_written_with_the_format_extension(self, tmp_path):
        from tasks.mediaserver import ampache

        with patch.object(ampache.config, 'AMPACHE_URL', 'http://ampache.test'), \
             patch.object(ampache.config, 'AMPACHE_USER', 'amp'), \
             patch.object(ampache.config, 'AMPACHE_PASSWORD', 'secret'), \
             patch.object(ampache, 'requests') as http:
            http.get.side_effect = [_json_response({'auth': 'tok'}), _audio_response()]
            path = ampache.download_track(str(tmp_path), {'Id': '1', 'suffix': 'mp3'})

        assert path is not None
        assert path.endswith('1.mp3')
        assert (tmp_path / '1.mp3').read_bytes() == b'audio-bytes'

    def test_a_track_with_no_format_falls_back_to_the_path_extension(self, tmp_path):
        from tasks.mediaserver import ampache

        with patch.object(ampache.config, 'AMPACHE_URL', 'http://ampache.test'), \
             patch.object(ampache.config, 'AMPACHE_USER', 'amp'), \
             patch.object(ampache.config, 'AMPACHE_PASSWORD', 'secret'), \
             patch.object(ampache, 'requests') as http:
            http.get.side_effect = [_json_response({'auth': 'tok'}), _audio_response()]
            path = ampache.download_track(str(tmp_path), {'Id': '1', 'Path': '/music/x.flac'})

        assert path.endswith('1.flac')


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

    def test_a_changed_password_does_not_reuse_the_cached_token(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'auth': 'tok'}),
                _json_response({'song': []}),
                _json_response({'auth': 'tok2'}),
                _json_response({'song': []}),
            ]
            ampache._request('songs', user_creds=creds)
            ampache._request('songs', user_creds={**creds, 'password': 'rotated'})

        assert http.get.call_count == 4

    def test_the_cache_key_never_carries_the_password_in_the_clear(self):
        from tasks.mediaserver import ampache

        key = ampache._cache_key('http://ampache.test', 'amp', 'secret')

        assert 'secret' not in key

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

    def test_a_session_that_stays_expired_reports_that_it_could_not_be_renewed(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'auth': 'tok'}),
                _json_response({'error': {'errorCode': '4701'}}),
                _json_response({'auth': 'tok2'}),
                _json_response({'error': {'errorCode': '4701'}}),
            ]
            body, err = ampache._request_ex('songs', user_creds=creds)

        assert body is None
        assert err == {'kind': 'auth', 'message': 'Ampache session could not be renewed'}

    def test_an_api_key_shaped_secret_falls_back_to_the_key_handshake(self, creds):
        from tasks.mediaserver import ampache

        api_key = 'a' * 64
        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'error': {'message': 'bad password'}}),
                _json_response({'auth': 'key-session'}),
            ]
            url, token, err = ampache._token(user_creds={**creds, 'password': api_key})

        assert err is None
        assert token == 'key-session'
        assert http.get.call_args_list[1].kwargs['params']['auth'] == api_key

    def test_a_real_password_is_never_sent_as_a_plaintext_api_key(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [_json_response({'error': {'message': 'bad password'}})]
            url, token, err = ampache._token(user_creds=creds)

        assert token is None
        assert err['kind'] == 'auth'
        assert http.get.call_count == 1
        assert http.get.call_args_list[0].kwargs['params']['auth'] != 'secret'

    def test_an_empty_username_authenticates_with_the_key_alone(self):
        from tasks.mediaserver import ampache

        api_key = 'b' * 64
        with patch.object(ampache.config, 'AMPACHE_USER', ''), \
             patch.object(ampache, 'requests') as http:
            http.get.side_effect = [_json_response({'auth': 'key-session'})]
            url, token, err = ampache._token(
                user_creds={'url': 'http://ampache.test', 'user': '', 'password': api_key}
            )

        assert err is None
        assert token == 'key-session'
        params = http.get.call_args_list[0].kwargs['params']
        assert params['auth'] == api_key
        assert 'user' not in params

    def test_a_missing_url_or_password_is_reported_as_a_config_error(self):
        from tasks.mediaserver import ampache

        with patch.object(ampache.config, 'AMPACHE_URL', ''), \
             patch.object(ampache.config, 'AMPACHE_PASSWORD', ''):
            url, token, err = ampache._token(user_creds={'url': '', 'user': '', 'password': ''})

        assert token is None
        assert err['kind'] == 'config'

    def test_the_callers_lyrics_budget_also_bounds_the_handshake(self, configured):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'auth': 'tok'}),
                _json_response({'song': [{'lyrics': 'la'}]}),
            ]
            ampache.get_lyrics('1', timeout=2.5)

        assert http.get.call_args_list[0].kwargs['timeout'] == 2.5
        assert http.get.call_args_list[1].kwargs['timeout'] == 2.5

    def test_an_ordinary_call_keeps_the_default_handshake_timeout(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [_json_response({'auth': 'tok'}), _json_response({'song': []})]
            ampache._request('songs', user_creds=creds)

        assert http.get.call_args_list[0].kwargs['timeout'] == ampache._HANDSHAKE_TIMEOUT_SECONDS


class TestLibraryFilter:
    def test_the_catalog_filter_is_pushed_into_the_query(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_target_catalog_ids', return_value={'2'}), \
             patch.object(ampache, '_request_ex') as request:
            request.return_value = ({'song': [{'id': 1, 'catalog': 2}]}, None)
            songs = ampache.get_all_songs(user_creds=creds)

        action, params = request.call_args[0][0], request.call_args[0][1]
        assert action == 'advanced_search'
        assert params['rule_1'] == 'catalog'
        assert params['rule_1_input'] == '2'
        assert [s['Id'] for s in songs] == ['1']

    def test_a_server_that_ignores_the_rule_is_still_filtered_locally(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_target_catalog_ids', return_value={'2'}), \
             patch.object(ampache, '_request_ex') as request:
            request.return_value = (
                {'song': [{'id': 1, 'catalog': 2}, {'id': 2, 'catalog': 7}]},
                None,
            )
            songs = ampache.get_all_songs(user_creds=creds)

        assert [s['Id'] for s in songs] == ['1']

    def test_an_unsupported_advanced_search_falls_back_to_a_full_fetch(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_target_catalog_ids', return_value={'2'}), \
             patch.object(ampache, '_request_ex') as request:
            request.side_effect = [
                (None, {'kind': 'api', 'message': 'bad request'}),
                ({'song': [{'id': 1, 'catalog': 2}, {'id': 2, 'catalog': 7}]}, None),
            ]
            songs = ampache.get_all_songs(user_creds=creds)

        assert request.call_args_list[1][0][0] == 'songs'
        assert [s['Id'] for s in songs] == ['1']

    def test_a_catalogue_row_carries_only_the_keys_consumers_read(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_target_catalog_ids', return_value=None), \
             patch.object(ampache, '_request_ex') as request:
            request.side_effect = [
                ({'song': [{'id': 1, 'title': 'S', 'art': 'http://art', 'r128_track_gain': -7}]}, None),
                ({'song': []}, None),
            ]
            songs = ampache.get_all_songs(user_creds=creds)

        assert set(songs[0]) == set(ampache._CATALOGUE_KEYS)

    def test_recent_albums_keeps_paging_until_the_filter_yields_enough(self):
        from tasks.mediaserver import ampache

        first = [{'id': i, 'name': f'A{i}', 'catalog': 7} for i in range(500)]
        second = [{'id': 500, 'name': 'Wanted', 'catalog': 2}]
        with patch.object(ampache, '_target_catalog_ids', return_value={'2'}), \
             patch.object(ampache, '_request') as request:
            request.side_effect = [{'album': first}, {'album': second}]
            albums = ampache.get_recent_albums(1)

        assert [a['Id'] for a in albums] == ['500']
        assert request.call_args_list[1][0][1]['offset'] == 500

    def test_recent_albums_with_no_filter_asks_the_server_for_exactly_the_limit(self):
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_target_catalog_ids', return_value=None), \
             patch.object(ampache, '_request') as request:
            request.return_value = {'album': [{'id': 1, 'name': 'A'}, {'id': 2, 'name': 'B'}]}
            albums = ampache.get_recent_albums(2)

        assert request.call_args_list[0][0][1]['limit'] == 2
        assert [a['Id'] for a in albums] == ['1', '2']

    def test_a_filter_matching_no_catalog_returns_no_albums(self):
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_target_catalog_ids', return_value=set()), \
             patch.object(ampache, '_request') as request:
            albums = ampache.get_recent_albums(10)

        assert albums == []
        request.assert_not_called()


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


class TestConnectionTest:
    def test_relative_paths_are_reported_as_a_warning_not_a_silent_pass(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_request_ex') as request:
            request.return_value = ({'song': [{'id': 1, 'filename': 'music/song.mp3'}]}, None)
            result = ampache.test_connection(user_creds=creds)

        assert result['ok'] is True
        assert result['path_format'] == 'relative'
        assert result['warnings']

    def test_absolute_paths_produce_no_warning(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_request_ex') as request:
            request.return_value = ({'song': [{'id': 1, 'filename': '/music/song.mp3'}]}, None)
            result = ampache.test_connection(user_creds=creds)

        assert result['path_format'] == 'absolute'
        assert result['warnings'] == []


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


class TestApiKeyOnlyInstall:
    def test_an_api_key_only_ampache_server_is_accepted_by_the_registry(self):
        from database import missing_required_creds

        assert missing_required_creds(
            'ampache', {'url': 'http://amp', 'user': '', 'password': 'k' * 64}
        ) == []

    def test_a_missing_url_or_password_is_still_refused(self):
        from database import missing_required_creds

        assert missing_required_creds('ampache', {'url': '', 'user': 'amp', 'password': ''}) == [
            'url',
            'password',
        ]
