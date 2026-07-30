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
* Handshake caches its token per credential SET, takes its window from the
  server's session_expire and slides it forward on every accepted call rather than
  re-handshaking on a timer, re-handshakes once on a genuinely lapsed session, and
  never sends a real password as a plaintext API key
* The library filter is pushed into the query and still enforced locally
* _map_song exposes the keys analysis and provider_probe read
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_token_cache():
    from tasks.mediaserver import ampache

    ampache._token_cache.clear()
    ampache._header_auth.clear()
    yield
    ampache._token_cache.clear()
    ampache._header_auth.clear()


@pytest.fixture
def creds():
    return {'url': 'http://ampache.test', 'user': 'amp', 'password': 'secret'}


@pytest.fixture
def key_creds():
    """Credentials whose secret is API-key shaped, so header auth is eligible."""
    return {'url': 'http://ampache.test', 'user': '', 'password': 'a' * 64}


def _ping_response(api='8.0.0', **extra):
    """A bearer-authenticated ping, which carries the whole server_details payload."""
    return _json_response({
        'session_expire': '2126-07-30T17:06:45+10:00',
        'server': api,
        'version': api,
        'api': api,
        'auth': 'a201455bfaecb00b082a5716d3dee64d',
        'username': 'user',
        **extra,
    })


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

    def test_playlist_writes_use_the_callers_creds_not_the_default_server(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'get_playlist_by_name', return_value=None), \
             patch.object(ampache, 'create_playlist', return_value={'Id': '9'}) as created:
            ampache.create_or_replace_playlist('Nightly', ['1'], creds)
            ampache.create_instant_playlist('My Mix', ['1'], user_creds=creds)

        assert [call.kwargs.get('user_creds') for call in created.call_args_list] == [creds, creds]

    def test_create_or_replace_playlist_deletes_the_existing_playlist_first(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'get_playlist_by_name', return_value={'Id': '3'}), \
             patch.object(ampache, 'delete_playlist', return_value=True) as deleted, \
             patch.object(ampache, 'create_playlist', return_value={'Id': '9'}):
            ampache.create_or_replace_playlist('Nightly', ['1'], creds)

        deleted.assert_called_once_with('3', user_creds=creds)

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

    def test_a_successful_call_slides_the_session_window_forward(self, creds):
        import time

        from tasks.mediaserver import ampache

        key = ampache._cache_key('http://ampache.test', 'amp', 'secret')
        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'auth': 'tok'}),
                _json_response({'song': []}),
                _json_response({'song': []}),
            ]
            ampache._request('songs', user_creds=creds)
            # Stand the cache up as it would look late in a long run, a moment
            # before the window the handshake opened would have lapsed.
            ampache._token_cache[key]['expires'] = time.time() + 1
            ampache._request('songs', user_creds=creds)
            slid = ampache._token_cache[key]['expires']

        handshakes = [
            call for call in http.get.call_args_list
            if (call.kwargs.get('params') or {}).get('action') == 'handshake'
        ]
        # Ampache extends a session on every accepted call, so the second request
        # must reuse the token AND push the expiry back out - not re-handshake on
        # a timer while the session it holds is still valid.
        assert len(handshakes) == 1
        assert slid > time.time() + 60

    def test_the_session_window_comes_from_the_servers_session_expire(self, creds):
        from datetime import datetime, timedelta, timezone

        from tasks.mediaserver import ampache

        expires_at = datetime.now(timezone.utc) + timedelta(seconds=1800)
        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'auth': 'tok', 'session_expire': expires_at.isoformat()}),
                _json_response({'song': []}),
            ]
            ampache._request('songs', user_creds=creds)

        cached = ampache._token_cache[ampache._cache_key('http://ampache.test', 'amp', 'secret')]
        assert 1700 < cached['lifetime'] <= 1800

    @pytest.mark.parametrize(
        'session_expire',
        [
            None,                           # older server, field absent
            '',
            'not-a-date',                   # a proxy rewrote it
            0,                              # perpetual_api_session=true reports 0
            '1970-01-01T00:00:00+00:00',    # already past, or our clock disagrees
        ],
    )
    def test_an_unusable_session_expire_falls_back_to_the_default_window(self, session_expire):
        from tasks.mediaserver import ampache

        body = {'auth': 'tok'}
        if session_expire is not None:
            body['session_expire'] = session_expire

        assert ampache._session_lifetime(body) == ampache._TOKEN_TTL_SECONDS


def _authorization(call):
    return (call.kwargs.get('headers') or {}).get('Authorization')


class TestHeaderAuth:
    """An API key goes in an Authorization header, and Ampache opens the session.

    ApiHandler only reads the header when NO auth parameter is present, so these
    tests assert the absence of `auth` from the query string as much as the
    presence of the header.
    """

    def test_a_key_shaped_secret_pings_once_then_sends_a_bearer_header(self, key_creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [_ping_response(), _json_response({'song': []})]
            body, err = ampache._request_ex('songs', user_creds=key_creds)

        assert err is None
        assert body == {'song': []}
        assert http.get.call_count == 2

        ping, call = http.get.call_args_list
        assert ping.kwargs['params']['action'] == 'ping'
        assert _authorization(ping) == 'Bearer ' + 'a' * 64
        assert _authorization(call) == 'Bearer ' + 'a' * 64
        # The header is only honoured when the query string carries no auth at all.
        assert 'auth' not in ping.kwargs['params']
        assert 'auth' not in call.kwargs['params']

    def test_the_bearer_ping_fills_the_version_gate(self, key_creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [_ping_response(api='8.0.0'), _json_response({'song': []})]
            ampache._request('songs', user_creds=key_creds)

        # Without this the API-8 warning would go quiet, since nothing handshaked.
        assert ampache._cached_api_version(key_creds) == '8.0.0'
        assert ampache._api_version_warning(key_creds) is None

    def test_an_old_server_still_warns_under_header_auth(self, key_creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [_ping_response(api='6.6.0'), _json_response({'song': []})]
            ampache._request('songs', user_creds=key_creds)

        assert 'API 6' in (ampache._api_version_warning(key_creds) or '')

    def test_a_refused_bearer_ping_falls_back_to_the_handshake(self, key_creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'error': {'errorCode': '4742', 'errorMessage': 'Access Denied'}}),
                _json_response({'auth': 'tok'}),
                _json_response({'song': []}),
            ]
            body, err = ampache._request_ex('songs', user_creds=key_creds)

        assert err is None
        assert body == {'song': []}

        ping, handshake, call = http.get.call_args_list
        assert ping.kwargs['params']['action'] == 'ping'
        assert handshake.kwargs['params']['action'] == 'handshake'
        # Back on the session path: token in the query string, no header.
        assert call.kwargs['params']['auth'] == 'tok'
        assert _authorization(call) is None

    def test_an_unauthenticated_ping_reply_is_not_mistaken_for_acceptance(self, key_creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            # Ampache answers a ping it did not authenticate with server/version
            # only - no `api`, no session. That must not read as success.
            http.get.side_effect = [
                _json_response({'server': '8.0.0', 'version': '8.0.0'}),
                _json_response({'auth': 'tok'}),
                _json_response({'song': []}),
            ]
            ampache._request('songs', user_creds=key_creds)

        assert http.get.call_args_list[1].kwargs['params']['action'] == 'handshake'
        assert ampache._header_auth[ampache._cache_key('http://ampache.test', '', 'a' * 64)] is False

    def test_the_bearer_ping_happens_once_per_credential_set(self, key_creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _ping_response(),
                _json_response({'song': []}),
                _json_response({'song': []}),
            ]
            ampache._request('songs', user_creds=key_creds)
            ampache._request('songs', user_creds=key_creds)

        pings = [
            call for call in http.get.call_args_list
            if call.kwargs['params']['action'] == 'ping'
        ]
        assert len(pings) == 1
        assert http.get.call_count == 3

    def test_a_password_never_travels_as_a_bearer_header(self, creds):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [_json_response({'auth': 'tok'}), _json_response({'song': []})]
            ampache._request('songs', user_creds=creds)

        # No ping, no header: a password would only be refused by findByApiKey, and
        # being refused means having sent it for nothing.
        assert http.get.call_count == 2
        assert http.get.call_args_list[0].kwargs['params']['action'] == 'handshake'
        assert all(_authorization(call) is None for call in http.get.call_args_list)


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
             patch.object(ampache, '_request_ex') as request:
            request.side_effect = [({'album': first}, None), ({'album': second}, None)]
            albums = ampache.get_recent_albums(1)

        assert [a['Id'] for a in albums] == ['500']
        assert request.call_args_list[1][0][1]['offset'] == 500

    def test_recent_albums_with_no_filter_asks_the_server_for_exactly_the_limit(self):
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_target_catalog_ids', return_value=None), \
             patch.object(ampache, '_request_ex') as request:
            request.return_value = (
                {'album': [{'id': 1, 'name': 'A'}, {'id': 2, 'name': 'B'}]}, None
            )
            albums = ampache.get_recent_albums(2)

        assert request.call_args_list[0][0][0] == 'albums'
        assert request.call_args_list[0][0][1]['limit'] == 2
        assert [a['Id'] for a in albums] == ['1', '2']

    def test_a_filter_matching_no_catalog_returns_no_albums(self):
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_target_catalog_ids', return_value=set()), \
             patch.object(ampache, '_request_ex') as request:
            albums = ampache.get_recent_albums(10)

        assert albums == []
        request.assert_not_called()

    def test_a_library_filter_is_pushed_into_the_album_browse(self):
        """Ampache album objects carry no catalog id, so the filter must be server-side."""
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_target_catalog_ids', return_value={'2'}), \
             patch.object(ampache, '_request_ex') as request:
            request.return_value = (
                {'album': [{'id': 9, 'name': 'Filtered', 'catalog': '2'}]}, None
            )
            albums = ampache.get_recent_albums(1)

        action, params = request.call_args_list[0][0][0], request.call_args_list[0][0][1]
        assert action == 'albums'
        assert params['cond'] == 'catalog,2'
        assert [a['Id'] for a in albums] == ['9']

    def test_several_catalogues_are_browsed_separately_and_merged_without_duplicates(self):
        """`cond` conditions combine, so one browse per catalogue is the portable form."""
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_target_catalog_ids', return_value={'2', '3'}), \
             patch.object(ampache, '_request_ex') as request:
            request.side_effect = [
                ({'album': [{'id': 1, 'name': 'From 2', 'catalog': '2'}]}, None),
                ({'album': [
                    {'id': 1, 'name': 'From 2', 'catalog': '2'},
                    {'id': 5, 'name': 'From 3', 'catalog': '3'},
                ]}, None),
            ]
            albums = ampache.get_recent_albums(0)

        assert [c[0][1]['cond'] for c in request.call_args_list] == ['catalog,2', 'catalog,3']
        assert [a['Id'] for a in albums] == ['1', '5']

    def test_an_ignored_cond_cannot_leak_albums_from_other_catalogues(self):
        """Ampache ignores an unknown `cond` and returns everything, so rows are re-checked."""
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_target_catalog_ids', return_value={'32'}), \
             patch.object(ampache, '_request_ex') as request:
            request.return_value = (
                {'album': [
                    {'id': 1, 'name': 'Wanted', 'catalog': '32'},
                    {'id': 2, 'name': 'Other catalogue', 'catalog': '2'},
                ]},
                None,
            )
            albums = ampache.get_recent_albums(0)

        assert [a['Id'] for a in albums] == ['1']

    def test_albums_with_no_catalogue_ids_stop_instead_of_analysing_everything(self, caplog):
        """A server too old to report catalogue ids must not be filtered by guesswork."""
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_target_catalog_ids', return_value={'2'}), \
             patch.object(ampache, '_request_ex') as request:
            request.return_value = ({'album': [{'id': 1, 'name': 'No catalog key'}]}, None)
            albums = ampache.get_recent_albums(1)

        assert albums == []
        assert 'AMPACHE REPORTED NO CATALOGUE IDS' in caplog.text

    def test_fetching_every_album_pages_instead_of_asking_for_limit_zero(self):
        from tasks.mediaserver import ampache

        first = [{'id': i, 'name': f'A{i}'} for i in range(ampache._PAGE_SIZE)]
        second = [{'id': 9001, 'name': 'Last'}]
        with patch.object(ampache, '_target_catalog_ids', return_value=None), \
             patch.object(ampache, '_request_ex') as request:
            request.side_effect = [({'album': first}, None), ({'album': second}, None)]
            albums = ampache.get_recent_albums(0)

        assert request.call_args_list[0][0][1]['limit'] == ampache._PAGE_SIZE
        assert request.call_args_list[1][0][1]['offset'] == ampache._PAGE_SIZE
        assert len(albums) == ampache._PAGE_SIZE + 1

    def test_a_failed_album_page_is_reported_not_treated_as_an_empty_library(self, caplog):
        from tasks.mediaserver import ampache

        with patch.object(ampache, '_target_catalog_ids', return_value=None), \
             patch.object(ampache, '_request_ex') as request:
            request.return_value = (None, {'kind': 'api', 'message': 'boom'})
            albums = ampache.get_recent_albums(5)

        assert albums == []
        assert 'AMPACHE ALBUM FETCH FAILED' in caplog.text


class TestApiVersionGate:
    """This backend targets Ampache API 8; older servers must be told, not guessed at."""

    @pytest.mark.parametrize(
        ('reported', 'expected'),
        [('8.0.0', 8), ('6.6.6', 6), ('600000', 6), ('400001', 4), ('8', 8), ('', None),
         ('nonsense', None)],
    )
    def test_api_major_reads_both_version_forms(self, reported, expected):
        from tasks.mediaserver import ampache

        assert ampache._api_major(reported) == expected

    def test_test_connection_warns_when_the_server_predates_api_8(self, configured):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'auth': 'tok', 'api': '6.6.6'}),
                _json_response({'song': [{'id': 1, 'title': 'S', 'filename': '/music/a.mp3'}]}),
            ]
            result = ampache.test_connection()

        assert result['ok'] is True
        assert any('API 8 or newer' in w for w in result['warnings'])

    def test_test_connection_is_quiet_on_api_8(self, configured):
        from tasks.mediaserver import ampache

        with patch.object(ampache, 'requests') as http:
            http.get.side_effect = [
                _json_response({'auth': 'tok', 'api': '8.0.0'}),
                _json_response({'song': [{'id': 1, 'title': 'S', 'filename': '/music/a.mp3'}]}),
            ]
            result = ampache.test_connection()

        assert result['warnings'] == []


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
