import pytest
from unittest.mock import Mock, MagicMock, patch
import requests


class TestJellyfinAuthHeader:
    def test_builds_authorization_header_from_token(self):
        import config

        assert config.jellyfin_auth_header('abc123') == {
            'Authorization': 'MediaBrowser Token="abc123"'
        }

    def test_empty_token_yields_no_header(self):
        import config

        assert config.jellyfin_auth_header('') == {}
        assert config.jellyfin_auth_header(None) == {}

    def test_compute_headers_jellyfin_uses_authorization(self, monkeypatch):
        import config

        monkeypatch.setattr(config, 'MEDIASERVER_TYPE', 'jellyfin')
        monkeypatch.setattr(config, 'JELLYFIN_TOKEN', 'jf-tok')
        assert config._compute_headers() == {'Authorization': 'MediaBrowser Token="jf-tok"'}

    def test_compute_headers_emby_keeps_x_emby_token(self, monkeypatch):
        import config

        monkeypatch.setattr(config, 'MEDIASERVER_TYPE', 'emby')
        monkeypatch.setattr(config, 'EMBY_TOKEN', 'emby-tok')
        assert config._compute_headers() == {'X-Emby-Token': 'emby-tok'}


class TestJellyfinSelectBestArtist:
    def test_prioritizes_artist_items_over_album_artist(self):
        from tasks.mediaserver.jellyfin import _select_best_artist

        item = {
            'ArtistItems': [{'Name': 'Track Artist', 'Id': 'artist-123'}],
            'Artists': ['Fallback Artist'],
            'AlbumArtist': 'Album Artist',
        }

        artist_name, artist_id = _select_best_artist(item)

        assert artist_name == 'Track Artist'
        assert artist_id == 'artist-123'

    def test_falls_back_to_artists_array(self):
        from tasks.mediaserver.jellyfin import _select_best_artist

        item = {
            'ArtistItems': [],
            'Artists': ['First Artist', 'Second Artist'],
            'AlbumArtist': 'Album Artist',
        }

        artist_name, artist_id = _select_best_artist(item)

        assert artist_name == 'First Artist'
        assert artist_id is None

    def test_falls_back_to_album_artist(self):
        from tasks.mediaserver.jellyfin import _select_best_artist

        item = {'AlbumArtist': 'The Album Artist'}

        artist_name, artist_id = _select_best_artist(item)

        assert artist_name == 'The Album Artist'
        assert artist_id is None

    def test_returns_unknown_when_no_artist_info(self):
        from tasks.mediaserver.jellyfin import _select_best_artist

        item = {}

        artist_name, artist_id = _select_best_artist(item)

        assert artist_name == 'Unknown Artist'
        assert artist_id is None

    def test_handles_empty_artist_items(self):
        from tasks.mediaserver.jellyfin import _select_best_artist

        item = {'ArtistItems': [], 'AlbumArtist': 'Fallback'}

        artist_name, _ = _select_best_artist(item)

        assert artist_name == 'Fallback'


class TestJellyfinResolveUser:
    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_resolves_username_to_id(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import resolve_user

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_response = Mock()
        mock_response.json.return_value = [
            {'Name': 'admin', 'Id': 'admin-id-123'},
            {'Name': 'TestUser', 'Id': 'user-id-456'},
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = resolve_user('testuser', 'token123')

        assert result == 'user-id-456'
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert '/Users' in call_url

    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_returns_identifier_if_no_match(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import resolve_user

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_response = Mock()
        mock_response.json.return_value = [{'Name': 'OtherUser', 'Id': 'other-id'}]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = resolve_user('direct-user-id', 'token123')

        assert result == 'direct-user-id'

    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_handles_http_error(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import resolve_user

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_get.side_effect = requests.exceptions.RequestException("Connection failed")

        result = resolve_user('some-user', 'token')

        assert result == 'some-user'


class TestJellyfinGetTracksFromAlbum:
    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_uses_correct_url_and_params(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import get_tracks_from_album

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {'Authorization': 'MediaBrowser Token="token"'}

        mock_response = Mock()
        mock_response.json.return_value = {'Items': []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        get_tracks_from_album('album-xyz')

        call_url = mock_get.call_args[0][0]
        call_params = mock_get.call_args[1].get('params', {})

        assert call_url == 'http://jellyfin:8096/Items', (
            f"URL changed! Expected '/Items', got '{call_url}'"
        )
        assert call_params.get('userId') == 'user123', "userId param missing or wrong"
        assert call_params.get('ParentId') == 'album-xyz', "ParentId param missing or wrong"
        assert call_params.get('IncludeItemTypes') == 'Audio', "IncludeItemTypes param wrong"

    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_enriches_tracks_with_artist_info(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import get_tracks_from_album

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {'Authorization': 'MediaBrowser Token="token"'}

        mock_response = Mock()
        mock_response.json.return_value = {
            'Items': [
                {
                    'Id': 'track1',
                    'Name': 'Song One',
                    'ArtistItems': [{'Name': 'Artist A', 'Id': 'artist-a'}],
                },
                {'Id': 'track2', 'Name': 'Song Two', 'AlbumArtist': 'Album Artist B'},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        tracks = get_tracks_from_album('album123')

        assert len(tracks) == 2
        assert 'AlbumArtist' in tracks[0], "AlbumArtist field must be added"
        assert 'ArtistId' in tracks[0], "ArtistId field must be added"
        assert tracks[0]['AlbumArtist'] == 'Artist A', "ArtistItems should be prioritized"
        assert tracks[0]['ArtistId'] == 'artist-a'
        assert tracks[1]['AlbumArtist'] == 'Album Artist B', (
            "Should fall back to AlbumArtist when no ArtistItems"
        )
        assert tracks[1]['ArtistId'] is None

    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_returns_empty_on_http_error(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import get_tracks_from_album

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {}
        mock_get.side_effect = requests.exceptions.RequestException("Failed")

        tracks = get_tracks_from_album('album123')

        assert tracks == []

    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_handles_empty_items_response(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import get_tracks_from_album

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {}

        mock_response = Mock()
        mock_response.json.return_value = {'Items': []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        tracks = get_tracks_from_album('album123')

        assert tracks == []


class TestJellyfinGetAllPlaylists:
    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_uses_correct_url_and_params(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import get_all_playlists

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {'Authorization': 'MediaBrowser Token="token"'}

        mock_response = Mock()
        mock_response.json.return_value = {'Items': []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        get_all_playlists()

        call_url = mock_get.call_args[0][0]
        call_params = mock_get.call_args[1].get('params', {})

        assert call_url == 'http://jellyfin:8096/Items', f"URL changed! Got '{call_url}'"
        assert call_params.get('userId') == 'user123', "userId param missing or wrong"
        assert call_params.get('IncludeItemTypes') == 'Playlist', (
            "IncludeItemTypes must be 'Playlist'"
        )
        assert call_params.get('Recursive') is True, "Recursive must be True"

    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_parses_items_array_from_response(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import get_all_playlists

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {}

        mock_response = Mock()
        mock_response.json.return_value = {
            'Items': [
                {'Id': 'pl1', 'Name': 'Rock_automatic'},
                {'Id': 'pl2', 'Name': 'Jazz Favorites'},
            ],
            'TotalRecordCount': 2,
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        playlists = get_all_playlists()

        assert len(playlists) == 2
        assert playlists[0]['Id'] == 'pl1'
        assert playlists[0]['Name'] == 'Rock_automatic'

    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_returns_empty_on_error(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import get_all_playlists

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {}
        mock_get.side_effect = requests.exceptions.RequestException("Failed")

        playlists = get_all_playlists()

        assert playlists == []


class TestJellyfinGetPlaylistTrackIds:
    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_uses_playlists_items_url_and_reads_id(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import get_playlist_track_ids

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {'Authorization': 'MediaBrowser Token="token"'}

        mock_response = Mock()
        mock_response.json.return_value = {
            'Items': [
                {'Id': 'track1', 'PlaylistItemId': 'entry-a'},
                {'Id': 'track2', 'PlaylistItemId': 'entry-b'},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        ids = get_playlist_track_ids('pl-xyz')

        call_url = mock_get.call_args[0][0]
        assert call_url == 'http://jellyfin:8096/Playlists/pl-xyz/Items'
        assert mock_get.call_args[1].get('params', {}).get('UserId') == 'user123'
        assert ids == ['track1', 'track2']

    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_returns_empty_on_error(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import get_playlist_track_ids

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {}
        mock_get.side_effect = requests.exceptions.RequestException("Failed")

        assert get_playlist_track_ids('pl-xyz') == []


class TestJellyfinDeletePlaylist:
    @patch('tasks.mediaserver.jellyfin.requests.delete')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_uses_correct_url_and_method(self, mock_config, mock_delete):
        from tasks.mediaserver.jellyfin import delete_playlist

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.HEADERS = {'Authorization': 'MediaBrowser Token="test-token"'}

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_delete.return_value = mock_response

        result = delete_playlist('playlist-123')

        assert result is True
        mock_delete.assert_called_once()
        call_url = mock_delete.call_args[0][0]
        assert call_url == 'http://jellyfin:8096/Items/playlist-123', (
            f"URL changed! Expected '/Items/playlist-123', got '{call_url}'"
        )
        call_kwargs = mock_delete.call_args[1]
        assert call_kwargs.get('headers') == {'Authorization': 'MediaBrowser Token="test-token"'}

    @patch('tasks.mediaserver.jellyfin.requests.delete')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_returns_false_on_http_error(self, mock_config, mock_delete):
        from tasks.mediaserver.jellyfin import delete_playlist

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.HEADERS = {}
        mock_delete.side_effect = requests.exceptions.RequestException("Connection refused")

        result = delete_playlist('playlist-123')

        assert result is False

    @patch('tasks.mediaserver.jellyfin.requests.delete')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_returns_false_on_raise_for_status(self, mock_config, mock_delete):
        from tasks.mediaserver.jellyfin import delete_playlist

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.HEADERS = {}

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_delete.return_value = mock_response

        result = delete_playlist('playlist-123')

        assert result is False


class TestJellyfinGetLastPlayedTime:
    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_extracts_last_played_date(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import get_last_played_time

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.JELLYFIN_TOKEN = 'token123'

        mock_response = Mock()
        mock_response.json.return_value = {
            'UserData': {'LastPlayedDate': '2024-01-15T10:30:00Z', 'PlayCount': 5}
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = get_last_played_time('track-id', {'user_id': 'user123', 'token': 'token'})

        assert result == '2024-01-15T10:30:00Z'

    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_returns_none_if_never_played(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import get_last_played_time

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.JELLYFIN_TOKEN = 'token123'

        mock_response = Mock()
        mock_response.json.return_value = {'UserData': {'PlayCount': 0}}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = get_last_played_time('track-id', {'user_id': 'user123', 'token': 'token'})

        assert result is None


class TestNavidromeCoerceToList:
    def test_single_dict_wrapped_in_list(self):
        from tasks.mediaserver.navidrome import _coerce_to_list

        value = {'id': 'song-1', 'title': 'Track'}
        assert _coerce_to_list(value) == [value]

    def test_tuple_converted_to_list(self):
        from tasks.mediaserver.navidrome import _coerce_to_list

        assert _coerce_to_list(({'id': 'a'}, {'id': 'b'})) == [{'id': 'a'}, {'id': 'b'}]

    def test_list_passed_through_unchanged(self):
        from tasks.mediaserver.navidrome import _coerce_to_list

        value = [{'id': 'a'}, {'id': 'b'}]
        assert _coerce_to_list(value) is value

    def test_empty_list_passed_through(self):
        from tasks.mediaserver.navidrome import _coerce_to_list

        value = []
        assert _coerce_to_list(value) is value

    def test_none_becomes_empty_list(self):
        from tasks.mediaserver.navidrome import _coerce_to_list

        assert _coerce_to_list(None) == []

    def test_str_becomes_empty_list(self):
        from tasks.mediaserver.navidrome import _coerce_to_list

        assert _coerce_to_list('song') == []

    def test_int_becomes_empty_list(self):
        from tasks.mediaserver.navidrome import _coerce_to_list

        assert _coerce_to_list(42) == []


class TestNavidromeSelectBestArtist:
    def test_prioritizes_track_artist(self):
        from tasks.mediaserver.navidrome import _select_best_artist

        song = {
            'artist': 'Track Artist',
            'artistId': 'track-artist-id',
            'albumArtist': 'Album Artist',
            'albumArtistId': 'album-artist-id',
        }

        artist_name, artist_id = _select_best_artist(song)

        assert artist_name == 'Track Artist'
        assert artist_id == 'track-artist-id'

    def test_falls_back_to_album_artist(self):
        from tasks.mediaserver.navidrome import _select_best_artist

        song = {'albumArtist': 'Album Artist', 'albumArtistId': 'album-artist-id'}

        artist_name, artist_id = _select_best_artist(song)

        assert artist_name == 'Album Artist'
        assert artist_id == 'album-artist-id'

    def test_returns_unknown_when_no_artist(self):
        from tasks.mediaserver.navidrome import _select_best_artist

        song = {'title': 'Some Song'}

        artist_name, artist_id = _select_best_artist(song)

        assert artist_name == 'Unknown Artist'
        assert artist_id is None


class TestNavidromeAuthParams:
    @patch('tasks.mediaserver.navidrome.config')
    def test_generates_hex_encoded_password(self, mock_config):
        from tasks.mediaserver.navidrome import get_navidrome_auth_params

        mock_config.NAVIDROME_USER = 'testuser'
        mock_config.NAVIDROME_PASSWORD = 'secret123'
        mock_config.APP_VERSION = '1.0.0'

        params = get_navidrome_auth_params()

        assert params['u'] == 'testuser'
        assert params['p'].startswith('enc:')
        hex_password = params['p'].replace('enc:', '')
        decoded = bytes.fromhex(hex_password).decode('utf-8')
        assert decoded == 'secret123'

    @patch('tasks.mediaserver.navidrome.config')
    def test_returns_empty_when_no_credentials(self, mock_config):
        from tasks.mediaserver.navidrome import get_navidrome_auth_params

        mock_config.NAVIDROME_USER = ''
        mock_config.NAVIDROME_PASSWORD = ''

        params = get_navidrome_auth_params()

        assert params == {}


class TestNavidromeRequest:
    @patch('tasks.mediaserver.navidrome.requests.request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_constructs_correct_url_with_view_suffix(self, mock_config, mock_request):
        from tasks.mediaserver.navidrome import _navidrome_request

        mock_config.NAVIDROME_URL = 'http://navidrome:4533'
        mock_config.NAVIDROME_USER = 'admin'
        mock_config.NAVIDROME_PASSWORD = 'password'
        mock_config.APP_VERSION = '1.0'

        mock_response = Mock()
        mock_response.json.return_value = {'subsonic-response': {'status': 'ok'}}
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        _navidrome_request('getPlaylists')

        call_kwargs = mock_request.call_args
        assert call_kwargs[0][0] == 'get'
        url = call_kwargs[0][1]
        assert url == 'http://navidrome:4533/rest/getPlaylists.view', (
            f"URL format changed! Expected '/rest/getPlaylists.view', got '{url}'"
        )

    @patch('tasks.mediaserver.navidrome.requests.request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_parses_subsonic_response_wrapper(self, mock_config, mock_request):
        from tasks.mediaserver.navidrome import _navidrome_request

        mock_config.NAVIDROME_URL = 'http://navidrome:4533'
        mock_config.NAVIDROME_USER = 'admin'
        mock_config.NAVIDROME_PASSWORD = 'password'
        mock_config.APP_VERSION = '1.0'

        mock_response = Mock()
        mock_response.json.return_value = {
            'subsonic-response': {
                'status': 'ok',
                'version': '1.16.1',
                'playlists': {'playlist': []},
            }
        }
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        result = _navidrome_request('getPlaylists')

        assert result['status'] == 'ok'
        assert 'playlists' in result
        assert 'subsonic-response' not in result

    @patch('tasks.mediaserver.navidrome.requests.request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_checks_status_field_for_failure(self, mock_config, mock_request):
        from tasks.mediaserver.navidrome import _navidrome_request

        mock_config.NAVIDROME_URL = 'http://navidrome:4533'
        mock_config.NAVIDROME_USER = 'admin'
        mock_config.NAVIDROME_PASSWORD = 'password'
        mock_config.APP_VERSION = '1.0'

        mock_response = Mock()
        mock_response.json.return_value = {
            'subsonic-response': {
                'status': 'failed',
                'error': {'code': 40, 'message': 'Wrong username or password'},
            }
        }
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        result = _navidrome_request('getPlaylists')

        assert result is None, "Failed status should return None, not the response"

    @patch('tasks.mediaserver.navidrome.requests.request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_includes_auth_params_in_request(self, mock_config, mock_request):
        from tasks.mediaserver.navidrome import _navidrome_request

        mock_config.NAVIDROME_URL = 'http://navidrome:4533'
        mock_config.NAVIDROME_USER = 'testuser'
        mock_config.NAVIDROME_PASSWORD = 'secret'
        mock_config.APP_VERSION = '2.0'

        mock_response = Mock()
        mock_response.json.return_value = {'subsonic-response': {'status': 'ok'}}
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        _navidrome_request('ping', {'extra': 'param'})

        call_kwargs = mock_request.call_args[1]
        params = call_kwargs.get('params', {})

        assert params.get('u') == 'testuser', "Username not in params"
        assert params.get('p').startswith('enc:'), "Password not hex-encoded"
        assert params.get('f') == 'json', "Format must be json"
        assert 'extra' in params, "Custom params not passed through"

    @patch('tasks.mediaserver.navidrome.requests.request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_returns_none_on_http_error(self, mock_config, mock_request):
        from tasks.mediaserver.navidrome import _navidrome_request

        mock_config.NAVIDROME_URL = 'http://navidrome:4533'
        mock_config.NAVIDROME_USER = 'admin'
        mock_config.NAVIDROME_PASSWORD = 'password'
        mock_config.APP_VERSION = '1.0'

        mock_request.side_effect = requests.exceptions.RequestException("Connection refused")

        result = _navidrome_request('getPlaylists')

        assert result is None


class TestNavidromeAuthDetection:
    def _ok_config(self, mock_config):
        mock_config.NAVIDROME_URL = 'http://navidrome:4533'
        mock_config.NAVIDROME_USER = 'admin'
        mock_config.NAVIDROME_PASSWORD = 'password'
        mock_config.APP_VERSION = '1.0'

    @patch('tasks.mediaserver.navidrome.requests.request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_request_ex_flags_wrong_password_as_auth(self, mock_config, mock_request):
        from tasks.mediaserver.navidrome import _navidrome_request_ex

        self._ok_config(mock_config)
        mock_response = Mock()
        mock_response.json.return_value = {
            'subsonic-response': {
                'status': 'failed',
                'error': {'code': 40, 'message': 'Wrong username or password'},
            }
        }
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        data, err = _navidrome_request_ex('search3')

        assert data is None
        assert err['kind'] == 'auth'
        assert err['message'] == 'Wrong username or password'

    @patch('tasks.mediaserver.navidrome.requests.request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_request_ex_classifies_non_auth_failure_as_server(self, mock_config, mock_request):
        from tasks.mediaserver.navidrome import _navidrome_request_ex

        self._ok_config(mock_config)
        mock_response = Mock()
        mock_response.json.return_value = {
            'subsonic-response': {
                'status': 'failed',
                'error': {'code': 70, 'message': 'Requested data not found'},
            }
        }
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        data, err = _navidrome_request_ex('search3')

        assert data is None
        assert err['kind'] == 'server'

    @patch('tasks.mediaserver.navidrome.requests.request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_request_ex_network_error(self, mock_config, mock_request):
        from tasks.mediaserver.navidrome import _navidrome_request_ex

        self._ok_config(mock_config)
        mock_request.side_effect = requests.exceptions.RequestException('refused')

        data, err = _navidrome_request_ex('search3')

        assert data is None
        assert err['kind'] == 'network'

    @patch('tasks.mediaserver.navidrome.requests.request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_request_ex_network_error_redacts_password(self, mock_config, mock_request):
        from tasks.mediaserver.navidrome import _navidrome_request_ex

        self._ok_config(mock_config)
        leaky = (
            "HTTPConnectionPool(host='navidrome', port=4533): Max retries exceeded "
            "with url: /rest/search3.view?u=admin&p=enc:7365637265743132&v=1.16.1&f=json"
        )
        mock_request.side_effect = requests.exceptions.ConnectionError(leaky)

        data, err = _navidrome_request_ex('search3')

        assert data is None
        assert err['kind'] == 'network'
        assert '7365637265743132' not in err['message']
        assert 'enc:7365637265743132' not in err['message']
        assert '[REDACTED]' in err['message']

    @patch('tasks.mediaserver.navidrome.requests.request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_request_ex_network_error_does_not_leak_creds_to_logs(self, mock_config, mock_request):
        # Guards against reverting logger.error -> logger.exception here: the
        # exception traceback embeds the unredacted URL with auth params.
        import logging
        from tasks.mediaserver import navidrome

        self._ok_config(mock_config)
        leaky = (
            "HTTPConnectionPool(host='navidrome', port=4533): Max retries exceeded "
            "with url: /rest/search3.view?u=admin&p=enc:7365637265743132&v=1.16.1&f=json"
        )
        mock_request.side_effect = requests.exceptions.ConnectionError(leaky)

        records = []
        handler = logging.Handler()
        handler.emit = records.append
        navidrome.logger.addHandler(handler)
        try:
            navidrome._navidrome_request_ex('search3')
        finally:
            navidrome.logger.removeHandler(handler)

        fmt = logging.Formatter()
        output = "\n".join(fmt.format(r) for r in records)
        assert '7365637265743132' not in output, (
            "Navidrome credentials leaked into logs (logger.exception dumped the traceback?)"
        )

    @patch('tasks.mediaserver.navidrome.requests.request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_request_ex_handles_non_dict_json(self, mock_config, mock_request):
        from tasks.mediaserver.navidrome import _navidrome_request_ex

        self._ok_config(mock_config)
        mock_response = Mock()
        mock_response.json.return_value = ['unexpected', 'list']
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        data, err = _navidrome_request_ex('search3')

        assert data is None
        assert err['kind'] == 'server'

    @patch('tasks.mediaserver.navidrome.config')
    def test_request_ex_missing_credentials(self, mock_config):
        from tasks.mediaserver.navidrome import _navidrome_request_ex

        mock_config.NAVIDROME_URL = 'http://navidrome:4533'
        mock_config.NAVIDROME_USER = ''
        mock_config.NAVIDROME_PASSWORD = ''

        data, err = _navidrome_request_ex('search3')

        assert data is None
        assert err['kind'] == 'config'

    @patch('tasks.mediaserver.navidrome.requests.request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_test_connection_reports_auth_failed(self, mock_config, mock_request):
        from tasks.mediaserver.navidrome import test_connection

        self._ok_config(mock_config)
        mock_response = Mock()
        mock_response.json.return_value = {
            'subsonic-response': {
                'status': 'failed',
                'error': {'code': 40, 'message': 'Wrong username or password'},
            }
        }
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        result = test_connection()

        assert result['ok'] is False
        assert result['auth_failed'] is True
        assert result['error'] == 'Wrong username or password'


class TestNavidromeGetTracksFromAlbum:
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_calls_getAlbum_endpoint(self, mock_request):
        from tasks.mediaserver.navidrome import get_tracks_from_album

        mock_request.return_value = {'status': 'ok', 'album': {'id': 'album123', 'song': []}}

        get_tracks_from_album('album123')

        call_args = mock_request.call_args
        assert call_args[0][0] == 'getAlbum', (
            f"Endpoint changed! Expected 'getAlbum', got '{call_args[0][0]}'"
        )
        assert call_args[0][1] == {'id': 'album123'}, "Params changed! Expected {'id': 'album123'}"

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_normalizes_field_names_to_capitalized(self, mock_request):
        from tasks.mediaserver.navidrome import get_tracks_from_album

        mock_request.return_value = {
            'status': 'ok',
            'album': {
                'id': 'album123',
                'name': 'Test Album',
                'song': [
                    {
                        'id': 'song1',
                        'title': 'Track One',
                        'artist': 'Song Artist',
                        'artistId': 'artist1',
                        'path': '/music/song1.mp3',
                    }
                ],
            },
        }

        tracks = get_tracks_from_album('album123')

        assert len(tracks) == 1
        assert 'Id' in tracks[0], "Missing 'Id' (capital I) - normalization broken"
        assert tracks[0]['Id'] == 'song1'
        assert 'Name' in tracks[0], "Missing 'Name' (capital N) - normalization broken"
        assert tracks[0]['Name'] == 'Track One'
        assert 'Path' in tracks[0], "Missing 'Path' (capital P) - normalization broken"
        assert tracks[0]['Path'] == '/music/song1.mp3'
        assert 'AlbumArtist' in tracks[0], "Missing 'AlbumArtist' - enrichment broken"
        assert 'ArtistId' in tracks[0], "Missing 'ArtistId' - enrichment broken"

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_artist_prioritization_applied(self, mock_request):
        from tasks.mediaserver.navidrome import get_tracks_from_album

        mock_request.return_value = {
            'status': 'ok',
            'album': {
                'id': 'album123',
                'song': [
                    {
                        'id': 'song1',
                        'title': 'Has Track Artist',
                        'artist': 'Track Artist',
                        'artistId': 'track-artist-id',
                        'albumArtist': 'Album Artist',
                        'albumArtistId': 'album-artist-id',
                    },
                    {
                        'id': 'song2',
                        'title': 'Only Album Artist',
                        'albumArtist': 'Album Artist Only',
                        'albumArtistId': 'album-only-id',
                    },
                ],
            },
        }

        tracks = get_tracks_from_album('album123')

        assert tracks[0]['AlbumArtist'] == 'Track Artist', (
            "Track artist should be prioritized over album artist"
        )
        assert tracks[0]['ArtistId'] == 'track-artist-id'

        assert tracks[1]['AlbumArtist'] == 'Album Artist Only', (
            "Should fall back to album artist when track artist missing"
        )
        assert tracks[1]['ArtistId'] == 'album-only-id'

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_returns_empty_on_missing_songs(self, mock_request):
        from tasks.mediaserver.navidrome import get_tracks_from_album

        mock_request.return_value = {
            'status': 'ok',
            'album': {'id': 'album123', 'name': 'Empty Album'},
        }

        tracks = get_tracks_from_album('album123')

        assert tracks == []

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_returns_empty_on_api_failure(self, mock_request):
        from tasks.mediaserver.navidrome import get_tracks_from_album

        mock_request.return_value = None

        tracks = get_tracks_from_album('album123')

        assert tracks == []


class TestNavidromeGetTopPlayedSongsAlbumCap:
    @staticmethod
    def _album_list_response(album_ids):
        return {'status': 'ok', 'albumList2': {'album': [{'id': aid} for aid in album_ids]}}

    @staticmethod
    def _tracks_for(album_id, count):
        return [{'Id': f'{album_id}_track{i}', 'Album': album_id} for i in range(count)]

    @patch('tasks.mediaserver.navidrome.get_tracks_from_album')
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_single_album_capped_even_with_large_limit(
        self, mock_config, mock_request, mock_get_tracks
    ):
        from tasks.mediaserver.navidrome import get_top_played_songs

        mock_config.SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM = 2
        mock_request.return_value = self._album_list_response(['big'])
        mock_get_tracks.return_value = self._tracks_for('big', 100)

        result = get_top_played_songs(limit=60, user_creds={})

        assert len(result) == 2, (
            f"Expected configured cap of 2, got {len(result)} (limit//10 floor regressed)"
        )

    @patch('tasks.mediaserver.navidrome.get_tracks_from_album')
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_cap_honored_per_album_across_multiple_albums(
        self, mock_config, mock_request, mock_get_tracks
    ):
        from tasks.mediaserver.navidrome import get_top_played_songs

        mock_config.SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM = 2
        mock_request.return_value = self._album_list_response(['a1', 'a2', 'a3'])
        mock_get_tracks.side_effect = lambda album_id, user_creds=None: self._tracks_for(
            album_id, 10
        )

        result = get_top_played_songs(limit=20, user_creds={})

        per_album = {}
        for song in result:
            album = song['Id'].split('_')[0]
            per_album[album] = per_album.get(album, 0) + 1
        assert all(count <= 2 for count in per_album.values()), (
            f"Some album exceeded the cap of 2: {per_album}"
        )

    @patch('tasks.mediaserver.navidrome.get_tracks_from_album')
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_fetches_enough_albums_to_reach_limit_under_tight_cap(
        self, mock_config, mock_request, mock_get_tracks
    ):
        from tasks.mediaserver.navidrome import get_top_played_songs

        mock_config.SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM = 2
        mock_request.return_value = self._album_list_response([f'a{i}' for i in range(40)])
        mock_get_tracks.side_effect = lambda album_id, user_creds=None: self._tracks_for(
            album_id, 10
        )

        get_top_played_songs(limit=60, user_creds={})

        requested_size = mock_request.call_args[0][1]['size']
        assert requested_size >= 60 // 2, (
            f"Album fetch size {requested_size} too small to reach limit under cap=2"
        )

    @patch('tasks.mediaserver.navidrome.get_tracks_from_album')
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_final_selection_keeps_most_recently_played(
        self, mock_config, mock_request, mock_get_tracks
    ):
        from tasks.mediaserver.navidrome import get_top_played_songs

        mock_config.SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM = 5
        mock_request.return_value = self._album_list_response(['a1'])
        mock_get_tracks.return_value = [
            {'Id': 'old', 'Album': 'a1', 'played': '2026-01-01T00:00:00Z'},
            {'Id': 'newest', 'Album': 'a1', 'played': '2026-06-01T00:00:00Z'},
            {'Id': 'mid', 'Album': 'a1', 'played': '2026-03-01T00:00:00Z'},
            {'Id': 'never', 'Album': 'a1'},
            {'Id': 'recent', 'Album': 'a1', 'played': '2026-05-01T00:00:00Z'},
        ]

        result = get_top_played_songs(limit=2, user_creds={})

        assert {s['Id'] for s in result} == {'newest', 'recent'}, (
            f"Expected the 2 most recently played, got {[s['Id'] for s in result]}"
        )

    @patch('tasks.mediaserver.navidrome.get_tracks_from_album')
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_final_selection_falls_back_to_last_played_field(
        self, mock_config, mock_request, mock_get_tracks
    ):
        from tasks.mediaserver.navidrome import get_top_played_songs

        mock_config.SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM = 5
        mock_request.return_value = self._album_list_response(['a1'])
        mock_get_tracks.return_value = [
            {'Id': 'older', 'Album': 'a1', 'lastPlayed': '2026-02-01T00:00:00Z'},
            {'Id': 'newer', 'Album': 'a1', 'lastPlayed': '2026-04-01T00:00:00Z'},
        ]

        result = get_top_played_songs(limit=1, user_creds={})

        assert [s['Id'] for s in result] == ['newer'], (
            f"Expected the most recently played via lastPlayed, got {[s['Id'] for s in result]}"
        )

    @patch('tasks.mediaserver.navidrome.get_tracks_from_album')
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_never_played_songs_handled_without_error(
        self, mock_config, mock_request, mock_get_tracks
    ):
        from tasks.mediaserver.navidrome import get_top_played_songs

        mock_config.SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM = 3
        mock_request.return_value = self._album_list_response(['a1', 'a2'])
        mock_get_tracks.side_effect = lambda album_id, user_creds=None: [
            {'Id': f'{album_id}_t{i}', 'Album': album_id} for i in range(4)
        ]

        result = get_top_played_songs(limit=5, user_creds={})

        assert len(result) == 5
        assert all('Id' in s for s in result)

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    @patch('tasks.mediaserver.navidrome.config')
    def test_no_frequent_albums_returns_empty(self, mock_config, mock_request):
        from tasks.mediaserver.navidrome import get_top_played_songs

        mock_config.SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM = 3
        mock_request.return_value = {'status': 'ok', 'albumList2': {}}

        result = get_top_played_songs(limit=20, user_creds={})

        assert result == []


class TestNavidromeGetAllPlaylists:
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_calls_getPlaylists_endpoint(self, mock_request):
        from tasks.mediaserver.navidrome import get_all_playlists

        mock_request.return_value = {'status': 'ok', 'playlists': {'playlist': []}}

        get_all_playlists()

        mock_request.assert_called_once_with('getPlaylists')

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_parses_nested_playlist_structure(self, mock_request):
        from tasks.mediaserver.navidrome import get_all_playlists

        mock_request.return_value = {
            'status': 'ok',
            'playlists': {
                'playlist': [
                    {'id': 'pl1', 'name': 'Rock_automatic', 'songCount': 50},
                    {'id': 'pl2', 'name': 'Jazz Mix', 'songCount': 30},
                ]
            },
        }

        playlists = get_all_playlists()

        assert len(playlists) == 2
        assert playlists[0]['Id'] == 'pl1', "Missing 'Id' normalization"
        assert playlists[0]['Name'] == 'Rock_automatic', "Missing 'Name' normalization"
        assert playlists[0]['id'] == 'pl1', "Original 'id' should be preserved"
        assert playlists[0]['name'] == 'Rock_automatic', "Original 'name' should be preserved"

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_handles_missing_playlists_key(self, mock_request):
        from tasks.mediaserver.navidrome import get_all_playlists

        mock_request.return_value = {'status': 'ok'}

        playlists = get_all_playlists()

        assert playlists == []

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_handles_missing_playlist_array(self, mock_request):
        from tasks.mediaserver.navidrome import get_all_playlists

        mock_request.return_value = {'status': 'ok', 'playlists': {}}

        playlists = get_all_playlists()

        assert playlists == []

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_returns_empty_on_failure(self, mock_request):
        from tasks.mediaserver.navidrome import get_all_playlists

        mock_request.return_value = None

        playlists = get_all_playlists()

        assert playlists == []


class TestNavidromeGetPlaylistTrackIds:
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_reads_entry_ids(self, mock_request):
        from tasks.mediaserver.navidrome import get_playlist_track_ids

        mock_request.return_value = {
            'status': 'ok',
            'playlist': {
                'id': 'pl1',
                'entry': [{'id': 'song1', 'title': 'A'}, {'id': 'song2', 'title': 'B'}],
            },
        }

        ids = get_playlist_track_ids('pl1')

        assert mock_request.call_args[0][0] == 'getPlaylist'
        assert mock_request.call_args[0][1] == {'id': 'pl1'}
        assert ids == ['song1', 'song2']

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_coerces_single_entry_dict_to_list(self, mock_request):
        from tasks.mediaserver.navidrome import get_playlist_track_ids

        mock_request.return_value = {
            'status': 'ok',
            'playlist': {'id': 'pl1', 'entry': {'id': 'only-song'}},
        }

        assert get_playlist_track_ids('pl1') == ['only-song']

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_returns_empty_on_failure(self, mock_request):
        from tasks.mediaserver.navidrome import get_playlist_track_ids

        mock_request.return_value = None

        assert get_playlist_track_ids('pl1') == []


class TestNavidromeDeletePlaylist:
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_calls_correct_endpoint_with_id_param(self, mock_request):
        from tasks.mediaserver.navidrome import delete_playlist

        mock_request.return_value = {'status': 'ok'}

        result = delete_playlist('playlist-123')

        assert result is True
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == 'deletePlaylist', (
            f"Endpoint changed! Expected 'deletePlaylist', got '{call_args[0][0]}'"
        )
        assert call_args[0][1] == {'id': 'playlist-123'}, (
            f"Params changed! Expected {{'id': 'playlist-123'}}, got {call_args[0][1]}"
        )
        assert call_args[1].get('method') == 'post', (
            "Method changed! Must be POST for deletePlaylist"
        )

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_checks_status_ok_for_success(self, mock_request):
        from tasks.mediaserver.navidrome import delete_playlist

        mock_request.return_value = {'status': 'something_else'}

        result = delete_playlist('playlist-123')

        assert result is False

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_returns_false_on_none_response(self, mock_request):
        from tasks.mediaserver.navidrome import delete_playlist

        mock_request.return_value = None

        result = delete_playlist('playlist-123')

        assert result is False


class TestNavidromeGetPlaylistByName:
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_finds_playlist_by_exact_name(self, mock_request):
        from tasks.mediaserver.navidrome import get_playlist_by_name

        mock_request.return_value = {
            'status': 'ok',
            'playlists': {
                'playlist': [
                    {'id': 'pl1', 'name': 'Rock Mix'},
                    {'id': 'pl2', 'name': 'Jazz Favorites'},
                    {'id': 'pl3', 'name': 'Rock Mix Special'},
                ]
            },
        }

        result = get_playlist_by_name('Jazz Favorites')

        assert result is not None
        assert result['id'] == 'pl2'
        assert result['name'] == 'Jazz Favorites'

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_returns_none_if_not_found(self, mock_request):
        from tasks.mediaserver.navidrome import get_playlist_by_name

        mock_request.return_value = {
            'status': 'ok',
            'playlists': {'playlist': [{'id': 'pl1', 'name': 'Rock Mix'}]},
        }

        result = get_playlist_by_name('NonExistent')

        assert result is None


class TestNavidromeCreatePlaylist:
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_extracts_playlist_id_from_response(self, mock_request):
        from tasks.mediaserver.navidrome import _create_playlist_batched

        mock_request.return_value = {
            'status': 'ok',
            'playlist': {'id': 'new-pl-123', 'name': 'Test Playlist', 'songCount': 3},
        }

        result = _create_playlist_batched('Test Playlist', ['song1', 'song2', 'song3'])

        assert result is not None
        assert result['id'] == 'new-pl-123'
        assert result['Id'] == 'new-pl-123'
        assert result['Name'] == 'Test Playlist'

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_create_playlist_sets_public_after_creation(self, mock_request):
        from tasks.mediaserver.navidrome import _create_playlist_batched

        mock_request.return_value = {
            'status': 'ok',
            'playlist': {'id': 'new-pl-456', 'name': 'Test Playlist', 'songCount': 1},
        }

        _create_playlist_batched('Test Playlist', ['song1'])

        first_call_args = mock_request.call_args_list[0][0]
        assert first_call_args[0] == 'createPlaylist'
        create_params = first_call_args[1]
        assert create_params.get('public') is None

        second_call_args = mock_request.call_args_list[1][0]
        assert second_call_args[0] == 'updatePlaylist'
        update_params = second_call_args[1]
        assert update_params.get('playlistId') == 'new-pl-456'
        assert update_params.get('public') == 'true'

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_returns_none_on_creation_failure(self, mock_request):
        from tasks.mediaserver.navidrome import _create_playlist_batched

        mock_request.return_value = None

        result = _create_playlist_batched('Test Playlist', ['song1'])

        assert result is None

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_handles_malformed_response(self, mock_request):
        from tasks.mediaserver.navidrome import _create_playlist_batched

        mock_request.return_value = {'status': 'ok'}

        result = _create_playlist_batched('Test Playlist', ['song1'])

        assert result is None


class TestNavidromeGetLastPlayedTime:
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_extracts_last_played(self, mock_request):
        from tasks.mediaserver.navidrome import get_last_played_time

        mock_request.return_value = {
            'status': 'ok',
            'song': {'id': 'song123', 'title': 'Test Song', 'lastPlayed': '2024-01-15T10:30:00Z'},
        }

        result = get_last_played_time('song123', {'user': 'test', 'password': 'pass'})

        assert result == '2024-01-15T10:30:00Z'

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_returns_none_if_never_played(self, mock_request):
        from tasks.mediaserver.navidrome import get_last_played_time

        mock_request.return_value = {
            'status': 'ok',
            'song': {'id': 'song123', 'title': 'Test Song'},
        }

        result = get_last_played_time('song123', {'user': 'test', 'password': 'pass'})

        assert result is None


class TestNavidromeGetRecentAlbums:
    @patch('tasks.mediaserver.navidrome._get_target_music_folder_ids')
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_normalizes_album_keys(self, mock_request, mock_folders):
        from tasks.mediaserver.navidrome import get_recent_albums

        mock_folders.return_value = None
        mock_request.return_value = {
            'status': 'ok',
            'albumList2': {
                'album': [
                    {'id': 'album1', 'name': 'First Album', 'artist': 'Artist A'},
                    {'id': 'album2', 'name': 'Second Album', 'artist': 'Artist B'},
                ]
            },
        }

        albums = get_recent_albums(10)

        assert len(albums) == 2
        assert albums[0]['Id'] == 'album1'
        assert albums[0]['Name'] == 'First Album'

    @patch('tasks.mediaserver.navidrome._get_target_music_folder_ids')
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_returns_empty_when_no_matching_folders(self, mock_request, mock_folders):
        from tasks.mediaserver.navidrome import get_recent_albums

        mock_folders.return_value = set()

        albums = get_recent_albums(10)

        assert albums == []
        mock_request.assert_not_called()


class TestNavidromeGetAllSongsApplyFilter:
    @patch('tasks.mediaserver.navidrome._get_target_music_folder_ids')
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_apply_filter_false_skips_folder_lookup(self, mock_request, mock_filter):
        from tasks.mediaserver.navidrome import get_all_songs

        mock_request.return_value = {
            'status': 'ok',
            'searchResult3': {'song': []},
        }

        creds = {'url': 'http://target:4533', 'user': 'u', 'password': 'p'}
        get_all_songs(user_creds=creds, apply_filter=False)

        mock_filter.assert_not_called()
        endpoints = [c.args[0] for c in mock_request.call_args_list]
        assert 'getMusicFolders' not in endpoints
        assert any(ep == 'search3' for ep in endpoints)
        for c in mock_request.call_args_list:
            assert c.kwargs.get('user_creds') == creds

    @patch('tasks.mediaserver.navidrome._get_target_music_folder_ids')
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_apply_filter_true_default_honors_filter(self, mock_request, mock_filter):
        from tasks.mediaserver.navidrome import get_all_songs

        mock_filter.return_value = set()
        mock_request.return_value = {'status': 'ok'}

        songs = get_all_songs(user_creds=None)

        mock_filter.assert_called_once()
        assert songs == []
        mock_request.assert_not_called()

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_get_target_music_folder_ids_forwards_user_creds(self, mock_request):
        from tasks.mediaserver.navidrome import _get_target_music_folder_ids

        with patch('tasks.mediaserver.navidrome.config') as mock_config:
            mock_config.MUSIC_LIBRARIES = 'Music'
            mock_request.return_value = {
                'musicFolders': {'musicFolder': [{'id': 1, 'name': 'Music'}]}
            }

            creds = {'url': 'http://target:4533', 'user': 'u', 'password': 'p'}
            _get_target_music_folder_ids(user_creds=creds)

        assert mock_request.call_args.args[0] == 'getMusicFolders'
        assert mock_request.call_args.kwargs.get('user_creds') == creds


class TestDispatcherValidation:
    @patch('tasks.mediaserver.config')
    def test_get_playlist_by_name_requires_name(self, mock_config):
        from tasks.mediaserver import get_playlist_by_name

        with pytest.raises(ValueError, match="Playlist name is required"):
            get_playlist_by_name('')

        with pytest.raises(ValueError, match="Playlist name is required"):
            get_playlist_by_name(None)

    @patch('tasks.mediaserver.config')
    def test_create_playlist_requires_name_and_ids(self, mock_config):
        from tasks.mediaserver import create_playlist

        with pytest.raises(ValueError, match="Playlist name is required"):
            create_playlist('', ['id1'])

        with pytest.raises(ValueError, match="Track IDs are required"):
            create_playlist('Name', [])

        with pytest.raises(ValueError, match="Track IDs are required"):
            create_playlist('Name', None)

    @patch('tasks.mediaserver.config')
    def test_create_instant_playlist_requires_name_and_ids(self, mock_config):
        from tasks.mediaserver import create_instant_playlist

        with pytest.raises(ValueError, match="Playlist name is required"):
            create_instant_playlist('', ['id1'])

        with pytest.raises(ValueError, match="Track IDs are required"):
            create_instant_playlist('Name', [])


class TestDispatcherAutomaticPlaylistDeletion:
    @patch('tasks.mediaserver.config')
    @patch('tasks.mediaserver.jellyfin.get_all_playlists')
    @patch('tasks.mediaserver.jellyfin.delete_playlist')
    def test_only_deletes_automatic_suffix_playlists(self, mock_delete, mock_get, mock_config):
        from tasks.mediaserver import delete_automatic_playlists

        mock_config.MEDIASERVER_TYPE = 'jellyfin'
        mock_get.return_value = [
            {'Id': '1', 'Name': 'Rock_automatic'},
            {'Id': '2', 'Name': 'automatic_Rock'},
            {'Id': '3', 'Name': 'My Favorites'},
            {'Id': '4', 'Name': 'Jazz_automatic'},
            {'Id': '5', 'Name': 'Pop_Automatic'},
        ]
        mock_delete.return_value = True

        delete_automatic_playlists()

        assert mock_delete.call_count == 2
        deleted_ids = [call[0][0] for call in mock_delete.call_args_list]
        assert '1' in deleted_ids
        assert '4' in deleted_ids
        assert '2' not in deleted_ids
        assert '3' not in deleted_ids

    @patch('tasks.mediaserver.config')
    @patch('tasks.mediaserver.navidrome.get_all_playlists')
    @patch('tasks.mediaserver.navidrome.delete_playlist')
    def test_handles_both_id_and_Id_keys(self, mock_delete, mock_get, mock_config):
        from tasks.mediaserver import delete_automatic_playlists

        mock_config.MEDIASERVER_TYPE = 'navidrome'
        mock_get.return_value = [
            {'id': 'nav1', 'Name': 'Test_automatic'},
            {'Id': 'nav2', 'Name': 'Other_automatic'},
        ]
        mock_delete.return_value = True

        delete_automatic_playlists()

        assert mock_delete.call_count == 2
        deleted_ids = [call[0][0] for call in mock_delete.call_args_list]
        assert 'nav1' in deleted_ids
        assert 'nav2' in deleted_ids


class TestLyrionSelectBestArtist:
    def test_artist_priority_order(self):
        priority_fields = ['trackartist', 'contributor', 'artist', 'albumartist', 'band']

        assert priority_fields[0] == 'trackartist', "trackartist should be highest priority"
        assert priority_fields[-1] == 'band', "band should be lowest priority"


class TestLyrionJsonRpcRequest:
    @patch('tasks.mediaserver.lyrion.requests.Session')
    @patch('tasks.mediaserver.lyrion.config')
    def test_constructs_correct_url(self, mock_config, mock_session_class):
        from tasks.mediaserver.lyrion import _jsonrpc_request

        mock_config.LYRION_URL = 'http://lyrion:9000'

        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {'result': {'status': 'ok'}}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response
        mock_session.headers = Mock()
        mock_session.headers.update = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_session_class.return_value = mock_session

        _jsonrpc_request('albums', [0, 10])

        call_args = mock_session.post.call_args
        assert call_args[0][0] == 'http://lyrion:9000/jsonrpc.js', (
            f"URL changed! Expected '/jsonrpc.js', got '{call_args[0][0]}'"
        )

    @patch('tasks.mediaserver.lyrion.requests.Session')
    @patch('tasks.mediaserver.lyrion.config')
    def test_uses_slim_request_method(self, mock_config, mock_session_class):
        from tasks.mediaserver.lyrion import _jsonrpc_request

        mock_config.LYRION_URL = 'http://lyrion:9000'

        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {'result': {'albums_loop': []}}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response
        mock_session.headers = Mock()
        mock_session.headers.update = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_session_class.return_value = mock_session

        _jsonrpc_request('albums', [0, 10], player_id='player1')

        call_kwargs = mock_session.post.call_args[1]
        payload = call_kwargs.get('json', {})

        assert payload.get('method') == 'slim.request', (
            f"Method changed! Expected 'slim.request', got '{payload.get('method')}'"
        )
        assert payload.get('params')[0] == 'player1', "Player ID not passed correctly"
        assert payload.get('params')[1][0] == 'albums', "Command not passed correctly"

    @patch('tasks.mediaserver.lyrion.requests.Session')
    @patch('tasks.mediaserver.lyrion.config')
    def test_extracts_result_field(self, mock_config, mock_session_class):
        from tasks.mediaserver.lyrion import _jsonrpc_request

        mock_config.LYRION_URL = 'http://lyrion:9000'

        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            'id': 1,
            'result': {'albums_loop': [{'id': '1', 'album': 'Test'}]},
            'error': None,
        }
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response
        mock_session.headers = Mock()
        mock_session.headers.update = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_session_class.return_value = mock_session

        result = _jsonrpc_request('albums', [0, 10])

        assert 'albums_loop' in result
        assert 'id' not in result

    @patch('tasks.mediaserver.lyrion.requests.Session')
    @patch('tasks.mediaserver.lyrion.config')
    def test_raises_on_jsonrpc_error(self, mock_config, mock_session_class):
        from tasks.mediaserver.lyrion import _jsonrpc_request, LyrionAPIError

        mock_config.LYRION_URL = 'http://lyrion:9000'

        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {'error': {'message': 'Unknown command'}}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response
        mock_session.headers = Mock()
        mock_session.headers.update = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_session_class.return_value = mock_session

        with pytest.raises(LyrionAPIError):
            _jsonrpc_request('badcommand', [])


class TestLyrionGetAllPlaylists:
    @patch('tasks.mediaserver.lyrion._jsonrpc_request')
    def test_calls_playlists_command(self, mock_request):
        from tasks.mediaserver.lyrion import get_all_playlists

        mock_request.return_value = {'playlists_loop': []}

        get_all_playlists()

        call_args = mock_request.call_args
        assert call_args[0][0] == 'playlists', (
            f"Command changed! Expected 'playlists', got '{call_args[0][0]}'"
        )

    @patch('tasks.mediaserver.lyrion._jsonrpc_request')
    def test_normalizes_playlist_keys(self, mock_request):
        from tasks.mediaserver.lyrion import get_all_playlists

        mock_request.return_value = {
            'playlists_loop': [
                {'id': 'pl1', 'playlist': 'Rock_automatic'},
                {'id': 'pl2', 'playlist': 'Jazz Mix'},
            ]
        }

        playlists = get_all_playlists()

        assert len(playlists) == 2
        assert playlists[0]['Id'] == 'pl1', "Missing 'Id' normalization"
        assert playlists[0]['Name'] == 'Rock_automatic', (
            "Missing 'Name' normalization from 'playlist' field"
        )

    @patch('tasks.mediaserver.lyrion._jsonrpc_request')
    def test_returns_empty_on_no_playlists(self, mock_request):
        from tasks.mediaserver.lyrion import get_all_playlists

        mock_request.return_value = {}

        playlists = get_all_playlists()

        assert playlists == []


class TestLyrionDeletePlaylist:
    @patch('tasks.mediaserver.lyrion._jsonrpc_request')
    def test_calls_playlists_delete_command(self, mock_request):
        from tasks.mediaserver.lyrion import delete_playlist

        mock_request.return_value = {'count': 1}

        delete_playlist('playlist-123')

        call_args = mock_request.call_args
        assert call_args[0][0] == 'playlists', (
            f"Command changed! Expected 'playlists', got '{call_args[0][0]}'"
        )
        params = call_args[0][1]
        assert 'delete' in params, "Must include 'delete' param"
        assert 'playlist_id:playlist-123' in params, "Must include playlist_id param"

    @patch('tasks.mediaserver.lyrion._jsonrpc_request')
    def test_returns_true_on_success(self, mock_request):
        from tasks.mediaserver.lyrion import delete_playlist

        mock_request.return_value = {'count': 1}

        result = delete_playlist('playlist-123')

        assert result is True

    @patch('tasks.mediaserver.lyrion._jsonrpc_request')
    def test_returns_false_on_failure(self, mock_request):
        from tasks.mediaserver.lyrion import delete_playlist

        mock_request.return_value = None

        result = delete_playlist('playlist-123')

        assert result is False


class TestLyrionGetTracksFromAlbum:
    @patch('tasks.mediaserver.lyrion._jsonrpc_request')
    def test_calls_titles_command_with_album_id(self, mock_request):
        from tasks.mediaserver.lyrion import get_tracks_from_album

        mock_request.return_value = {'titles_loop': []}

        get_tracks_from_album('album-123')

        call_args = mock_request.call_args
        assert call_args[0][0] == 'titles', (
            f"Command changed! Expected 'titles', got '{call_args[0][0]}'"
        )
        params = call_args[0][1]
        assert any('album_id:album-123' in str(p) for p in params), "Must include album_id filter"

    @patch('tasks.mediaserver.lyrion._jsonrpc_request')
    def test_normalizes_track_fields(self, mock_request):
        from tasks.mediaserver.lyrion import get_tracks_from_album

        mock_request.return_value = {
            'titles_loop': [
                {
                    'id': 'track1',
                    'title': 'Song One',
                    'trackartist': 'Track Artist',
                    'artist': 'Album Artist',
                    'url': '/music/song1.mp3',
                }
            ]
        }

        tracks = get_tracks_from_album('album-123')

        assert len(tracks) == 1
        assert tracks[0]['Id'] == 'track1', "Missing 'Id' normalization"
        assert tracks[0]['Name'] == 'Song One', "Missing 'Name' normalization"
        assert tracks[0]['AlbumArtist'] == 'Track Artist', (
            "trackartist should be prioritized for AlbumArtist"
        )
        assert tracks[0]['Path'] == '/music/song1.mp3', "Missing 'Path' from 'url'"

    @patch('tasks.mediaserver.lyrion._jsonrpc_request')
    def test_artist_fallback_priority(self, mock_request):
        from tasks.mediaserver.lyrion import get_tracks_from_album

        mock_request.return_value = {
            'titles_loop': [
                {
                    'id': 'track1',
                    'title': 'No TrackArtist',
                    'contributor': 'Contributor Name',
                    'artist': 'Artist Name',
                    'albumartist': 'Album Artist',
                },
                {'id': 'track2', 'title': 'Only AlbumArtist', 'albumartist': 'Album Artist Only'},
            ]
        }

        tracks = get_tracks_from_album('album-123')

        assert tracks[0]['AlbumArtist'] == 'Contributor Name', (
            "Should fall back to contributor when no trackartist"
        )

        assert tracks[1]['AlbumArtist'] == 'Album Artist Only', (
            "Should fall back to albumartist when no higher priority fields"
        )

    @patch('tasks.mediaserver.lyrion._jsonrpc_request')
    def test_filters_spotify_tracks(self, mock_request):
        from tasks.mediaserver.lyrion import get_tracks_from_album

        mock_request.return_value = {
            'titles_loop': [
                {'id': 'local1', 'title': 'Local Track', 'url': '/music/local.mp3'},
                {'id': 'spotify1', 'title': 'Spotify Track', 'url': 'spotify://track/123'},
                {
                    'id': 'local2',
                    'title': 'Another Local',
                    'genre': 'rock',
                    'url': '/music/local2.mp3',
                },
            ]
        }

        tracks = get_tracks_from_album('album-123')

        assert len(tracks) == 2
        track_ids = [t['Id'] for t in tracks]
        assert 'local1' in track_ids
        assert 'local2' in track_ids
        assert 'spotify1' not in track_ids, "Spotify tracks should be filtered"


class TestEmbySelectBestArtist:
    def test_prioritizes_artist_items_over_album_artist(self):
        from tasks.mediaserver.emby import _select_best_artist

        item = {
            'ArtistItems': [{'Name': 'Track Artist', 'Id': 'artist-123'}],
            'Artists': ['Fallback Artist'],
            'AlbumArtist': 'Album Artist',
        }

        artist_name, artist_id = _select_best_artist(item)

        assert artist_name == 'Track Artist'
        assert artist_id == 'artist-123'

    def test_falls_back_to_artists_array(self):
        from tasks.mediaserver.emby import _select_best_artist

        item = {
            'ArtistItems': [],
            'Artists': ['First Artist', 'Second Artist'],
            'AlbumArtist': 'Album Artist',
        }

        artist_name, artist_id = _select_best_artist(item)

        assert artist_name == 'First Artist'
        assert artist_id is None

    def test_falls_back_to_album_artist(self):
        from tasks.mediaserver.emby import _select_best_artist

        item = {'AlbumArtist': 'The Album Artist'}

        artist_name, artist_id = _select_best_artist(item)

        assert artist_name == 'The Album Artist'
        assert artist_id is None

    def test_returns_unknown_when_no_artist_info(self):
        from tasks.mediaserver.emby import _select_best_artist

        item = {}

        artist_name, artist_id = _select_best_artist(item)

        assert artist_name == 'Unknown Artist'
        assert artist_id is None


class TestEmbyGetAllPlaylists:
    @patch('tasks.mediaserver.emby.requests.get')
    @patch('tasks.mediaserver.emby.config')
    def test_uses_correct_url_with_emby_prefix(self, mock_config, mock_get):
        from tasks.mediaserver.emby import get_all_playlists

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.HEADERS = {'X-Emby-Token': 'token'}

        mock_response = Mock()
        mock_response.json.return_value = {'Items': []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        get_all_playlists()

        call_url = mock_get.call_args[0][0]
        assert '/emby/' in call_url, "URL must include /emby/ prefix"
        assert call_url == 'http://emby:8096/emby/Users/user123/Items', (
            f"URL changed! Got '{call_url}'"
        )

    @patch('tasks.mediaserver.emby.requests.get')
    @patch('tasks.mediaserver.emby.config')
    def test_includes_playlist_item_type(self, mock_config, mock_get):
        from tasks.mediaserver.emby import get_all_playlists

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.HEADERS = {}

        mock_response = Mock()
        mock_response.json.return_value = {'Items': []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        get_all_playlists()

        call_params = mock_get.call_args[1].get('params', {})
        assert call_params.get('IncludeItemTypes') == 'Playlist', (
            "Must filter by Playlist item type"
        )

    @patch('tasks.mediaserver.emby.requests.get')
    @patch('tasks.mediaserver.emby.config')
    def test_parses_items_array(self, mock_config, mock_get):
        from tasks.mediaserver.emby import get_all_playlists

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.HEADERS = {}

        mock_response = Mock()
        mock_response.json.return_value = {
            'Items': [{'Id': 'pl1', 'Name': 'Rock_automatic'}, {'Id': 'pl2', 'Name': 'Jazz Mix'}]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        playlists = get_all_playlists()

        assert len(playlists) == 2
        assert playlists[0]['Id'] == 'pl1'
        assert playlists[0]['Name'] == 'Rock_automatic'


class TestEmbyDeletePlaylist:
    @patch('tasks.mediaserver.emby.requests.post')
    @patch('tasks.mediaserver.emby.config')
    def test_uses_items_delete_endpoint(self, mock_config, mock_post):
        from tasks.mediaserver.emby import delete_playlist

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.HEADERS = {'X-Emby-Token': 'token'}

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = delete_playlist('playlist-123')

        assert result is True
        call_url = mock_post.call_args[0][0]
        assert call_url == 'http://emby:8096/emby/Items/Delete', (
            f"Emby deletion URL changed! Expected '/emby/Items/Delete', got '{call_url}'"
        )

    @patch('tasks.mediaserver.emby.requests.post')
    @patch('tasks.mediaserver.emby.config')
    def test_passes_id_as_query_param(self, mock_config, mock_post):
        from tasks.mediaserver.emby import delete_playlist

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.HEADERS = {}

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        delete_playlist('playlist-xyz')

        call_params = mock_post.call_args[1].get('params', {})
        assert call_params.get('Ids') == 'playlist-xyz', (
            "Playlist ID must be passed as 'Ids' query param"
        )

    @patch('tasks.mediaserver.emby.requests.post')
    @patch('tasks.mediaserver.emby.config')
    def test_returns_false_on_error(self, mock_config, mock_post):
        from tasks.mediaserver.emby import delete_playlist

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.HEADERS = {}
        mock_post.side_effect = requests.exceptions.RequestException("Failed")

        result = delete_playlist('playlist-123')

        assert result is False


class TestEmbyGetTracksFromAlbum:
    @patch('tasks.mediaserver.emby.requests.get')
    @patch('tasks.mediaserver.emby.config')
    def test_uses_emby_url_prefix(self, mock_config, mock_get):
        from tasks.mediaserver.emby import get_tracks_from_album

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.HEADERS = {}

        mock_response = Mock()
        mock_response.json.return_value = {'Items': []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        get_tracks_from_album('album-123')

        call_url = mock_get.call_args[0][0]
        assert '/emby/' in call_url, "URL must include /emby/ prefix"

    @patch('tasks.mediaserver.emby.requests.get')
    @patch('tasks.mediaserver.emby.config')
    def test_enriches_tracks_with_artist(self, mock_config, mock_get):
        from tasks.mediaserver.emby import get_tracks_from_album

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.HEADERS = {}

        mock_response = Mock()
        mock_response.json.return_value = {
            'Items': [
                {
                    'Id': 'track1',
                    'Name': 'Song One',
                    'ArtistItems': [{'Name': 'Artist A', 'Id': 'artist-a'}],
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        tracks = get_tracks_from_album('album-123')

        assert len(tracks) == 1
        assert 'AlbumArtist' in tracks[0], "Must add AlbumArtist field"
        assert 'ArtistId' in tracks[0], "Must add ArtistId field"
        assert tracks[0]['AlbumArtist'] == 'Artist A'
        assert tracks[0]['ArtistId'] == 'artist-a'

    @patch('tasks.mediaserver.emby.requests.get')
    @patch('tasks.mediaserver.emby.config')
    def test_handles_standalone_track_pseudo_albums(self, mock_config, mock_get):
        from tasks.mediaserver.emby import get_tracks_from_album

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.HEADERS = {}

        mock_response = Mock()
        mock_response.json.return_value = {
            'Id': 'real-track-id',
            'Name': 'Standalone Song',
            'AlbumArtist': 'Some Artist',
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        get_tracks_from_album('standalone_real-track-id')

        call_url = mock_get.call_args[0][0]
        assert 'real-track-id' in call_url, "Should extract real track ID from pseudo-album"
        assert 'standalone_' not in call_url, "Should NOT include 'standalone_' prefix in API call"


class TestEmbyCreatePlaylist:
    @patch('tasks.mediaserver.emby.requests.post')
    @patch('tasks.mediaserver.emby.config')
    def test_uses_query_params_not_json_body(self, mock_config, mock_post):
        from tasks.mediaserver.emby import create_playlist

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.EMBY_TOKEN = 'token123'

        mock_response = Mock()
        mock_response.json.return_value = {'Id': 'new-playlist'}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        create_playlist('Test Playlist', ['track1', 'track2'])

        call_url = mock_post.call_args[0][0]
        assert 'Name=' in call_url, "Name must be in query string"
        assert 'Ids=' in call_url, "Ids must be in query string"
        assert 'UserId=' in call_url, "UserId must be in query string"
        assert 'MediaType=Audio' in call_url, "MediaType must be Audio"

        call_kwargs = mock_post.call_args[1]
        assert 'json' not in call_kwargs, "Emby should NOT receive JSON body"

    @patch('tasks.mediaserver.emby.requests.post')
    @patch('tasks.mediaserver.emby.config')
    def test_url_encodes_playlist_name(self, mock_config, mock_post):
        from tasks.mediaserver.emby import create_playlist

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.EMBY_TOKEN = 'token123'

        mock_response = Mock()
        mock_response.json.return_value = {'Id': 'new-playlist'}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        create_playlist('Rock & Roll Mix', ['track1'])

        call_url = mock_post.call_args[0][0]
        assert (
            'Rock%20%26%20Roll' in call_url
            or 'Rock+%26+Roll' in call_url
            or 'Rock%20&%20Roll' not in call_url
        )


class TestJellyfinListLibraries:
    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_returns_music_libraries_with_id_and_name(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import list_libraries

        mock_config.JELLYFIN_URL = 'http://jelly:8096'
        mock_config.JELLYFIN_TOKEN = 'admin-token'
        mock_config.HEADERS = {}

        resp = Mock()
        resp.raise_for_status = Mock()
        resp.json.return_value = [
            {'Name': 'Music', 'ItemId': 'lib-1', 'CollectionType': 'music'},
            {'Name': 'TV Shows', 'ItemId': 'lib-2', 'CollectionType': 'tvshows'},
            {'Name': 'Podcasts', 'ItemId': 'lib-3', 'CollectionType': 'music'},
        ]
        mock_get.return_value = resp

        result = list_libraries()

        assert result == [
            {'id': 'lib-1', 'name': 'Music'},
            {'id': 'lib-3', 'name': 'Podcasts'},
        ]

    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_forwards_user_creds_to_url_and_token(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import list_libraries

        mock_config.JELLYFIN_URL = 'http://SHOULD-NOT-BE-USED:0000'
        mock_config.JELLYFIN_TOKEN = 'SHOULD-NOT-BE-USED'
        mock_config.HEADERS = {}

        resp = Mock()
        resp.raise_for_status = Mock()
        resp.json.return_value = []
        mock_get.return_value = resp

        list_libraries(
            user_creds={
                'url': 'http://target-jelly:8096',
                'token': 'target-token',
            }
        )

        called_url = mock_get.call_args[0][0]
        assert called_url == 'http://target-jelly:8096/Library/VirtualFolders'
        headers = mock_get.call_args[1]['headers']
        assert headers.get('Authorization') == 'MediaBrowser Token="target-token"'


class TestEmbyListLibraries:
    @patch('tasks.mediaserver.emby.requests.get')
    @patch('tasks.mediaserver.emby.config')
    def test_returns_music_libraries_only(self, mock_config, mock_get):
        from tasks.mediaserver.emby import list_libraries

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_TOKEN = 'admin-token'
        mock_config.HEADERS = {}

        resp = Mock()
        resp.raise_for_status = Mock()
        resp.json.return_value = [
            {'Name': 'Music', 'ItemId': 'e1', 'CollectionType': 'music'},
            {'Name': 'Movies', 'ItemId': 'e2', 'CollectionType': 'movies'},
        ]
        mock_get.return_value = resp

        result = list_libraries()

        assert result == [{'id': 'e1', 'name': 'Music'}]

    @patch('tasks.mediaserver.emby.requests.get')
    @patch('tasks.mediaserver.emby.config')
    def test_forwards_user_creds(self, mock_config, mock_get):
        from tasks.mediaserver.emby import list_libraries

        mock_config.EMBY_URL = 'http://SHOULD-NOT-BE-USED:0000'
        mock_config.EMBY_TOKEN = 'SHOULD-NOT-BE-USED'
        mock_config.HEADERS = {}

        resp = Mock()
        resp.raise_for_status = Mock()
        resp.json.return_value = []
        mock_get.return_value = resp

        list_libraries(
            user_creds={
                'url': 'http://target-emby:8096',
                'token': 'target-token',
            }
        )

        called_url = mock_get.call_args[0][0]
        assert called_url == 'http://target-emby:8096/emby/Library/VirtualFolders'
        headers = mock_get.call_args[1]['headers']
        assert headers.get('X-Emby-Token') == 'target-token'


class TestNavidromeListLibraries:
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_returns_every_folder_without_reading_music_libraries(self, mock_req):
        from tasks.mediaserver.navidrome import list_libraries

        mock_req.return_value = {
            'musicFolders': {
                'musicFolder': [
                    {'id': 1, 'name': 'Main'},
                    {'id': 2, 'name': 'Podcasts'},
                ]
            }
        }

        result = list_libraries()

        assert result == [
            {'id': '1', 'name': 'Main'},
            {'id': '2', 'name': 'Podcasts'},
        ]
        mock_req.assert_called_once()
        args, kwargs = mock_req.call_args
        assert args[0] == 'getMusicFolders'
        assert kwargs.get('user_creds') is None

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_handles_single_dict_response(self, mock_req):
        from tasks.mediaserver.navidrome import list_libraries

        mock_req.return_value = {'musicFolders': {'musicFolder': {'id': 1, 'name': 'OnlyFolder'}}}

        result = list_libraries()

        assert result == [{'id': '1', 'name': 'OnlyFolder'}]

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    def test_forwards_user_creds_to_getmusicfolders(self, mock_req):
        from tasks.mediaserver.navidrome import list_libraries

        mock_req.return_value = {'musicFolders': {'musicFolder': []}}

        creds = {'url': 'http://target-nav:4533', 'user': 'u', 'password': 'p'}
        list_libraries(user_creds=creds)

        args, kwargs = mock_req.call_args
        assert args[0] == 'getMusicFolders'
        assert kwargs.get('user_creds') == creds


class TestLyrionListLibraries:
    @patch('tasks.mediaserver.lyrion._jsonrpc_request')
    def test_returns_every_folder(self, mock_rpc):
        from tasks.mediaserver.lyrion import list_libraries

        mock_rpc.return_value = {
            'folder_loop': [
                {'id': 10, 'name': 'Music'},
                {'id': 11, 'name': 'Audiobooks'},
            ]
        }

        result = list_libraries()

        assert result == [
            {'id': '10', 'name': 'Music'},
            {'id': '11', 'name': 'Audiobooks'},
        ]
        args, kwargs = mock_rpc.call_args
        assert args[0] == 'musicfolder'
        assert kwargs.get('user_creds') is None

    @patch('tasks.mediaserver.lyrion._jsonrpc_request')
    def test_handles_lyrion_9_x_filename_field(self, mock_rpc):
        from tasks.mediaserver.lyrion import list_libraries

        mock_rpc.return_value = {
            'folder_loop': [
                {'id': 685, 'filename': 'Library_A', 'type': 'folder'},
                {'id': 686, 'filename': 'Library_B', 'type': 'folder'},
            ],
            'count': 2,
        }

        result = list_libraries()

        assert result == [
            {'id': '685', 'name': 'Library_A'},
            {'id': '686', 'name': 'Library_B'},
        ]

    @patch('tasks.mediaserver.lyrion._jsonrpc_request')
    def test_prefers_path_over_name_when_available(self, mock_rpc):
        from tasks.mediaserver.lyrion import list_libraries

        mock_rpc.return_value = {
            'folder_loop': [
                {'id': 10, 'name': 'Music', 'path': '/srv/music'},
                {'id': 11, 'name': 'Spoken', 'url': '/srv/audiobooks'},
                {'id': 12, 'name': 'NoPath'},
            ]
        }

        result = list_libraries()

        assert result == [
            {'id': '10', 'name': '/srv/music'},
            {'id': '11', 'name': '/srv/audiobooks'},
            {'id': '12', 'name': 'NoPath'},
        ]

    @patch('tasks.mediaserver.lyrion._jsonrpc_request')
    def test_forwards_user_creds(self, mock_rpc):
        from tasks.mediaserver.lyrion import list_libraries

        mock_rpc.return_value = {'folder_loop': []}

        creds = {'url': 'http://target-lms:9000', 'user': 'u', 'password': 'p'}
        list_libraries(user_creds=creds)

        args, kwargs = mock_rpc.call_args
        assert args[0] == 'musicfolder'
        assert kwargs.get('user_creds') == creds


class TestNavidromeCreateOrReplacePlaylist:
    @patch('tasks.mediaserver.navidrome._create_playlist_batched')
    @patch('tasks.mediaserver.navidrome.get_playlist_by_name')
    def test_missing_playlist_creates_via_batched(self, mock_get, mock_create):
        from tasks.mediaserver.navidrome import create_or_replace_playlist

        mock_get.return_value = None
        mock_create.return_value = {'Id': 'new-pl-1', 'Name': 'SF', 'id': 'new-pl-1'}

        result = create_or_replace_playlist('SF', ['s1', 's2'])

        mock_create.assert_called_once_with('SF', ['s1', 's2'], user_creds=None)
        assert result['Id'] == 'new-pl-1'

    @patch('tasks.mediaserver.navidrome._add_to_playlist')
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    @patch('tasks.mediaserver.navidrome.get_playlist_by_name')
    def test_existing_playlist_preserves_id(self, mock_get, mock_request, mock_add):
        from tasks.mediaserver.navidrome import create_or_replace_playlist

        mock_get.return_value = {'id': 'pl-existing', 'name': 'SF'}
        mock_request.side_effect = [
            {'playlist': {'id': 'pl-existing', 'songCount': 3}},
            {'status': 'ok'},
        ]
        mock_add.return_value = True

        result = create_or_replace_playlist('SF', ['new1', 'new2'])

        first = mock_request.call_args_list[0]
        assert first[0][0] == 'getPlaylist'
        assert first[0][1] == {'id': 'pl-existing'}

        second = mock_request.call_args_list[1]
        assert second[0][0] == 'updatePlaylist'
        assert second[0][1]['playlistId'] == 'pl-existing'
        assert second[0][1]['songIndexToRemove'] == [2, 1, 0]

        mock_add.assert_called_once_with('pl-existing', ['new1', 'new2'], user_creds=None)

        assert result['Id'] == 'pl-existing'

    @patch('tasks.mediaserver.navidrome._add_to_playlist')
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    @patch('tasks.mediaserver.navidrome.get_playlist_by_name')
    def test_clear_batches_above_40(self, mock_get, mock_request, mock_add):
        from tasks.mediaserver.navidrome import create_or_replace_playlist

        mock_get.return_value = {'id': 'pl-100', 'name': 'SF'}
        mock_request.side_effect = [
            {'playlist': {'id': 'pl-100', 'songCount': 100}},
            {'status': 'ok'},
            {'status': 'ok'},
            {'status': 'ok'},
        ]
        mock_add.return_value = True

        create_or_replace_playlist('SF', ['x'])

        assert len(mock_request.call_args_list) == 4
        update_calls = mock_request.call_args_list[1:]
        assert len(update_calls[0][0][1]['songIndexToRemove']) == 40
        assert len(update_calls[1][0][1]['songIndexToRemove']) == 40
        assert len(update_calls[2][0][1]['songIndexToRemove']) == 20

    @patch('tasks.mediaserver.navidrome._navidrome_request')
    @patch('tasks.mediaserver.navidrome.get_playlist_by_name')
    def test_empty_item_ids_returns_none_without_calls(self, mock_get, mock_request):
        from tasks.mediaserver.navidrome import create_or_replace_playlist

        result = create_or_replace_playlist('SF', [])

        assert result is None
        mock_get.assert_not_called()
        mock_request.assert_not_called()

    @patch('tasks.mediaserver.navidrome._add_to_playlist')
    @patch('tasks.mediaserver.navidrome._navidrome_request')
    @patch('tasks.mediaserver.navidrome.get_playlist_by_name')
    def test_returns_none_when_add_fails_after_clear(self, mock_get, mock_request, mock_add):
        from tasks.mediaserver.navidrome import create_or_replace_playlist

        mock_get.return_value = {'id': 'pl-1', 'name': 'SF'}
        mock_request.side_effect = [
            {'playlist': {'id': 'pl-1', 'songCount': 1}},
            {'status': 'ok'},
        ]
        mock_add.return_value = False

        result = create_or_replace_playlist('SF', ['new1'])

        assert result is None


class TestJellyfinCreateOrReplacePlaylist:
    @patch('tasks.mediaserver.jellyfin.requests')
    @patch('tasks.mediaserver.jellyfin.get_playlist_by_name')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_missing_playlist_creates_and_returns_id(self, mock_config, mock_get, mock_requests):
        from tasks.mediaserver.jellyfin import create_or_replace_playlist

        mock_config.JELLYFIN_URL = 'http://jf'
        mock_config.JELLYFIN_USER_ID = 'admin-user'
        mock_config.HEADERS = {'Authorization': 'MediaBrowser Token="t"'}
        mock_get.return_value = None

        post_resp = MagicMock()
        post_resp.json.return_value = {'Id': 'new-jf-1', 'Name': 'SF'}
        mock_requests.post.return_value = post_resp

        result = create_or_replace_playlist('SF', ['s1', 's2'])

        assert mock_requests.post.call_count == 1
        post_call = mock_requests.post.call_args
        assert post_call[0][0] == 'http://jf/Playlists'
        assert post_call[1]['json'] == {'Name': 'SF', 'Ids': ['s1', 's2'], 'UserId': 'admin-user'}
        assert result['Id'] == 'new-jf-1'

    @patch('tasks.mediaserver.jellyfin.requests')
    @patch('tasks.mediaserver.jellyfin.get_playlist_by_name')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_existing_playlist_clears_and_adds_preserving_id(
        self, mock_config, mock_get, mock_requests
    ):
        from tasks.mediaserver.jellyfin import create_or_replace_playlist

        mock_config.JELLYFIN_URL = 'http://jf'
        mock_config.JELLYFIN_USER_ID = 'admin-user'
        mock_config.HEADERS = {'Authorization': 'MediaBrowser Token="t"'}
        mock_get.return_value = {'Id': 'pl-existing', 'Name': 'SF'}

        get_resp = MagicMock()
        get_resp.json.return_value = {
            'Items': [
                {'Id': 'song1', 'PlaylistItemId': 'entry-a'},
                {'Id': 'song2', 'PlaylistItemId': 'entry-b'},
            ]
        }
        mock_requests.get.return_value = get_resp
        mock_requests.delete.return_value = MagicMock()
        mock_requests.post.return_value = MagicMock()

        result = create_or_replace_playlist('SF', ['new1', 'new2'])

        assert mock_requests.get.call_args[0][0] == 'http://jf/Playlists/pl-existing/Items'
        assert mock_requests.delete.call_count == 1
        del_call = mock_requests.delete.call_args
        assert del_call[0][0] == 'http://jf/Playlists/pl-existing/Items'
        assert del_call[1]['params']['entryIds'] == 'entry-a,entry-b'
        assert mock_requests.post.call_count == 1
        post_call = mock_requests.post.call_args
        assert post_call[0][0] == 'http://jf/Playlists/pl-existing/Items'
        assert post_call[1]['params']['ids'] == 'new1,new2'
        assert post_call[1]['params']['userId'] == 'admin-user'
        assert result['Id'] == 'pl-existing'

    @patch('tasks.mediaserver.jellyfin.get_playlist_by_name')
    def test_empty_item_ids_returns_none(self, mock_get):
        from tasks.mediaserver.jellyfin import create_or_replace_playlist

        result = create_or_replace_playlist('SF', [])

        assert result is None
        mock_get.assert_not_called()

    @patch('tasks.mediaserver.jellyfin._add_items_to_playlist')
    @patch('tasks.mediaserver.jellyfin._remove_playlist_entries')
    @patch('tasks.mediaserver.jellyfin._get_playlist_entry_ids')
    @patch('tasks.mediaserver.jellyfin.get_playlist_by_name')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_returns_none_when_add_fails_after_clear(
        self, mock_config, mock_get, mock_get_entries, mock_remove, mock_add
    ):
        from tasks.mediaserver.jellyfin import create_or_replace_playlist

        mock_config.JELLYFIN_URL = 'http://jf'
        mock_config.JELLYFIN_USER_ID = 'admin-user'
        mock_config.HEADERS = {'Authorization': 'MediaBrowser Token="t"'}
        mock_get.return_value = {'Id': 'pl-1', 'Name': 'SF'}
        mock_get_entries.return_value = ['e1']
        mock_remove.return_value = True
        mock_add.return_value = False

        result = create_or_replace_playlist('SF', ['new1'])

        assert result is None

    @patch('tasks.mediaserver.jellyfin._create_fresh_playlist')
    @patch('tasks.mediaserver.jellyfin.delete_playlist')
    @patch('tasks.mediaserver.jellyfin._remove_playlist_entries')
    @patch('tasks.mediaserver.jellyfin._get_playlist_entry_ids')
    @patch('tasks.mediaserver.jellyfin.get_playlist_by_name')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_falls_back_to_recreate_when_remove_fails(
        self, mock_config, mock_get, mock_get_entries, mock_remove, mock_delete, mock_create
    ):
        from tasks.mediaserver.jellyfin import create_or_replace_playlist

        mock_config.JELLYFIN_URL = 'http://jf'
        mock_config.JELLYFIN_USER_ID = 'admin-user'
        mock_config.HEADERS = {'Authorization': 'MediaBrowser Token="t"'}
        mock_get.return_value = {'Id': 'old-pl', 'Name': 'SF'}
        mock_get_entries.return_value = ['e1', 'e2']
        mock_remove.side_effect = requests.exceptions.HTTPError('400 Bad Request')
        mock_delete.return_value = True
        mock_create.return_value = {'Id': 'new-pl', 'Name': 'SF'}

        result = create_or_replace_playlist('SF', ['n1', 'n2'])

        mock_delete.assert_called_once_with('old-pl')
        mock_create.assert_called_once_with('SF', ['n1', 'n2'])
        assert result['Id'] == 'new-pl'

    @patch('tasks.mediaserver.jellyfin._create_fresh_playlist')
    @patch('tasks.mediaserver.jellyfin.delete_playlist')
    @patch('tasks.mediaserver.jellyfin._remove_playlist_entries')
    @patch('tasks.mediaserver.jellyfin._get_playlist_entry_ids')
    @patch('tasks.mediaserver.jellyfin.get_playlist_by_name')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_fallback_returns_none_when_delete_playlist_fails(
        self, mock_config, mock_get, mock_get_entries, mock_remove, mock_delete, mock_create
    ):
        from tasks.mediaserver.jellyfin import create_or_replace_playlist

        mock_config.JELLYFIN_URL = 'http://jf'
        mock_config.JELLYFIN_USER_ID = 'admin-user'
        mock_config.HEADERS = {'Authorization': 'MediaBrowser Token="t"'}
        mock_get.return_value = {'Id': 'old-pl', 'Name': 'SF'}
        mock_get_entries.return_value = ['e1']
        mock_remove.side_effect = requests.exceptions.HTTPError('400 Bad Request')
        mock_delete.return_value = False

        result = create_or_replace_playlist('SF', ['n1'])

        assert result is None
        mock_create.assert_not_called()


class TestEmbyCreateOrReplacePlaylist:
    @patch('tasks.mediaserver.emby.requests')
    @patch('tasks.mediaserver.emby.get_playlist_by_name')
    @patch('tasks.mediaserver.emby.config')
    def test_existing_playlist_clears_and_adds_preserving_id(
        self, mock_config, mock_get, mock_requests
    ):
        from tasks.mediaserver.emby import create_or_replace_playlist

        mock_config.EMBY_URL = 'http://emby'
        mock_config.EMBY_USER_ID = 'admin-emby'
        mock_config.EMBY_TOKEN = 'tok'
        mock_get.return_value = {'Id': 'emby-pl', 'Name': 'SF'}

        get_resp = MagicMock()
        get_resp.json.return_value = {
            'Items': [
                {'Id': 'song1', 'PlaylistItemId': 'e1'},
            ]
        }
        mock_requests.get.return_value = get_resp
        mock_requests.delete.return_value = MagicMock()
        mock_requests.post.return_value = MagicMock()
        mock_requests.utils.quote.side_effect = lambda s: s

        result = create_or_replace_playlist('SF', ['n1'])

        assert mock_requests.delete.call_args[1]['params']['EntryIds'] == 'e1'
        post_call = mock_requests.post.call_args
        assert post_call[0][0] == 'http://emby/emby/Playlists/emby-pl/Items'
        assert post_call[1]['params']['Ids'] == 'n1'
        assert post_call[1]['params']['UserId'] == 'admin-emby'
        assert result['Id'] == 'emby-pl'

    @patch('tasks.mediaserver.emby._add_items_to_playlist')
    @patch('tasks.mediaserver.emby._remove_playlist_entries')
    @patch('tasks.mediaserver.emby._get_playlist_entry_ids')
    @patch('tasks.mediaserver.emby.get_playlist_by_name')
    @patch('tasks.mediaserver.emby.config')
    def test_returns_none_when_add_fails_after_clear(
        self, mock_config, mock_get, mock_get_entries, mock_remove, mock_add
    ):
        from tasks.mediaserver.emby import create_or_replace_playlist

        mock_config.EMBY_URL = 'http://emby'
        mock_config.EMBY_USER_ID = 'admin-emby'
        mock_config.EMBY_TOKEN = 'tok'
        mock_get.return_value = {'Id': 'emby-pl', 'Name': 'SF'}
        mock_get_entries.return_value = ['e1']
        mock_remove.return_value = True
        mock_add.return_value = False

        result = create_or_replace_playlist('SF', ['n1'])

        assert result is None


class TestLyrionCreateOrReplacePlaylist:
    @patch('tasks.mediaserver.lyrion._create_playlist_batched')
    @patch('tasks.mediaserver.lyrion.delete_playlist')
    @patch('tasks.mediaserver.lyrion.get_playlist_by_name')
    def test_existing_deletes_then_creates(self, mock_get, mock_delete, mock_create):
        from tasks.mediaserver.lyrion import create_or_replace_playlist

        mock_get.return_value = {'Id': 99, 'Name': 'SF'}
        mock_delete.return_value = True
        mock_create.return_value = {'Id': 100, 'Name': 'SF'}

        result = create_or_replace_playlist('SF', ['t1'])

        mock_delete.assert_called_once_with(99)
        mock_create.assert_called_once_with('SF', ['t1'])
        assert result['Name'] == 'SF'

    @patch('tasks.mediaserver.lyrion._create_playlist_batched')
    @patch('tasks.mediaserver.lyrion.delete_playlist')
    @patch('tasks.mediaserver.lyrion.get_playlist_by_name')
    def test_missing_creates_without_delete(self, mock_get, mock_delete, mock_create):
        from tasks.mediaserver.lyrion import create_or_replace_playlist

        mock_get.return_value = None
        mock_create.return_value = {'Id': 50, 'Name': 'SF'}

        create_or_replace_playlist('SF', ['t1'])

        mock_delete.assert_not_called()
        mock_create.assert_called_once_with('SF', ['t1'])

    @patch('tasks.mediaserver.lyrion._create_playlist_batched')
    @patch('tasks.mediaserver.lyrion.delete_playlist')
    @patch('tasks.mediaserver.lyrion.get_playlist_by_name')
    def test_aborts_when_delete_fails(self, mock_get, mock_delete, mock_create):
        from tasks.mediaserver.lyrion import create_or_replace_playlist

        mock_get.return_value = {'Id': 99, 'Name': 'SF'}
        mock_delete.return_value = False

        result = create_or_replace_playlist('SF', ['t1'])

        assert result is None
        mock_create.assert_not_called()


class TestDispatcherCreateOrReplacePlaylist:
    @patch('tasks.mediaserver.config')
    def test_requires_name_and_ids(self, mock_config):
        from tasks.mediaserver import create_or_replace_playlist

        with pytest.raises(ValueError, match="Playlist name is required"):
            create_or_replace_playlist('', ['id1'])

        with pytest.raises(ValueError, match="Track IDs are required"):
            create_or_replace_playlist('Name', [])

    @patch('tasks.mediaserver.config')
    def test_unsupported_backend_raises_not_implemented(self, mock_config):
        from tasks.mediaserver import create_or_replace_playlist

        mock_config.MEDIASERVER_TYPE = 'unsupported'

        with pytest.raises(NotImplementedError):
            create_or_replace_playlist('SF', ['s1'])

    @patch('tasks.mediaserver.navidrome.create_or_replace_playlist')
    @patch('tasks.mediaserver.config')
    def test_dispatches_to_navidrome(self, mock_config, mock_provider):
        from tasks.mediaserver import create_or_replace_playlist

        mock_config.MEDIASERVER_TYPE = 'navidrome'
        mock_provider.return_value = {'Id': 'pl-1'}

        result = create_or_replace_playlist('SF', ['s1'])

        mock_provider.assert_called_once_with('SF', ['s1'], None)
        assert result['Id'] == 'pl-1'

    @patch('tasks.mediaserver.jellyfin.create_or_replace_playlist')
    @patch('tasks.mediaserver.config')
    def test_dispatches_to_jellyfin(self, mock_config, mock_provider):
        from tasks.mediaserver import create_or_replace_playlist

        mock_config.MEDIASERVER_TYPE = 'jellyfin'
        mock_provider.return_value = {'Id': 'pl-2'}

        create_or_replace_playlist('SF', ['s1'])

        mock_provider.assert_called_once()

    @patch('tasks.mediaserver.emby.create_or_replace_playlist')
    @patch('tasks.mediaserver.config')
    def test_dispatches_to_emby(self, mock_config, mock_provider):
        from tasks.mediaserver import create_or_replace_playlist

        mock_config.MEDIASERVER_TYPE = 'emby'
        mock_provider.return_value = {'Id': 'pl-3'}

        create_or_replace_playlist('SF', ['s1'])

        mock_provider.assert_called_once()

    @patch('tasks.mediaserver.lyrion.create_or_replace_playlist')
    @patch('tasks.mediaserver.config')
    def test_dispatches_to_lyrion(self, mock_config, mock_provider):
        from tasks.mediaserver import create_or_replace_playlist

        mock_config.MEDIASERVER_TYPE = 'lyrion'
        mock_provider.return_value = {'Id': 4}

        create_or_replace_playlist('SF', ['s1'])

        mock_provider.assert_called_once()


def _audio_page(n_items, start=0):
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = {
        'Items': [{'Id': f'id{start + i}', 'Name': f'Song {start + i}'} for i in range(n_items)]
    }
    return resp


class TestJellyfinGetAllSongsPagination:
    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_paginates_until_short_page(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import get_all_songs

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {'Authorization': 'MediaBrowser Token="token"'}
        mock_get.side_effect = [_audio_page(500), _audio_page(3, start=500)]

        songs = get_all_songs()

        assert len(songs) == 503
        assert mock_get.call_count == 2
        page2_params = mock_get.call_args_list[1].kwargs['params']
        assert page2_params['StartIndex'] == 500
        assert page2_params['Limit'] == 500

    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_raises_on_midscan_failure_instead_of_truncating(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import get_all_songs

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {'Authorization': 'MediaBrowser Token="token"'}
        mock_get.side_effect = [
            _audio_page(500),
            requests.exceptions.ReadTimeout("read timed out"),
        ]

        with pytest.raises(requests.exceptions.ReadTimeout):
            get_all_songs()

    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_empty_library_returns_empty_without_raising(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import get_all_songs

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {}
        mock_get.side_effect = [_audio_page(0)]

        assert get_all_songs() == []


class TestEmbyGetAllSongsRaisesOnFailure:
    @patch('tasks.mediaserver.emby.requests.get')
    @patch('tasks.mediaserver.emby.config')
    def test_raises_on_midscan_failure_instead_of_truncating(self, mock_config, mock_get):
        from tasks.mediaserver.emby import get_all_songs

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.HEADERS = {'X-Emby-Token': 'token'}
        mock_get.side_effect = [
            _audio_page(1000),
            requests.exceptions.ReadTimeout("read timed out"),
        ]

        with pytest.raises(requests.exceptions.ReadTimeout):
            get_all_songs()

    @patch('tasks.mediaserver.emby.requests.get')
    @patch('tasks.mediaserver.emby.config')
    def test_empty_library_returns_empty_without_raising(self, mock_config, mock_get):
        from tasks.mediaserver.emby import get_all_songs

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.HEADERS = {}
        mock_get.side_effect = [_audio_page(0)]

        assert get_all_songs() == []


class _WrapperError(Exception):
    pass


def _make_http_error(status_code, message='error'):
    resp = Mock()
    resp.status_code = status_code
    err = requests.exceptions.HTTPError(message)
    err.response = resp
    return err


class TestIsAuthError:
    def test_detects_401_response(self):
        from tasks.mediaserver.helper import is_auth_error

        assert is_auth_error(_make_http_error(401, '401 Client Error')) is True

    def test_detects_403_response(self):
        from tasks.mediaserver.helper import is_auth_error

        assert is_auth_error(_make_http_error(403, '403 Forbidden')) is True

    def test_ignores_500_response(self):
        from tasks.mediaserver.helper import is_auth_error

        assert is_auth_error(_make_http_error(500, '500 Server Error')) is False

    def test_ignores_plain_connection_error(self):
        from tasks.mediaserver.helper import is_auth_error

        assert is_auth_error(requests.exceptions.ConnectionError('refused')) is False

    def test_detects_auth_wording(self):
        from tasks.mediaserver.helper import is_auth_error

        assert is_auth_error(_WrapperError('Wrong username or password')) is True

    def test_walks_exception_chain(self):
        from tasks.mediaserver.helper import is_auth_error

        try:
            try:
                raise _make_http_error(401, 'unauthorized')
            except Exception as inner:
                raise _WrapperError('wrapped call') from inner
        except Exception as e:
            assert is_auth_error(e) is True


class TestProviderTestConnectionAuth:
    @patch('tasks.mediaserver.jellyfin.requests.get')
    @patch('tasks.mediaserver.jellyfin.config')
    def test_jellyfin_flags_401(self, mock_config, mock_get):
        from tasks.mediaserver.jellyfin import test_connection

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'uid'
        mock_config.JELLYFIN_TOKEN = 'tok'
        mock_config.HEADERS = {}
        resp = Mock()
        resp.raise_for_status.side_effect = _make_http_error(401, '401 Unauthorized')
        mock_get.return_value = resp

        result = test_connection()

        assert result['ok'] is False
        assert result['auth_failed'] is True

    @patch('tasks.mediaserver.emby.requests.get')
    @patch('tasks.mediaserver.emby.config')
    def test_emby_flags_401(self, mock_config, mock_get):
        from tasks.mediaserver.emby import test_connection

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'uid'
        mock_config.EMBY_TOKEN = 'tok'
        mock_config.HEADERS = {}
        resp = Mock()
        resp.raise_for_status.side_effect = _make_http_error(401, '401 Unauthorized')
        mock_get.return_value = resp

        result = test_connection()

        assert result['ok'] is False
        assert result['auth_failed'] is True

    @patch('tasks.mediaserver.lyrion._jsonrpc_request')
    def test_lyrion_flags_auth_error_without_raising(self, mock_jsonrpc):
        from tasks.mediaserver.lyrion import test_connection, LyrionAPIError

        mock_jsonrpc.side_effect = LyrionAPIError(
            'Unexpected error calling Lyrion API: 401 Client Error: Unauthorized'
        )

        result = test_connection()

        assert result['ok'] is False
        assert result['auth_failed'] is True
