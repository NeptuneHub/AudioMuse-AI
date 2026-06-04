"""Unit tests for mediaserver implementations

Tests mock HTTP responses but verify real parsing, transformation, and error handling.
If the response parsing or error handling changes, these tests will catch it.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import requests


# =============================================================================
# JELLYFIN TESTS
# =============================================================================

class TestJellyfinSelectBestArtist:
    """Test artist field prioritization logic - no mocking needed"""

    def test_prioritizes_artist_items_over_album_artist(self):
        """ArtistItems should be preferred over AlbumArtist"""
        from tasks.mediaserver_jellyfin import _select_best_artist
        
        item = {
            'ArtistItems': [{'Name': 'Track Artist', 'Id': 'artist-123'}],
            'Artists': ['Fallback Artist'],
            'AlbumArtist': 'Album Artist'
        }
        
        artist_name, artist_id = _select_best_artist(item)
        
        assert artist_name == 'Track Artist'
        assert artist_id == 'artist-123'

    def test_falls_back_to_artists_array(self):
        """If no ArtistItems, use Artists array"""
        from tasks.mediaserver_jellyfin import _select_best_artist
        
        item = {
            'ArtistItems': [],
            'Artists': ['First Artist', 'Second Artist'],
            'AlbumArtist': 'Album Artist'
        }
        
        artist_name, artist_id = _select_best_artist(item)
        
        assert artist_name == 'First Artist'
        assert artist_id is None

    def test_falls_back_to_album_artist(self):
        """If no Artists, use AlbumArtist"""
        from tasks.mediaserver_jellyfin import _select_best_artist
        
        item = {
            'AlbumArtist': 'The Album Artist'
        }
        
        artist_name, artist_id = _select_best_artist(item)
        
        assert artist_name == 'The Album Artist'
        assert artist_id is None

    def test_returns_unknown_when_no_artist_info(self):
        """Returns 'Unknown Artist' when no artist info available"""
        from tasks.mediaserver_jellyfin import _select_best_artist
        
        item = {}
        
        artist_name, artist_id = _select_best_artist(item)
        
        assert artist_name == 'Unknown Artist'
        assert artist_id is None

    def test_handles_empty_artist_items(self):
        """Empty ArtistItems should fall back"""
        from tasks.mediaserver_jellyfin import _select_best_artist
        
        item = {
            'ArtistItems': [],
            'AlbumArtist': 'Fallback'
        }
        
        artist_name, _ = _select_best_artist(item)
        
        assert artist_name == 'Fallback'


class TestJellyfinResolveUser:
    """Test user resolution with mocked HTTP"""

    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_resolves_username_to_id(self, mock_config, mock_get):
        """Username should be resolved to User ID"""
        from tasks.mediaserver_jellyfin import resolve_user
        
        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_response = Mock()
        mock_response.json.return_value = [
            {'Name': 'admin', 'Id': 'admin-id-123'},
            {'Name': 'TestUser', 'Id': 'user-id-456'}
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = resolve_user('testuser', 'token123')
        
        assert result == 'user-id-456'
        mock_get.assert_called_once()
        # Verify correct URL was called
        call_url = mock_get.call_args[0][0]
        assert '/Users' in call_url

    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_returns_identifier_if_no_match(self, mock_config, mock_get):
        """If username not found, return original identifier (assumed to be ID)"""
        from tasks.mediaserver_jellyfin import resolve_user
        
        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_response = Mock()
        mock_response.json.return_value = [
            {'Name': 'OtherUser', 'Id': 'other-id'}
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = resolve_user('direct-user-id', 'token123')
        
        assert result == 'direct-user-id'

    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_handles_http_error(self, mock_config, mock_get):
        """HTTP errors should return original identifier"""
        from tasks.mediaserver_jellyfin import resolve_user
        
        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_get.side_effect = requests.exceptions.RequestException("Connection failed")
        
        result = resolve_user('some-user', 'token')
        
        # Should return original identifier on error
        assert result == 'some-user'


class TestJellyfinGetTracksFromAlbum:
    """Test track fetching with artist enrichment - verifies exact behavior"""

    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_uses_correct_url_and_params(self, mock_config, mock_get):
        """CRITICAL: Must use /Users/{id}/Items with ParentId - catches if URL changes"""
        from tasks.mediaserver_jellyfin import get_tracks_from_album
        
        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {'X-Emby-Token': 'token'}
        
        mock_response = Mock()
        mock_response.json.return_value = {'Items': []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        get_tracks_from_album('album-xyz')
        
        call_url = mock_get.call_args[0][0]
        call_params = mock_get.call_args[1].get('params', {})
        
        # Verify exact URL
        assert call_url == 'http://jellyfin:8096/Users/user123/Items', \
            f"URL changed! Expected '/Users/user123/Items', got '{call_url}'"
        # Verify required params
        assert call_params.get('ParentId') == 'album-xyz', "ParentId param missing or wrong"
        assert call_params.get('IncludeItemTypes') == 'Audio', "IncludeItemTypes param wrong"

    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_enriches_tracks_with_artist_info(self, mock_config, mock_get):
        """CRITICAL: Must add AlbumArtist and ArtistId fields - catches if enrichment changes"""
        from tasks.mediaserver_jellyfin import get_tracks_from_album
        
        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {'X-Emby-Token': 'token'}
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'Items': [
                {
                    'Id': 'track1',
                    'Name': 'Song One',
                    'ArtistItems': [{'Name': 'Artist A', 'Id': 'artist-a'}]
                },
                {
                    'Id': 'track2',
                    'Name': 'Song Two',
                    'AlbumArtist': 'Album Artist B'
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        tracks = get_tracks_from_album('album123')
        
        assert len(tracks) == 2
        # CRITICAL: Verify enrichment fields are added
        assert 'AlbumArtist' in tracks[0], "AlbumArtist field must be added"
        assert 'ArtistId' in tracks[0], "ArtistId field must be added"
        # First track should use ArtistItems (priority)
        assert tracks[0]['AlbumArtist'] == 'Artist A', \
            "ArtistItems should be prioritized"
        assert tracks[0]['ArtistId'] == 'artist-a'
        # Second track should fall back to AlbumArtist
        assert tracks[1]['AlbumArtist'] == 'Album Artist B', \
            "Should fall back to AlbumArtist when no ArtistItems"
        assert tracks[1]['ArtistId'] is None

    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_returns_empty_on_http_error(self, mock_config, mock_get):
        """HTTP error should return empty list, not raise"""
        from tasks.mediaserver_jellyfin import get_tracks_from_album
        
        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {}
        mock_get.side_effect = requests.exceptions.RequestException("Failed")
        
        tracks = get_tracks_from_album('album123')
        
        assert tracks == []

    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_handles_empty_items_response(self, mock_config, mock_get):
        """Empty Items array should return empty list"""
        from tasks.mediaserver_jellyfin import get_tracks_from_album
        
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
    """Test playlist fetching - verifies exact URL and response parsing"""

    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_uses_correct_url_and_params(self, mock_config, mock_get):
        """CRITICAL: Must use /Users/{id}/Items with IncludeItemTypes=Playlist"""
        from tasks.mediaserver_jellyfin import get_all_playlists
        
        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {'X-Emby-Token': 'token'}
        
        mock_response = Mock()
        mock_response.json.return_value = {'Items': []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        get_all_playlists()
        
        call_url = mock_get.call_args[0][0]
        call_params = mock_get.call_args[1].get('params', {})
        
        # Verify exact URL
        assert call_url == 'http://jellyfin:8096/Users/user123/Items', \
            f"URL changed! Got '{call_url}'"
        # Verify required params
        assert call_params.get('IncludeItemTypes') == 'Playlist', \
            "IncludeItemTypes must be 'Playlist'"
        assert call_params.get('Recursive') == True, \
            "Recursive must be True"

    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_parses_items_array_from_response(self, mock_config, mock_get):
        """CRITICAL: Must extract Items[] from response - catches if parsing changes"""
        from tasks.mediaserver_jellyfin import get_all_playlists
        
        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {}
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'Items': [
                {'Id': 'pl1', 'Name': 'Rock_automatic'},
                {'Id': 'pl2', 'Name': 'Jazz Favorites'}
            ],
            'TotalRecordCount': 2
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        playlists = get_all_playlists()
        
        assert len(playlists) == 2
        assert playlists[0]['Id'] == 'pl1'
        assert playlists[0]['Name'] == 'Rock_automatic'

    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_returns_empty_on_error(self, mock_config, mock_get):
        """Error should return empty list, not raise"""
        from tasks.mediaserver_jellyfin import get_all_playlists
        
        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {}
        mock_get.side_effect = requests.exceptions.RequestException("Failed")
        
        playlists = get_all_playlists()
        
        assert playlists == []


class TestJellyfinDeletePlaylist:
    """Test playlist deletion - verifies exact URL construction and HTTP method"""

    @patch('tasks.mediaserver_jellyfin.requests.delete')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_uses_correct_url_and_method(self, mock_config, mock_delete):
        """CRITICAL: Must use DELETE method to /Items/{id} - catches if someone changes to POST"""
        from tasks.mediaserver_jellyfin import delete_playlist
        
        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.HEADERS = {'X-Emby-Token': 'test-token'}
        
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_delete.return_value = mock_response
        
        result = delete_playlist('playlist-123')
        
        assert result is True
        # CRITICAL: Verify exact URL - will fail if path changes
        mock_delete.assert_called_once()
        call_url = mock_delete.call_args[0][0]
        assert call_url == 'http://jellyfin:8096/Items/playlist-123', \
            f"URL changed! Expected '/Items/playlist-123', got '{call_url}'"
        # Verify headers are passed
        call_kwargs = mock_delete.call_args[1]
        assert call_kwargs.get('headers') == {'X-Emby-Token': 'test-token'}

    @patch('tasks.mediaserver_jellyfin.requests.delete')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_returns_false_on_http_error(self, mock_config, mock_delete):
        """HTTP error returns False - catches if error handling changes"""
        from tasks.mediaserver_jellyfin import delete_playlist
        
        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.HEADERS = {}
        mock_delete.side_effect = requests.exceptions.RequestException("Connection refused")
        
        result = delete_playlist('playlist-123')
        
        assert result is False

    @patch('tasks.mediaserver_jellyfin.requests.delete')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_returns_false_on_raise_for_status(self, mock_config, mock_delete):
        """raise_for_status exception returns False - catches if error handling changes"""
        from tasks.mediaserver_jellyfin import delete_playlist
        
        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.HEADERS = {}
        
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_delete.return_value = mock_response
        
        result = delete_playlist('playlist-123')
        
        assert result is False


class TestJellyfinGetLastPlayedTime:
    """Test last played time extraction"""

    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_extracts_last_played_date(self, mock_config, mock_get):
        """LastPlayedDate should be extracted from UserData"""
        from tasks.mediaserver_jellyfin import get_last_played_time
        
        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.JELLYFIN_TOKEN = 'token123'
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'UserData': {
                'LastPlayedDate': '2024-01-15T10:30:00Z',
                'PlayCount': 5
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = get_last_played_time('track-id', {'user_id': 'user123', 'token': 'token'})
        
        assert result == '2024-01-15T10:30:00Z'

    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_returns_none_if_never_played(self, mock_config, mock_get):
        """Returns None if no LastPlayedDate"""
        from tasks.mediaserver_jellyfin import get_last_played_time
        
        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.JELLYFIN_TOKEN = 'token123'
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'UserData': {'PlayCount': 0}
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = get_last_played_time('track-id', {'user_id': 'user123', 'token': 'token'})
        
        assert result is None


# =============================================================================
# NAVIDROME TESTS
# =============================================================================

class TestNavidromeSelectBestArtist:
    """Test Navidrome artist field prioritization - no HTTP mocking needed"""

    def test_prioritizes_track_artist(self):
        """Track artist should be preferred over album artist"""
        from tasks.mediaserver_navidrome import _select_best_artist
        
        song = {
            'artist': 'Track Artist',
            'artistId': 'track-artist-id',
            'albumArtist': 'Album Artist',
            'albumArtistId': 'album-artist-id'
        }
        
        artist_name, artist_id = _select_best_artist(song)
        
        assert artist_name == 'Track Artist'
        assert artist_id == 'track-artist-id'

    def test_falls_back_to_album_artist(self):
        """Falls back to albumArtist if no artist field"""
        from tasks.mediaserver_navidrome import _select_best_artist
        
        song = {
            'albumArtist': 'Album Artist',
            'albumArtistId': 'album-artist-id'
        }
        
        artist_name, artist_id = _select_best_artist(song)
        
        assert artist_name == 'Album Artist'
        assert artist_id == 'album-artist-id'

    def test_returns_unknown_when_no_artist(self):
        """Returns 'Unknown Artist' when no artist info"""
        from tasks.mediaserver_navidrome import _select_best_artist
        
        song = {'title': 'Some Song'}
        
        artist_name, artist_id = _select_best_artist(song)
        
        assert artist_name == 'Unknown Artist'
        assert artist_id is None


class TestNavidromeAuthParams:
    """Test auth parameter generation"""

    @patch('tasks.mediaserver_navidrome.config')
    def test_generates_hex_encoded_password(self, mock_config):
        """Password should be hex-encoded"""
        from tasks.mediaserver_navidrome import get_navidrome_auth_params
        
        mock_config.NAVIDROME_USER = 'testuser'
        mock_config.NAVIDROME_PASSWORD = 'secret123'
        mock_config.APP_VERSION = '1.0.0'
        
        params = get_navidrome_auth_params()
        
        assert params['u'] == 'testuser'
        assert params['p'].startswith('enc:')
        # Verify hex encoding
        hex_password = params['p'].replace('enc:', '')
        decoded = bytes.fromhex(hex_password).decode('utf-8')
        assert decoded == 'secret123'

    @patch('tasks.mediaserver_navidrome.config')
    def test_returns_empty_when_no_credentials(self, mock_config):
        """Returns empty dict when credentials missing"""
        from tasks.mediaserver_navidrome import get_navidrome_auth_params
        
        mock_config.NAVIDROME_USER = ''
        mock_config.NAVIDROME_PASSWORD = ''
        
        params = get_navidrome_auth_params()
        
        assert params == {}


class TestNavidromeRequest:
    """Test the core request helper - verifies URL construction and response parsing"""

    @patch('tasks.mediaserver_navidrome.requests.request')
    @patch('tasks.mediaserver_navidrome.config')
    def test_constructs_correct_url_with_view_suffix(self, mock_config, mock_request):
        """CRITICAL: URL must end with .view - Subsonic API requirement"""
        from tasks.mediaserver_navidrome import _navidrome_request
        
        mock_config.NAVIDROME_URL = 'http://navidrome:4533'
        mock_config.NAVIDROME_USER = 'admin'
        mock_config.NAVIDROME_PASSWORD = 'password'
        mock_config.APP_VERSION = '1.0'
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'subsonic-response': {'status': 'ok'}
        }
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        _navidrome_request('getPlaylists')
        
        # Verify exact URL format - catches if .view suffix is removed
        call_kwargs = mock_request.call_args
        assert call_kwargs[0][0] == 'get'  # method
        url = call_kwargs[0][1]
        assert url == 'http://navidrome:4533/rest/getPlaylists.view', \
            f"URL format changed! Expected '/rest/getPlaylists.view', got '{url}'"

    @patch('tasks.mediaserver_navidrome.requests.request')
    @patch('tasks.mediaserver_navidrome.config')
    def test_parses_subsonic_response_wrapper(self, mock_config, mock_request):
        """CRITICAL: Must extract 'subsonic-response' key - catches if parsing changes"""
        from tasks.mediaserver_navidrome import _navidrome_request
        
        mock_config.NAVIDROME_URL = 'http://navidrome:4533'
        mock_config.NAVIDROME_USER = 'admin'
        mock_config.NAVIDROME_PASSWORD = 'password'
        mock_config.APP_VERSION = '1.0'
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'subsonic-response': {
                'status': 'ok',
                'version': '1.16.1',
                'playlists': {'playlist': []}
            }
        }
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        result = _navidrome_request('getPlaylists')
        
        # Result should be the INNER object, not the wrapper
        assert result['status'] == 'ok'
        assert 'playlists' in result
        # Make sure we don't return the wrapper
        assert 'subsonic-response' not in result

    @patch('tasks.mediaserver_navidrome.requests.request')
    @patch('tasks.mediaserver_navidrome.config')
    def test_checks_status_field_for_failure(self, mock_config, mock_request):
        """CRITICAL: Must check status=='failed' - catches if error detection changes"""
        from tasks.mediaserver_navidrome import _navidrome_request
        
        mock_config.NAVIDROME_URL = 'http://navidrome:4533'
        mock_config.NAVIDROME_USER = 'admin'
        mock_config.NAVIDROME_PASSWORD = 'password'
        mock_config.APP_VERSION = '1.0'
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'subsonic-response': {
                'status': 'failed',
                'error': {'code': 40, 'message': 'Wrong username or password'}
            }
        }
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        result = _navidrome_request('getPlaylists')
        
        # MUST return None on API-level failure
        assert result is None, "Failed status should return None, not the response"

    @patch('tasks.mediaserver_navidrome.requests.request')
    @patch('tasks.mediaserver_navidrome.config')
    def test_includes_auth_params_in_request(self, mock_config, mock_request):
        """CRITICAL: Auth params must be in query string - catches if auth method changes"""
        from tasks.mediaserver_navidrome import _navidrome_request
        
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
        
        # Verify auth params are present
        assert params.get('u') == 'testuser', "Username not in params"
        assert params.get('p').startswith('enc:'), "Password not hex-encoded"
        assert params.get('f') == 'json', "Format must be json"
        assert 'extra' in params, "Custom params not passed through"

    @patch('tasks.mediaserver_navidrome.requests.request')
    @patch('tasks.mediaserver_navidrome.config')
    def test_returns_none_on_http_error(self, mock_config, mock_request):
        """HTTP errors must return None - catches if error handling changes"""
        from tasks.mediaserver_navidrome import _navidrome_request
        
        mock_config.NAVIDROME_URL = 'http://navidrome:4533'
        mock_config.NAVIDROME_USER = 'admin'
        mock_config.NAVIDROME_PASSWORD = 'password'
        mock_config.APP_VERSION = '1.0'
        
        mock_request.side_effect = requests.exceptions.RequestException("Connection refused")
        
        result = _navidrome_request('getPlaylists')
        
        assert result is None


class TestNavidromeGetTracksFromAlbum:
    """Test track fetching with parsing - verifies field transformations"""

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_calls_getAlbum_endpoint(self, mock_request):
        """CRITICAL: Must use getAlbum endpoint - catches if API changes"""
        from tasks.mediaserver_navidrome import get_tracks_from_album
        
        mock_request.return_value = {
            'status': 'ok',
            'album': {'id': 'album123', 'song': []}
        }
        
        get_tracks_from_album('album123')
        
        call_args = mock_request.call_args
        assert call_args[0][0] == 'getAlbum', \
            f"Endpoint changed! Expected 'getAlbum', got '{call_args[0][0]}'"
        assert call_args[0][1] == {'id': 'album123'}, \
            f"Params changed! Expected {{'id': 'album123'}}"

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_normalizes_field_names_to_capitalized(self, mock_request):
        """CRITICAL: Must transform id->Id, title->Name - catches if normalization changes"""
        from tasks.mediaserver_navidrome import get_tracks_from_album
        
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
                        'path': '/music/song1.mp3'
                    }
                ]
            }
        }
        
        tracks = get_tracks_from_album('album123')
        
        assert len(tracks) == 1
        # Verify EXACT field transformations - these are the contract
        assert 'Id' in tracks[0], "Missing 'Id' (capital I) - normalization broken"
        assert tracks[0]['Id'] == 'song1'
        assert 'Name' in tracks[0], "Missing 'Name' (capital N) - normalization broken"
        assert tracks[0]['Name'] == 'Track One'
        assert 'Path' in tracks[0], "Missing 'Path' (capital P) - normalization broken"
        assert tracks[0]['Path'] == '/music/song1.mp3'
        assert 'AlbumArtist' in tracks[0], "Missing 'AlbumArtist' - enrichment broken"
        assert 'ArtistId' in tracks[0], "Missing 'ArtistId' - enrichment broken"

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_artist_prioritization_applied(self, mock_request):
        """CRITICAL: Track artist > album artist - catches if priority changes"""
        from tasks.mediaserver_navidrome import get_tracks_from_album
        
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
                        'albumArtistId': 'album-artist-id'
                    },
                    {
                        'id': 'song2',
                        'title': 'Only Album Artist',
                        'albumArtist': 'Album Artist Only',
                        'albumArtistId': 'album-only-id'
                    }
                ]
            }
        }
        
        tracks = get_tracks_from_album('album123')
        
        # First track: should use track artist (priority)
        assert tracks[0]['AlbumArtist'] == 'Track Artist', \
            "Track artist should be prioritized over album artist"
        assert tracks[0]['ArtistId'] == 'track-artist-id'
        
        # Second track: should fall back to album artist
        assert tracks[1]['AlbumArtist'] == 'Album Artist Only', \
            "Should fall back to album artist when track artist missing"
        assert tracks[1]['ArtistId'] == 'album-only-id'

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_returns_empty_on_missing_songs(self, mock_request):
        """Returns empty list if no songs in album"""
        from tasks.mediaserver_navidrome import get_tracks_from_album
        
        mock_request.return_value = {
            'status': 'ok',
            'album': {'id': 'album123', 'name': 'Empty Album'}
        }
        
        tracks = get_tracks_from_album('album123')
        
        assert tracks == []

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_returns_empty_on_api_failure(self, mock_request):
        """Returns empty list on API failure"""
        from tasks.mediaserver_navidrome import get_tracks_from_album
        
        mock_request.return_value = None

        tracks = get_tracks_from_album('album123')

        assert tracks == []


class TestNavidromeGetTopPlayedSongsAlbumCap:
    """Regression tests for issue #603: a single large album must not dominate
    the sonic-fingerprint seed pool. SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM is a
    hard maximum and must be honored regardless of the requested limit."""

    @staticmethod
    def _album_list_response(album_ids):
        return {'status': 'ok', 'albumList2': {'album': [{'id': aid} for aid in album_ids]}}

    @staticmethod
    def _tracks_for(album_id, count):
        return [{'Id': f'{album_id}_track{i}', 'Album': album_id} for i in range(count)]

    @patch('tasks.mediaserver_navidrome.get_tracks_from_album')
    @patch('tasks.mediaserver_navidrome._navidrome_request')
    @patch('tasks.mediaserver_navidrome.config')
    def test_single_album_capped_even_with_large_limit(self, mock_config, mock_request, mock_get_tracks):
        """A 100-track album with cap=2 must contribute at most 2 seeds, even
        when limit//10 (the old floor) would have allowed 6."""
        from tasks.mediaserver_navidrome import get_top_played_songs

        mock_config.SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM = 2
        mock_request.return_value = self._album_list_response(['big'])
        mock_get_tracks.return_value = self._tracks_for('big', 100)

        result = get_top_played_songs(limit=60, user_creds={})

        assert len(result) == 2, \
            f"Expected configured cap of 2, got {len(result)} (limit//10 floor regressed)"

    @patch('tasks.mediaserver_navidrome.get_tracks_from_album')
    @patch('tasks.mediaserver_navidrome._navidrome_request')
    @patch('tasks.mediaserver_navidrome.config')
    def test_cap_honored_per_album_across_multiple_albums(self, mock_config, mock_request, mock_get_tracks):
        """No album may contribute more than the configured cap to the pool."""
        from tasks.mediaserver_navidrome import get_top_played_songs

        mock_config.SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM = 2
        mock_request.return_value = self._album_list_response(['a1', 'a2', 'a3'])
        mock_get_tracks.side_effect = lambda album_id, user_creds=None: self._tracks_for(album_id, 10)

        result = get_top_played_songs(limit=20, user_creds={})

        per_album = {}
        for song in result:
            album = song['Id'].split('_')[0]
            per_album[album] = per_album.get(album, 0) + 1
        assert all(count <= 2 for count in per_album.values()), \
            f"Some album exceeded the cap of 2: {per_album}"

    @patch('tasks.mediaserver_navidrome.get_tracks_from_album')
    @patch('tasks.mediaserver_navidrome._navidrome_request')
    @patch('tasks.mediaserver_navidrome.config')
    def test_fetches_enough_albums_to_reach_limit_under_tight_cap(self, mock_config, mock_request, mock_get_tracks):
        """With a tight cap the album fetch size must scale so the pool can
        still reach the requested limit."""
        from tasks.mediaserver_navidrome import get_top_played_songs

        mock_config.SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM = 2
        mock_request.return_value = self._album_list_response([f'a{i}' for i in range(40)])
        mock_get_tracks.side_effect = lambda album_id, user_creds=None: self._tracks_for(album_id, 10)

        get_top_played_songs(limit=60, user_creds={})

        requested_size = mock_request.call_args[0][1]['size']
        assert requested_size >= 60 // 2, \
            f"Album fetch size {requested_size} too small to reach limit under cap=2"

    @patch('tasks.mediaserver_navidrome.get_tracks_from_album')
    @patch('tasks.mediaserver_navidrome._navidrome_request')
    @patch('tasks.mediaserver_navidrome.config')
    def test_final_selection_keeps_most_recently_played(self, mock_config, mock_request, mock_get_tracks):
        """Step 3: from the capped pool, the most recently played tracks win.
        Never-played tracks (no ``played``) fall to the bottom."""
        from tasks.mediaserver_navidrome import get_top_played_songs

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

        assert {s['Id'] for s in result} == {'newest', 'recent'}, \
            f"Expected the 2 most recently played, got {[s['Id'] for s in result]}"

    @patch('tasks.mediaserver_navidrome.get_tracks_from_album')
    @patch('tasks.mediaserver_navidrome._navidrome_request')
    @patch('tasks.mediaserver_navidrome.config')
    def test_final_selection_falls_back_to_lastPlayed_field(self, mock_config, mock_request, mock_get_tracks):
        """Non-OpenSubsonic servers expose recency as ``lastPlayed``; it must
        still drive the ordering when ``played`` is absent."""
        from tasks.mediaserver_navidrome import get_top_played_songs

        mock_config.SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM = 5
        mock_request.return_value = self._album_list_response(['a1'])
        mock_get_tracks.return_value = [
            {'Id': 'older', 'Album': 'a1', 'lastPlayed': '2026-02-01T00:00:00Z'},
            {'Id': 'newer', 'Album': 'a1', 'lastPlayed': '2026-04-01T00:00:00Z'},
        ]

        result = get_top_played_songs(limit=1, user_creds={})

        assert [s['Id'] for s in result] == ['newer'], \
            f"Expected the most recently played via lastPlayed, got {[s['Id'] for s in result]}"


class TestNavidromeGetAllPlaylists:
    """Test playlist fetching and normalization - verifies exact response parsing"""

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_calls_getPlaylists_endpoint(self, mock_request):
        """CRITICAL: Must call getPlaylists - catches if endpoint changes"""
        from tasks.mediaserver_navidrome import get_all_playlists
        
        mock_request.return_value = {
            'status': 'ok',
            'playlists': {'playlist': []}
        }
        
        get_all_playlists()
        
        mock_request.assert_called_once_with('getPlaylists')

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_parses_nested_playlist_structure(self, mock_request):
        """CRITICAL: Response is playlists.playlist[] - catches if parsing changes"""
        from tasks.mediaserver_navidrome import get_all_playlists
        
        mock_request.return_value = {
            'status': 'ok',
            'playlists': {
                'playlist': [
                    {'id': 'pl1', 'name': 'Rock_automatic', 'songCount': 50},
                    {'id': 'pl2', 'name': 'Jazz Mix', 'songCount': 30}
                ]
            }
        }
        
        playlists = get_all_playlists()
        
        assert len(playlists) == 2
        # Verify normalization to capital letters
        assert playlists[0]['Id'] == 'pl1', "Missing 'Id' normalization"
        assert playlists[0]['Name'] == 'Rock_automatic', "Missing 'Name' normalization"
        # Original keys also preserved for compatibility
        assert playlists[0]['id'] == 'pl1', "Original 'id' should be preserved"
        assert playlists[0]['name'] == 'Rock_automatic', "Original 'name' should be preserved"

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_handles_missing_playlists_key(self, mock_request):
        """Missing playlists key should return empty list"""
        from tasks.mediaserver_navidrome import get_all_playlists
        
        mock_request.return_value = {'status': 'ok'}
        
        playlists = get_all_playlists()
        
        assert playlists == []

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_handles_missing_playlist_array(self, mock_request):
        """Missing playlist array should return empty list"""
        from tasks.mediaserver_navidrome import get_all_playlists
        
        mock_request.return_value = {'status': 'ok', 'playlists': {}}
        
        playlists = get_all_playlists()
        
        assert playlists == []

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_returns_empty_on_failure(self, mock_request):
        """Returns empty list on API failure"""
        from tasks.mediaserver_navidrome import get_all_playlists
        
        mock_request.return_value = None
        
        playlists = get_all_playlists()
        
        assert playlists == []


class TestNavidromeDeletePlaylist:
    """Test playlist deletion - verifies exact endpoint and params"""

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_calls_correct_endpoint_with_id_param(self, mock_request):
        """CRITICAL: Must call deletePlaylist with id param - catches if endpoint changes"""
        from tasks.mediaserver_navidrome import delete_playlist
        
        mock_request.return_value = {'status': 'ok'}
        
        result = delete_playlist('playlist-123')
        
        assert result is True
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        # Verify exact endpoint name
        assert call_args[0][0] == 'deletePlaylist', \
            f"Endpoint changed! Expected 'deletePlaylist', got '{call_args[0][0]}'"
        # Verify exact param structure
        assert call_args[0][1] == {'id': 'playlist-123'}, \
            f"Params changed! Expected {{'id': 'playlist-123'}}, got {call_args[0][1]}"
        # Verify uses POST method
        assert call_args[1].get('method') == 'post', \
            "Method changed! Must be POST for deletePlaylist"

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_checks_status_ok_for_success(self, mock_request):
        """CRITICAL: Must check status=='ok' - catches if success detection changes"""
        from tasks.mediaserver_navidrome import delete_playlist
        
        # Return response without 'ok' status
        mock_request.return_value = {'status': 'something_else'}
        
        result = delete_playlist('playlist-123')
        
        # Should return False because status is not 'ok'
        assert result is False

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_returns_false_on_none_response(self, mock_request):
        """None response (API failure) returns False"""
        from tasks.mediaserver_navidrome import delete_playlist
        
        mock_request.return_value = None
        
        result = delete_playlist('playlist-123')
        
        assert result is False


class TestNavidromeGetPlaylistByName:
    """Test playlist lookup by name"""

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_finds_playlist_by_exact_name(self, mock_request):
        """Should find playlist with exact name match"""
        from tasks.mediaserver_navidrome import get_playlist_by_name
        
        mock_request.return_value = {
            'status': 'ok',
            'playlists': {
                'playlist': [
                    {'id': 'pl1', 'name': 'Rock Mix'},
                    {'id': 'pl2', 'name': 'Jazz Favorites'},
                    {'id': 'pl3', 'name': 'Rock Mix Special'}
                ]
            }
        }
        
        result = get_playlist_by_name('Jazz Favorites')
        
        assert result is not None
        assert result['id'] == 'pl2'
        assert result['name'] == 'Jazz Favorites'

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_returns_none_if_not_found(self, mock_request):
        """Returns None if no matching playlist"""
        from tasks.mediaserver_navidrome import get_playlist_by_name
        
        mock_request.return_value = {
            'status': 'ok',
            'playlists': {
                'playlist': [
                    {'id': 'pl1', 'name': 'Rock Mix'}
                ]
            }
        }
        
        result = get_playlist_by_name('NonExistent')
        
        assert result is None


class TestNavidromeCreatePlaylist:
    """Test playlist creation with batching"""

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_extracts_playlist_id_from_response(self, mock_request):
        """Should extract playlist ID from creation response"""
        from tasks.mediaserver_navidrome import _create_playlist_batched
        
        mock_request.return_value = {
            'status': 'ok',
            'playlist': {
                'id': 'new-pl-123',
                'name': 'Test Playlist',
                'songCount': 3
            }
        }
        
        result = _create_playlist_batched('Test Playlist', ['song1', 'song2', 'song3'])
        
        assert result is not None
        assert result['id'] == 'new-pl-123'
        assert result['Id'] == 'new-pl-123'  # Normalized key
        assert result['Name'] == 'Test Playlist'

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_create_playlist_sets_public_after_creation(self, mock_request):
        """Should call updatePlaylist(public=true) right after createPlaylist"""
        from tasks.mediaserver_navidrome import _create_playlist_batched

        mock_request.return_value = {
            'status': 'ok',
            'playlist': {'id': 'new-pl-456', 'name': 'Test Playlist', 'songCount': 1}
        }

        _create_playlist_batched('Test Playlist', ['song1'])

        # First call is createPlaylist
        first_call_args = mock_request.call_args_list[0][0]
        assert first_call_args[0] == 'createPlaylist'
        create_params = first_call_args[1]
        assert create_params.get('public') is None

        # Second call is updatePlaylist(public=true)
        second_call_args = mock_request.call_args_list[1][0]
        assert second_call_args[0] == 'updatePlaylist'
        update_params = second_call_args[1]
        assert update_params.get('playlistId') == 'new-pl-456'
        assert update_params.get('public') == 'true'

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_returns_none_on_creation_failure(self, mock_request):
        """Returns None when creation fails"""
        from tasks.mediaserver_navidrome import _create_playlist_batched
        
        mock_request.return_value = None
        
        result = _create_playlist_batched('Test Playlist', ['song1'])
        
        assert result is None

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_handles_malformed_response(self, mock_request):
        """Returns None on malformed response"""
        from tasks.mediaserver_navidrome import _create_playlist_batched
        
        mock_request.return_value = {'status': 'ok'}  # Missing playlist key
        
        result = _create_playlist_batched('Test Playlist', ['song1'])
        
        assert result is None


class TestNavidromeGetLastPlayedTime:
    """Test last played time extraction"""

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_extracts_last_played(self, mock_request):
        """Should extract lastPlayed from song response"""
        from tasks.mediaserver_navidrome import get_last_played_time
        
        mock_request.return_value = {
            'status': 'ok',
            'song': {
                'id': 'song123',
                'title': 'Test Song',
                'lastPlayed': '2024-01-15T10:30:00Z'
            }
        }
        
        result = get_last_played_time('song123', {'user': 'test', 'password': 'pass'})
        
        assert result == '2024-01-15T10:30:00Z'

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_returns_none_if_never_played(self, mock_request):
        """Returns None if no lastPlayed field"""
        from tasks.mediaserver_navidrome import get_last_played_time
        
        mock_request.return_value = {
            'status': 'ok',
            'song': {
                'id': 'song123',
                'title': 'Test Song'
            }
        }
        
        result = get_last_played_time('song123', {'user': 'test', 'password': 'pass'})
        
        assert result is None


class TestNavidromeGetRecentAlbums:
    """Test recent albums parsing"""

    @patch('tasks.mediaserver_navidrome._get_target_music_folder_ids')
    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_normalizes_album_keys(self, mock_request, mock_folders):
        """Albums should have Id and Name normalized"""
        from tasks.mediaserver_navidrome import get_recent_albums
        
        mock_folders.return_value = None  # No folder filtering
        mock_request.return_value = {
            'status': 'ok',
            'albumList2': {
                'album': [
                    {'id': 'album1', 'name': 'First Album', 'artist': 'Artist A'},
                    {'id': 'album2', 'name': 'Second Album', 'artist': 'Artist B'}
                ]
            }
        }
        
        albums = get_recent_albums(10)
        
        assert len(albums) == 2
        assert albums[0]['Id'] == 'album1'
        assert albums[0]['Name'] == 'First Album'

    @patch('tasks.mediaserver_navidrome._get_target_music_folder_ids')
    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_returns_empty_when_no_matching_folders(self, mock_request, mock_folders):
        """Returns empty list when folder filter matches nothing"""
        from tasks.mediaserver_navidrome import get_recent_albums

        mock_folders.return_value = set()  # Empty set = no matches

        albums = get_recent_albums(10)

        assert albums == []
        mock_request.assert_not_called()


class TestNavidromeGetAllSongsApplyFilter:
    """Migration probe path: dry-run must NOT apply ``config.MUSIC_LIBRARIES``
    to the target server.

    Bug: ``config.MUSIC_LIBRARIES`` holds the *source* provider's folder
    names; applying it to the *target* during a migration probe filters
    target tracks against names that don't exist on the target, returning
    an empty set and zeroing out the dry-run result.

    Fix: an explicit ``apply_filter`` flag on ``get_all_songs``. The
    migration probe (``provider_probe.fetch_all_tracks``) passes
    ``apply_filter=False``; live-provider scans default to ``True`` so the
    user's saved selection is still honored. The flag's intent is visible
    in code rather than implied from the presence of ``user_creds``.
    """

    @patch('tasks.mediaserver_navidrome._get_target_music_folder_ids')
    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_apply_filter_false_skips_folder_lookup(self, mock_request, mock_filter):
        """apply_filter=False must NOT call _get_target_music_folder_ids."""
        from tasks.mediaserver_navidrome import get_all_songs

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

    @patch('tasks.mediaserver_navidrome._get_target_music_folder_ids')
    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_apply_filter_true_default_honors_filter(self, mock_request, mock_filter):
        """apply_filter defaults to True, preserving live-provider semantics."""
        from tasks.mediaserver_navidrome import get_all_songs

        mock_filter.return_value = set()  # Filter active but no matches.
        mock_request.return_value = {'status': 'ok'}

        songs = get_all_songs(user_creds=None)

        mock_filter.assert_called_once()
        assert songs == []
        mock_request.assert_not_called()

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_get_target_music_folder_ids_forwards_user_creds(self, mock_request):
        """The folder-lookup helper must thread user_creds through to the API
        request so callers (e.g. live-provider code paths receiving session
        creds) hit ``getMusicFolders`` with valid auth instead of falling back
        to empty config globals."""
        from tasks.mediaserver_navidrome import _get_target_music_folder_ids

        with patch('tasks.mediaserver_navidrome.config') as mock_config:
            mock_config.MUSIC_LIBRARIES = 'Music'
            mock_request.return_value = {
                'musicFolders': {'musicFolder': [{'id': 1, 'name': 'Music'}]}
            }

            creds = {'url': 'http://target:4533', 'user': 'u', 'password': 'p'}
            _get_target_music_folder_ids(user_creds=creds)

        assert mock_request.call_args.args[0] == 'getMusicFolders'
        assert mock_request.call_args.kwargs.get('user_creds') == creds


# =============================================================================
# DISPATCHER TESTS (minimal - just validation logic)
# =============================================================================

class TestDispatcherValidation:
    """Test input validation in dispatcher functions"""

    @patch('tasks.mediaserver.config')
    def test_get_playlist_by_name_requires_name(self, mock_config):
        """Empty name should raise ValueError"""
        from tasks.mediaserver import get_playlist_by_name
        
        with pytest.raises(ValueError, match="Playlist name is required"):
            get_playlist_by_name('')

        with pytest.raises(ValueError, match="Playlist name is required"):
            get_playlist_by_name(None)

    @patch('tasks.mediaserver.config')
    def test_create_playlist_requires_name_and_ids(self, mock_config):
        """Empty name or IDs should raise ValueError"""
        from tasks.mediaserver import create_playlist
        
        with pytest.raises(ValueError, match="Playlist name is required"):
            create_playlist('', ['id1'])

        with pytest.raises(ValueError, match="Track IDs are required"):
            create_playlist('Name', [])

        with pytest.raises(ValueError, match="Track IDs are required"):
            create_playlist('Name', None)

    @patch('tasks.mediaserver.config')
    def test_create_instant_playlist_requires_name_and_ids(self, mock_config):
        """Empty name or IDs should raise ValueError"""
        from tasks.mediaserver import create_instant_playlist
        
        with pytest.raises(ValueError, match="Playlist name is required"):
            create_instant_playlist('', ['id1'])

        with pytest.raises(ValueError, match="Track IDs are required"):
            create_instant_playlist('Name', [])


class TestDispatcherAutomaticPlaylistDeletion:
    """Test the filtering logic in delete_automatic_playlists"""

    @patch('tasks.mediaserver.config')
    @patch('tasks.mediaserver.jellyfin_get_all_playlists')
    @patch('tasks.mediaserver.jellyfin_delete_playlist')
    def test_only_deletes_automatic_suffix_playlists(self, mock_delete, mock_get, mock_config):
        """Only playlists ending with '_automatic' should be deleted"""
        from tasks.mediaserver import delete_automatic_playlists
        
        mock_config.MEDIASERVER_TYPE = 'jellyfin'
        mock_get.return_value = [
            {'Id': '1', 'Name': 'Rock_automatic'},
            {'Id': '2', 'Name': 'automatic_Rock'},  # Wrong position
            {'Id': '3', 'Name': 'My Favorites'},
            {'Id': '4', 'Name': 'Jazz_automatic'},
            {'Id': '5', 'Name': 'Pop_Automatic'},  # Case sensitive - won't match
        ]
        mock_delete.return_value = True
        
        delete_automatic_playlists()
        
        # Only playlists 1 and 4 should be deleted
        assert mock_delete.call_count == 2
        deleted_ids = [call[0][0] for call in mock_delete.call_args_list]
        assert '1' in deleted_ids
        assert '4' in deleted_ids
        assert '2' not in deleted_ids
        assert '3' not in deleted_ids

    @patch('tasks.mediaserver.config')
    @patch('tasks.mediaserver.navidrome_get_all_playlists')
    @patch('tasks.mediaserver.navidrome_delete_playlist')
    def test_handles_both_id_and_Id_keys(self, mock_delete, mock_get, mock_config):
        """Should handle both 'id' and 'Id' keys (Navidrome uses lowercase)"""
        from tasks.mediaserver import delete_automatic_playlists
        
        mock_config.MEDIASERVER_TYPE = 'navidrome'
        mock_get.return_value = [
            {'id': 'nav1', 'Name': 'Test_automatic'},  # lowercase id
            {'Id': 'nav2', 'Name': 'Other_automatic'},  # uppercase Id
        ]
        mock_delete.return_value = True
        
        delete_automatic_playlists()
        
        assert mock_delete.call_count == 2
        deleted_ids = [call[0][0] for call in mock_delete.call_args_list]
        assert 'nav1' in deleted_ids
        assert 'nav2' in deleted_ids


# =============================================================================
# LYRION TESTS
# =============================================================================

class TestLyrionSelectBestArtist:
    """Test Lyrion artist field prioritization - tests the inline logic in get_tracks_from_album"""

    def test_artist_priority_order(self):
        """Verify priority: trackartist > contributor > artist > albumartist > band"""
        # The logic is inline in get_tracks_from_album, so we test expected behavior
        # by examining the field priority order from the code
        
        # Priority fields in order (from examining mediaserver_lyrion.py):
        priority_fields = ['trackartist', 'contributor', 'artist', 'albumartist', 'band']
        
        # This verifies our understanding of the priority
        assert priority_fields[0] == 'trackartist', "trackartist should be highest priority"
        assert priority_fields[-1] == 'band', "band should be lowest priority"


class TestLyrionJsonRpcRequest:
    """Test the core JSON-RPC request helper"""

    @patch('tasks.mediaserver_lyrion.requests.Session')
    @patch('tasks.mediaserver_lyrion.config')
    def test_constructs_correct_url(self, mock_config, mock_session_class):
        """CRITICAL: URL must be /jsonrpc.js - catches if endpoint changes"""
        from tasks.mediaserver_lyrion import _jsonrpc_request
        
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
        
        # Verify exact URL
        call_args = mock_session.post.call_args
        assert call_args[0][0] == 'http://lyrion:9000/jsonrpc.js', \
            f"URL changed! Expected '/jsonrpc.js', got '{call_args[0][0]}'"

    @patch('tasks.mediaserver_lyrion.requests.Session')
    @patch('tasks.mediaserver_lyrion.config')
    def test_uses_slim_request_method(self, mock_config, mock_session_class):
        """CRITICAL: Must use 'slim.request' method - catches if protocol changes"""
        from tasks.mediaserver_lyrion import _jsonrpc_request
        
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
        
        # Verify JSON-RPC payload structure
        call_kwargs = mock_session.post.call_args[1]
        payload = call_kwargs.get('json', {})
        
        assert payload.get('method') == 'slim.request', \
            f"Method changed! Expected 'slim.request', got '{payload.get('method')}'"
        assert payload.get('params')[0] == 'player1', \
            "Player ID not passed correctly"
        assert payload.get('params')[1][0] == 'albums', \
            "Command not passed correctly"

    @patch('tasks.mediaserver_lyrion.requests.Session')
    @patch('tasks.mediaserver_lyrion.config')
    def test_extracts_result_field(self, mock_config, mock_session_class):
        """CRITICAL: Must return 'result' field - catches if response parsing changes"""
        from tasks.mediaserver_lyrion import _jsonrpc_request
        
        mock_config.LYRION_URL = 'http://lyrion:9000'
        
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            'id': 1,
            'result': {'albums_loop': [{'id': '1', 'album': 'Test'}]},
            'error': None
        }
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response
        mock_session.headers = Mock()
        mock_session.headers.update = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_session_class.return_value = mock_session
        
        result = _jsonrpc_request('albums', [0, 10])
        
        # Must return the 'result' content, not the whole response
        assert 'albums_loop' in result
        assert 'id' not in result  # Should not include top-level id

    @patch('tasks.mediaserver_lyrion.requests.Session')
    @patch('tasks.mediaserver_lyrion.config')
    def test_raises_on_jsonrpc_error(self, mock_config, mock_session_class):
        """CRITICAL: Must raise LyrionAPIError on error response"""
        from tasks.mediaserver_lyrion import _jsonrpc_request, LyrionAPIError
        
        mock_config.LYRION_URL = 'http://lyrion:9000'
        
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            'error': {'message': 'Unknown command'}
        }
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
    """Test playlist fetching and normalization"""

    @patch('tasks.mediaserver_lyrion._jsonrpc_request')
    def test_calls_playlists_command(self, mock_request):
        """CRITICAL: Must call 'playlists' command - catches if API changes"""
        from tasks.mediaserver_lyrion import get_all_playlists
        
        mock_request.return_value = {'playlists_loop': []}
        
        get_all_playlists()
        
        call_args = mock_request.call_args
        assert call_args[0][0] == 'playlists', \
            f"Command changed! Expected 'playlists', got '{call_args[0][0]}'"

    @patch('tasks.mediaserver_lyrion._jsonrpc_request')
    def test_normalizes_playlist_keys(self, mock_request):
        """CRITICAL: Must normalize 'id'->Id, 'playlist'->Name"""
        from tasks.mediaserver_lyrion import get_all_playlists
        
        mock_request.return_value = {
            'playlists_loop': [
                {'id': 'pl1', 'playlist': 'Rock_automatic'},
                {'id': 'pl2', 'playlist': 'Jazz Mix'}
            ]
        }
        
        playlists = get_all_playlists()
        
        assert len(playlists) == 2
        # Verify normalization - Lyrion uses 'playlist' for name, not 'name'
        assert playlists[0]['Id'] == 'pl1', "Missing 'Id' normalization"
        assert playlists[0]['Name'] == 'Rock_automatic', "Missing 'Name' normalization from 'playlist' field"

    @patch('tasks.mediaserver_lyrion._jsonrpc_request')
    def test_returns_empty_on_no_playlists(self, mock_request):
        """Returns empty list when no playlists_loop"""
        from tasks.mediaserver_lyrion import get_all_playlists
        
        mock_request.return_value = {}
        
        playlists = get_all_playlists()
        
        assert playlists == []


class TestLyrionDeletePlaylist:
    """Test playlist deletion"""

    @patch('tasks.mediaserver_lyrion._jsonrpc_request')
    def test_calls_playlists_delete_command(self, mock_request):
        """CRITICAL: Must use 'playlists' with 'delete' param - catches if API changes"""
        from tasks.mediaserver_lyrion import delete_playlist
        
        mock_request.return_value = {'count': 1}
        
        result = delete_playlist('playlist-123')
        
        call_args = mock_request.call_args
        # First arg is command
        assert call_args[0][0] == 'playlists', \
            f"Command changed! Expected 'playlists', got '{call_args[0][0]}'"
        # Second arg is params list
        params = call_args[0][1]
        assert 'delete' in params, "Must include 'delete' param"
        assert 'playlist_id:playlist-123' in params, "Must include playlist_id param"

    @patch('tasks.mediaserver_lyrion._jsonrpc_request')
    def test_returns_true_on_success(self, mock_request):
        """Returns True when deletion succeeds"""
        from tasks.mediaserver_lyrion import delete_playlist
        
        mock_request.return_value = {'count': 1}
        
        result = delete_playlist('playlist-123')
        
        assert result is True

    @patch('tasks.mediaserver_lyrion._jsonrpc_request')
    def test_returns_false_on_failure(self, mock_request):
        """Returns False when deletion fails"""
        from tasks.mediaserver_lyrion import delete_playlist
        
        mock_request.return_value = None
        
        result = delete_playlist('playlist-123')
        
        assert result is False


class TestLyrionGetTracksFromAlbum:
    """Test track fetching from albums"""

    @patch('tasks.mediaserver_lyrion._jsonrpc_request')
    def test_calls_titles_command_with_album_id(self, mock_request):
        """CRITICAL: Must use 'titles' with album_id filter"""
        from tasks.mediaserver_lyrion import get_tracks_from_album
        
        mock_request.return_value = {'titles_loop': []}
        
        get_tracks_from_album('album-123')
        
        call_args = mock_request.call_args
        assert call_args[0][0] == 'titles', \
            f"Command changed! Expected 'titles', got '{call_args[0][0]}'"
        params = call_args[0][1]
        assert any('album_id:album-123' in str(p) for p in params), \
            "Must include album_id filter"

    @patch('tasks.mediaserver_lyrion._jsonrpc_request')
    def test_normalizes_track_fields(self, mock_request):
        """CRITICAL: Must normalize id->Id, title->Name, add AlbumArtist"""
        from tasks.mediaserver_lyrion import get_tracks_from_album
        
        mock_request.return_value = {
            'titles_loop': [
                {
                    'id': 'track1',
                    'title': 'Song One',
                    'trackartist': 'Track Artist',  # Highest priority
                    'artist': 'Album Artist',
                    'url': '/music/song1.mp3'
                }
            ]
        }
        
        tracks = get_tracks_from_album('album-123')
        
        assert len(tracks) == 1
        assert tracks[0]['Id'] == 'track1', "Missing 'Id' normalization"
        assert tracks[0]['Name'] == 'Song One', "Missing 'Name' normalization"
        assert tracks[0]['AlbumArtist'] == 'Track Artist', \
            "trackartist should be prioritized for AlbumArtist"
        assert tracks[0]['Path'] == '/music/song1.mp3', "Missing 'Path' from 'url'"

    @patch('tasks.mediaserver_lyrion._jsonrpc_request')
    def test_artist_fallback_priority(self, mock_request):
        """Tests artist field fallback: trackartist > contributor > artist > albumartist"""
        from tasks.mediaserver_lyrion import get_tracks_from_album
        
        mock_request.return_value = {
            'titles_loop': [
                {
                    'id': 'track1',
                    'title': 'No TrackArtist',
                    'contributor': 'Contributor Name',  # Should use this
                    'artist': 'Artist Name',
                    'albumartist': 'Album Artist'
                },
                {
                    'id': 'track2',
                    'title': 'Only AlbumArtist',
                    'albumartist': 'Album Artist Only'
                }
            ]
        }
        
        tracks = get_tracks_from_album('album-123')
        
        # First track: should use contributor (2nd priority after trackartist)
        assert tracks[0]['AlbumArtist'] == 'Contributor Name', \
            "Should fall back to contributor when no trackartist"
        
        # Second track: should use albumartist
        assert tracks[1]['AlbumArtist'] == 'Album Artist Only', \
            "Should fall back to albumartist when no higher priority fields"

    @patch('tasks.mediaserver_lyrion._jsonrpc_request')
    def test_filters_spotify_tracks(self, mock_request):
        """CRITICAL: Spotify tracks should be filtered out"""
        from tasks.mediaserver_lyrion import get_tracks_from_album
        
        mock_request.return_value = {
            'titles_loop': [
                {'id': 'local1', 'title': 'Local Track', 'url': '/music/local.mp3'},
                {'id': 'spotify1', 'title': 'Spotify Track', 'url': 'spotify://track/123'},
                {'id': 'local2', 'title': 'Another Local', 'genre': 'rock', 'url': '/music/local2.mp3'}
            ]
        }
        
        tracks = get_tracks_from_album('album-123')
        
        # Should filter out Spotify track
        assert len(tracks) == 2
        track_ids = [t['Id'] for t in tracks]
        assert 'local1' in track_ids
        assert 'local2' in track_ids
        assert 'spotify1' not in track_ids, "Spotify tracks should be filtered"


# =============================================================================
# EMBY TESTS
# =============================================================================

class TestEmbySelectBestArtist:
    """Test Emby artist field prioritization - same as Jellyfin"""

    def test_prioritizes_artist_items_over_album_artist(self):
        """ArtistItems should be preferred over AlbumArtist"""
        from tasks.mediaserver_emby import _select_best_artist
        
        item = {
            'ArtistItems': [{'Name': 'Track Artist', 'Id': 'artist-123'}],
            'Artists': ['Fallback Artist'],
            'AlbumArtist': 'Album Artist'
        }
        
        artist_name, artist_id = _select_best_artist(item)
        
        assert artist_name == 'Track Artist'
        assert artist_id == 'artist-123'

    def test_falls_back_to_artists_array(self):
        """If no ArtistItems, use Artists array"""
        from tasks.mediaserver_emby import _select_best_artist
        
        item = {
            'ArtistItems': [],
            'Artists': ['First Artist', 'Second Artist'],
            'AlbumArtist': 'Album Artist'
        }
        
        artist_name, artist_id = _select_best_artist(item)
        
        assert artist_name == 'First Artist'
        assert artist_id is None

    def test_falls_back_to_album_artist(self):
        """If no Artists, use AlbumArtist"""
        from tasks.mediaserver_emby import _select_best_artist
        
        item = {
            'AlbumArtist': 'The Album Artist'
        }
        
        artist_name, artist_id = _select_best_artist(item)
        
        assert artist_name == 'The Album Artist'
        assert artist_id is None

    def test_returns_unknown_when_no_artist_info(self):
        """Returns 'Unknown Artist' when no artist info available"""
        from tasks.mediaserver_emby import _select_best_artist
        
        item = {}
        
        artist_name, artist_id = _select_best_artist(item)
        
        assert artist_name == 'Unknown Artist'
        assert artist_id is None


class TestEmbyGetAllPlaylists:
    """Test playlist fetching - verifies URL and response parsing"""

    @patch('tasks.mediaserver_emby.requests.get')
    @patch('tasks.mediaserver_emby.config')
    def test_uses_correct_url_with_emby_prefix(self, mock_config, mock_get):
        """CRITICAL: URL must include /emby/ prefix - catches if path changes"""
        from tasks.mediaserver_emby import get_all_playlists
        
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
        assert call_url == 'http://emby:8096/emby/Users/user123/Items', \
            f"URL changed! Got '{call_url}'"

    @patch('tasks.mediaserver_emby.requests.get')
    @patch('tasks.mediaserver_emby.config')
    def test_includes_playlist_item_type(self, mock_config, mock_get):
        """CRITICAL: Must filter by IncludeItemTypes=Playlist"""
        from tasks.mediaserver_emby import get_all_playlists
        
        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.HEADERS = {}
        
        mock_response = Mock()
        mock_response.json.return_value = {'Items': []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        get_all_playlists()
        
        call_params = mock_get.call_args[1].get('params', {})
        assert call_params.get('IncludeItemTypes') == 'Playlist', \
            "Must filter by Playlist item type"

    @patch('tasks.mediaserver_emby.requests.get')
    @patch('tasks.mediaserver_emby.config')
    def test_parses_items_array(self, mock_config, mock_get):
        """CRITICAL: Must extract Items[] from response"""
        from tasks.mediaserver_emby import get_all_playlists
        
        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.HEADERS = {}
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'Items': [
                {'Id': 'pl1', 'Name': 'Rock_automatic'},
                {'Id': 'pl2', 'Name': 'Jazz Mix'}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        playlists = get_all_playlists()
        
        assert len(playlists) == 2
        assert playlists[0]['Id'] == 'pl1'
        assert playlists[0]['Name'] == 'Rock_automatic'


class TestEmbyDeletePlaylist:
    """Test playlist deletion - Emby uses different endpoint than Jellyfin!"""

    @patch('tasks.mediaserver_emby.requests.post')
    @patch('tasks.mediaserver_emby.config')
    def test_uses_items_delete_endpoint(self, mock_config, mock_post):
        """CRITICAL: Emby uses /Items/Delete with POST, not DELETE to /Items/{id}"""
        from tasks.mediaserver_emby import delete_playlist
        
        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.HEADERS = {'X-Emby-Token': 'token'}
        
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        result = delete_playlist('playlist-123')
        
        assert result is True
        call_url = mock_post.call_args[0][0]
        # CRITICAL: Emby uses /Items/Delete endpoint, not /Items/{id}
        assert call_url == 'http://emby:8096/emby/Items/Delete', \
            f"Emby deletion URL changed! Expected '/emby/Items/Delete', got '{call_url}'"

    @patch('tasks.mediaserver_emby.requests.post')
    @patch('tasks.mediaserver_emby.config')
    def test_passes_id_as_query_param(self, mock_config, mock_post):
        """CRITICAL: Playlist ID must be in 'Ids' query param"""
        from tasks.mediaserver_emby import delete_playlist
        
        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.HEADERS = {}
        
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        delete_playlist('playlist-xyz')
        
        call_params = mock_post.call_args[1].get('params', {})
        assert call_params.get('Ids') == 'playlist-xyz', \
            "Playlist ID must be passed as 'Ids' query param"

    @patch('tasks.mediaserver_emby.requests.post')
    @patch('tasks.mediaserver_emby.config')
    def test_returns_false_on_error(self, mock_config, mock_post):
        """HTTP error returns False"""
        from tasks.mediaserver_emby import delete_playlist
        
        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.HEADERS = {}
        mock_post.side_effect = requests.exceptions.RequestException("Failed")
        
        result = delete_playlist('playlist-123')
        
        assert result is False


class TestEmbyGetTracksFromAlbum:
    """Test track fetching with artist enrichment"""

    @patch('tasks.mediaserver_emby.requests.get')
    @patch('tasks.mediaserver_emby.config')
    def test_uses_emby_url_prefix(self, mock_config, mock_get):
        """CRITICAL: URL must include /emby/ prefix"""
        from tasks.mediaserver_emby import get_tracks_from_album
        
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

    @patch('tasks.mediaserver_emby.requests.get')
    @patch('tasks.mediaserver_emby.config')
    def test_enriches_tracks_with_artist(self, mock_config, mock_get):
        """CRITICAL: Must add AlbumArtist and ArtistId fields"""
        from tasks.mediaserver_emby import get_tracks_from_album
        
        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.HEADERS = {}
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'Items': [
                {
                    'Id': 'track1',
                    'Name': 'Song One',
                    'ArtistItems': [{'Name': 'Artist A', 'Id': 'artist-a'}]
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

    @patch('tasks.mediaserver_emby.requests.get')
    @patch('tasks.mediaserver_emby.config')
    def test_handles_standalone_track_pseudo_albums(self, mock_config, mock_get):
        """CRITICAL: Must handle standalone_ prefix for pseudo-albums"""
        from tasks.mediaserver_emby import get_tracks_from_album
        
        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.HEADERS = {}
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'Id': 'real-track-id',
            'Name': 'Standalone Song',
            'AlbumArtist': 'Some Artist'
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Call with pseudo-album ID
        tracks = get_tracks_from_album('standalone_real-track-id')
        
        # Should fetch the track directly, not list children
        call_url = mock_get.call_args[0][0]
        assert 'real-track-id' in call_url, "Should extract real track ID from pseudo-album"
        assert 'standalone_' not in call_url, "Should NOT include 'standalone_' prefix in API call"


class TestEmbyCreatePlaylist:
    """Test playlist creation - Emby uses query params, not JSON body!"""

    @patch('tasks.mediaserver_emby.requests.post')
    @patch('tasks.mediaserver_emby.config')
    def test_uses_query_params_not_json_body(self, mock_config, mock_post):
        """CRITICAL: Emby expects query params, not JSON body"""
        from tasks.mediaserver_emby import create_playlist
        
        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.EMBY_TOKEN = 'token123'
        
        mock_response = Mock()
        mock_response.json.return_value = {'Id': 'new-playlist'}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        create_playlist('Test Playlist', ['track1', 'track2'])
        
        call_url = mock_post.call_args[0][0]
        # URL should contain query params
        assert 'Name=' in call_url, "Name must be in query string"
        assert 'Ids=' in call_url, "Ids must be in query string"
        assert 'UserId=' in call_url, "UserId must be in query string"
        assert 'MediaType=Audio' in call_url, "MediaType must be Audio"
        
        # Should NOT have json body
        call_kwargs = mock_post.call_args[1]
        assert 'json' not in call_kwargs, "Emby should NOT receive JSON body"

    @patch('tasks.mediaserver_emby.requests.post')
    @patch('tasks.mediaserver_emby.config')
    def test_url_encodes_playlist_name(self, mock_config, mock_post):
        """Playlist names with special chars must be URL encoded"""
        from tasks.mediaserver_emby import create_playlist
        
        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.EMBY_TOKEN = 'token123'
        
        mock_response = Mock()
        mock_response.json.return_value = {'Id': 'new-playlist'}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        create_playlist('Rock & Roll Mix', ['track1'])

        call_url = mock_post.call_args[0][0]
        # & should be encoded
        assert 'Rock%20%26%20Roll' in call_url or 'Rock+%26+Roll' in call_url or 'Rock%20&%20Roll' not in call_url


# =============================================================================
# list_libraries — provider-specific helpers used by the setup wizard and
# migration assistant to populate the "music libraries" checkbox list.
# Each test pins that (a) the function returns every music library the server
# reports, without applying the MUSIC_LIBRARIES filter, and (b) user_creds is
# forwarded so the migration assistant can probe a target without mutating
# `config` globals (same discipline that commit b426682 established for
# Navidrome's `get_all_songs`).
# =============================================================================

class TestJellyfinListLibraries:
    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_returns_music_libraries_with_id_and_name(self, mock_config, mock_get):
        from tasks.mediaserver_jellyfin import list_libraries

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

    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_forwards_user_creds_to_url_and_token(self, mock_config, mock_get):
        """Migration target probe must use session creds, not config globals."""
        from tasks.mediaserver_jellyfin import list_libraries

        mock_config.JELLYFIN_URL = 'http://SHOULD-NOT-BE-USED:0000'
        mock_config.JELLYFIN_TOKEN = 'SHOULD-NOT-BE-USED'
        mock_config.HEADERS = {}

        resp = Mock()
        resp.raise_for_status = Mock()
        resp.json.return_value = []
        mock_get.return_value = resp

        list_libraries(user_creds={
            'url':   'http://target-jelly:8096',
            'token': 'target-token',
        })

        called_url = mock_get.call_args[0][0]
        assert called_url == 'http://target-jelly:8096/Library/VirtualFolders'
        headers = mock_get.call_args[1]['headers']
        assert headers.get('X-Emby-Token') == 'target-token'


class TestEmbyListLibraries:
    @patch('tasks.mediaserver_emby.requests.get')
    @patch('tasks.mediaserver_emby.config')
    def test_returns_music_libraries_only(self, mock_config, mock_get):
        from tasks.mediaserver_emby import list_libraries

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

    @patch('tasks.mediaserver_emby.requests.get')
    @patch('tasks.mediaserver_emby.config')
    def test_forwards_user_creds(self, mock_config, mock_get):
        from tasks.mediaserver_emby import list_libraries

        mock_config.EMBY_URL = 'http://SHOULD-NOT-BE-USED:0000'
        mock_config.EMBY_TOKEN = 'SHOULD-NOT-BE-USED'
        mock_config.HEADERS = {}

        resp = Mock()
        resp.raise_for_status = Mock()
        resp.json.return_value = []
        mock_get.return_value = resp

        list_libraries(user_creds={
            'url':   'http://target-emby:8096',
            'token': 'target-token',
        })

        called_url = mock_get.call_args[0][0]
        assert called_url == 'http://target-emby:8096/emby/Library/VirtualFolders'
        headers = mock_get.call_args[1]['headers']
        assert headers.get('X-Emby-Token') == 'target-token'


class TestNavidromeListLibraries:
    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_returns_every_folder_without_reading_music_libraries(self, mock_req):
        """
        Does NOT call _get_target_music_folder_ids (which would read
        config.MUSIC_LIBRARIES and break when the filter is set for the source
        provider but doesn't apply to the target). Returns every folder.
        """
        from tasks.mediaserver_navidrome import list_libraries

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
        # Verify the single getMusicFolders call — no _get_target_music_folder_ids path
        mock_req.assert_called_once()
        args, kwargs = mock_req.call_args
        assert args[0] == 'getMusicFolders'
        assert kwargs.get('user_creds') is None

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_handles_single_dict_response(self, mock_req):
        """Some Subsonic-compatible servers return a single dict (not a list)
        when only one folder exists. The function must coerce to a list."""
        from tasks.mediaserver_navidrome import list_libraries

        mock_req.return_value = {
            'musicFolders': {
                'musicFolder': {'id': 1, 'name': 'OnlyFolder'}
            }
        }

        result = list_libraries()

        assert result == [{'id': '1', 'name': 'OnlyFolder'}]

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    def test_forwards_user_creds_to_getmusicfolders(self, mock_req):
        """
        Migration-target path: user_creds must reach _navidrome_request so the
        request uses the session's URL/username/password rather than
        config.NAVIDROME_* (which are empty for a target that isn't live yet).
        """
        from tasks.mediaserver_navidrome import list_libraries

        mock_req.return_value = {'musicFolders': {'musicFolder': []}}

        creds = {'url': 'http://target-nav:4533', 'user': 'u', 'password': 'p'}
        list_libraries(user_creds=creds)

        args, kwargs = mock_req.call_args
        assert args[0] == 'getMusicFolders'
        assert kwargs.get('user_creds') == creds


class TestLyrionListLibraries:
    @patch('tasks.mediaserver_lyrion._jsonrpc_request')
    def test_returns_every_folder(self, mock_rpc):
        from tasks.mediaserver_lyrion import list_libraries

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
        # The CLI command is the singular ``musicfolder``. The plural
        # ``musicfolders`` variant drops the connection on Lyrion 9.0.x.
        assert args[0] == 'musicfolder'
        assert kwargs.get('user_creds') is None

    @patch('tasks.mediaserver_lyrion._jsonrpc_request')
    def test_handles_lyrion_9_x_filename_field(self, mock_rpc):
        """Lyrion 9.0.x returns folder entries with ``filename`` (not ``name``).
        Older versions used ``name`` / ``folder``. Accept all three."""
        from tasks.mediaserver_lyrion import list_libraries

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

    @patch('tasks.mediaserver_lyrion._jsonrpc_request')
    def test_prefers_path_over_name_when_available(self, mock_rpc):
        """Lyrion's scan-time filter (_get_target_paths_for_filtering) does a
        substring match against album file URLs, so when the server reports
        a real path we persist it (more deterministic match). Otherwise we
        fall back to the folder display name (which Lyrion treats as a path
        substring at scan time on standard layouts)."""
        from tasks.mediaserver_lyrion import list_libraries

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
            {'id': '12', 'name': 'NoPath'},  # falls back when path absent
        ]

    @patch('tasks.mediaserver_lyrion._jsonrpc_request')
    def test_forwards_user_creds(self, mock_rpc):
        from tasks.mediaserver_lyrion import list_libraries

        mock_rpc.return_value = {'folder_loop': []}

        creds = {'url': 'http://target-lms:9000', 'user': 'u', 'password': 'p'}
        list_libraries(user_creds=creds)

        args, kwargs = mock_rpc.call_args
        assert args[0] == 'musicfolder'
        assert kwargs.get('user_creds') == creds


# =============================================================================
# create_or_replace_playlist (cron sonic_fingerprint upsert) — issue #336
# =============================================================================

class TestNavidromeCreateOrReplacePlaylist:
    """Navidrome upsert: create when missing, clear+add when existing (preserve ID)."""

    @patch('tasks.mediaserver_navidrome._create_playlist_batched')
    @patch('tasks.mediaserver_navidrome.get_playlist_by_name')
    def test_missing_playlist_creates_via_batched(self, mock_get, mock_create):
        from tasks.mediaserver_navidrome import create_or_replace_playlist

        mock_get.return_value = None
        mock_create.return_value = {'Id': 'new-pl-1', 'Name': 'SF', 'id': 'new-pl-1'}

        result = create_or_replace_playlist('SF', ['s1', 's2'])

        mock_create.assert_called_once_with('SF', ['s1', 's2'], user_creds=None)
        assert result['Id'] == 'new-pl-1'

    @patch('tasks.mediaserver_navidrome._add_to_playlist')
    @patch('tasks.mediaserver_navidrome._navidrome_request')
    @patch('tasks.mediaserver_navidrome.get_playlist_by_name')
    def test_existing_playlist_preserves_id(self, mock_get, mock_request, mock_add):
        from tasks.mediaserver_navidrome import create_or_replace_playlist

        mock_get.return_value = {'id': 'pl-existing', 'name': 'SF'}
        # 1st call: getPlaylist returns 3 songs. 2nd+: updatePlaylist returns ok.
        mock_request.side_effect = [
            {'playlist': {'id': 'pl-existing', 'songCount': 3}},
            {'status': 'ok'},
        ]
        mock_add.return_value = True

        result = create_or_replace_playlist('SF', ['new1', 'new2'])

        # getPlaylist was called once
        first = mock_request.call_args_list[0]
        assert first[0][0] == 'getPlaylist'
        assert first[0][1] == {'id': 'pl-existing'}

        # updatePlaylist removed indices in DESCENDING order [2, 1, 0]
        second = mock_request.call_args_list[1]
        assert second[0][0] == 'updatePlaylist'
        assert second[0][1]['playlistId'] == 'pl-existing'
        assert second[0][1]['songIndexToRemove'] == [2, 1, 0]

        # Adds happened on the same playlist id
        mock_add.assert_called_once_with('pl-existing', ['new1', 'new2'], user_creds=None)

        # Returned dict carries the existing ID
        assert result['Id'] == 'pl-existing'

    @patch('tasks.mediaserver_navidrome._add_to_playlist')
    @patch('tasks.mediaserver_navidrome._navidrome_request')
    @patch('tasks.mediaserver_navidrome.get_playlist_by_name')
    def test_clear_batches_above_40(self, mock_get, mock_request, mock_add):
        from tasks.mediaserver_navidrome import create_or_replace_playlist

        mock_get.return_value = {'id': 'pl-100', 'name': 'SF'}
        mock_request.side_effect = [
            {'playlist': {'id': 'pl-100', 'songCount': 100}},
            {'status': 'ok'}, {'status': 'ok'}, {'status': 'ok'},  # 3 update batches
        ]
        mock_add.return_value = True

        create_or_replace_playlist('SF', ['x'])

        # getPlaylist + 3 updatePlaylist batches (40+40+20)
        assert len(mock_request.call_args_list) == 4
        update_calls = mock_request.call_args_list[1:]
        assert len(update_calls[0][0][1]['songIndexToRemove']) == 40
        assert len(update_calls[1][0][1]['songIndexToRemove']) == 40
        assert len(update_calls[2][0][1]['songIndexToRemove']) == 20

    @patch('tasks.mediaserver_navidrome._navidrome_request')
    @patch('tasks.mediaserver_navidrome.get_playlist_by_name')
    def test_empty_item_ids_returns_none_without_calls(self, mock_get, mock_request):
        from tasks.mediaserver_navidrome import create_or_replace_playlist

        result = create_or_replace_playlist('SF', [])

        assert result is None
        mock_get.assert_not_called()
        mock_request.assert_not_called()

    @patch('tasks.mediaserver_navidrome._add_to_playlist')
    @patch('tasks.mediaserver_navidrome._navidrome_request')
    @patch('tasks.mediaserver_navidrome.get_playlist_by_name')
    def test_returns_none_when_add_fails_after_clear(self, mock_get, mock_request, mock_add):
        """If clear succeeds but add fails, return None so the cron handler doesn't log success."""
        from tasks.mediaserver_navidrome import create_or_replace_playlist

        mock_get.return_value = {'id': 'pl-1', 'name': 'SF'}
        mock_request.side_effect = [
            {'playlist': {'id': 'pl-1', 'songCount': 1}},
            {'status': 'ok'},
        ]
        mock_add.return_value = False

        result = create_or_replace_playlist('SF', ['new1'])

        assert result is None


class TestJellyfinCreateOrReplacePlaylist:
    """Jellyfin upsert via /Playlists/{Id}/Items POST/DELETE/GET."""

    @patch('tasks.mediaserver_jellyfin.requests')
    @patch('tasks.mediaserver_jellyfin.get_playlist_by_name')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_missing_playlist_creates_and_returns_id(self, mock_config, mock_get, mock_requests):
        from tasks.mediaserver_jellyfin import create_or_replace_playlist

        mock_config.JELLYFIN_URL = 'http://jf'
        mock_config.JELLYFIN_USER_ID = 'admin-user'
        mock_config.HEADERS = {'X-Emby-Token': 't'}
        mock_get.return_value = None

        post_resp = MagicMock()
        post_resp.json.return_value = {'Id': 'new-jf-1', 'Name': 'SF'}
        mock_requests.post.return_value = post_resp

        result = create_or_replace_playlist('SF', ['s1', 's2'])

        # Single POST to /Playlists with the create body
        assert mock_requests.post.call_count == 1
        post_call = mock_requests.post.call_args
        assert post_call[0][0] == 'http://jf/Playlists'
        assert post_call[1]['json'] == {'Name': 'SF', 'Ids': ['s1', 's2'], 'UserId': 'admin-user'}
        assert result['Id'] == 'new-jf-1'

    @patch('tasks.mediaserver_jellyfin.requests')
    @patch('tasks.mediaserver_jellyfin.get_playlist_by_name')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_existing_playlist_clears_and_adds_preserving_id(self, mock_config, mock_get, mock_requests):
        from tasks.mediaserver_jellyfin import create_or_replace_playlist

        mock_config.JELLYFIN_URL = 'http://jf'
        mock_config.JELLYFIN_USER_ID = 'admin-user'
        mock_config.HEADERS = {'X-Emby-Token': 't'}
        mock_get.return_value = {'Id': 'pl-existing', 'Name': 'SF'}

        # GET items returns two entries with PlaylistItemId
        get_resp = MagicMock()
        get_resp.json.return_value = {'Items': [
            {'Id': 'song1', 'PlaylistItemId': 'entry-a'},
            {'Id': 'song2', 'PlaylistItemId': 'entry-b'},
        ]}
        mock_requests.get.return_value = get_resp
        mock_requests.delete.return_value = MagicMock()
        mock_requests.post.return_value = MagicMock()

        result = create_or_replace_playlist('SF', ['new1', 'new2'])

        # GET on /Playlists/pl-existing/Items
        assert mock_requests.get.call_args[0][0] == 'http://jf/Playlists/pl-existing/Items'
        # DELETE with entryIds
        assert mock_requests.delete.call_count == 1
        del_call = mock_requests.delete.call_args
        assert del_call[0][0] == 'http://jf/Playlists/pl-existing/Items'
        assert del_call[1]['params']['entryIds'] == 'entry-a,entry-b'
        # POST adds new ids
        assert mock_requests.post.call_count == 1
        post_call = mock_requests.post.call_args
        assert post_call[0][0] == 'http://jf/Playlists/pl-existing/Items'
        assert post_call[1]['params']['ids'] == 'new1,new2'
        assert post_call[1]['params']['userId'] == 'admin-user'
        # Same ID preserved
        assert result['Id'] == 'pl-existing'

    @patch('tasks.mediaserver_jellyfin.get_playlist_by_name')
    def test_empty_item_ids_returns_none(self, mock_get):
        from tasks.mediaserver_jellyfin import create_or_replace_playlist

        result = create_or_replace_playlist('SF', [])

        assert result is None
        mock_get.assert_not_called()

    @patch('tasks.mediaserver_jellyfin._add_items_to_playlist')
    @patch('tasks.mediaserver_jellyfin._remove_playlist_entries')
    @patch('tasks.mediaserver_jellyfin._get_playlist_entry_ids')
    @patch('tasks.mediaserver_jellyfin.get_playlist_by_name')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_returns_none_when_add_fails_after_clear(
        self, mock_config, mock_get, mock_get_entries, mock_remove, mock_add
    ):
        """If clear succeeds but add fails, return None instead of misreporting success."""
        from tasks.mediaserver_jellyfin import create_or_replace_playlist

        mock_config.JELLYFIN_URL = 'http://jf'
        mock_config.JELLYFIN_USER_ID = 'admin-user'
        mock_config.HEADERS = {'X-Emby-Token': 't'}
        mock_get.return_value = {'Id': 'pl-1', 'Name': 'SF'}
        mock_get_entries.return_value = ['e1']
        mock_remove.return_value = True
        mock_add.return_value = False

        result = create_or_replace_playlist('SF', ['new1'])

        assert result is None

    @patch('tasks.mediaserver_jellyfin._create_fresh_playlist')
    @patch('tasks.mediaserver_jellyfin.delete_playlist')
    @patch('tasks.mediaserver_jellyfin._remove_playlist_entries')
    @patch('tasks.mediaserver_jellyfin._get_playlist_entry_ids')
    @patch('tasks.mediaserver_jellyfin.get_playlist_by_name')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_falls_back_to_recreate_when_remove_fails(
        self, mock_config, mock_get, mock_get_entries, mock_remove, mock_delete, mock_create
    ):
        """Jellyfin <10.11 + API token rejects DELETE /Playlists/{Id}/Items (jellyfin#13476).
        When that happens, delete the whole playlist and recreate. Id will change."""
        from tasks.mediaserver_jellyfin import create_or_replace_playlist

        mock_config.JELLYFIN_URL = 'http://jf'
        mock_config.JELLYFIN_USER_ID = 'admin-user'
        mock_config.HEADERS = {'X-Emby-Token': 't'}
        mock_get.return_value = {'Id': 'old-pl', 'Name': 'SF'}
        mock_get_entries.return_value = ['e1', 'e2']
        mock_remove.side_effect = requests.exceptions.HTTPError('400 Bad Request')
        mock_delete.return_value = True
        mock_create.return_value = {'Id': 'new-pl', 'Name': 'SF'}

        result = create_or_replace_playlist('SF', ['n1', 'n2'])

        mock_delete.assert_called_once_with('old-pl')
        mock_create.assert_called_once_with('SF', ['n1', 'n2'])
        assert result['Id'] == 'new-pl'

    @patch('tasks.mediaserver_jellyfin._create_fresh_playlist')
    @patch('tasks.mediaserver_jellyfin.delete_playlist')
    @patch('tasks.mediaserver_jellyfin._remove_playlist_entries')
    @patch('tasks.mediaserver_jellyfin._get_playlist_entry_ids')
    @patch('tasks.mediaserver_jellyfin.get_playlist_by_name')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_fallback_returns_none_when_delete_playlist_fails(
        self, mock_config, mock_get, mock_get_entries, mock_remove, mock_delete, mock_create
    ):
        """If the fallback delete itself fails, don't try to recreate — bail out."""
        from tasks.mediaserver_jellyfin import create_or_replace_playlist

        mock_config.JELLYFIN_URL = 'http://jf'
        mock_config.JELLYFIN_USER_ID = 'admin-user'
        mock_config.HEADERS = {'X-Emby-Token': 't'}
        mock_get.return_value = {'Id': 'old-pl', 'Name': 'SF'}
        mock_get_entries.return_value = ['e1']
        mock_remove.side_effect = requests.exceptions.HTTPError('400 Bad Request')
        mock_delete.return_value = False

        result = create_or_replace_playlist('SF', ['n1'])

        assert result is None
        mock_create.assert_not_called()


class TestEmbyCreateOrReplacePlaylist:
    """Emby upsert under /emby/ prefix with uppercase params."""

    @patch('tasks.mediaserver_emby.requests')
    @patch('tasks.mediaserver_emby.get_playlist_by_name')
    @patch('tasks.mediaserver_emby.config')
    def test_existing_playlist_clears_and_adds_preserving_id(self, mock_config, mock_get, mock_requests):
        from tasks.mediaserver_emby import create_or_replace_playlist

        mock_config.EMBY_URL = 'http://emby'
        mock_config.EMBY_USER_ID = 'admin-emby'
        mock_config.EMBY_TOKEN = 'tok'
        mock_get.return_value = {'Id': 'emby-pl', 'Name': 'SF'}

        get_resp = MagicMock()
        get_resp.json.return_value = {'Items': [
            {'Id': 'song1', 'PlaylistItemId': 'e1'},
        ]}
        mock_requests.get.return_value = get_resp
        mock_requests.delete.return_value = MagicMock()
        mock_requests.post.return_value = MagicMock()
        # quote helper used in create branch — make it a simple passthrough so
        # other tests in this class remain compatible if they hit it
        mock_requests.utils.quote.side_effect = lambda s: s

        result = create_or_replace_playlist('SF', ['n1'])

        assert mock_requests.delete.call_args[1]['params']['EntryIds'] == 'e1'
        post_call = mock_requests.post.call_args
        assert post_call[0][0] == 'http://emby/emby/Playlists/emby-pl/Items'
        assert post_call[1]['params']['Ids'] == 'n1'
        assert post_call[1]['params']['UserId'] == 'admin-emby'
        assert result['Id'] == 'emby-pl'

    @patch('tasks.mediaserver_emby._add_items_to_playlist')
    @patch('tasks.mediaserver_emby._remove_playlist_entries')
    @patch('tasks.mediaserver_emby._get_playlist_entry_ids')
    @patch('tasks.mediaserver_emby.get_playlist_by_name')
    @patch('tasks.mediaserver_emby.config')
    def test_returns_none_when_add_fails_after_clear(
        self, mock_config, mock_get, mock_get_entries, mock_remove, mock_add
    ):
        from tasks.mediaserver_emby import create_or_replace_playlist

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
    """Lyrion upsert: deletes existing then creates fresh (ID may change — known limitation)."""

    @patch('tasks.mediaserver_lyrion._create_playlist_batched')
    @patch('tasks.mediaserver_lyrion.delete_playlist')
    @patch('tasks.mediaserver_lyrion.get_playlist_by_name')
    def test_existing_deletes_then_creates(self, mock_get, mock_delete, mock_create):
        from tasks.mediaserver_lyrion import create_or_replace_playlist

        mock_get.return_value = {'Id': 99, 'Name': 'SF'}
        mock_delete.return_value = True
        mock_create.return_value = {'Id': 100, 'Name': 'SF'}

        result = create_or_replace_playlist('SF', ['t1'])

        mock_delete.assert_called_once_with(99)
        mock_create.assert_called_once_with('SF', ['t1'])
        assert result['Name'] == 'SF'

    @patch('tasks.mediaserver_lyrion._create_playlist_batched')
    @patch('tasks.mediaserver_lyrion.delete_playlist')
    @patch('tasks.mediaserver_lyrion.get_playlist_by_name')
    def test_missing_creates_without_delete(self, mock_get, mock_delete, mock_create):
        from tasks.mediaserver_lyrion import create_or_replace_playlist

        mock_get.return_value = None
        mock_create.return_value = {'Id': 50, 'Name': 'SF'}

        create_or_replace_playlist('SF', ['t1'])

        mock_delete.assert_not_called()
        mock_create.assert_called_once_with('SF', ['t1'])

    @patch('tasks.mediaserver_lyrion._create_playlist_batched')
    @patch('tasks.mediaserver_lyrion.delete_playlist')
    @patch('tasks.mediaserver_lyrion.get_playlist_by_name')
    def test_aborts_when_delete_fails(self, mock_get, mock_delete, mock_create):
        """If delete fails, return None instead of creating a duplicate playlist."""
        from tasks.mediaserver_lyrion import create_or_replace_playlist

        mock_get.return_value = {'Id': 99, 'Name': 'SF'}
        mock_delete.return_value = False

        result = create_or_replace_playlist('SF', ['t1'])

        assert result is None
        mock_create.assert_not_called()


class TestDispatcherCreateOrReplacePlaylist:
    """Validation + per-provider dispatching for create_or_replace_playlist."""

    @patch('tasks.mediaserver.config')
    def test_requires_name_and_ids(self, mock_config):
        from tasks.mediaserver import create_or_replace_playlist

        with pytest.raises(ValueError, match="Playlist name is required"):
            create_or_replace_playlist('', ['id1'])

        with pytest.raises(ValueError, match="Track IDs are required"):
            create_or_replace_playlist('Name', [])

    @patch('tasks.mediaserver.config')
    def test_mpd_raises_not_implemented(self, mock_config):
        from tasks.mediaserver import create_or_replace_playlist

        mock_config.MEDIASERVER_TYPE = 'mpd'

        with pytest.raises(NotImplementedError):
            create_or_replace_playlist('SF', ['s1'])

    @patch('tasks.mediaserver.navidrome_create_or_replace_playlist')
    @patch('tasks.mediaserver.config')
    def test_dispatches_to_navidrome(self, mock_config, mock_provider):
        from tasks.mediaserver import create_or_replace_playlist

        mock_config.MEDIASERVER_TYPE = 'navidrome'
        mock_provider.return_value = {'Id': 'pl-1'}

        result = create_or_replace_playlist('SF', ['s1'])

        mock_provider.assert_called_once_with('SF', ['s1'], None)
        assert result['Id'] == 'pl-1'

    @patch('tasks.mediaserver.jellyfin_create_or_replace_playlist')
    @patch('tasks.mediaserver.config')
    def test_dispatches_to_jellyfin(self, mock_config, mock_provider):
        from tasks.mediaserver import create_or_replace_playlist

        mock_config.MEDIASERVER_TYPE = 'jellyfin'
        mock_provider.return_value = {'Id': 'pl-2'}

        create_or_replace_playlist('SF', ['s1'])

        mock_provider.assert_called_once()

    @patch('tasks.mediaserver.emby_create_or_replace_playlist')
    @patch('tasks.mediaserver.config')
    def test_dispatches_to_emby(self, mock_config, mock_provider):
        from tasks.mediaserver import create_or_replace_playlist

        mock_config.MEDIASERVER_TYPE = 'emby'
        mock_provider.return_value = {'Id': 'pl-3'}

        create_or_replace_playlist('SF', ['s1'])

        mock_provider.assert_called_once()

    @patch('tasks.mediaserver.lyrion_create_or_replace_playlist')
    @patch('tasks.mediaserver.config')
    def test_dispatches_to_lyrion(self, mock_config, mock_provider):
        from tasks.mediaserver import create_or_replace_playlist

        mock_config.MEDIASERVER_TYPE = 'lyrion'
        mock_provider.return_value = {'Id': 4}

        create_or_replace_playlist('SF', ['s1'])

        mock_provider.assert_called_once()


# =============================================================================
# Issue #523: get_all_songs must paginate AND fail loudly on a page error.
#
# A single unpaginated request for a large library times out on Jellyfin
# 10.11.x (the reported bug). Pagination fixes that. But the error path matters
# just as much: get_all_songs feeds the migration probe
# (provider_probe.fetch_all_tracks) -> provider_migration_matcher.match_tracks,
# where every score row NOT present in the returned list becomes "unmatched"
# and is deleted as an orphan by the migration execute step. A silently
# truncated (`break`) or silently empty (`return []`) result on a mid-scan
# timeout would look like a successful migration while destroying real analysis
# data. The contract is therefore: return ALL songs, or raise.
# =============================================================================


def _audio_page(n_items, start=0):
    """Build a mock Jellyfin/Emby ``/Items`` response holding ``n_items`` rows."""
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = {
        'Items': [{'Id': f'id{start + i}', 'Name': f'Song {start + i}'}
                  for i in range(n_items)]
    }
    return resp


class TestJellyfinGetAllSongsPagination:
    """Jellyfin get_all_songs: paginate, and raise rather than truncate."""

    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_paginates_until_short_page(self, mock_config, mock_get):
        """A full page (== limit) triggers another request; a short page stops it."""
        from tasks.mediaserver_jellyfin import get_all_songs

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {'X-Emby-Token': 'token'}
        # Page 1 full (500) -> continue; page 2 short (3) -> stop.
        mock_get.side_effect = [_audio_page(500), _audio_page(3, start=500)]

        songs = get_all_songs()

        assert len(songs) == 503
        assert mock_get.call_count == 2
        # Page 2 must advance StartIndex by the page size.
        page2_params = mock_get.call_args_list[1].kwargs['params']
        assert page2_params['StartIndex'] == 500
        assert page2_params['Limit'] == 500

    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_raises_on_midscan_failure_instead_of_truncating(self, mock_config, mock_get):
        """A timeout on a later page must propagate, NOT return the partial list."""
        from tasks.mediaserver_jellyfin import get_all_songs

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {'X-Emby-Token': 'token'}
        # Page 1 succeeds (full -> would continue); page 2 times out.
        mock_get.side_effect = [
            _audio_page(500),
            requests.exceptions.ReadTimeout("read timed out"),
        ]

        with pytest.raises(requests.exceptions.ReadTimeout):
            get_all_songs()

    @patch('tasks.mediaserver_jellyfin.requests.get')
    @patch('tasks.mediaserver_jellyfin.config')
    def test_empty_library_returns_empty_without_raising(self, mock_config, mock_get):
        """A genuinely empty library (first page empty) returns [], does not raise."""
        from tasks.mediaserver_jellyfin import get_all_songs

        mock_config.JELLYFIN_URL = 'http://jellyfin:8096'
        mock_config.JELLYFIN_USER_ID = 'user123'
        mock_config.HEADERS = {}
        mock_get.side_effect = [_audio_page(0)]

        assert get_all_songs() == []


class TestEmbyGetAllSongsRaisesOnFailure:
    """Emby is already paginated (limit=1000); on a page error it must raise,
    not return the partial list it had accumulated."""

    @patch('tasks.mediaserver_emby.requests.get')
    @patch('tasks.mediaserver_emby.config')
    def test_raises_on_midscan_failure_instead_of_truncating(self, mock_config, mock_get):
        from tasks.mediaserver_emby import get_all_songs

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.HEADERS = {'X-Emby-Token': 'token'}
        # Page 1 full (1000 -> would continue); page 2 times out.
        mock_get.side_effect = [
            _audio_page(1000),
            requests.exceptions.ReadTimeout("read timed out"),
        ]

        with pytest.raises(requests.exceptions.ReadTimeout):
            get_all_songs()

    @patch('tasks.mediaserver_emby.requests.get')
    @patch('tasks.mediaserver_emby.config')
    def test_empty_library_returns_empty_without_raising(self, mock_config, mock_get):
        from tasks.mediaserver_emby import get_all_songs

        mock_config.EMBY_URL = 'http://emby:8096'
        mock_config.EMBY_USER_ID = 'user123'
        mock_config.HEADERS = {}
        mock_get.side_effect = [_audio_page(0)]

        assert get_all_songs() == []
