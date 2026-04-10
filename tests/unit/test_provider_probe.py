"""Unit tests for tasks.provider_probe.

Mocks `requests.get` / `requests.post` and verifies the probe module produces
the unified track/album shape regardless of which provider it talks to.

Uses the _import_module bypass so `tasks/__init__.py` (librosa) isn't touched.
"""
import os
import sys
import importlib.util
import pytest
from unittest.mock import MagicMock, patch


def _load_probe():
    mod_name = 'tasks.provider_probe'
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    )
    mod_path = os.path.join(repo_root, 'tasks', 'provider_probe.py')
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def probe():
    return _load_probe()


def _mock_response(json_data=None, status_code=200, ok=True):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = ok
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Jellyfin
# ---------------------------------------------------------------------------

class TestJellyfinProbe:
    CREDS = {'url': 'http://jellyfin.local:8096', 'user_id': 'uid', 'token': 'tok'}

    def test_fetch_all_tracks_returns_unified_shape(self, probe):
        api_response = {
            'Items': [
                {
                    'Id': 'j1', 'Name': 'Song One', 'Album': 'Album A',
                    'AlbumArtist': 'Artist A', 'Path': '/media/music/A/song1.flac',
                    'ProductionYear': 2020, 'IndexNumber': 1, 'ParentIndexNumber': 1,
                },
                {
                    'Id': 'j2', 'Name': 'Song Two', 'Album': 'Album A',
                    'AlbumArtist': 'Artist A', 'Path': '/media/music/A/song2.flac',
                    'ProductionYear': 2020, 'IndexNumber': 2, 'ParentIndexNumber': 1,
                },
            ]
        }
        with patch.object(probe.requests, 'get', return_value=_mock_response(api_response)) as mock_get:
            tracks = probe.fetch_all_tracks('jellyfin', self.CREDS)

        assert len(tracks) == 2
        t = tracks[0]
        # Unified shape — must have these keys
        for key in ('id', 'path', 'title', 'artist', 'album', 'album_artist', 'year'):
            assert key in t, f"missing key {key}"
        assert t['id'] == 'j1'
        assert t['title'] == 'Song One'
        assert t['path'] == '/media/music/A/song1.flac'
        assert t['album'] == 'Album A'
        assert t['album_artist'] == 'Artist A'
        assert t['year'] == 2020

        # Verify the request was sent correctly
        call_args = mock_get.call_args
        assert call_args[0][0] == 'http://jellyfin.local:8096/Users/uid/Items'
        # Token must be sent in header
        headers = call_args[1]['headers']
        assert headers['X-Emby-Token'] == 'tok'

    def test_test_connection_detects_absolute_paths(self, probe):
        api_response = {
            'Items': [
                {'Id': f'j{i}', 'Name': f'S{i}', 'Path': f'/media/music/a/s{i}.flac'}
                for i in range(5)
            ]
        }
        with patch.object(probe.requests, 'get', return_value=_mock_response(api_response)):
            result = probe.test_connection('jellyfin', self.CREDS)

        assert result['ok'] is True
        assert result['sample_count'] == 5
        assert result['path_format'] == 'absolute'

    def test_test_connection_error_returns_not_ok(self, probe):
        with patch.object(probe.requests, 'get', side_effect=Exception('boom')):
            result = probe.test_connection('jellyfin', self.CREDS)

        assert result['ok'] is False
        assert 'boom' in result['error']

    def test_search_albums(self, probe):
        api_response = {
            'Items': [
                {
                    'Id': 'album1', 'Name': 'Abbey Road',
                    'AlbumArtist': 'The Beatles', 'ProductionYear': 1969,
                    'ChildCount': 17,
                }
            ]
        }
        with patch.object(probe.requests, 'get', return_value=_mock_response(api_response)) as mock_get:
            albums = probe.search_albums('jellyfin', self.CREDS, 'Abbey Road')

        assert len(albums) == 1
        a = albums[0]
        assert a['id'] == 'album1'
        assert a['name'] == 'Abbey Road'
        assert a['artist'] == 'The Beatles'
        assert a['year'] == 1969
        assert a['track_count'] == 17

        # Should have used SearchTerm param
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs['params']['IncludeItemTypes'] == 'MusicAlbum'
        assert call_kwargs['params']['SearchTerm'] == 'Abbey Road'


# ---------------------------------------------------------------------------
# Navidrome
# ---------------------------------------------------------------------------

class TestNavidromeProbe:
    CREDS = {'url': 'http://nav.local:4533', 'user': 'admin', 'password': 'hunter2'}

    def _subsonic_wrap(self, inner):
        return {'subsonic-response': {'status': 'ok', **inner}}

    def test_fetch_all_tracks_returns_unified_shape(self, probe):
        # Simulate single page of results
        songs_page = self._subsonic_wrap({
            'searchResult3': {
                'song': [
                    {'id': 'n1', 'title': 'Song One', 'artist': 'Artist A',
                     'album': 'Album A', 'path': '/music/A/song1.flac', 'year': 2020},
                    {'id': 'n2', 'title': 'Song Two', 'artist': 'Artist A',
                     'album': 'Album A', 'path': '/music/A/song2.flac', 'year': 2020},
                ]
            }
        })
        # Second call returns empty to end pagination
        empty_page = self._subsonic_wrap({'searchResult3': {}})

        with patch.object(probe.requests, 'get', side_effect=[
            _mock_response(songs_page),
            _mock_response(empty_page),
        ]) as mock_get:
            tracks = probe.fetch_all_tracks('navidrome', self.CREDS)

        assert len(tracks) == 2
        t = tracks[0]
        assert t['id'] == 'n1'
        assert t['title'] == 'Song One'
        assert t['path'] == '/music/A/song1.flac'
        assert t['album'] == 'Album A'
        # Navidrome's `artist` should map to both `artist` and `album_artist`
        assert t['artist'] == 'Artist A'
        assert t['year'] == 2020

        # Verify auth params were sent (hex-encoded password)
        first_call = mock_get.call_args_list[0]
        params = first_call[1]['params']
        assert params['u'] == 'admin'
        assert params['p'] == 'enc:' + 'hunter2'.encode('utf-8').hex()
        assert params['c'] == 'AudioMuse-AI'
        assert params['f'] == 'json'

    def test_test_connection_detects_absolute(self, probe):
        api_response = self._subsonic_wrap({
            'searchResult3': {
                'song': [{'id': f'n{i}', 'title': f'S{i}', 'path': f'/music/a/s{i}.flac'}
                         for i in range(5)]
            }
        })
        empty = self._subsonic_wrap({'searchResult3': {}})
        with patch.object(probe.requests, 'get', side_effect=[_mock_response(api_response), _mock_response(empty)]):
            result = probe.test_connection('navidrome', self.CREDS)

        assert result['ok'] is True
        assert result['path_format'] == 'absolute'

    def test_test_connection_detects_relative_when_no_leading_slash(self, probe):
        api_response = self._subsonic_wrap({
            'searchResult3': {
                'song': [{'id': f'n{i}', 'title': f'S{i}', 'path': f'Artist/Album/s{i}.flac'}
                         for i in range(5)]
            }
        })
        empty = self._subsonic_wrap({'searchResult3': {}})
        with patch.object(probe.requests, 'get', side_effect=[_mock_response(api_response), _mock_response(empty)]):
            result = probe.test_connection('navidrome', self.CREDS)

        assert result['ok'] is True
        assert result['path_format'] == 'relative'
        # Should emit a warning for Navidrome + non-absolute
        assert any('Report Real Path' in w or 'RealPath' in w for w in result['warnings'])

    def test_test_connection_detects_none_when_all_paths_missing(self, probe):
        api_response = self._subsonic_wrap({
            'searchResult3': {
                'song': [{'id': f'n{i}', 'title': f'S{i}'} for i in range(5)]
            }
        })
        empty = self._subsonic_wrap({'searchResult3': {}})
        with patch.object(probe.requests, 'get', side_effect=[_mock_response(api_response), _mock_response(empty)]):
            result = probe.test_connection('navidrome', self.CREDS)

        assert result['ok'] is True
        assert result['path_format'] == 'none'
        assert any('Report Real Path' in w or 'RealPath' in w for w in result['warnings'])

    def test_test_connection_detects_mixed(self, probe):
        songs = (
            [{'id': f'a{i}', 'title': f'S{i}', 'path': f'/music/a/s{i}.flac'} for i in range(3)]
            + [{'id': f'b{i}', 'title': f'S{i}', 'path': f'Artist/s{i}.flac'} for i in range(3)]
        )
        api_response = self._subsonic_wrap({'searchResult3': {'song': songs}})
        empty = self._subsonic_wrap({'searchResult3': {}})
        with patch.object(probe.requests, 'get', side_effect=[_mock_response(api_response), _mock_response(empty)]):
            result = probe.test_connection('navidrome', self.CREDS)

        assert result['ok'] is True
        assert result['path_format'] == 'mixed'

    def test_search_albums(self, probe):
        api_response = self._subsonic_wrap({
            'searchResult3': {
                'album': [
                    {'id': 'alb1', 'name': 'Abbey Road', 'artist': 'The Beatles',
                     'year': 1969, 'songCount': 17}
                ]
            }
        })
        with patch.object(probe.requests, 'get', return_value=_mock_response(api_response)) as mock_get:
            albums = probe.search_albums('navidrome', self.CREDS, 'Abbey Road')

        assert len(albums) == 1
        a = albums[0]
        assert a['id'] == 'alb1'
        assert a['name'] == 'Abbey Road'
        assert a['artist'] == 'The Beatles'
        assert a['year'] == 1969
        assert a['track_count'] == 17
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs['params']['query'] == 'Abbey Road'

    def test_get_album_tracks(self, probe):
        api_response = self._subsonic_wrap({
            'album': {
                'id': 'alb1',
                'name': 'Abbey Road',
                'song': [
                    {'id': 's1', 'title': 'Come Together', 'artist': 'Beatles',
                     'album': 'Abbey Road', 'path': '/music/beatles/ar/come_together.flac',
                     'year': 1969, 'track': 1},
                ]
            }
        })
        with patch.object(probe.requests, 'get', return_value=_mock_response(api_response)):
            tracks = probe.get_album_tracks('navidrome', self.CREDS, 'alb1')

        assert len(tracks) == 1
        assert tracks[0]['id'] == 's1'
        assert tracks[0]['title'] == 'Come Together'


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class TestDispatcher:
    def test_unsupported_provider_raises(self, probe):
        with pytest.raises(Exception) as exc:
            probe.test_connection('unknown_provider', {})
        assert 'unknown_provider' in str(exc.value).lower() or 'not supported' in str(exc.value).lower()

    def test_emby_uses_jellyfin_logic(self, probe):
        # Emby has the same shape as Jellyfin; probe should treat it the same
        api_response = {
            'Items': [
                {'Id': 'e1', 'Name': 'Song', 'Path': '/music/a.flac',
                 'Album': 'Album', 'AlbumArtist': 'Artist', 'ProductionYear': 2020}
            ]
        }
        creds = {'url': 'http://emby.local:8096', 'user_id': 'u', 'token': 't'}
        with patch.object(probe.requests, 'get', return_value=_mock_response(api_response)) as mock_get:
            tracks = probe.fetch_all_tracks('emby', creds)
        assert len(tracks) == 1
        assert tracks[0]['id'] == 'e1'

    def test_lyrion_is_stub_or_implemented(self, probe):
        # Lyrion may be implemented or may cleanly raise "not yet supported"
        # Either way, calling it should not crash with a generic traceback
        try:
            result = probe.test_connection('lyrion', {'url': 'http://lms.local:9000'})
            # If implemented, it should return the standard result dict shape
            assert 'ok' in result
        except NotImplementedError as e:
            assert 'lyrion' in str(e).lower()

    def test_mpd_is_stub_or_implemented(self, probe):
        try:
            result = probe.test_connection('mpd', {'host': 'localhost', 'port': 6600})
            assert 'ok' in result
        except NotImplementedError as e:
            assert 'mpd' in str(e).lower()
