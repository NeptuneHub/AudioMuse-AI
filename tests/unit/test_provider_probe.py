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
        with patch.object(probe.mediaserver_jellyfin.requests, 'get', return_value=_mock_response(api_response)) as mock_get:
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
        # Fields must request the artist-resolution fields so the probe can
        # apply the same ArtistItems → Artists → AlbumArtist hierarchy that
        # mediaserver_jellyfin uses.
        fields = call_args[1]['params']['Fields']
        assert 'ArtistItems' in fields
        assert 'Artists' in fields
        assert 'AlbumArtist' in fields

    def test_artist_hierarchy_prefers_artist_items_first(self, probe):
        # Mirrors mediaserver_jellyfin._select_best_artist:
        # ArtistItems[0].Name > Artists[0] > AlbumArtist > "Unknown Artist".
        api_response = {
            'Items': [{
                'Id': 'j1', 'Name': 'Compilation Track', 'Album': 'Various Hits',
                'AlbumArtist': 'Various Artists',
                'Artists': ['Backup Artist'],
                'ArtistItems': [{'Id': 'a1', 'Name': 'Real Performer'}],
                'Path': '/music/compilation/01.flac',
            }]
        }
        with patch.object(probe.mediaserver_jellyfin.requests, 'get', return_value=_mock_response(api_response)):
            tracks = probe.fetch_all_tracks('jellyfin', self.CREDS)
        t = tracks[0]
        # Track-level = real performer; album-level preserved separately
        assert t['artist'] == 'Real Performer'
        assert t['album_artist'] == 'Various Artists'

    def test_artist_hierarchy_falls_back_to_artists_array(self, probe):
        api_response = {
            'Items': [{
                'Id': 'j1', 'Name': 'x', 'Album': 'y',
                'AlbumArtist': 'Various Artists',
                'Artists': ['Real Performer'],
                'Path': '/music/x.flac',
            }]
        }
        with patch.object(probe.mediaserver_jellyfin.requests, 'get', return_value=_mock_response(api_response)):
            tracks = probe.fetch_all_tracks('jellyfin', self.CREDS)
        assert tracks[0]['artist'] == 'Real Performer'
        assert tracks[0]['album_artist'] == 'Various Artists'

    def test_artist_hierarchy_falls_back_to_album_artist(self, probe):
        api_response = {
            'Items': [{
                'Id': 'j1', 'Name': 'x', 'Album': 'y',
                'AlbumArtist': 'Album Only Artist',
                'Path': '/music/x.flac',
            }]
        }
        with patch.object(probe.mediaserver_jellyfin.requests, 'get', return_value=_mock_response(api_response)):
            tracks = probe.fetch_all_tracks('jellyfin', self.CREDS)
        assert tracks[0]['artist'] == 'Album Only Artist'
        assert tracks[0]['album_artist'] == 'Album Only Artist'

    def test_artist_hierarchy_ultimate_fallback_unknown(self, probe):
        api_response = {
            'Items': [{'Id': 'j1', 'Name': 'x', 'Album': 'y', 'Path': '/music/x.flac'}]
        }
        with patch.object(probe.mediaserver_jellyfin.requests, 'get', return_value=_mock_response(api_response)):
            tracks = probe.fetch_all_tracks('jellyfin', self.CREDS)
        assert tracks[0]['artist'] == 'Unknown Artist'
        assert tracks[0]['album_artist'] is None

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

        with patch.object(probe.mediaserver_navidrome.requests, 'request', side_effect=[
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

    def test_artist_hierarchy_prefers_artist_over_album_artist(self, probe):
        # Compilation track where albumArtist='Various Artists' but
        # track-level artist is the real performer.
        songs_page = self._subsonic_wrap({
            'searchResult3': {
                'song': [{
                    'id': 'n1', 'title': 'Hotel California',
                    'artist': 'Eagles',
                    'albumArtist': 'Various Artists',
                    'album': 'Rock Hits', 'path': '/music/va/01.flac',
                }]
            }
        })
        empty_page = self._subsonic_wrap({'searchResult3': {}})
        with patch.object(probe.requests, 'get', side_effect=[
            _mock_response(songs_page), _mock_response(empty_page),
        ]):
            tracks = probe.fetch_all_tracks('navidrome', self.CREDS)
        assert tracks[0]['artist'] == 'Eagles'
        assert tracks[0]['album_artist'] == 'Various Artists'

    def test_artist_hierarchy_falls_back_to_unknown(self, probe):
        songs_page = self._subsonic_wrap({
            'searchResult3': {
                'song': [{'id': 'n1', 'title': 'x', 'album': 'y', 'path': '/a.flac'}]
            }
        })
        empty_page = self._subsonic_wrap({'searchResult3': {}})
        with patch.object(probe.requests, 'get', side_effect=[
            _mock_response(songs_page), _mock_response(empty_page),
        ]):
            tracks = probe.fetch_all_tracks('navidrome', self.CREDS)
        assert tracks[0]['artist'] == 'Unknown Artist'
        # album_artist stays None (not overwritten with 'Unknown Artist')
        assert tracks[0]['album_artist'] is None


# ---------------------------------------------------------------------------
# Lyrion (LMS JSON-RPC)
# ---------------------------------------------------------------------------

def _lyrion_wrap(result):
    """LMS JSON-RPC response envelope: ``{'id':1,'method':..,'params':[..],'result':result}``."""
    return {'id': 1, 'method': 'slim.request', 'params': ['', []], 'result': result}


class TestLyrionProbe:
    CREDS = {'url': 'http://lms.local:9000'}

    def test_decode_url_file_scheme(self, probe):
        assert probe._lyrion_decode_url('file:///music/artist/album/01.flac') == '/music/artist/album/01.flac'

    def test_decode_url_url_encoded(self, probe):
        assert probe._lyrion_decode_url('file:///music/My%20Artist/My%20Album/01%20Intro.flac') \
            == '/music/My Artist/My Album/01 Intro.flac'

    def test_decode_url_none(self, probe):
        assert probe._lyrion_decode_url(None) is None
        assert probe._lyrion_decode_url('') is None

    def test_decode_url_passthrough_for_non_file_scheme(self, probe):
        # Non-file URIs come back URL-decoded but otherwise untouched.
        assert probe._lyrion_decode_url('spotify:track:abc') == 'spotify:track:abc'

    def test_is_spotify_url(self, probe):
        assert probe._lyrion_is_spotify({'url': 'spotify:track:xyz'}) is True
        assert probe._lyrion_is_spotify({'url': 'https://api.spotify.com/v1/tracks/xyz'}) is True

    def test_is_spotify_genre(self, probe):
        assert probe._lyrion_is_spotify({'url': 'file:///a.flac', 'genre': 'Spotify'}) is True

    def test_is_spotify_negative(self, probe):
        assert probe._lyrion_is_spotify({'url': 'file:///a.flac', 'genre': 'Rock'}) is False
        assert probe._lyrion_is_spotify({}) is False

    def test_track_shape_full(self, probe):
        raw = {
            'id': 42, 'title': 'Song', 'artist': 'Track Artist',
            'albumartist': 'Album Artist', 'album': 'Album Name',
            'url': 'file:///music/a/b/01.flac', 'year': 2021,
            'tracknum': 1, 'disc': 1,
        }
        t = probe._lyrion_track(raw)
        assert t['id'] == '42'
        assert t['title'] == 'Song'
        assert t['artist'] == 'Track Artist'
        assert t['album_artist'] == 'Album Artist'
        assert t['album'] == 'Album Name'
        assert t['path'] == '/music/a/b/01.flac'
        assert t['year'] == 2021
        assert t['track_number'] == 1

    def test_track_artist_fallback_to_albumartist(self, probe):
        raw = {'id': 1, 'title': 'x', 'albumartist': 'AA', 'album': 'x', 'url': 'file:///a.flac'}
        t = probe._lyrion_track(raw)
        assert t['artist'] == 'AA'
        assert t['album_artist'] == 'AA'

    def test_track_artist_prefers_trackartist_over_everything(self, probe):
        # Hierarchy: trackartist > contributor > artist > albumartist > band
        raw = {
            'id': 1, 'title': 'x', 'album': 'y',
            'trackartist': 'Track Role',
            'contributor': 'Contributor Role',
            'artist': 'Artist Role',
            'albumartist': 'Album Role',
            'band': 'Band Role',
            'url': 'file:///a.flac',
        }
        t = probe._lyrion_track(raw)
        assert t['artist'] == 'Track Role'
        # album_artist is the album-level value, NOT the chosen track artist
        assert t['album_artist'] == 'Album Role'

    def test_track_artist_falls_back_to_contributor(self, probe):
        raw = {
            'id': 1, 'title': 'x', 'album': 'y',
            'contributor': 'Contributor Role',
            'artist': 'Artist Role',
            'albumartist': 'Album Role',
            'url': 'file:///a.flac',
        }
        t = probe._lyrion_track(raw)
        assert t['artist'] == 'Contributor Role'

    def test_track_artist_falls_back_to_band(self, probe):
        # When only `band` is populated — e.g. classical orchestra recordings.
        raw = {
            'id': 1, 'title': 'x', 'album': 'y',
            'band': 'Berlin Philharmonic',
            'url': 'file:///a.flac',
        }
        t = probe._lyrion_track(raw)
        assert t['artist'] == 'Berlin Philharmonic'
        assert t['album_artist'] is None

    def test_track_artist_unknown_when_all_missing(self, probe):
        raw = {'id': 1, 'title': 'x', 'album': 'y', 'url': 'file:///a.flac'}
        t = probe._lyrion_track(raw)
        assert t['artist'] == 'Unknown Artist'
        assert t['album_artist'] is None

    def test_track_year_coerced_from_string(self, probe):
        t = probe._lyrion_track({'id': 1, 'title': 'x', 'year': '2015', 'url': None})
        assert t['year'] == 2015

    def test_track_year_invalid_becomes_none(self, probe):
        t = probe._lyrion_track({'id': 1, 'title': 'x', 'year': 'not a year', 'url': None})
        assert t['year'] is None

    def test_fetch_all_tracks_paginates(self, probe):
        page1 = _lyrion_wrap({'titles_loop': [
            {'id': i, 'title': f'T{i}', 'artist': 'A', 'albumartist': 'A',
             'album': 'Album', 'url': f'file:///m/t{i}.flac'}
            for i in range(1, 501)
        ]})
        page2 = _lyrion_wrap({'titles_loop': [
            {'id': i, 'title': f'T{i}', 'artist': 'A', 'albumartist': 'A',
             'album': 'Album', 'url': f'file:///m/t{i}.flac'}
            for i in range(501, 701)
        ]})
        responses = [_mock_response(page1), _mock_response(page2)]
        with patch.object(probe.mediaserver_lyrion.requests, 'post', side_effect=responses) as mock_post:
            tracks = probe.fetch_all_tracks('lyrion', self.CREDS)
        assert len(tracks) == 700
        # Two POST calls (page 2 was a short page, so the loop terminated)
        assert mock_post.call_count == 2
        # Offset advances correctly
        body_p2 = mock_post.call_args_list[1][1]['json']
        assert body_p2['params'][1][1] == 500  # offset

    def test_fetch_all_tracks_empty(self, probe):
        with patch.object(probe.mediaserver_lyrion.requests, 'post', return_value=_mock_response(_lyrion_wrap({'titles_loop': []}))):
            tracks = probe.fetch_all_tracks('lyrion', self.CREDS)
        assert tracks == []

    def test_fetch_all_tracks_skips_spotify(self, probe):
        page = _lyrion_wrap({'titles_loop': [
            {'id': 1, 'title': 'Local', 'artist': 'X', 'album': 'Y', 'url': 'file:///a.flac'},
            {'id': 2, 'title': 'Remote', 'artist': 'X', 'album': 'Y', 'url': 'spotify:track:xyz'},
            {'id': 3, 'title': 'AlsoSpotify', 'genre': 'Spotify', 'url': 'file:///b.flac'},
        ]})
        with patch.object(probe.mediaserver_lyrion.requests, 'post', return_value=_mock_response(page)):
            tracks = probe.fetch_all_tracks('lyrion', self.CREDS)
        assert len(tracks) == 1
        assert tracks[0]['id'] == '1'

    def test_search_albums(self, probe):
        resp = _lyrion_wrap({'albums_loop': [
            {'id': 10, 'album': 'Abbey Road', 'artist': 'The Beatles', 'year': 1969},
            {'id': 11, 'album': 'Let It Be', 'artist': 'The Beatles', 'year': 1970},
        ]})
        with patch.object(probe.requests, 'post', return_value=_mock_response(resp)) as mock_post:
            albums = probe.search_albums('lyrion', self.CREDS, 'Abbey')

        assert len(albums) == 2
        assert albums[0]['id'] == '10'
        assert albums[0]['name'] == 'Abbey Road'
        assert albums[0]['artist'] == 'The Beatles'
        assert albums[0]['year'] == 1969
        # Verify the JSON-RPC payload carried the search param
        body = mock_post.call_args[1]['json']
        params = body['params'][1]
        assert params[0] == 'albums'
        assert any(p == 'search:Abbey' for p in params)

    def test_get_album_tracks(self, probe):
        resp = _lyrion_wrap({'titles_loop': [
            {'id': 100, 'title': 'Come Together', 'artist': 'Beatles', 'album': 'Abbey Road',
             'albumartist': 'The Beatles', 'url': 'file:///music/beatles/come_together.flac',
             'year': 1969},
        ]})
        with patch.object(probe.requests, 'post', return_value=_mock_response(resp)) as mock_post:
            tracks = probe.get_album_tracks('lyrion', self.CREDS, 'alb1')

        assert len(tracks) == 1
        assert tracks[0]['id'] == '100'
        assert tracks[0]['title'] == 'Come Together'
        # JSON-RPC should carry album_id filter
        body = mock_post.call_args[1]['json']
        params = body['params'][1]
        assert params[0] == 'titles'
        assert any(p == 'album_id:alb1' for p in params)

    def test_test_connection_ok(self, probe):
        resp = _lyrion_wrap({'titles_loop': [
            {'id': 1, 'title': 'a', 'artist': 'x', 'album': 'y', 'url': 'file:///music/a.flac'}
        ]})
        with patch.object(probe.requests, 'post', return_value=_mock_response(resp)):
            result = probe.test_connection('lyrion', self.CREDS)
        assert result['ok'] is True
        assert result['sample_count'] == 1
        assert result['path_format'] == 'absolute'
        assert result['warnings'] == []

    def test_test_connection_failure(self, probe):
        def _raise(*_a, **_k):
            raise RuntimeError('connection refused')
        with patch.object(probe.requests, 'post', side_effect=_raise):
            result = probe.test_connection('lyrion', self.CREDS)
        assert result['ok'] is False
        assert 'connection refused' in result['error']
        assert result['sample_count'] == 0

    def test_test_connection_warns_when_paths_not_absolute(self, probe):
        resp = _lyrion_wrap({'titles_loop': [
            {'id': 1, 'title': 'a', 'artist': 'x', 'album': 'y', 'url': 'spotify:track:abc'}
        ]})
        # Spotify tracks are skipped, so sample ends up empty → path_format='none' → warning
        with patch.object(probe.requests, 'post', return_value=_mock_response(resp)):
            result = probe.test_connection('lyrion', self.CREDS)
        assert result['ok'] is True
        assert len(result['warnings']) == 1
        assert 'path' in result['warnings'][0].lower()

    def test_jsonrpc_sends_basic_auth_when_creds_have_user(self, probe):
        creds = {'url': 'http://lms.local:9000', 'user': 'admin', 'password': 'secret'}
        resp = _lyrion_wrap({'titles_loop': []})
        with patch.object(probe.requests, 'post', return_value=_mock_response(resp)) as mock_post:
            probe.fetch_all_tracks('lyrion', creds)
        kwargs = mock_post.call_args[1]
        assert kwargs.get('auth') == ('admin', 'secret')

    def test_jsonrpc_no_auth_when_creds_empty(self, probe):
        resp = _lyrion_wrap({'titles_loop': []})
        with patch.object(probe.requests, 'post', return_value=_mock_response(resp)) as mock_post:
            probe.fetch_all_tracks('lyrion', self.CREDS)
        kwargs = mock_post.call_args[1]
        assert kwargs.get('auth') is None


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
        with patch.object(probe.mediaserver_emby.requests, 'get', return_value=_mock_response(api_response)) as mock_get:
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
