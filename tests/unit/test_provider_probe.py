"""Unit tests for tasks.provider_probe.

The probe module is a thin shim: it normalizes the provider type, delegates
to :mod:`tasks.mediaserver` (passing ``user_creds`` / ``provider_type``
explicitly so it never reads ``config.py``), and normalizes the per-track
dicts returned by the mediaserver layer into a unified shape.

Tests exercise the shim's actual behavior:
  * delegation to ``tasks.mediaserver`` with the right arguments,
  * provider-type validation (normalize + reject unsupported),
  * unified track shape produced by ``_normalize_track``.

Uses the ``importlib.util`` bypass so ``tasks/__init__.py`` (which pulls in
heavy ML deps like librosa) isn't imported.
"""
import importlib.util
import os
import sys
from unittest.mock import patch

import pytest


def _load_probe():
    mod_name = 'tasks.provider_probe'
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    )
    # Ensure `tasks.mediaserver` and friends are importable as a real package
    # when provider_probe does `from tasks import mediaserver`.
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    mod_path = os.path.join(repo_root, 'tasks', 'provider_probe.py')
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def probe():
    return _load_probe()


# ---------------------------------------------------------------------------
# _normalize_track — unified track shape
# ---------------------------------------------------------------------------

class TestNormalizeTrack:
    REQUIRED_KEYS = {
        'id', 'path', 'title', 'artist', 'album_artist', 'album',
        'year', 'track_number', 'disc_number',
    }

    def test_none_item_returns_empty_shape(self, probe):
        t = probe._normalize_track(None)
        assert set(t.keys()) == self.REQUIRED_KEYS
        assert all(v is None for v in t.values())

    def test_jellyfin_style_item(self, probe):
        item = {
            'Id': 'j1', 'Name': 'Song One', 'Album': 'Album A',
            'AlbumArtist': 'Artist A', 'Path': '/m/a/song1.flac',
            'Year': 2020, 'IndexNumber': 3, 'ParentIndexNumber': 1,
        }
        t = probe._normalize_track(item)
        assert t['id'] == 'j1'
        assert t['title'] == 'Song One'
        assert t['album'] == 'Album A'
        assert t['artist'] == 'Artist A'
        assert t['path'] == '/m/a/song1.flac'
        assert t['year'] == 2020
        assert t['track_number'] == 3
        assert t['disc_number'] == 1

    def test_lowercase_style_item(self, probe):
        item = {
            'id': 'n1', 'title': 'Song', 'album': 'Album',
            'artist': 'Artist', 'path': '/m/song.flac',
            'year': 2019, 'track_number': 5, 'disc_number': 2,
        }
        t = probe._normalize_track(item)
        assert t['id'] == 'n1'
        assert t['title'] == 'Song'
        assert t['album'] == 'Album'
        assert t['artist'] == 'Artist'
        assert t['path'] == '/m/song.flac'
        assert t['year'] == 2019
        assert t['track_number'] == 5
        assert t['disc_number'] == 2

    def test_year_string_is_coerced_to_int(self, probe):
        t = probe._normalize_track({'Id': 'x', 'Year': '2018'})
        assert t['year'] == 2018

    def test_year_invalid_string_becomes_none(self, probe):
        t = probe._normalize_track({'Id': 'x', 'Year': 'not a year'})
        assert t['year'] is None

    def test_keys_always_present(self, probe):
        t = probe._normalize_track({'Id': 'only-id'})
        assert set(t.keys()) == self.REQUIRED_KEYS


# ---------------------------------------------------------------------------
# _normalize_provider_type — validation
# ---------------------------------------------------------------------------

class TestNormalizeProviderType:
    def test_supported_providers_normalized_lowercase(self, probe):
        for t in ('jellyfin', 'Jellyfin', 'EMBY', 'Navidrome', 'LYRION'):
            assert probe._normalize_provider_type(t) == t.lower()

    def test_unsupported_provider_raises(self, probe):
        with pytest.raises(ValueError) as ei:
            probe._normalize_provider_type('mpd')
        assert 'not supported' in str(ei.value)

    def test_empty_or_none_raises(self, probe):
        with pytest.raises(ValueError):
            probe._normalize_provider_type(None)
        with pytest.raises(ValueError):
            probe._normalize_provider_type('')


# ---------------------------------------------------------------------------
# fetch_all_tracks — delegation + normalization
# ---------------------------------------------------------------------------

class TestFetchAllTracks:
    CREDS = {'url': 'http://host', 'token': 'tok'}

    def test_delegates_to_mediaserver_and_normalizes(self, probe):
        fake_items = [
            {'Id': 'a', 'Name': 'A', 'Path': '/m/a.flac'},
            {'Id': 'b', 'Name': 'B', 'Path': '/m/b.flac'},
        ]
        with patch.object(probe.mediaserver, 'get_all_songs', return_value=fake_items) as m:
            tracks = probe.fetch_all_tracks('jellyfin', self.CREDS)
        m.assert_called_once_with(user_creds=self.CREDS, provider_type='jellyfin')
        assert len(tracks) == 2
        assert tracks[0]['id'] == 'a'
        assert tracks[1]['id'] == 'b'
        assert all(set(t.keys()) >= TestNormalizeTrack.REQUIRED_KEYS for t in tracks)

    def test_empty_result_is_handled(self, probe):
        with patch.object(probe.mediaserver, 'get_all_songs', return_value=None):
            assert probe.fetch_all_tracks('navidrome', self.CREDS) == []
        with patch.object(probe.mediaserver, 'get_all_songs', return_value=[]):
            assert probe.fetch_all_tracks('navidrome', self.CREDS) == []

    def test_unsupported_provider_raises_before_call(self, probe):
        with patch.object(probe.mediaserver, 'get_all_songs') as m:
            with pytest.raises(ValueError):
                probe.fetch_all_tracks('mpd', self.CREDS)
        m.assert_not_called()


# ---------------------------------------------------------------------------
# search_albums — delegation (raw pass-through)
# ---------------------------------------------------------------------------

class TestSearchAlbums:
    CREDS = {'url': 'http://host', 'token': 'tok'}

    def test_delegates_to_mediaserver(self, probe):
        fake_albums = [{'id': '1', 'name': 'Album 1'}]
        with patch.object(probe.mediaserver, 'search_albums', return_value=fake_albums) as m:
            result = probe.search_albums('emby', self.CREDS, 'query text')
        m.assert_called_once_with('query text', user_creds=self.CREDS, provider_type='emby')
        assert result == fake_albums

    def test_unsupported_provider_raises(self, probe):
        with patch.object(probe.mediaserver, 'search_albums') as m:
            with pytest.raises(ValueError):
                probe.search_albums('mpd', self.CREDS, 'q')
        m.assert_not_called()


# ---------------------------------------------------------------------------
# get_album_tracks — delegation + normalization
# ---------------------------------------------------------------------------

class TestGetAlbumTracks:
    CREDS = {'url': 'http://host', 'token': 'tok'}

    def test_delegates_and_normalizes(self, probe):
        fake_items = [{'Id': 'x', 'Name': 'Title'}]
        with patch.object(probe.mediaserver, 'get_tracks_from_album', return_value=fake_items) as m:
            tracks = probe.get_album_tracks('jellyfin', self.CREDS, 'album-42')
        m.assert_called_once_with('album-42', user_creds=self.CREDS, provider_type='jellyfin')
        assert len(tracks) == 1
        assert tracks[0]['id'] == 'x'
        assert tracks[0]['title'] == 'Title'

    def test_empty_result_is_handled(self, probe):
        with patch.object(probe.mediaserver, 'get_tracks_from_album', return_value=None):
            assert probe.get_album_tracks('lyrion', self.CREDS, 'album-1') == []


# ---------------------------------------------------------------------------
# test_connection — delegation (raw pass-through)
# ---------------------------------------------------------------------------

class TestTestConnection:
    CREDS = {'url': 'http://host'}

    def test_delegates_to_mediaserver(self, probe):
        fake_result = {
            'ok': True, 'error': None, 'sample_count': 10,
            'path_format': 'absolute', 'warnings': [],
        }
        with patch.object(probe.mediaserver, 'test_connection', return_value=fake_result) as m:
            result = probe.test_connection('navidrome', self.CREDS)
        m.assert_called_once_with(user_creds=self.CREDS, provider_type='navidrome')
        assert result == fake_result

    def test_unsupported_provider_raises(self, probe):
        with patch.object(probe.mediaserver, 'test_connection') as m:
            with pytest.raises(ValueError):
                probe.test_connection('mpd', self.CREDS)
        m.assert_not_called()
