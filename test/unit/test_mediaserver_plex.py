# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Plex media-server backend adapter unit tests.

Exercises the Plex provider helpers with mocked HTTP calls only; every stub
payload is shaped from the published Plex API contract at https://plexapi.dev/
so the request URLs, params, headers and response parsing stay accurate without
a live Plex server.

Main Features:
* Header/auth construction, library-section discovery and MUSIC_LIBRARIES filter
* Container-paginated song scan, album-track and recent-album parsing
* Track downloads, play stats, lyrics and server:// playlist create/replace flows
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import requests


PLEX_URL = 'http://plex:32400'
MACHINE_ID = 'abc123'


def _resp(payload):
    r = Mock()
    r.raise_for_status = Mock()
    r.json.return_value = payload
    return r


def _text_resp(text):
    r = Mock()
    r.raise_for_status = Mock()
    r.text = text
    return r


def _error_resp(status):
    r = Mock()
    err = requests.exceptions.HTTPError(f'{status} error')
    err_response = Mock()
    err_response.status_code = status
    err.response = err_response
    r.raise_for_status.side_effect = err
    r.json.return_value = {}
    return r


def _stream_resp(chunks=(b'audio-bytes',)):
    r = MagicMock()
    r.__enter__.return_value = r
    r.__exit__.return_value = False
    r.raise_for_status = Mock()
    r.iter_content = Mock(return_value=list(chunks))
    return r


def _mc(**fields):
    return {'MediaContainer': fields}


def _resp_mc(**fields):
    return _resp(_mc(**fields))


def _sections_payload():
    return _mc(
        size=2,
        title1='Plex Library',
        Directory=[
            {
                'key': '5',
                'type': 'artist',
                'title': 'Music',
                'agent': 'tv.plex.agents.music',
                'scanner': 'Plex Music',
                'uuid': 'sec-uuid-5',
            },
            {'key': '1', 'type': 'movie', 'title': 'Movies', 'agent': 'tv.plex.agents.movie'},
        ],
    )


def _identity_payload():
    return _mc(size=0, claimed=True, machineIdentifier=MACHINE_ID, version='1.41.0.8992')


def _min_track(rating_key):
    return {'ratingKey': str(rating_key), 'type': 'track', 'title': f'Song {rating_key}'}


def _track(
    rating_key='101',
    title='One More Time',
    grandparent='Daft Punk',
    parent='Discovery',
    original=None,
    index=1,
    parent_index=1,
    year=2001,
    parent_year=2001,
    view_count=5,
    last_viewed=1600000000,
    container='flac',
    file='/music/Daft Punk/Discovery/1-01 One More Time.flac',
    part_key=None,
):
    item = {
        'ratingKey': str(rating_key),
        'key': f'/library/metadata/{rating_key}',
        'parentRatingKey': '55',
        'grandparentRatingKey': '9',
        'guid': 'plex://track/aaa',
        'type': 'track',
        'title': title,
        'grandparentTitle': grandparent,
        'parentTitle': parent,
        'index': index,
        'parentIndex': parent_index,
        'year': year,
        'parentYear': parent_year,
        'viewCount': view_count,
        'duration': 210000,
        'addedAt': 1600000000,
        'Media': [
            {
                'id': 1,
                'container': container,
                'audioCodec': container,
                'Part': [
                    {
                        'id': 1,
                        'key': part_key or f'/library/parts/{rating_key}/1600000000/file.{container}',
                        'file': file,
                        'size': 20000000,
                        'container': container,
                    }
                ],
            }
        ],
    }
    if original is not None:
        item['originalTitle'] = original
    if last_viewed is not None:
        item['lastViewedAt'] = last_viewed
    return item


def _track_with_lyrics(rating_key='101', stream_key='/library/streams/999'):
    return {
        'ratingKey': str(rating_key),
        'type': 'track',
        'title': 'One More Time',
        'Media': [
            {
                'id': 1,
                'Part': [
                    {
                        'id': 1,
                        'key': '/library/parts/1/1/file.flac',
                        'Stream': [
                            {'id': 1, 'streamType': 2, 'codec': 'flac'},
                            {'id': 2, 'streamType': 4, 'format': 'lrc', 'key': stream_key},
                        ],
                    }
                ],
            }
        ],
    }


def _album(rating_key='700', title='Discovery', artist='Daft Punk', year=2001, added=1600000000, leaf=14):
    return {
        'ratingKey': str(rating_key),
        'key': f'/library/metadata/{rating_key}/children',
        'type': 'album',
        'title': title,
        'parentTitle': artist,
        'parentRatingKey': '9',
        'year': year,
        'addedAt': added,
        'leafCount': leaf,
    }


def _track_page(n, start=0, total=None):
    meta = [_min_track(start + i) for i in range(n)]
    fields = {'size': n, 'offset': start, 'Metadata': meta}
    if total is not None:
        fields['totalSize'] = total
    return _mc(**fields)


def _set_config(mock_config, music_libraries=''):
    mock_config.PLEX_URL = PLEX_URL
    mock_config.PLEX_TOKEN = 'tok'
    mock_config.MUSIC_LIBRARIES = music_libraries


@pytest.fixture(autouse=True)
def _clear_machine_cache():
    from tasks.mediaserver import plex

    plex._MACHINE_ID_CACHE.clear()
    yield
    plex._MACHINE_ID_CACHE.clear()


class TestPlexHeadersAndBaseUrl:
    @patch('tasks.mediaserver.plex.config')
    def test_headers_include_token_and_json_accept(self, mock_config):
        from tasks.mediaserver.plex import _headers

        mock_config.PLEX_TOKEN = 'tok'
        assert _headers() == {'Accept': 'application/json', 'X-Plex-Token': 'tok'}

    @patch('tasks.mediaserver.plex.config')
    def test_headers_prefer_user_creds_token(self, mock_config):
        from tasks.mediaserver.plex import _headers

        mock_config.PLEX_TOKEN = 'tok'
        assert _headers({'token': 'other'})['X-Plex-Token'] == 'other'

    @patch('tasks.mediaserver.plex.config')
    def test_headers_omit_token_when_absent(self, mock_config):
        from tasks.mediaserver.plex import _headers

        mock_config.PLEX_TOKEN = ''
        assert _headers() == {'Accept': 'application/json'}

    @patch('tasks.mediaserver.plex.config')
    def test_base_url_strips_trailing_slash(self, mock_config):
        from tasks.mediaserver.plex import _base_url

        mock_config.PLEX_URL = 'http://plex:32400/'
        assert _base_url() == 'http://plex:32400'
        assert _base_url({'url': 'http://other:32400'}) == 'http://other:32400'


class TestPlexListLibraries:
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_returns_only_artist_sections(self, mock_config, mock_get):
        from tasks.mediaserver.plex import list_libraries

        _set_config(mock_config)
        mock_get.return_value = _resp(_sections_payload())

        libraries = list_libraries()

        assert libraries == [{'id': '5', 'name': 'Music'}]
        assert mock_get.call_args[0][0] == f'{PLEX_URL}/library/sections'
        assert mock_get.call_args.kwargs['headers']['X-Plex-Token'] == 'tok'

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_returns_empty_on_error(self, mock_config, mock_get):
        from tasks.mediaserver.plex import list_libraries

        _set_config(mock_config)
        mock_get.side_effect = requests.exceptions.ConnectionError('down')

        assert list_libraries() == []


class TestPlexMusicLibrariesFilter:
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_matches_configured_name(self, mock_config, mock_get):
        from tasks.mediaserver.plex import _target_sections

        _set_config(mock_config, music_libraries='Music')
        mock_get.return_value = _resp(_sections_payload())

        assert _target_sections() == [{'id': '5', 'name': 'Music'}]

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_no_match_returns_empty_and_skips_scan(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_all_songs

        _set_config(mock_config, music_libraries='Nonexistent')
        mock_get.return_value = _resp(_sections_payload())

        assert get_all_songs() == []
        assert mock_get.call_count == 1

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_empty_filter_uses_all_sections(self, mock_config, mock_get):
        from tasks.mediaserver.plex import _target_sections

        _set_config(mock_config, music_libraries='')
        mock_get.return_value = _resp(_sections_payload())

        assert _target_sections() == [{'id': '5', 'name': 'Music'}]

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_user_creds_skip_music_libraries_filter(self, mock_config, mock_get):
        from tasks.mediaserver.plex import _target_sections

        _set_config(mock_config, music_libraries='Nonexistent')
        mock_get.return_value = _resp(_sections_payload())

        assert _target_sections({'url': PLEX_URL, 'token': 'x'}) == [{'id': '5', 'name': 'Music'}]


class TestPlexGetAllSongsPagination:
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_paginates_until_short_page(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_all_songs

        _set_config(mock_config)
        mock_get.side_effect = [
            _resp(_sections_payload()),
            _resp(_track_page(1000, start=0, total=1003)),
            _resp(_track_page(3, start=1000, total=1003)),
        ]

        songs = get_all_songs()

        assert len(songs) == 1003
        assert mock_get.call_count == 3
        page1 = mock_get.call_args_list[1]
        assert page1[0][0] == f'{PLEX_URL}/library/sections/5/all'
        assert page1.kwargs['params'] == {'type': 10}
        assert page1.kwargs['headers']['X-Plex-Container-Start'] == '0'
        assert page1.kwargs['headers']['X-Plex-Container-Size'] == '1000'
        assert mock_get.call_args_list[2].kwargs['headers']['X-Plex-Container-Start'] == '1000'

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_normalizes_track_fields(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_all_songs

        _set_config(mock_config)
        mock_get.side_effect = [
            _resp(_sections_payload()),
            _resp_mc(size=1, Metadata=[_track(original='Daft Punk feat. Romanthony')]),
        ]

        songs = get_all_songs()

        assert len(songs) == 1
        song = songs[0]
        assert song['Id'] == '101'
        assert song['Name'] == 'One More Time'
        assert song['AlbumArtist'] == 'Daft Punk feat. Romanthony'
        assert song['OriginalAlbumArtist'] == 'Daft Punk'
        assert song['ArtistId'] == '9'
        assert song['Album'] == 'Discovery'
        assert song['Year'] == 2001
        assert song['IndexNumber'] == 1
        assert song['ParentIndexNumber'] == 1
        assert song['Path'] == '/music/Daft Punk/Discovery/1-01 One More Time.flac'
        assert song['FilePath'] == song['Path']
        assert song['Container'] == 'flac'
        assert song['PartKey'] == '/library/parts/101/1600000000/file.flac'

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_falls_back_to_grandparent_when_no_original_title(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_all_songs

        _set_config(mock_config)
        mock_get.side_effect = [
            _resp(_sections_payload()),
            _resp_mc(size=1, Metadata=[_track(original=None)]),
        ]

        assert get_all_songs()[0]['AlbumArtist'] == 'Daft Punk'

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_raises_on_midscan_failure_instead_of_truncating(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_all_songs

        _set_config(mock_config)
        mock_get.side_effect = [
            _resp(_sections_payload()),
            _resp(_track_page(1000, start=0, total=2000)),
            requests.exceptions.ReadTimeout('read timed out'),
        ]

        with pytest.raises(requests.exceptions.ReadTimeout):
            get_all_songs()

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_empty_library_returns_empty(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_all_songs

        _set_config(mock_config)
        mock_get.side_effect = [
            _resp(_sections_payload()),
            _resp(_track_page(0, start=0, total=0)),
        ]

        assert get_all_songs() == []


class TestPlexGetTracksFromAlbum:
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_fetches_children_and_normalizes(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_tracks_from_album

        _set_config(mock_config)
        mock_get.return_value = _resp_mc(
            size=1, parentTitle='Discovery', Metadata=[_track(original='Feat Guest')]
        )

        tracks = get_tracks_from_album('55')

        assert mock_get.call_args[0][0] == f'{PLEX_URL}/library/metadata/55/children'
        assert tracks[0]['AlbumArtist'] == 'Feat Guest'
        assert tracks[0]['OriginalAlbumArtist'] == 'Daft Punk'
        assert tracks[0]['Path'] == '/music/Daft Punk/Discovery/1-01 One More Time.flac'
        assert tracks[0]['Year'] == 2001

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_year_falls_back_to_track_year(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_tracks_from_album

        _set_config(mock_config)
        track = _track(parent_year=None, year=1999)
        mock_get.return_value = _resp_mc(size=1, Metadata=[track])

        assert get_tracks_from_album('55')[0]['Year'] == 1999

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_returns_empty_on_error(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_tracks_from_album

        _set_config(mock_config)
        mock_get.side_effect = requests.exceptions.RequestException('boom')

        assert get_tracks_from_album('55') == []


class TestPlexGetRecentAlbums:
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_sorts_by_added_desc_and_honours_limit(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_recent_albums

        _set_config(mock_config)
        albums = [
            _album(rating_key='700', title='Discovery', added=1600000000),
            _album(rating_key='701', title='Homework', artist='Daft Punk', added=1500000000),
        ]
        mock_get.side_effect = [
            _resp(_sections_payload()),
            _resp_mc(size=2, Metadata=albums),
        ]

        result = get_recent_albums(5)

        page = mock_get.call_args_list[1]
        assert page[0][0] == f'{PLEX_URL}/library/sections/5/all'
        assert page.kwargs['params'] == {'type': 9, 'sort': 'addedAt:desc'}
        assert page.kwargs['headers']['X-Plex-Container-Size'] == '5'
        assert [a['Id'] for a in result] == ['700', '701']
        assert result[0] == {
            'Id': '700',
            'Name': 'Discovery',
            'AlbumArtist': 'Daft Punk',
            'Year': 2001,
            'DateCreated': 1600000000,
        }

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_limit_zero_fetches_all_pages(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_recent_albums

        _set_config(mock_config)
        mock_get.side_effect = [
            _resp(_sections_payload()),
            _resp_mc(size=1, totalSize=1, Metadata=[_album()]),
        ]

        result = get_recent_albums(0)

        assert len(result) == 1
        assert mock_get.call_args_list[1].kwargs['headers']['X-Plex-Container-Size'] == '1000'

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_no_sections_returns_empty(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_recent_albums

        _set_config(mock_config)
        mock_get.return_value = _resp_mc(size=0, Directory=[])

        assert get_recent_albums(10) == []


class TestPlexSearchAlbums:
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_sends_title_filter_and_shapes_results(self, mock_config, mock_get):
        from tasks.mediaserver.plex import search_albums

        _set_config(mock_config)
        mock_get.side_effect = [
            _resp(_sections_payload()),
            _resp_mc(size=1, Metadata=[_album()]),
        ]

        results = search_albums('discovery')

        search_call = mock_get.call_args_list[1]
        assert search_call[0][0] == f'{PLEX_URL}/library/sections/5/all'
        assert search_call.kwargs['params'] == {'type': 9, 'title': 'discovery'}
        assert results == [
            {'id': '700', 'name': 'Discovery', 'artist': 'Daft Punk', 'year': 2001, 'track_count': 14}
        ]

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_returns_empty_on_error(self, mock_config, mock_get):
        from tasks.mediaserver.plex import search_albums

        _set_config(mock_config)
        mock_get.side_effect = [
            _resp(_sections_payload()),
            requests.exceptions.RequestException('boom'),
        ]

        assert search_albums('x') == []


class TestPlexDownloadTrack:
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_streams_from_part_key(self, mock_config, mock_get, tmp_path):
        from tasks.mediaserver.plex import download_track

        _set_config(mock_config)
        mock_get.return_value = _stream_resp((b'aaa', b'bbb'))

        item = {
            'Id': '101',
            'Name': 'One More Time',
            'PartKey': '/library/parts/101/1600000000/file.flac',
            'Container': 'flac',
            'Path': '/music/x.flac',
        }
        path = download_track(str(tmp_path), item)

        assert path == str(tmp_path / '101.flac')
        assert mock_get.call_args[0][0] == f'{PLEX_URL}/library/parts/101/1600000000/file.flac'
        assert mock_get.call_args.kwargs['params'] == {'download': 1}
        with open(path, 'rb') as f:
            assert f.read() == b'aaabbb'

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_resolves_part_when_missing(self, mock_config, mock_get, tmp_path):
        from tasks.mediaserver.plex import download_track

        _set_config(mock_config)
        mock_get.side_effect = [
            _resp(_mc(size=1, Metadata=[_track(rating_key='101')])),
            _stream_resp((b'zzz',)),
        ]

        item = {'Id': '101', 'id': '101', 'Name': 'One More Time', 'Path': ''}
        path = download_track(str(tmp_path), item)

        assert path == str(tmp_path / '101.flac')
        assert mock_get.call_args_list[0][0][0] == f'{PLEX_URL}/library/metadata/101'
        assert mock_get.call_args_list[1][0][0] == f'{PLEX_URL}/library/parts/101/1600000000/file.flac'

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_returns_none_when_no_part_found(self, mock_config, mock_get, tmp_path):
        from tasks.mediaserver.plex import download_track

        _set_config(mock_config)
        mock_get.return_value = _resp(_mc(size=1, Metadata=[{'ratingKey': '101', 'title': 'x'}]))

        assert download_track(str(tmp_path), {'Id': '101', 'Name': 'x'}) is None


class TestPlexTestConnection:
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_ok_reports_sample_and_path_format(self, mock_config, mock_get):
        from tasks.mediaserver.plex import test_connection

        _set_config(mock_config)
        mock_get.side_effect = [
            _resp(_sections_payload()),
            _resp_mc(size=2, Metadata=[_track(rating_key='101'), _track(rating_key='102')]),
        ]

        result = test_connection()

        assert result['ok'] is True
        assert result['sample_count'] == 2
        assert result['path_format'] == 'absolute'
        probe = mock_get.call_args_list[1]
        assert probe[0][0] == f'{PLEX_URL}/library/sections/5/all'
        assert probe.kwargs['params'] == {'type': 10}
        assert probe.kwargs['headers']['X-Plex-Container-Size'] == '100'

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_auth_failure_sets_auth_failed(self, mock_config, mock_get):
        from tasks.mediaserver.plex import test_connection

        _set_config(mock_config)
        mock_get.return_value = _error_resp(401)

        result = test_connection()

        assert result['ok'] is False
        assert result['auth_failed'] is True

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_no_music_section_reports_error(self, mock_config, mock_get):
        from tasks.mediaserver.plex import test_connection

        _set_config(mock_config)
        mock_get.return_value = _resp_mc(size=1, Directory=[{'key': '1', 'type': 'movie', 'title': 'Movies'}])

        result = test_connection()

        assert result['ok'] is False
        assert result['auth_failed'] is False
        assert 'music library' in result['error'].lower()

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_samples_across_sections_when_first_empty(self, mock_config, mock_get):
        from tasks.mediaserver.plex import test_connection

        _set_config(mock_config)
        two_music = _mc(
            size=2,
            Directory=[
                {'key': '5', 'type': 'artist', 'title': 'Empty'},
                {'key': '6', 'type': 'artist', 'title': 'Music'},
            ],
        )
        mock_get.side_effect = [
            _resp(two_music),
            _resp_mc(size=0, Metadata=[]),
            _resp_mc(size=1, Metadata=[_track(rating_key='101')]),
        ]

        result = test_connection()

        assert result['ok'] is True
        assert result['sample_count'] == 1
        assert mock_get.call_args_list[1][0][0] == f'{PLEX_URL}/library/sections/5/all'
        assert mock_get.call_args_list[2][0][0] == f'{PLEX_URL}/library/sections/6/all'


class TestPlexPlaylistReads:
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_get_all_playlists_normalizes(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_all_playlists

        _set_config(mock_config)
        mock_get.return_value = _resp_mc(
            size=2,
            Metadata=[
                {'ratingKey': '55', 'title': 'My Mix', 'type': 'playlist', 'playlistType': 'audio', 'leafCount': 3},
                {'ratingKey': '56', 'title': 'Chill_automatic', 'type': 'playlist', 'playlistType': 'audio'},
            ],
        )

        playlists = get_all_playlists()

        assert playlists == [{'Id': '55', 'Name': 'My Mix'}, {'Id': '56', 'Name': 'Chill_automatic'}]
        assert mock_get.call_args.kwargs['params'] == {'playlistType': 'audio'}

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_get_playlist_by_name_exact_match(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_playlist_by_name

        _set_config(mock_config)
        mock_get.return_value = _resp_mc(
            size=1, Metadata=[{'ratingKey': '55', 'title': 'My Mix'}]
        )

        assert get_playlist_by_name('My Mix') == {'Id': '55', 'Name': 'My Mix'}
        assert get_playlist_by_name('Missing') is None

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_get_playlist_track_ids(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_playlist_track_ids

        _set_config(mock_config)
        mock_get.return_value = _resp_mc(
            size=2, Metadata=[_track(rating_key='101'), _track(rating_key='102')]
        )

        assert get_playlist_track_ids('55') == ['101', '102']
        assert mock_get.call_args[0][0] == f'{PLEX_URL}/playlists/55/items'


class TestPlexDeletePlaylist:
    @patch('tasks.mediaserver.plex.requests.delete')
    @patch('tasks.mediaserver.plex.config')
    def test_delete_success(self, mock_config, mock_delete):
        from tasks.mediaserver.plex import delete_playlist

        _set_config(mock_config)
        mock_delete.return_value = _resp({})

        assert delete_playlist('55') is True
        assert mock_delete.call_args[0][0] == f'{PLEX_URL}/playlists/55'

    @patch('tasks.mediaserver.plex.requests.delete')
    @patch('tasks.mediaserver.plex.config')
    def test_delete_failure_returns_false(self, mock_config, mock_delete):
        from tasks.mediaserver.plex import delete_playlist

        _set_config(mock_config)
        mock_delete.side_effect = requests.exceptions.RequestException('boom')

        assert delete_playlist('55') is False


class TestPlexCreatePlaylist:
    @patch('tasks.mediaserver.plex.requests.post')
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_instant_playlist_posts_metadata_uri(self, mock_config, mock_get, mock_post):
        from tasks.mediaserver.plex import create_instant_playlist

        _set_config(mock_config)
        mock_get.return_value = _resp(_identity_payload())
        mock_post.return_value = _resp(_mc(Metadata=[{'ratingKey': '900', 'title': 'Mix_instant'}]))

        result = create_instant_playlist('Mix', ['1', '2', '3'])

        assert result == {'Id': '900', 'Name': 'Mix_instant'}
        assert mock_get.call_args[0][0] == f'{PLEX_URL}/identity'
        create_call = mock_post.call_args
        assert create_call[0][0] == f'{PLEX_URL}/playlists'
        assert create_call.kwargs['params'] == {
            'type': 'audio',
            'title': 'Mix_instant',
            'smart': 0,
            'uri': f'server://{MACHINE_ID}/com.plexapp.plugins.library/library/metadata/1,2,3',
        }

    @patch('tasks.mediaserver.plex.requests.put')
    @patch('tasks.mediaserver.plex.requests.post')
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_overflow_ids_added_in_second_request(self, mock_config, mock_get, mock_post, mock_put):
        from tasks.mediaserver.plex import create_instant_playlist

        _set_config(mock_config)
        mock_get.return_value = _resp(_identity_payload())
        mock_post.return_value = _resp(_mc(Metadata=[{'ratingKey': '900', 'title': 'Big_instant'}]))
        mock_put.return_value = _resp(_mc())

        ids = [str(i) for i in range(150)]
        create_instant_playlist('Big', ids)

        assert mock_post.call_count == 1
        first_uri = mock_post.call_args_list[0].kwargs['params']['uri']
        assert first_uri.endswith('/library/metadata/' + ','.join(str(i) for i in range(100)))
        add_call = mock_put.call_args_list[0]
        assert add_call[0][0] == f'{PLEX_URL}/playlists/900/items'
        assert add_call.kwargs['params']['uri'].endswith(
            '/library/metadata/' + ','.join(str(i) for i in range(100, 150))
        )

    @patch('tasks.mediaserver.plex.requests.put')
    @patch('tasks.mediaserver.plex.requests.post')
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_overflow_add_failure_returns_none(self, mock_config, mock_get, mock_post, mock_put):
        from tasks.mediaserver.plex import create_instant_playlist

        _set_config(mock_config)
        mock_get.return_value = _resp(_identity_payload())
        mock_post.return_value = _resp(_mc(Metadata=[{'ratingKey': '900', 'title': 'Big_instant'}]))
        mock_put.return_value = _error_resp(500)

        result = create_instant_playlist('Big', [str(i) for i in range(150)])

        assert result is None

    @patch('tasks.mediaserver.plex.requests.post')
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_create_playlist_swallows_errors(self, mock_config, mock_get, mock_post):
        from tasks.mediaserver.plex import create_playlist

        _set_config(mock_config)
        mock_get.return_value = _resp(_identity_payload())
        mock_post.side_effect = requests.exceptions.RequestException('boom')

        create_playlist('Mix', ['1'])

        assert mock_post.called


class TestPlexCreateOrReplacePlaylist:
    @patch('tasks.mediaserver.plex.requests.delete')
    @patch('tasks.mediaserver.plex.requests.post')
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_missing_playlist_creates_new(self, mock_config, mock_get, mock_post, mock_delete):
        from tasks.mediaserver.plex import create_or_replace_playlist

        _set_config(mock_config)
        mock_get.side_effect = [
            _resp(_mc(size=0, Metadata=[])),
            _resp(_identity_payload()),
        ]
        mock_post.return_value = _resp(_mc(Metadata=[{'ratingKey': '900', 'title': 'SF'}]))

        result = create_or_replace_playlist('SF', ['1', '2'])

        assert result == {'Id': '900', 'Name': 'SF'}
        assert mock_delete.called is False

    @patch('tasks.mediaserver.plex.requests.delete')
    @patch('tasks.mediaserver.plex.requests.post')
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_existing_playlist_deleted_then_created(self, mock_config, mock_get, mock_post, mock_delete):
        from tasks.mediaserver.plex import create_or_replace_playlist

        _set_config(mock_config)
        mock_get.side_effect = [
            _resp(_mc(size=1, Metadata=[{'ratingKey': '55', 'title': 'SF'}])),
            _resp(_identity_payload()),
        ]
        mock_delete.return_value = _resp({})
        mock_post.return_value = _resp(_mc(Metadata=[{'ratingKey': '900', 'title': 'SF'}]))

        result = create_or_replace_playlist('SF', ['1', '2'])

        assert mock_delete.call_args[0][0] == f'{PLEX_URL}/playlists/55'
        assert result == {'Id': '900', 'Name': 'SF'}


class TestPlexTopPlayedAndLastPlayed:
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_top_played_sorts_and_limits(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_top_played_songs

        _set_config(mock_config)
        mock_get.side_effect = [
            _resp(_sections_payload()),
            _resp_mc(
                size=3,
                Metadata=[
                    _track(rating_key='101', view_count=2),
                    _track(rating_key='102', view_count=9),
                    _track(rating_key='103', view_count=5),
                ],
            ),
        ]

        result = get_top_played_songs(2)

        probe = mock_get.call_args_list[1]
        assert probe.kwargs['params'] == {'type': 10, 'sort': 'viewCount:desc'}
        assert probe.kwargs['headers']['X-Plex-Container-Size'] == '2'
        assert [t['Id'] for t in result] == ['102', '103']

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_last_played_converts_epoch_to_iso(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_last_played_time

        _set_config(mock_config)
        mock_get.return_value = _resp_mc(size=1, Metadata=[_track(rating_key='101', last_viewed=1600000000)])

        assert get_last_played_time('101') == '2020-09-13T12:26:40.000Z'
        assert mock_get.call_args[0][0] == f'{PLEX_URL}/library/metadata/101'

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_last_played_none_when_never_played(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_last_played_time

        _set_config(mock_config)
        mock_get.return_value = _resp_mc(size=1, Metadata=[_track(rating_key='101', last_viewed=None)])

        assert get_last_played_time('101') is None


class TestPlexGetLyrics:
    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_resolves_and_fetches_lyric_stream(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_lyrics

        _set_config(mock_config)
        mock_get.side_effect = [
            _resp(_mc(size=1, Metadata=[_track_with_lyrics(stream_key='/library/streams/999')])),
            _text_resp('la la la\nsecond line'),
        ]

        assert get_lyrics('101') == 'la la la\nsecond line'
        assert mock_get.call_args_list[1][0][0] == f'{PLEX_URL}/library/streams/999'

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_returns_none_when_no_lyric_stream(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_lyrics

        _set_config(mock_config)
        mock_get.return_value = _resp(_mc(size=1, Metadata=[_track(rating_key='101')]))

        assert get_lyrics('101') is None

    @patch('tasks.mediaserver.plex.requests.get')
    @patch('tasks.mediaserver.plex.config')
    def test_forwards_timeout(self, mock_config, mock_get):
        from tasks.mediaserver.plex import get_lyrics

        _set_config(mock_config)
        mock_get.side_effect = [
            _resp(_mc(size=1, Metadata=[_track_with_lyrics()])),
            _text_resp('words'),
        ]

        get_lyrics('101', timeout=1.0)

        assert mock_get.call_args_list[0].kwargs['timeout'] == 1.0
