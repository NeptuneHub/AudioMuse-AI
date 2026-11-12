# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""VLC media-player backend adapter unit tests.

Drives tasks/mediaserver/vlc.py against an in-memory fake of VLC 3's Lua HTTP
interface (requests/browse.json, status.json, playlist.json, vlm_cmd.xml, vlm.xml
and /art), shaped from the real responses of VLC 3.0.24, so the URI handling, id
scheme, metadata, download and playlist behaviour are pinned without a running VLC.

Main Features:
* Windows, Linux, macOS, UNC and file:// music folders become the file:// URI VLC
  needs; queries are percent-encoded (VLC's Lua parser never decodes '+').
* Track ids are the hash of the path relative to the music folder: stable across
  scans and hosts, NFC/separator independent; album ids round-trip and refuse '..'.
* Folder-layout metadata, catalogue walk (hidden/non-audio skipped, an error page
  aborts instead of returning a partial catalogue), recent-album order.
* Downloads: VLM copy job (raw MP3, mkv otherwise) into Folder.jpg in the VLC
  host's temp folder, play-queue item served whole by /art, the copy job as the
  lock (wait, stale takeover), 404 until preparsed, empty or short copies
  rejected, play-queue and job cleanup, and the probe download in test_connection.
* The empty/no-op playlist, play-history and lyrics answers, plus the
  registry/setup validation rules.
"""

import os
import re
import time
import unicodedata
from unittest.mock import Mock, patch
from urllib.parse import parse_qsl, quote

import pytest
import requests

import config
from tasks import mediaserver
from tasks.mediaserver import context, vlc

VLC_URL = 'http://vlc:8080'
PASSWORD = 'secret'
ROOT_URI = 'file:///C:/Music'
ROOT_PATH = 'C:\\Music'


def _set_config(music_folder=ROOT_PATH, user=''):
    config.MEDIASERVER_TYPE = 'vlc'
    config.VLC_URL = VLC_URL
    config.VLC_USER = user
    config.VLC_PASSWORD = PASSWORD
    config.VLC_MUSIC_FOLDER = music_folder


@pytest.fixture(autouse=True)
def _no_polling_delay():
    with patch.object(vlc, 'COPY_POLL_SECONDS', 0), patch.object(vlc, 'COPY_RETRY_SECONDS', 0):
        yield


def _json_resp(payload, status=200):
    r = Mock()
    r.status_code = status
    r.json.return_value = payload
    r.text = ''
    r.content = b''
    r.raise_for_status = Mock()
    return r


def _text_resp(text, status=200):
    r = Mock()
    r.status_code = status
    r.text = text
    r.content = text.encode('utf-8')
    r.json.side_effect = ValueError('not json')
    if status >= 400:
        err = requests.exceptions.HTTPError(f'{status} error')
        err.response = Mock(status_code=status)
        r.raise_for_status.side_effect = err
    else:
        r.raise_for_status = Mock()
    return r


class _BodyResp:
    def __init__(self, body=b'', status=200, length=None):
        self.status_code = status
        self._body = body
        self.headers = {'Content-Length': str(len(body) if length is None else length)}

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.exceptions.HTTPError(f'{self.status_code} error')
            err.response = self
            raise err

    def iter_content(self, chunk_size=None):
        return iter([self._body[i:i + 4] for i in range(0, len(self._body), 4)])

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _error_page(message):
    return (
        '<html><head><title>Error loading /requests/browse.json</title></head>'
        f'<body><h1>Error loading</h1><pre>httprequests.lua:419: {message}</pre></body></html>'
    )


_NEW = re.compile(r'new (\S+) broadcast enabled input "([^"]+)" output "(.+)" option (\S+)$')
_COPY_OUTPUT = re.compile(r'#std\{access=file,mux=(\w+),dst=\\"([^"\\]+)\\"\}$')


class FakeVlc:
    def __init__(self, files, root_uri=ROOT_URI, root_path=ROOT_PATH, sep='\\', mtimes=None, sizes=None):
        self.root_uri = root_uri
        self.listings = {}
        self.broken = set()
        self.urls = []
        self.commands = []
        self.enqueued = []
        self.media = {}
        self.disk = {}
        self.queue = []
        self.next_id = 10
        self.copy_bytes = None
        self.copy_polls = 1
        self.art_misses = 0
        self.art_length = None
        self.busy_attempts = 0
        mtimes = mtimes or {}
        sizes = sizes or {}
        self._add_dir(root_uri, root_path, sep)
        for rel in files:
            parts = rel.split('/')
            uri, path = root_uri, root_path
            for index, part in enumerate(parts):
                child_uri = uri.rstrip('/') + '/' + quote(part, safe='')
                child_path = path.rstrip(sep) + sep + part
                is_file = index == len(parts) - 1
                entry = {
                    'type': 'file' if is_file else 'dir',
                    'name': part,
                    'uri': child_uri,
                    'path': child_path,
                    'size': sizes.get(rel, 1000) if is_file else 0,
                    'modification_time': mtimes.get(rel, 1700000000) if is_file else 1700000000,
                }
                if entry not in self.listings[uri]:
                    self.listings[uri].append(entry)
                if not is_file:
                    self._add_dir(child_uri, child_path, sep)
                uri, path = child_uri, child_path

    def _add_dir(self, uri, path, sep):
        self.listings.setdefault(uri, [{
            'type': 'dir', 'name': '..', 'uri': uri.rsplit('/', 1)[0] + '/',
            'path': path + sep + '..', 'size': 0,
        }])

    def hold_copy_job(self, started_at):
        self.media[vlc.COPY_MEDIA] = {
            'uri': 'file:///elsewhere.mp3', 'output': '#std{}', 'mux': 'raw', 'dst': '/x',
            'options': [f'{vlc.COPY_SINCE_OPTION}={int(started_at)}'], 'polls': 0,
        }

    def get(self, url, auth=None, timeout=None, stream=False):
        self.urls.append(url)
        assert url.startswith(VLC_URL)
        path, _, query = url[len(VLC_URL):].partition('?')
        params = dict(parse_qsl(query, keep_blank_values=True))
        if auth != ('', PASSWORD):
            return _text_resp('<html>401 Unauthorized</html>', status=401)
        if path == '/requests/browse.json':
            uri = params['uri']
            if uri in self.broken or uri not in self.listings:
                return _text_resp(_error_page(f"cannot open directory `{uri}'."))
            return _json_resp({'element': self.listings[uri]})
        if path == '/requests/status.json':
            return self._status(params)
        if path == '/requests/playlist.json':
            return _json_resp({'type': 'node', 'name': '', 'children': [
                {'type': 'node', 'name': 'Playlist', 'id': '1', 'children': list(self.queue)},
                {'type': 'node', 'name': 'Media Library', 'id': '2', 'children': []},
            ]})
        if path == '/requests/vlm_cmd.xml':
            return self._vlm(params['command'])
        if path == '/requests/vlm.xml':
            return self._vlm_listing()
        if path == '/art':
            return self._art(params.get('item'))
        return _text_resp('<html>not found</html>', status=404)

    def _status(self, params):
        command = params.get('command')
        if command == 'in_enqueue':
            self.enqueued.append(params['input'])
            self.queue.append({
                'type': 'leaf', 'id': str(self.next_id), 'uri': params['input'],
                'name': params.get('name') or params['input'].rsplit('/', 1)[-1],
            })
            self.next_id += 1
        elif command == 'pl_delete':
            self.queue = [leaf for leaf in self.queue if leaf['id'] != params['id']]
        return _json_resp({'state': 'stopped', 'version': '3.0.24 Vetinari'})

    def _art(self, item):
        leaf = next((leaf for leaf in self.queue if leaf['id'] == item), None)
        if leaf is None or leaf['uri'] not in self.disk:
            return _BodyResp(b'Error\n', status=404)
        if self.art_misses:
            self.art_misses -= 1
            return _BodyResp(b'Error\n', status=404)
        return _BodyResp(self.disk[leaf['uri']], length=self.art_length)

    def _vlm(self, command):
        self.commands.append(command)
        error = ''
        if command.startswith('new '):
            match = _NEW.match(command)
            copy = _COPY_OUTPUT.match(match.group(3)) if match else None
            if not copy:
                error = 'Unknown VLM command'
            elif match.group(1) in self.media:
                error = f'new : {match.group(1)}: Name already in use'
                if self.busy_attempts:
                    self.busy_attempts -= 1
                    if not self.busy_attempts:
                        self.media.pop(match.group(1))
            else:
                self.media[match.group(1)] = {
                    'uri': match.group(2), 'output': match.group(3), 'mux': copy.group(1),
                    'dst': copy.group(2), 'options': [match.group(4)], 'polls': 0,
                }
        elif command.startswith('control '):
            name = command.split()[1]
            media = self.media.get(name)
            if media is None:
                error = f'{name}: media unknown'
            else:
                body = self.copy_bytes
                if body is None:
                    body = f"{media['mux']}:{media['uri']}".encode()
                self.disk[vlc.folder_uri(media['dst'])] = body
                media['polls'] = self.copy_polls
        elif command.startswith('del '):
            self.media.pop(command.split()[1], None)
        else:
            error = 'Unknown VLM command'
        return _text_resp(
            '<?xml version="1.0" encoding="utf-8" standalone="yes" ?>\n'
            f'<vlm><error>{error}</error>\n</vlm>'
        )

    def _vlm_listing(self):
        nodes = []
        for name, media in self.media.items():
            instance = ''
            if media['polls']:
                media['polls'] -= 1
                instance = '<instance name="default" state="playing" time="0" length="0" />'
            options = ''.join(f'<option>{o}</option>' for o in media['options'])
            nodes.append(
                f'<broadcast name="{name}" enabled="yes" loop="no"><output>x</output>'
                f'<inputs><input>{media["uri"]}</input></inputs><options>{options}</options>'
                f'<instances>{instance}</instances></broadcast>'
            )
        return _text_resp('<vlm><media>' + ''.join(nodes) + '</media></vlm>')


def _install(fake):
    return patch('tasks.mediaserver.vlc.requests.get', side_effect=fake.get)


ALBUM_FILES = [
    'Artist A/Album B (2020)/01 - Song One.mp3',
    'Artist A/Album B (2020)/02 - Song Two.flac',
    'Artist A/Album B (2020)/cover.jpg',
    'Artist A/Album B (2020)/._02 - Song Two.flac',
    'Artist A/.hidden/03 - Ghost.mp3',
    'Artist C/Album D/CD1/01 - Song Three.mp3',
]


class TestFolderUri:
    @pytest.mark.parametrize('folder, expected', [
        ('C:\\Users\\me\\Music', 'file:///C:/Users/me/Music'),
        ('C:/Users/me/Music/', 'file:///C:/Users/me/Music'),
        ('C:\\', 'file:///C:/'),
        ('D:', 'file:///D:/'),
        ('/home/me/Music', 'file:///home/me/Music'),
        ('/Users/me/Music/', 'file:///Users/me/Music'),
        ('/', 'file:///'),
        ('\\\\nas\\share\\Music', 'file://nas/share/Music'),
        ('file:///C:/Music/My%20Album%20(2020)', 'file:///C:/Music/My%20Album%20%282020%29'),
        ('file:///home/me/My Music', 'file:///home/me/My%20Music'),
        ('file://C:/Music', 'file:///C:/Music'),
        ('  /srv/music  ', 'file:///srv/music'),
        ('', ''),
        (None, ''),
    ])
    def test_every_platform_becomes_a_file_uri(self, folder, expected):
        assert vlc.folder_uri(folder) == expected

    def test_non_ascii_folder_is_percent_encoded_utf8(self):
        assert vlc.folder_uri('/home/me/M\u00fcsik') == 'file:///home/me/M%C3%BCsik'


class TestIds:
    def test_track_id_is_stable_hash_of_relative_path(self):
        rel = 'Artist A/Album B/01 - Song One.mp3'
        assert vlc.track_id(rel) == vlc.track_id(rel)
        assert len(vlc.track_id(rel)) == 32
        assert vlc.track_id(rel) != vlc.track_id('Artist A/Album B/02 - Song Two.mp3')

    def test_track_id_ignores_separator_and_unicode_normal_form(self):
        composed = 'Artist A/Caf\u00e9/01 - Song.mp3'
        decomposed = unicodedata.normalize('NFD', composed)
        assert decomposed != composed
        assert vlc.track_id(decomposed) == vlc.track_id(composed)
        assert vlc.track_id(composed.replace('/', '\\')) == vlc.track_id(composed)

    def test_same_relative_song_has_same_id_on_windows_and_linux_hosts(self):
        rel = 'Artist A/Album B (2020)/01 - Song One.mp3'
        _set_config('C:\\Music')
        with _install(FakeVlc([rel])):
            windows = mediaserver.get_all_songs()
        _set_config('/srv/media/music')
        linux_fake = FakeVlc([rel], root_uri='file:///srv/media/music', root_path='/srv/media/music', sep='/')
        with _install(linux_fake):
            linux = mediaserver.get_all_songs()
        assert windows[0]['Id'] == linux[0]['Id'] == vlc.track_id(rel)
        assert windows[0]['Path'] == 'C:\\Music\\Artist A\\Album B (2020)\\01 - Song One.mp3'
        assert linux[0]['Path'] == '/srv/media/music/Artist A/Album B (2020)/01 - Song One.mp3'

    def test_album_id_round_trips_unicode(self):
        parts = ('Artist A', 'Album \u00e9 (2020)')
        value = vlc.album_id(parts)
        assert re.fullmatch(r'd-[A-Za-z0-9_-]*', value)
        assert vlc.album_parts(value) == parts
        assert vlc.album_parts(vlc.album_id(())) == ()

    @pytest.mark.parametrize('value', ['d-' + 'Li4vZXRj', 'plain-id', '', None])
    def test_album_parts_refuses_escape_and_foreign_ids(self, value):
        with pytest.raises(ValueError):
            vlc.album_parts(value)


class TestMetadataFromLayout:
    @pytest.mark.parametrize('rel, expected', [
        ('Artist A/Album B (2020)/01 - Song One.mp3',
         ('Song One', 'Artist A', 'Artist A', 'Album B', 2020)),
        ('Artist A/Album B (2020)/Artist A - Album B - 07 - Song One.mp3',
         ('Song One', 'Artist A', 'Artist A', 'Album B', 2020)),
        ('Artist A/2019 - Album B/03. Song One.flac',
         ('Song One', 'Artist A', 'Artist A', 'Album B', 2019)),
        ('Artist A/Album B [2018]/1-04 Song One.flac',
         ('Song One', 'Artist A', 'Artist A', 'Album B', 2018)),
        ('Artist A/Album B/CD2/05 - Song One.mp3',
         ('Song One', 'Artist A', 'Artist A', 'Album B', None)),
        ('Mixed/Mixed/Artist E - Song One.mp3',
         ('Song One', 'Artist E', 'Mixed', 'Mixed', None)),
        ('Artist A - Album B/01 - Song One.mp3',
         ('Song One', 'Artist A', 'Artist A', 'Album B', None)),
        ('Loose/Song One.ogg', ('Song One', 'Unknown Artist', 'Unknown Artist', 'Loose', None)),
        ('Song One.mp3', ('Song One', 'Unknown Artist', 'Unknown Artist', None, None)),
        ('Artist A/Album B/99 Problems Song.mp3',
         ('99 Problems Song', 'Artist A', 'Artist A', 'Album B', None)),
    ])
    def test_describe(self, rel, expected):
        parts = tuple(rel.split('/'))
        assert vlc._describe(parts[:-1], parts[-1]) == expected

    def test_album_name_strips_year_and_keeps_disc(self):
        assert vlc._album_name(('Artist A', 'Album B (2020)')) == 'Album B'
        assert vlc._album_name(('Artist A', 'Album B', 'CD1')) == 'Album B (CD1)'
        assert vlc._album_name(('Artist A - Album B',)) == 'Album B'
        assert vlc._album_name(()) == 'Unknown Album'


class TestSettingsAndRequests:
    def test_config_fallback_when_no_server_bound(self):
        _set_config('/srv/music')
        assert vlc._settings() == {
            'url': VLC_URL, 'user': '', 'password': PASSWORD, 'root': 'file:///srv/music',
        }

    def test_bound_server_creds_win_over_config(self):
        _set_config('/srv/music')
        server = {
            'server_id': 's2', 'server_type': 'vlc', 'music_libraries': '',
            'creds': {'url': 'http://other:9090/', 'user': '', 'password': 'p2',
                      'music_folder': 'D:\\Songs'},
        }
        with context.use_server(server):
            settings = vlc._settings()
        assert settings == {'url': 'http://other:9090', 'user': '', 'password': 'p2', 'root': 'file:///D:/Songs'}

    def test_explicit_creds_are_used_without_context(self):
        _set_config('/srv/music')
        settings = vlc._settings({'url': 'http://x:1', 'password': 'q', 'music_folder': '/m'})
        assert settings['url'] == 'http://x:1' and settings['root'] == 'file:///m'

    def test_query_never_uses_plus_for_spaces(self):
        query = vlc._query({'command': 'new a broadcast', 'input': 'file:///C:/a b+c'})
        assert '+' not in query
        assert query == 'command=new%20a%20broadcast&input=file%3A%2F%2F%2FC%3A%2Fa%20b%2Bc'

    def test_requests_send_empty_user_and_password(self):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        with patch('tasks.mediaserver.vlc.requests.get', side_effect=fake.get) as mock_get:
            vlc._get_json(vlc._settings(), '/requests/status.json')
        assert mock_get.call_args.kwargs['auth'] == ('', PASSWORD)


class TestCatalogue:
    def test_get_all_songs_walks_folders_and_skips_hidden_and_non_audio(self):
        _set_config()
        with _install(FakeVlc(ALBUM_FILES)):
            songs = mediaserver.get_all_songs()
        assert sorted(s['RelativePath'] for s in songs) == [
            'Artist A/Album B (2020)/01 - Song One.mp3',
            'Artist A/Album B (2020)/02 - Song Two.flac',
            'Artist C/Album D/CD1/01 - Song Three.mp3',
        ]
        first = next(s for s in songs if s['Name'] == 'Song One')
        assert first['Id'] == vlc.track_id('Artist A/Album B (2020)/01 - Song One.mp3')
        assert first['AlbumArtist'] == 'Artist A'
        assert first['Album'] == 'Album B'
        assert first['Year'] == 2020
        assert first['FilePath'] == 'C:\\Music\\Artist A\\Album B (2020)\\01 - Song One.mp3'
        assert first['url'] == 'file:///C:/Music/Artist%20A/Album%20B%20%282020%29/01%20-%20Song%20One.mp3'

    def test_bare_windows_path_is_browsed_as_file_uri(self):
        _set_config('C:\\Music')
        fake = FakeVlc(ALBUM_FILES)
        with _install(fake):
            mediaserver.get_all_songs()
        assert fake.urls[0] == VLC_URL + '/requests/browse.json?uri=' + quote(ROOT_URI, safe='')

    def test_error_page_aborts_instead_of_returning_partial_catalogue(self):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        fake.broken.add(ROOT_URI + '/Artist%20C')
        with _install(fake), pytest.raises(vlc.VlcAPIError, match='cannot open directory'):
            mediaserver.get_all_songs()

    def test_recent_albums_newest_first_with_limit(self):
        _set_config()
        files = ['Old/Album X/01 - A.mp3', 'New/Album Y/01 - B.mp3', 'Mid/Album Z/01 - C.mp3']
        mtimes = {files[0]: 100, files[1]: 300, files[2]: 200}
        with _install(FakeVlc(files, mtimes=mtimes)):
            albums = mediaserver.get_recent_albums(0)
            limited = mediaserver.get_recent_albums(2)
        assert [a['Name'] for a in albums] == ['Album Y', 'Album Z', 'Album X']
        assert [a['Name'] for a in limited] == ['Album Y', 'Album Z']
        assert vlc.album_parts(albums[0]['Id']) == ('New', 'Album Y')

    def test_get_tracks_from_album_lists_one_folder_in_name_order(self):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        album = vlc.album_id(('Artist A', 'Album B (2020)'))
        with _install(fake):
            tracks = mediaserver.get_tracks_from_album(album)
        assert [t['Name'] for t in tracks] == ['Song One', 'Song Two']
        assert fake.urls == [
            VLC_URL + '/requests/browse.json?uri='
            + quote(ROOT_URI + '/Artist%20A/Album%20B%20%282020%29', safe='')
        ]

    def test_get_tracks_from_album_bad_id_or_error_returns_empty(self):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        with _install(fake):
            assert mediaserver.get_tracks_from_album('not-a-vlc-id') == []
            assert mediaserver.get_tracks_from_album(vlc.album_id(('Missing',))) == []

    def test_search_albums_matches_album_or_artist(self):
        _set_config()
        with _install(FakeVlc(ALBUM_FILES)):
            by_album = mediaserver.search_albums('album b')
            by_artist = mediaserver.search_albums('artist c')
        assert by_album == [{
            'id': vlc.album_id(('Artist A', 'Album B (2020)')), 'name': 'Album B',
            'artist': 'Artist A', 'year': 2020, 'track_count': 2,
        }]
        assert [a['name'] for a in by_artist] == ['Album D (CD1)']

    def test_list_libraries_is_empty_so_the_wizard_hides_the_picker(self):
        assert mediaserver.list_libraries(provider_type='vlc') == {'libraries': [], 'unsupported': False}


class TestConnection:
    def test_ok_after_downloading_the_smallest_sampled_track(self):
        _set_config()
        fake = FakeVlc(ALBUM_FILES, sizes={'Artist C/Album D/CD1/01 - Song Three.mp3': 10})
        with _install(fake):
            result = mediaserver.test_connection()
        assert result['ok'] is True
        assert result['sample_count'] == 3
        assert result['path_format'] == 'absolute'
        news = [c for c in fake.commands if c.startswith('new ')]
        assert len(news) == 1 and '/CD1/01%20-%20Song%20Three.mp3"' in news[0]
        assert fake.queue == [] and fake.media == {}

    def test_fails_when_vlc_cannot_serve_the_copy(self):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        fake.art_misses = 10 ** 6
        with _install(fake), patch.object(vlc, 'ART_TIMEOUT', 0):
            result = mediaserver.test_connection()
        assert result['ok'] is False
        assert 'Automatically preparse items' in result['error']

    def test_wrong_password_is_reported_as_auth_failure(self):
        _set_config()
        config.VLC_PASSWORD = 'wrong'
        with _install(FakeVlc(ALBUM_FILES)):
            result = mediaserver.test_connection()
        assert result['ok'] is False
        assert result['auth_failed'] is True

    def test_missing_folder_fails_with_vlc_message(self):
        _set_config('C:\\Nowhere')
        with _install(FakeVlc(ALBUM_FILES)):
            result = mediaserver.test_connection()
        assert result['ok'] is False
        assert 'cannot open directory' in result['error']

    def test_folder_without_audio_fails(self):
        _set_config()
        with _install(FakeVlc(['Docs/readme.txt'])):
            result = mediaserver.test_connection()
        assert result['ok'] is False
        assert 'no audio files' in result['error']

    def test_missing_settings_short_circuit(self):
        _set_config('')
        config.VLC_PASSWORD = ''
        with patch('tasks.mediaserver.vlc.requests.get') as mock_get:
            result = mediaserver.test_connection()
        assert result['ok'] is False
        assert 'password' in result['error'] and 'music folder' in result['error']
        mock_get.assert_not_called()

    def test_explicit_creds_test_a_server_that_is_not_configured(self):
        _set_config('')
        creds = {'url': VLC_URL, 'user': '', 'password': PASSWORD, 'music_folder': ROOT_PATH}
        with _install(FakeVlc(ALBUM_FILES)):
            result = mediaserver.test_connection(user_creds=creds, provider_type='vlc')
        assert result['ok'] is True


def _album_tracks(fake, parts=('Artist A', 'Album B (2020)')):
    with _install(fake):
        return mediaserver.get_tracks_from_album(vlc.album_id(parts))


SCRATCH_URI = 'file:///C:/Windows/Temp/Folder.jpg'


class TestDownload:
    def test_mp3_is_copied_raw_and_served_whole_by_art(self, tmp_path):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        track = _album_tracks(fake)[0]
        with _install(fake):
            path = vlc.download_track(str(tmp_path), track)
        assert path == str(tmp_path / f"{track['Id']}.mp3")
        with open(path, 'rb') as handle:
            assert handle.read() == f"raw:{track['url']}".encode()
        new = next(c for c in fake.commands if c.startswith('new '))
        assert new.startswith(f'new {vlc.COPY_MEDIA} broadcast enabled input "{track["url"]}" ')
        assert 'output "#std{access=file,mux=raw,dst=\\"C:/Windows/Temp/Folder.jpg\\"}"' in new
        assert f'control {vlc.COPY_MEDIA} play' in fake.commands
        assert fake.commands[-1] == f'del {vlc.COPY_MEDIA}'
        assert fake.enqueued == [SCRATCH_URI]
        assert fake.queue == [] and fake.media == {}
        assert os.listdir(tmp_path) == [os.path.basename(path)]

    def test_other_formats_are_remuxed_to_mkv(self, tmp_path):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        track = _album_tracks(fake)[1]
        with _install(fake):
            path = vlc.download_track(str(tmp_path), track)
        assert path.endswith('.mkv')
        with open(path, 'rb') as handle:
            assert handle.read() == f"mkv:{track['url']}".encode()

    def test_posix_vlc_copies_into_tmp(self, tmp_path):
        _set_config('/srv/music')
        fake = FakeVlc(ALBUM_FILES, root_uri='file:///srv/music', root_path='/srv/music', sep='/')
        track = _album_tracks(fake)[0]
        with _install(fake):
            assert vlc.download_track(str(tmp_path), track)
        new = next(c for c in fake.commands if c.startswith('new '))
        assert 'dst=\\"/tmp/Folder.jpg\\"' in new
        assert fake.enqueued == ['file:///tmp/Folder.jpg']

    def test_unc_music_folder_is_a_windows_vlc(self):
        assert vlc._scratch_path({'root': 'file://nas/share/Music'}) == 'C:/Windows/Temp/Folder.jpg'
        assert vlc._scratch_path({'root': 'file:///D:/Music'}) == 'C:/Windows/Temp/Folder.jpg'
        assert vlc._scratch_path({'root': 'file:///Users/me/Music'}) == '/tmp/Folder.jpg'

    def test_waits_while_another_download_holds_the_copy_job(self, tmp_path):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        track = _album_tracks(fake)[0]
        fake.hold_copy_job(time.time())
        fake.busy_attempts = 3
        with _install(fake):
            assert vlc.download_track(str(tmp_path), track)
        news = [c for c in fake.commands if c.startswith('new ')]
        assert len(news) == 4
        assert [c for c in fake.commands if c.startswith('del ')] == [f'del {vlc.COPY_MEDIA}']
        assert fake.commands.index(f'del {vlc.COPY_MEDIA}') > fake.commands.index(news[-1])

    def test_copy_job_of_a_dead_download_is_taken_over(self, tmp_path, caplog):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        track = _album_tracks(fake)[0]
        fake.hold_copy_job(time.time() - vlc.COPY_STALE_SECONDS - 1)
        with _install(fake), caplog.at_level('WARNING'):
            assert vlc.download_track(str(tmp_path), track)
        assert fake.commands[:2] == [fake.commands[0], f'del {vlc.COPY_MEDIA}']
        assert 'stopped download left behind' in caplog.text

    def test_waits_for_the_copy_and_for_the_preparsed_art(self, tmp_path):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        track = _album_tracks(fake)[0]
        fake.copy_polls = 3
        fake.art_misses = 2
        with _install(fake):
            assert vlc.download_track(str(tmp_path), track)
        assert len([u for u in fake.urls if '/art?' in u]) == 3

    def test_art_never_served_fails_and_cleans_up(self, tmp_path):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        track = _album_tracks(fake)[0]
        fake.art_misses = 10 ** 6
        with _install(fake), patch.object(vlc, 'ART_TIMEOUT', 0):
            assert vlc.download_track(str(tmp_path), track) is None
        assert fake.queue == [] and fake.media == {}
        assert os.listdir(tmp_path) == []

    def test_empty_copy_is_rejected(self, tmp_path):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        track = _album_tracks(fake)[0]
        fake.copy_bytes = b''
        with _install(fake):
            assert vlc.download_track(str(tmp_path), track) is None
        assert os.listdir(tmp_path) == []

    def test_short_body_is_rejected(self, tmp_path):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        track = _album_tracks(fake)[0]
        fake.art_length = 10 ** 6
        with _install(fake):
            assert vlc.download_track(str(tmp_path), track) is None
        assert os.listdir(tmp_path) == []

    def test_leftover_copies_leave_the_play_queue_and_user_items_stay(self, tmp_path):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        track = _album_tracks(fake)[0]
        fake.queue = [
            {'type': 'leaf', 'id': '1', 'name': f'{vlc.COPY_MEDIA}_dead', 'uri': SCRATCH_URI},
            {'type': 'leaf', 'id': '2', 'name': 'my song', 'uri': 'file:///C:/Music/x.mp3'},
        ]
        with _install(fake):
            assert vlc.download_track(str(tmp_path), track)
        assert [leaf['id'] for leaf in fake.queue] == ['2']

    def test_item_without_uri_is_not_downloaded(self, tmp_path):
        _set_config()
        with patch('tasks.mediaserver.vlc.requests.get') as mock_get:
            assert vlc.download_track(str(tmp_path), {'Id': 'abc', 'Name': 'x'}) is None
        mock_get.assert_not_called()

    def test_dispatcher_keeps_the_extension(self, tmp_path):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        tracks = _album_tracks(fake)
        with _install(fake):
            assert mediaserver.download_track(str(tmp_path), tracks[0]).endswith('.mp3')
            assert mediaserver.download_track(str(tmp_path), tracks[1]).endswith('.mkv')

    def test_download_uses_the_bound_server(self, tmp_path):
        server = {
            'server_id': 's2', 'server_type': 'vlc', 'music_libraries': '',
            'creds': {'url': VLC_URL, 'user': '', 'password': PASSWORD, 'music_folder': ROOT_PATH},
        }
        _set_config()
        config.VLC_URL = 'http://wrong:1'
        fake = FakeVlc(ALBUM_FILES)
        with context.use_server(server), _install(fake):
            tracks = mediaserver.get_tracks_from_album(vlc.album_id(('Artist A', 'Album B (2020)')))
            assert mediaserver.download_track(str(tmp_path), tracks[0]) is not None

    def test_vlm_error_is_raised(self):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        with _install(fake), pytest.raises(vlc.VlcAPIError, match='Unknown VLM command'):
            vlc._vlm(vlc._settings(), 'bogus')


class TestPlaylistsAndHistory:
    def test_instant_playlist_is_queued_in_vlc_in_order(self):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        tracks = _album_tracks(fake)
        ids = [tracks[1]['Id'], 'unknown-id', tracks[0]['Id']]
        with _install(fake), patch('tasks.mediaserver._to_server_ids', side_effect=lambda x: x):
            result = mediaserver.create_instant_playlist('Mix', ids)
        assert result == {'Id': vlc.PLAY_QUEUE_ID, 'Name': 'Mix_instant'}
        assert fake.enqueued == [tracks[1]['url'], tracks[0]['url']]
        enqueue_urls = [u for u in fake.urls if 'in_enqueue' in u]
        assert all('+' not in u for u in enqueue_urls)

    def test_instant_playlist_with_no_known_track_fails(self):
        _set_config()
        fake = FakeVlc(ALBUM_FILES)
        with _install(fake), patch('tasks.mediaserver._to_server_ids', side_effect=lambda x: x):
            assert mediaserver.create_instant_playlist('Mix', ['nope']) is None
        assert fake.enqueued == []

    def test_saved_playlists_are_not_stored_but_never_fail_the_caller(self):
        _set_config()
        with patch('tasks.mediaserver.vlc.requests.get') as mock_get, \
                patch('tasks.mediaserver._to_server_ids', side_effect=lambda x: x):
            assert mediaserver.create_playlist('Cluster_automatic', ['a', 'b']) == {
                'Id': '', 'Name': 'Cluster_automatic'}
            assert mediaserver.create_or_replace_playlist('Radio', ['a']) == {'Id': '', 'Name': 'Radio'}
            mediaserver.delete_automatic_playlists()
        mock_get.assert_not_called()

    def test_reads_vlc_has_no_answer_for_are_empty(self):
        _set_config()
        with patch('tasks.mediaserver.vlc.requests.get') as mock_get:
            assert mediaserver.get_all_playlists() == []
            assert mediaserver.get_playlist_by_name('x') is None
            assert mediaserver.get_playlist_track_ids('x') == []
            assert mediaserver.get_top_played_songs(10) == []
            assert mediaserver.get_last_played_time('abc') is None
            assert mediaserver.get_lyrics('abc') is None
        mock_get.assert_not_called()


class TestRegistryAndSetupRules:
    def test_vlc_user_is_optional_everything_else_required(self):
        from database import missing_required_creds

        complete = {'url': VLC_URL, 'user': '', 'password': PASSWORD, 'music_folder': '/m'}
        assert missing_required_creds('vlc', complete) == []
        assert missing_required_creds('vlc', {'url': VLC_URL}) == ['password', 'music_folder']

    def test_setup_validation_accepts_blank_vlc_user(self):
        from types import SimpleNamespace
        from tasks.setup_manager import SetupManager

        manager = SetupManager.__new__(SetupManager)
        valid = SimpleNamespace(
            MEDIASERVER_TYPE='vlc', VLC_URL=VLC_URL, VLC_USER='', VLC_PASSWORD=PASSWORD,
            VLC_MUSIC_FOLDER='C:\\Music',
        )
        assert manager._is_valid_server_config(valid) is True
        valid.VLC_MUSIC_FOLDER = ''
        assert manager._is_valid_server_config(valid) is False

    def test_registry_creds_and_default_name(self):
        from tasks.mediaserver import registry

        _set_config('/srv/music')
        assert registry.creds_from_config('vlc') == {
            'url': VLC_URL, 'user': '', 'password': PASSWORD, 'music_folder': '/srv/music',
        }
        assert registry._default_server_name('vlc') == 'VLC'

    def test_vlc_is_a_supported_type_everywhere(self):
        import app_music_servers
        import app_provider_migration
        from tasks import provider_probe

        assert 'vlc' in app_music_servers._SUPPORTED_TYPES
        assert 'vlc' in app_provider_migration._SUPPORTED_TARGETS
        assert provider_probe._normalize_provider_type('VLC') == 'vlc'
        assert mediaserver._provider('vlc') is vlc

    def test_vlc_api_error_is_a_media_server_error(self):
        from error import error_manager
        from error.error_dictionary import ERR_MEDIASERVER_UNREACHABLE

        assert error_manager.classify(vlc.VlcAPIError('boom'), 'fallback') == ERR_MEDIASERVER_UNREACHABLE

    def test_vlc_password_is_a_wizard_secret(self):
        import app_setup

        assert 'VLC_PASSWORD' in app_setup.SECRET_FIELDS
        assert {'VLC_URL', 'VLC_USER', 'VLC_PASSWORD', 'VLC_MUSIC_FOLDER'} <= set(app_setup.BASIC_SERVER_FIELDS)
