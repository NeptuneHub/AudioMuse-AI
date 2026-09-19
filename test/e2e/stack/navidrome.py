# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The real Navidrome the end-to-end stack runs, and a Subsonic client for assertions.

Navidrome is the media server the app talks to through tasks/mediaserver/navidrome.py.
A pinned release binary serves the committed fixture library, the admin user is
created by ND_DEVAUTOCREATEADMINPASSWORD on the empty database, and readiness
means the startup scan has listed every manifest file. SubsonicClient signs
requests exactly like the app does (u, p=enc:<hex>, v, c, f=json) but under its
own client name, and is used only to seed state (plays) and to assert what the
app wrote (playlists), never to stand in for app behaviour.

Main Features:
* NAVIDROME_ASSET pins the version and sha256 per architecture
* NavidromeServer.start / wait_ready / wait_scanned / rescan / stop
* SubsonicClient: ping, scan status, songs, albums, playlists, scrobble, star
"""

import os
import time

import requests

from .binaries import Asset
from .env import NAVIDROME_ADMIN_PASSWORD, NAVIDROME_ADMIN_USER
from .errors import StackError
from .processes import ManagedProcess, wait_until

NAVIDROME_VERSION = '0.64.0'
NAVIDROME_ASSET = Asset(
    name='navidrome',
    version=NAVIDROME_VERSION,
    member='navidrome',
    urls={
        'x86_64': (
            'https://github.com/navidrome/navidrome/releases/download/'
            f'v{NAVIDROME_VERSION}/navidrome_{NAVIDROME_VERSION}_linux_amd64.tar.gz'
        ),
        'aarch64': (
            'https://github.com/navidrome/navidrome/releases/download/'
            f'v{NAVIDROME_VERSION}/navidrome_{NAVIDROME_VERSION}_linux_arm64.tar.gz'
        ),
    },
    sha256={
        'x86_64': 'efd94d11253234035b5d3cf284da917111736e72daeeda4f45a58251c60553ac',
        'aarch64': '74acac22979b5a3587f7ab0fd5616bf17deec0b0ad0a06fdc4c4f04deeb85c74',
    },
)

DEFAULT_PORT = 4533
SUBSONIC_API_VERSION = '1.16.1'
CLIENT_NAME = 'AudioMuse-E2E'
PAGE_SIZE = 500


def _as_list(value):
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    return []


class SubsonicClient:
    def __init__(self, base_url, user=NAVIDROME_ADMIN_USER, password=NAVIDROME_ADMIN_PASSWORD, timeout=30):
        self.base_url = base_url.rstrip('/')
        self.user = user
        self.password = password
        self.timeout = timeout
        self.session = requests.Session()

    def auth_params(self):
        return {
            'u': self.user,
            'p': 'enc:' + self.password.encode('utf-8').hex(),
            'v': SUBSONIC_API_VERSION,
            'c': CLIENT_NAME,
            'f': 'json',
        }

    def call(self, endpoint, **params):
        merged = {**self.auth_params(), **params}
        response = self.session.get(
            f'{self.base_url}/rest/{endpoint}.view', params=merged, timeout=self.timeout
        )
        response.raise_for_status()
        body = response.json().get('subsonic-response', {})
        if body.get('status') != 'ok':
            error = body.get('error') or {}
            raise StackError(
                f'Subsonic {endpoint} failed: code {error.get("code")}: {error.get("message")}'
            )
        return body

    def ping(self):
        return self.call('ping')

    def server_version(self):
        return str(self.ping().get('serverVersion', ''))

    def scan_status(self):
        return self.call('getScanStatus').get('scanStatus', {})

    def start_scan(self, full=False):
        params = {'fullScan': 'true'} if full else {}
        return self.call('startScan', **params).get('scanStatus', {})

    def all_songs(self):
        songs = []
        offset = 0
        while True:
            body = self.call(
                'search3', query='', songCount=PAGE_SIZE, songOffset=offset,
                artistCount=0, albumCount=0,
            )
            page = _as_list((body.get('searchResult3') or {}).get('song'))
            songs.extend(page)
            if len(page) < PAGE_SIZE:
                return songs
            offset += len(page)

    def song_count(self):
        return len(self.all_songs())

    def song(self, song_id):
        return self.call('getSong', id=song_id).get('song', {})

    def albums(self, size=PAGE_SIZE):
        body = self.call('getAlbumList2', type='alphabeticalByName', size=size)
        return _as_list((body.get('albumList2') or {}).get('album'))

    def album_songs(self, album_id):
        body = self.call('getAlbum', id=album_id)
        return _as_list((body.get('album') or {}).get('song'))

    def playlists(self):
        body = self.call('getPlaylists')
        return _as_list((body.get('playlists') or {}).get('playlist'))

    def playlist(self, playlist_id):
        body = self.call('getPlaylist', id=playlist_id).get('playlist') or {}
        body['entries'] = _as_list(body.get('entry'))
        return body

    def playlist_by_name(self, name):
        return next((p for p in self.playlists() if p.get('name') == name), None)

    def playlist_entry_ids(self, playlist_id):
        return [str(e.get('id')) for e in self.playlist(playlist_id)['entries'] if e.get('id')]

    def delete_playlist(self, playlist_id):
        self.call('deletePlaylist', id=playlist_id)

    def delete_playlists_named(self, predicate):
        removed = 0
        for playlist in self.playlists():
            if predicate(playlist.get('name') or ''):
                self.delete_playlist(playlist['id'])
                removed += 1
        return removed

    def scrobble(self, song_id, submission=True):
        self.call('scrobble', id=song_id, submission='true' if submission else 'false')

    def star(self, song_id):
        self.call('star', id=song_id)

    def set_rating(self, song_id, rating):
        self.call('setRating', id=song_id, rating=int(rating))


class NavidromeServer:
    def __init__(self, executable, data_root, music_folder, port, log_path, parent_env, instance='navidrome'):
        self.port = port
        self.instance = instance
        self.base_url = f'http://127.0.0.1:{port}'
        self.music_folder = music_folder
        data_dir = os.path.join(data_root, instance, 'data')
        cache_dir = os.path.join(data_root, instance, 'cache')
        env = {
            'PATH': parent_env.get('PATH', ''),
            'HOME': parent_env.get('HOME', data_dir),
            'TZ': 'UTC',
            'ND_ADDRESS': '127.0.0.1',
            'ND_PORT': str(port),
            'ND_MUSICFOLDER': music_folder,
            'ND_DATAFOLDER': data_dir,
            'ND_CACHEFOLDER': cache_dir,
            'ND_DEVAUTOCREATEADMINPASSWORD': NAVIDROME_ADMIN_PASSWORD,
            'ND_SCANNER_SCANONSTARTUP': 'true',
            'ND_SCANNER_SCHEDULE': '0',
            'ND_SUBSONIC_DEFAULTREPORTREALPATH': 'true',
            'ND_ENABLEINSIGHTSCOLLECTOR': 'false',
            'ND_ENABLEEXTERNALSERVICES': 'false',
            'ND_LASTFM_ENABLED': 'false',
            'ND_LISTENBRAINZ_ENABLED': 'false',
            'ND_ENABLESHARING': 'false',
            'ND_SESSIONTIMEOUT': '24h',
            'ND_LOGLEVEL': 'info',
        }
        self.process = ManagedProcess(instance, [executable], env, data_dir, log_path)
        self.client = SubsonicClient(self.base_url)

    def start(self):
        self.process.start()

    def stop(self):
        self.process.stop(10)

    def running(self):
        return self.process.running()

    def _dead(self):
        return not self.process.running()

    def wait_ready(self, timeout=60):
        wait_until(
            lambda: requests.get(self.base_url + '/ping', timeout=3).status_code == 200,
            timeout, 'Navidrome /ping', dead=self._dead, detail=self.process.describe,
        )
        wait_until(
            lambda: self.client.ping().get('status') == 'ok',
            30, 'Navidrome admin login (ND_DEVAUTOCREATEADMINPASSWORD)',
            dead=self._dead, detail=self.process.describe,
        )

    def wait_scanned(self, expected_count, timeout=120):
        def done():
            status = self.client.scan_status()
            if str(status.get('scanning')).lower() == 'true':
                return False
            return expected_count is None or self.client.song_count() == expected_count

        wait_until(
            done, timeout, f'Navidrome scan listing {expected_count} songs',
            dead=self._dead,
            detail=lambda: f'scan status {self.client.scan_status()}, songs {self.client.song_count()}. {self.process.describe()}',
        )

    def rescan(self, expected_count, timeout=120):
        self.client.start_scan()
        time.sleep(1.0)
        self.wait_scanned(expected_count, timeout)
