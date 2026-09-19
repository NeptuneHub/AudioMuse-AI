# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The committed fixture library as the tests see it.

test/e2e/library/manifest.json describes every audio file Navidrome serves:
the clips, the two derived copies that exercise catalogue identity, the broken
file that exercises analysis exclusions, the lyric sidecars and the counts the
tests assert against. Tests never hard-code a count or a file name; they ask
this object, so the library can change without touching the catalogue.

Main Features:
* load() parses the manifest into Track objects addressed by short keys (A01)
* bind_provider_ids maps each key to its Navidrome song id by path suffix
* remove_file / restore_file move a file out of and back into the served folder
  for the cleaning scenario, keeping the committed tree intact
"""

import json
import os
import shutil

from .errors import StackError
from .paths import LIBRARY_DIR, MANIFEST_PATH


class Track:
    def __init__(self, key, album, data):
        self.key = key
        self.album_key = album['key']
        self.folder = album['folder']
        self.album = album['album']
        self.album_artist = album['album_artist']
        self.year = album.get('year')
        self.file = data['file']
        self.title = data['title']
        self.artist = data['artist']
        self.track = data.get('track')
        self.duration_s = data.get('duration_s')
        self.role = data.get('role', 'clip')
        self.lyrics = data.get('lyrics')
        self.character = data.get('character', '')
        self.provider_id = None

    @property
    def relpath(self):
        return f'{self.folder}/{self.file}'


class Library:
    def __init__(self, manifest, root):
        self.root = root
        self.manifest = manifest
        self.tracks = {}
        self.albums = manifest['albums']
        for album in self.albums:
            for data in album['tracks']:
                track = Track(data['key'], album, data)
                if track.key in self.tracks:
                    raise StackError(f'manifest: duplicate track key {track.key}')
                self.tracks[track.key] = track
        self.copies = manifest.get('copies', {})
        self.unanalyzable = list(manifest.get('unanalyzable', []))
        self.lyrics = manifest.get('lyrics', {})
        self.clap_probe = manifest.get('clap_probe', {})
        self.counts = manifest['counts']
        self.holding_dir = None
        self._held = {}

    def keys(self, role=None):
        return [k for k, t in self.tracks.items() if role is None or t.role == role]

    def track(self, key):
        try:
            return self.tracks[key]
        except KeyError:
            raise StackError(f'manifest has no track {key}') from None

    def path(self, key):
        return os.path.join(self.root, self.track(key).folder, self.track(key).file)

    def relpath(self, key):
        return self.track(key).relpath

    def files(self):
        return [self.path(k) for k in self.tracks]

    def clip_keys(self):
        return [k for k in self.keys('clip')]

    def album_keys(self):
        return [album['key'] for album in self.albums]

    def tracks_in_album(self, album_key):
        return [t for t in self.tracks.values() if t.album_key == album_key]

    def lyric_keys(self):
        return list(self.lyrics)

    def bind_provider_ids(self, songs):
        by_suffix = {}
        for song in songs:
            path = (song.get('path') or '').replace('\\', '/')
            if path:
                by_suffix[path] = song.get('id')
        unbound = []
        for track in self.tracks.values():
            match = next((sid for p, sid in by_suffix.items() if p.endswith(track.relpath)), None)
            if match is None:
                unbound.append(track.relpath)
            track.provider_id = match
        if unbound:
            raise StackError(
                f'Navidrome did not report {len(unbound)} manifest files: {unbound}; '
                f'server paths were {sorted(by_suffix)[:5]}...'
            )

    def pid(self, key):
        provider_id = self.track(key).provider_id
        if provider_id is None:
            raise StackError(f'track {key} has no Navidrome id yet (bind_provider_ids not run)')
        return provider_id

    def remove_file(self, key):
        if self.holding_dir is None:
            raise StackError('library.holding_dir is not set')
        src = self.path(key)
        os.makedirs(self.holding_dir, exist_ok=True)
        dest = os.path.join(self.holding_dir, f'{key}__{self.track(key).file}')
        shutil.move(src, dest)
        self._held[key] = dest
        return dest

    def restore_file(self, key):
        held = self._held.pop(key, None)
        if held is None:
            return None
        dest = self.path(key)
        shutil.move(held, dest)
        return dest

    def restore_all(self):
        for key in list(self._held):
            self.restore_file(key)


def load(root=LIBRARY_DIR, manifest_path=MANIFEST_PATH):
    try:
        with open(manifest_path, encoding='utf-8') as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise StackError(f'cannot read the fixture manifest {manifest_path}: {exc}') from exc
    library = Library(manifest, root)
    missing = [t.relpath for t in library.tracks.values() if not os.path.isfile(library.path(t.key))]
    if missing:
        raise StackError(f'fixture files listed in the manifest are missing: {missing}')
    return library
