# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The seed catalogue: analysis rows of a few hundred real songs, without their audio.

test/e2e/seed/catalogue.json.gz holds the anonymized rows the real analysis
produced once for CC0 songs from the AudioMuse-AI-DCLAP dataset (see
test/e2e/seed_builder.py): score features, the MusiCNN embedding, the CLAP
embedding and the neural fingerprint. Neither the audio nor any per-song file
is committed: Navidrome only assigns an id to a file it finds on disk, so the
harness writes a one-second silent MP3 per seeded song into the library at
boot, tagged from the catalogue ("Various Artists - song N"); those folders
are git-ignored. Applying the seed inserts the rows and binds them to the ids
Navidrome gave those files, after which the app treats the songs as already
analyzed and every index build, clustering run and search works at a
realistic library size.

Main Features:
* load() reads the seed or returns an empty one when the file is absent
* ensure_placeholders() writes the missing silent files and removes seed
  folders the catalogue no longer names
* apply() inserts score, embedding, clap_embedding, lyrics_embedding,
  track_server_map and artist_server_map rows bound to the server's song ids,
  idempotently (every insert is ON CONFLICT DO NOTHING)
"""

import base64
import gzip
import json
import os
import re
import shutil
from fractions import Fraction

import av
import numpy as np

from .errors import StackError
from .paths import E2E_DIR

SEED_DIR = os.path.join(E2E_DIR, 'seed')
CATALOGUE_PATH = os.path.join(SEED_DIR, 'catalogue.json.gz')
SEED_FOLDER = re.compile(r'^S[A-Z]\d\d - ')
PLACEHOLDER_RATE = 22050
PLACEHOLDER_SECONDS = 1.0
PLACEHOLDER_BIT_RATE = 16000
PLACEHOLDER_FRAME = 1152
PLACEHOLDER_COMMENT = 'AudioMuse-AI e2e seed placeholder: silent stand-in written at boot, the analysis rows come from test/e2e/seed/catalogue.json.gz'


def write_silent_mp3(dest, tags):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    samples = np.zeros(int(PLACEHOLDER_SECONDS * PLACEHOLDER_RATE), dtype=np.int16)
    out = av.open(dest, 'w', options={'id3v2_version': '3', 'write_id3v1': '0'})
    for key, value in tags.items():
        out.metadata[key] = str(value)
    stream = out.add_stream('libmp3lame', rate=PLACEHOLDER_RATE)
    stream.bit_rate = PLACEHOLDER_BIT_RATE
    stream.layout = 'mono'
    stream.format = 's16p'
    stream.time_base = Fraction(1, PLACEHOLDER_RATE)
    pts = 0
    for start in range(0, len(samples), PLACEHOLDER_FRAME):
        block = samples[start:start + PLACEHOLDER_FRAME].reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(block), format='s16p', layout='mono')
        frame.sample_rate = PLACEHOLDER_RATE
        frame.pts = pts
        frame.time_base = Fraction(1, PLACEHOLDER_RATE)
        pts += block.shape[1]
        for packet in stream.encode(frame):
            out.mux(packet)
    for packet in stream.encode(None):
        out.mux(packet)
    out.close()


def _decode(value):
    if isinstance(value, dict) and 'b64' in value:
        return base64.b64decode(value['b64'])
    return value


class Seed:
    def __init__(self, payload):
        self.payload = payload or {}
        self.tracks = list(self.payload.get('tracks') or [])
        self.tables = self.payload.get('tables') or {}
        self.sentinel = self.payload.get('lyrics_sentinel')
        self.count = len(self.tracks)
        self.applied = 0

    @property
    def relpaths(self):
        return [f"{t['folder']}/{t['file']}" for t in self.tracks]

    @property
    def artists(self):
        return {t['artist'] for t in self.tracks}

    @property
    def albums(self):
        return {t['folder'] for t in self.tracks}

    def ensure_placeholders(self, library_dir):
        wanted = self.albums
        for name in sorted(os.listdir(library_dir)):
            path = os.path.join(library_dir, name)
            if SEED_FOLDER.match(name) and name not in wanted and os.path.isdir(path):
                shutil.rmtree(path)
        written = 0
        for track in self.tracks:
            dest = os.path.join(library_dir, track['folder'], track['file'])
            if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                continue
            write_silent_mp3(dest, {
                'title': track['title'], 'artist': track['artist'], 'album_artist': track['album_artist'],
                'album': track['album'], 'date': '2024', 'track': f"{track['track_no']}/10", 'genre': 'Seed',
                'comment': PLACEHOLDER_COMMENT,
            })
            written += 1
        return written

    def apply(self, conn, server_id, songs):
        if not self.tracks:
            return 0
        by_path = {}
        for song in songs:
            path = (song.get('path') or '').replace('\\', '/')
            if path:
                by_path[path] = song
        score_cols = list(self.tables['score'])
        embedding_cols = list(self.tables['embedding'])
        clap_cols = list(self.tables['clap_embedding'])
        inserted = 0
        with conn.cursor() as cur:
            for track in self.tracks:
                relpath = f"{track['folder']}/{track['file']}"
                song = next((s for p, s in by_path.items() if p.endswith(relpath)), None)
                if song is None:
                    raise StackError(f'seed placeholder not served by Navidrome: {relpath}')
                item_id = track['item_id']
                cur.execute(
                    f"INSERT INTO score (item_id, {', '.join(score_cols)}) VALUES (%s{', %s' * len(score_cols)}) "
                    "ON CONFLICT (item_id) DO NOTHING",
                    [item_id] + [_decode(v) for v in track['score']],
                )
                inserted += cur.rowcount
                cur.execute(
                    f"INSERT INTO embedding (item_id, {', '.join(embedding_cols)}) VALUES (%s{', %s' * len(embedding_cols)}) "
                    "ON CONFLICT (item_id) DO NOTHING",
                    [item_id] + [_decode(v) for v in track['embedding']],
                )
                cur.execute(
                    f"INSERT INTO clap_embedding (item_id, {', '.join(clap_cols)}) VALUES (%s{', %s' * len(clap_cols)}) "
                    "ON CONFLICT (item_id) DO NOTHING",
                    [item_id] + [_decode(v) for v in track['clap']],
                )
                if self.sentinel:
                    cur.execute(
                        "INSERT INTO lyrics_embedding (item_id, embedding, axis_vector) VALUES (%s, %s, %s) "
                        "ON CONFLICT (item_id) DO NOTHING",
                        (item_id, _decode(self.sentinel['embedding']), _decode(self.sentinel['axis_vector'])),
                    )
                cur.execute(
                    "INSERT INTO track_server_map (item_id, server_id, provider_track_id, match_tier, file_path) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (item_id, server_id, str(song['id']), 'fingerprint', song.get('path')),
                )
                if song.get('artistId'):
                    cur.execute(
                        "INSERT INTO artist_server_map (artist_name, server_id, provider_artist_id) "
                        "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        (track['artist'], server_id, str(song['artistId'])),
                    )
        conn.commit()
        self.applied = inserted
        return inserted


def empty():
    return Seed({})


def load(path=CATALOGUE_PATH):
    if not os.path.isfile(path):
        return empty()
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise StackError(f'cannot read the seed catalogue {path}: {exc}') from exc
    seed = Seed(payload)
    for table in ('score', 'embedding', 'clap_embedding'):
        if table not in seed.tables:
            raise StackError(f'seed catalogue lacks the column list for {table}')
    return seed
