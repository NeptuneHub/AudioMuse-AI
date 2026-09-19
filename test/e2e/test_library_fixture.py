# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Contract of the committed fixture library, checked without booting anything.

The catalogue tests derive every expected count from manifest.json, so the
manifest and the audio files on disk must agree: tags, durations, the pairwise
duration spacing that keeps distinct clips from ever being merged as one
recording, the shapes of the two derived copies, the deliberately undecodable
file, and the lyric sidecars passing the text gates the lyrics stage applies.

Main Features:
* counts in the manifest match the files it lists
* every playable file is 44.1 kHz stereo MP3 with the manifest's tags
* clip durations are at least 2 s apart; copies have the intended durations
* the broken file is tagged yet yields no decodable audio
* each .lrc sidecar clears the minimum-length and English-confidence gates
* the harness can write a tagged one-second silent stand-in for every seeded
  song from the catalogue alone (none is committed), and the seed is CC0
"""

import os
import re

import av
import pytest
from langdetect import detect_langs

from test.e2e.stack import library, seed

pytestmark = pytest.mark.e2e

CC0 = 'https://creativecommons.org/publicdomain/zero/1.0/'
MIN_LYRIC_CHARS = 250
MIN_ENGLISH_CONFIDENCE = 0.70
MIN_CLIP_GAP_SECONDS = 2.0
_LRC_STAMP = re.compile(r'\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]')


@pytest.fixture(scope='module')
def lib():
    return library.load()


def _open(path):
    container = av.open(path)
    stream = container.streams.audio[0]
    duration = float(container.duration) / av.time_base
    meta = dict(container.metadata)
    info = (duration, stream.rate, stream.channels, meta)
    container.close()
    return info


def test_manifest_counts_match_files(lib):
    counts = lib.counts
    assert counts['files'] == len(lib.tracks)
    assert counts['clips'] == len(lib.clip_keys())
    assert counts['analyzable_files'] == counts['files'] - len(lib.unanalyzable)
    merged = sum(1 for c in lib.copies.values() if c['expect'] == 'merged')
    assert counts['catalogue_rows'] == counts['analyzable_files'] - merged
    assert counts['lyric_clips'] == len(lib.lyrics)
    assert counts['album_folders'] == len(lib.albums)
    clip_artists = {lib.track(k).album_artist for k in lib.clip_keys()}
    assert counts['clip_artists'] == len(clip_artists)
    assert counts['clips'] == 15


def test_every_playable_file_has_tags_and_duration(lib):
    for key, track in lib.tracks.items():
        if track.role == 'broken':
            continue
        duration, rate, channels, meta = _open(lib.path(key))
        assert rate == 44100, key
        assert channels == 2, key
        assert meta.get('title') == track.title, (key, meta)
        assert meta.get('artist') == track.artist, (key, meta)
        assert meta.get('album') == track.album, (key, meta)
        assert meta.get('album_artist') == track.album_artist, (key, meta)
        assert meta.get('date') == str(track.year), (key, meta)
        assert meta.get('track', '').startswith(str(track.track)), (key, meta)
        assert meta.get('genre'), (key, meta)
        assert abs(duration - track.duration_s) < 0.5, (key, duration, track.duration_s)


def test_clip_durations_are_pairwise_apart(lib):
    durations = sorted(lib.track(k).duration_s for k in lib.clip_keys())
    gaps = [b - a for a, b in zip(durations, durations[1:])]
    assert all(gap >= MIN_CLIP_GAP_SECONDS - 0.01 for gap in gaps), gaps


def test_copies_have_the_intended_shapes(lib):
    for copy_key, spec in lib.copies.items():
        original = lib.track(spec['of'])
        copy = lib.track(copy_key)
        assert copy.title.startswith(original.title.rstrip('.')), (copy_key, copy.title)
        assert copy.artist == original.artist
        assert copy.album != original.album
        copy_duration = _open(lib.path(copy_key))[0]
        original_duration = _open(lib.path(spec['of']))[0]
        if spec['expect'] == 'merged':
            assert abs(copy_duration - original_duration) < 0.1, (copy_key, copy_duration, original_duration)
            with open(lib.path(copy_key), 'rb') as a, open(lib.path(spec['of']), 'rb') as b:
                assert a.read()[-4096:] == b.read()[-4096:], 'stream copy must keep the audio bytes'
        else:
            extra = float(spec['extra_seconds'])
            assert abs(copy_duration - (original_duration + extra)) < 0.3, (copy_key, copy_duration)
            assert extra >= 7.0


def test_broken_file_is_tagged_yet_undecodable(lib):
    for key in lib.unanalyzable:
        path = lib.path(key)
        assert os.path.getsize(path) < 8192
        with open(path, 'rb') as handle:
            assert handle.read(3) == b'ID3'
        decoded = 0
        try:
            container = av.open(path)
            for frame in container.decode(audio=0):
                decoded += frame.samples
            container.close()
        except Exception:
            decoded = 0
        assert decoded < 44100, f'{key} decoded {decoded} samples, it must not be playable'


def test_lyric_sidecars_clear_the_text_gates(lib):
    for key, spec in lib.lyrics.items():
        track = lib.track(key)
        assert track.role == 'vocal', (key, 'only real sung songs may carry lyrics')
        assert spec['probe_phrase'], key
        if not spec.get('file'):
            assert track.lyrics is None, (key, 'a song transcribed by the ASR has no sidecar')
            continue
        assert track.lyrics == spec['file'], key
        sidecar = os.path.join(lib.root, track.folder, spec['file'])
        assert os.path.isfile(sidecar), sidecar
        assert os.path.splitext(spec['file'])[0] == os.path.splitext(track.file)[0], key
        with open(sidecar, encoding='utf-8') as handle:
            raw = handle.read()
        text = '\n'.join(_LRC_STAMP.sub('', line).strip() for line in raw.splitlines())
        assert len(text) >= MIN_LYRIC_CHARS, (key, len(text))
        english = max((lang.prob for lang in detect_langs(text) if lang.lang == 'en'), default=0.0)
        assert english >= MIN_ENGLISH_CONFIDENCE, (key, english)
        assert spec['probe_phrase'].split(',')[0].lower() in text.lower(), key
    for key in lib.copies:
        assert lib.track(key).lyrics is None, 'copies must not carry lyrics'
    assert lib.track(lib.copies['F01']['of']).lyrics is None, 'the merged original must not carry lyrics'


def test_probe_keys_exist(lib):
    assert lib.clap_probe['expect_key'] in lib.tracks
    probe = lib.manifest['recording_probe']
    assert probe['key'] in lib.tracks
    assert probe['start_s'] + probe['length_s'] <= lib.track(probe['key']).duration_s


def test_seed_placeholders_are_generated_from_the_catalogue(tmp_path):
    catalogue = seed.load()
    assert catalogue.count >= 100, 'the seed catalogue is part of the committed fixture'
    stale = tmp_path / 'SZ99 - Various Artists - Seed 99'
    stale.mkdir()
    (stale / '01 - old.mp3').write_bytes(b'x')
    assert catalogue.ensure_placeholders(str(tmp_path)) == catalogue.count
    assert catalogue.ensure_placeholders(str(tmp_path)) == 0, 'a second call must find every file in place'
    assert not stale.exists(), 'a seed folder the catalogue no longer names must go'
    for track in catalogue.tracks:
        path = os.path.join(str(tmp_path), track['folder'], track['file'])
        assert os.path.isfile(path), path
        duration, rate, channels, meta = _open(path)
        assert 0.5 <= duration <= 2.0, (track['file'], duration)
        assert meta.get('title') == track['title'], (track['file'], meta)
        assert meta.get('artist') == track['artist'], (track['file'], meta)
        assert meta.get('album') == track['album'], (track['file'], meta)
        assert meta.get('album_artist') == track['album_artist'], (track['file'], meta)
        assert track['artist'] == 'Various Artists', (track['file'], track)
        assert track['title'].startswith('song '), (track['file'], track)
        assert 'source' not in track, 'seed rows must stay anonymous'
        assert 'license' not in track, 'seed rows must stay anonymous'
    assert len({t['item_id'] for t in catalogue.tracks}) == catalogue.count
    assert catalogue.payload['provenance']['license'] == CC0
    assert catalogue.sentinel
    assert catalogue.tables['score']
