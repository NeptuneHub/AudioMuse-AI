# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Builds the seed catalogue (test/e2e/seed) from the CC0 songs of the AudioMuse-AI-DCLAP dataset list.

`fetch` reads the Wikimedia Commons list published with the DCLAP model, keeps
only the rows released under CC0 1.0 (no attribution owed, so the suite needs
no attribution page), skips names that describe loops or sound samples rather
than songs, downloads a fixed number of songs, cuts each to a short clip and
writes a tagged temporary library. `build` boots the real stack against that
library, runs the real analysis on it, exports the rows the analysis produced
(score, embedding, clap_embedding, one lyrics sentinel) into
test/e2e/seed/catalogue.json.gz, and writes a silent placeholder file into
test/e2e/library for every song. The export is anonymized: every song
becomes "Various Artists - song N" in the placeholder tags and in the
exported rows alike, no per-song source is kept, and a small seeded Gaussian
noise is mixed into the MusiCNN and CLAP vectors (norm preserved), so the
repository holds a realistic catalogue without carrying anyone's real song
data. The audio itself is never committed. The stack state of `build`
lives in AUDIOMUSE_E2E_STATE_DIR or
$HOME/audiomuse_e2e_seed_state (a native filesystem, initdb refuses /mnt/c),
so a second `build` reuses the analysis instead of repeating it. Run inside
WSL or Linux with the repo .venv:

    python -m test.e2e.seed_builder fetch --count 300
    python -m test.e2e.seed_builder build

Main Features:
* fetch: parse the licence list, keep CC0 only, download, cut to 60 s, tag
* build: analyze that library with the production pipeline and export the rows
* anonymize: "Various Artists - song N" everywhere, seeded noise on the vectors
* placeholders: one-second silent MP3s carrying the anonymized tags
"""

import argparse
import base64
import gzip
import json
import os
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from fractions import Fraction

import av
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from test.e2e.stack import paths, postgres  # noqa: E402
from test.e2e.stack.boot import Stack, require_linux  # noqa: E402
from test.e2e.stack.seed import CATALOGUE_PATH, SEED_DIR, Seed  # noqa: E402

LICENSE_LIST = 'https://raw.githubusercontent.com/NeptuneHub/AudioMuse-AI-DCLAP/main/dataset_license/WIKIMEDIA_SONGS_LICENSE.md'
CC0_URL = 'https://creativecommons.org/publicdomain/zero/1.0/'
DEFAULT_WORK_DIR = os.path.join(paths.CACHE_DIR, 'seed_src')
DEFAULT_STATE_DIR = os.path.join(os.path.expanduser('~'), 'audiomuse_e2e_seed_state')
DEFAULT_COUNT = 300
CLIP_START = 15.0
CLIP_LEN = 60.0
RATE = 44100
ALBUM_ARTIST = 'Various Artists'
ALBUM_PREFIX = 'Seed Commons'
FOLDER_PREFIX = 'SC'
ANONYMOUS_ALBUM_PREFIX = 'Seed'
NOISE_SCALE = 0.03
NOISE_SEED = 20260918
KEPT_AS_REAL_CLIPS = frozenset((
    'MC_Shadow_-_What_Im_sayin_streetmix.ogg',
    'U-Man_-_28_-_Pizza.ogg',
    'Frederic_Lardon_-_04_-_Un_week-end_avec_papy.ogg',
))
SEED_FOLDER = re.compile(r'^S[A-Z]\d\d - ')
UA = 'AudioMuse-AI-e2e-seed-builder/1.0 (https://github.com/NeptuneHub/AudioMuse-AI; one-off fixture build)'
REQUEST_SPACING_SECONDS = 2.0

WIKIMEDIA_ROW = re.compile(
    r'^\|\s*[^|]*\|\s*https://commons\.wikimedia\.org/wiki/File:(\S+)\s*\|\s*\[CC0\]\('
    + re.escape(CC0_URL) + r'\)'
)
NOT_A_SONG = re.compile(
    r'bauchamp|snare|hi.?hat|loop|sample|sfx|noise|drum|kick|click|tone|test|sound|bell|chord|'
    r'scale|note|pronunc|voice|speech|spoken|interview|reading|lecture|talk|pitch|\bhz\b|sine|'
    r'chime|whistle|bird|animal|jingle|ringtone|alarm',
    re.IGNORECASE,
)
NOT_FOR_A_FIXTURE = re.compile(r'fuck|shit|bitch|dick|porn|sex|nazi|kill', re.IGNORECASE)
SAFE = re.compile(r'[^A-Za-z0-9 ._()-]+')

SCORE_EXCLUDED = ('item_id', 'created_at', 'search_u')


def _safe_name(text):
    return SAFE.sub('_', text).strip()[:70] or 'untitled'


def _download(url, dest, retries=4):
    if os.path.isfile(dest) and os.path.getsize(dest) > 20000:
        return True
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(request, timeout=180) as response, open(dest + '.part', 'wb') as out:  # nosec B310 - https dataset sources
                shutil.copyfileobj(response, out)
            os.replace(dest + '.part', dest)
            time.sleep(REQUEST_SPACING_SECONDS)
            return os.path.getsize(dest) > 20000
        except Exception as exc:
            wait = 30 * attempt if '429' in str(exc) else 10 * attempt
            print(f'retry {attempt}: {url}: {exc} (waiting {wait}s)', flush=True)
            time.sleep(wait)
    return False


def _split_name(file_name):
    stem = os.path.splitext(file_name)[0].replace('_', ' ').strip()
    parts = [p.strip() for p in stem.split(' - ') if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return 'Wikimedia Commons', stem


def parse_wikimedia_cc0(text):
    entries = []
    seen = set()
    for line in text.splitlines():
        match = WIKIMEDIA_ROW.match(line)
        if not match:
            continue
        file_name = match.group(1)
        if NOT_A_SONG.search(file_name) or NOT_FOR_A_FIXTURE.search(file_name) or file_name in seen:
            continue
        seen.add(file_name)
        artist, title = _split_name(file_name)
        entries.append({
            'source': 'wikimedia', 'title': title, 'artist': artist, 'track_id': file_name,
            'page': 'https://commons.wikimedia.org/wiki/File:' + file_name, 'license': CC0_URL,
            'audio': 'https://commons.wikimedia.org/wiki/Special:FilePath/' + urllib.parse.quote(file_name),
        })
    return entries


def pick_round_robin(entries, count):
    by_artist = {}
    for entry in entries:
        by_artist.setdefault(entry['artist'], []).append(entry)
    queues = list(by_artist.values())
    picked = []
    while queues and len(picked) < count:
        queues = [q for q in queues if q]
        for queue in queues:
            if len(picked) >= count:
                break
            picked.append(queue.pop(0))
    return picked


def decode_stereo(path):
    container = av.open(path)
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format='s16', layout='stereo', rate=RATE)
    chunks = []
    for frame in container.decode(stream):
        for out in resampler.resample(frame):
            chunks.append(out.to_ndarray())
    for out in resampler.resample(None):
        chunks.append(out.to_ndarray())
    container.close()
    if not chunks:
        return np.zeros((0, 2), dtype=np.int16)
    return np.concatenate(chunks, axis=1).reshape(-1, 2)


def encode_mp3(samples, dest, tags, rate=RATE, layout='stereo', bit_rate=96000):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out = av.open(dest, 'w', options={'id3v2_version': '3', 'write_id3v1': '0'})
    for key, value in tags.items():
        out.metadata[key] = str(value)
    stream = out.add_stream('libmp3lame', rate=rate)
    stream.bit_rate = bit_rate
    stream.layout = layout
    stream.format = 's16p'
    stream.time_base = Fraction(1, rate)
    channels = 2 if layout == 'stereo' else 1
    pts = 0
    for start in range(0, len(samples), 1152):
        block = samples[start:start + 1152].reshape(-1, channels)
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(block.T), format='s16p', layout=layout)
        frame.sample_rate = rate
        frame.pts = pts
        frame.time_base = Fraction(1, rate)
        pts += len(block)
        for packet in stream.encode(frame):
            out.mux(packet)
    for packet in stream.encode(None):
        out.mux(packet)
    out.close()


def _tags(entry, album, album_artist, track_no, comment):
    return {
        'title': entry['title'], 'artist': entry['artist'], 'album_artist': album_artist, 'album': album,
        'date': '2024', 'track': f'{track_no}/10', 'genre': 'Seed', 'comment': comment,
    }


def fetch(work_dir, count):
    raw_dir = os.path.join(work_dir, 'raw')
    lib_dir = os.path.join(work_dir, 'library')
    os.makedirs(raw_dir, exist_ok=True)
    shutil.rmtree(lib_dir, ignore_errors=True)
    os.makedirs(lib_dir, exist_ok=True)
    dest = os.path.join(work_dir, 'wikimedia.md')
    if not os.path.isfile(dest):
        request = urllib.request.Request(LICENSE_LIST, headers={'User-Agent': UA})
        with urllib.request.urlopen(request, timeout=60) as response, open(dest, 'wb') as out:  # nosec B310 - pinned https raw URL
            shutil.copyfileobj(response, out)
    with open(dest, encoding='utf-8') as handle:
        pool = parse_wikimedia_cc0(handle.read())
    print(f'{len(pool)} CC0 songs in the list', flush=True)
    manifest = []
    kept = 0
    for entry in pick_round_robin(pool, len(pool)):
        if kept >= count:
            break
        key = f"{entry['source']}-{entry['track_id']}"
        raw = os.path.join(raw_dir, _safe_name(key) + os.path.splitext(entry['track_id'])[1].lower())
        if not _download(entry['audio'], raw):
            print(f'FAILED {key} {entry["audio"]}', flush=True)
            continue
        try:
            samples = decode_stereo(raw)
        except Exception as exc:
            print(f'UNDECODABLE {key}: {exc}', flush=True)
            continue
        total = len(samples) / RATE
        if total < 25:
            print(f'TOO SHORT {key}: {total:.1f}s', flush=True)
            continue
        start = CLIP_START if total >= CLIP_START + CLIP_LEN else 0.0
        length = min(CLIP_LEN, total - start)
        clip = samples[int(start * RATE):int((start + length) * RATE)]
        album_no = kept // 10 + 1
        album = f'{ALBUM_PREFIX} {album_no:02d}'
        folder = f'{FOLDER_PREFIX}{album_no:02d} - {ALBUM_ARTIST} - {album}'
        track_no = kept % 10 + 1
        file_name = f"{track_no:02d} - {_safe_name(entry['title'])}.mp3"
        encode_mp3(clip, os.path.join(lib_dir, folder, file_name), _tags(entry, album, ALBUM_ARTIST, track_no, 'AudioMuse-AI e2e seed source'))
        manifest.append({
            **entry, 'album': album, 'album_artist': ALBUM_ARTIST, 'folder': folder, 'file': file_name,
            'track_no': track_no, 'clip_start_s': start, 'clip_len_s': round(length, 2),
        })
        kept += 1
        if kept % 10 == 0:
            print(f'{kept} clips', flush=True)
    with open(os.path.join(work_dir, 'seed_sources.json'), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=1)
    print(f'fetched {len(manifest)} clips into {lib_dir}', flush=True)
    return manifest


def _track_no(entry):
    return int(entry.get('track_no') or entry['file'].split(' - ', 1)[0])


def _anonymous(index):
    album_no = (index - 1) // 10 + 1
    track_no = (index - 1) % 10 + 1
    album = f'{ANONYMOUS_ALBUM_PREFIX} {album_no:02d}'
    return {
        'title': f'song {index}', 'artist': ALBUM_ARTIST, 'album': album, 'album_artist': ALBUM_ARTIST,
        'track_no': track_no, 'folder': f'{FOLDER_PREFIX}{album_no:02d} - {ALBUM_ARTIST} - {album}',
        'file': f'{track_no:02d} - song {index}.mp3',
    }


def _rename(columns, row, names):
    mapping = {
        'title': names['title'], 'author': names['artist'], 'artist': names['artist'],
        'album': names['album'], 'album_artist': names['album_artist'],
    }
    return [mapping[col] if col in mapping and value is not None else value for col, value in zip(columns, row)]


def _noisy(blob, rng):
    vector = np.frombuffer(bytes(blob), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not norm:
        return bytes(blob)
    rms = norm / np.sqrt(vector.size)
    noisy = vector + rng.normal(0.0, NOISE_SCALE * rms, vector.size).astype(np.float32)
    noisy = noisy * (norm / float(np.linalg.norm(noisy)))
    return noisy.astype(np.float32).tobytes()


def _perturb(columns, row, rng):
    return [_noisy(value, rng) if col == 'embedding' and value is not None else value for col, value in zip(columns, row)]


def _encode_value(value):
    if isinstance(value, memoryview):
        value = bytes(value)
    if isinstance(value, (bytes, bytearray)):
        return {'b64': base64.b64encode(bytes(value)).decode('ascii')}
    return value


def _columns(conn, table, excluded=()):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s "
            "AND is_generated = 'NEVER' ORDER BY ordinal_position",
            (table,),
        )
        return [row[0] for row in cur.fetchall() if row[0] not in excluded]


def export_rows(dsn, sources, lib_dir):
    conn = postgres.connect(dsn)
    try:
        score_cols = _columns(conn, 'score', SCORE_EXCLUDED)
        embedding_cols = _columns(conn, 'embedding', ('item_id',))
        clap_cols = _columns(conn, 'clap_embedding', ('item_id',))
        with conn.cursor() as cur:
            cur.execute('SELECT item_id, provider_track_id, file_path FROM track_server_map WHERE file_path IS NOT NULL')
            maps = cur.fetchall()
            cur.execute('SELECT embedding, axis_vector FROM lyrics_embedding LIMIT 1')
            sentinel = cur.fetchone()
        by_relpath = {}
        for item_id, _provider_id, file_path in maps:
            rel = os.path.relpath(file_path, lib_dir).replace('\\', '/')
            by_relpath[rel] = item_id
        tracks = []
        missing = []
        merged = []
        exported = set()
        with conn.cursor() as cur:
            for entry in sources:
                if entry.get('track_id') in KEPT_AS_REAL_CLIPS:
                    continue
                rel = f"{entry['folder']}/{entry['file']}"
                item_id = by_relpath.get(rel)
                if item_id is None:
                    missing.append(rel)
                    continue
                if item_id in exported:
                    merged.append(rel)
                    continue
                exported.add(item_id)
                cur.execute(f"SELECT {', '.join(score_cols)} FROM score WHERE item_id = %s", (item_id,))
                score = cur.fetchone()
                cur.execute(f"SELECT {', '.join(embedding_cols)} FROM embedding WHERE item_id = %s", (item_id,))
                embedding = cur.fetchone()
                cur.execute(f"SELECT {', '.join(clap_cols)} FROM clap_embedding WHERE item_id = %s", (item_id,))
                clap = cur.fetchone()
                if score is None or embedding is None or clap is None:
                    missing.append(rel)
                    continue
                names = _anonymous(len(tracks) + 1)
                rng = np.random.default_rng(NOISE_SEED + len(tracks))
                tracks.append({
                    'item_id': item_id, **names,
                    'score': [_encode_value(v) for v in _rename(score_cols, score, names)],
                    'embedding': [_encode_value(v) for v in _perturb(embedding_cols, embedding, rng)],
                    'clap': [_encode_value(v) for v in _perturb(clap_cols, clap, rng)],
                })
    finally:
        conn.close()
    if missing:
        print(f'{len(missing)} source clips have no complete analysis rows and are left out: {missing[:5]}', flush=True)
    if merged:
        print(f'{len(merged)} source clips are the same recording as an earlier one and are left out: {merged}', flush=True)
    payload = {
        'version': 2,
        'generated': time.strftime('%Y-%m-%d'),
        'provenance': {
            'list': LICENSE_LIST,
            'license': CC0_URL,
            'note': 'every song is CC0 1.0 per the DCLAP dataset list; the audio was downloaded, cut, '
                    'analyzed once by the real pipeline and discarded; the rows are anonymized '
                    '("Various Artists - song N") and the vectors carry seeded Gaussian noise',
            'noise': {'scale': NOISE_SCALE, 'seed': NOISE_SEED},
        },
        'tables': {'score': score_cols, 'embedding': embedding_cols, 'clap_embedding': clap_cols},
        'lyrics_sentinel': {'embedding': _encode_value(sentinel[0]), 'axis_vector': _encode_value(sentinel[1])} if sentinel else None,
        'tracks': tracks,
        'counts': {'tracks': len(tracks), 'artists': len({t['artist'] for t in tracks}), 'albums': len({t['folder'] for t in tracks})},
    }
    return payload


def remove_previous_seed():
    for name in sorted(os.listdir(paths.LIBRARY_DIR)):
        path = os.path.join(paths.LIBRARY_DIR, name)
        if SEED_FOLDER.match(name) and os.path.isdir(path):
            shutil.rmtree(path)
    stale = os.path.join(SEED_DIR, 'ATTRIBUTION.md')
    if os.path.isfile(stale):
        os.unlink(stale)


def write_placeholders(tracks):
    return Seed({'tracks': tracks}).ensure_placeholders(paths.LIBRARY_DIR)


def build(work_dir):
    require_linux()
    lib_dir = os.path.join(work_dir, 'library')
    with open(os.path.join(work_dir, 'seed_sources.json'), encoding='utf-8') as handle:
        sources = json.load(handle)
    os.environ.setdefault(paths.STATE_DIR_ENV, DEFAULT_STATE_DIR)
    print(f'stack state: {os.environ[paths.STATE_DIR_ENV]}', flush=True)
    stack = Stack(music_folder=lib_dir, load_manifest=False, second_instance=False, load_seed=False)
    try:
        if not stack.boot():
            raise SystemExit('no database available for the seed build')
        expected = sum(1 for _root, _dirs, files in os.walk(lib_dir) for f in files if f.endswith('.mp3'))
        stack.navidrome.wait_scanned(expected, 300)
        task_id = stack.api.start_task('/api/analysis/start', {'num_recent_albums': 0, 'top_n_moods': 5})
        final = stack.api.wait_for_task(task_id, timeout=7200)
        print(f'analysis {task_id}: {final["state"]} {final["status_message"]}', flush=True)
        stack.api.wait_idle(300)
        payload = export_rows(stack.dsn, sources, lib_dir)
    finally:
        stack.shutdown()
    os.makedirs(SEED_DIR, exist_ok=True)
    remove_previous_seed()
    with gzip.open(CATALOGUE_PATH, 'wt', encoding='utf-8') as handle:
        json.dump(payload, handle, separators=(',', ':'))
    write_placeholders(payload['tracks'])
    print(f"seed catalogue: {payload['counts']} -> {CATALOGUE_PATH} ({os.path.getsize(CATALOGUE_PATH) // 1024} KB)", flush=True)


def main(argv):
    parser = argparse.ArgumentParser(description='build the e2e seed catalogue')
    sub = parser.add_subparsers(dest='command', required=True)
    fetch_parser = sub.add_parser('fetch')
    fetch_parser.add_argument('--work-dir', default=DEFAULT_WORK_DIR)
    fetch_parser.add_argument('--count', type=int, default=DEFAULT_COUNT)
    build_parser = sub.add_parser('build')
    build_parser.add_argument('--work-dir', default=DEFAULT_WORK_DIR)
    args = parser.parse_args(argv)
    if args.command == 'fetch':
        fetch(args.work_dir, args.count)
    else:
        build(args.work_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
