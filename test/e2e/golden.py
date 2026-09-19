# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Recorded exact answers for every deterministic API call of the suite.

A module asks `golden.check(key, payload)` for each answer it wants pinned.
The payload is normalized first so that it is stable across machines: every
provider id is replaced by the song, album, artist, playlist or server it
names, generated ids and timestamps are dropped, tag strings such as
"jazz:0.627,rock:0.515" become label-to-score maps, the 2D map coordinates are
dropped (the UMAP projection is not seeded, so they change at every
analysis), unique test names are masked and floats are rounded. In compare
mode every name, label, count and order must equal the one stored in
seed/golden/<module>.json, while numbers may differ by the few thousandths
that a different CPU or thread count puts into the models' floating-point
output for the real clips; a mismatch fails with the differing paths and
their expected and actual values. In record mode
(AUDIOMUSE_E2E_RECORD_GOLDEN=1) the answers are written instead, merged
into the existing file so that a partial run never erases other keys.

Main Features:
* normalize: ids to names, volatile keys dropped, floats rounded
* Golden.check: assert against, or record into, the module's golden file
* one JSON file per test module under seed/golden/
"""

import json
import os
import re

from test.e2e.stack.seed import SEED_DIR

GOLDEN_DIR = os.path.join(SEED_DIR, 'golden')
RECORD_ENV = 'AUDIOMUSE_E2E_RECORD_GOLDEN'
REMEDY = f'if the change is intended, re-record with {RECORD_ENV}=1 bash test/e2e/run_local.sh --no-browser'
DROP_KEYS = frozenset((
    'timestamp', 'last_run', 'next_run', 'restore_log',
    'restore_log_name', 'size_bytes', 'elapsed', 'elapsed_seconds', 'memory_mb',
    'task_id', 'job_id', 'sweep_task_id', 'server_id', 'default_id', 'creds',
    'pid', 'run_dir', 'search_u', 'generated', 'build', 'build_id', 'loaded_at', 'age_seconds', 'uptime',
    'version', 'app_version', 'restore_pid', 'playlist_id', 'session_id', 'run_id', 'cache_age', 'log',
    'details_log', 'progress_log', 'processed_at', 'duration_ms', 'took_ms', 'query_time', 'time',
))
INT_ID_KEYS = frozenset(('id', 'anchor_id', 'radio_id', 'row_id'))
STRING_ONLY_DROP_KEYS = frozenset(('path', 'file_path', 'url', 'base_url', 'filename'))
PROJECTION_KEYS = frozenset(('umap_x', 'umap_y'))
LIBRARY_MARKER = '/test/e2e/library/'
DATETIME_RE = re.compile(r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}')
VECTOR_KEYS = frozenset(('embedding', 'embedding_vector', 'poincare_embedding', 'embedding_2d', 'centroid', 'vector'))
UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
HEX32_RE = re.compile(r'^[0-9a-f]{32}$')
E2E_NAME_RE = re.compile(r'^e2e-[A-Za-z0-9_.-]*[0-9a-f]{8}$')
FLOAT_PLACES = 3
VECTOR_PLACES = 2
MAX_DIFF_LINES = 25
TOLERANCE = 0.0025
VECTOR_TOLERANCE = 0.025
TAG_KEYS = frozenset(('mood_vector', 'other_features'))


class Resolver:
    def __init__(self):
        self.names = {}

    @classmethod
    def build(cls, stack):
        resolver = cls()
        for client in (stack.subsonic, stack.subsonic2):
            if client is None:
                continue
            for song in client.all_songs():
                resolver.names[str(song['id'])] = f"track:{song.get('title', '')} | {song.get('artist', '')}"
            for album in client.albums():
                resolver.names[str(album['id'])] = f"album:{album.get('name', '')} | {album.get('artist', '')}"
            for playlist in client.playlists():
                resolver.names[str(playlist['id'])] = f"playlist:{playlist.get('name', '')}"
        return resolver

    def add_artists(self, pairs):
        for provider_artist_id, name in pairs:
            self.names[str(provider_artist_id)] = f'artist:{name}'

    def add_servers(self, pairs):
        for server_id, name in pairs:
            self.names[str(server_id)] = f'server:{name}'

    def token(self, value):
        named = self.names.get(value)
        if named is not None:
            return named
        if UUID_RE.match(value):
            return '<uuid>'
        if HEX32_RE.match(value):
            return '<hex32>'
        if E2E_NAME_RE.match(value):
            return '<e2e-name>'
        if DATETIME_RE.match(value):
            return '<datetime>'
        if LIBRARY_MARKER in value:
            return '<library>/' + value.split(LIBRARY_MARKER, 1)[1]
        if value.startswith('/') and value.count('/') >= 2:
            return '<path>/' + value.rsplit('/', 1)[1]
        return value


def _round(value, places):
    if value != value or value in (float('inf'), float('-inf')):
        return str(value)
    return round(value, places)


def parse_tags(text):
    tags = {}
    for part in text.split(','):
        label, _sep, score = part.rpartition(':')
        try:
            tags[label.strip()] = round(float(score), FLOAT_PLACES)
        except ValueError:
            return text
    return tags if tags and '' not in tags else text


def normalize(value, resolver, places=FLOAT_PLACES):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in TAG_KEYS and isinstance(item, str) and item:
                out[key] = parse_tags(item)
                continue
            if key in DROP_KEYS or key.endswith('_at') or key.endswith('_2d') or key in PROJECTION_KEYS:
                continue
            if key in STRING_ONLY_DROP_KEYS and isinstance(item, str):
                continue
            if key in INT_ID_KEYS and isinstance(item, int) and not isinstance(item, bool):
                continue
            out[key] = normalize(item, resolver, VECTOR_PLACES if key in VECTOR_KEYS else places)
        return out
    if isinstance(value, (list, tuple)):
        inner = places
        if len(value) > 8 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            inner = VECTOR_PLACES
        return [normalize(v, resolver, inner) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        return _round(value, places)
    if isinstance(value, str):
        return resolver.token(value)
    return str(value)


def _short(value):
    return json.dumps(value, ensure_ascii=False)[:160]


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def differences(expected, actual, where='$', tolerance=TOLERANCE):
    if _is_number(expected) and _is_number(actual):
        if isinstance(expected, int) and isinstance(actual, int):
            close = expected == actual
        else:
            close = abs(float(expected) - float(actual)) <= tolerance
        return [] if close else [f'{where}: expected {_short(expected)}, got {_short(actual)}']
    if type(expected) is not type(actual):
        return [f'{where}: expected {_short(expected)}, got {_short(actual)}']
    if isinstance(expected, dict):
        lines = []
        for name in sorted(set(expected) | set(actual)):
            if name not in actual:
                lines.append(f'{where}.{name}: missing, expected {_short(expected[name])}')
            elif name not in expected:
                lines.append(f'{where}.{name}: unexpected {_short(actual[name])}')
            else:
                inner = VECTOR_TOLERANCE if name in VECTOR_KEYS else tolerance
                lines.extend(differences(expected[name], actual[name], f'{where}.{name}', inner))
        return lines
    if isinstance(expected, list):
        lines = []
        if len(expected) != len(actual):
            lines.append(f'{where}: expected {len(expected)} items, got {len(actual)}')
        inner = tolerance
        if len(expected) > 8 and all(_is_number(v) for v in expected):
            inner = VECTOR_TOLERANCE
        for index, (left, right) in enumerate(zip(expected, actual)):
            lines.extend(differences(left, right, f'{where}[{index}]', inner))
        return lines
    if expected != actual:
        return [f'{where}: expected {_short(expected)}, got {_short(actual)}']
    return []


class Golden:
    def __init__(self, module_name, resolver, record):
        self.path = os.path.join(GOLDEN_DIR, f'{module_name}.json')
        self.resolver = resolver
        self.record = record
        self.recorded = {}
        self.expected = None
        if os.path.isfile(self.path):
            with open(self.path, encoding='utf-8') as handle:
                self.expected = json.load(handle)

    def check(self, key, payload):
        value = normalize(payload, self.resolver)
        if self.record:
            self.recorded[key] = value
            return value
        assert self.expected is not None, f'{self.path} is missing; {REMEDY}'
        assert key in self.expected, f'{key!r} was never recorded in {os.path.basename(self.path)}; {REMEDY}'
        expected = self.expected[key]
        lines = differences(expected, value)
        if lines:
            shown = '\n'.join(lines[:MAX_DIFF_LINES])
            more = f'\n... and {len(lines) - MAX_DIFF_LINES} more' if len(lines) > MAX_DIFF_LINES else ''
            raise AssertionError(f'{key!r} changed in {len(lines)} place(s):\n{shown}{more}\n{REMEDY}')
        return value

    def flush(self):
        if not self.record or not self.recorded:
            return
        merged = dict(self.expected or {})
        merged.update(self.recorded)
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(self.path, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(merged, handle, indent=1, sort_keys=True, ensure_ascii=False)
            handle.write('\n')
