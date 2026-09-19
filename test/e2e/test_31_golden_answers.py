# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Golden answers: the exact songs the ranking features return for fixed seeds.

The seeded catalogue carries committed embeddings, so for a seeded song the
neighbours, the path, the similar artists, the text-search hits and the
hyperbolic neighbours are fully determined by the app's code and index
builds (the IVF training is seeded). This module records those answers once
into seed/golden.json and fails when any of them changes, so a regression in
a distance, an index build, a filter or a serializer shows up as "expected
song A, got song B" instead of passing as "some result came back". The stored
score row of each seed is compared value by value as well. A brute-force
cosine ranking computed from the database embeddings cross-checks the
similar-song answer independently of the recorded file: the first answer is
the true nearest song and every answer sits within the brute-force top 25
(the IVF search is approximate and quantized, so it is not exhaustive).

Re-record after an intended change:

    AUDIOMUSE_E2E_RECORD_GOLDEN=1 bash test/e2e/run_local.sh --no-browser -k golden

Main Features:
* similar songs, song path, similar artists, text search, hyperbolic
  neighbours for three fixed seeded songs match the recorded answers
* the score row of each seed is returned value by value
* the top similar songs agree with a brute-force cosine ranking
* recording mode rewrites seed/golden.json from the live answers
"""

import json
import os
import urllib.parse

import numpy as np
import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, provider_ids_for, rows
from test.e2e.stack.seed import SEED_DIR

pytestmark = pytest.mark.e2e

GOLDEN_PATH = os.path.join(SEED_DIR, 'golden.json')
RECORD_ENV = 'AUDIOMUSE_E2E_RECORD_GOLDEN'
SEED_POSITIONS = (0, 1, 2)
NEIGHBOURS = 5
CLAP_QUERIES = ('upbeat chiptune video game music', 'slow orchestral strings', 'electronic dance beat')
SCORE_FIELDS = ('title', 'author', 'tempo', 'key', 'scale', 'energy', 'mood_vector', 'other_features')
ORACLE_SLACK = 25
REMEDY = f'if the change is intended, re-record with {RECORD_ENV}=1 bash test/e2e/run_local.sh --no-browser -k golden'


def _who(row):
    return row.get('author') or row.get('artist') or ''


def _seeded_pairs(stack):
    return {(t['title'], t['artist']) for t in stack.seed.tracks}


def _seeded_only(stack, entries, limit):
    allowed = _seeded_pairs(stack)
    kept = []
    for row in entries:
        pair = (row.get('title') or '', _who(row))
        if pair in allowed:
            kept.append(list(pair))
        if len(kept) >= limit:
            break
    return kept


def _round(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [_round(v) for v in value]
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    return value


@pytest.fixture
def seeds(stack, db, analyzed_library):
    ordered = sorted(stack.seed.tracks, key=lambda t: t['item_id'])
    step = max(1, len(ordered) // len(SEED_POSITIONS))
    chosen = []
    for position in SEED_POSITIONS:
        track = ordered[position * step]
        pids = provider_ids_for(db, track['item_id'])
        assert pids, track['title']
        chosen.append({'item_id': track['item_id'], 'pid': pids[0], 'title': track['title'], 'artist': track['artist']})
    return chosen


def _collect(stack, api, seeds):
    answers = {}
    for index, seed in enumerate(seeds):
        label = f"{index}:{seed['artist']} - {seed['title']}"
        similar = api.json('GET', f"/api/similar_tracks?item_id={seed['pid']}&n=20")
        assert_no_fp_ids(similar)
        answers[f'similar {label}'] = _seeded_only(stack, similar, NEIGHBOURS)
        score = api.json('GET', f"/external/get_score?id={seed['pid']}")
        assert_no_fp_ids(score)
        answers[f'score {label}'] = _round({k: score.get(k) for k in SCORE_FIELDS})
        for mode in ('similar', 'roots', 'niche'):
            body = api.json('POST', '/api/hyperbolic/similar', json={'item_id': seed['pid'], 'limit': 20, 'mode': mode})
            assert_no_fp_ids(body)
            answers[f'hyperbolic {mode} {label}'] = _seeded_only(stack, body['results'], NEIGHBOURS)
    start, end = seeds[0], seeds[-1]
    path = api.json('GET', f"/api/find_path?start_song_id={start['pid']}&end_song_id={end['pid']}&max_steps=6")
    assert_no_fp_ids(path)
    answers['path'] = _seeded_only(stack, path['path'], 8)
    artists = api.json('GET', '/api/similar_artists?artist=' + urllib.parse.quote(seeds[0]['artist']) + '&n=10')
    assert_no_fp_ids(artists)
    answers['artists'] = [(r.get('artist') or r.get('name')) for r in artists][:NEIGHBOURS]
    for query in CLAP_QUERIES:
        body = api.json('POST', '/api/clap/search', json={'query': query, 'limit': 20}, timeout=300)
        assert_no_fp_ids(body)
        answers[f'clap {query}'] = _seeded_only(stack, body['results'], NEIGHBOURS)
    return answers


def test_recorded_answers(stack, api, seeds):
    api.wait_idle(180)
    api.json('POST', '/api/hyperbolic/warmup', timeout=300)
    api.json('POST', '/api/clap/warmup', timeout=300)
    answers = _collect(stack, api, seeds)
    assert all(answers[k] for k in answers), {k: v for k, v in answers.items() if not v}
    if os.environ.get(RECORD_ENV, '').strip():
        with open(GOLDEN_PATH, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(answers, handle, indent=1, sort_keys=True, ensure_ascii=False)
            handle.write('\n')
    assert os.path.isfile(GOLDEN_PATH), f'{GOLDEN_PATH} is missing; {REMEDY}'
    with open(GOLDEN_PATH, encoding='utf-8') as handle:
        golden = json.load(handle)
    assert set(golden) == set(answers), (sorted(set(golden) ^ set(answers)), REMEDY)
    for key in sorted(golden):
        assert answers[key] == golden[key], f'{key}: expected {golden[key]}, got {answers[key]}; {REMEDY}'


def _embeddings(db):
    table = {}
    for item_id, blob in rows(db, 'SELECT item_id, embedding FROM embedding WHERE embedding IS NOT NULL'):
        vector = np.frombuffer(bytes(blob), dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        table[item_id] = vector / norm if norm else vector
    return table


def test_similar_songs_agree_with_a_brute_force_cosine_ranking(stack, api, db, seeds):
    vectors = _embeddings(db)
    assert len(vectors) == stack.catalogue_rows
    for seed in seeds:
        query = vectors[seed['item_id']]
        scored = sorted(
            ((float(np.dot(query, vector)), item_id) for item_id, vector in vectors.items() if item_id != seed['item_id']),
            reverse=True,
        )
        rank_of = {item_id: rank for rank, (_score, item_id) in enumerate(scored, start=1)}
        answered = api.json('GET', f"/api/similar_tracks?item_id={seed['pid']}&n={NEIGHBOURS}")
        assert answered, seed
        ranks = []
        for row in answered:
            item = rows(db, 'SELECT item_id FROM track_server_map WHERE provider_track_id = %s', (row['item_id'],))
            assert item, row
            ranks.append(rank_of[item[0][0]])
        assert ranks[0] == 1, (seed['title'], ranks, [t for _s, t in scored[:3]])
        assert max(ranks) <= ORACLE_SLACK, (seed['title'], ranks)
