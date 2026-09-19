# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Song analysis (ALGORITHM.md section 1): the shared run and what it persisted.

The session fixture already ran the analysis of every album through the real
queue; this module asserts on the task outcome it captured and on every table
the analysis fills, then runs the analysis again to show it is idempotent and
refuses to start twice.

Main Features:
* the main task ended SUCCESS with every album completed and every
  analyzable file counted
* score, embedding, clap_embedding, lyrics_embedding, chromaprint and
  track_server_map hold the expected rows with the expected columns filled
* the broken file is recorded as an analysis exclusion, not a failed album
* a second run analyzes nothing new, and a start during a run answers 409
"""

import re

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, rows, scalar

pytestmark = pytest.mark.e2e

ITEM_ID_RE = re.compile(r'^fp_[1-9][0-9a-f]{50}$')


def test_task_outcome(stack, library, analyzed_library, golden):
    final = analyzed_library.final
    assert_no_fp_ids(final)
    assert final['state'] == 'SUCCESS'
    assert final['task_type_from_db'] == 'main_analysis'
    details = final['details']
    if analyzed_library.fresh:
        golden.check('fresh analysis outcome', {
            'state': final['state'], 'status_message': final['status_message'],
            'tracks_analyzed': details.get('tracks_analyzed'), 'albums_completed': details.get('albums_completed'),
            'failed_albums': details.get('failed_albums', 0),
        })
    albums = library.counts['album_folders'] + len(stack.seed.albums)
    if analyzed_library.fresh:
        assert details['tracks_analyzed'] == library.counts['analyzable_files'], details
        assert details['albums_completed'] == library.counts['album_folders'], details
    else:
        assert details['tracks_analyzed'] == 0, details
    assert details.get('failed_albums', 0) == 0, details
    assert final['status_message'].startswith(f'Albums {albums}/{albums}'), final['status_message']


def test_score_rows(stack, db, library, analyzed_library):
    catalogue = rows(db, 'SELECT item_id, title, author, album_artist, tempo, key, scale, energy, mood_vector, other_features, duration FROM score')
    assert len(catalogue) == stack.catalogue_rows
    for item_id, title, author, album_artist, tempo, key, scale, energy, mood_vector, other_features, duration in catalogue:
        assert ITEM_ID_RE.match(item_id), item_id
        assert title and author and album_artist
        assert tempo and key and scale and energy is not None and duration, (title, tempo, key, scale, energy, duration)
        assert len(mood_vector.split(',')) == 5, mood_vector
        assert other_features and 'danceable:' in other_features, other_features
    titles = {r[1] for r in catalogue}
    for key in library.clip_keys():
        assert library.track(key).title in titles, key


def test_embedding_tables(stack, db, library, analyzed_library):
    expected = stack.catalogue_rows
    filled = rows(db, 'SELECT count(*), count(embedding), count(neural_fingerprint), count(poincare_embedding), count(hyperbolic_radius) FROM embedding')[0]
    assert all(n == expected for n in filled), filled
    assert scalar(db, 'SELECT max(length(embedding)) FROM embedding') == 800
    assert scalar(db, 'SELECT count(*) FROM clap_embedding') == expected
    assert scalar(db, 'SELECT min(length(embedding)) FROM clap_embedding') == 2048
    assert scalar(db, 'SELECT count(*) FROM lyrics_embedding WHERE embedding IS NOT NULL AND axis_vector IS NOT NULL') == expected


def test_chromaprint_and_mappings(stack, db, library, analyzed_library):
    assert scalar(db, 'SELECT count(fingerprint) FROM chromaprint') == library.counts['analyzable_files']
    maps = rows(db, 'SELECT item_id, provider_track_id, match_tier, file_path FROM track_server_map')
    assert len(maps) == stack.analyzable_files
    assert len({m[0] for m in maps}) == stack.catalogue_rows
    assert all(m[2] == 'fingerprint' and m[3] for m in maps), maps
    provider_ids = {m[1] for m in maps}
    for key in library.tracks:
        if key in library.unanalyzable:
            assert library.pid(key) not in provider_ids
        else:
            assert library.pid(key) in provider_ids, key
    assert scalar(db, 'SELECT count(*) FROM artist_server_map') >= library.counts['clip_artists']


def test_broken_file_is_excluded(stack, db, library, analyzed_library):
    exclusions = rows(db, 'SELECT provider_item_id, title, reason_code FROM analysis_exclusions')
    assert len(exclusions) == len(library.unanalyzable), exclusions
    for key in library.unanalyzable:
        match = next((e for e in exclusions if e[0] == library.pid(key)), None)
        assert match is not None, (key, exclusions)
        assert match[1] == library.track(key).title
        assert match[2]


def test_rerun_is_idempotent_and_refuses_concurrency(stack, api, db, library, analyzed_library):
    api.wait_idle(180)
    before = {
        t: scalar(db, f'SELECT count(*) FROM {t}')
        for t in ('score', 'embedding', 'clap_embedding', 'lyrics_embedding', 'track_server_map', 'analysis_exclusions')
    }
    task_id = api.start_task('/api/analysis/start', {'num_recent_albums': 0, 'top_n_moods': 5})
    refused = api.post('/api/analysis/start', json={'num_recent_albums': 0})
    assert refused.status_code == 409, refused.text
    assert refused.json().get('task_id') == task_id, refused.text
    final = api.wait_for_task(task_id, timeout=600)
    api.wait_idle(120)
    assert final['details']['tracks_analyzed'] == 0, final['details']
    after = {t: scalar(db, f'SELECT count(*) FROM {t}') for t in before}
    assert after == before
    assert api.status(analyzed_library.task_id) is None or api.status(analyzed_library.task_id)['state'] != 'SUCCESS'
