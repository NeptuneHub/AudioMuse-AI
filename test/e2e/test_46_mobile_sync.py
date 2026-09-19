# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The mobile sync endpoint: a manifest of fingerprints and paged full payloads.

A client first fetches the manifest (provider id plus a read-time fingerprint
over the analysis columns), then pages through the payload for the tracks it
lacks. Every id is a provider id, the manifest covers the whole catalogue, and
the fingerprint is exactly the md5 prefix the server computes.

Main Features:
* fields=index lists every catalogue row once with a stable fingerprint
* payload pages carry the analysis fields and stop when has_more is false
* ids= filters the payload to the requested provider ids
"""

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, item_id_of, scalar

pytestmark = pytest.mark.e2e

FP_SQL = (
    "substr(md5(coalesce(s.mood_vector,'')||'|'||coalesce(s.energy::text,'')||'|'||"
    "coalesce(s.other_features,'')||'|'||coalesce(s.tempo::text,'')||'|'||"
    "coalesce(s.key,'')||'|'||coalesce(s.scale,'')), 1, 16)"
)


def test_manifest_covers_the_catalogue(stack, api, db, library, analyzed_library):
    body = api.json('GET', '/api/sync?fields=index&limit=500')
    assert_no_fp_ids(body)
    assert body['provider_type'] == 'navidrome'
    assert body['total_tracks'] == stack.catalogue_rows
    assert body['has_more'] is False
    assert body['next_page'] is None
    tracks = body['tracks']
    assert len(tracks) == stack.catalogue_rows
    assert len({t['id'] for t in tracks}) == len(tracks)
    pid = library.pid('B02')
    entry = next(t for t in tracks if t['id'] == pid)
    expected = scalar(db, f'SELECT {FP_SQL} FROM score s WHERE s.item_id = %s', (item_id_of(db, pid),))
    assert entry['fp'] == expected


def test_payload_pages_through_everything(stack, api, library, analyzed_library):
    seen = set()
    page = 1
    while True:
        body = api.json('GET', f'/api/sync?limit=50&page={page}')
        assert_no_fp_ids(body)
        tracks = body['tracks']
        assert 0 < len(tracks) <= 50, body
        for track in tracks:
            assert track['id']
            assert track.get('title')
            seen.add(track['id'])
        if not body['has_more']:
            assert body['next_page'] is None
            break
        assert body['next_page'] == page + 1
        page = body['next_page']
        assert page < 50
    assert len(seen) == stack.catalogue_rows


def test_id_filter_and_embedding_toggle(stack, api, library, analyzed_library):
    wanted = [library.pid('A03'), library.pid('B01')]
    body = api.json('GET', f'/api/sync?ids={",".join(wanted)}&limit=50')
    assert {t['id'] for t in body['tracks']} == set(wanted)
    with_embeddings = api.json('GET', f'/api/sync?ids={wanted[0]}&include_embeddings=true')
    without = api.json('GET', f'/api/sync?ids={wanted[0]}&include_embeddings=false')
    assert with_embeddings['tracks']
    assert without['tracks']
    assert 'embedding' in with_embeddings['tracks'][0]
    assert 'embedding' not in without['tracks'][0]
