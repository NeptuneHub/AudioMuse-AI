# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Text search (ALGORITHM.md section 12): DCLAP text-to-audio retrieval on the library.

The CLAP text model embeds a free-text query and the audio index ranks the
clips; on this library a "solo piano" query must rank a piano clip near the
top. Warmup, concept steering and the top-queries list are exercised as well.

Main Features:
* warmup loads the text model and reports it active
* a semantic query ranks a piano clip in the top results, sorted by similarity
* missing query and non-numeric limit are 400
* concept steering answers or explains why it is unavailable
"""

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids

pytestmark = pytest.mark.e2e

PIANO_KEYS = ('A01', 'A02', 'A03', 'F01', 'F02')


def test_warmup(stack, api, analyzed_library):
    body = api.json('POST', '/api/clap/warmup', timeout=300)
    assert body['loaded'] is True, body
    status = api.json('GET', '/api/clap/warmup/status')
    assert status.get('active') is True, status


def test_semantic_query_ranks_piano(stack, api, library, analyzed_library):
    probe = library.clap_probe
    limit = 5 if stack.seed_count == 0 else 15
    body = api.json('POST', '/api/clap/search', json={'query': probe['query'], 'limit': limit})
    assert_no_fp_ids(body)
    assert body['query'] == probe['query']
    results = body['results']
    assert 1 <= len(results) <= limit
    assert body['count'] == len(results)
    similarities = [r['similarity'] for r in results]
    assert similarities == sorted(similarities, reverse=True), similarities
    piano = {library.pid(k) for k in PIANO_KEYS}
    assert piano & {r['item_id'] for r in results}, [r.get('title') for r in results]


def test_validation(stack, api, analyzed_library):
    assert api.post('/api/clap/search', json={}).status_code == 400
    assert api.post('/api/clap/search', json={'query': '   '}).status_code == 400
    assert api.post('/api/clap/search', json={'query': 'piano', 'limit': 'x'}).status_code == 400


def test_concepts_and_top_queries(stack, api, db, analyzed_library):
    concepts = api.json('GET', '/api/clap/concepts')
    assert 'available' in concepts
    if concepts['available']:
        categories = concepts.get('categories') or []
        term = categories[0]['terms'][0]['term'] if categories and categories[0].get('terms') else None
        if term:
            steered = api.json(
                'POST', '/api/clap/search', timeout=300,
                json={'query': 'piano', 'limit': 3, 'steering': [{'term': term, 'weight': 1.0, 'direction': 'more'}], 'explain': True},
            )
            assert steered.get('steering'), steered
        bad = api.post('/api/clap/search', json={'query': 'piano', 'steering': [{'term': 'zzz-not-a-term', 'weight': 1.0}]})
        assert bad.status_code == 400, bad.text
    else:
        assert concepts.get('reason'), concepts
    top = api.json('GET', '/api/clap/top_queries')
    assert isinstance(top.get('queries'), list)
