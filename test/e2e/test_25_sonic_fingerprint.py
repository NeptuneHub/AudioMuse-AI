# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Sonic fingerprint (ALGORITHM.md section 10) from real Navidrome play history.

Plays are seeded through the Subsonic scrobble endpoint, so the app's own
provider code reads them back from getAlbumList2 type=frequent, builds the
listening profile and returns the seeds plus their neighbours, which then
become a real playlist.

Main Features:
* scrobbled tracks are the most played and lead the fingerprint result
* n caps the result, a non-numeric n is 400, a wrong password yields nothing
* the result is created as a playlist on Navidrome
"""

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, create_playlist, unique_name

pytestmark = pytest.mark.e2e

SEED_KEYS = ('B01', 'B02', 'B03')


@pytest.fixture(scope='module')
def seeded_plays(stack, library, navidrome, analyzed_library):
    for key in SEED_KEYS:
        for _ in range(2):
            navidrome.scrobble(library.pid(key), submission=True)
    return [library.pid(k) for k in SEED_KEYS]


def test_fingerprint_reflects_play_history(stack, api, library, seeded_plays, golden):
    results = api.json('POST', '/api/sonic_fingerprint/generate', json={'n': 8})
    assert_no_fp_ids(results)
    golden.check('sonic_fingerprint n8 after the seeded plays', results)
    assert 1 <= len(results) <= 8, results
    ids = [r['item_id'] for r in results]
    assert set(seeded_plays) <= set(ids), (seeded_plays, ids)
    for row in results:
        assert row.get('title') and 'author' in row


def test_fingerprint_count_and_validation(stack, api, seeded_plays):
    three = api.json('GET', '/api/sonic_fingerprint/generate?n=3')
    assert len(three) == 3, three
    assert api.post('/api/sonic_fingerprint/generate', json={'n': 'x'}).status_code == 400
    wrong = api.json('POST', '/api/sonic_fingerprint/generate', json={'n': 5, 'navidrome_password': 'wrong'})
    assert wrong == [], wrong


def test_fingerprint_becomes_a_playlist(stack, api, navidrome, seeded_plays):
    results = api.json('POST', '/api/sonic_fingerprint/generate', json={'n': 6})
    ids = [r['item_id'] for r in results]
    created = create_playlist(api, unique_name('fingerprint'), ids)
    try:
        assert navidrome.playlist_entry_ids(created['playlist_id']) == ids
    finally:
        navidrome.delete_playlist(created['playlist_id'])
