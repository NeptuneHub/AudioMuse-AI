# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Catalogue identity and deduplication (ALGORITHM.md section 2) on the derived copies.

The retagged packet copy of a clip is the same recording, so it maps N:1 onto
the same catalogue row; the copy padded with seven seconds of silence is a
different duration, so it stays a separate row. The API accepts the catalogue
id on input but only ever echoes provider ids, and the browser exposes the
duplicate copy.

Main Features:
* the stream copy shares its catalogue row with the original
* the padded copy is a separate row with the longer duration
* /api/track resolves both ids and echoes a provider id
* similar_tracks never lists the merged copy next to its original
"""

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, item_id_of, provider_ids_for, rows

pytestmark = pytest.mark.e2e


def test_stream_copy_is_the_same_recording(stack, db, library, analyzed_library):
    for copy_key, spec in library.copies.items():
        if spec['expect'] != 'merged':
            continue
        original = item_id_of(db, library.pid(spec['of']))
        copy = item_id_of(db, library.pid(copy_key))
        assert original and original == copy, (copy_key, original, copy)
        assert sorted(provider_ids_for(db, original)) == sorted([library.pid(spec['of']), library.pid(copy_key)])


def test_padded_copy_is_a_separate_row(stack, db, library, analyzed_library):
    for copy_key, spec in library.copies.items():
        if spec['expect'] != 'separate':
            continue
        original = item_id_of(db, library.pid(spec['of']))
        copy = item_id_of(db, library.pid(copy_key))
        assert original and copy and original != copy, (copy_key, original, copy)
        durations = dict(rows(db, 'SELECT item_id, duration FROM score WHERE item_id = ANY(%s)', ([original, copy],)))
        assert durations[copy] - durations[original] >= spec['extra_seconds'] - 0.5, durations


def test_track_endpoint_accepts_catalogue_id_but_echoes_provider_id(stack, api, db, library, analyzed_library):
    pid = library.pid('A01')
    catalogue_id = item_id_of(db, pid)
    by_catalogue = api.json('GET', f'/api/track?item_id={catalogue_id}')
    assert_no_fp_ids(by_catalogue)
    assert by_catalogue['item_id'] in set(provider_ids_for(db, catalogue_id)), by_catalogue
    assert by_catalogue['title'] == library.track('A01').title
    by_copy = api.json('GET', f'/api/track?item_id={library.pid("F01")}')
    assert_no_fp_ids(by_copy)
    assert by_copy['title'] == library.track('A01').title


def test_similar_tracks_never_lists_the_merged_copy(stack, api, db, library, analyzed_library):
    original = library.pid('A01')
    results = api.json('GET', f'/api/similar_tracks?item_id={original}&n=20')
    assert_no_fp_ids(results)
    catalogue_id = item_id_of(db, original)
    merged = set(provider_ids_for(db, catalogue_id))
    assert not (merged & {r['item_id'] for r in results}), results
    assert 1 <= len(results) < stack.catalogue_rows


def test_catalogue_counts(stack, db, library, analyzed_library):
    assert rows(db, 'SELECT count(*) FROM score')[0][0] == stack.catalogue_rows
    assert rows(db, 'SELECT count(*) FROM track_server_map')[0][0] == stack.analyzable_files
    assert rows(db, 'SELECT count(DISTINCT item_id) FROM track_server_map')[0][0] == stack.catalogue_rows
