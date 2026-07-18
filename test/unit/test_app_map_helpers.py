# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Coordinate and mood helpers for the song map in app_map.

Covers _pick_top_mood, _round_coord, _sample_items, and _translated_bucket used
to build and per-server-scope the 2D projection payload sent to the map view.

Main Features:
* _pick_top_mood returns the highest-scoring label or "unknown" on bad input
* _round_coord rounds to three decimals and zeroes out malformed coordinates
* _sample_items samples a deterministic fraction, returning a fresh list
* _translated_bucket rewrites canonical ids to a server's provider ids, drops
  fp_/unmapped rows, and fails closed on a registry error (never leaks fp_)
"""

import gzip
import json

from app_map import (
    _pick_top_mood,
    _round_coord,
    _sample_items,
    _translated_bucket,
)


class TestPickTopMood:
    def test_returns_highest_scoring_label(self):
        assert _pick_top_mood('happy:0.8,sad:0.2') == 'happy'

    def test_empty_string_returns_unknown(self):
        assert _pick_top_mood('') == 'unknown'

    def test_none_returns_unknown(self):
        assert _pick_top_mood(None) == 'unknown'

    def test_no_colon_parts_returns_unknown(self):
        assert _pick_top_mood('justalabel') == 'unknown'

    def test_unparseable_score_treated_as_zero(self):
        assert _pick_top_mood('happy:abc,sad:0.2') == 'sad'

    def test_single_unparseable_score_still_returns_label(self):
        assert _pick_top_mood('happy:abc') == 'happy'


class TestRoundCoord:
    def test_rounds_to_three_decimals(self):
        assert _round_coord([1.23456789, 2.98765432]) == [1.235, 2.988]

    def test_non_numeric_entries_return_zeros(self):
        assert _round_coord(['a', 'b']) == [0.0, 0.0]

    def test_none_returns_zeros(self):
        assert _round_coord(None) == [0.0, 0.0]

    def test_too_short_returns_zeros(self):
        assert _round_coord([1.0]) == [0.0, 0.0]


class TestSampleItems:
    def test_deterministic_for_same_input(self):
        items = list(range(40))
        assert _sample_items(items, 0.5) == _sample_items(items, 0.5)

    def test_fraction_075_of_100_returns_75(self):
        items = list(range(100))
        assert len(_sample_items(items, 0.75)) == 75

    def test_empty_list_returns_empty(self):
        assert _sample_items([], 0.5) == []

    def test_fraction_one_returns_all_items(self):
        items = list(range(10))
        result = _sample_items(items, 1.0)
        assert result == items
        assert result is not items


def _entry_from_items(items, projection='umap'):
    payload = {'items': items, 'projection': projection, 'count': len(items)}
    js = json.dumps(payload).encode('utf-8')
    return {'json_gzip_bytes': gzip.compress(js), 'projection': projection, 'count': len(items)}


def _items_from_bucket(bucket):
    raw = bucket.get('json_gzip_bytes')
    raw = gzip.decompress(raw) if raw else bucket['json_bytes']
    return json.loads(raw)


class TestTranslatedBucket:
    def test_rewrites_ids_to_provider_and_drops_unmapped(self, monkeypatch):
        import tasks.mediaserver.registry as reg
        items = [
            {'item_id': 'fp_a', 'title': 'A', 'artist': 'x'},
            {'item_id': 'fp_b', 'title': 'B', 'artist': 'y'},
        ]
        monkeypatch.setattr(
            reg, 'translate_ids', lambda ids, server_id=None, conn=None: {'fp_a': 'prov_a'}
        )
        payload = _items_from_bucket(_translated_bucket(_entry_from_items(items), None))
        assert [it['item_id'] for it in payload['items']] == ['prov_a']
        assert payload['count'] == 1
        assert payload['projection'] == 'umap'

    def test_fails_closed_dropping_fp_but_keeping_legacy_on_registry_error(self, monkeypatch):
        import tasks.mediaserver.registry as reg

        def boom(ids, server_id=None, conn=None):
            raise RuntimeError('registry down')

        monkeypatch.setattr(reg, 'translate_ids', boom)
        items = [{'item_id': 'fp_a', 'title': 'A'}, {'item_id': 'legacy1', 'title': 'B'}]
        payload = _items_from_bucket(_translated_bucket(_entry_from_items(items), None))
        assert [it['item_id'] for it in payload['items']] == ['legacy1']

    def test_empty_entry_returns_none(self):
        assert _translated_bucket({}, None) is None
