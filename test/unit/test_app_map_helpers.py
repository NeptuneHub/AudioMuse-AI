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
* build_map_cache reads the catalogue through a plain client-side cursor with
  no embedding column, fetches embeddings in chunks only for rows the stored
  projection does not cover, and drops a row whose embedding is unreadable
"""

import gc
import gzip
import json
from unittest.mock import MagicMock

import numpy as np

import app_map
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
    def test_fraction_half_of_ten_returns_evenly_spaced_items_0_2_4_6_9(self):
        assert _sample_items(list(range(10)), 0.5) == [0, 2, 4, 6, 9]

    def test_fraction_below_one_item_still_returns_the_first_item(self):
        assert _sample_items(list(range(10)), 0.01) == [0]

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


class TestBuildMapCacheStreaming:
    def _run(self, monkeypatch, rows, id_map, proj, emb_rows=None, warm=None):
        cur = MagicMock()
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        cur.fetchall = MagicMock(side_effect=[list(rows), list(emb_rows or [])])
        conn = MagicMock()
        conn.cursor.return_value = cur

        monkeypatch.setattr(app_map, 'get_db', lambda: conn)
        monkeypatch.setattr(app_map, 'load_map_projection', lambda *a, **k: (id_map, proj))
        monkeypatch.setattr(app_map, '_warm_server_buckets', warm or (lambda: None))
        monkeypatch.setattr(app_map, 'MAP_JSON_CACHE', {})
        app_map.build_map_cache()
        return conn, cur

    def test_catalogue_scan_uses_a_plain_client_side_cursor(self, monkeypatch):
        conn, cur = self._run(
            monkeypatch,
            [('a', 'T', 'A', 'happy:1')],
            ['a'],
            np.array([[1.0, 2.0]], dtype=np.float32),
        )
        assert conn.cursor.call_args.kwargs.get('name') is None
        cur.fetchall.assert_called_once()

    def test_catalogue_scan_selects_no_embedding_column(self, monkeypatch):
        _, cur = self._run(
            monkeypatch,
            [('a', 'T', 'A', 'happy:1')],
            ['a'],
            np.array([[1.0, 2.0]], dtype=np.float32),
        )
        sql = cur.execute.call_args_list[0].args[0]
        select_list = sql.split('FROM')[0]
        assert 'embedding' not in select_list
        assert 'e.embedding IS NOT NULL' in sql

    def test_projected_library_never_queries_embeddings(self, monkeypatch):
        _, cur = self._run(
            monkeypatch,
            [('a', 'Title', 'Artist', 'happy:1'), ('b', 'Title', 'Artist', 'sad:1')],
            ['a', 'b'],
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )
        assert cur.execute.call_count == 1
        payload = _items_from_bucket(app_map.MAP_JSON_CACHE['100'])
        coords = {it['item_id']: it['embedding_2d'] for it in payload['items']}
        assert coords == {'a': [1.0, 2.0], 'b': [3.0, 4.0]}

    def test_only_rows_missing_from_the_projection_reach_the_projector(self, monkeypatch):
        emb = np.array([0.1, 0.2], dtype=np.float32).tobytes()
        seen = {}

        def fake_project(mat):
            seen['rows'] = mat.shape[0]
            return [(9.0, 9.0)] * mat.shape[0]

        monkeypatch.setattr(app_map, '_project_with_umap', None)
        monkeypatch.setattr(app_map, '_project_to_2d', fake_project)
        _, cur = self._run(
            monkeypatch,
            [
                ('a', 'T', 'A', 'happy:1'),
                ('b', 'T', 'A', 'happy:1'),
            ],
            ['a'],
            np.array([[1.0, 2.0]], dtype=np.float32),
            emb_rows=[('b', emb)],
        )
        assert seen['rows'] == 1
        assert cur.execute.call_args_list[1].args[1] == (['b'],)
        payload = _items_from_bucket(app_map.MAP_JSON_CACHE['100'])
        coords = {it['item_id']: it['embedding_2d'] for it in payload['items']}
        assert coords == {'a': [1.0, 2.0], 'b': [9.0, 9.0]}

    def test_unprojected_row_with_no_readable_embedding_is_left_off_the_map(self, monkeypatch):
        emb = np.array([0.1, 0.2], dtype=np.float32).tobytes()
        monkeypatch.setattr(app_map, '_project_with_umap', None)
        monkeypatch.setattr(app_map, '_project_to_2d', lambda mat: [(9.0, 9.0)] * mat.shape[0])
        self._run(
            monkeypatch,
            [
                ('a', 'T', 'A', 'happy:1'),
                ('b', 'T', 'A', 'happy:1'),
                ('c', 'T', 'A', 'happy:1'),
                ('d', 'T', 'A', 'happy:1'),
            ],
            ['a'],
            np.array([[1.0, 2.0]], dtype=np.float32),
            emb_rows=[('c', b'not-float32'), ('d', emb)],
        )
        payload = _items_from_bucket(app_map.MAP_JSON_CACHE['100'])
        coords = {it['item_id']: it['embedding_2d'] for it in payload['items']}
        assert coords == {'a': [1.0, 2.0], 'd': [9.0, 9.0]}

    def test_missing_embeddings_are_fetched_in_chunks(self, monkeypatch):
        monkeypatch.setattr(app_map, '_MISSING_EMBEDDING_CHUNK', 2)
        cur = MagicMock()
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        emb = np.array([0.5], dtype=np.float32).tobytes()
        cur.fetchall = MagicMock(side_effect=[[('x', emb), ('y', emb)], [('z', emb)]])
        conn = MagicMock()
        conn.cursor.return_value = cur
        found = app_map._fetch_missing_embeddings(conn, ['x', 'y', 'z'])
        assert [c.args[1] for c in cur.execute.call_args_list] == [(['x', 'y'],), (['z'],)]
        assert sorted(found) == ['x', 'y', 'z']

    def test_item_rows_are_released_before_the_server_warmup(self, monkeypatch):
        alive = {}

        def warm():
            gc.collect()
            alive['first_row'] = any(
                type(o) is dict and o.get('item_id') == 'row-first' and 'embedding_2d' in o
                for o in gc.get_objects()
            )

        self._run(
            monkeypatch,
            [('row-first', 'T', 'A', 'happy:1'), ('row-last', 'T', 'A', 'happy:1')],
            ['row-first', 'row-last'],
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            warm=warm,
        )
        assert alive == {'first_row': False}

    def test_empty_catalogue_leaves_an_empty_cache(self, monkeypatch):
        self._run(monkeypatch, [], [], np.zeros((0, 2), dtype=np.float32))
        assert app_map.MAP_JSON_CACHE == {}
