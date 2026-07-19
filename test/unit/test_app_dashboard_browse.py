# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Contract for the dashboard Browse view (paginated Song/Artist/Album listing).

Guards the two rules the feature is built on: every query is LIMIT-bounded so no
request can ever return the whole catalogue, and the response is metadata only so
the internal fp_ id can never leak.

Main Features:
* No query builder puts an id in its SELECT projection; the response carries none
* Every list is fetched with LIMIT page_size+1 (has_more) and a clamped OFFSET
* The duplicates filter needs a server, carries the per-copy file paths, and is
  ordered most-duplicated first
* Orphan is catalogue-wide - a stray server param is dropped - and short-circuits
  to empty when the snapshot already knows the count is 0
* has_more never advertises a page past the offset cap
* User search text is escaped so % / _ are literal, not wildcards
"""

from pathlib import Path
from unittest.mock import MagicMock

import jinja2
import pytest

import app_dashboard as dash


class TestBrowseLike:
    def test_percent_underscore_backslash_are_escaped(self):
        assert dash._browse_like('50%') == '%50\\%%'
        assert dash._browse_like('a_b') == '%a\\_b%'
        assert dash._browse_like('c\\d') == '%c\\\\d%'


def _projection(sql):
    return sql.split(' FROM ')[0]


class TestBrowseSongsSql:
    def test_no_id_in_projection_and_no_limit_in_builder(self):
        sql, params = dash._browse_songs_sql(None, 'all', '')
        assert 'item_id' not in _projection(sql)
        assert 'LIMIT' not in sql  # the caller appends LIMIT/OFFSET
        assert 'ORDER BY' in sql
        assert params == []

    def test_server_scope_adds_exists_and_param(self):
        sql, params = dash._browse_songs_sql('srv1', 'all', '')
        assert 'EXISTS' in sql and 't.server_id = %s' in sql
        assert params == ['srv1']

    def test_duplicates_carries_copies_and_file_paths_ordered_by_copies(self):
        sql, params = dash._browse_songs_sql('srv1', 'duplicates', '')
        assert 'HAVING COUNT(*) > 1' in sql
        assert 'd.copies' in _projection(sql)
        assert 'd.files' in _projection(sql)          # per-copy file paths
        assert 'array_agg' in sql
        assert 'file_path' in sql and 'provider_track_id' not in sql  # paths only, no ids
        assert 'ORDER BY d.copies DESC' in sql        # most-duplicated on top
        assert 'item_id' not in _projection(sql)
        assert params == ['srv1']

    def test_orphan_is_an_anti_join(self):
        sql, params = dash._browse_songs_sql(None, 'orphan', '')
        assert 'NOT EXISTS' in sql
        assert 'item_id' not in _projection(sql)
        assert params == []

    def test_search_appends_escaped_ilike(self):
        sql, params = dash._browse_songs_sql(None, 'all', 'love')
        assert 's.search_u ILIKE %s' in sql
        assert params == ['%love%']


class TestBrowseArtistsAlbumsSql:
    def test_artists_group_by_and_no_id(self):
        sql, params = dash._browse_artists_sql(None, '')
        assert 'GROUP BY 1' in sql and 'ORDER BY 1' in sql
        assert 'item_id' not in _projection(sql)
        assert params == []

    def test_artists_server_scope(self):
        sql, params = dash._browse_artists_sql('srv1', 'ab')
        assert 'm.server_id = %s' in sql
        assert params == ['srv1', '%ab%']

    def test_albums_group_by_pair(self):
        sql, params = dash._browse_albums_sql(None, '')
        assert 'GROUP BY 1, 2' in sql
        assert 'item_id' not in _projection(sql)
        assert params == []

    def test_albums_search_matches_album_or_artist(self):
        sql, params = dash._browse_albums_sql(None, 'x')
        assert sql.count('ILIKE %s') == 2
        assert params == ['%x%', '%x%']


class TestBrowseSerialize:
    def test_songs_carry_no_id_and_no_fp_value(self):
        out = dash._browse_serialize('songs', [('Title', 'Artist', 'Album', 'AA', 2020, None)])
        assert out == [{'title': 'Title', 'author': 'Artist', 'album': 'Album',
                        'album_artist': 'AA', 'year': 2020, 'copies': None}]
        assert 'item_id' not in out[0]
        assert not any(isinstance(v, str) and v.startswith('fp_') for v in out[0].values())

    def test_duplicates_copies_is_int(self):
        out = dash._browse_serialize('songs', [('T', 'A', 'Al', 'AA', 2020, 3)])
        assert out[0]['copies'] == 3

    def test_duplicates_row_carries_file_paths(self):
        out = dash._browse_serialize(
            'songs', [('T', 'A', 'Al', 'AA', 2020, 2, ['/m/a.flac', '/m/b.flac'])])
        assert out[0]['copies'] == 2
        assert out[0]['files'] == ['/m/a.flac', '/m/b.flac']
        assert 'item_id' not in out[0]

    def test_non_duplicate_song_has_no_files_key(self):
        out = dash._browse_serialize('songs', [('T', 'A', 'Al', 'AA', 2020, None)])
        assert 'files' not in out[0]

    def test_artists_and_albums_shape(self):
        assert dash._browse_serialize('artists', [('X',)]) == [{'artist': 'X'}]
        assert dash._browse_serialize('albums', [('AA', 'Alb')]) == [
            {'album_artist': 'AA', 'album': 'Alb'}]


class TestBrowseTotal:
    SNAP = {
        'total_songs': 100, 'distinct_artists': 20, 'distinct_albums': 15,
        'music_servers': [
            {'name': 'Jelly', 'unique_songs': 90, 'duplicate_copies': 5},
            {'name': 'On multiple servers', 'unique_songs': -2, 'is_overlap': True},
            {'name': 'Orphan', 'unique_songs': 12, 'is_orphan': True},
        ],
    }

    def test_catalogue_songs(self):
        assert dash._browse_total(self.SNAP, 'songs', None, None, 'all', False) == 100

    def test_per_server_unique(self):
        assert dash._browse_total(self.SNAP, 'songs', 'id', 'Jelly', 'all', False) == 90

    def test_orphan(self):
        assert dash._browse_total(self.SNAP, 'songs', None, None, 'orphan', False) == 12

    def test_duplicates_is_unknown(self):
        assert dash._browse_total(self.SNAP, 'songs', 'id', 'Jelly', 'duplicates', False) is None

    def test_artists_and_albums_catalogue(self):
        assert dash._browse_total(self.SNAP, 'artists', None, None, 'all', False) == 20
        assert dash._browse_total(self.SNAP, 'albums', None, None, 'all', False) == 15

    def test_search_is_unknown(self):
        assert dash._browse_total(self.SNAP, 'songs', None, None, 'all', True) is None


class TestBrowseApi:
    @pytest.fixture
    def client(self):
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(dash.dashboard_bp)
        app.config['TESTING'] = True
        return app.test_client()

    @staticmethod
    def _mock_db(monkeypatch, rows):
        cur = MagicMock()
        cur.__enter__ = lambda self: self
        cur.__exit__ = lambda self, *a: None
        cur.fetchall.return_value = rows
        cur.fetchone.return_value = None  # _load_dashboard_stats -> ({}, None)
        conn = MagicMock()
        conn.__enter__ = lambda self: self
        conn.__exit__ = lambda self, *a: None
        conn.cursor.return_value = cur
        monkeypatch.setattr(dash, 'get_db', lambda: conn)
        return cur

    def test_default_songs_page_is_limit_bounded(self, client, monkeypatch):
        monkeypatch.setattr(dash.config, 'DASHBOARD_BROWSE_PAGE_SIZE', 3)
        rows = [('T%d' % i, 'A', 'Al', 'AA', 2020, None) for i in range(4)]  # page_size + 1
        cur = self._mock_db(monkeypatch, rows)

        resp = client.get('/api/dashboard/browse?kind=songs')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['kind'] == 'songs'
        assert len(data['results']) == 3           # trimmed to page_size
        assert data['has_more'] is True            # the 4th row proved there is more
        browse_calls = [c for c in cur.execute.call_args_list
                        if 'LIMIT %s OFFSET %s' in c[0][0]]
        assert browse_calls, "the list query must be LIMIT-bounded"
        params = browse_calls[0][0][1]
        assert params[-2] == 4 and params[-1] == 0  # LIMIT page_size+1 OFFSET 0
        assert all('item_id' not in row for row in data['results'])

    def test_has_more_is_false_on_the_last_reachable_page(self, client, monkeypatch):
        monkeypatch.setattr(dash.config, 'DASHBOARD_BROWSE_PAGE_SIZE', 3)
        monkeypatch.setattr(dash.config, 'DASHBOARD_BROWSE_MAX_OFFSET', 6)
        rows = [('T%d' % i, 'A', 'Al', 'AA', 2020, None) for i in range(4)]
        self._mock_db(monkeypatch, rows)
        # page 3 -> offset 6 (== cap, still runs); the NEXT offset (9) exceeds the
        # cap, so Next must be disabled even though 4 rows came back.
        data = client.get('/api/dashboard/browse?kind=songs&page=3').get_json()
        assert data['capped'] is False
        assert data['has_more'] is False

    def test_has_more_true_when_the_next_page_is_still_reachable(self, client, monkeypatch):
        monkeypatch.setattr(dash.config, 'DASHBOARD_BROWSE_PAGE_SIZE', 3)
        monkeypatch.setattr(dash.config, 'DASHBOARD_BROWSE_MAX_OFFSET', 6)
        rows = [('T%d' % i, 'A', 'Al', 'AA', 2020, None) for i in range(4)]
        self._mock_db(monkeypatch, rows)
        data = client.get('/api/dashboard/browse?kind=songs&page=2').get_json()  # next offset 6 <= 6
        assert data['has_more'] is True

    def test_duplicates_without_server_is_400(self, client, monkeypatch):
        self._mock_db(monkeypatch, [])
        resp = client.get('/api/dashboard/browse?kind=songs&filter=duplicates')
        assert resp.status_code == 400

    def test_unknown_server_is_400(self, client, monkeypatch):
        self._mock_db(monkeypatch, [])
        monkeypatch.setattr(dash.registry, 'get_server', lambda x: None)
        monkeypatch.setattr(dash.registry, 'get_server_by_name', lambda x: None)
        resp = client.get('/api/dashboard/browse?kind=songs&server=nope')
        assert resp.status_code == 400

    def test_page_beyond_max_offset_is_capped_not_scanned(self, client, monkeypatch):
        monkeypatch.setattr(dash.config, 'DASHBOARD_BROWSE_PAGE_SIZE', 100)
        monkeypatch.setattr(dash.config, 'DASHBOARD_BROWSE_MAX_OFFSET', 200)
        cur = self._mock_db(monkeypatch, [])

        resp = client.get('/api/dashboard/browse?kind=songs&page=10')  # offset 900 > 200

        data = resp.get_json()
        assert data['capped'] is True
        assert data['results'] == []
        cur.execute.assert_not_called()  # never touched the DB past the clamp

    def test_orphan_drops_a_stray_server(self, client, monkeypatch):
        srv = {'server_id': 'id1', 'name': 'Jelly'}
        monkeypatch.setattr(dash.registry, 'get_server', lambda x: srv if x == 'id1' else None)
        monkeypatch.setattr(dash.registry, 'get_server_by_name', lambda x: srv if x == 'Jelly' else None)
        self._mock_db(monkeypatch, [])

        resp = client.get('/api/dashboard/browse?kind=songs&filter=orphan&server=Jelly')

        data = resp.get_json()
        assert data['filter'] == 'orphan'
        assert data['server'] is None

    def test_orphan_short_circuits_when_snapshot_count_is_zero(self, client, monkeypatch):
        cur = self._mock_db(monkeypatch, [('x', 'y', 'z', 'w', 2020, None)])
        monkeypatch.setattr(dash, '_load_dashboard_stats', lambda c: ({'orphan_songs': 0}, None))
        data = client.get('/api/dashboard/browse?kind=songs&filter=orphan').get_json()
        assert data['results'] == []
        assert data['has_more'] is False
        cur.execute.assert_not_called()  # the score anti-join was never run

    def test_orphan_runs_the_query_when_count_is_nonzero(self, client, monkeypatch):
        cur = self._mock_db(monkeypatch, [('Orphan', 'A', 'Al', 'AA', 2020, None)])
        monkeypatch.setattr(dash, '_load_dashboard_stats', lambda c: ({'orphan_songs': 5}, None))
        data = client.get('/api/dashboard/browse?kind=songs&filter=orphan').get_json()
        assert len(data['results']) == 1
        cur.execute.assert_called()

    def test_no_fp_id_anywhere_in_the_response(self, client, monkeypatch):
        self._mock_db(monkeypatch, [('Song', 'Artist', 'Album', 'AA', 2021, None)])
        resp = client.get('/api/dashboard/browse?kind=songs')
        assert 'fp_' not in resp.get_data(as_text=True)


class TestBrowseTemplateParses:
    def test_browse_html_is_syntactically_valid_jinja(self):
        src = (Path(__file__).resolve().parents[2] / 'templates' / 'browse.html').read_text(
            encoding='utf-8')
        jinja2.Environment().parse(src)
