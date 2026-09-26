# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Every search box follows one contract: search from the first letter, pages of 20.

The song, artist and playlist pickers of every page run on static/autocomplete.js,
and the endpoints they call share app_helper.search_query_arg / search_page_window
/ search_page_response. These tests pin both halves so no page and no endpoint
can drift from the standard again.

Main Features:
* track, artist and playlist search answer a one-character query and page with
  start/end (default 20 rows, per-endpoint cap)
* whitespace-only queries and empty windows never reach the backend
* X-Search-Has-More is decided on the SQL page, so a row scoped away after the
  SQL LIMIT never ends paging early
* playlist search pages a name-sorted list and a blank query lists nothing
* the library browser searches from one character
* the page size and minimum have ONE source (app_helper), rendered onto the
  widget's script tag by the layout, which loads the widget for every page
* no template keeps a private copy of the paging code
"""

import contextlib
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from werkzeug.datastructures import MultiDict

import app_alchemy
import app_artist_similarity
import app_dashboard
import app_helper
import app_ivf

ROOT = Path(__file__).resolve().parents[2]


def _read(relative):
    return (ROOT / relative).read_text(encoding='utf-8')


def _args(**kwargs):
    return MultiDict({key: str(value) for key, value in kwargs.items()})


def _tracks(count):
    return [{'item_id': 'id%d' % i, 'title': 't%d' % i, 'author': 'a', 'album': 'b'} for i in range(count)]


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(app_ivf.ivf_bp)
    app.register_blueprint(app_artist_similarity.artist_similarity_bp)
    app.register_blueprint(app_alchemy.alchemy_bp)
    app.register_blueprint(app_dashboard.dashboard_bp)
    app.config['TESTING'] = True
    return app.test_client()


@pytest.fixture
def unscoped(monkeypatch):
    ctx = app_ivf.app_server_context
    monkeypatch.setattr(ctx, 'selected_server_scope', lambda: (None, False))
    monkeypatch.setattr(ctx, 'scope_results', lambda rows, *a, **k: rows)
    monkeypatch.setattr(ctx, 'scope_artist_results', lambda rows, *a, **k: rows)
    monkeypatch.setattr(ctx, 'use_request_server', lambda *a, **k: contextlib.nullcontext())


class TestSearchPageWindow:
    def test_default_is_the_first_page_of_twenty(self):
        assert app_helper.search_page_window(_args(), 500) == (0, 20)

    def test_start_end_pick_the_next_page(self):
        assert app_helper.search_page_window(_args(start=20, end=40), 500) == (20, 20)

    def test_the_page_size_is_capped(self):
        assert app_helper.search_page_window(_args(start=0, end=999999), 100) == (0, 100)

    def test_an_empty_or_inverted_window_is_none(self):
        assert app_helper.search_page_window(_args(start=40, end=40), 500) is None
        assert app_helper.search_page_window(_args(start=40, end=20), 500) is None

    def test_a_negative_or_garbage_start_is_zero(self):
        assert app_helper.search_page_window(_args(start=-5, end=20), 500) == (0, 20)
        assert app_helper.search_page_window(_args(start='x'), 500) == (0, 20)

    def test_an_endpoint_may_keep_its_own_default(self):
        assert app_helper.search_page_window(_args(), 100, default=50) == (0, 50)


class TestSearchQueryArg:
    def test_one_character_is_a_search(self):
        assert app_helper.SEARCH_MIN_QUERY_LENGTH == 1
        assert app_helper.search_query_arg(_args(q='r'), 'q') == 'r'

    def test_text_is_trimmed_and_blank_is_no_search(self):
        assert app_helper.search_query_arg(_args(q='  re  '), 'q') == 're'
        assert app_helper.search_query_arg(_args(q='   '), 'q') == ''
        assert app_helper.search_query_arg(_args(), 'q') == ''


class TestTrackSearch:
    def test_one_character_reaches_the_backend_with_the_first_page(self, client, unscoped):
        with patch.object(app_ivf, 'search_tracks_unified', return_value=[]) as backend:
            resp = client.get('/api/search_tracks', query_string={'search_query': 'r', 'start': 0, 'end': 20})
        assert resp.status_code == 200
        assert backend.call_args.args[0] == 'r'
        assert backend.call_args.kwargs['offset'] == 0
        assert backend.call_args.kwargs['limit'] == 20

    def test_scrolling_asks_for_the_next_page(self, client, unscoped):
        with patch.object(app_ivf, 'search_tracks_unified', return_value=[]) as backend:
            client.get('/api/search_tracks', query_string={'search_query': 'r', 'start': 20, 'end': 40})
        assert backend.call_args.kwargs['offset'] == 20
        assert backend.call_args.kwargs['limit'] == 20

    def test_blank_text_never_reaches_the_backend(self, client, unscoped):
        with patch.object(app_ivf, 'search_tracks_unified', return_value=[]) as backend:
            resp = client.get('/api/search_tracks', query_string={'search_query': '   '})
        assert resp.get_json() == []
        backend.assert_not_called()

    def test_the_legacy_title_artist_params_still_search(self, client, unscoped):
        with patch.object(app_ivf, 'search_tracks_unified', return_value=[]) as backend:
            client.get('/api/search_tracks', query_string={'title': 't', 'artist': 'a'})
        assert backend.call_args.args[0] == 'a t'

    def test_a_full_sql_page_says_more_even_when_scoping_drops_a_row(self, client, unscoped, monkeypatch):
        monkeypatch.setattr(app_ivf.app_server_context, 'scope_results', lambda rows, *a, **k: rows[1:])
        with patch.object(app_ivf, 'search_tracks_unified', return_value=_tracks(20)):
            resp = client.get('/api/search_tracks', query_string={'search_query': 'r', 'start': 0, 'end': 20})
        assert len(resp.get_json()) == 19
        assert resp.headers['X-Search-Has-More'] == '1'

    def test_a_short_sql_page_is_the_last(self, client, unscoped):
        with patch.object(app_ivf, 'search_tracks_unified', return_value=_tracks(7)):
            resp = client.get('/api/search_tracks', query_string={'search_query': 'r', 'start': 0, 'end': 20})
        assert len(resp.get_json()) == 7
        assert resp.headers['X-Search-Has-More'] == '0'


class TestArtistSearch:
    def test_one_character_reaches_the_backend(self, client, unscoped):
        with patch.object(app_artist_similarity, 'search_artists_by_name', return_value=[]) as backend:
            resp = client.get('/api/search_artists', query_string={'query': 'r', 'start': 0, 'end': 20})
        assert resp.status_code == 200
        assert backend.call_args.args[0] == 'r'
        assert backend.call_args.kwargs['offset'] == 0
        assert backend.call_args.kwargs['limit'] == 20

    def test_scrolling_asks_for_the_next_page(self, client, unscoped):
        with patch.object(app_artist_similarity, 'search_artists_by_name', return_value=[]) as backend:
            client.get('/api/search_artists', query_string={'query': 'r', 'start': 20, 'end': 40})
        assert backend.call_args.kwargs['offset'] == 20
        assert backend.call_args.kwargs['limit'] == 20

    def test_blank_text_never_reaches_the_backend(self, client, unscoped):
        with patch.object(app_artist_similarity, 'search_artists_by_name', return_value=[]) as backend:
            resp = client.get('/api/search_artists', query_string={'query': ' '})
        assert resp.get_json() == []
        backend.assert_not_called()

    def test_a_full_sql_page_says_more_even_when_scoping_drops_an_artist(self, client, unscoped, monkeypatch):
        artists = [{'artist': 'n%d' % i, 'artist_id': 'x%d' % i, 'track_count': 1} for i in range(20)]
        monkeypatch.setattr(app_artist_similarity.app_server_context, 'scope_artist_results', lambda rows, *a, **k: rows[1:])
        with patch.object(app_artist_similarity, 'search_artists_by_name', return_value=artists):
            resp = client.get('/api/search_artists', query_string={'query': 'n', 'start': 0, 'end': 20})
        assert len(resp.get_json()) == 19
        assert resp.headers['X-Search-Has-More'] == '1'

    def test_there_is_exactly_one_artist_search_route(self, client):
        rules = [rule for rule in client.application.url_map.iter_rules() if rule.rule == '/api/search_artists']
        assert [rule.endpoint for rule in rules] == ['artist_similarity_bp.search_artists_endpoint']


class TestPlaylistSearch:
    PLAYLISTS = [
        {'Id': 'p3', 'Name': 'charlie mix', 'ChildCount': 3},
        {'id': 'p1', 'name': 'Alpha mix', 'songCount': 1},
        {'Id': 'p2', 'Name': 'bravo mix', 'ChildCount': 0},
        {'Id': 'p4', 'Name': 'other'},
    ]

    def _get(self, client, monkeypatch, playlists=None, **params):
        monkeypatch.setattr(app_alchemy, '_cached_all_playlists', lambda server_id: playlists or self.PLAYLISTS)
        return client.get('/api/search_playlists', query_string=params)

    def test_matches_are_sorted_by_name_and_paged(self, client, unscoped, monkeypatch):
        first = self._get(client, monkeypatch, query='mix', start=0, end=2)
        second = self._get(client, monkeypatch, query='mix', start=2, end=4)
        assert [row['id'] for row in first.get_json()] == ['p1', 'p2']
        assert first.headers['X-Search-Has-More'] == '1'
        assert [row['id'] for row in second.get_json()] == ['p3']
        assert second.headers['X-Search-Has-More'] == '0'

    def test_one_character_searches(self, client, unscoped, monkeypatch):
        assert [row['id'] for row in self._get(client, monkeypatch, query='t', start=0, end=20).get_json()] == ['p4']

    def test_a_blank_query_lists_nothing_like_every_other_search(self, client, unscoped, monkeypatch):
        assert self._get(client, monkeypatch, query='  ', start=0, end=20).get_json() == []

    def test_without_a_window_the_old_default_of_fifty_stays(self, client, unscoped, monkeypatch):
        many = [{'Id': 'id%03d' % i, 'Name': 'mix %03d' % i} for i in range(80)]
        assert len(self._get(client, monkeypatch, playlists=many, query='mix').get_json()) == 50


class TestBrowseSearch:
    def test_one_character_filters_the_listing(self, client, monkeypatch):
        cur = MagicMock()
        cur.__enter__ = lambda self: self
        cur.__exit__ = lambda self, *a: None
        cur.fetchall.return_value = []
        cur.fetchone.return_value = None
        conn = MagicMock()
        conn.__enter__ = lambda self: self
        conn.__exit__ = lambda self, *a: None
        conn.cursor.return_value = cur
        monkeypatch.setattr(app_dashboard, 'get_db', lambda: conn)
        resp = client.get('/api/dashboard/browse', query_string={'kind': 'songs', 'q': 'r'})
        assert resp.status_code == 200
        listing = [c for c in cur.execute.call_args_list if 'LIMIT %s OFFSET %s' in c[0][0]]
        assert listing
        assert 'ILIKE' in listing[0][0][0]
        assert '%r%' in listing[0][0][1]


class TestOneSourceForTheContract:
    def test_the_widget_has_no_copy_of_the_page_size_or_minimum(self):
        widget = _read('static/autocomplete.js')
        assert 'settings.pageSize' in widget and 'settings.minChars' in widget
        assert not re.search(r'(PAGE_SIZE|MIN_CHARS)\s*=\s*\d', widget)

    def test_the_layout_renders_the_backend_constants_onto_the_widget(self):
        layout = _read('templates/includes/layout.html')
        tag = re.search(r'<script[^>]*autocomplete\.js[^>]*>', layout).group(0)
        assert 'data-page-size="{{ search_page_size }}"' in tag
        assert 'data-min-chars="{{ search_min_chars }}"' in tag
        entry = _read('app.py')
        assert 'search_page_size=SEARCH_PAGE_SIZE' in entry
        assert 'search_min_chars=SEARCH_MIN_QUERY_LENGTH' in entry

    def test_the_layout_loads_the_widget_before_every_page_script(self):
        layout = _read('templates/includes/layout.html')
        widget = layout.index("filename='autocomplete.js'")
        assert layout.index("filename='error_display.js'") < widget < layout.index('{% block bodyAdditions %}')

    def test_no_template_keeps_its_own_copy_of_the_picker(self):
        endpoints = ('search_tracks_endpoint', 'search_artists_endpoint', 'search_playlists')
        templates = sorted((ROOT / 'templates').rglob('*.html'))
        assert len(templates) > 10
        offenders = []
        for path in templates:
            text = path.read_text(encoding='utf-8')
            if 'IntersectionObserver' in text or 'autocomplete-load-more' in text:
                offenders.append(path.name + ': private infinite-scroll code')
            if any(name in text for name in endpoints) and 'AudioMuseAutocomplete.attach' not in text:
                offenders.append(path.name + ': calls a search endpoint without the shared widget')
        assert offenders == []
