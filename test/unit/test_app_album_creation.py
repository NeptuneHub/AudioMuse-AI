# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Album Creation blueprint, page, menu entry and cron wiring.

Main Features:
* The page renders under the album_creation scope and says when the feature is off
* A song seed is canonicalized from the caller's provider id, an album seed is the
  album plus its album artist, an artist seed is a name or a server artist id
* Bad input answers 400, a seed with nothing around it 404, a missing index 503 and
  an unexpected failure the album creation code, never the text of the exception
* No response ever carries a canonical fp_ id, and the slots are renumbered when the
  server scope drops a row
* The album search is paged, capped, scoped to the selected server and needs 2 letters
* The page, its API and its menu entry are off while lyrics are off. The schedule is
  not: its row is always on the Scheduled Tasks page, which alone enables or disables it
* A due album_of_the_week cron row is enqueued on the default queue for all servers
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

import app_album_creation
import config
from tasks import album_creation_manager as acm

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def _read(*parts):
    with open(os.path.join(_ROOT, *parts), encoding='utf-8') as handle:
        return handle.read()


def _album(ids=('fp_a', 'fp_b', 'fp_c')):
    roles = [acm.ROLE_OPENER] + [acm.ROLE_TRACK] * (len(ids) - 2) + [acm.ROLE_CLOSER]
    return {
        'seed': {'type': 'song', 'label': 'Seed - Artist'},
        'suggested_name': 'Seed - The Album',
        'stats': {'tracks': len(ids), 'minutes': 11, 'artists': 3, 'cohesion': 0.8,
                  'target_cohesion': 0.8, 'opener_style': 'bang'},
        'tracks': [
            {'slot': slot, 'role': role, 'item_id': item_id, 'title': item_id.upper(), 'author': 'x',
             'album': 'y', 'album_artist': 'x', 'year': 2000, 'duration': 200.0,
             'mood_vector': 'rock:0.9', 'other_features': 'happy:0.6'}
            for slot, (item_id, role) in enumerate(zip(ids, roles), start=1)
        ],
    }


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(app_album_creation.album_creation_bp)
    app.config['TESTING'] = True
    return app.test_client()


@pytest.fixture(autouse=True)
def neutral_scope(monkeypatch):
    import app_helper
    import app_server_context

    monkeypatch.setattr(config, 'LYRICS_ENABLED', True)
    monkeypatch.setattr(app_server_context, 'resolve_request_server_id', lambda data=None: None)
    monkeypatch.setattr(app_server_context, 'selected_server_scope', lambda data=None: ('srv-1', True))
    monkeypatch.setattr(app_server_context, 'resolve_input_item_id', lambda raw_id, data=None: raw_id)
    monkeypatch.setattr(
        app_server_context, 'scope_results',
        lambda rows, requested_n=None, id_key='item_id', translate=True: rows,
    )
    monkeypatch.setattr(app_helper, 'attach_song_features', lambda rows, id_key='item_id': rows)


@pytest.fixture
def created(monkeypatch):
    calls = []

    def create_album(seed_type, **seed):
        calls.append((seed_type, seed))
        return _album()

    monkeypatch.setattr(acm, 'create_album', create_album)
    return calls


def _generate(client, body):
    return client.post('/api/album_creation/generate', json=body)


class TestThePage:
    def _render(self, enabled):
        with (
            patch.object(config, 'LYRICS_ENABLED', enabled),
            patch('app_album_creation.render_template', return_value='page') as render,
        ):
            app = Flask(__name__)
            app.register_blueprint(app_album_creation.album_creation_bp)
            assert app.test_client().get('/album_creation').status_code == 200
        return render.call_args

    def test_it_renders_under_its_own_active_key_with_the_cd_size(self):
        args = self._render(True)
        assert args.args == ('album_creation.html',)
        assert args.kwargs['active'] == 'album_creation'
        assert args.kwargs['title'] == 'AudioMuse-AI - Album Creation'
        assert args.kwargs['album_creation_enabled'] is True
        assert args.kwargs['album_tracks'] == config.ALBUM_CREATION_TRACKS

    def test_it_tells_the_template_when_lyrics_are_off(self):
        assert self._render(False).kwargs['album_creation_enabled'] is False

    def test_the_template_reuses_the_shared_widgets_and_posts_to_the_shared_playlist_route(self):
        page = _read('templates', 'album_creation.html')
        for needle in (
            'id="album-creation-form"', 'autocomplete-container', 'autocomplete-results',
            'renderSongList(', 'id="playlist-creator"', 'id="playlist-status"', 'id="status"',
            "url_for('ivf_bp.create_media_server_playlist')", "url_for('ivf_bp.search_tracks_endpoint')",
            "url_for('album_creation_bp.generate_album_endpoint')", 'album-creation-disabled',
        ):
            assert needle in page, needle
        for seed in ('data-seed="song"', 'data-seed="text"'):
            assert seed in page
        for widget in ('>Text Search<', 'id="seed-examples-list"', 'id="refine-box"',
                       'id="refine-picker"', "clap_search_bp.clap_concepts_api",
                       "clap_search_bp.top_queries_api"):
            assert widget in page, widget
        for gone in ('data-seed="album"', 'data-seed="artist"', 'search_albums_endpoint',
                     'search_artists_endpoint', 'album or artist'):
            assert gone not in page, gone

    def test_the_layout_marks_the_page_per_server(self):
        assert "'album_creation': 'server'" in _read('templates', 'includes', 'layout.html')

    def test_the_menu_entry_sits_right_under_artist_similarity(self):
        lines = _read('templates', 'sidebar_navi.html').splitlines()
        index = next(i for i, line in enumerate(lines) if '>Artist Similarity</a>' in line)
        assert '>Album Creation</a>' in lines[index + 1]
        assert lines[index + 1].startswith('{% if lyrics_enabled and clap_enabled %}')
        assert lines[index + 1].endswith('{% endif %}')

    def test_the_blueprint_is_registered_by_the_app(self):
        source = _read('app.py')
        assert 'from app_album_creation import album_creation_bp' in source
        assert 'album_creation_bp,' in source.split('for blueprint in (')[1].split('):')[0]


class TestGenerate:
    def test_a_song_seed_is_canonicalized_before_it_reaches_the_manager(self, client, created, monkeypatch):
        import app_server_context

        monkeypatch.setattr(app_server_context, 'resolve_input_item_id', lambda raw_id, data=None: f'fp_{raw_id}')
        response = _generate(client, {'seed_type': 'Song', 'item_id': ' 42 '})
        assert response.status_code == 200
        assert created == [('song', {'item_id': 'fp_42'})]
        body = response.get_json()
        assert body['suggested_name'] == 'Seed - The Album'
        assert [track['role'] for track in body['tracks']] == ['opener', 'track', 'closer']

    @pytest.mark.parametrize('body', [
        {'seed_type': 'song'}, {'seed_type': 'song', 'item_id': '  '},
        {'seed_type': 'text'}, {'seed_type': 'text', 'query': '   '},
        {'seed_type': 'album', 'album': 'Blue Lines'}, {'seed_type': 'artist', 'artist': 'Nina'},
        {'seed_type': 'playlist', 'item_id': '1'}, {},
    ])
    def test_a_missing_or_unknown_seed_answers_400_and_builds_nothing(self, client, created, body):
        response = _generate(client, body)
        assert response.status_code == 400
        assert response.get_json()['error_code'] == 1003
        assert created == []

    def test_a_description_seed_passes_the_words_through(self, client, created):
        response = _generate(client, {'seed_type': 'Text', 'query': '  jazz with trumpet '})
        assert response.status_code == 200
        assert created == [('text', {'query': 'jazz with trumpet'})]

    def test_a_body_that_is_not_a_json_object_answers_400(self, client, created):
        assert client.post('/api/album_creation/generate', data='nope').status_code == 400
        assert client.post('/api/album_creation/generate', json=['song']).status_code == 400
        assert created == []

    def test_an_unknown_server_answers_400(self, client, created, monkeypatch):
        import app_server_context

        def unknown(data=None):
            raise app_server_context.UnknownServerError("Unknown server 'x'")

        monkeypatch.setattr(app_server_context, 'resolve_request_server_id', unknown)
        response = _generate(client, {'seed_type': 'song', 'item_id': '1', 'server': 'x'})
        assert response.status_code == 400 and created == []

    @pytest.mark.parametrize('error, status, code', [
        (acm.AlbumSeedNotFound('SECRET'), 404, 1004),
        (acm.AlbumSeedError('SECRET'), 400, 1003),
        (RuntimeError('SECRET'), 503, None),
        (KeyError('SECRET'), 500, 6011),
    ])
    def test_a_failure_answers_its_own_code_and_never_the_exception_text(self, client, monkeypatch, error, status, code):
        def create_album(seed_type, **seed):
            raise error

        monkeypatch.setattr(acm, 'create_album', create_album)
        response = _generate(client, {'seed_type': 'song', 'item_id': '1'})
        assert response.status_code == status
        assert 'SECRET' not in response.get_data(as_text=True)
        if code is not None:
            assert response.get_json()['error_code'] == code

    def test_no_response_of_this_api_ever_carries_a_canonical_id(self, client, created, monkeypatch):
        import app_server_context

        def scope(rows, requested_n=None, id_key='item_id', translate=True):
            return [dict(row, **{id_key: row[id_key][3:].upper()}) for row in rows]

        monkeypatch.setattr(app_server_context, 'scope_results', scope)
        monkeypatch.setattr(app_server_context, 'resolve_input_item_id', lambda raw_id, data=None: raw_id if raw_id.startswith('fp_') else f'fp_{raw_id}')
        for sent in ('42', 'fp_42'):
            response = _generate(client, {'seed_type': 'song', 'item_id': sent})
            assert response.status_code == 200
            assert 'fp_' not in response.get_data(as_text=True)
            assert [track['item_id'] for track in response.get_json()['tracks']] == ['A', 'B', 'C']

    def test_a_row_the_server_does_not_have_is_dropped_and_the_slots_stay_gapless(self, client, created, monkeypatch):
        import app_server_context

        monkeypatch.setattr(
            app_server_context, 'scope_results',
            lambda rows, requested_n=None, id_key='item_id', translate=True: [r for r in rows if r['item_id'] != 'fp_b'],
        )
        body = _generate(client, {'seed_type': 'song', 'item_id': '1'}).get_json()
        assert [(track['slot'], track['item_id']) for track in body['tracks']] == [(1, 'fp_a'), (2, 'fp_c')]
        assert body['stats']['tracks'] == 2

    def test_the_rows_get_the_shared_song_tags(self, client, created, monkeypatch):
        import app_helper

        def attach(rows, id_key='item_id'):
            for row in rows:
                row['top_genre'] = 'rock'
            return rows

        monkeypatch.setattr(app_helper, 'attach_song_features', attach)
        body = _generate(client, {'seed_type': 'song', 'item_id': '1'}).get_json()
        assert all(track['top_genre'] == 'rock' for track in body['tracks'])


class TestOffWithoutBothAnalyses:
    @pytest.mark.parametrize('off', ['LYRICS_ENABLED', 'CLAP_ENABLED'])
    def test_the_api_answers_400_and_touches_nothing(self, client, created, monkeypatch, off):
        monkeypatch.setattr(config, off, False)
        for response in (
            _generate(client, {'seed_type': 'song', 'item_id': '1'}),
            _generate(client, {'seed_type': 'text', 'query': 'jazz with trumpet'}),
        ):
            assert response.status_code == 400
            assert 'disabled' in response.get_json()['error']
        assert created == []

    def test_the_schedule_row_is_behind_no_flag_and_is_saved_like_the_other_rows(self):
        page = _read('templates', 'cron.html')
        assert 'lyrics_enabled' not in page and '{% if' not in page.split('id="album-of-the-week-cron"')[0].split('id="sonic-fingerprint-enabled"')[1]
        assert 'id="album-of-the-week-cron"' in page and 'id="album-of-the-week-enabled"' in page
        assert "{id:wn.id, name:'Album of the Week', task_type:'album_of_the_week', cron_expr:wexpr, enabled:wen}" in page
        assert "{name:'Album of the Week', task_type:'album_of_the_week', cron_expr:wexpr, enabled:wen}" in page
        assert "document.getElementById('album-of-the-week-cron').value = '30 0 * * 6'" in page

    def test_the_scheduled_task_itself_reads_no_feature_flag(self):
        import ast
        import inspect

        source = inspect.getsource(acm.create_album_of_the_week) + inspect.getsource(acm.run_album_of_the_week_task)
        names = {node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)}
        attributes = {node.attr for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Attribute)}
        assert 'is_enabled' not in names | attributes
        assert 'LYRICS_ENABLED' not in names | attributes


class TestTheCronRow:
    def test_a_due_row_is_enqueued_for_all_servers_on_the_default_queue(self):
        import taskqueue
        from app_cron import run_due_cron_jobs

        cur = MagicMock()
        cur.fetchall.return_value = [{
            'id': 1, 'name': 'Album of the Week', 'task_type': 'album_of_the_week',
            'cron_expr': '* * * * *', 'enabled': True, 'last_run': 0,
        }]
        cur.fetchone.return_value = None
        cur.__enter__.return_value = cur
        cur.rowcount = 1
        db = MagicMock()
        db.cursor.return_value = cur
        with (
            patch('app_cron.cron_matches_now', return_value=True),
            patch('app_cron.get_db', return_value=db),
            patch('app_cron.save_task_status'),
            patch('app_cron.get_queue_blocking_task', return_value=None),
            patch('app_cron.clean_up_previous_main_tasks') as clean,
            patch('app_cron.taskqueue.enqueue') as enqueue,
            patch.object(acm, 'create_album_of_the_week') as build,
        ):
            run_due_cron_jobs()
        build.assert_not_called()
        clean.assert_not_called()
        enqueue.assert_called_once()
        assert enqueue.call_args[0][0] == 'tasks.album_creation_manager.run_album_of_the_week_task'
        assert enqueue.call_args[1]['kwargs'] == {'server_scope': 'all'}
        assert enqueue.call_args[1]['task_type'] == 'album_of_the_week'
        assert enqueue.call_args[1]['queue'] == taskqueue.QUEUE_DEFAULT

    def test_the_task_function_is_allowed_on_the_queue_with_its_own_error_code(self):
        import taskqueue
        from error.error_dictionary import ERR_ALBUM_CREATION_FAILED

        func = 'tasks.album_creation_manager.run_album_of_the_week_task'
        assert func in taskqueue.ALLOWED_FUNCS
        assert taskqueue.TASK_FUNC_ERROR_CODES[func] == ERR_ALBUM_CREATION_FAILED == 6011

    def test_a_blocked_run_is_retried_like_the_sonic_fingerprint(self):
        import app_cron

        assert app_cron._cron_retry_eligible('album_of_the_week')
        assert app_cron._queue_type_for_cron_task_type('album_of_the_week') == 'album_of_the_week'
