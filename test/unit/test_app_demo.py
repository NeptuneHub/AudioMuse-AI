# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Demo blueprint and the page pieces that play its answer.

Main Features:
* POST /api/demo answers 400 without a list of ids or with a length that is not
  a number of at least MIN_SECONDS, resolves the ids for the selected server,
  answers one row per id in the order sent under the ids as sent (never a
  canonical id), null times for a song without a fingerprint, and the length
  used (SONG_DEMO_SECONDS when none is sent, 400 naming it when under 5)
* SONG_DEMO_SECONDS is a visible setup wizard field in its Song Player section,
  documented in PARAMETERS.md, and the wizard refuses to save one under 5
* With NEURAL_FINGERPRINT_ENABLED false it answers 503 naming the flag; any
  manager failure is a generic 500 without the detail
* The app registers the blueprint
* static/song_demo.js posts the ids (and the length only when one is set) to the
  URL of its script tag and asks each id once
* static/song_player.js is the one interface: the Smart fade and Demo
  checkboxes (only with the neural fingerprint, kept in step) are read at the
  search; Play plays whole songs, the demos (last one stopping at its end) or
  the smart fade; the bar sits above the list, Skip only with Smart fade; the
  songs are the rendered rows (data-item-id, data-title)
* The layout loads it on every song-list page, never on the map or clustering
  page; each such page has the options in its form and wraps its rendering
  with arrange/show/hide; a path or an album keeps its order
"""

import os

import pytest

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def _source(*parts):
    with open(os.path.join(_ROOT, *parts), encoding='utf-8') as handle:
        return handle.read()


@pytest.fixture(autouse=True)
def neural_fingerprint_on(monkeypatch):
    import config

    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_ENABLED', True)


@pytest.fixture
def client(monkeypatch):
    from flask import Flask

    import app_demo
    import app_server_context

    monkeypatch.setattr(
        app_server_context, 'resolve_input_item_ids',
        lambda ids, data=None: {i: i if i.startswith('fp_') else f'fp_{i}' for i in ids},
    )
    monkeypatch.setattr(app_server_context, 'provider_echo_id', lambda raw_id: raw_id[3:].upper() if str(raw_id).startswith('fp_') else raw_id)
    app = Flask(__name__)
    app.register_blueprint(app_demo.demo_bp)
    app.config['TESTING'] = True
    return app.test_client()


def _sections(monkeypatch, answer):
    import tasks.demo_manager as dm

    seen = []

    def sections(item_ids, seconds):
        seen.append((list(item_ids), seconds))
        return answer(item_ids)

    monkeypatch.setattr(dm, 'demo_sections', sections)
    return seen


def test_a_missing_list_or_a_bad_length_answers_400(client, monkeypatch):
    seen = _sections(monkeypatch, lambda ids: [None] * len(ids))
    for body in ({}, {'item_ids': []}, {'item_ids': 'a,b'}, {'item_ids': [None, ' ']}):
        assert client.post('/api/demo', json=body).status_code == 400
    for seconds in ('30', True, 0, 4.9, -10, float('inf')):
        response = client.post('/api/demo', json={'item_ids': ['a'], 'seconds': seconds})
        assert response.status_code == 400
        assert 'seconds' in response.get_json()['error']
    assert seen == []


def test_the_demos_come_back_in_the_order_sent_under_the_ids_as_sent(client, monkeypatch):
    section = {'start_seconds': 61.5, 'end_seconds': 91.5, 'duration_seconds': 213.0}
    seen = _sections(monkeypatch, lambda ids: [None if i == 'fp_b' else dict(section) for i in ids])
    response = client.post('/api/demo', json={'item_ids': ['a', 'b', 'fp_c']})
    body = response.get_json()
    assert response.status_code == 200
    assert seen == [(['fp_a', 'fp_b', 'fp_c'], 30.0)]
    assert 'fp_' not in response.get_data(as_text=True)
    assert body['seconds'] == 30.0 and body['count'] == 3
    assert body['results'] == [
        dict({'item_id': 'a'}, **section),
        {'item_id': 'b', 'start_seconds': None, 'end_seconds': None, 'duration_seconds': None},
        dict({'item_id': 'C'}, **section),
    ]
    response = client.post('/api/demo', json={'item_ids': ['a'], 'seconds': 60})
    assert response.status_code == 200 and response.get_json()['seconds'] == 60.0
    assert seen[-1] == (['fp_a'], 60.0)


def test_the_length_sent_by_default_is_the_wizard_setting(client, monkeypatch):
    import config

    seen = _sections(monkeypatch, lambda ids: [None] * len(ids))
    monkeypatch.setattr(config, 'SONG_DEMO_SECONDS', 45.0)
    response = client.post('/api/demo', json={'item_ids': ['a']})
    assert response.status_code == 200
    assert response.get_json()['seconds'] == 45.0
    assert seen == [(['fp_a'], 45.0)]
    monkeypatch.setattr(config, 'SONG_DEMO_SECONDS', 2.0)
    response = client.post('/api/demo', json={'item_ids': ['a']})
    assert response.status_code == 400
    assert 'SONG_DEMO_SECONDS' in response.get_json()['error']
    assert len(seen) == 1


def test_the_wizard_offers_the_demo_length_and_refuses_one_under_the_minimum():
    import app_setup
    import config

    assert config.SONG_DEMO_SECONDS == 30.0
    assert app_setup.should_show_advanced('SONG_DEMO_SECONDS')
    assert 'SONG_DEMO_SECONDS' not in config.SETUP_BOOTSTRAP_EXCLUDED_KEYS
    sections = _source('static', 'setup.js')
    sections = sections[sections.index('var ADVANCED_SECTIONS'):sections.index('var ADVANCED_OTHER_TITLE')]
    assert "title: 'Song Player',\n        items: [\n            'SONG_DEMO_SECONDS'\n        ]" in sections
    assert '| `SONG_DEMO_SECONDS`' in _source('docs', 'PARAMETERS.md')
    for value in ('30', '5', 60, '12.5'):
        assert app_setup._song_demo_seconds_problem({'SONG_DEMO_SECONDS': value}) is None
    for value in ('4.9', '0', '-30', 'abc', '', 'inf', 'nan', None):
        assert 'SONG_DEMO_SECONDS' in app_setup._song_demo_seconds_problem({'SONG_DEMO_SECONDS': value})
    assert app_setup._song_demo_seconds_problem({'OTHER': '1'}) is None


def test_a_disabled_fingerprint_answers_503_and_a_failure_is_a_generic_500(client, monkeypatch):
    import config

    def boom(ids):
        raise KeyError('secret-internal-detail')

    _sections(monkeypatch, boom)
    response = client.post('/api/demo', json={'item_ids': ['a']})
    assert response.status_code == 500
    assert 'secret' not in response.get_data(as_text=True)
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_ENABLED', False)
    response = client.post('/api/demo', json={'item_ids': ['a']})
    assert response.status_code == 503
    assert 'NEURAL_FINGERPRINT_ENABLED' in response.get_json()['error']


def test_the_app_registers_the_demo_blueprint():
    source = _source('app.py')
    assert 'from app_demo import demo_bp' in source
    assert 'recording_search_bp, demo_bp, models_bp' in source


def test_the_demo_component_posts_the_ids_once_and_the_length_only_when_set():
    source = _source('static', 'song_demo.js')
    assert "const missing = [...new Set(ids)].filter(id => !known.has(id));" in source
    assert "if (typeof options.seconds === 'number') body.seconds = options.seconds;" in source
    assert 'fetch(options.url, {' in source and "method: 'POST'" in source
    assert 'window.AudioMuseDemo = { create };' in source


_PLAYER_PAGES = (
    'similarity', 'path', 'hyperbolic', 'clap_search', 'lyrics_search', 'sonic_fingerprint',
    'recording_search', 'alchemy', 'album_creation',
)


def test_the_demo_client_takes_its_url_from_its_script_tag():
    source = _source('static', 'song_demo.js')
    assert 'const defaults = document.currentScript ? document.currentScript.dataset : {};' in source
    assert 'const options = Object.assign({ url: defaults.url }, settings || {});' in source


def test_the_player_plays_whole_songs_demos_or_smart_fade_as_ticked_before_the_search():
    source = _source('static', 'song_player.js')
    assert "const nextChoice = { smart: NEURAL && picked.smart && !picked.demo, demo: NEURAL && picked.demo && !picked.smart, seed, fades: new Map(), note: '' };" in source
    assert "if (choice && choice.demo) return { songs: await demos().queue(songs), beatMatch: false, stopAtEnd: true, label: 'Demo' };" in source
    assert "if (choice && choice.smart) return { songs: songs.map(withFade), beatMatch: true, stopAtEnd: false, label: 'Smart fade' };" in source
    assert "return { songs, beatMatch: false, stopAtEnd: false, label: '' };" in source
    assert "return typeof song.end === 'number' && (index + 1 < playQueue.length || playMode.stopAtEnd) ? song.end : null;" in source
    assert 'const fadeEnd = fadeEndOf(index);' in source
    assert 'const ending = fadeEnd !== null && playMode.stopAtEnd;' in source
    assert 'if (!playMode.beatMatch || !planning || !canPlan(from, to)) return null;' in source
    assert "console.error('Could not prepare the beat-matched mix; crossfading into the next song instead.', error);" in source


def test_the_bar_has_play_and_stop_and_shows_skip_only_with_smart_fade():
    source = _source('static', 'song_player.js')
    assert '<button type="button" class="btn btn-primary play-btn">Play</button>' in source
    assert '<button type="button" class="btn btn-ghost play-skip-btn hidden" disabled>Skip to the fade</button>' in source
    assert '<button type="button" class="btn btn-ghost play-stop-btn" disabled>Stop</button>' in source
    assert "skipBtn.classList.toggle('hidden', !choice.smart);" in source
    assert 'list.parentNode.insertBefore(bar, list);' in source
    assert 'Play demos' not in source


def test_the_options_are_two_exclusive_checkboxes_drawn_only_with_the_neural_fingerprint_and_kept_in_step():
    source = _source('static', 'song_player.js')
    assert "label: 'Smart fade'," in source and "label: 'Demo'," in source
    assert 'if (!READY || !NEURAL || box.dataset.ready) return;' in source
    assert "if (input.checked) Object.keys(picked).forEach(other => { if (other !== name) picked[other] = false; });" in source
    assert "document.querySelectorAll('.song-player-options input[data-option]').forEach(other => { other.checked = picked[other.dataset.option]; });" in source
    assert "const NEURAL = settings.neural === 'true' && !!URLS.smartFade;" in source


def test_the_songs_played_are_the_rendered_rows_which_carry_their_id_and_title():
    tags = _source('static', 'song_tags.js')
    assert 'data-item-id="${escapeHtml(s.item_id || \'\')}" data-title="${escapeHtml(s.title || \'Unknown\')}"' in tags
    journey = _source('templates', 'hyperbolic.html')
    assert "'<div class=\"result-item\" data-item-id=\"' + esc(row.item_id || '') + '\" data-title=\"' + esc(row.title || 'Unknown') + '\">'" in journey
    source = _source('static', 'song_player.js')
    assert ".map((item, row) => ({ item_id: item.dataset.itemId, title: item.dataset.title || 'Unknown', row }))" in source


def test_the_layout_loads_the_one_player_on_every_song_list_page_and_never_on_the_map_or_clustering():
    layout = _source('templates', 'includes', 'layout.html')
    line = next(row for row in layout.splitlines() if "{% if active in [" in row and "'similarity'" in row)
    assert all(f"'{page}'" in line for page in _PLAYER_PAGES)
    assert "'map'" not in line and "'index'" not in line and "'clustering'" not in line
    assert "filename='song_demo.js'" in layout and "data-url=\"{{ player_urls.get('demo', '') }}\"" in layout
    assert "data-play-url=\"{{ player_urls.get('play', '') }}\"" in layout
    assert "data-neural=\"{{ 'true' if neural_enabled else 'false' }}\"" in layout
    app_source = _source('app.py')
    assert "'demo': url_for('demo_bp.demo_api')," in app_source and 'song_player=song_player,' in app_source


def test_every_song_list_page_ticks_before_the_search_and_wraps_its_rendering():
    for page in _PLAYER_PAGES:
        source = _source('templates', f'{page}.html')
        assert '<div class="song-player-options"></div>' in source, page
        assert 'data-options' not in source, page
        assert 'await AudioMusePlayer.arrange(' in source, page
        assert 'AudioMusePlayer.show(' in source and 'AudioMusePlayer.hide(' in source, page
        assert "filename='song_player.js'" not in source and 'AudioMusePlayer.create(' not in source, page
    for page in ('map', 'index'):
        source = _source('templates', f'{page}.html')
        assert 'song-player-options' not in source and 'AudioMusePlayer' not in source, page


def test_a_path_and_an_album_keep_their_order_under_smart_fade():
    for page, needle in (
        ('path', 'await AudioMusePlayer.arrange(resultsTableWrapper, currentPathData, { keepOrder: true });'),
        ('hyperbolic', "await AudioMusePlayer.arrange(document.getElementById('journey-list'), data.results || [], { keepOrder: true });"),
        ('album_creation', 'await AudioMusePlayer.arrange(resultsWrapper, album.tracks, { keepOrder: true });'),
    ):
        assert needle in _source('templates', f'{page}.html'), page
    player = _source('static', 'song_player.js')
    assert 'if (options.keepOrder) body.keep_order = true;' in player
    assert 'if (options.keepOrder) return rows;' in player


def test_every_options_box_offers_both_options():
    source = _source('static', 'song_player.js')
    assert 'box.innerHTML = Object.entries(OPTIONS).map(' in source
    assert 'dataset.options' not in source
