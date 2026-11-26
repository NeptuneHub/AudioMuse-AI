# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Browser regressions for untrusted metadata rendered by the frontend.

Main Features:
* Saved anchor names and selected map metadata remain literal text.
* Player transition titles cannot introduce event-handler attributes.
* Exercises production rendering code in Chromium without starting the stack.
"""

import html
import json
import re
import urllib.parse
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = '<img src=x onerror="window.metadata_xss=1">'
ORIGIN = 'http://audiomuse.test'
pytestmark = pytest.mark.browser


@pytest.fixture
def dom_page(browser):
    context = browser.new_context()
    page = context.new_page()
    page.evaluate('window.metadata_xss = 0')
    yield page
    context.close()


def test_saved_anchor_name_is_literal_text(dom_page):
    source = (ROOT / 'templates/alchemy.html').read_text(encoding='utf-8')
    start = source.index("const tr = document.createElement('tr');", source.index('const refreshAnchors'))
    end = source.index('anchorsTable.appendChild(tr);', start) + len('anchorsTable.appendChild(tr);')
    dom_page.set_content('<table><tbody id="anchors"></tbody></table>')
    dom_page.evaluate(
        '(name) => { const anchor = {id: 7, name}; '
        'const anchorsTable = document.getElementById("anchors"); ' + source[start:end] + ' }',
        PAYLOAD,
    )
    assert dom_page.locator('#anchors td').first.text_content() == PAYLOAD
    assert dom_page.locator('#anchors img').count() == 0
    assert dom_page.locator('.btn-rename-anchor').get_attribute('data-id') == '7'
    assert dom_page.locator('.btn-delete-anchor').get_attribute('data-id') == '7'
    assert dom_page.evaluate('window.metadata_xss') == 0


def test_map_metadata_is_literal_text(dom_page):
    source = (ROOT / 'templates/map.html').read_text(encoding='utf-8')
    start = source.index('function renderSelectionPanel()')
    end = source.index('function removeFromSelection(', start)
    dom_page.set_content('<div id="selection-panel-inline"><div id="selection-list"></div></div>')
    dom_page.evaluate(
        '(value) => { window._plotSelection = ["7"]; '
        'const findPointById = () => ({title: value, artist: value, genre: value}); '
        + source[start:end] + ' renderSelectionPanel(); }',
        PAYLOAD,
    )
    assert dom_page.locator('.selection-list-item > div').text_content() == f'[{PAYLOAD}] {PAYLOAD} - {PAYLOAD}'
    assert dom_page.locator('#selection-list img').count() == 0
    assert dom_page.locator('#selection-list button').text_content() == 'REMOVE'
    assert dom_page.evaluate('window.metadata_xss') == 0


def _smart_fade_page(title):
    rows = ''.join(f'<div class="result-item" data-item-id="{item}" data-title="{html.escape(name)}"><div class="result-main"></div></div>'
                   for item, name in (('a', 'First'), ('b', title)))
    return ('<div class="song-player-options"></div><div id="results">' + rows + '</div>'
            '<script src="/static/error_display.js"></script>'
            '<script src="/static/song_player.js" data-play-url="/play" data-smart-fade-url="/smart_fade"'
            ' data-transition-url="/transition" data-neural="true"></script>')


def _crossfade_plan():
    flat = [0.0, 0.0, 0.0]
    side = {'gain': 1.0, 'speed': [1.0, 1.0, 1.0], 'high_db': flat, 'mid_db': flat, 'bass_db': flat}
    return {'type': 'crossfade', 'start': 90.0, 'duration': 10.0, 'middle': 5.0, 'step': 5.0,
            'filters': {'high_hz': 4000.0, 'mid_hz': 1000.0, 'mid_q': 0.5, 'bass_hz': 180.0},
            'from': dict(side, volume=[1.0, 0.7, 0.0]),
            'to': dict(side, volume=[0.0, 0.7, 1.0], start=5.0, highpass_hz=None, continue_from=15.0), 'info': {}}


def test_transition_title_cannot_add_an_event_attribute(dom_page):
    title = 'Song" onpointerenter="window.metadata_xss=1" data-tail="'
    order = {'count': 2, 'results': [{'item_id': item, 'fade_in_seconds': start, 'fade_out_seconds': 100.0, 'duration_seconds': 110.0}
                                     for item, start in (('a', 0.0), ('b', 5.0))]}
    answers = {'/': ('text/html', _smart_fade_page(title)),
               **{f'/static/{name}': ('application/javascript', (ROOT / 'static' / name).read_text(encoding='utf-8'))
                  for name in ('error_display.js', 'song_player.js')},
               '/smart_fade': ('application/json', json.dumps(order)),
               '/transition': ('application/json', json.dumps(_crossfade_plan()))}

    def serve(route):
        content_type, body = answers[urllib.parse.urlsplit(route.request.url).path]
        route.fulfill(status=200, content_type=content_type, body=body)

    dom_page.route(re.compile('^' + re.escape(ORIGIN) + '/'), serve)
    dom_page.goto(ORIGIN + '/')
    dom_page.evaluate('window.metadata_xss = 0')
    dom_page.check('input[data-option="smart"]')
    dom_page.evaluate('async () => { const list = document.getElementById("results"); '
                      'await AudioMusePlayer.arrange(list, [{item_id: "a"}, {item_id: "b"}], {keepOrder: true}); '
                      'AudioMusePlayer.show(list); }')
    mark = dom_page.locator('.result-item[data-item-id="b"] .play-mark')
    assert mark.get_attribute('title') == f'Default crossfade into {title}'
    assert mark.get_attribute('onpointerenter') is None
    assert mark.get_attribute('data-tail') is None
    mark.hover()
    assert dom_page.evaluate('window.metadata_xss') == 0
