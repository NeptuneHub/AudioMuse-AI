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

from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = '<img src=x onerror="window.metadata_xss=1">'
pytestmark = pytest.mark.browser


@pytest.fixture
def dom_page():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.evaluate('window.metadata_xss = 0')
        yield page
        browser.close()


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


def test_transition_title_cannot_add_an_event_attribute(dom_page):
    title = 'Song" onpointerenter="window.metadata_xss=1" data-tail="'
    dom_page.set_content('<div id="results"><div class="result-item"><div class="result-main"></div></div></div>')
    dom_page.add_script_tag(path=str(ROOT / 'static/song_player.js'))
    dom_page.evaluate(
        '(title) => AudioMusePlayer.showSpans(document.getElementById("results"), '
        '[{row: 0, start: 0, end: 10, duration: 12}], "Smart fade", '
        '() => ({kind: "crossfade", title}))',
        title,
    )
    mark = dom_page.locator('.play-mark')
    assert mark.get_attribute('title') == title
    assert mark.get_attribute('onpointerenter') is None
    assert mark.get_attribute('data-tail') is None
    mark.hover()
    assert dom_page.evaluate('window.metadata_xss') == 0
