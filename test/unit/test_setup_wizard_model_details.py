# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Structural guards for the "What it does" list under each Machine Learning Model.

Each model row carries a button that expands a short list, one bullet per page
the model feeds, saying what the model gives that page and why it matters.
The list is plain text inside the row (no box, no headings, no diagram): the
row simply grows. There is no javascript runner in this repo, so setup.html
and setup.js are read as sources.

Main Features:
* Every model, MusiCNN and Whisper included, has a toggle wired to a list that
  ships collapsed
* Every list is one bullet list, two to seven bullets, each opening with a
  bold feature name and a colon, with no heading, diagram, cost line or boxed
  panel; DCLAP adds one plain sentence after the list, and Whisper is that one
  sentence alone (it is a fallback, not a page of its own)
* The lists are plain ASCII and every sentence stays short for a B2 reader
* The DCLAP list names Text Search as the main use, the mood pills, Clustering,
  the Dashboard and Instant Playlist, and the sentence after it suggests
  keeping the model on
* The lyrics, Whisper and neural lists name what the code actually does
* Opening one list closes every other one (one open at a time), the toggle
  flips aria-expanded and the list's hidden flag in setup.js, and the subtitle
  points at the list instead of a hover
"""

import os
import re

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)

MODELS = ('musicnn', 'clap', 'lyrics', 'whisper', 'neural-fingerprint')

MAX_WORDS_PER_SENTENCE = 30


def _read(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path), encoding='utf-8') as handle:
        return handle.read()


def _template():
    return _read(os.path.join('templates', 'setup.html'))


def _setup_js():
    return _read(os.path.join('static', 'setup.js'))


def _section(template):
    match = re.search(
        r'<section class="section" id="ml-models-section">(.*?)</section>',
        template,
        re.DOTALL,
    )
    assert match, 'the Machine Learning Models section is missing from setup.html'
    return match.group(0)


def _panel(section, model):
    match = re.search(
        r'<div class="ml-model-details" id="ml-model-' + re.escape(model) + r'-details" hidden>(.*?)</div>\s*<div class="ml-model-switch">',
        section,
        re.DOTALL,
    )
    assert match, 'no details list for model ' + model
    return match.group(1)


def _bullets(panel):
    return re.findall(r'<li>(.*?)</li>', panel, re.DOTALL)


def _text(html):
    broken = re.sub(r'</(li|strong)>', '. ', html)
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', broken)).strip()


def _sentences(text):
    return [s.strip() for s in re.split(r'[.!?](?:\s|$)', text) if s.strip()]


def test_every_model_has_a_toggle_wired_to_a_collapsed_list():
    section = _section(_template())
    for model in MODELS:
        toggle = re.search(
            r'<button type="button" class="ml-model-details-toggle" aria-expanded="false" aria-controls="ml-model-'
            + re.escape(model) + r'-details">What it does</button>',
            section,
        )
        assert toggle, model + ' has no toggle'
        assert _panel(section, model)
    assert section.count('class="ml-model-details-toggle"') == len(MODELS)
    assert section.count('class="ml-model-details"') == len(MODELS)


def test_every_list_is_one_bullet_list_of_two_to_seven_bold_feature_lines_and_nothing_else():
    section = _section(_template())
    for model in MODELS:
        panel = _panel(section, model)
        bullets = _bullets(panel)
        if model == 'whisper':
            assert panel.count('<ul>') == 0 and not bullets, 'Whisper is one sentence, not a list'
        else:
            assert panel.count('<ul>') == 1, model
            assert 2 <= len(bullets) <= 7, '%s has %d bullets' % (model, len(bullets))
        for bullet in bullets:
            assert bullet.lstrip().startswith('<strong>'), '%s bullet without a bold feature name: %s' % (model, bullet)
            assert re.search(r':</strong>', bullet), model
            assert not bullet.startswith('<strong>Cost:'), model + ' still carries a cost line'
            assert not bullet.startswith('<strong>Always on:'), model
            assert not bullet.startswith('<strong>Instrumental songs:'), model
        for forbidden in ('<h4', 'ml-flow', '<table'):
            assert forbidden not in panel, '%s list carries %s' % (model, forbidden)
        notes = re.findall(r'<p class="ml-model-details-note">(.*?)</p>', panel)
        assert len(notes) == (1 if model in ('clap', 'whisper') else 0), model
        assert '<p>' not in panel, model


def test_the_list_is_not_a_box_inside_the_row():
    template = _template()
    rule = re.search(r'\.ml-model-details \{(.*?)\}', template, re.DOTALL).group(1)
    for boxy in ('border', 'background', 'padding', 'box-shadow'):
        assert boxy not in rule, 'the list must not be a box: ' + boxy
    assert 'flex: 1 1 100%' in rule
    assert re.search(r'\.ml-model-row \{[^}]*flex-wrap: wrap', template, re.DOTALL)
    assert re.search(r'\.ml-model-info \{[^}]*flex: 1 1 0', template, re.DOTALL)


def test_the_lists_are_plain_ascii():
    section = _section(_template())
    for model in MODELS:
        assert _panel(section, model).isascii(), model + ' list carries a non-ASCII character'


def test_every_sentence_in_a_list_stays_short_for_a_b2_reader():
    section = _section(_template())
    for model in MODELS:
        for sentence in _sentences(_text(_panel(section, model))):
            words = len(sentence.split())
            assert words <= MAX_WORDS_PER_SENTENCE, '%s: %d words: %s' % (model, words, sentence)


def test_the_dclap_list_names_text_search_first_the_mood_pages_and_a_sentence_suggests_keeping_it_on():
    panel = _panel(_section(_template()), 'clap')
    bullets = _bullets(panel)
    assert bullets[0].startswith('<strong>Text Search:</strong>')
    text = _text(' '.join(bullets))
    for name in ('main use', 'pop song with a female vocalist', 'mood pills', 'Clustering', 'Dashboard', 'Instant Playlist'):
        assert name in text, name
    for label in ('danceable', 'aggressive', 'happy', 'party', 'relaxed', 'sad'):
        assert label in text, label
    assert 'Keep it on' not in text
    note = re.search(r'<p class="ml-model-details-note">(.*?)</p>', panel).group(1)
    assert note.startswith('We suggest keeping it on')
    assert 'mood scores feed the other pages' in note


def test_the_lyrics_list_names_the_three_search_tabs_semgrove_and_song_path():
    text = _text(_panel(_section(_template()), 'lyrics'))
    for phrase in ('By Text', 'By Axes', 'By Song (SemGrove)', '75 percent', 'Song Path', 'Instant Playlist', 'any language'):
        assert phrase in text, phrase
    assert 'Instrumental songs' not in text


def test_the_whisper_sentence_says_it_is_a_fallback_after_the_music_server_and_the_lyrics_api():
    text = _text(_panel(_section(_template()), 'whisper'))
    for phrase in ('fallback', 'music server', 'Lyrics API', 'Whisper transcribes', 'Lyrics Search'):
        assert phrase in text, phrase
    for gone in ('Cost', 'Voice check', 'instrumental'):
        assert gone not in text, gone
    assert len(_sentences(text)) <= 2


def test_the_neural_list_names_both_tabs_and_the_clip_length():
    text = _text(_panel(_section(_template()), 'neural-fingerprint'))
    for phrase in ('Search by Recording', 'Search by Song', '20 seconds', 'where in the song', 'other versions'):
        assert phrase in text, phrase
    assert 'Cost' not in text


def test_the_musicnn_list_names_the_pages_its_tooltip_names():
    text = _text(_panel(_section(_template()), 'musicnn'))
    for page in ('Playlist from Similar Song', 'Music Map', 'Song Path', 'Clustering', 'Song Alchemy',
                 'Artist Similarity', 'Sonic Fingerprint', 'Hyperbolic Explorer', 'Instant Playlist'):
        assert page in text, page
    assert 'Always on' not in text


def test_opening_one_list_closes_the_others_and_the_toggle_flips_aria_expanded_and_hidden():
    source = _setup_js()
    wiring = source[source.index('var modelDetailsToggles'):source.index('function buildAdvancedFieldRow')]
    assert "button.getAttribute('aria-controls')" in wiring
    assert "button.setAttribute('aria-expanded', open ? 'true' : 'false')" in wiring
    assert 'panel.hidden = !open' in wiring
    handler = wiring[wiring.index("button.addEventListener('click'"):]
    assert handler.index('setModelDetailsOpen(other, false)') < handler.index('setModelDetailsOpen(button, !open)')


def test_the_subtitle_points_at_the_list_instead_of_a_hover():
    section = _section(_template())
    assert 'open "What it does" under a model' in section
    assert 'hover the name' not in section
