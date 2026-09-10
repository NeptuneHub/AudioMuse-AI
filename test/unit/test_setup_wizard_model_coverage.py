# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Structural guards for the coverage bar under each Machine Learning Model.

There is no javascript runner in this repo, so the bar that tells a user how
much of the library a model's index can find is pinned by reading setup.html
and setup.js as sources: four segments and a word under every model that owns
an index, shown as a band and never as a number, hidden while the model is
off, colored per band in both themes, with no hover tooltip and no help cursor.

Main Features:
* Every model but Whisper carries a bar of four segments and a text label
* Every bar ships hidden at band 0, so a wizard without a database shows nothing
* The bar carries no tooltip and no help cursor
* The five labels carry no digit and no percent sign: a band, never a number
* The bar is hidden while its model's switch is off and re-evaluated on every toggle
* The wizard renders the bands the API hands it right after the switches
* One color per band lights the segments, in light and dark mode alike
* The copy promises no refresh cadence: the bands are read on every open
"""

import os
import re

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)

BAR_MODELS = ('musicnn', 'clap', 'lyrics', 'neural-fingerprint')


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


def _bar(section, model):
    match = re.search(
        r'<div class="ml-model-coverage" data-coverage="' + re.escape(model) + r'"([^>]*)>(.*?)</div>',
        section,
        re.DOTALL,
    )
    assert match, 'no coverage bar for model ' + model
    return match.group(1), match.group(2)


def test_every_model_but_whisper_carries_a_bar_of_four_segments_and_a_label():
    section = _section(_template())
    for model in BAR_MODELS:
        _attrs, body = _bar(section, model)
        assert body.count('<span class="ml-model-coverage-seg"></span>') == 4, model
        assert '<span class="ml-model-coverage-label">' in body, model
        assert 'aria-hidden="true"' in body, model
    assert 'data-coverage="whisper"' not in section
    assert section.count('class="ml-model-coverage"') == len(BAR_MODELS)


def test_every_bar_ships_hidden_at_band_zero():
    section = _section(_template())
    for model in BAR_MODELS:
        attrs, body = _bar(section, model)
        assert 'data-level="0"' in attrs, model
        assert ' hidden' in attrs, model
        assert 'No index yet' in body, model


def test_a_hidden_bar_is_really_not_displayed_despite_its_flex_rule():
    template = _template()
    shown = re.search(r'\.ml-model-coverage \{(.*?)\}', template, re.DOTALL)
    assert shown and 'display: flex' in shown.group(1)
    hidden = re.search(r'\.ml-model-coverage\[hidden\] \{(.*?)\}', template, re.DOTALL)
    assert hidden, 'an author display:flex rule beats the browser [hidden] rule; the template must re-hide the bar'
    assert 'display: none' in hidden.group(1)


def test_the_bar_carries_no_tooltip_and_no_help_cursor():
    template = _template()
    section = _section(template)
    for model in BAR_MODELS:
        attrs, _body = _bar(section, model)
        assert 'title=' not in attrs, model
    rule = re.search(r'\.ml-model-coverage \{(.*?)\}', template, re.DOTALL)
    assert rule
    assert 'cursor' not in rule.group(1)


def test_the_labels_are_bands_never_numbers():
    source = _setup_js()
    match = re.search(r'var MODEL_COVERAGE_LABELS = \[(.*?)\];', source)
    assert match
    labels = re.findall(r"'([^']*)'", match.group(1))
    assert len(labels) == 5
    for label in labels:
        assert not re.search(r'[0-9%]', label), label
    assert labels[0] == 'No index yet'
    assert labels[-1] == 'Ready'


def test_the_bar_is_hidden_while_its_switch_is_off_and_re_evaluated_on_every_toggle():
    source = _setup_js()
    coverage = source[source.index('var MODEL_COVERAGE_LABELS'):source.index('function buildAdvancedFieldRow')]
    assert 'bar.hidden = !(known && switchOn)' in coverage
    assert 'var switchOn = flag ? !!(checkbox && checkbox.checked) : true' in coverage
    assert 'renderModelCoverage(modelCoverageLevels)' in coverage
    flags = re.search(r'var MODEL_COVERAGE_FLAGS = \{(.*?)\};', coverage, re.DOTALL)
    assert flags
    assert "'musicnn': null" in flags.group(1)
    assert "'clap': 'CLAP_ENABLED'" in flags.group(1)
    assert "'lyrics': 'LYRICS_ENABLED'" in flags.group(1)
    assert "'neural-fingerprint': 'NEURAL_FINGERPRINT_ENABLED'" in flags.group(1)


def test_the_wizard_renders_the_bands_right_after_the_switches():
    source = _setup_js()
    load = source[source.index('function loadSetupData'):source.index('function saveCurrentServerValues')]
    assert load.index('renderModelSwitches(advancedData)') < load.index('renderModelCoverage(data.model_coverage)')


def test_one_color_per_band_lights_the_segments_in_both_themes():
    template = _template()
    for level in ('1', '2', '3', '4'):
        light = r'\.ml-model-coverage\[data-level="%s"\] \{ --coverage-color: #[0-9A-Fa-f]{6}; \}' % level
        dark = r'body\.dark-mode ' + light
        assert re.search(light, template), 'no light color for band ' + level
        assert re.search(dark, template), 'no dark color for band ' + level
    for level, lit in (('1', 1), ('2', 2), ('3', 3)):
        rule = '.ml-model-coverage[data-level="%s"] .ml-model-coverage-seg:nth-child(-n+%d)' % (level, lit)
        assert rule in template, rule
    assert '.ml-model-coverage[data-level="4"] .ml-model-coverage-seg {' in template


def test_the_copy_promises_no_refresh_cadence():
    section = _section(_template())
    assert 'how much of your library that feature can find right now' in section
    assert 'minutes' not in section
    assert 'Refreshed' not in section
