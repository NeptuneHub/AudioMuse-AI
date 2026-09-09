# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Structural guards for the setup wizard's Machine Learning Models section.

There is no javascript runner in this repo, so the section that lets a user switch
each analysis model on or off is pinned by reading the template and setup.js as
sources: the section sits between Authentication and the advanced parameters, every
switchable model carries the hidden input that submits its config flag, MusiCNN is
shown as always on, each name has a tooltip that names the features the model
powers, and the flags are removed from the advanced list so no value is rendered
twice.

Main Features:
* The section is rendered under Authentication and before the advanced parameters
* Every model flag is submitted through a hidden input that carries the flag name
* MusiCNN is a checked, disabled switch with no flag to submit
* Each model name carries a tooltip naming the features that depend on it
* The flags are no longer listed in ADVANCED_SECTIONS nor rendered as leftovers
* The Whisper switch is locked whenever the GTE Lyrics switch is off
"""

import os
import re

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)

MODEL_FLAGS = ('CLAP_ENABLED', 'LYRICS_ENABLED', 'LYRICS_ASR_ENABLE', 'NEURAL_FINGERPRINT_ENABLED')

FEATURES_PER_MODEL = {
    'musicnn': ('Clustering', 'Playlist from Similar Song', 'Song Path', 'Song Alchemy', 'Hyperbolic Explorer'),
    'clap': ('Text Search',),
    'lyrics': ('Lyrics Search',),
    'whisper': ('Lyrics Search',),
    'neural-fingerprint': ('Search by Recording',),
}


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


def _row(section, model):
    match = re.search(
        r'<div class="ml-model-row[^"]*" data-model="' + re.escape(model) + r'">(.*?)<div class="ml-model-switch">(.*?)</div>\s*</div>',
        section,
        re.DOTALL,
    )
    assert match, 'no row for model ' + model
    return match.group(1), match.group(2)


def test_the_section_sits_between_authentication_and_the_advanced_parameters():
    template = _template()
    section = _section(template)
    auth_at = template.index('<h2>Authentication</h2>')
    section_at = template.index(section)
    advanced_at = template.index('<details id="advanced-config">')
    assert auth_at < section_at < advanced_at
    assert '<h2>Machine Learning Models</h2>' in section
    assert 'Turning a model off also turns off every feature that depends on it' in section


def test_every_model_flag_is_submitted_through_a_hidden_input_named_after_it():
    section = _section(_template())
    for flag in MODEL_FLAGS:
        hidden = 'type="hidden" id="%s" name="%s"' % (flag, flag)
        assert hidden in section, flag + ' has no hidden input'
        switch = 'type="checkbox" id="ml-model-'
        assert re.search(switch + r'[a-z-]+" data-flag="' + flag + '"', section), flag + ' has no switch'


def test_musicnn_is_a_checked_disabled_switch_with_no_flag():
    section = _section(_template())
    info, switch = _row(section, 'musicnn')
    assert 'checked disabled' in switch
    assert 'data-flag' not in switch
    assert 'type="hidden"' not in switch
    assert 'Always on' in switch
    assert 'is-locked' in re.search(r'<div class="ml-model-row[^"]*" data-model="musicnn">', section).group(0)


def test_each_model_name_carries_a_tooltip_naming_the_features_it_powers():
    section = _section(_template())
    for model, features in FEATURES_PER_MODEL.items():
        info, _switch = _row(section, model)
        tooltip = re.search(r'<span class="tooltip-text"[^>]*>(.*?)</span>', info, re.DOTALL)
        assert tooltip, model + ' has no tooltip'
        for feature in features:
            assert feature in tooltip.group(1), '%s tooltip does not name %s' % (model, feature)


def test_the_flags_are_no_longer_rendered_in_the_advanced_list():
    source = _setup_js()
    sections = source[source.index('var ADVANCED_SECTIONS'):source.index('var ADVANCED_OTHER_TITLE')]
    for flag in MODEL_FLAGS:
        assert "'%s'" % flag not in sections, flag + ' is still listed in ADVANCED_SECTIONS'
    match = re.search(r'var ML_MODEL_FLAGS = \[(.*?)\];', source)
    assert match
    listed = re.findall(r"'([A-Z_]+)'", match.group(1))
    assert tuple(listed) == MODEL_FLAGS
    load = source[source.index('function loadSetupData'):source.index('function saveCurrentServerValues')]
    filter_match = re.search(r'visibleAdvancedData = (.*?);', load, re.DOTALL)
    assert filter_match
    assert '!ML_MODEL_FLAGS.includes(f.name)' in filter_match.group(1)
    assert 'renderModelSwitches(advancedData)' in load


def test_the_switches_write_true_or_false_into_the_hidden_input_and_remember_the_original():
    source = _setup_js()
    render = source[source.index('function renderModelSwitches'):source.index('function buildAdvancedFieldRow')]
    assert 'hidden.dataset.originalValue = current' in render
    assert 'hidden.value = current' in render
    assert "checkbox.checked = current === 'true'" in render
    assert "hidden.value = checkbox.checked ? 'true' : 'false'" in render


def test_the_whisper_switch_is_locked_whenever_the_lyrics_switch_is_off():
    source = _setup_js()
    start = source.index('function updateModelSwitchDependencies')
    body = source[start:source.index('function renderModelSwitches')]
    assert "modelSwitchFor('LYRICS_ENABLED')" in body
    assert "modelSwitchFor('LYRICS_ASR_ENABLE')" in body
    assert 'whisper.disabled = !lyricsOn' in body
    assert "classList.toggle('is-locked', !lyricsOn)" in body


def test_a_disabled_switch_is_greyed_out_by_the_shared_stylesheet():
    css = _read(os.path.join('static', 'style.css'))
    rule = re.search(r'\.toggle-switch input:disabled \+ \.toggle-slider \{(.*?)\}', css, re.DOTALL)
    assert rule
    assert 'opacity' in rule.group(1)
    assert 'cursor: not-allowed' in rule.group(1)


def test_neural_fingerprint_ships_off_in_config_and_the_wizard_agrees_until_a_saved_value_arrives():
    config_source = _read('config.py')
    assert 'os.environ.get("NEURAL_FINGERPRINT_ENABLED", "false")' in config_source
    section = _section(_template())
    _info, switch = _row(section, 'neural-fingerprint')
    assert 'name="NEURAL_FINGERPRINT_ENABLED" value="false"' in switch
    for flag in ('CLAP_ENABLED', 'LYRICS_ENABLED', 'LYRICS_ASR_ENABLE'):
        assert 'name="%s" value="true"' % flag in section
    source = _setup_js()
    render = source[source.index('function renderModelSwitches'):source.index('function buildAdvancedFieldRow')]
    assert "normalizeFlagValue(hidden.defaultValue, 'true')" in render
    assert 'normalizeFlagValue(field.default, shipped)' in render
    assert 'normalizeFlagValue(field.value, fallback)' in render
