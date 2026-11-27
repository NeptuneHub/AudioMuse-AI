# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Structural guards for the setup wizard's Smart Fade panel.

There is no javascript runner in this repo, so the panel is pinned by reading
setup.html, setup.js and the docs as sources, like the Machine Learning Models
switches it reuses.

Main Features:
* A closed details panel after the AI Prompt one, the last before Save
* One hidden input per transition switch, shipped "true"; the default
  crossfade is a checked, disabled switch with no flag
* setup.js lists the same flags as tasks.smart_fade_mix.SWITCHES, loads them
  like the model switches and keeps them out of the advanced list
* With the Neural Fingerprint switch off every transition switch and tuning
  field is disabled and a note says why; the Neural Fingerprint list names
  smart fade
* The other smart fade parameters sit in the panel's Fine tuning box, never in
  the advanced list, and every one has a PARAMETERS.md row
* Its test box (saved configuration and Neural Fingerprint on) picks two songs
  with the shared search box and plays them through the shared player with
  the switches as they stand; its inputs have no name, so a Save never sends them
"""

import os
import re

import app_setup
import config
from tasks import smart_fade_mix as mix

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
FLAGS = tuple(mix.SWITCHES.values())


def _read(*parts):
    with open(os.path.join(REPO_ROOT, *parts), encoding='utf-8') as handle:
        return handle.read()


def _panel(template):
    match = re.search(r'<details id="smart-fade-config">(.*?)</details>', template, re.DOTALL)
    assert match, 'the Smart Fade panel is missing from setup.html'
    return match.group(0)


def _js_block(source, start, end):
    begin = source.index(start)
    return source[begin:source.index(end, begin)]


def test_the_panel_is_closed_and_the_last_one_before_save():
    template = _read('templates', 'setup.html')
    panel = _panel(template)
    assert '<summary>Smart Fade</summary>' in panel
    assert template.index('<details id="ai-prompt-config">') < template.index(panel) < template.index('id="save-button"')
    assert template.count('<details', template.index(panel) + 1) == 0
    assert '<details id="smart-fade-config" open' not in template


def test_every_transition_has_a_switch_shipped_on_and_the_crossfade_is_always_on():
    panel = _panel(_read('templates', 'setup.html'))
    for flag in FLAGS:
        assert 'type="hidden" id="%s" name="%s" value="true"' % (flag, flag) in panel, flag
        assert re.search(r'type="checkbox" id="smart-fade-[a-z-]+" data-flag="' + flag + '"', panel), flag
    crossfade = re.search(r'data-transition="crossfade">(.*?)</div>\s*</div>', panel, re.DOTALL).group(1)
    assert 'checked disabled' in crossfade and 'data-flag' not in crossfade and 'Always on' in crossfade
    for flag in FLAGS:
        assert getattr(config, flag) is True
        assert app_setup.should_show_advanced(flag) and flag not in config.SETUP_BOOTSTRAP_EXCLUDED_KEYS


def test_setup_js_loads_the_switches_and_keeps_them_out_of_the_advanced_list():
    source = _read('static', 'setup.js')
    listed = re.findall(r"'([A-Z_]+)'", re.search(r'var SMART_FADE_FLAGS = \[(.*?)\];', source, re.DOTALL).group(1))
    assert tuple(listed) == FLAGS
    sections = _js_block(source, 'var ADVANCED_SECTIONS', 'var ADVANCED_OTHER_TITLE')
    for flag in FLAGS:
        assert "'%s'" % flag not in sections, flag
    load = _js_block(source, 'function loadSetupData', 'function saveCurrentServerValues')
    assert '!SMART_FADE_FLAGS.includes(f.name)' in re.search(r'visibleAdvancedData = (.*?);', load, re.DOTALL).group(1)
    assert 'renderSmartFadeSwitches(advancedData)' in load
    render = _js_block(source, 'function renderSmartFadeSwitches', 'function loadFlagSwitches')
    assert 'loadFlagSwitches(SMART_FADE_FLAGS, smartFadeSwitchFor, fields)' in render
    assert "#smart-fade-config input[type=\"checkbox\"][data-flag=\"' + flag + '\"]" in source
    listener = _js_block(source, 'SMART_FADE_FLAGS.forEach', 'var modelDetailsToggles')
    assert "hidden.value = checkbox.checked ? 'true' : 'false'" in listener


def test_the_switches_are_disabled_while_the_neural_fingerprint_is_off():
    source = _read('static', 'setup.js')
    body = _js_block(source, 'function updateSmartFadeAvailability', 'function updateModelSwitchDependencies')
    assert "modelSwitchFor('NEURAL_FINGERPRINT_ENABLED')" in body
    assert 'checkbox.disabled = !neuralOn' in body
    assert "querySelectorAll('#smart-fade-fields input, #smart-fade-fields select')" in body
    assert 'input.disabled = !neuralOn' in body
    assert "classList.toggle('is-locked', !neuralOn || !checkbox)" in body
    assert 'note.hidden = neuralOn' in body
    dependencies = _js_block(source, 'function updateModelSwitchDependencies', 'function renderModelSwitches')
    assert 'updateSmartFadeAvailability()' in dependencies
    template = _read('templates', 'setup.html')
    assert 'id="smart-fade-needs-neural" class="smart-fade-needs-neural" hidden>' in _panel(template)
    neural = re.search(r'id="ml-model-neural-fingerprint-details" hidden>(.*?)</div>', template, re.DOTALL).group(1)
    assert '<strong>Smart fade:</strong>' in neural


def test_the_other_smart_fade_parameters_sit_in_the_panel_and_every_one_is_documented():
    source = _read('static', 'setup.js')
    knobs = sorted(name for name in dir(config) if name.startswith('SMART_FADE_') and name not in FLAGS)
    tuning = re.findall(r"'([A-Z_]+)'", re.search(r'var SMART_FADE_TUNING = \[(.*?)\];', source, re.DOTALL).group(1))
    assert knobs and sorted(tuning) == knobs and len(tuning) == len(knobs)
    assert 'SMART_FADE_' not in _js_block(source, 'var ADVANCED_SECTIONS', 'var ADVANCED_OTHER_TITLE')
    for name in knobs:
        assert app_setup.should_show_advanced(name) and name not in app_setup.BASIC_FIELDS, name
    load = _js_block(source, 'function loadSetupData', 'function saveCurrentServerValues')
    assert '!SMART_FADE_TUNING.includes(f.name)' in re.search(r'visibleAdvancedData = (.*?);', load, re.DOTALL).group(1)
    render = _js_block(source, 'function renderSmartFadeSwitches', 'function fieldsByName')
    assert "getElementById('smart-fade-fields')" in render
    assert "renderAdvancedSection(tuning, 'Fine tuning', SMART_FADE_TUNING, fieldsByName(fields), {})" in render
    assert render.index('renderAdvancedSection(tuning') < render.index('updateSmartFadeAvailability()')
    panel = _panel(_read('templates', 'setup.html'))
    assert panel.index('class="ml-model-list"') < panel.index('<div id="smart-fade-fields"></div>')
    parameters = _read('docs', 'PARAMETERS.md')
    for name in knobs + list(FLAGS):
        row = re.search(r'^\| `' + name + r'` .*\| `([^`]+)` +\|$', parameters, re.MULTILINE)
        assert row, name + ' has no PARAMETERS.md row'
        default = getattr(config, name)
        expected = str(default).lower() if isinstance(default, bool) else ('%g' % default)
        assert row.group(1) == expected, name


def test_the_panel_tests_two_picked_songs_with_the_unsaved_switches_through_the_shared_player():
    template = _read('templates', 'setup.html')
    panel = _panel(template)
    assert panel.index('class="ml-model-list"') < panel.index('id="smart-fade-test"') < panel.index('id="smart-fade-fields"')
    box = panel[panel.index('id="smart-fade-test"'):panel.index('<div id="smart-fade-fields"></div>')]
    assert '{% if neural_enabled and setup_saved %}' in box and 'needs a saved configuration' in box
    for picker in ('smart-fade-test-first', 'smart-fade-test-second'):
        assert '<input type="text" id="%s"' % picker in box
        assert '<div id="%s-results" class="autocomplete-results hidden"></div>' % picker in box
    assert 'name=' not in box and '<div id="smart-fade-test-list"></div>' in box
    script = template[template.index("var list = document.getElementById('smart-fade-test-list');"):]
    script = script[:script.index('</script>')]
    assert template.rindex('{% if neural_enabled and setup_saved %}') < template.index(script)
    assert 'AudioMuseAutocomplete.attach({' in script and "url_for('ivf_bp.search_tracks_endpoint')" in script
    assert '#smart-fade-config .ml-model-row[data-transition] input[type="checkbox"][data-flag]' in script
    assert '{ keepOrder: true, smartFade: true, transitions: transitions }' in script
    assert 'list.innerHTML = renderSongList(rows, {});' in script and 'AudioMusePlayer.show(list);' in script
    assert "if (event.key === 'Enter')" in script and 'fetch(' not in script
    layout = _read('templates', 'includes', 'layout.html')
    assert "'album_creation', 'setup'] %}" in layout
