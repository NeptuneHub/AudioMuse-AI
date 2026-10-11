# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Structural guards for the setup wizard's P2P Sharing section.

There is no javascript runner in this repo, so setup.html and setup.js are
pinned by reading them as sources: where the section sits, which form fields it
owns, the privacy notice, the bounds of its number inputs, and the script that
loads, guards, mirrors and refreshes it.

Main Features:
* The collapsed section sits after AI Prompt and before the save button, inside
  the form and outside the lyrics gate, and carries the privacy notice verbatim
* It submits exactly the five P2P settings, the switch through a hidden input
  with no data-flag, and every number input's static bounds match the backend
* The script never submits the settings before they were loaded, mirrors the
  switch, reads the status only on open or Refresh (no timer) and locks the
  switch when Tor or the P2P libraries are missing
* The advanced list gains a "P2P Sharing internals" group of visible keys
"""

import os
import re

import app_setup

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)

PRIVACY_NOTICE = (
    'P2P Sharing is off until you turn it on. When on, this server swaps analysis results with other '
    'AudioMuse-AI servers through the Tor network, so new songs can skip slow analysis steps. No open '
    'port or public IP address is needed and no server run by the project is involved. Other servers '
    "never see this server's IP address: they see its onion address, its AudioMuse-AI version, a short "
    'audio signature and the length of each song it looks up, and whether this library has a song they '
    'ask about. The onion address, the AudioMuse-AI version and the models in use are listed in a public '
    'directory on Nostr relays. Audio, file names, titles, artists, lyrics text, play history and '
    'account details are never sent. Only results from AudioMuse-AI v3.0.0 or newer are exchanged, and '
    'only when the song matches.'
)
INTERNAL_FIELDS = ('P2P_MAX_PEERS', 'P2P_LOOKUP_TIMEOUT_SECONDS')


def _read(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path), encoding='utf-8') as handle:
        return handle.read()


def _template():
    return _read(os.path.join('templates', 'setup.html'))


def _setup_js():
    return _read(os.path.join('static', 'setup.js'))


def _section():
    markup = _template()
    start = markup.index('<details id="p2p-config">')
    return markup[start:markup.index('</details>', start) + len('</details>')]


def _text(html):
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', html)).strip()


def _function(name):
    source = _setup_js()
    body = source[source.index('function ' + name):]
    return body[:body.index('\n}') + 2]


def _p2p_script():
    source = _setup_js()
    start = source.index('var P2P_FORM_FIELDS')
    return source[start:source.index('// Lyrics API section', start)]


class TestPlacement:
    def test_the_section_is_collapsed_by_default(self):
        match = re.search(r'<details id="p2p-config"([^>]*)>', _template())
        assert match and 'open' not in match.group(1)

    def test_the_section_sits_after_ai_prompt_and_before_the_save_button(self):
        markup = _template()
        ai_end = markup.index('</details>', markup.index('<details id="ai-prompt-config">'))
        p2p_at = markup.index('<details id="p2p-config">')
        assert ai_end < p2p_at < markup.index('id="save-button"')
        between = markup[ai_end + len('</details>'):p2p_at]
        assert between.strip() == ''

    def test_the_section_is_inside_the_form_and_outside_the_lyrics_gate(self):
        markup = _template()
        p2p_at = markup.index('<details id="p2p-config">')
        assert markup.index('<form id="setup-form"') < p2p_at < markup.index('</form>')
        assert markup.index('{% endif %}', markup.index('{% if lyrics_enabled %}')) < p2p_at
        assert '{% if' not in _section()

    def test_the_summary_names_the_section(self):
        assert '<summary>P2P Sharing</summary>' in _section()

    def test_the_privacy_notice_is_shown_verbatim(self):
        notice = re.search(r'<div class="notice"[^>]*>(.*?)</div>', _section(), re.S)
        assert notice and _text(notice.group(1)) == PRIVACY_NOTICE

    def test_the_section_is_plain_ascii_with_no_em_dash(self):
        section = _section()
        assert section.isascii()
        assert '&mdash;' not in section

    def test_the_section_names_no_removed_network_piece(self):
        text = _text(_section()).lower()
        for marker in ('libp2p', 'main network server', 'main server', 'bootstrap', 'upnp', '4101'):
            assert marker not in text, marker


class TestFields:
    def test_only_the_p2p_settings_are_form_fields(self):
        names = re.findall(r'\bname="([^"]+)"', _section())
        assert sorted(names) == sorted(app_setup.P2P_CONFIG_FIELDS)

    def test_the_switch_is_mirrored_into_a_hidden_input_without_a_flag(self):
        section = _section()
        assert '<input type="hidden" id="P2P_ENABLED" name="P2P_ENABLED" value="false">' in section
        switch = re.search(r'<input type="checkbox" id="p2p-enabled-switch"[^>]*>', section)
        assert switch
        assert 'data-flag' not in switch.group(0) and 'name=' not in switch.group(0)
        assert 'data-flag' not in section

    def test_every_number_input_carries_the_backend_bounds(self):
        section = _section()
        for key, (low, high, _label) in app_setup.P2P_NUMBER_RULES.items():
            tag = re.search(r'<input id="%s"[^>]*>' % key, section)
            assert tag, key
            tag = tag.group(0)
            assert 'type="number"' in tag, key
            assert 'min="%d"' % low in tag, key
            if high is None:
                assert 'max=' not in tag, key
            else:
                assert 'max="%d"' % high in tag, key
            assert 'value="{{' not in tag, key

    def test_the_rate_labels_say_kib_per_second(self):
        section = _section()
        for key in ('P2P_MAX_DOWNLOAD_KIBPS', 'P2P_MAX_UPLOAD_KIBPS'):
            label = re.search(r'<label for="%s">([^<]*)</label>' % key, section)
            assert label and 'KiB/s' in label.group(1), key

    def test_the_status_line_and_refresh_controls_exist(self):
        section = _section()
        for element_id in ('p2p-status-word', 'p2p-status-refresh', 'p2p-unavailable-note'):
            assert 'id="%s"' % element_id in section, element_id

    def test_the_network_group_and_the_identity_reset_are_gone(self):
        section = _section()
        for marker in ('<textarea', '<select', '>Network<', 'Network identity', 'p2p-reset'):
            assert marker not in section, marker

    def test_nothing_forces_a_width_wider_than_a_phone(self):
        section = _section()
        for match in re.finditer(r'(?:min-)?width:\s*(\d+)px', section):
            assert int(match.group(1)) <= 300, match.group(0)
        assert 'white-space:nowrap' not in section.replace(' ', '')

    def test_the_section_reuses_the_toolbar_styles(self):
        markup = _template()
        assert '#p2p-config .ai-prompt-toolbar button' in markup
        assert '#p2p-config > summary' in markup


class TestScript:
    def test_the_form_field_list_is_the_backend_list(self):
        match = re.search(r'var P2P_FORM_FIELDS = new Set\(\[(.*?)\]\);', _setup_js(), re.S)
        assert match
        assert re.findall(r"'([A-Z0-9_]+)'", match.group(1)) == app_setup.P2P_CONFIG_FIELDS

    def test_fields_are_not_submitted_before_the_section_is_loaded(self):
        collect = _function('collectConfigFromForm')
        assert '!p2pState.loaded && P2P_FORM_FIELDS.has(key)' in collect

    def test_the_payload_is_populated_on_load(self):
        load = _function('loadSetupData')
        assert 'populateP2pFields(data.p2p_fields)' in load
        assert load.index('populateAiPromptFields') < load.index('populateP2pFields')

    def test_populate_remembers_originals_marks_loaded_and_follows_availability(self):
        populate = _function('populateP2pFields')
        assert 'input.dataset.originalValue = input.value' in populate
        assert 'p2pState.loaded = true' in populate
        assert "p2pState.available = p2pData.available !== false" in populate
        assert "toggle.checked = hidden.value === 'true'" in populate
        assert 'widenP2pNumberBounds(input, value)' in populate
        assert populate.index('p2pState.loaded = true') > populate.index('P2P_FORM_FIELDS.forEach')

    def test_a_loaded_value_outside_the_static_bounds_widens_them(self):
        widen = _function('widenP2pNumberBounds')
        assert 'input.min = String(number)' in widen
        assert 'input.max = String(number)' in widen

    def test_the_switch_is_locked_while_p2p_is_unavailable_unless_it_must_be_turned_off(self):
        lock = _function('updateP2pSwitchLock')
        assert 'toggle.disabled = !p2pState.available && !toggle.checked' in lock

    def test_the_switch_writes_true_or_false_into_the_hidden_input(self):
        script = _p2p_script()
        assert "hidden.value = _p2pSwitch.checked ? 'true' : 'false'" in script

    def test_the_status_is_read_on_open_and_on_refresh_only(self):
        script = _p2p_script()
        assert "fetch('/api/setup/p2p/status'" in _function('refreshP2pStatus')
        assert "_p2pDetails.addEventListener('toggle'" in script
        assert 'if (_p2pDetails.open)' in script
        assert "_p2pStatusRefresh.addEventListener('click', refreshP2pStatus)" in script
        assert 'setInterval' not in script and 'setTimeout' not in script

    def test_the_status_word_is_text_not_html(self):
        refresh = _function('refreshP2pStatus')
        assert 'word.textContent' in refresh and 'innerHTML' not in refresh

    def test_the_script_only_calls_the_status_route(self):
        script = _p2p_script()
        assert re.findall(r"fetch\('(/api/setup/p2p/[^']*)'", script) == ['/api/setup/p2p/status']
        assert 'window.confirm' not in script

    def test_the_script_is_plain_ascii(self):
        assert _p2p_script().isascii()


class TestAdvancedGroup:
    def _sections(self):
        source = _setup_js()
        return source[source.index('var ADVANCED_SECTIONS'):source.index('var ADVANCED_OTHER_TITLE')]

    def test_the_internals_have_their_own_group_after_the_plugin_system(self):
        sections = self._sections()
        group = re.search(r"title: 'P2P Sharing internals',\s*items: \[(.*?)\]", sections, re.S)
        assert group
        assert tuple(re.findall(r"'([A-Z0-9_]+)'", group.group(1))) == INTERNAL_FIELDS
        assert sections.index("title: 'Plugin System'") < sections.index("title: 'P2P Sharing internals'")

    def test_the_internals_are_visible_and_the_section_settings_are_not_listed(self):
        sections = self._sections()
        for field in INTERNAL_FIELDS:
            assert app_setup.should_show_advanced(field), field
        for field in app_setup.P2P_CONFIG_FIELDS:
            assert "'%s'" % field not in sections, field
