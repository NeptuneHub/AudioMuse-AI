# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Sidebar menu: a feature whose master switch is off leaves the menu.

Main Features:
* Text Search (DCLAP), Lyrics Search and Search by Recording are each wrapped
  in the guard of their own flag in the sidebar template
* The layout context processor hands the template all three flags straight
  from config, so the guards see the running configuration
"""

import os
import re

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

_ENTRIES = {
    'clap_enabled': 'Text Search (DCLAP)',
    'lyrics_enabled': 'Lyrics Search',
    'neural_enabled': 'Search by Recording',
}


def _read(*parts):
    with open(os.path.join(_ROOT, *parts), encoding='utf-8') as handle:
        return handle.read()


def test_each_switchable_page_is_guarded_by_its_flag_in_the_sidebar():
    sidebar = _read('templates', 'sidebar_navi.html')
    for flag, label in _ENTRIES.items():
        lines = [line for line in sidebar.splitlines() if f'>{label}</a>' in line]
        assert len(lines) == 1, label
        assert re.search(r'\{%\s*if\s+' + flag + r'\s*%\}.*' + re.escape(label) + r'.*\{%\s*endif\s*%\}', lines[0]), (flag, label)


def test_the_layout_context_hands_the_sidebar_all_three_flags_from_config():
    app_source = _read('app.py')
    start = app_source.index('def inject_globals():')
    end = app_source.index('\n@app.', start)
    body = app_source[start:end]
    for flag, setting in (
        ('clap_enabled', 'CLAP_ENABLED'),
        ('lyrics_enabled', 'LYRICS_ENABLED'),
        ('neural_enabled', 'NEURAL_FINGERPRINT_ENABLED'),
    ):
        assert f'{flag}={setting},' in body, flag
        assert setting in body.split('return dict(')[0], setting
