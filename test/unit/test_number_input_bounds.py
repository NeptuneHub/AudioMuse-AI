# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Guard that a legal config value never renders a form the browser refuses.

Every search page seeds its count input from a config key and also carries a
hard-coded min / max. When an operator moves the config value outside that pair,
an <input type="number"> whose value sits outside its own bounds fails HTML5
constraint validation and the submit button silently does nothing. The min_bound
/ max_bound filters widen the rendered bounds to admit the value they are given,
so the hard-coded pair stays the guidance it was meant to be rather than a trap.

Main Features:
* min_bound never returns a floor above the value it is given, for the integer
  count boxes and the float radial-spread / ancestry-dive sliders alike
* max_bound never returns a ceiling below the value it is given
* Both leave the hard-coded bound alone when the value sits inside it
* A non-numeric or missing value falls back to the hard-coded bound
* app.py registers both as Jinja filters under the names the templates use
* Every server-rendered number input in templates/ uses both filters, so a new
  page cannot reintroduce the trap
"""

import glob
import os
import re

import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
TEMPLATES = os.path.join(REPO_ROOT, 'templates')

NUMBER_INPUT = re.compile(r'<input[^>]*type="number"[^>]*>', re.S)


def _filters():
    from app_helper import max_bound, min_bound

    return min_bound, max_bound


class TestTheBoundsAlwaysAdmitTheRenderedValue:
    @pytest.mark.parametrize('value,floor,expected', [
        (20, 40, 20),
        (50, 40, 40),
        (40, 40, 40),
        (1, 1, 1),
    ])
    def test_min_bound_never_sits_above_the_value(self, value, floor, expected):
        min_bound, _ = _filters()

        assert min_bound(value, floor) == expected

    @pytest.mark.parametrize('value,ceiling,expected', [
        (300, 200, 300),
        (50, 200, 200),
        (200, 200, 200),
    ])
    def test_max_bound_never_sits_below_the_value(self, value, ceiling, expected):
        _, max_bound = _filters()

        assert max_bound(value, ceiling) == expected

    @pytest.mark.parametrize('value,floor,expected', [
        (0.15, 0, 0),
        (-0.1, 0, -0.1),
    ])
    def test_min_bound_handles_the_float_sliders_too(self, value, floor, expected):
        min_bound, _ = _filters()

        assert min_bound(value, floor) == expected

    @pytest.mark.parametrize('value,ceiling,expected', [
        (1.5, 0.99, 1.5),
        (0.15, 0.99, 0.99),
    ])
    def test_max_bound_handles_the_float_sliders_too(self, value, ceiling, expected):
        _, max_bound = _filters()

        assert max_bound(value, ceiling) == expected

    @pytest.mark.parametrize('bad', [None, '', 'abc'])
    def test_a_non_numeric_value_falls_back_to_the_hard_coded_bound(self, bad):
        min_bound, max_bound = _filters()

        assert min_bound(bad, 40) == 40
        assert max_bound(bad, 200) == 200

    def test_a_config_value_outside_the_hard_coded_pair_still_renders_a_valid_input(self):
        min_bound, max_bound = _filters()
        value = 300

        assert min_bound(value, 40) <= value <= max_bound(value, 200)


class TestTheFiltersAreRegisteredUnderTheNamesTheTemplatesUse:
    def test_app_registers_both_filters(self):
        import re as _re

        with open(os.path.join(REPO_ROOT, 'app.py'), 'r', encoding='utf-8') as handle:
            source = handle.read()
        for name in ('min_bound', 'max_bound'):
            assert _re.search(r"add_template_filter\([^)]*'%s'\)" % name, source), (
                '%s is used by templates/ but app.py does not register it' % name
            )


class TestNoPageCanReintroduceTheTrap:
    def _server_rendered_number_inputs(self):
        found = []
        for path in sorted(glob.glob(os.path.join(TEMPLATES, '*.html'))):
            with open(path, 'r', encoding='utf-8') as handle:
                text = handle.read()
            for tag in NUMBER_INPUT.findall(text):
                if 'value="{{' in tag:
                    found.append((os.path.basename(path), tag))
        return found

    def test_the_scan_finds_the_known_pages(self):
        names = {name for name, _ in self._server_rendered_number_inputs()}

        assert len(names) >= 8, 'expected every search page to seed its count input: %s' % names

    def test_every_server_seeded_number_input_widens_both_bounds(self):
        offenders = []
        for name, tag in self._server_rendered_number_inputs():
            if 'min=' in tag and 'min_bound' not in tag:
                offenders.append('%s: min= is hard coded against a rendered value' % name)
            if 'max=' in tag and 'max_bound' not in tag:
                offenders.append('%s: max= is hard coded against a rendered value' % name)
        assert not offenders, (
            'a number input seeded from config must widen its bounds with '
            'min_bound / max_bound, or a legal config value renders an '
            'unsubmittable form:\n' + '\n'.join(offenders)
        )
