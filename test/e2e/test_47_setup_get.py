# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The setup wizard's read side on a configured, analyzed instance.

GET /api/setup is what the wizard renders: the instance reports itself as
configured, the model coverage bands of every index are at the top once the library
is analyzed, secret fields never carry a value, and fields of media-server
types other than the active one are blank. POST is out of scope here because
it publishes a worker restart that needs the control plane.

Main Features:
* setup_saved is true and every model coverage band is the top one after analysis
* secret fields are masked; inactive provider fields carry no value
"""

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids

pytestmark = pytest.mark.e2e

TOP_COVERAGE_BAND = 5


def _fields(setup):
    return list(setup.get('basic_fields', [])) + list(setup.get('advanced_fields', []))


def test_setup_reports_configured_and_covered(stack, api, analyzed_library):
    setup = api.json('GET', '/api/setup')
    assert_no_fp_ids(setup)
    assert setup['setup_saved'] is True
    assert isinstance(setup['has_admin_user'], bool)
    coverage = setup['model_coverage']
    assert coverage, setup
    assert all(band == TOP_COVERAGE_BAND for band in coverage.values()), coverage
    assert isinstance(setup.get('music_libraries', ''), str)


def test_secret_and_inactive_fields_are_blank(stack, api, analyzed_library):
    setup = api.json('GET', '/api/setup')
    fields = _fields(setup)
    assert fields, setup
    secret = [f for f in fields if f.get('secret')]
    assert secret, 'the wizard must expose at least one secret field'
    for field in secret:
        assert field.get('value') in ('', None), field
    jellyfin = [f for f in fields if str(f.get('key', f.get('name', ''))).startswith('JELLYFIN_')]
    for field in jellyfin:
        assert field.get('value') in ('', None), field
