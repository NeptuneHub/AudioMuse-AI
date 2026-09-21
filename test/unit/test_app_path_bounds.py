# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The Song Path API clamps max_steps the way its own page already does.

The handler clamped only the low end (max(1, n)), so the page's own cap of 200
was advisory: a hand-edited URL or a script with an extra zero asked for any
number of steps. Measured on the live 200k-song instance, max_steps=1000 held a
request thread for 34 seconds against 1.2 seconds at the default, and this app
is one gunicorn worker with a handful of threads, so a few such calls starve
every other page. The ceiling is config.PATH_MAX_LENGTH, matching the page.

Main Features:
* A request above the ceiling is clamped to it rather than refused
* The low-end clamp and the config default are unchanged
* A value inside the range reaches the path finder untouched
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import config  # noqa: E402
from app_path import path_bp  # noqa: E402


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(path_bp)
    app.config['TESTING'] = True
    return app.test_client()


@pytest.fixture
def asked():
    calls = []

    def _find(start_id, end_id, max_steps, **kwargs):
        calls.append(max_steps)
        return []

    with patch('app_path.find_path_between_songs', side_effect=_find):
        yield calls


def _get(client, steps):
    return client.get(f'/api/find_path?start_song_id=a&end_song_id=b&max_steps={steps}')


class TestTheStepCeiling:
    @pytest.mark.parametrize('asked_for', [201, 1000, 100000])
    def test_a_request_above_the_ceiling_is_clamped_to_it(self, client, asked, asked_for):
        _get(client, asked_for)

        assert asked and asked[0] == config.PATH_MAX_LENGTH, (
            'an unbounded max_steps let one GET hold a request thread for '
            'minutes; the API must cap it where the page caps it'
        )

    @pytest.mark.parametrize('asked_for', [1, 25, 200])
    def test_a_value_inside_the_range_is_passed_through(self, client, asked, asked_for):
        _get(client, asked_for)

        assert asked and asked[0] == asked_for

    @pytest.mark.parametrize('asked_for', [0, -5])
    def test_the_low_end_still_clamps_to_one(self, client, asked, asked_for):
        _get(client, asked_for)

        assert asked and asked[0] == 1

    def test_the_ceiling_is_at_least_the_default_so_the_default_is_reachable(self):
        assert config.PATH_MAX_LENGTH >= config.PATH_DEFAULT_LENGTH
