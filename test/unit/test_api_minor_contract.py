# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Four small API-contract defects a live fuzz of the running instance found.

None of them could corrupt anything; all four told a caller something untrue.
provider_echo_id answers None for an internal id that is not on the selected
server - that is the fp_-leak guard doing its job - but three error messages
rendered that None into the text, so a caller got "Item 'None' not found". The
instant-playlist success message named the playlist the caller ASKED for, while
every provider saves it with its own suffix, so a client that looked the name up
afterwards found nothing. And the published OpenAPI disagreed with the server on
two points a client codes against.

Main Features:
* A not-found message never contains the literal "None"
* It still quotes a real provider id when there is one to quote
* The instant-playlist message does not name a playlist the server did not create
* The published spec's query minLength and steering weight match the server
"""

import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _source(name):
    return (REPO / name).read_text(encoding='utf-8')


class TestANotFoundMessageNeverSaysNone:
    def test_no_handler_interpolates_provider_echo_id_straight_into_a_message(self):
        source = _source('app_ivf.py')
        offenders = re.findall(r"f\"[^\"]*\{app_server_context\.provider_echo_id\([^\"]*\"", source)
        assert not offenders, (
            'provider_echo_id returns None for an id that is not on the selected '
            'server, so interpolating it directly produced "Item \'None\' not '
            f'found". Guard the None first. Offenders: {offenders}'
        )

    def test_every_echo_site_has_a_fallback_phrase(self):
        source = _source('app_ivf.py')
        calls = source.count('app_server_context.provider_echo_id(')
        guarded = source.count('echo = app_server_context.provider_echo_id(')
        assert calls and calls == guarded, (
            f'{calls - guarded} call site(s) still use the echo id without first '
            'checking it for None'
        )
        assert source.count('not on this server') == calls, (
            'each not-found message needs its own id-less fallback'
        )


class TestTheInstantPlaylistMessageIsHonest:
    def test_it_does_not_quote_the_requested_name(self):
        source = _source('app_ivf.py')
        assert "f\"Playlist '{playlist_name}' created" not in source, (
            'every provider saves an instant playlist under its own suffixed '
            'name, so quoting the requested name announced a playlist that does '
            'not exist under that name'
        )

    def test_it_still_reports_the_counts_and_the_id(self):
        source = _source('app_ivf.py')
        assert '"message": f"Playlist created on the selected server' in source
        assert '"playlist_id": new_playlist_id' in source


class TestThePublishedSpecMatchesTheServer:
    @pytest.mark.parametrize('module,guard', [
        ('app_clap_search.py', 'Query must be at least 1 character'),
        ('app_lyrics.py', 'Query must be at least 1 character.'),
    ])
    def test_the_query_min_length_is_the_one_the_handler_enforces(self, module, guard):
        source = _source(module)
        assert guard in source, 'the handler still enforces a 1-character floor'
        assert 'minLength: 3' not in source, (
            f'{module} published minLength 3 while accepting 1, so a client '
            'generated from the spec rejects queries the server would answer'
        )
        assert 'minLength: 1' in source

    def test_the_steering_weight_is_not_published_as_an_enum(self):
        import config

        source = _source('app_clap_search.py')
        assert 'enum: [0.1, 0.2, 0.5, 1.0, 2.0]' not in source, (
            'the server snaps any number to CLAP_SAE_ALPHA_STEPS instead of '
            'rejecting it, and those steps are not those five values'
        )
        assert f'default: {config.CLAP_SAE_DEFAULT_ALPHA}' in source
        assert config.CLAP_SAE_ALPHA_STEPS == [1.0, 2.0, 3.0, 5.0]

    def test_the_weight_really_is_snapped_not_rejected(self):
        from tasks.clap_steering import _snap_weight

        import config

        assert _snap_weight(0.1) in config.CLAP_SAE_ALPHA_STEPS
        assert _snap_weight(99) == max(config.CLAP_SAE_ALPHA_STEPS)
        assert _snap_weight(2.2) == 2.0
