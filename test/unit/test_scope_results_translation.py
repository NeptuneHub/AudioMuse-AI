# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""A list-of-ids API response must never expose the internal canonical (fp_) id.

scope_results / filter_rows_for_request_server rewrite every surviving row's id to
the request server's OWN provider id, so a Jellyfin/Navidrome plugin gets ids it can
use. Internal callers (a pool handed to create_instant_playlist_for_server, which
re-translates) opt out with translate=False and keep the canonical id.

Main Features:
* Default translate=True rewrites item_id to the server's provider id.
* Rows with no mapping on the target server are dropped (availability filter).
* translate=False keeps the canonical id for internal, re-translated flows.
"""

import app_server_context
from tasks.mediaserver import registry


def _wire(monkeypatch, server_id, mapping):
    monkeypatch.setattr(app_server_context, 'resolve_request_server_id',
                        lambda *a, **k: server_id)
    monkeypatch.setattr(registry, 'has_secondary_servers', lambda *a, **k: True)
    # translate_ids returns {canonical_item_id: provider_track_id} for mapped ids only
    monkeypatch.setattr(registry, 'translate_ids',
                        lambda ids, sid, conn=None: {i: mapping[i] for i in ids if i in mapping})


def test_output_id_is_rewritten_to_the_servers_provider_id(monkeypatch):
    _wire(monkeypatch, 'srv1', {'fp_2aaa': 'jelly-1', 'fp_2bbb': 'jelly-2'})
    rows = [{'item_id': 'fp_2aaa', 'title': 'A'}, {'item_id': 'fp_2bbb', 'title': 'B'}]

    out = app_server_context.scope_results(rows, None, id_key='item_id')

    assert [r['item_id'] for r in out] == ['jelly-1', 'jelly-2']
    assert not any(str(r['item_id']).startswith('fp_') for r in out)


def test_rows_not_on_the_server_are_dropped(monkeypatch):
    _wire(monkeypatch, 'srv1', {'fp_2aaa': 'jelly-1'})  # fp_2bbb not on this server
    rows = [{'item_id': 'fp_2aaa', 'title': 'A'}, {'item_id': 'fp_2bbb', 'title': 'B'}]

    out = app_server_context.scope_results(rows, None, id_key='item_id')

    assert [r['item_id'] for r in out] == ['jelly-1']


def test_translate_false_keeps_canonical_id_for_internal_use(monkeypatch):
    _wire(monkeypatch, 'srv1', {'fp_2aaa': 'jelly-1', 'fp_2bbb': 'jelly-2'})
    rows = [{'item_id': 'fp_2aaa'}, {'item_id': 'fp_2bbb'}]

    out = app_server_context.scope_results(rows, None, id_key='item_id', translate=False)

    assert [r['item_id'] for r in out] == ['fp_2aaa', 'fp_2bbb']


def test_requested_n_still_trims_after_translation(monkeypatch):
    _wire(monkeypatch, 'srv1', {'fp_2aaa': 'jelly-1', 'fp_2bbb': 'jelly-2'})
    rows = [{'item_id': 'fp_2aaa'}, {'item_id': 'fp_2bbb'}]

    out = app_server_context.scope_results(rows, 1, id_key='item_id')

    assert out == [{'item_id': 'jelly-1'}]


def _wire_translate_error(monkeypatch):
    monkeypatch.setattr(app_server_context, 'resolve_request_server_id',
                        lambda *a, **k: 'srv1')
    monkeypatch.setattr(registry, 'has_secondary_servers', lambda *a, **k: True)

    def _boom(ids, sid, conn=None):
        raise RuntimeError('registry down')
    monkeypatch.setattr(registry, 'translate_ids', _boom)


def test_translation_error_fails_closed_dropping_fp_but_keeping_legacy_ids(monkeypatch):
    _wire_translate_error(monkeypatch)
    rows = [{'item_id': 'fp_2aaa'}, {'item_id': 'legacy-1'}]

    out = app_server_context.scope_results(rows, None, id_key='item_id')

    assert [r['item_id'] for r in out] == ['legacy-1']
    assert not any(str(r['item_id']).startswith('fp_') for r in out)


def test_translate_ids_for_request_drops_fp_and_keeps_legacy_on_error(monkeypatch):
    _wire_translate_error(monkeypatch)

    mapping = app_server_context.translate_ids_for_request(['fp_2aaa', 'legacy-1'])

    assert 'fp_2aaa' not in mapping
    assert mapping.get('legacy-1') == 'legacy-1'
