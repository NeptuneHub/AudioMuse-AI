# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Assertion helpers shared by the end-to-end test modules.

The catalogue ids (fp_...) are internal and must never reach an API response,
so every payload a test receives goes through assert_no_fp_ids. The other
helpers read the two tables that map provider ids to catalogue rows and create
the playlists the tests then look for on Navidrome.

Main Features:
* assert_no_fp_ids walks any JSON payload and fails on a leaked fp_ id
* item_id_of / provider_ids_for translate between provider and catalogue ids
  straight from track_server_map
* create_playlist posts to /api/create_playlist and returns the checked body
"""

import uuid


def assert_no_fp_ids(payload, where='response'):
    found = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f'{path}.{key}')
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f'{path}[{index}]')
        elif isinstance(node, str) and node.startswith('fp_'):
            found.append(f'{path}={node[:20]}...')

    walk(payload, where)
    assert not found, f'internal fp_ ids leaked into the API payload: {found[:10]}'


def unique_name(prefix):
    return f'e2e-{prefix}-{uuid.uuid4().hex[:8]}'


def rows(db, sql, params=None):
    with db.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def scalar(db, sql, params=None):
    result = rows(db, sql, params)
    return None if not result else result[0][0]


def default_server_id(db):
    return scalar(db, 'SELECT server_id FROM music_servers WHERE is_default')


def item_id_of(db, provider_id, server_id=None):
    server_id = server_id or default_server_id(db)
    return scalar(
        db,
        'SELECT item_id FROM track_server_map WHERE server_id = %s AND provider_track_id = %s',
        (server_id, provider_id),
    )


def provider_ids_for(db, item_id, server_id=None):
    server_id = server_id or default_server_id(db)
    return [
        r[0]
        for r in rows(
            db,
            'SELECT provider_track_id FROM track_server_map WHERE server_id = %s AND item_id = %s '
            'ORDER BY provider_track_id',
            (server_id, item_id),
        )
    ]


def create_playlist(api, name, track_ids, expect=201):
    body = api.json(
        'POST', '/api/create_playlist', expect=expect,
        json={'playlist_name': name, 'track_ids': list(track_ids)},
    )
    assert_no_fp_ids(body)
    return body


def delete_navidrome_playlists(navidrome, names):
    wanted = set(names)
    return navidrome.delete_playlists_named(lambda name: name in wanted)
