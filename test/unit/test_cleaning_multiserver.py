# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Multi-server library cleanup: unbind-only semantics.

Drives identify_and_clean_orphaned_albums_task with the media-server registry,
provider fetches and DB helpers faked, asserting that cleanup NEVER deletes
catalogue rows: it only prunes each healthy server's own track_server_map rows,
skips servers whose library view is incomplete, and reports (without touching)
the tracks currently bound to no server.

Main Features:
* A failed or empty fetch skips ONLY that server's unbinding, others proceed
* Per-server pruning receives exactly that server's present provider ids
* Full coverage across servers unbinds nothing and reports zero orphans
* Tracks on no server are reported as kept, never deleted
"""

import sys
import types

from unittest.mock import MagicMock

from flask import Flask

import config


def _server(server_id, name, default=False):
    return {
        'server_id': server_id, 'name': name, 'server_type': 'jellyfin',
        'creds': {}, 'music_libraries': '', 'is_default': default, 'enabled': True,
    }


def _run_cleaning(monkeypatch, servers, albums_by_server, tracks_by_album,
                  reverse_by_server, db_track_ids, author_by_id=None,
                  prune_results=None):
    from tasks import cleaning
    from tasks import multiserver_sync

    statuses = []
    pruned_calls = []
    authors = author_by_id or {}

    fake_flask_app = types.ModuleType('flask_app')
    fake_flask_app.app = Flask('cleaning-test')
    monkeypatch.setitem(sys.modules, 'flask_app', fake_flask_app)

    cur = MagicMock()
    state = {'last': (None, None)}

    def record_execute(sql, params=None):
        state['last'] = (sql, params)

    def answer_fetchall():
        sql, params = state['last']
        if sql and 'JOIN embedding' in sql:
            return [(item_id,) for item_id in sorted(db_track_ids)]
        if sql and sql.startswith('SELECT item_id, title, author FROM score'):
            return [
                (item_id, f'Title {item_id}', authors.get(item_id, f'Artist {item_id}'))
                for item_id in params[0]
            ]
        return []

    cur.execute.side_effect = record_execute
    cur.fetchall.side_effect = answer_fetchall
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    get_db_cm = MagicMock()
    get_db_cm.__enter__.return_value = conn
    get_db_cm.__exit__.return_value = False

    fake_app_helper = types.ModuleType('app_helper')
    fake_app_helper.redis_conn = object()
    fake_app_helper.get_db = lambda: get_db_cm
    fake_app_helper.save_task_status = (
        lambda task_id, task_type, status, progress=None, details=None:
        statuses.append((status, progress, details))
    )
    monkeypatch.setitem(sys.modules, 'app_helper', fake_app_helper)

    monkeypatch.setattr(cleaning, 'get_current_job', lambda *a, **k: None)
    monkeypatch.setattr(
        cleaning.registry, 'servers_for_scope', lambda scope, conn=None: servers
    )
    by_id = {s['server_id']: s for s in servers}
    monkeypatch.setattr(
        cleaning.registry, 'context_for', lambda sid, conn=None: by_id[sid]
    )

    def fake_reverse(chunk, server_id, conn=None):
        mapping = reverse_by_server.get(server_id, {})
        return {pid: mapping[pid] for pid in chunk if pid in mapping}

    monkeypatch.setattr(cleaning.registry, 'reverse_translate_ids', fake_reverse)

    def fake_recent_albums(limit):
        sid = cleaning.ms_context.active_server_id()
        result = albums_by_server[sid]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(cleaning, 'get_recent_albums', fake_recent_albums)
    monkeypatch.setattr(
        cleaning, 'get_tracks_from_album', lambda album_id: tracks_by_album[album_id]
    )

    def fake_prune(db, server_id, present_ids):
        pruned_calls.append((server_id, sorted(present_ids)))
        return (prune_results or {}).get(server_id, 0)

    monkeypatch.setattr(multiserver_sync, '_prune_stale_mappings', fake_prune)

    result = cleaning.identify_and_clean_orphaned_albums_task()
    return result, statuses, pruned_calls


class TestCleaningSkipsUnreadableServers:
    def test_failed_fetch_skips_that_server_but_prunes_the_healthy_one(self, monkeypatch):
        result, statuses, pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            albums_by_server={
                's1': RuntimeError('fetch failed'),
                's2': [{'Id': 'alb2', 'Name': 'A2'}],
            },
            tracks_by_album={'alb2': [{'Id': 'n1'}]},
            reverse_by_server={'s2': {'n1': 'fp_1'}},
            db_track_ids={'fp_1', 'fp_2'},
            prune_results={'s2': 3},
        )
        assert result['status'] == 'FAILURE'
        assert result['deleted_count'] == 0
        assert 'One' in result['failed_servers']
        assert pruned == [('s2', ['n1'])]
        assert result['unbound_mappings'] == 3
        assert statuses[-1][0] == config.TASK_STATUS_FAILURE

    def test_zero_albums_skips_that_server_and_reports_no_orphans(self, monkeypatch):
        result, statuses, pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            albums_by_server={
                's1': [],
                's2': [{'Id': 'alb2', 'Name': 'A2'}],
            },
            tracks_by_album={'alb2': [{'Id': 'n1'}]},
            reverse_by_server={'s2': {'n1': 'fp_1'}},
            db_track_ids={'fp_1', 'fp_2'},
        )
        assert result['deleted_count'] == 0
        assert 'One' in result['failed_servers']
        assert pruned == [('s2', ['n1'])]
        assert result['orphaned_tracks_count'] == 0


class TestCleaningUnbindOnly:
    def test_full_coverage_unbinds_nothing_and_reports_clean(self, monkeypatch):
        result, statuses, pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            albums_by_server={
                's1': [{'Id': 'alb1', 'Name': 'A1'}],
                's2': [{'Id': 'alb2', 'Name': 'A2'}],
            },
            tracks_by_album={
                'alb1': [{'Id': 'j1'}, {'Id': 'j2'}],
                'alb2': [{'Id': 'n1'}],
            },
            reverse_by_server={
                's1': {'j1': 'fp_1', 'j2': 'fp_2'},
                's2': {'n1': 'fp_3'},
            },
            db_track_ids={'fp_1', 'fp_2', 'fp_3'},
        )
        assert result['status'] == 'SUCCESS'
        assert result['orphaned_tracks_count'] == 0
        assert result['deleted_count'] == 0
        assert result['unbound_mappings'] == 0
        assert pruned == [('s1', ['j1', 'j2']), ('s2', ['n1'])]
        assert statuses[-1][0] == config.TASK_STATUS_SUCCESS

    def test_tracks_on_no_server_are_reported_kept_never_deleted(self, monkeypatch):
        result, statuses, pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            albums_by_server={
                's1': [{'Id': 'alb1', 'Name': 'A1'}],
                's2': [{'Id': 'alb2', 'Name': 'A2'}],
            },
            tracks_by_album={
                'alb1': [{'Id': 'j1'}, {'Id': 'j9'}],
                'alb2': [{'Id': 'n1'}],
            },
            reverse_by_server={
                's1': {'j1': 'fp_1'},
                's2': {'n1': 'fp_2'},
            },
            db_track_ids={'fp_1', 'fp_2', 'fp_3', 'fp_4'},
            prune_results={'s1': 1, 's2': 2},
        )
        assert result['status'] == 'SUCCESS'
        assert result['orphaned_tracks_count'] == 2
        assert result['deleted_count'] == 0
        assert result['unbound_mappings'] == 3
        assert result['unbound_by_server'] == {'One': 1, 'Two': 2}
        reported = {
            t['item_id']
            for album in result['orphaned_albums']
            for t in album['tracks']
        }
        assert reported == {'fp_3', 'fp_4'}
        assert statuses[-1][0] == config.TASK_STATUS_SUCCESS
