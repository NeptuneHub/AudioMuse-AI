# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Multi-server orphan cleaning: abort rules and orphan identification.

Drives identify_and_clean_orphaned_albums_task with the media-server registry,
provider fetches and DB helpers faked, asserting the multi-server abort rules
(any failed or empty server aborts the run with zero deletions) and that only
canonical ids uncovered by every server's reverse-translated catalogue reach
the deletion helper.

Main Features:
* A failed album fetch on any server aborts the run and never deletes
* Zero albums on one of several servers aborts the run and never deletes
* Full canonical coverage across servers finishes clean with no deletion
* Uncovered canonical ids are exactly what reaches delete_orphaned_albums_sync
* The safety cap limits deletion to the largest orphaned albums
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
                  safety_limit=None):
    from tasks import cleaning

    statuses = []
    deleted = []
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

    fake_analysis = types.ModuleType('tasks.analysis')
    fake_analysis._run_all_index_builds = lambda log_fn=None: None
    monkeypatch.setitem(sys.modules, 'tasks.analysis', fake_analysis)

    monkeypatch.setattr(cleaning, 'get_current_job', lambda *a, **k: None)
    if safety_limit is not None:
        monkeypatch.setattr(cleaning, 'CLEANING_SAFETY_LIMIT', safety_limit)
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

    def fake_delete(ids):
        deleted.append(sorted(ids))
        return {
            'status': 'SUCCESS', 'message': 'ok',
            'deleted_count': len(ids), 'failed_deletions': [],
        }

    monkeypatch.setattr(cleaning, 'delete_orphaned_albums_sync', fake_delete)

    result = cleaning.identify_and_clean_orphaned_albums_task()
    return result, statuses, deleted


class TestCleaningAbortRules:
    def test_failed_album_fetch_on_one_server_aborts_without_deleting(self, monkeypatch):
        result, statuses, deleted = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            albums_by_server={
                's1': RuntimeError('fetch failed'),
                's2': [{'Id': 'alb2', 'Name': 'A2'}],
            },
            tracks_by_album={'alb2': [{'Id': 'n1'}]},
            reverse_by_server={'s2': {'n1': 'fp_1'}},
            db_track_ids={'fp_1', 'fp_2'},
        )
        assert result['status'] == 'ABORTED'
        assert result['deleted_count'] == 0
        assert 'One' in result['failed_servers']
        assert deleted == []
        assert statuses[-1][0] == config.TASK_STATUS_FAILURE

    def test_zero_albums_on_one_of_two_servers_aborts_without_deleting(self, monkeypatch):
        result, statuses, deleted = _run_cleaning(
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
        assert result['status'] == 'ABORTED'
        assert result['deleted_count'] == 0
        assert 'One' in result['failed_servers']
        assert deleted == []
        assert statuses[-1][0] == config.TASK_STATUS_FAILURE


class TestCleaningOrphanIdentification:
    def test_full_coverage_across_servers_deletes_nothing(self, monkeypatch):
        result, statuses, deleted = _run_cleaning(
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
        assert deleted == []
        assert statuses[-1][0] == config.TASK_STATUS_SUCCESS

    def test_uncovered_canonical_ids_are_exactly_what_gets_deleted(self, monkeypatch):
        result, statuses, deleted = _run_cleaning(
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
        )
        assert result['status'] == 'SUCCESS'
        assert result['orphaned_tracks_count'] == 2
        assert result['deleted_count'] == 2
        assert deleted == [['fp_3', 'fp_4']]
        assert statuses[-1][0] == config.TASK_STATUS_SUCCESS

    def test_safety_cap_limits_deletion_to_largest_orphaned_albums(self, monkeypatch):
        result, statuses, deleted = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            albums_by_server={
                's1': [{'Id': 'alb1', 'Name': 'A1'}],
                's2': [{'Id': 'alb2', 'Name': 'A2'}],
            },
            tracks_by_album={
                'alb1': [{'Id': 'j1'}],
                'alb2': [{'Id': 'n1'}],
            },
            reverse_by_server={
                's1': {'j1': 'fp_1'},
                's2': {'n1': 'fp_2'},
            },
            db_track_ids={'fp_1', 'fp_2', 'fp_3', 'fp_4', 'fp_5'},
            author_by_id={'fp_3': 'ArtistA', 'fp_4': 'ArtistA', 'fp_5': 'ArtistB'},
            safety_limit=1,
        )
        assert result['status'] == 'SUCCESS'
        assert deleted == [['fp_3', 'fp_4']]
        assert result['deleted_count'] == 2
        assert result['orphaned_albums_count'] == 1
        assert statuses[-1][0] == config.TASK_STATUS_SUCCESS
