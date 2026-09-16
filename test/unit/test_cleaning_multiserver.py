# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Multi-server library cleanup: per-server unbind plus orphan delete.

Drives identify_and_clean_orphaned_albums_task with the media-server registry,
provider fetches and DB helpers faked, asserting that it prunes each server's
own track_server_map rows from exactly what that server returned, and DELETES
the tracks found on no server, at most CLEANING_SAFETY_LIMIT albums per run.

Main Features:
* Uses the same full-catalogue fetch the alignment sweeps use (fetch_all_tracks)
* A fetch that raises skips ONLY that server and keeps only the songs it holds
* An empty fetch from a server that still holds songs (or a default/legacy
  server with a non-empty catalogue) is unread too; an empty server that holds
  nothing is read normally
* Per-server pruning receives exactly that server's present provider ids
* Full coverage across servers unbinds nothing and reports zero orphans
* The legacy [None] registry fallback still counts its tracks as present
* No share guard: a large orphan share is deleted, capped only by album count
"""

import sys
import types

import taskqueue

from unittest.mock import MagicMock

from flask import Flask

import config


def _server(server_id, name, default=False):
    return {
        'server_id': server_id, 'name': name, 'server_type': 'jellyfin',
        'creds': {}, 'music_libraries': '', 'is_default': default,
    }


def _run_cleaning(monkeypatch, servers, tracks_by_server,
                  reverse_by_server, db_track_ids, author_by_id=None,
                  prune_results=None, stored_counts=None,
                  clean_catalogue=True, rebuild_calls=None,
                  chromaprint_split_result=None, current_job=None,
                  task_info=None, album_by_id=None, mapped_by_server=None,
                  unmapped_ids=None, deleted_calls=None, title_by_id=None):
    from tasks import cleaning
    from tasks import multiserver_sync

    statuses = []
    pruned_calls = []
    authors = author_by_id or {}
    albums = album_by_id or {}
    titles = title_by_id or {}
    mapped = mapped_by_server or {}
    deletes = deleted_calls if deleted_calls is not None else []

    fake_flask_app = types.ModuleType('flask_app')
    fake_flask_app.app = Flask('cleaning-test')
    monkeypatch.setitem(sys.modules, 'flask_app', fake_flask_app)

    cur = MagicMock()
    state = {'last': (None, None)}

    def record_execute(sql, params=None):
        state['last'] = (sql, params)
        if sql and sql.startswith('DELETE FROM score'):
            deletes.extend(params[0])

    def answer_fetchall():
        sql, params = state['last']
        if sql and 'JOIN embedding' in sql:
            return [(item_id,) for item_id in sorted(db_track_ids)]
        if sql and sql.startswith('SELECT item_id, title, author, album, album_artist FROM score'):
            return [
                (
                    item_id, titles.get(item_id, f'Title {item_id}'),
                    authors.get(item_id, f'Artist {item_id}'),
                    albums.get(item_id, f'Album {item_id}'), None,
                )
                for item_id in params[0]
            ]
        if sql and sql.startswith('SELECT item_id FROM track_server_map WHERE server_id = ANY'):
            return [
                (item_id,) for sid in params[0] for item_id in sorted(mapped.get(sid, ()))
            ]
        if sql and 'NOT EXISTS' in sql and 'track_server_map' in sql:
            return [(item_id,) for item_id in sorted(unmapped_ids or ())]
        return []

    def answer_fetchone():
        sql, params = state['last']
        if sql and 'EXISTS (SELECT 1 FROM track_server_map WHERE server_id' in sql:
            return (bool(mapped.get(params[0])),)
        if sql and 'EXISTS (SELECT 1 FROM score)' in sql:
            return (bool(db_track_ids),)
        return None

    cur.execute.side_effect = record_execute
    cur.fetchall.side_effect = answer_fetchall
    cur.fetchone.side_effect = answer_fetchone
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    get_db_cm = MagicMock()
    get_db_cm.__enter__.return_value = conn
    get_db_cm.__exit__.return_value = False

    rebuilds = rebuild_calls if rebuild_calls is not None else []

    def _fake_run_all_index_builds(log_fn=None, progress_start=95, progress_end=98,
                                   task_id=None):
        rebuilds.append((progress_start, progress_end))
        if log_fn:
            log_fn("Similarity indexes rebuilt.", progress_end)

    fake_index = types.ModuleType('tasks.analysis.index')
    fake_index._run_all_index_builds = _fake_run_all_index_builds
    monkeypatch.setitem(sys.modules, 'tasks.analysis.index', fake_index)

    cp_result = chromaprint_split_result or {'split': 0, 'removed': 0}
    fake_dup_repair = types.ModuleType('tasks.duplicate_repair')
    fake_dup_repair.split_chromaprint_false_merges = lambda conn=None: cp_result
    monkeypatch.setitem(sys.modules, 'tasks.duplicate_repair', fake_dup_repair)

    import database as db_module
    from tasks import task_run as task_run_module
    monkeypatch.setattr(db_module, 'get_db', lambda: get_db_cm)
    monkeypatch.setattr(db_module, 'get_task_info_from_db', lambda _task_id: task_info)

    def _record_status(task_id, task_type, status, progress=None, details=None, **_kwargs):
        statuses.append((status, progress, details))
        return True

    monkeypatch.setattr(db_module, 'save_task_status', _record_status)
    monkeypatch.setattr(task_run_module, 'save_task_status', _record_status)

    monkeypatch.setattr(
        taskqueue, 'current_task_id',
        lambda: current_job.id if current_job is not None else None,
    )
    monkeypatch.setattr(
        cleaning.registry, 'servers_for_scope', lambda scope, conn=None: servers
    )
    by_id = {s['server_id']: s for s in servers if s}
    monkeypatch.setattr(
        cleaning.registry, 'context_for', lambda sid, conn=None: by_id[sid]
    )

    def fake_reverse(chunk, server_id, conn=None):
        mapping = reverse_by_server.get(server_id, {})
        return {pid: mapping[pid] for pid in chunk if pid in mapping}

    monkeypatch.setattr(cleaning.registry, 'reverse_translate_ids', fake_reverse)

    from tasks.mediaserver import context as ms_context

    def fake_fetch(stype, creds, apply_filter=False):
        sid = ms_context.active_server_id()
        result = tracks_by_server[sid]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(multiserver_sync.provider_probe, 'fetch_all_tracks', fake_fetch)

    def fake_prune(db, server_id, present_ids):
        pruned_calls.append((server_id, sorted(present_ids)))
        return (prune_results or {}).get(server_id, 0)

    monkeypatch.setattr(multiserver_sync, 'prune_stale_mappings', fake_prune)

    counts = stored_counts if stored_counts is not None else []
    monkeypatch.setattr(
        multiserver_sync, '_store_server_track_count',
        lambda db, server_id, count: counts.append((server_id, count)),
    )

    try:
        result = cleaning.identify_and_clean_orphaned_albums_task(clean_catalogue)
    except cleaning.CleaningIncomplete as exc:
        summary = (statuses[-1][2] or {}).get('final_summary_details') or {}
        result = {'status': 'FAIL', 'message': str(exc), **summary}
    return result, statuses, pruned_calls


class TestAPartialRunIsNotRetriedByTheQueue:
    def test_an_unread_server_is_a_permanent_failure_the_next_cron_run_retries(self):
        from taskqueue import TaskFailed
        from tasks import cleaning

        assert issubclass(cleaning.CleaningIncomplete, TaskFailed), (
            'a plain raise would cost QUEUE_MAX_ATTEMPTS full runs, each one a '
            'whole-catalogue fetch per server plus the full index rebuild with the '
            'one-live-main slot held, to reach the identical outcome; the run '
            'completed, the summary is on the row, and tomorrow is the retry'
        )


class TestCleaningRefreshesTrackCounts:
    def test_each_fetched_server_gets_its_track_count_stored(self, monkeypatch):
        stored = []
        result, _statuses, _pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            tracks_by_server={
                's1': [{'id': 'a1'}, {'id': 'a2'}],
                's2': [{'id': 'n1'}],
            },
            reverse_by_server={'s1': {'a1': 'fp_1', 'a2': 'fp_2'}, 's2': {'n1': 'fp_1'}},
            db_track_ids={'fp_1', 'fp_2'},
            stored_counts=stored,
        )
        assert result['status'] == 'SUCCESS'
        assert stored == [('s1', 2), ('s2', 1)]

    def test_failed_fetch_stores_no_count_for_that_server(self, monkeypatch):
        stored = []
        _result, _statuses, _pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            tracks_by_server={
                's1': RuntimeError('fetch failed'),
                's2': [{'id': 'n1'}],
            },
            reverse_by_server={'s2': {'n1': 'fp_1'}},
            db_track_ids={'fp_1'},
            stored_counts=stored,
        )
        assert stored == [('s2', 1)]


def test_dequeued_cleaning_with_wiped_claim_stops_before_writing(monkeypatch):
    import pytest
    from taskqueue import TaskCancelled

    job = MagicMock(id='cleaning-cancelled')
    monkeypatch.setattr('tasks.task_run._read_task_statuses', lambda _conn, ids: {})
    counts = []
    servers = [_server('s1', 'One', default=True)]

    with pytest.raises(TaskCancelled):
        _run_cleaning(
            monkeypatch,
            servers=servers,
            tracks_by_server={'s1': [{'id': 'a1'}]},
            reverse_by_server={'s1': {'a1': 'fp_1'}},
            db_track_ids={'fp_1'},
            current_job=job,
            task_info=None,
            stored_counts=counts,
        )

    assert counts == [], (
        'the shared cancel check is forced once before the first report, so a '
        'row the cancel wiped fetches nothing, prunes nothing and writes nothing'
    )


class TestCleaningSkipsUnreadableServers:
    def test_failed_fetch_keeps_only_that_servers_songs_and_cleans_the_rest(self, monkeypatch):
        deleted = []
        result, statuses, pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One'), _server('s2', 'Two', default=True)],
            tracks_by_server={
                's1': RuntimeError('fetch failed'),
                's2': [{'id': 'n1'}],
            },
            reverse_by_server={'s2': {'n1': 'fp_1'}},
            db_track_ids={'fp_1', 'fp_2', 'fp_3'},
            mapped_by_server={'s1': {'fp_2'}},
            prune_results={'s2': 3},
            deleted_calls=deleted,
        )
        assert result['status'] == 'FAIL'
        assert 'One' in result['failed_servers']
        assert pruned == [('s2', ['n1'])]
        assert result['unbound_mappings'] == 3
        assert deleted == ['fp_3'], (
            'fp_2 is still mapped to the unread server so it was never checked; '
            'fp_3 is on no server and is cleaned in the same run'
        )
        assert statuses[-1][0] == config.TASK_STATUS_RUNNING, (
            'the task narrates the summary on its last progress write and then '
            "raises; FAIL itself is the queue's row to write, and its retry"
        )
        assert 'One' in statuses[-1][2]['final_summary_details']['failed_servers']

    def test_an_unread_default_server_also_keeps_unmapped_legacy_rows(self, monkeypatch):
        deleted = []
        result, _statuses, _pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            tracks_by_server={
                's1': RuntimeError('fetch failed'),
                's2': [{'id': 'n1'}],
            },
            reverse_by_server={'s2': {'n1': 'fp_1'}},
            db_track_ids={'fp_1', 'legacy1', 'fp_3'},
            unmapped_ids={'legacy1'},
            deleted_calls=deleted,
        )
        assert deleted == ['fp_3']
        assert result['orphaned_tracks_count'] == 1

    def test_an_unread_legacy_fallback_server_checks_nothing(self, monkeypatch):
        deleted = []
        result, _statuses, pruned = _run_cleaning(
            monkeypatch,
            servers=[None],
            tracks_by_server={None: RuntimeError('fetch failed')},
            reverse_by_server={},
            db_track_ids={'a1', 'a2'},
            deleted_calls=deleted,
        )
        assert deleted == []
        assert result['orphaned_tracks_count'] == 0
        assert pruned == []

    def test_an_empty_list_from_a_server_holding_songs_is_unread_not_emptied(self, monkeypatch):
        deleted = []
        result, statuses, pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One'), _server('s2', 'Two', default=True)],
            tracks_by_server={
                's1': [],
                's2': [{'id': 'n1'}],
            },
            reverse_by_server={'s2': {'n1': 'fp_1'}},
            db_track_ids={'fp_1', 'fp_2', 'fp_3'},
            mapped_by_server={'s1': {'fp_2'}},
            deleted_calls=deleted,
        )
        assert result['status'] == 'FAIL'
        assert result['failed_servers'] == ['One']
        assert pruned == [('s2', ['n1'])], 'an empty answer must never unbind that server'
        assert deleted == ['fp_3'], 'fp_2 is still mapped to the silent server and is kept'

    def test_an_empty_default_server_keeps_the_whole_catalogue(self, monkeypatch):
        deleted = []
        result, _statuses, pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True)],
            tracks_by_server={'s1': []},
            reverse_by_server={},
            db_track_ids={'fp_1', 'legacy1'},
            mapped_by_server={'s1': {'fp_1'}},
            unmapped_ids={'legacy1'},
            deleted_calls=deleted,
        )
        assert result['status'] == 'FAIL'
        assert pruned == []
        assert deleted == []
        assert result['orphaned_tracks_count'] == 0

    def test_an_empty_legacy_fallback_server_deletes_nothing(self, monkeypatch):
        deleted = []
        result, _statuses, pruned = _run_cleaning(
            monkeypatch,
            servers=[None],
            tracks_by_server={None: []},
            reverse_by_server={},
            db_track_ids={'a1', 'a2'},
            deleted_calls=deleted,
        )
        assert result['status'] == 'FAIL'
        assert deleted == []
        assert pruned == []

    def test_an_empty_server_that_holds_nothing_is_simply_read(self, monkeypatch):
        deleted = []
        result, _statuses, pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            tracks_by_server={
                's1': [{'id': 'j1'}],
                's2': [],
            },
            reverse_by_server={'s1': {'j1': 'fp_1'}},
            db_track_ids={'fp_1', 'fp_2'},
            deleted_calls=deleted,
        )
        assert result['status'] == 'SUCCESS'
        assert result['failed_servers'] == []
        assert pruned == [('s1', ['j1']), ('s2', [])]
        assert deleted == ['fp_2']


class TestCleaningOrphanHandling:
    def test_full_coverage_unbinds_nothing_and_reports_clean(self, monkeypatch):
        result, statuses, pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            tracks_by_server={
                's1': [{'id': 'j1'}, {'id': 'j2'}],
                's2': [{'id': 'n1'}],
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
        assert statuses[-1][0] == config.TASK_STATUS_RUNNING

    def test_tracks_on_no_server_are_deleted_when_view_is_complete(self, monkeypatch):
        result, statuses, pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            tracks_by_server={
                's1': [{'id': 'j1'}, {'id': 'j9'}],
                's2': [{'id': 'n1'}],
            },
            reverse_by_server={
                's1': {'j1': 'fp_1', 'j9': 'fp_5'},
                's2': {'n1': 'fp_2'},
            },
            db_track_ids={'fp_1', 'fp_2', 'fp_3', 'fp_4', 'fp_5'},
            prune_results={'s1': 1, 's2': 2},
        )
        assert result['status'] == 'SUCCESS'
        assert result['orphaned_tracks_count'] == 2
        assert result['deleted_count'] == 2
        assert result['unbound_mappings'] == 3
        assert result['unbound_by_server'] == {'One': 1, 'Two': 2}
        reported = {
            t['item_id']
            for album in result['orphaned_albums']
            for t in album['tracks']
        }
        assert reported == {'fp_3', 'fp_4'}
        assert statuses[-1][0] == config.TASK_STATUS_RUNNING

    def test_index_rebuild_runs_inline_before_a_cleaning_run_completes(self, monkeypatch):
        rebuilds = []
        result, _statuses, _pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True)],
            tracks_by_server={'s1': [{'id': 'j1'}]},
            reverse_by_server={'s1': {'j1': 'fp_1'}},
            db_track_ids={'fp_1'},
            rebuild_calls=rebuilds,
        )
        assert result['status'] == 'SUCCESS'
        assert len(rebuilds) == 1

    def test_chromaprint_false_merge_splits_are_reported(self, monkeypatch):
        result, _statuses, _pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True)],
            tracks_by_server={'s1': [{'id': 'j1'}]},
            reverse_by_server={'s1': {'j1': 'fp_1'}},
            db_track_ids={'fp_1'},
            chromaprint_split_result={'split': 3, 'removed': 6},
        )
        assert result['status'] == 'SUCCESS'
        assert result['chromaprint_splits'] == 3

    def test_orphans_are_kept_when_catalogue_cleaning_is_disabled(self, monkeypatch):
        result, _statuses, _pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            tracks_by_server={
                's1': [{'id': 'j1'}, {'id': 'j9'}],
                's2': [{'id': 'n1'}],
            },
            reverse_by_server={
                's1': {'j1': 'fp_1', 'j9': 'fp_5'},
                's2': {'n1': 'fp_2'},
            },
            db_track_ids={'fp_1', 'fp_2', 'fp_3', 'fp_4', 'fp_5'},
            prune_results={'s1': 1, 's2': 2},
            clean_catalogue=False,
        )
        assert result['status'] == 'SUCCESS'
        assert result['orphaned_tracks_count'] == 2
        assert result['deleted_count'] == 0
        assert result['catalogue_deletion'] is False

    def test_a_large_orphan_share_is_deleted_with_no_share_guard(self, monkeypatch):
        result, _statuses, _pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True)],
            tracks_by_server={'s1': [{'id': 'j1'}]},
            reverse_by_server={'s1': {'j1': 'fp_1'}},
            db_track_ids={'fp_1', 'fp_2', 'fp_3', 'fp_4', 'fp_5'},
        )
        assert result['status'] == 'SUCCESS'
        assert result['orphaned_tracks_count'] == 4
        assert result['deleted_count'] == 4
        assert result['remaining_orphans_count'] == 0

    def test_the_safety_limit_caps_the_albums_deleted_per_run(self, monkeypatch):
        from tasks import cleaning

        monkeypatch.setattr(cleaning, 'CLEANING_SAFETY_LIMIT', 1)
        deleted = []
        result, _statuses, _pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True)],
            tracks_by_server={'s1': [{'id': 'j1'}]},
            reverse_by_server={'s1': {'j1': 'fp_1'}},
            db_track_ids={'fp_1', 'fp_2', 'fp_3', 'fp_4'},
            author_by_id={'fp_2': 'Band', 'fp_3': 'Band', 'fp_4': 'Solo'},
            album_by_id={'fp_2': 'Big', 'fp_3': 'Big', 'fp_4': 'Small'},
            deleted_calls=deleted,
        )
        assert sorted(deleted) == ['fp_2', 'fp_3'], 'the largest album goes first'
        assert result['deleted_albums_count'] == 1
        assert result['orphaned_albums_count'] == 2
        assert result['remaining_orphans_count'] == 1
        assert [a['album'] for a in result['orphaned_albums']] == ['Big']

    def test_untagged_orphans_group_per_artist_under_unknown_album(self, monkeypatch):
        from tasks import cleaning

        monkeypatch.setattr(cleaning, 'CLEANING_SAFETY_LIMIT', 1)
        deleted = []
        result, _statuses, _pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True)],
            tracks_by_server={'s1': [{'id': 'j1'}]},
            reverse_by_server={'s1': {'j1': 'fp_1'}},
            db_track_ids={'fp_1', 'fp_2', 'fp_3', 'fp_4'},
            author_by_id={'fp_2': 'Band', 'fp_3': 'Band', 'fp_4': 'Solo'},
            album_by_id={'fp_2': None, 'fp_3': None, 'fp_4': ''},
            title_by_id={'fp_2': None, 'fp_3': None, 'fp_4': None},
            deleted_calls=deleted,
        )
        assert sorted(deleted) == ['fp_2', 'fp_3'], 'the untagged pile of one artist is one album'
        assert result['deleted_albums_count'] == 1
        assert result['orphaned_albums_count'] == 2
        assert result['remaining_orphans_count'] == 1
        assert result['orphaned_albums'][0]['album'] == 'Unknown Album'
        assert result['orphaned_albums'][0]['artist'] == 'Band'
        assert not any('fp_' in a['album'] or 'fp_' in a['artist'] for a in result['orphaned_albums'])


class TestCleaningLegacyFallback:
    def test_none_server_fallback_counts_tracks_present_and_never_prunes(self, monkeypatch):
        result, statuses, pruned = _run_cleaning(
            monkeypatch,
            servers=[None],
            tracks_by_server={None: [{'id': 'a1'}, {'id': 'a2'}]},
            reverse_by_server={None: {'a1': 'a1', 'a2': 'a2'}},
            db_track_ids={'a1', 'a2'},
        )
        assert result['status'] == 'SUCCESS'
        assert result['orphaned_tracks_count'] == 0
        assert result['unbound_mappings'] == 0
        assert pruned == []
        assert statuses[-1][0] == config.TASK_STATUS_RUNNING
