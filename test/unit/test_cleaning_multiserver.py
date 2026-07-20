# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Multi-server library cleanup: per-server unbind plus orphan delete.

Drives identify_and_clean_orphaned_albums_task with the media-server registry,
provider fetches and DB helpers faked, asserting that it prunes each healthy
server's own track_server_map rows, skips servers whose fetch fails, returns
nothing or looks partial, and DELETES the tracks bound to no server only when
every server was read completely.

Main Features:
* Uses the same full-catalogue fetch the alignment sweeps use (fetch_all_tracks)
* A failed or empty fetch skips ONLY that server's unbinding, others proceed
* Per-server pruning receives exactly that server's present provider ids
* Full coverage across servers unbinds nothing and reports zero orphans
* The legacy [None] registry fallback still counts its tracks as present
* Tracks on no server are deleted on a complete view, kept on an incomplete one
"""

import sys
import types

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
                  prune_results=None, stored_counts=None, mark_refused=None,
                  clean_catalogue=True, rebuild_calls=None,
                  chromaprint_split_result=None):
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

    # Cleaning finishes by running the shared final rebuild inline (the same one
    # analysis runs). Stub it with a fake tasks.analysis.index module so the unit
    # test records the call without touching the real builders or a live index.
    rebuilds = rebuild_calls if rebuild_calls is not None else []

    def _fake_run_all_index_builds(log_fn=None, progress_start=95, progress_end=98):
        rebuilds.append((progress_start, progress_end))
        if log_fn:
            log_fn("Similarity indexes rebuilt.", progress_end)

    fake_index = types.ModuleType('tasks.analysis.index')
    fake_index._run_all_index_builds = _fake_run_all_index_builds
    monkeypatch.setitem(sys.modules, 'tasks.analysis.index', fake_index)

    # Chromaprint dedup (Path B) is invoked inline before the rebuild; stub it so the
    # unit test controls the split count without a real fpcalc/DB round trip.
    cp_result = chromaprint_split_result or {'split': 0, 'removed': 0}
    fake_dup_repair = types.ModuleType('tasks.duplicate_repair')
    fake_dup_repair.split_chromaprint_false_merges = lambda conn=None: cp_result
    monkeypatch.setitem(sys.modules, 'tasks.duplicate_repair', fake_dup_repair)

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
        # The cleaning loop must have bound this server's context before fetching.
        sid = ms_context.active_server_id()
        result = tracks_by_server[sid]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(multiserver_sync.provider_probe, 'fetch_all_tracks', fake_fetch)

    refused_ids = set(mark_refused or ())

    def fake_prune(db, server_id, present_ids, refused=None):
        pruned_calls.append((server_id, sorted(present_ids)))
        if refused is not None and server_id in refused_ids:
            refused.append(server_id)
        return (prune_results or {}).get(server_id, 0)

    monkeypatch.setattr(multiserver_sync, 'prune_stale_mappings', fake_prune)

    counts = stored_counts if stored_counts is not None else []
    monkeypatch.setattr(
        multiserver_sync, '_store_server_track_count',
        lambda db, server_id, count: counts.append((server_id, count)),
    )

    result = cleaning.identify_and_clean_orphaned_albums_task(clean_catalogue)
    return result, statuses, pruned_calls


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


class TestCleaningSkipsUnreadableServers:
    def test_failed_fetch_skips_that_server_but_prunes_the_healthy_one(self, monkeypatch):
        result, statuses, pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            tracks_by_server={
                's1': RuntimeError('fetch failed'),
                's2': [{'id': 'n1'}],
            },
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

    def test_zero_tracks_skips_that_server_and_reports_no_orphans(self, monkeypatch):
        result, statuses, pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            tracks_by_server={
                's1': [],
                's2': [{'id': 'n1'}],
            },
            reverse_by_server={'s2': {'n1': 'fp_1'}},
            db_track_ids={'fp_1', 'fp_2'},
        )
        assert result['deleted_count'] == 0
        assert 'One' in result['failed_servers']
        assert pruned == [('s2', ['n1'])]
        assert result['orphaned_tracks_count'] == 0


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
        assert statuses[-1][0] == config.TASK_STATUS_SUCCESS

    def test_tracks_on_no_server_are_deleted_when_view_is_complete(self, monkeypatch):
        # Both servers were read completely, so the tracks bound to no server
        # (fp_3, fp_4) are gone from every library and get deleted from the catalogue.
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
        assert statuses[-1][0] == config.TASK_STATUS_SUCCESS

    def test_index_rebuild_runs_inline_before_a_cleaning_run_completes(self, monkeypatch):
        # Cleaning changes what each server maps and can remove catalogue rows, so it
        # runs the SAME final rebuild analysis runs INLINE - the task is not reported
        # complete until every server's similarity results reflect the cleaned
        # catalogue and the reload has been published.
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
        # Cleaning runs the Chromaprint dedup (Path B) and surfaces how many false
        # merges it split, so the user sees the benefit in the run summary.
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
        # Same complete view as above, but the per-run flag is off (the default): the
        # orphans are reported, never deleted, and the catalogue is left untouched.
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

    def test_refused_partial_listing_reports_orphans_but_deletes_nothing(self, monkeypatch):
        # A server that returned a partial listing (prune refused) means the view is
        # unreliable, so orphans are reported but NOT deleted - an unbound-but-present
        # track must never be dropped on incomplete data.
        result, _statuses, _pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            tracks_by_server={'s1': [{'id': 'j1'}], 's2': [{'id': 'n1'}]},
            reverse_by_server={'s1': {'j1': 'fp_1'}, 's2': {'n1': 'fp_2'}},
            db_track_ids={'fp_1', 'fp_2', 'fp_3'},
            mark_refused={'s2'},
        )
        assert 'Two' in result['prune_refused_servers']
        assert result['orphaned_tracks_count'] == 1
        assert result['deleted_count'] == 0

    def test_implausibly_many_orphans_are_not_deleted(self, monkeypatch):
        # More than half the catalogue looking orphaned on a complete view is a bogus
        # listing, so the guard reports them and deletes nothing.
        result, _statuses, _pruned = _run_cleaning(
            monkeypatch,
            servers=[_server('s1', 'One', default=True)],
            tracks_by_server={'s1': [{'id': 'j1'}]},
            reverse_by_server={'s1': {'j1': 'fp_1'}},
            db_track_ids={'fp_1', 'fp_2', 'fp_3', 'fp_4', 'fp_5'},
        )
        assert result['status'] == 'SUCCESS'
        assert result['orphaned_tracks_count'] == 4
        assert result['deleted_count'] == 0


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
        assert statuses[-1][0] == config.TASK_STATUS_SUCCESS
