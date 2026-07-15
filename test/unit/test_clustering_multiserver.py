# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Multi-server clustering orchestration: per-server last-run persistence.

Drives run_clustering_task with the registry, per-server pipeline and DB
writers faked, asserting the sequential loop persists each server's playlists
incrementally (scoped replace, bare names), prunes rows of unconfigured
servers at start, and enriches the per-server summary.

Main Features:
* Playlists persist per server as soon as that server succeeds, before the
  next server clusters
* A revoke mid-loop keeps already-persisted servers and stops the loop
* Every 'all'-scope run starts by pruning rows of servers no longer
  configured; the legacy [None] fallback never prunes
* One server raising mid-pipeline or failing its DB persist is marked failed
  and the remaining servers still run
* Stored names stay bare (no [server] suffix); summary carries best_params
  and playlist names per server
"""

import sys
import types

from flask import Flask

import config


def _server(server_id, name, default=False):
    return {
        'server_id': server_id, 'name': name, 'server_type': 'jellyfin',
        'creds': {}, 'music_libraries': '', 'is_default': default,
    }


def _payload(names, score=0.5, params=None, calibrated=None):
    return {
        'playlists': {name: [(f'fp_{name}', f'Title {name}', 'Artist')] for name in names},
        'best_score': score,
        'best_params': params or {'method': 'kmeans'},
        'calibrated_params': calibrated or {
            'num_clusters_min': 8, 'num_clusters_max': 8, 'stratification_percentile': 65,
        },
    }


def _task_kwargs():
    return dict(
        clustering_method='kmeans',
        num_clusters_min=2,
        num_clusters_max=5,
        dbscan_eps_min=0.1,
        dbscan_eps_max=0.5,
        dbscan_min_samples_min=2,
        dbscan_min_samples_max=5,
        pca_components_min=0,
        pca_components_max=8,
        num_clustering_runs=10,
        max_songs_per_cluster_val=20,
        gmm_n_components_min=2,
        gmm_n_components_max=5,
        spectral_n_clusters_min=2,
        spectral_n_clusters_max=5,
        min_songs_per_genre_for_stratification_param=5,
        stratified_sampling_target_percentile_param=50,
        score_weight_diversity_param=1.0,
        score_weight_silhouette_param=0.0,
        score_weight_davies_bouldin_param=0.0,
        score_weight_calinski_harabasz_param=0.0,
        score_weight_purity_param=1.0,
        score_weight_other_feature_diversity_param=0.0,
        score_weight_other_feature_purity_param=0.0,
        ai_model_provider_param='NONE',
        ollama_server_url_param='',
        ollama_model_name_param='',
        openai_server_url_param='',
        openai_model_name_param='',
        openai_api_key_param='',
        gemini_api_key_param='',
        gemini_model_name_param='',
        mistral_api_key_param='',
        mistral_model_name_param='',
        top_n_moods_for_clustering_param=3,
        top_n_playlists_param=8,
        enable_clustering_embeddings_param=False,
        output_server_scope='all',
    )


def _run_clustering(monkeypatch, servers, results_by_server, fail_persist_for=()):
    from unittest.mock import MagicMock
    from tasks import clustering

    events = []
    statuses = []

    fake_flask_app = types.ModuleType('flask_app')
    fake_flask_app.app = Flask('clustering-test')
    monkeypatch.setitem(sys.modules, 'flask_app', fake_flask_app)

    monkeypatch.setattr(clustering, 'get_current_job', lambda conn=None: None)
    monkeypatch.setattr(clustering, 'get_task_info_from_db', lambda task_id: None)
    monkeypatch.setattr(clustering, 'error_manager', MagicMock())
    monkeypatch.setattr(
        clustering, 'save_task_status',
        lambda task_id, task_type, status, progress=None, details=None:
        statuses.append((status, progress, details)),
    )
    monkeypatch.setattr(
        clustering, 'prune_playlist_rows_for_missing_servers',
        lambda server_ids: events.append(('prune', list(server_ids))),
    )

    def fake_persist(playlists, server_id):
        if server_id in fail_persist_for:
            raise RuntimeError('db write failed')
        events.append(('persist', server_id, dict(playlists)))

    monkeypatch.setattr(clustering, 'update_playlist_table', fake_persist)
    monkeypatch.setattr(
        clustering.registry, 'servers_for_scope', lambda scope, conn=None: servers
    )

    def scripted(target_server, state, report, *args):
        server_id = target_server['server_id'] if target_server else None
        events.append(('cluster', server_id))
        result = results_by_server[server_id]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(clustering, '_cluster_one_server', scripted)

    raised = None
    result = None
    try:
        result = clustering.run_clustering_task(**_task_kwargs())
    except Exception as exc:
        raised = exc
    return result, events, statuses, raised


def _persists(events):
    return [e for e in events if e[0] == 'persist']


class TestPerServerPersistence:
    def test_each_server_persists_its_playlists_before_the_next_server_runs(self, monkeypatch):
        result, events, _statuses, raised = _run_clustering(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            results_by_server={
                's1': ('success', _payload(['Rock_automatic'], score=0.4)),
                's2': ('success', _payload(['Jazz_automatic'], score=0.7)),
            },
        )
        assert raised is None
        assert result['status'] == 'SUCCESS'
        ordered = [(e[0], e[1]) for e in events]
        assert ordered == [
            ('prune', ['s1', 's2']),
            ('cluster', 's1'),
            ('persist', 's1'),
            ('cluster', 's2'),
            ('persist', 's2'),
        ]

    def test_stored_playlist_names_keep_their_bare_media_server_names(self, monkeypatch):
        _result, events, _statuses, _raised = _run_clustering(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            results_by_server={
                's1': ('success', _payload(['Rock_automatic'])),
                's2': ('success', _payload(['Rock_automatic'])),
            },
        )
        for _kind, _server_id, playlists in _persists(events):
            assert list(playlists.keys()) == ['Rock_automatic']

    def test_legacy_none_server_persists_under_null_server_id_without_pruning(self, monkeypatch):
        result, events, _statuses, _raised = _run_clustering(
            monkeypatch,
            servers=[None],
            results_by_server={None: ('success', _payload(['Rock_automatic']))},
        )
        assert result['status'] == 'SUCCESS'
        assert [e for e in events if e[0] == 'prune'] == []
        assert [(e[0], e[1]) for e in _persists(events)] == [('persist', None)]


class TestRevokeAndFailure:
    def test_revoke_after_first_server_keeps_first_servers_fresh_rows(self, monkeypatch):
        result, events, _statuses, _raised = _run_clustering(
            monkeypatch,
            servers=[
                _server('s1', 'One', default=True),
                _server('s2', 'Two'),
                _server('s3', 'Three'),
            ],
            results_by_server={
                's1': ('success', _payload(['Rock_automatic'])),
                's2': ('revoked', None),
                's3': ('success', _payload(['Jazz_automatic'])),
            },
        )
        assert result['status'] == 'REVOKED'
        assert [(e[0], e[1]) for e in _persists(events)] == [('persist', 's1')]
        assert ('cluster', 's3') not in [(e[0], e[1]) for e in events]

    def test_all_servers_failing_raises_and_writes_nothing(self, monkeypatch):
        _result, events, _statuses, raised = _run_clustering(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            results_by_server={
                's1': ('skipped', 'only 0 clusterable tracks available'),
                's2': ('skipped', 'only 1 clusterable tracks available'),
            },
        )
        assert isinstance(raised, ValueError)
        assert _persists(events) == []

    def test_failed_server_keeps_its_previous_runs_rows_until_a_success_replaces_them(self, monkeypatch):
        result, events, statuses, _raised = _run_clustering(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            results_by_server={
                's1': ('skipped', 'only 0 clusterable tracks available'),
                's2': ('success', _payload(['Jazz_automatic'])),
            },
        )
        assert result['status'] == 'SUCCESS'
        assert [(e[0], e[1]) for e in _persists(events)] == [('persist', 's2')]
        final_details = statuses[-1][2]
        assert statuses[-1][0] == config.TASK_STATUS_SUCCESS
        assert final_details['per_server'][0]['status'] == 'skipped'
        assert final_details['per_server'][1]['status'] == 'success'

    def test_a_server_raising_mid_pipeline_does_not_stop_the_remaining_servers(self, monkeypatch):
        result, events, statuses, raised = _run_clustering(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            results_by_server={
                's1': RuntimeError('provider unreachable'),
                's2': ('success', _payload(['Jazz_automatic'])),
            },
        )
        assert raised is None
        assert result['status'] == 'SUCCESS'
        assert [(e[0], e[1]) for e in events] == [
            ('prune', ['s1', 's2']),
            ('cluster', 's1'),
            ('cluster', 's2'),
            ('persist', 's2'),
        ]
        per_server = statuses[-1][2]['per_server']
        assert per_server[0]['status'] == 'failed'
        assert 'provider unreachable' in per_server[0]['reason']

    def test_a_failed_db_persist_marks_the_server_failed_not_success(self, monkeypatch):
        result, events, statuses, raised = _run_clustering(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            results_by_server={
                's1': ('success', _payload(['Rock_automatic'])),
                's2': ('success', _payload(['Jazz_automatic'])),
            },
            fail_persist_for={'s1'},
        )
        assert raised is None
        assert result['status'] == 'SUCCESS'
        assert [(e[0], e[1]) for e in _persists(events)] == [('persist', 's2')]
        per_server = statuses[-1][2]['per_server']
        assert per_server[0]['status'] == 'failed'
        assert 'persistence' in per_server[0]['reason']
        assert per_server[1]['status'] == 'success'


class TestRetentionAndSummary:
    def test_new_run_prunes_rows_of_servers_no_longer_configured(self, monkeypatch):
        _result, events, _statuses, _raised = _run_clustering(
            monkeypatch,
            servers=[_server('s1', 'One', default=True)],
            results_by_server={'s1': ('success', _payload(['Rock_automatic']))},
        )
        assert events[0] == ('prune', ['s1'])
        assert ('cluster', 's1') in [(e[0], e[1]) for e in events[1:]]

    def test_per_server_details_carry_best_params_and_playlist_names(self, monkeypatch):
        _result, _events, statuses, _raised = _run_clustering(
            monkeypatch,
            servers=[_server('s1', 'One', default=True), _server('s2', 'Two')],
            results_by_server={
                's1': ('success', _payload(['Rock_automatic', 'Jazz_automatic'], score=0.4,
                                           params={'method': 'kmeans', 'k': 3})),
                's2': ('success', _payload(['Pop_automatic'], score=0.9,
                                           params={'method': 'kmeans', 'k': 7})),
            },
        )
        final_details = statuses[-1][2]
        per_server = final_details['per_server']
        assert per_server[0]['best_params'] == {'method': 'kmeans', 'k': 3}
        assert per_server[0]['calibrated_params'] == {
            'num_clusters_min': 8, 'num_clusters_max': 8, 'stratification_percentile': 65,
        }
        assert per_server[0]['playlist_names'] == ['Jazz_automatic', 'Rock_automatic']
        assert per_server[1]['best_params'] == {'method': 'kmeans', 'k': 7}
        assert per_server[1]['playlist_names'] == ['Pop_automatic']
        assert final_details['best_score'] == 0.9
        assert final_details['best_params'] == {'method': 'kmeans', 'k': 7}
        assert final_details['num_playlists_created'] == 3
