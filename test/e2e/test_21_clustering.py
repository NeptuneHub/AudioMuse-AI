# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Song clustering (ALGORITHM.md section 5): every algorithm on the real queue.

Each of the four algorithms (k-means, GMM, spectral, DBSCAN) fans out its
batch jobs on the seeded catalogue of a few hundred real songs, picks a best
solution, writes the playlist rows for the default server and creates the
playlists on Navidrome; each run replaces the previous run's playlists there,
and the playlists listing exposes them with provider ids.

Main Features:
* four runs end SUCCESS with a score and a playlist count in their details
* the playlist table holds the last run per server, names ending _automatic
* Navidrome holds exactly the current run's automatic playlists
* /api/playlists lists them per server with provider ids
"""

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, default_server_id, rows

pytestmark = pytest.mark.e2e

BASE = {
    'enable_clustering_embeddings': True,
    'clustering_runs': 10,
    'pca_components_min': 0,
    'pca_components_max': 8,
    'auto_parameter_discovery': False,
    'top_n_clustering_playlist': 5,
    'max_songs_per_cluster': 0,
    'stratified_sampling_target_percentile': 100,
    'ai_model_provider': 'NONE',
}
SMALL = {'num_clusters_min': 2, 'num_clusters_max': 4, 'gmm_n_components_min': 2, 'gmm_n_components_max': 4,
         'spectral_n_clusters_min': 2, 'spectral_n_clusters_max': 4}
LARGE = {'num_clusters_min': 5, 'num_clusters_max': 12, 'gmm_n_components_min': 5, 'gmm_n_components_max': 12,
         'spectral_n_clusters_min': 5, 'spectral_n_clusters_max': 12,
         'dbscan_eps_min': 0.1, 'dbscan_eps_max': 0.6, 'dbscan_min_samples_min': 3, 'dbscan_min_samples_max': 8}
SUFFIX = '_automatic'
MAX_PLAYLISTS = BASE['top_n_clustering_playlist']


def _payload(method, stack):
    sizes = LARGE if stack.seed_count >= 100 else SMALL
    return dict(BASE, clustering_method=method, **sizes)


@pytest.fixture(scope='module', autouse=True)
def _cleanup_navidrome(stack, navidrome, analyzed_library):
    yield
    navidrome.delete_playlists_named(lambda name: name.endswith(SUFFIX))


def _run(api, payload):
    api.wait_idle(180)
    task_id = api.start_task('/api/clustering/start', payload)
    final = api.wait_for_task(task_id, timeout=1200)
    api.wait_idle(60)
    return final


def _db_playlists(db):
    server_id = default_server_id(db)
    result = {}
    for name, item_id, row_server in rows(db, 'SELECT playlist_name, item_id, server_id FROM playlist'):
        assert row_server == server_id, (name, row_server)
        result.setdefault(name, set()).add(item_id)
    return result


def _remote_automatic(navidrome):
    return {p['name']: p['id'] for p in navidrome.playlists() if (p.get('name') or '').endswith(SUFFIX)}


def _assert_run(final, db, navidrome):
    details = final['details']
    assert isinstance(details.get('best_score'), (int, float)), details
    created = details.get('num_playlists_created')
    assert isinstance(created, int) and 1 <= created <= MAX_PLAYLISTS, details
    playlists = _db_playlists(db)
    assert len(playlists) == created, (created, list(playlists))
    assert all(name.endswith(SUFFIX) for name in playlists), list(playlists)
    catalogue = {r[0] for r in rows(db, 'SELECT item_id FROM score')}
    for items in playlists.values():
        assert items <= catalogue
    remote = _remote_automatic(navidrome)
    assert set(remote) == set(playlists), (set(remote), set(playlists))
    provider_ids = {r[0] for r in rows(db, 'SELECT provider_track_id FROM track_server_map')}
    for name, playlist_id in remote.items():
        entries = navidrome.playlist_entry_ids(playlist_id)
        assert entries and set(entries) <= provider_ids, (name, entries)
    return playlists


@pytest.mark.parametrize('method', ['kmeans', 'gmm', 'spectral', 'dbscan'])
def test_algorithm_run(stack, api, db, navidrome, analyzed_library, method):
    before = _remote_automatic(navidrome)
    final = _run(api, _payload(method, stack))
    assert final['task_type_from_db'] == 'main_clustering'
    playlists = _assert_run(final, db, navidrome)
    remote = _remote_automatic(navidrome)
    stale = set(before) - set(playlists)
    assert not (stale & set(remote)), stale
    assert rows(db, 'SELECT count(*) FROM playlist_name_history')[0][0] >= len(playlists)


def test_playlists_listing(stack, api, db, analyzed_library):
    body = api.json('GET', '/api/playlists')
    assert_no_fp_ids(body)
    assert body.get('multi_server') is False
    servers = body['servers']
    assert len(servers) == 1 and servers[0]['is_default'] is True
    listed = servers[0]['playlists']
    assert set(listed) == set(_db_playlists(db)), (set(listed), set(_db_playlists(db)))
    for tracks in listed.values():
        assert tracks and all(t.get('item_id') and 'title' in t for t in tracks)
