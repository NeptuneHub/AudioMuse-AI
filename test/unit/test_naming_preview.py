# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the setup wizard full-title naming preview.

The preview clusters a sample of the library once and names every playlist with the
unsaved prompt. These tests pin that it stays a read-only side trip: it never runs
more than one at a time, never samples more than 10,000 songs, only ever issues
SELECT statements, always clusters with K-Means at a fixed K, and names playlists
through the same title function and duplicate-suffix rule as a real run.

Main Features:
* A second preview is refused while one runs, and a crash ends in a generic message.
* The sample is capped at PREVIEW_MAX_SONGS and read with SELECT only.
* Clustering is one K-Means pass at a fixed K, followed by the real minimum-size filter
  and, when configured, the diverse top-N selection.
* Naming falls back to the tag name and de-duplicates titles like a real run.
"""

import pytest

import config
from tasks import naming_preview


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(naming_preview, '_STATE', {
        'id': None, 'status': 'idle', 'message': '', 'song_count': 0, 'done': 0,
        'total': 0, 'titles': [], 'started_at': None, 'finished_at': None,
    })


class _NoStartThread:
    started = 0

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        _NoStartThread.started += 1


class TestStartAndStatus:
    def test_a_second_preview_is_refused_while_one_runs(self, monkeypatch):
        monkeypatch.setattr(naming_preview.threading, 'Thread', _NoStartThread)
        _NoStartThread.started = 0
        assert naming_preview.start_preview('Name it.', {'provider': 'OLLAMA'}) is True
        assert naming_preview.preview_status()['status'] == 'running'
        assert naming_preview.start_preview('Name it.', {'provider': 'OLLAMA'}) is False
        assert _NoStartThread.started == 1

    def test_status_is_a_copy(self):
        snapshot = naming_preview.preview_status()
        snapshot['titles'].append('mutated')
        assert naming_preview.preview_status()['titles'] == []

    def test_a_crash_ends_with_a_generic_message(self, monkeypatch):
        naming_preview._STATE.update(id='p1', status='running')

        def boom(_preview_id):
            raise RuntimeError('secret internal detail')

        monkeypatch.setattr(naming_preview, '_cluster_sample', boom)
        naming_preview._run('p1', 'Name it.', {'provider': 'OLLAMA'})
        state = naming_preview.preview_status()
        assert state['status'] == 'failed'
        assert 'secret' not in state['message']
        assert 'container logs' in state['message']

    def test_a_stale_preview_cannot_overwrite_a_newer_one(self):
        naming_preview._STATE.update(id='new', status='running', message='new run')
        naming_preview._finish('old', 'done', 'old run')
        assert naming_preview.preview_status()['message'] == 'new run'


class TestClusterCount:
    @pytest.fixture(autouse=True)
    def limits(self, monkeypatch):
        monkeypatch.setattr(config, 'CLUSTERING_MAX_PLAYLIST_SONGS', 200)
        monkeypatch.setattr(config, 'MIN_PLAYLIST_SIZE_FOR_TOP_N', 20)
        monkeypatch.setattr(config, 'NUM_CLUSTERS_MAX', 100)

    def test_a_large_sample_uses_the_configured_maximum(self, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 0)
        assert naming_preview.preview_cluster_count(10000) == 100

    def test_a_small_sample_is_bounded_by_playlist_size(self, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 0)
        assert naming_preview.preview_cluster_count(400) == 10

    def test_k_never_reaches_the_song_count(self, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 0)
        assert naming_preview.preview_cluster_count(3) == 2


class _FakeCursor:
    def __init__(self, log):
        self.log = log

    def execute(self, sql, params=None):
        self.log.append(sql)

    def fetchall(self):
        return [{'item_id': 'x', 'mood_vector': 'rock:1'}]

    def close(self):
        pass


class _FakeConnection:
    def __init__(self, log):
        self.log = log

    def cursor(self, cursor_factory=None):
        return _FakeCursor(self.log)


class TestSampling:
    def test_the_sample_is_capped_and_read_only(self, monkeypatch):
        import database
        from tasks import clustering, clustering_helper

        statements = []
        monkeypatch.setattr(database, 'get_db', lambda: _FakeConnection(statements))
        monkeypatch.setattr(clustering, '_prepare_genre_map', lambda rows: {'rock': rows})
        monkeypatch.setattr(clustering, '_calculate_target_songs_per_genre', lambda *a: 50)
        monkeypatch.setattr(
            clustering_helper,
            '_get_stratified_song_subset',
            lambda genre_map, target: [{'item_id': str(i)} for i in range(12000)],
        )
        item_ids, genre_map = naming_preview._sample_item_ids()
        assert len(item_ids) == naming_preview.PREVIEW_MAX_SONGS == 10000
        assert len(set(item_ids)) == 10000
        assert genre_map == {'rock': [{'item_id': 'x', 'mood_vector': 'rock:1'}]}
        assert statements and all(sql.lstrip().upper().startswith('SELECT') for sql in statements)


class TestClusterSample:
    @pytest.fixture
    def patched(self, monkeypatch):
        from tasks import clustering_helper, clustering_postprocessing

        calls = {}
        monkeypatch.setattr(config, 'MIN_PLAYLIST_SIZE_FOR_TOP_N', 2)
        monkeypatch.setattr(
            naming_preview,
            '_sample_item_ids',
            lambda: (['a', 'b', 'c', 'd'], {'rock': [1, 2], '__other__': [3]}),
        )
        monkeypatch.setattr(naming_preview, 'preview_cluster_count', lambda count: 3)

        def fake_iteration(**kwargs):
            calls['iteration'] = kwargs
            return calls.get('result', {'named_playlists': {'Rock_automatic': [('a', 'T', 'A')]}})

        def fake_min_size(result, min_size, log_prefix=''):
            calls['min_size'] = min_size
            return result

        def fake_top_n(result, limit, primary_genre_counts=None):
            calls['top_n'] = (limit, primary_genre_counts)
            return result

        monkeypatch.setattr(clustering_helper, '_perform_single_clustering_iteration', fake_iteration)
        monkeypatch.setattr(
            clustering_postprocessing, 'apply_minimum_size_filter_to_clustering_result', fake_min_size
        )
        monkeypatch.setattr(
            clustering_postprocessing, 'select_diverse_playlists_with_genre_coverage', fake_top_n
        )
        naming_preview._STATE.update(id='p1', status='running')
        return calls

    def test_one_kmeans_pass_at_a_fixed_k(self, patched, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 0)
        result = naming_preview._cluster_sample('p1')
        assert result['named_playlists']
        iteration = patched['iteration']
        assert iteration['clustering_method'] == 'kmeans'
        assert iteration['num_clusters_min_max'] == (3, 3)
        assert iteration['elite_solutions_params_list'] == []
        assert iteration['item_ids_for_subset'] == ['a', 'b', 'c', 'd']
        assert patched['min_size'] == 2
        assert 'top_n' not in patched

    def test_the_diverse_top_n_selection_runs_when_configured(self, patched, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 8)
        naming_preview._cluster_sample('p1')
        assert patched['top_n'] == (8, {'rock': 2})

    def test_too_few_songs_fail_cleanly(self, patched, monkeypatch):
        monkeypatch.setattr(config, 'MIN_PLAYLIST_SIZE_FOR_TOP_N', 50)
        assert naming_preview._cluster_sample('p1') is None
        assert naming_preview.preview_status()['status'] == 'failed'
        assert 'iteration' not in patched

    def test_no_playlists_fail_cleanly(self, patched, monkeypatch):
        monkeypatch.setattr(config, 'TOP_N_CLUSTERING_PLAYLIST', 0)
        patched['result'] = {'fitness_score': -1.0}
        assert naming_preview._cluster_sample('p1') is None
        assert naming_preview.preview_status()['status'] == 'failed'


class TestNamePlaylists:
    def test_titles_fall_back_and_are_de_duplicated_like_a_real_run(self, monkeypatch):
        from tasks.ai import api

        answers = {
            'A_automatic': 'Velvet Light',
            'B_automatic': 'Velvet Light',
            'C_automatic': None,
        }
        seen = []

        def fake_title(instructions, songs, ai_config):
            seen.append(instructions)
            name = songs[0][0]
            if name == 'D_automatic':
                raise RuntimeError('provider exploded')
            return answers[name]

        monkeypatch.setattr(api, 'get_ai_playlist_title', fake_title)
        naming_preview._STATE.update(id='p1', status='running')
        result = {'named_playlists': {
            'A_automatic': [('A_automatic', 'Song 1', 'Artist 1'), ('x', 'Song 2', None)],
            'B_automatic': [('B_automatic', 'Song 3', 'Artist 3')],
            'C_automatic': [('C_automatic', 'Song 4', 'Artist 4')],
            'D_automatic': [('D_automatic', 'Song 5', 'Artist 5')],
            'Empty_automatic': [],
        }}
        naming_preview._name_playlists('p1', result, 'Edited.', {'provider': 'OLLAMA'})
        state = naming_preview.preview_status()
        assert state['total'] == 4 and state['done'] == 4
        assert [t['title'] for t in state['titles']] == [
            'Velvet Light', 'Velvet Light (2)', 'C_automatic', 'D_automatic',
        ]
        assert [t['from_ai'] for t in state['titles']] == [True, True, False, False]
        assert state['titles'][0]['sample'] == ['Song 1 - Artist 1', 'Song 2 - Unknown Artist']
        assert state['titles'][0]['song_count'] == 2
        assert set(seen) == {'Edited.'}
