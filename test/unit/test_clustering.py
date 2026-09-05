# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Clustering pipeline helpers across clustering, clustering_helper, and postprocessing.

Covers parameter mutation, data scaling, genre stratification, JSON sanitization,
and playlist post-processing that together drive the evolutionary clustering run.

Main Features:
* Param mutation stays within bounds; data prep scales features/embeddings and
  returns None for empty input
* Genre maps, per-genre target counts, primary-genre lookup, and cluster naming
  by tempo and top moods
* sanitize_for_json unwraps numpy types; postprocessing applies min-size filter,
  title/artist dedup, and top-N diverse playlist selection
* Finished batches are absorbed on the parent connection that also writes the
  progress, and a failed write rolls the reap and every absorbed key back
  together, the two stop counters and the evolving subset included
"""

import numpy as np
from unittest.mock import MagicMock, Mock, patch
from collections import defaultdict


class TestParameterMutation:
    def test_mutate_param_integer_within_bounds(self):
        from tasks.clustering_helper import _mutate_param

        for _ in range(10):
            result = _mutate_param(50, min_val=0, max_val=100, delta=10, is_float=False)
            assert 0 <= result <= 100
            assert isinstance(result, int)

    def test_mutate_param_integer_at_min_boundary(self):
        from tasks.clustering_helper import _mutate_param

        result = _mutate_param(0, min_val=0, max_val=100, delta=10, is_float=False)
        assert 0 <= result <= 100

    def test_mutate_param_integer_at_max_boundary(self):
        from tasks.clustering_helper import _mutate_param

        result = _mutate_param(100, min_val=0, max_val=100, delta=10, is_float=False)
        assert 0 <= result <= 100

    def test_mutate_param_float_within_bounds(self):
        from tasks.clustering_helper import _mutate_param

        for _ in range(10):
            result = _mutate_param(0.5, min_val=0.0, max_val=1.0, delta=0.1, is_float=True)
            assert 0.0 <= result <= 1.0
            assert isinstance(result, float)

    def test_mutate_param_float_precision(self):
        from tasks.clustering_helper import _mutate_param

        result = _mutate_param(0.5, min_val=0.0, max_val=1.0, delta=0.05, is_float=True)
        assert abs(result - 0.5) <= 0.1

    def test_mutate_param_clipping_low(self):
        from tasks.clustering_helper import _mutate_param

        result = _mutate_param(5, min_val=10, max_val=100, delta=1, is_float=False)
        assert result == 10

    def test_mutate_param_clipping_high(self):
        from tasks.clustering_helper import _mutate_param

        result = _mutate_param(95, min_val=0, max_val=90, delta=1, is_float=False)
        assert result == 90


class TestViablePlaylistSelection:
    def test_viable_playlists_counts_only_min_size_playlists_capped_at_minimum(self):
        from tasks.clustering import _viable_playlists
        import config

        result = {
            'named_playlists': {
                **{
                    f'big{i}': list(range(25))
                    for i in range(config.TOP_N_CLUSTERING_PLAYLIST + 3)
                },
                'tiny': list(range(5)),
            }
        }
        assert _viable_playlists(result) == config.TOP_N_CLUSTERING_PLAYLIST
        assert _viable_playlists({'named_playlists': {'tiny': list(range(5))}}) == 0
        assert _viable_playlists(None) == 0
        assert _viable_playlists(result, target=3) == 3
        assert _viable_playlists({'fitness_score': -1.0}) == 0

    def test_a_viable_result_outranks_a_higher_scoring_shredded_result(self):
        from tasks.clustering import _viable_playlists

        shredded = {
            'named_playlists': {'A': list(range(30)), 'B': list(range(5))},
            'fitness_score': 99.0,
        }
        viable = {
            'named_playlists': {f'P{i}': list(range(25)) for i in range(8)},
            'fitness_score': 10.0,
        }
        shredded_rank = (_viable_playlists(shredded), shredded['fitness_score'])
        viable_rank = (_viable_playlists(viable), viable['fitness_score'])
        assert viable_rank > shredded_rank


class TestEarlyStopCounting:
    @staticmethod
    def _batch_result(score):
        import config

        return {
            'status': config.TASK_STATUS_SUCCESS,
            'iterations_completed_in_batch': 20,
            'best_result_from_batch': {
                'fitness_score': score,
                'parameters': {'method': 'kmeans'},
                'named_playlists': {f'P{i}': list(range(25)) for i in range(8)},
            },
        }

    @staticmethod
    def _state(top_n=10):
        return {
            'runs_completed': 0, 'total_runs': 100, 'best_score': -1.0,
            'best_result': None, 'elite_solutions': [], 'last_subset_ids': [],
            'failed_batches': 0, 'stale_batches': 0, 'job_prefix': 'p',
            'top_n_clustering_playlist': top_n, 'batches_launched': 0,
        }

    @staticmethod
    def _reaped_rows(batch_results):
        import config

        rows = []
        for i, result in enumerate(batch_results):
            status = config.TASK_STATUS_SUCCESS if result else config.TASK_STATUS_FAILURE
            details = {}
            if result:
                details = {
                    'iterations_completed_in_batch': result['iterations_completed_in_batch'],
                    'full_best_result_from_batch': result['best_result_from_batch'],
                }
            rows.append({
                'task_id': f'p_batch_{i}', 'status': status,
                'sub_type_identifier': f'Batch_{i}', 'details': details,
            })
        return rows

    @staticmethod
    def _patch_reap(monkeypatch, reap):
        from tasks import clustering

        db = MagicMock()
        monkeypatch.setattr(clustering, 'get_db', lambda: db)
        monkeypatch.setattr(clustering.taskqueue, 'reap_finished_children', reap)
        return db

    @staticmethod
    def _persisted(_message):
        return True

    def _run_monitor(self, monkeypatch, batch_results, top_n=10):
        from tasks import clustering

        rows = self._reaped_rows(batch_results)
        self._patch_reap(monkeypatch, lambda _pid, conn=None: rows)
        state = self._state(top_n)
        clustering._absorb_finished_batches(state, 'p', self._persisted)
        return state

    def test_a_batch_from_another_server_phase_is_not_credited_to_this_one(
        self, monkeypatch
    ):
        from tasks import clustering
        import config

        self._patch_reap(
            monkeypatch,
            lambda _pid, conn=None: [{
                'task_id': 'p_s9_batch_0', 'status': config.TASK_STATUS_SUCCESS,
                'sub_type_identifier': 'Batch_0',
                'details': {'iterations_completed_in_batch': 20},
            }],
        )
        state = self._state()

        clustering._absorb_finished_batches(state, 'p', self._persisted)

        assert state['runs_completed'] == 0, (
            'reap returns every finished child of the parent, so a batch belonging to '
            'a different server phase must not be counted into this one'
        )

    def test_three_batches_without_a_better_result_mark_the_search_stale(self, monkeypatch):
        state = self._run_monitor(
            monkeypatch,
            [self._batch_result(10), self._batch_result(9),
             self._batch_result(8), self._batch_result(7)],
        )
        assert state['stale_batches'] == 3
        assert state['best_score'] == 10

    def test_a_better_batch_resets_the_stale_counter(self, monkeypatch):
        state = self._run_monitor(
            monkeypatch,
            [self._batch_result(10), self._batch_result(9),
             self._batch_result(12), self._batch_result(11)],
        )
        assert state['stale_batches'] == 1
        assert state['best_score'] == 12

    def test_failed_batches_count_toward_the_early_stop(self, monkeypatch):
        state = self._run_monitor(
            monkeypatch,
            [self._batch_result(10), None, None],
        )
        assert state['stale_batches'] == 2

    def test_a_batch_is_credited_exactly_once_because_reaping_deletes_its_row(
        self, monkeypatch
    ):
        from tasks import clustering
        import config

        rows = [{
            'task_id': 'p_batch_0', 'status': config.TASK_STATUS_SUCCESS,
            'sub_type_identifier': 'Batch_0',
            'details': {'iterations_completed_in_batch': 20},
        }]

        def reap(_pid, conn=None):
            drained, rows[:] = list(rows), []
            return drained

        self._patch_reap(monkeypatch, reap)
        state = self._state()

        clustering._absorb_finished_batches(state, 'p', self._persisted)
        clustering._absorb_finished_batches(state, 'p', self._persisted)

        assert state['runs_completed'] == 20, (
            'the row is deleted as it is reaped, so there is no processed-id set to '
            'keep and nothing to double count'
        )

    def test_an_explicit_zero_keep_all_target_is_not_coerced_to_the_default(self, monkeypatch):
        from tasks import clustering

        seen_targets = []
        original = clustering._viable_playlists

        def spy(result, target):
            seen_targets.append(target)
            return original(result, target)

        monkeypatch.setattr(clustering, '_viable_playlists', spy)
        self._run_monitor(monkeypatch, [self._batch_result(10)], top_n=0)
        assert seen_targets
        assert all(target == 0 for target in seen_targets)

    def test_the_child_row_is_deleted_on_the_same_connection_the_progress_is_written_on(
        self, monkeypatch
    ):
        from tasks import clustering

        seen = {}
        rows = self._reaped_rows([self._batch_result(10)])

        def reap(_pid, conn=None):
            seen['conn'] = conn
            return rows

        db = self._patch_reap(monkeypatch, reap)
        state = self._state()

        absorbed = clustering._absorb_finished_batches(state, 'p', self._persisted)

        assert absorbed == 1
        assert seen['conn'] is db, (
            'the reap must run on the parent connection that also writes the absorbed '
            'runs, so a parent that dies between the two cannot lose a finished batch '
            'whose row is already gone'
        )

    def test_a_progress_write_that_fails_rolls_the_reap_back_and_forgets_the_batch(
        self, monkeypatch
    ):
        from tasks import clustering

        rows = self._reaped_rows([self._batch_result(10)])
        db = self._patch_reap(monkeypatch, lambda _pid, conn=None: rows)
        state = self._state()

        absorbed = clustering._absorb_finished_batches(state, 'p', lambda _message: False)

        assert absorbed == 0
        assert db.rollback.called, (
            'the reaped row must come back so the next pass can absorb it again'
        )
        assert state['runs_completed'] == 0
        assert state['best_score'] == -1.0
        assert state['best_result'] is None
        assert state['elite_solutions'] == [], (
            'in-memory state that was never persisted must be rolled back with the '
            'row, otherwise the batch is counted but its result is lost'
        )

    def test_a_rolled_back_reap_restores_the_stop_counters_and_the_subset_too(
        self, monkeypatch
    ):
        from tasks import clustering
        import config

        resumed_result = {
            'fitness_score': 5.0, 'parameters': {'method': 'kmeans'},
            'named_playlists': {f'P{i}': list(range(25)) for i in range(8)},
        }
        rows = [
            {
                'task_id': 'p_batch_0', 'status': config.TASK_STATUS_FAILURE,
                'sub_type_identifier': 'Batch_0', 'details': {},
            },
            {
                'task_id': 'p_batch_1', 'status': config.TASK_STATUS_SUCCESS,
                'sub_type_identifier': 'Batch_1',
                'details': {
                    'iterations_completed_in_batch': 20,
                    'final_subset_track_ids': ['x', 'y'],
                    'full_best_result_from_batch': {
                        'fitness_score': 10.0, 'parameters': {'method': 'kmeans'},
                        'named_playlists': {
                            f'P{i}': list(range(25)) for i in range(8)
                        },
                    },
                },
            },
        ]
        db = self._patch_reap(monkeypatch, lambda _pid, conn=None: rows)
        state = self._state()
        state.update({
            'runs_completed': 40,
            'best_score': 5.0,
            'best_result': resumed_result,
            'elite_solutions': [{'score': 5.0, 'params': {'method': 'kmeans'}}],
            'failed_batches': 2,
            'stale_batches': 1,
            'last_subset_ids': ['a'],
        })

        absorbed = clustering._absorb_finished_batches(state, 'p', lambda _message: False)

        assert absorbed == 0
        assert db.rollback.called
        assert state['runs_completed'] == 40
        assert state['best_score'] == 5.0
        assert state['best_result'] is resumed_result
        assert state['elite_solutions'] == [
            {'score': 5.0, 'params': {'method': 'kmeans'}}
        ]
        assert state['failed_batches'] == 2, (
            'the failure was never recorded anywhere, so keeping its increment would '
            'double count it toward CLUSTERING_MAX_FAILED_BATCHES and stop the '
            'launches of a run that has not failed that often'
        )
        assert state['stale_batches'] == 1, (
            'the better result that reset this counter was rolled back with it, so a '
            'kept value trips CLUSTERING_EARLY_STOP_BATCHES against evidence that no '
            'longer exists and truncates the search'
        )
        assert state['last_subset_ids'] == ['a'], (
            'the subset the discarded batch ended on was never persisted, so the next '
            'batch must keep evolving from the one that was'
        )

    def test_a_reap_that_finds_nothing_ends_its_transaction_without_writing_progress(
        self, monkeypatch
    ):
        from tasks import clustering

        messages = []

        def persist(message):
            messages.append(message)
            return True

        db = self._patch_reap(monkeypatch, lambda _pid, conn=None: [])
        state = self._state()

        absorbed = clustering._absorb_finished_batches(state, 'p', persist)

        assert absorbed == 0
        assert messages == []
        assert db.commit.called, (
            'the drain loop reaps every few seconds, so a pass with nothing to absorb '
            'must not leave the parent connection idle in transaction'
        )


class TestSubsetExactSize:
    def test_an_oversized_stratified_sample_has_exact_precomputed_size(self, monkeypatch):
        from tasks import clustering_helper

        monkeypatch.setattr(clustering_helper, 'CLUSTERING_SUBSET_SONGS', 50)
        genre_map = {
            'rock': [
                {'item_id': str(i), 'mood_vector': 'rock:0.9'} for i in range(300)
            ]
        }
        subset = clustering_helper._get_stratified_song_subset(genre_map, 200)
        assert len(subset) == 50

    def test_oversized_sample_balances_genres_before_selecting_tracks(self, monkeypatch):
        from tasks import clustering_helper

        monkeypatch.setattr(clustering_helper, 'CLUSTERING_SUBSET_SONGS', 50)
        monkeypatch.setattr(clustering_helper, 'STRATIFIED_GENRES', ['rock', 'pop', 'jazz'])
        genre_map = {
            'rock': [
                {'item_id': f'r{i}', 'mood_vector': 'rock:0.9'} for i in range(300)
            ],
            'pop': [
                {'item_id': f'p{i}', 'mood_vector': 'pop:0.9'} for i in range(300)
            ],
            'jazz': [
                {'item_id': f'j{i}', 'mood_vector': 'jazz:0.9'} for i in range(10)
            ],
        }

        subset = clustering_helper._get_stratified_song_subset(genre_map, 200)
        counts = {
            genre: sum(
                clustering_helper._get_track_primary_genre(track) == genre
                for track in subset
            )
            for genre in genre_map
        }

        assert len(subset) == 50
        assert counts == {'rock': 20, 'pop': 20, 'jazz': 10}

    def test_real_configured_cap_is_exact_and_stratified(self, monkeypatch):
        from tasks import clustering_helper

        monkeypatch.setattr(clustering_helper, 'CLUSTERING_SUBSET_SONGS', 10_000)
        monkeypatch.setattr(clustering_helper, 'STRATIFIED_GENRES', ['rock', 'pop', 'jazz'])
        genre_map = {
            genre: [
                {'item_id': f'{genre}-{i}', 'mood_vector': f'{genre}:0.9'}
                for i in range(5000)
            ]
            for genre in ('rock', 'pop', 'jazz')
        }

        subset = clustering_helper._get_stratified_song_subset(genre_map, 5000)
        counts = [
            sum(
                clustering_helper._get_track_primary_genre(track) == genre
                for track in subset
            )
            for genre in genre_map
        ]

        assert len(subset) == 10_000
        assert max(counts) - min(counts) <= 1

    def test_top_up_stays_stratified_when_base_target_is_too_small(self, monkeypatch):
        from tasks import clustering_helper

        monkeypatch.setattr(clustering_helper, 'CLUSTERING_SUBSET_SONGS', 50)
        monkeypatch.setattr(clustering_helper, 'STRATIFIED_GENRES', ['rock', 'pop', 'jazz'])
        genre_map = {
            'rock': [
                {'item_id': f'r{i}', 'mood_vector': 'rock:0.9'} for i in range(1000)
            ],
            'pop': [
                {'item_id': f'p{i}', 'mood_vector': 'pop:0.9'} for i in range(100)
            ],
            'jazz': [
                {'item_id': f'j{i}', 'mood_vector': 'jazz:0.9'} for i in range(10)
            ],
        }

        subset = clustering_helper._get_stratified_song_subset(genre_map, 5)
        counts = {
            genre: sum(
                clustering_helper._get_track_primary_genre(track) == genre
                for track in subset
            )
            for genre in genre_map
        }

        assert len(subset) == 50
        assert counts == {'rock': 20, 'pop': 20, 'jazz': 10}

    def test_a_sparse_stratified_sample_is_topped_up_with_random_songs(self, monkeypatch):
        from tasks import clustering_helper

        monkeypatch.setattr(clustering_helper, 'CLUSTERING_SUBSET_SONGS', 50)
        genre_map = {
            'rock': [
                {'item_id': f'r{i}', 'mood_vector': 'rock:0.9'} for i in range(20)
            ],
            '__other__': [
                {'item_id': f'o{i}', 'mood_vector': ''} for i in range(100)
            ],
        }
        subset = clustering_helper._get_stratified_song_subset(genre_map, 10)
        assert len(subset) == 50
        assert len({t['item_id'] for t in subset}) == 50

    def test_a_library_smaller_than_the_subset_size_returns_every_song(self, monkeypatch):
        from tasks import clustering_helper

        monkeypatch.setattr(clustering_helper, 'CLUSTERING_SUBSET_SONGS', 50)
        genre_map = {
            'rock': [
                {'item_id': f'r{i}', 'mood_vector': 'rock:0.9'} for i in range(30)
            ],
            'pop': [
                {'item_id': f'p{i}', 'mood_vector': 'pop:0.9'} for i in range(10)
            ],
        }
        subset = clustering_helper._get_stratified_song_subset(genre_map, 10)
        expected_ids = {
            track['item_id']
            for tracks in genre_map.values()
            for track in tracks
        }

        assert len(subset) == 40
        assert {track['item_id'] for track in subset} == expected_ids


class TestDbscanOversizeSplit:
    def test_an_oversized_dbscan_cluster_is_split_into_playlist_sized_chunks(self):
        from tasks.clustering_helper import _split_oversized_clusters
        import config

        rng = np.random.default_rng(3)
        data = rng.standard_normal((600, 4))
        labels = np.zeros(600, dtype=int)
        split = _split_oversized_clusters(labels, data)
        sizes = [int((split == c).sum()) for c in set(split.tolist()) if c != -1]
        assert len(sizes) >= 2
        assert max(sizes) <= config.CLUSTERING_MAX_PLAYLIST_SONGS
        assert sum(sizes) == 600

    def test_small_clusters_and_noise_are_left_untouched(self):
        from tasks.clustering_helper import _split_oversized_clusters

        rng = np.random.default_rng(3)
        data = rng.standard_normal((60, 4))
        labels = np.array([0] * 30 + [1] * 20 + [-1] * 10)
        split = _split_oversized_clusters(labels, data)
        assert (split == labels).all()


class TestDataPreparationAndScaling:
    def test_prepare_and_normalize_data_with_features(self):
        from tasks.clustering_helper import _prepare_and_normalize_data

        x_feat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        x_embed = None

        normalized_data = _prepare_and_normalize_data(x_feat, x_embed, use_embeddings=False)

        assert normalized_data is not None
        assert normalized_data.shape == x_feat.shape

        assert np.allclose(np.linalg.norm(normalized_data, axis=1), 1.0, atol=1e-6)

    def test_prepare_and_normalize_data_with_embeddings(self):
        from tasks.clustering_helper import _prepare_and_normalize_data

        x_feat = np.array([[1.0, 2.0], [3.0, 4.0]])
        x_embed = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]])

        normalized_data = _prepare_and_normalize_data(x_feat, x_embed, use_embeddings=True)

        assert normalized_data is not None
        assert normalized_data.shape == x_embed.shape
        assert normalized_data.shape[1] == 4

    def test_prepare_and_normalize_data_returns_none_for_empty(self):
        from tasks.clustering_helper import _prepare_and_normalize_data

        x_feat = np.array([])
        x_embed = None

        result = _prepare_and_normalize_data(x_feat, x_embed, use_embeddings=False)

        assert result is None

    def test_prepare_and_normalize_data_returns_none_for_zero_rows(self):
        from tasks.clustering_helper import _prepare_and_normalize_data

        x_feat = np.empty((0, 5))
        x_embed = None

        result = _prepare_and_normalize_data(x_feat, x_embed, use_embeddings=False)

        assert result is None


class TestFeatureCentroidCalculation:
    def test_get_feature_centroid_for_embedding_cluster_basic(self):
        from tasks.clustering_helper import _get_feature_centroid_for_embedding_cluster

        labels = np.array([0, 0, 1, 1, 0])
        x_feat = np.array(
            [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [5.0, 6.0, 7.0], [5.5, 6.5, 7.5], [2.0, 3.0, 4.0]]
        )

        centroid = _get_feature_centroid_for_embedding_cluster(0, labels, x_feat)

        assert centroid is not None
        assert centroid.shape == (3,)

        expected_centroid = np.mean(x_feat[[0, 1, 4]], axis=0)
        np.testing.assert_array_almost_equal(centroid, expected_centroid)

    def test_get_feature_centroid_for_single_member_cluster(self):
        from tasks.clustering_helper import _get_feature_centroid_for_embedding_cluster

        labels = np.array([0, 1, 2])
        x_feat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        centroid = _get_feature_centroid_for_embedding_cluster(1, labels, x_feat)

        np.testing.assert_array_almost_equal(centroid, x_feat[1])

    def test_get_feature_centroid_for_empty_cluster(self):
        from tasks.clustering_helper import _get_feature_centroid_for_embedding_cluster

        labels = np.array([0, 0, 1, 1])
        x_feat = np.array([[1.0], [2.0], [3.0], [4.0]])

        result = _get_feature_centroid_for_embedding_cluster(5, labels, x_feat)

        assert result is None

    def test_get_feature_centroid_maintains_dimensionality(self):
        from tasks.clustering_helper import _get_feature_centroid_for_embedding_cluster

        labels = np.array([0, 0, 0, 1, 1])
        x_feat = np.random.rand(5, 50)

        centroid = _get_feature_centroid_for_embedding_cluster(0, labels, x_feat)

        assert centroid.shape == (50,)


class TestTrackPrimaryGenre:
    @patch('tasks.clustering_helper.STRATIFIED_GENRES', ['rock', 'pop', 'jazz', 'metal'])
    def test_get_track_primary_genre_with_mood_vector(self):
        from tasks.clustering_helper import _get_track_primary_genre

        track = {'mood_vector': 'rock:0.8,pop:0.2,jazz:0.1'}

        genre = _get_track_primary_genre(track)

        assert genre == 'rock'

    def test_get_track_primary_genre_with_no_mood_vector(self):
        from tasks.clustering_helper import _get_track_primary_genre

        track = {'title': 'Some Song'}

        genre = _get_track_primary_genre(track)

        assert genre == '__other__'

    def test_get_track_primary_genre_with_empty_mood_vector(self):
        from tasks.clustering_helper import _get_track_primary_genre

        track = {'mood_vector': ''}

        genre = _get_track_primary_genre(track)

        assert genre == '__other__'

    def test_get_track_primary_genre_with_none_mood_vector(self):
        from tasks.clustering_helper import _get_track_primary_genre

        track = {'mood_vector': None}

        genre = _get_track_primary_genre(track)

        assert genre == '__other__'


class TestGenreMapPreparation:
    @patch('tasks.clustering_helper.STRATIFIED_GENRES', ['rock', 'pop', 'jazz', 'metal'])
    def test_prepare_genre_map_groups_by_primary_genre_from_stratified_list(self):
        from tasks.clustering import _prepare_genre_map

        rows = [
            {'item_id': '1', 'mood_vector': 'rock:0.9,pop:0.1'},
            {'item_id': '2', 'mood_vector': 'rock:0.8,jazz:0.2'},
            {'item_id': '3', 'mood_vector': 'pop:0.9,rock:0.1'},
            {'item_id': '4', 'mood_vector': 'jazz:0.7,rock:0.3'},
            {'item_id': '5', 'mood_vector': 'electronic:0.9,rock:0.1'},
        ]

        genre_map = _prepare_genre_map(rows)

        assert 'rock' in genre_map
        assert 'pop' in genre_map
        assert 'jazz' in genre_map
        assert 'electronic' not in genre_map
        assert len(genre_map['rock']) == 3
        assert len(genre_map['pop']) == 1
        assert len(genre_map['jazz']) == 1

    def test_prepare_genre_map_with_no_mood_vector(self):
        from tasks.clustering import _prepare_genre_map

        rows = [
            {'item_id': '1', 'mood_vector': ''},
            {'item_id': '2', 'mood_vector': None},
            {'item_id': '3', 'title': 'Song'},
        ]

        genre_map = _prepare_genre_map(rows)

        assert len(genre_map) == 0

    def test_prepare_genre_map_empty_input(self):
        from tasks.clustering import _prepare_genre_map

        genre_map = _prepare_genre_map([])

        assert isinstance(genre_map, defaultdict)
        assert len(genre_map) == 0


class TestTargetSongsCalculation:
    @patch('tasks.clustering.STRATIFIED_GENRES', ['rock', 'pop', 'jazz', 'metal'])
    def test_calculate_target_songs_per_genre_basic(self):
        from tasks.clustering import _calculate_target_songs_per_genre

        genre_map = {
            'rock': [{'id': i} for i in range(100)],
            'pop': [{'id': i} for i in range(50)],
            'jazz': [{'id': i} for i in range(150)],
            'metal': [{'id': i} for i in range(75)],
        }

        target = _calculate_target_songs_per_genre(genre_map, percentile=50, min_songs=10)

        assert 70 <= target <= 100
        assert isinstance(target, int)

    def test_calculate_target_songs_respects_minimum(self):
        from tasks.clustering import _calculate_target_songs_per_genre

        genre_map = {'rock': [{'id': 1}], 'pop': [{'id': 2}]}

        target = _calculate_target_songs_per_genre(genre_map, percentile=50, min_songs=100)

        assert target == 100

    @patch('tasks.clustering.STRATIFIED_GENRES', ['rock', 'pop', 'jazz', 'metal'])
    def test_calculate_target_songs_high_percentile(self):
        from tasks.clustering import _calculate_target_songs_per_genre

        genre_map = {
            'rock': [{'id': i} for i in range(100)],
            'pop': [{'id': i} for i in range(200)],
            'jazz': [{'id': i} for i in range(50)],
        }

        target = _calculate_target_songs_per_genre(genre_map, percentile=90, min_songs=10)

        assert target >= 150

    def test_calculate_target_songs_empty_genre_map(self):
        from tasks.clustering import _calculate_target_songs_per_genre

        genre_map = {}

        target = _calculate_target_songs_per_genre(genre_map, percentile=50, min_songs=20)

        assert target == 20


class TestSanitizeForJson:
    def test_sanitize_numpy_array(self):
        from sanitization import sanitize_for_json as _sanitize_for_json

        obj = np.array([1.0, 2.0, 3.0])
        result = _sanitize_for_json(obj)

        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_sanitize_numpy_integers(self):
        from sanitization import sanitize_for_json as _sanitize_for_json

        obj = {
            'int8': np.int8(42),
            'int16': np.int16(100),
            'int32': np.int32(1000),
            'int64': np.int64(10000),
        }

        result = _sanitize_for_json(obj)

        for key, val in result.items():
            assert isinstance(val, int)
            assert not isinstance(val, np.integer)

    def test_sanitize_numpy_floats(self):
        from sanitization import sanitize_for_json as _sanitize_for_json

        obj = {'float32': np.float32(3.14), 'float64': np.float64(2.718)}

        result = _sanitize_for_json(obj)

        for key, val in result.items():
            assert isinstance(val, float)
            assert not isinstance(val, np.floating)

    def test_sanitize_numpy_bool(self):
        from sanitization import sanitize_for_json as _sanitize_for_json

        obj = {'flag': np.bool_(True)}
        result = _sanitize_for_json(obj)

        assert isinstance(result['flag'], bool)
        assert result['flag'] is True

    def test_sanitize_nested_structures(self):
        from sanitization import sanitize_for_json as _sanitize_for_json

        obj = {
            'array': np.array([1, 2, 3]),
            'nested': {'float': np.float64(1.5), 'list': [np.int32(5), np.int32(10)]},
        }

        result = _sanitize_for_json(obj)

        assert isinstance(result['array'], list)
        assert isinstance(result['nested']['float'], float)
        assert all(isinstance(x, int) for x in result['nested']['list'])

    def test_sanitize_preserves_native_types(self):
        from sanitization import sanitize_for_json as _sanitize_for_json

        obj = {
            'string': 'hello',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'list': [1, 2, 3],
            'none': None,
        }

        result = _sanitize_for_json(obj)

        assert result == obj


class TestGetVectorsFromDatabase:
    def test_get_vectors_from_database_basic(self):
        from tasks.clustering_postprocessing import get_vectors_from_database

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)

        vector1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        vector2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)

        mock_cursor.fetchall.return_value = [
            {'item_id': 'song1', 'embedding': vector1.tobytes()},
            {'item_id': 'song2', 'embedding': vector2.tobytes()},
        ]

        item_ids = ['song1', 'song2']
        result = get_vectors_from_database(item_ids, mock_conn)

        assert len(result) == 2
        assert 'song1' in result
        assert 'song2' in result
        np.testing.assert_array_almost_equal(result['song1'], vector1)
        np.testing.assert_array_almost_equal(result['song2'], vector2)

    def test_get_vectors_from_database_empty(self):
        from tasks.clustering_postprocessing import get_vectors_from_database

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.fetchall.return_value = []

        result = get_vectors_from_database(['song1'], mock_conn)

        assert len(result) == 0


class TestTitleArtistDeduplication:
    def test_title_artist_deduplication_removes_exact_duplicates(self):
        from tasks.clustering_postprocessing import apply_title_artist_deduplication

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.fetchall.return_value = [
            {'item_id': 'song1', 'title': 'Song A', 'author': 'Artist X'},
            {'item_id': 'song2', 'title': 'Song A', 'author': 'Artist X'},
            {'item_id': 'song3', 'title': 'Song B', 'author': 'Artist Y'},
        ]

        songs = [{'item_id': 'song1'}, {'item_id': 'song2'}, {'item_id': 'song3'}]
        result = apply_title_artist_deduplication(songs, mock_conn)

        assert len(result) == 2
        result_ids = [s['item_id'] for s in result]
        assert 'song1' in result_ids
        assert 'song3' in result_ids

    def test_title_artist_deduplication_case_insensitive(self):
        from tasks.clustering_postprocessing import apply_title_artist_deduplication

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.fetchall.return_value = [
            {'item_id': 'song1', 'title': 'Song A', 'author': 'Artist X'},
            {'item_id': 'song2', 'title': 'SONG A', 'author': 'ARTIST X'},
        ]

        songs = [{'item_id': 'song1'}, {'item_id': 'song2'}]
        result = apply_title_artist_deduplication(songs, mock_conn)

        assert len(result) == 1

    def test_title_artist_deduplication_removes_remastered_versions(self):
        from tasks.clustering_postprocessing import apply_title_artist_deduplication

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.fetchall.return_value = [
            {'item_id': 'song1', 'title': 'Song A', 'author': 'Artist X'},
            {'item_id': 'song2', 'title': 'Song A (Remastered)', 'author': 'Artist X'},
            {'item_id': 'song3', 'title': 'Song A [Explicit]', 'author': 'Artist X'},
        ]

        songs = [{'item_id': 'song1'}, {'item_id': 'song2'}, {'item_id': 'song3'}]
        result = apply_title_artist_deduplication(songs, mock_conn)

        assert len(result) == 1

    def test_title_artist_deduplication_preserves_different_songs(self):
        from tasks.clustering_postprocessing import apply_title_artist_deduplication

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.fetchall.return_value = [
            {'item_id': 'song1', 'title': 'Song A', 'author': 'Artist X'},
            {'item_id': 'song2', 'title': 'Song B', 'author': 'Artist X'},
            {'item_id': 'song3', 'title': 'Song A', 'author': 'Artist Y'},
        ]

        songs = [{'item_id': 'song1'}, {'item_id': 'song2'}, {'item_id': 'song3'}]
        result = apply_title_artist_deduplication(songs, mock_conn)

        assert len(result) == 3

    def test_title_artist_deduplication_empty_input(self):
        from tasks.clustering_postprocessing import apply_title_artist_deduplication

        mock_conn = Mock()
        result = apply_title_artist_deduplication([], mock_conn)

        assert len(result) == 0


class TestMinimumSizeFilter:
    def test_minimum_size_filter_removes_small_playlists(self):
        from tasks.clustering_postprocessing import apply_minimum_size_filter_to_clustering_result

        best_result = {
            'named_playlists': {
                'Large Playlist': [{'item_id': f'song{i}'} for i in range(25)],
                'Medium Playlist': [{'item_id': f'song{i}'} for i in range(20)],
                'Small Playlist': [{'item_id': f'song{i}'} for i in range(15)],
            },
            'playlist_centroids': {
                'Large Playlist': [0.5, 0.5, 0.5],
                'Medium Playlist': [0.3, 0.3, 0.3],
                'Small Playlist': [0.1, 0.1, 0.1],
            },
        }

        result = apply_minimum_size_filter_to_clustering_result(best_result, min_size=20)

        assert len(result['named_playlists']) == 2
        assert 'Large Playlist' in result['named_playlists']
        assert 'Medium Playlist' in result['named_playlists']
        assert 'Small Playlist' not in result['named_playlists']

    def test_minimum_size_filter_preserves_large_playlists(self):
        from tasks.clustering_postprocessing import apply_minimum_size_filter_to_clustering_result

        best_result = {
            'named_playlists': {
                'Playlist A': [{'item_id': f'song{i}'} for i in range(50)],
                'Playlist B': [{'item_id': f'song{i}'} for i in range(100)],
            },
            'playlist_centroids': {
                'Playlist A': [0.5],
                'Playlist B': [0.6],
            },
        }

        result = apply_minimum_size_filter_to_clustering_result(best_result, min_size=20)

        assert len(result['named_playlists']) == 2
        assert len(result['named_playlists']['Playlist A']) == 50
        assert len(result['named_playlists']['Playlist B']) == 100

    def test_minimum_size_filter_updates_centroids(self):
        from tasks.clustering_postprocessing import apply_minimum_size_filter_to_clustering_result

        best_result = {
            'named_playlists': {
                'Keep': [{'item_id': f'song{i}'} for i in range(25)],
                'Remove': [{'item_id': f'song{i}'} for i in range(5)],
            },
            'playlist_centroids': {
                'Keep': [0.5, 0.5],
                'Remove': [0.1, 0.1],
            },
        }

        result = apply_minimum_size_filter_to_clustering_result(best_result, min_size=20)

        assert 'Keep' in result['playlist_centroids']
        assert 'Remove' not in result['playlist_centroids']

    def test_minimum_size_filter_empty_input(self):
        from tasks.clustering_postprocessing import apply_minimum_size_filter_to_clustering_result

        best_result = None
        result = apply_minimum_size_filter_to_clustering_result(best_result, min_size=20)

        assert result is None

    def test_minimum_size_filter_all_playlists_removed(self):
        from tasks.clustering_postprocessing import apply_minimum_size_filter_to_clustering_result

        best_result = {
            'named_playlists': {
                'Small 1': [{'item_id': 'song1'}],
                'Small 2': [{'item_id': 'song2'}],
            },
            'playlist_centroids': {
                'Small 1': [0.1],
                'Small 2': [0.2],
            },
        }

        result = apply_minimum_size_filter_to_clustering_result(best_result, min_size=50)

        assert len(result['named_playlists']) == 0


class TestSelectDiversePlaylists:
    def test_default_strategy_returns_two_top_variants_and_four_other_genres(
        self, monkeypatch
    ):
        from tasks import clustering_postprocessing

        monkeypatch.setattr(
            clustering_postprocessing.secrets,
            'choice',
            lambda options: options[0],
        )
        specs = {
            'Rock Center': ('rock', 0.0, 100),
            'Rock Near': ('rock', 0.1, 90),
            'Rock Far': ('rock', 10.0, 70),
            'Pop Low': ('pop', 20.0, 60),
            'Pop Middle': ('pop', 25.0, 60),
            'Pop High': ('pop', 30.0, 60),
            'Indie Low': ('indie', 40.0, 60),
            'Indie Middle': ('indie', 45.0, 60),
            'Indie High': ('indie', 50.0, 60),
            'Jazz': ('jazz', 60.0, 60),
            'Soul': ('soul', 70.0, 60),
            'Folk': ('folk', 80.0, 60),
            'Country': ('country', 90.0, 60),
        }
        best_result = {
            'named_playlists': {
                name: [{'item_id': f'{name}-{i}'} for i in range(size)]
                for name, (_genre, _vector, size) in specs.items()
            },
            'playlist_centroids': {
                name: [vector] for name, (_genre, vector, _size) in specs.items()
            },
            'playlist_to_centroid_vector_map': {
                name: np.array([vector])
                for name, (_genre, vector, _size) in specs.items()
            },
            'playlist_primary_genres': {
                name: genre for name, (genre, _vector, _size) in specs.items()
            },
        }

        result = clustering_postprocessing.select_diverse_playlists_with_genre_coverage(
            best_result,
            limit=10,
            primary_genre_counts={
                'rock': 3000, 'pop': 2500, 'indie': 2000,
                'jazz': 500, 'soul': 400, 'folk': 300, 'country': 200,
            },
        )

        selected_genres = list(result['playlist_primary_genres'].values())
        assert len(selected_genres) == 10
        assert selected_genres.count('rock') == 2
        assert selected_genres.count('pop') == 2
        assert selected_genres.count('indie') == 2
        assert selected_genres.count('jazz') == 1
        assert selected_genres.count('soul') == 1
        assert selected_genres.count('folk') == 1
        assert selected_genres.count('country') == 1
        assert {'Rock Center', 'Rock Far'} <= set(result['named_playlists'])
        assert 'Rock Near' not in result['named_playlists']

    def test_other_four_use_distinct_non_top_genres_and_maximin_centroids(self):
        from tasks.clustering_postprocessing import (
            select_diverse_playlists_with_genre_coverage,
        )

        specs = {
            'Rock A': ('rock', 0.0), 'Rock B': ('rock', 1.0),
            'Pop A': ('pop', 10.0), 'Pop B': ('pop', 11.0),
            'Jazz A': ('jazz', 20.0), 'Jazz B': ('jazz', 21.0),
            'Soul Near': ('soul', 21.1), 'Soul Far': ('soul', 100.0),
            'Folk': ('folk', 200.0), 'Metal': ('metal', 300.0),
            'Country': ('country', 400.0),
        }
        best_result = {
            'named_playlists': {
                name: [{'item_id': f'{name}-{i}'} for i in range(30)]
                for name in specs
            },
            'playlist_centroids': {
                name: [vector] for name, (_genre, vector) in specs.items()
            },
            'playlist_to_centroid_vector_map': {
                name: np.array([vector]) for name, (_genre, vector) in specs.items()
            },
            'playlist_primary_genres': {
                name: genre for name, (genre, _vector) in specs.items()
            },
        }

        result = select_diverse_playlists_with_genre_coverage(
            best_result,
            limit=10,
            primary_genre_counts={
                'rock': 3000, 'pop': 2500, 'jazz': 2000, 'soul': 1000,
                'folk': 900, 'metal': 800, 'country': 700,
            },
        )

        selected = result['playlist_primary_genres']
        non_top = [genre for genre in selected.values() if genre not in {'rock', 'pop', 'jazz'}]
        assert len(non_top) == 4
        assert len(set(non_top)) == 4
        assert 'Soul Far' in result['named_playlists']
        assert 'Soul Near' not in result['named_playlists']

    def test_limit_is_a_hard_cap_and_short_candidate_sets_are_returned_whole(self):
        from tasks.clustering_postprocessing import select_diverse_playlists_with_genre_coverage

        many = {
            'named_playlists': {
                f'P{i}': [{'item_id': f's{i}'}] for i in range(20)
            },
            'playlist_centroids': {f'P{i}': [float(i)] for i in range(20)},
            'playlist_to_centroid_vector_map': {
                f'P{i}': np.array([float(i)]) for i in range(20)
            },
        }
        short = {
            key: dict(list(value.items())[:4])
            for key, value in many.items()
        }

        assert len(
            select_diverse_playlists_with_genre_coverage(many, 10)['named_playlists']
        ) == 10
        assert len(
            select_diverse_playlists_with_genre_coverage(short, 10)['named_playlists']
        ) == 4

    def test_naming_receives_recent_names_from_previous_runs(self, monkeypatch):
        from tasks import clustering

        received_avoid_names = []

        def fake_name(*args, **kwargs):
            received_avoid_names.extend(args[13])
            return 'Happy Pop'

        monkeypatch.setattr(clustering, '_try_ai_name_playlist', fake_name)
        result = clustering._name_and_prepare_playlists(
            {
                'named_playlists': {
                    'cluster': [('song-1', 'Song', 'Artist')],
                },
                'playlist_centroids': {},
                'playlist_primary_genres': {'cluster': 'pop'},
            },
            'OLLAMA', 'url', 'model', '', '', '', '', '', '', '',
            previous_playlist_names=['Pop Heartbreak_automatic'],
        )

        assert received_avoid_names == ['Pop Heartbreak_automatic']
        assert 'Happy Pop_automatic' in result

    def test_history_avoidance_is_off_by_default_so_recent_names_are_never_fetched(
        self, monkeypatch
    ):
        from tasks import clustering

        def fail_fetch(*args, **kwargs):
            raise AssertionError('history must not be queried when disabled')

        monkeypatch.setattr(clustering, 'get_recent_playlist_names', fail_fetch)

        assert clustering._previous_names_for_naming('server-1') == []

    def test_cluster_naming_ai_history_true_fetches_the_last_60_names(
        self, monkeypatch
    ):
        from tasks import clustering

        calls = []

        def fake_fetch(server_id, limit):
            calls.append((server_id, limit))
            return ['Pop Heartbreak_automatic']

        monkeypatch.setattr(clustering, 'CLUSTER_NAMING_AI_HISTORY', True)
        monkeypatch.setattr(clustering, 'get_recent_playlist_names', fake_fetch)

        assert clustering._previous_names_for_naming('server-1') == [
            'Pop Heartbreak_automatic'
        ]
        assert calls == [('server-1', 60)]

    def test_newest_first_history_is_reversed_so_the_prompt_window_stays_fresh(
        self, monkeypatch
    ):
        from tasks import clustering

        received_avoid_names = []

        def fake_name(*args, **kwargs):
            received_avoid_names.extend(args[13])
            return 'Happy Pop'

        monkeypatch.setattr(clustering, '_try_ai_name_playlist', fake_name)
        clustering._name_and_prepare_playlists(
            {
                'named_playlists': {
                    'cluster': [('song-1', 'Song', 'Artist')],
                },
                'playlist_centroids': {},
                'playlist_primary_genres': {},
            },
            'OLLAMA', 'url', 'model', '', '', '', '', '', '', '',
            previous_playlist_names=['Newest Pop_automatic', 'Oldest Rock_automatic'],
        )

        assert received_avoid_names == [
            'Oldest Rock_automatic', 'Newest Pop_automatic'
        ]

    def test_two_clusters_with_the_same_final_name_get_numbered_not_overwritten(
        self, monkeypatch
    ):
        from tasks import clustering

        monkeypatch.setattr(
            clustering, '_try_ai_name_playlist', lambda *args, **kwargs: 'Happy Pop'
        )
        result = clustering._name_and_prepare_playlists(
            {
                'named_playlists': {
                    'cluster_a': [('song-1', 'Song 1', 'Artist')],
                    'cluster_b': [('song-2', 'Song 2', 'Artist')],
                },
                'playlist_centroids': {},
                'playlist_primary_genres': {},
            },
            'OLLAMA', 'url', 'model', '', '', '', '', '', '', '',
        )

        assert 'Happy Pop_automatic' in result
        assert 'Happy Pop (2)_automatic' in result
        assert result['Happy Pop_automatic'] == [('song-1', 'Song 1', 'Artist')]
        assert result['Happy Pop (2)_automatic'] == [('song-2', 'Song 2', 'Artist')]

    def test_limit_zero_returns_the_original_result(self):
        from tasks.clustering_postprocessing import select_diverse_playlists_with_genre_coverage

        best_result = {
            'named_playlists': {'P1': [{'item_id': 'song1'}]},
            'playlist_centroids': {'P1': [1.0]},
            'playlist_to_centroid_vector_map': {'P1': np.array([1.0])},
        }

        result = select_diverse_playlists_with_genre_coverage(best_result, limit=0)

        assert result == best_result

    def test_empty_candidate_map_returns_the_original_result(self):
        from tasks.clustering_postprocessing import select_diverse_playlists_with_genre_coverage

        best_result = {
            'named_playlists': {},
            'playlist_centroids': {},
            'playlist_to_centroid_vector_map': {},
        }

        result = select_diverse_playlists_with_genre_coverage(best_result, limit=5)

        assert result == best_result


class TestClusterNaming:
    def test_name_cluster_basic(self):
        from tasks.clustering_helper import _name_cluster

        centroid = np.array([0.8, 0.6, 0.9, 0.1, 0.2])
        mood_labels = ['rock', 'pop', 'jazz']

        name, details = _name_cluster(centroid, mood_labels)

        assert isinstance(name, str)
        assert 'Fast' in name
        assert isinstance(details, dict)
        assert 'rock' in details

    def test_name_cluster_slow_tempo(self):
        from tasks.clustering_helper import _name_cluster

        centroid = np.array([0.2, 0.4, 0.5, 0.3, 0.2])
        mood_labels = ['chill', 'relaxed', 'ambient']

        name, _ = _name_cluster(centroid, mood_labels)

        assert 'Slow' in name

    def test_name_cluster_medium_tempo(self):
        from tasks.clustering_helper import _name_cluster

        centroid = np.array([0.5, 0.5, 0.4, 0.4, 0.2])
        mood_labels = ['pop', 'dance', 'electronic']

        name, _ = _name_cluster(centroid, mood_labels)

        assert 'Medium' in name

    def test_name_cluster_top_moods_in_name(self):
        from tasks.clustering_helper import _name_cluster

        centroid = np.array([0.6, 0.5, 0.9, 0.8, 0.1])
        mood_labels = ['rock', 'pop', 'jazz']

        name, details = _name_cluster(centroid, mood_labels)

        assert 'Rock' in name or 'Pop' in name

        assert len(details) == 3


def _batch_launch_args():
    return (
        'kmeans', 2, 4, 0.1, 0.5, 2, 5, 2, 4, 2, 4, 2, 4, 50,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5, True,
    )


def _batch_launch_state():
    return {
        'active_jobs': {},
        'batch_start_times': {},
        'elite_solutions': [],
        'last_subset_ids': [],
        'job_prefix': 'parent-1_s0',
    }


def test_a_batch_row_left_by_a_previous_attempt_is_left_to_finish(monkeypatch):
    import tasks.clustering as clustering
    from contextlib import nullcontext

    monkeypatch.setattr(clustering, 'main_task_start_lock', nullcontext)
    monkeypatch.setattr(
        clustering, 'get_task_info_from_db', lambda _task_id: {'status': 'RUNNING'}
    )

    def enqueue(_func, **_kwargs):
        raise clustering.taskqueue.TaskNotQueued('already exists')

    monkeypatch.setattr(clustering.taskqueue, 'enqueue', enqueue)

    assert clustering._launch_batch_job(
        _batch_launch_state(), 'parent-1', 0, 10, 'genre-token', 5,
        *_batch_launch_args(),
    ) is True, (
        'the row is already queued, so the launch has nothing left to do and the '
        'parent tracks it through live_children like any other batch'
    )


def test_a_run_resumes_from_its_own_row_instead_of_rescanning_its_children():
    import tasks.clustering as clustering
    import json

    task_info = {'details': json.dumps({
        'total_runs': 100, 'runs_completed': 40, 'batches_launched': 3,
        'server_idx': 1, 'elite_solutions': [{'score': 7.5}],
    })}

    resumed = clustering._resumable_progress(task_info, 100)

    assert resumed['runs_completed'] == 40
    assert resumed['batches_launched'] == 3
    assert resumed['server_idx'] == 1
    assert resumed['elite_solutions'] == [{'score': 7.5}]


def test_a_crashed_run_resumes_with_the_winning_result_it_had_found():
    import tasks.clustering as clustering
    import json

    task_info = {'details': json.dumps({
        'total_runs': 100, 'runs_completed': 80, 'batches_launched': 4,
        'best_score': 9.1, 'best_result': {'named_playlists': {'P1': ['a', 'b']}},
    })}

    resumed = clustering._resumable_progress(task_info, 100)

    assert resumed['best_score'] == 9.1
    assert resumed['best_result']['named_playlists'] == {'P1': ['a', 'b']}, (
        'without it a crashed main task would have to redo the whole search'
    )


def test_the_persisted_result_drops_the_blobs_nothing_reads():
    import tasks.clustering as clustering

    kept = clustering._persistable_best_result({
        'named_playlists': {'P1': ['a']},
        'playlist_centroids': {'P1': [0.1]},
        'playlist_to_centroid_vector_map': {'P1': [0.2]},
        'playlist_primary_genres': {'P1': 'rock'},
        'parameters': {'method': 'kmeans'},
        'fitness_score': 9.1,
        'pca_model_details': {'components': [[0.0] * 200] * 4},
    })

    assert 'pca_model_details' not in kept, (
        'the PCA matrix is n_components x EMBEDDING_DIMENSION floats and nothing '
        'reads it once the run is over'
    )
    assert kept['named_playlists'] == {'P1': ['a']}
    assert kept['playlist_to_centroid_vector_map'] == {'P1': [0.2]}, (
        'clustering_postprocessing does read this one'
    )


def test_a_run_whose_size_changed_does_not_resume_stale_progress():
    import tasks.clustering as clustering
    import json

    task_info = {'details': json.dumps({
        'total_runs': 100, 'runs_completed': 40, 'batches_launched': 3,
    })}

    assert clustering._resumable_progress(task_info, 200) is None, (
        'the persisted counters are meaningless against a different run size'
    )


def test_a_first_attempt_has_nothing_to_resume():
    import tasks.clustering as clustering
    import json

    task_info = {'details': json.dumps({
        'total_runs': 100, 'runs_completed': 0, 'batches_launched': 0,
    })}

    assert clustering._resumable_progress(task_info, 100) is None


def test_batches_from_an_earlier_server_phase_are_revoked_not_left_to_fail(monkeypatch):
    import tasks.clustering as clustering

    revoked = []
    monkeypatch.setattr(
        clustering, 'save_task_status',
        lambda job_id, *_a, **_k: revoked.append(job_id),
    )
    monkeypatch.setattr(clustering.taskqueue, 'request_cancel', lambda _job_id: None)
    monkeypatch.setattr(
        clustering.taskqueue, 'live_children',
        lambda _parent: [
            {'task_id': 'p_s0_batch_1'},
            {'task_id': 'p_s1_batch_7'},
            {'task_id': 'p_s2_batch_2'},
        ],
    )

    count = clustering._revoke_foreign_batches('p', 'p_s0')

    assert count == 2
    assert revoked == ['p_s1_batch_7', 'p_s2_batch_2'], (
        'there is one shared payload slot per parent, so a batch queued for another '
        'server phase can only fail once this phase overwrites the genre map'
    )


def test_cancelled_parent_cannot_enqueue_a_new_clustering_batch(monkeypatch):
    import tasks.clustering as clustering
    from contextlib import nullcontext

    monkeypatch.setattr(clustering, 'main_task_start_lock', nullcontext)
    monkeypatch.setattr(clustering, 'get_task_info_from_db', lambda _task_id: None)
    queue_calls = []
    monkeypatch.setattr(
        clustering.taskqueue, 'enqueue',
        lambda func, **kwargs: queue_calls.append((func, kwargs)),
    )

    launched = clustering._launch_batch_job(
        _batch_launch_state(), 'parent-1', 0, 10, 'genre-token', 5,
        *_batch_launch_args(),
    )

    assert launched is False
    assert queue_calls == []


def test_clustering_batch_parent_check_and_enqueue_share_the_cancel_lock(monkeypatch):
    import tasks.clustering as clustering

    held = {'value': False}

    class _Lock:
        def __enter__(self):
            held['value'] = True

        def __exit__(self, *_args):
            held['value'] = False

    monkeypatch.setattr(clustering, 'main_task_start_lock', _Lock)
    monkeypatch.setattr(
        clustering,
        'get_task_info_from_db',
        lambda _task_id: {'status': 'RUNNING'},
    )

    def enqueue(_func, **kwargs):
        assert held['value'] is True

    monkeypatch.setattr(clustering.taskqueue, 'enqueue', enqueue)

    assert clustering._launch_batch_job(
        _batch_launch_state(), 'parent-1', 0, 10, 'genre-token', 5,
        *_batch_launch_args(),
    ) is True


def _revoked_result(fn, **kwargs):
    from taskqueue import TaskCancelled

    try:
        return fn(**kwargs)
    except TaskCancelled:
        return {'status': 'REVOKED'}


def test_batch_start_racing_the_cancel_wipe_never_recreates_a_child_row(monkeypatch):
    import sys
    import types
    from contextlib import nullcontext
    from flask import Flask
    import tasks.clustering as clustering
    import tasks.task_run as task_run

    fake_flask_app = types.ModuleType('flask_app')
    fake_flask_app.app = Flask('cancelled-batch-start')
    monkeypatch.setitem(sys.modules, 'flask_app', fake_flask_app)
    job = Mock(id='parent-1_s0_batch_0', meta={})
    monkeypatch.setattr(clustering.taskqueue, 'current_task_id', lambda: job.id)
    monkeypatch.setattr(
        clustering, 'get_task_info_from_db', lambda _task_id: {'status': 'RUNNING'}
    )
    monkeypatch.setattr(task_run, '_read_task_statuses', lambda _conn, _ids: {})
    monkeypatch.setattr(clustering, 'main_task_start_lock', nullcontext)
    writes = []
    monkeypatch.setattr(
        clustering, 'save_task_status', lambda *a, **k: writes.append((a, k))
    )
    monkeypatch.setattr(
        task_run, 'save_task_status', lambda *a, **k: writes.append((a, k))
    )

    result = _revoked_result(
        clustering.run_clustering_batch_task,
        batch_id_str='Batch_0',
        start_run_idx=0,
        num_iterations_in_batch=1,
        genre_to_lightweight_track_data_map_json='{}',
        target_songs_per_genre=1,
        sampling_percentage_change_per_run=0.1,
        clustering_method='kmeans',
        active_mood_labels_for_batch=[],
        num_clusters_min_max_tuple=(2, 3),
        dbscan_params_ranges_dict={},
        gmm_params_ranges_dict={},
        spectral_params_ranges_dict={},
        pca_params_ranges_dict={'components_min': 2, 'components_max': 3},
        max_songs_per_cluster=50,
        parent_task_id='parent-1',
        score_weights_dict={},
        elite_solutions_params_list_json='[]',
        exploitation_probability=0.0,
        mutation_config_json='{}',
        initial_subset_track_ids_json='[]',
        enable_clustering_embeddings_param=True,
    )

    assert result['status'] == 'REVOKED'
    assert writes == [], (
        'the cancel wiped both rows between the claim and the first write; the '
        'shared cancel check runs BEFORE the opening write, so nothing recreates '
        'a child row for a parent that is gone'
    )


def _batch_kwargs():
    return dict(
        batch_id_str='Batch_0',
        start_run_idx=0,
        num_iterations_in_batch=1,
        genre_to_lightweight_track_data_map_json='{}',
        target_songs_per_genre=1,
        sampling_percentage_change_per_run=0.1,
        clustering_method='kmeans',
        active_mood_labels_for_batch=[],
        num_clusters_min_max_tuple=(2, 3),
        dbscan_params_ranges_dict={},
        gmm_params_ranges_dict={},
        spectral_params_ranges_dict={},
        pca_params_ranges_dict={'components_min': 2, 'components_max': 3},
        max_songs_per_cluster=50,
        parent_task_id='parent-1',
        score_weights_dict={},
        elite_solutions_params_list_json='[]',
        exploitation_probability=0.0,
        mutation_config_json='{}',
        initial_subset_track_ids_json='[]',
        enable_clustering_embeddings_param=True,
    )


def _live_batch_harness(monkeypatch):
    import sys
    import types
    from contextlib import nullcontext
    from flask import Flask
    import tasks.clustering as clustering
    import tasks.task_run as task_run

    fake_flask_app = types.ModuleType('flask_app')
    fake_flask_app.app = Flask('live-batch')
    monkeypatch.setitem(sys.modules, 'flask_app', fake_flask_app)
    monkeypatch.setattr(clustering.taskqueue, 'current_task_id', lambda: 'parent-1_s0_batch_0')
    monkeypatch.setattr(
        clustering, 'get_task_info_from_db', lambda _task_id: {'status': 'RUNNING'}
    )
    monkeypatch.setattr(task_run, '_open_check_connection', lambda: object())
    monkeypatch.setattr(
        task_run, '_read_task_statuses',
        lambda _conn, ids: {task_id: 'RUNNING' for task_id in ids},
    )
    monkeypatch.setattr(clustering, 'main_task_start_lock', nullcontext)
    monkeypatch.setattr(clustering, 'row_heartbeat', lambda *a, **k: nullcontext())
    return clustering, task_run


def test_a_batch_whose_opening_write_fails_raises_that_error_not_an_unbound_name(monkeypatch):
    import pytest

    clustering, task_run = _live_batch_harness(monkeypatch)

    def _db_gone(*_a, **_k):
        raise RuntimeError('db gone')

    monkeypatch.setattr(task_run, 'save_task_status', _db_gone)

    kwargs = _batch_kwargs()

    with pytest.raises(RuntimeError, match='db gone'):
        clustering.run_clustering_batch_task(**kwargs)


def test_a_batch_returns_its_best_result_so_the_queue_row_carries_it(monkeypatch):
    clustering, task_run = _live_batch_harness(monkeypatch)
    writes = []
    monkeypatch.setattr(
        task_run, 'save_task_status', lambda *a, **k: writes.append(k) or True
    )
    monkeypatch.setattr(
        clustering, '_get_stratified_song_subset', lambda *a, **k: [{'item_id': 'song-1'}]
    )
    best = {'fitness_score': 1.5, 'parameters': {'method': 'kmeans'}, 'named_playlists': {}}
    monkeypatch.setattr(
        clustering, '_perform_single_clustering_iteration', lambda **k: dict(best)
    )

    result = clustering.run_clustering_batch_task(**_batch_kwargs())

    assert result['status'] == 'SUCCESS'
    assert result['iterations_completed_in_batch'] == 1
    assert result['full_best_result_from_batch']['fitness_score'] == 1.5
    assert result['final_subset_track_ids'] == ['song-1']
    assert all(
        'full_best_result_from_batch' not in (k.get('details') or {}) for k in writes
    ), (
        'the winning result rides the dict the batch RETURNS, which the queue writes '
        'on the terminal row with a retry on a fresh connection; a progress write '
        'whose return value nobody checks is not a carrier for the whole batch'
    )
