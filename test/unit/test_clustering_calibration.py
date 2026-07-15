# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Per-server cluster calibration: parameters decided by code for every algorithm.

Drives _calibrate_cluster_params with the single-iteration probe faked,
asserting the up-to-three-tries rules that avoid all-tiny (under
MIN_PLAYLIST_SIZE_FOR_TOP_N) playlists and soft-cap oversized (over
CLUSTERING_MAX_PLAYLIST_SONGS) ones across kmeans, gmm, spectral and dbscan.

Main Features:
* KMeans, GMM and Spectral tune their own cluster/component range the same
  way: small libraries pin directly to top_n_playlists clusters (never below
  the oversize floor subset/CLUSTERING_MAX_PLAYLIST_SONGS) and the probe runs
  at the TOP of the range
* DBSCAN has no cluster count: eps is derived from the data (k-distance
  heuristic), then probes widen it when playlists are tiny and tighten it
  when oversized, alongside the percentile
* A probe passes only with at least top_n_playlists keeper playlists;
  otherwise clusters shrink toward the goal and the percentile rises
* Oversized playlists grow the cluster range and lower the percentile
* With no passing probe the best one wins and big always beats empty
* A probe failure still caps count-based ranges by library size
"""


import numpy as np
import pytest


def _result(sizes):
    return {'named_playlists': {f'P{i}': list(range(s)) for i, s in enumerate(sizes)}}


def _run_calibration(monkeypatch, probe_results, method='kmeans', num_min=40, num_max=100,
                     pct=50, subset_size=8000, top_n=8, genre_map=None,
                     derived_eps=(5.0, 9.0)):
    from tasks import clustering

    probes = []
    percentiles = []

    monkeypatch.setattr(
        clustering, '_calculate_target_songs_per_genre',
        lambda genre_map, percentile, min_songs: percentiles.append(percentile) or 10,
    )
    monkeypatch.setattr(
        clustering, '_get_stratified_song_subset',
        lambda genre_map, target: [{'item_id': str(i)} for i in range(subset_size)],
    )
    monkeypatch.setattr(
        clustering, '_derive_dbscan_eps',
        lambda item_ids, min_samples, moods, embeddings: derived_eps,
    )

    def fake_iteration(**kwargs):
        probes.append(kwargs)
        result = probe_results[len(probes) - 1]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(clustering, '_perform_single_clustering_iteration', fake_iteration)

    chosen = clustering._calibrate_cluster_params(
        method,
        genre_map if genre_map is not None else {'rock': [{'item_id': 'i1'}]},
        num_min,
        num_max,
        pct,
        100,
        0.1,
        0.5,
        5,
        20,
        0,
        199,
        0,
        top_n,
        5,
        True,
        lambda message, local_pct, task_state=None: None,
    )
    return chosen, probes, percentiles


class TestKmeansCalibration:
    def test_a_probe_with_enough_keeper_playlists_keeps_the_configured_defaults(self, monkeypatch):
        chosen, probes, _pcts = _run_calibration(monkeypatch, [_result([50] * 10)])
        assert chosen == (40, 100, 50)
        assert [p['num_clusters_min_max'] for p in probes] == [(100, 100)]

    def test_a_small_library_asks_directly_for_top_n_playlists(self, monkeypatch):
        chosen, probes, _pcts = _run_calibration(
            monkeypatch, [_result([40] * 8)], subset_size=800
        )
        assert chosen == (8, 8, 50)
        assert [p['num_clusters_min_max'] for p in probes] == [(8, 8)]

    def test_the_pinned_range_never_drops_below_the_oversize_floor(self, monkeypatch):
        chosen, probes, _pcts = _run_calibration(
            monkeypatch, [_result([150] * 12)], subset_size=2500
        )
        assert chosen == (12, 12, 50)
        assert [p['num_clusters_min_max'] for p in probes] == [(12, 12)]

    def test_too_few_keepers_shrink_clusters_and_raise_the_percentile_until_enough(self, monkeypatch):
        chosen, probes, pcts = _run_calibration(
            monkeypatch,
            [_result([5, 8]), _result([30] * 3), _result([30] * 9)],
        )
        assert chosen == (25, 25, 80)
        assert [p['num_clusters_min_max'] for p in probes] == [(100, 100), (50, 50), (25, 25)]
        assert pcts == [50, 65, 80]

    def test_oversized_playlists_grow_clusters_and_lower_the_percentile(self, monkeypatch):
        chosen, probes, pcts = _run_calibration(
            monkeypatch, [_result([250] * 9), _result([100] * 9)]
        )
        assert chosen == (60, 150, 35)
        assert [p['num_clusters_min_max'] for p in probes] == [(100, 100), (150, 150)]
        assert pcts == [50, 35]

    def test_with_no_passing_probe_big_playlists_beat_empty_results(self, monkeypatch):
        chosen, probes, _pcts = _run_calibration(
            monkeypatch, [_result([5]), _result([400]), _result([3])]
        )
        assert chosen == (40, 50, 65)
        assert len(probes) == 3

    def test_a_probe_failure_still_caps_the_range_by_library_size(self, monkeypatch):
        chosen, _probes, _pcts = _run_calibration(
            monkeypatch,
            [RuntimeError('probe blew up')],
            genre_map={'rock': [{'item_id': str(i)} for i in range(120)]},
        )
        assert chosen == (3, 3, 50)

    def test_a_probe_failure_on_a_big_library_keeps_the_configured_defaults(self, monkeypatch):
        chosen, _probes, _pcts = _run_calibration(
            monkeypatch,
            [RuntimeError('probe blew up')],
            genre_map={'rock': [{'item_id': str(i)} for i in range(10000)]},
        )
        assert chosen == (40, 100, 50)


class TestGmmAndSpectralCalibration:
    def test_gmm_pins_its_component_range_exactly_like_kmeans(self, monkeypatch):
        chosen, probes, _pcts = _run_calibration(
            monkeypatch, [_result([40] * 8)], method='gmm', subset_size=800
        )
        assert chosen == (8, 8, 50)
        assert probes[0]['clustering_method'] == 'gmm'
        assert probes[0]['gmm_params_ranges'] == {'n_components_min': 8, 'n_components_max': 8}
        assert probes[0]['num_clusters_min_max'] == (2, 2)

    def test_spectral_pins_its_cluster_range_exactly_like_kmeans(self, monkeypatch):
        chosen, probes, _pcts = _run_calibration(
            monkeypatch, [_result([40] * 8)], method='spectral', subset_size=800
        )
        assert chosen == (8, 8, 50)
        assert probes[0]['clustering_method'] == 'spectral'
        assert probes[0]['spectral_params_ranges'] == {'n_clusters_min': 8, 'n_clusters_max': 8}

    def test_gmm_too_few_keepers_shrink_components_and_raise_the_percentile(self, monkeypatch):
        chosen, probes, pcts = _run_calibration(
            monkeypatch,
            [_result([5, 8]), _result([30] * 9)],
            method='gmm',
        )
        assert chosen == (40, 50, 65)
        assert probes[1]['gmm_params_ranges'] == {'n_components_min': 50, 'n_components_max': 50}
        assert pcts == [50, 65]


class TestDbscanCalibration:
    def test_dbscan_derives_eps_from_the_data_instead_of_the_configured_range(self, monkeypatch):
        chosen, probes, _pcts = _run_calibration(
            monkeypatch,
            [_result([30] * 9)],
            method='dbscan',
            num_min=0.1,
            num_max=0.5,
        )
        assert chosen == (5.0, 9.0, 50)
        assert probes[0]['clustering_method'] == 'dbscan'
        assert probes[0]['dbscan_params_ranges'] == {
            'eps_min': 5.0, 'eps_max': 9.0, 'samples_min': 5, 'samples_max': 20,
        }

    def test_dbscan_too_few_keepers_widen_eps_and_raise_the_percentile(self, monkeypatch):
        chosen, probes, pcts = _run_calibration(
            monkeypatch,
            [_result([5, 8]), _result([30] * 9)],
            method='dbscan',
            num_min=0.1,
            num_max=0.5,
        )
        assert chosen == (7.5, 13.5, 65)
        assert probes[1]['dbscan_params_ranges']['eps_min'] == 7.5
        assert probes[1]['dbscan_params_ranges']['eps_max'] == 13.5
        assert pcts == [50, 65]

    def test_dbscan_oversized_playlists_tighten_eps_and_lower_the_percentile(self, monkeypatch):
        chosen, _probes, pcts = _run_calibration(
            monkeypatch,
            [_result([250] * 9), _result([100] * 9)],
            method='dbscan',
            num_min=0.1,
            num_max=0.5,
        )
        assert chosen == (pytest.approx(3.5), pytest.approx(6.3), 35)
        assert pcts == [50, 35]

    def test_dbscan_derivation_failure_keeps_the_configured_eps(self, monkeypatch):
        chosen, probes, _pcts = _run_calibration(
            monkeypatch,
            [_result([30] * 9)],
            method='dbscan',
            num_min=0.1,
            num_max=0.5,
            derived_eps=None,
        )
        assert chosen == (0.1, 0.5, 50)
        assert probes[0]['dbscan_params_ranges']['eps_min'] == 0.1

    def test_dbscan_eps_widening_is_capped_to_bound_memory(self, monkeypatch):
        chosen, probes, _pcts = _run_calibration(
            monkeypatch,
            [_result([5]), _result([5]), _result([30] * 9)],
            method='dbscan',
            num_min=0.1,
            num_max=0.5,
        )
        assert probes[1]['dbscan_params_ranges']['eps_max'] == 13.5
        assert probes[2]['dbscan_params_ranges']['eps_max'] == 13.5
        assert probes[2]['dbscan_params_ranges']['eps_min'] == pytest.approx(11.25)
        assert chosen == (pytest.approx(11.25), 13.5, 80)

    def test_a_single_giant_cluster_tightens_eps_instead_of_widening(self, monkeypatch):
        chosen, _probes, pcts = _run_calibration(
            monkeypatch,
            [_result([400]), _result([30] * 9)],
            method='dbscan',
            num_min=0.1,
            num_max=0.5,
        )
        assert chosen == (pytest.approx(3.5), pytest.approx(6.3), 35)
        assert pcts == [50, 35]

    def test_a_dbscan_probe_failure_falls_back_to_the_derived_eps(self, monkeypatch):
        chosen, _probes, _pcts = _run_calibration(
            monkeypatch,
            [RuntimeError('probe blew up')],
            method='dbscan',
            num_min=0.1,
            num_max=0.5,
        )
        assert chosen == (5.0, 9.0, 50)

    def test_a_dbscan_probe_failure_without_derivation_keeps_the_configured_values(self, monkeypatch):
        chosen, _probes, _pcts = _run_calibration(
            monkeypatch,
            [RuntimeError('probe blew up')],
            method='dbscan',
            num_min=0.1,
            num_max=0.5,
            derived_eps=None,
        )
        assert chosen == (0.1, 0.5, 50)

    def test_derive_dbscan_eps_scales_with_the_actual_neighbor_distances(self, monkeypatch):
        from tasks import clustering

        rng = np.random.default_rng(7)
        embeddings = rng.standard_normal((80, 50)).astype(np.float32)
        tracks = [{'item_id': str(i)} for i in range(80)]
        monkeypatch.setattr(
            clustering, '_prepare_iteration_data',
            lambda ids, moods, emb, prefix, run: (tracks, embeddings, embeddings),
        )
        derived = clustering._derive_dbscan_eps(
            [t['item_id'] for t in tracks], 10, ['rock'], True
        )
        assert derived is not None
        eps_min, eps_max = derived
        assert 0 < eps_min < eps_max
        assert eps_min > 1.0
