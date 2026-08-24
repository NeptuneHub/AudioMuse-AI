# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the Poincare-ball projection math and distance function.

Verifies the projection stays inside the open unit ball, the radius equals the
projected norm, the exact hyperbolic distance matches the analytic values on a
radial diameter, scale calibration resists saturation, and quantile radial
bands partition the radius distribution without gaps or overlaps.

Main Features:
* Projected vectors stay strictly inside the unit ball (||proj|| < 1)
* Radius equals ||proj(x)|| and saturates toward 1 as ||x|| grows past 3*s
* d_H(u,u) == 0, symmetry, and d_H(origin, r) == 2*arctanh(r) on a diameter
* Vectorized distances match the scalar distance per pair
* calibrate_scale returns the requested norm percentile and a sane fallback
* split_radial_bands/assign_radial_bands cover [min, max] with no empty band
* karcher_mean is a real Frechet minimiser: its cost never rises when it is
  given more iterations and no nudge off it lowers the cost, which is what an
  undamped unit Riemannian step fails once the members span more than a couple
  of units of hyperbolic distance
* einstein_midpoint agrees with the geodesic midpoint for a pair of points
* hyperbolic_distance_matrix returns the same matrix whether or not the caller
  hands it precomputed squared norms
* poincare_kmeans seeds k distinct centres, still fills k when every point
  coincides, and its k-means++ only ever measures against the centre it just
  added, so seeding stays linear in the catalogue instead of rebuilding the
  whole points-by-centres matrix on every pick
* nearest_centroid agrees with a full hyperbolic_distance_matrix argmin while
  skipping the arccosh, which is what keeps a full-catalogue assignment pass
  affordable, clips and upcasts one CHUNK at a time so its working set does not
  grow with the catalogue, and gives the same labels for float32 or float64 in
"""

import numpy as np
import pytest

from tasks.hyperbolic_geometry import (
    _BALL_LIMIT,
    _kmeans_plus_plus,
    assign_radial_bands,
    clip_into_ball,
    calibrate_scale,
    einstein_midpoint,
    hyperbolic_distance,
    hyperbolic_distance_matrix,
    hyperbolic_distances_to,
    karcher_mean,
    nearest_centroid,
    poincare_geodesic,
    poincare_kmeans,
    poincare_radius,
    project_to_poincare,
    split_radial_bands,
)


def test_projected_vector_stays_inside_unit_ball():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 8)).astype(np.float32)
    proj = project_to_poincare(x, scale=2.0)
    norms = np.linalg.norm(proj, axis=1)
    assert np.all(norms < 1.0)
    assert proj.dtype == np.float32


def test_projected_radius_equals_tanh_norm_over_scale():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((50, 4)).astype(np.float32)
    scale = 1.7
    proj = project_to_poincare(x, scale)
    radii = poincare_radius(x, scale)
    expected = np.tanh(np.linalg.norm(x, axis=1) / scale)
    np.testing.assert_allclose(np.linalg.norm(proj, axis=1), expected, rtol=1e-5)
    np.testing.assert_allclose(radii, expected, rtol=1e-5)


def test_projected_direction_matches_unit_vector():
    x = np.array([3.0, -4.0, 0.5], dtype=np.float32)
    proj = project_to_poincare(x, scale=2.0)
    unit = x / np.linalg.norm(x)
    assert np.allclose(proj / np.linalg.norm(proj), unit, atol=1e-6)


def test_large_norm_saturates_at_the_ball_limit_not_at_raw_tanh():
    """A saturating norm is capped at _BALL_LIMIT, not at tanh's own value.

    tanh(10) is 0.999999996, which is finer than float32 can represent once a
    200-dim norm is re-measured: 1 - ||u||^2 rounds to 0 and the Poincare
    denominator collapses. The clip is what prevents that, so the radius must
    land on the limit rather than on the mathematical tanh.
    """
    x = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    radius = float(poincare_radius(x, scale=1.0))
    assert radius == pytest.approx(_BALL_LIMIT, abs=1e-7)
    assert radius < 1.0
    assert radius > 0.999
    assert 1.0 - radius ** 2 > 1e-6


def test_zero_vector_projects_to_zero():
    x = np.zeros(5, dtype=np.float32)
    proj = project_to_poincare(x, scale=1.0)
    np.testing.assert_allclose(proj, 0.0, atol=1e-7)
    assert poincare_radius(x, scale=1.0) == 0.0


def test_distance_to_self_is_zero():
    x = np.array([0.3, -0.2, 0.1], dtype=np.float64)
    assert hyperbolic_distance(x, x) == pytest.approx(0.0, abs=1e-6)


def test_distance_is_symmetric():
    u = np.array([0.2, 0.1, -0.3], dtype=np.float64)
    v = np.array([-0.4, 0.5, 0.2], dtype=np.float64)
    assert hyperbolic_distance(u, v) == pytest.approx(hyperbolic_distance(v, u), abs=1e-6)


def test_distance_from_origin_equals_two_arctanh_r():
    r = 0.5
    origin = np.zeros(3, dtype=np.float64)
    point = np.array([r, 0.0, 0.0], dtype=np.float64)
    expected = 2.0 * np.arctanh(r)
    assert hyperbolic_distance(origin, point) == pytest.approx(float(expected), rel=1e-6)


def test_distance_grows_monotonically_along_diameter():
    origin = np.zeros(3, dtype=np.float64)
    d0 = hyperbolic_distance(origin, np.array([0.2, 0.0, 0.0]))
    d1 = hyperbolic_distance(origin, np.array([0.6, 0.0, 0.0]))
    d2 = hyperbolic_distance(origin, np.array([0.9, 0.0, 0.0]))
    assert d0 < d1 < d2


def test_vectorized_distances_match_scalar():
    rng = np.random.default_rng(3)
    target = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float64)
    cand = rng.uniform(-0.5, 0.5, (40, 4)).astype(np.float64)
    cand = cand / np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1.0) * 0.8
    vec = hyperbolic_distances_to(target, cand)
    scalar = np.array([hyperbolic_distance(target, c) for c in cand])
    np.testing.assert_allclose(vec, scalar, rtol=1e-5, atol=1e-6)


def test_calibrate_scale_returns_norm_percentile():
    norms = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float64)
    scale = calibrate_scale(norms, percentile=80.0)
    assert scale == pytest.approx(float(np.percentile(norms, 80.0)), rel=1e-9)


def test_calibrate_scale_fallbacks():
    assert calibrate_scale(np.array([], dtype=np.float64), percentile=95.0) == 1.0
    assert calibrate_scale(np.array([0.0, 0.0], dtype=np.float64), percentile=95.0) == 1.0


def test_radial_bands_partition_distribution():
    rng = np.random.default_rng(4)
    radii = np.tanh(rng.exponential(1.5, size=500))
    boundaries = split_radial_bands(radii, n_bands=3)
    assert len(boundaries) >= 1
    assert boundaries[0][0] == pytest.approx(float(radii.min()), abs=1e-6)
    assert boundaries[-1][1] == pytest.approx(float(radii.max()), abs=1e-6)
    assign = assign_radial_bands(radii, boundaries)
    assert assign.shape == radii.shape
    for band_index in range(len(boundaries)):
        lo, hi = boundaries[band_index]
        members = radii[assign == band_index]
        if members.size:
            assert members.min() >= lo - 1e-6
            assert members.max() <= hi + 1e-6


def test_radial_bands_with_ties_have_no_empty_bands():
    radii = np.array([0.5] * 100, dtype=np.float64)
    boundaries = split_radial_bands(radii, n_bands=3)
    assert len(boundaries) == 1
    assign = assign_radial_bands(radii, boundaries)
    assert set(assign.tolist()) == {0}


def test_radial_bands_empty_input():
    assert split_radial_bands(np.array([]), n_bands=3) == []


def test_karcher_mean_is_the_geodesic_midpoint_not_the_euclidean_mean():
    u = np.array([0.9, 0.0], dtype=np.float64)
    v = np.array([-0.5, 0.0], dtype=np.float64)
    mean = karcher_mean(np.stack([u, v]))
    assert mean is not None
    assert abs(mean[1]) < 1e-6
    assert mean[0] == pytest.approx(0.4313, abs=1e-2)


def _cloud_spanning_the_ball(seed=5, count=60, dim=8, rmax=0.95):
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((count, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rmax * rng.random((count, 1)) ** (1.0 / dim)
    return directions * radii


def _frechet_cost(centre, points):
    return float(np.sum(hyperbolic_distances_to(centre, points) ** 2))


def test_karcher_mean_cost_never_rises_when_given_more_iterations():
    points = _cloud_spanning_the_ball()
    costs = [
        _frechet_cost(karcher_mean(points, iterations=n), points)
        for n in (0, 1, 2, 4, 8, 16, 32)
    ]
    for earlier, later in zip(costs, costs[1:]):
        assert later <= earlier + 1e-4


def test_karcher_mean_is_not_beaten_by_a_nudge_off_it():
    points = _cloud_spanning_the_ball()
    mean = karcher_mean(points, iterations=32)
    base = _frechet_cost(mean, points)
    rng = np.random.default_rng(9)
    for _ in range(200):
        nudged = mean + rng.standard_normal(mean.shape) * 0.01
        if float(np.linalg.norm(nudged)) >= 1.0:
            continue
        assert _frechet_cost(nudged, points) >= base - 1e-6


def test_karcher_mean_stays_inside_the_hull_of_a_widely_spread_cloud():
    points = _cloud_spanning_the_ball()
    mean = karcher_mean(points, iterations=32)
    assert float(np.linalg.norm(mean)) < float(np.linalg.norm(points, axis=1).min())


def test_einstein_midpoint_of_two_points_is_the_geodesic_midpoint():
    u = np.array([0.9, 0.0, 0.0], dtype=np.float64)
    v = np.array([-0.1, 0.6, 0.0], dtype=np.float64)
    midpoint = poincare_geodesic(u, v, [0.5])[0]
    np.testing.assert_allclose(einstein_midpoint(np.stack([u, v])), midpoint, atol=1e-6)


def test_einstein_midpoint_of_one_point_is_that_point():
    point = np.array([0.3, -0.4, 0.1], dtype=np.float64)
    np.testing.assert_allclose(einstein_midpoint(point), point, atol=1e-6)


def test_einstein_midpoint_of_nothing_is_none():
    assert einstein_midpoint(np.zeros((0, 4))) is None


def test_distance_matrix_with_cached_norms_matches_the_uncached_matrix():
    rng = np.random.default_rng(17)
    targets = rng.uniform(-0.6, 0.6, (25, 6))
    candidates = rng.uniform(-0.6, 0.6, (7, 6))
    np.testing.assert_allclose(
        hyperbolic_distance_matrix(
            targets,
            candidates,
            target_norms2=np.sum(targets * targets, axis=1),
            candidate_norms2=np.sum(candidates * candidates, axis=1),
        ),
        hyperbolic_distance_matrix(targets, candidates),
        rtol=1e-5,
    )


def test_kmeans_plus_plus_seeds_k_distinct_points():
    rng = np.random.default_rng(2)
    pts = rng.uniform(-0.5, 0.5, (200, 6))
    chosen = _kmeans_plus_plus(pts, np.sum(pts * pts, axis=1), 12, np.random.RandomState(0))
    assert len(chosen) == 12
    assert len(set(chosen)) == 12


def test_kmeans_plus_plus_still_fills_k_when_every_point_coincides():
    pts = np.tile(np.array([0.3, -0.2, 0.1]), (40, 1))
    chosen = _kmeans_plus_plus(pts, np.sum(pts * pts, axis=1), 9, np.random.RandomState(0))
    assert len(chosen) == 9
    assert len(set(chosen)) == 9


def test_kmeans_plus_plus_only_measures_against_the_centre_it_just_added(monkeypatch):
    import tasks.hyperbolic_geometry as geometry

    widths = []
    real = geometry.hyperbolic_distance_matrix

    def counting(targets, candidates, **kwargs):
        widths.append(np.atleast_2d(np.asarray(candidates)).shape[0])
        return real(targets, candidates, **kwargs)

    monkeypatch.setattr(geometry, "hyperbolic_distance_matrix", counting)
    rng = np.random.default_rng(4)
    pts = rng.uniform(-0.5, 0.5, (300, 5))
    geometry._kmeans_plus_plus(pts, np.sum(pts * pts, axis=1), 20, np.random.RandomState(0))
    assert widths == [1] * 20


def test_nearest_centroid_matches_a_full_distance_matrix_argmin():
    rng = np.random.default_rng(11)
    pts = clip_into_ball(rng.uniform(-0.3, 0.3, (400, 12)))
    centroids = clip_into_ball(rng.uniform(-0.3, 0.3, (17, 12)))
    expected = np.argmin(hyperbolic_distance_matrix(pts, centroids), axis=1)
    np.testing.assert_array_equal(nearest_centroid(pts, centroids), expected)


def test_poincare_kmeans_recovers_separated_blobs():
    rng = np.random.default_rng(6)
    centres = rng.standard_normal((5, 12))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    centres *= 0.6
    truth = np.repeat(np.arange(5), 40)
    pts = centres[truth] + rng.standard_normal((200, 12)) * 0.005
    centroids, labels = poincare_kmeans(pts, 5, iterations=10)
    assert centroids.shape == (5, 12)
    for blob in range(5):
        assert len(set(labels[truth == blob])) == 1
    assert len(set(labels)) == 5


def _assignment_peak(n, dim, seed):
    import tracemalloc

    rng = np.random.default_rng(seed)
    points = clip_into_ball(rng.uniform(-0.3, 0.3, (n, dim))).astype(np.float32)
    centroids = clip_into_ball(rng.uniform(-0.3, 0.3, (32, dim)))
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    nearest_centroid(points, centroids)
    spike = tracemalloc.get_traced_memory()[1] - before
    tracemalloc.stop()
    return spike


def test_nearest_centroid_working_set_does_not_grow_with_the_catalogue():
    small = _assignment_peak(20_000, 64, 21)
    large = _assignment_peak(160_000, 64, 21)
    assert large < small * 1.5


def test_nearest_centroid_gives_the_same_labels_for_float32_and_float64():
    rng = np.random.default_rng(22)
    points = clip_into_ball(rng.uniform(-0.4, 0.4, (500, 24)))
    centroids = clip_into_ball(rng.uniform(-0.4, 0.4, (13, 24)))
    np.testing.assert_array_equal(
        nearest_centroid(points.astype(np.float32), centroids),
        nearest_centroid(points, centroids),
    )


def test_nearest_centroid_still_clips_points_that_sit_outside_the_ball():
    centroids = np.array([[0.5, 0.0], [-0.5, 0.0]], dtype=np.float64)
    outside = np.array([[9.0, 0.0], [-9.0, 0.0]], dtype=np.float64)
    np.testing.assert_array_equal(nearest_centroid(outside, centroids), [0, 1])
