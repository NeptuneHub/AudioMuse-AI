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
"""

import numpy as np
import pytest

from tasks.hyperbolic_geometry import (
    assign_radial_bands,
    calibrate_scale,
    hyperbolic_distance,
    hyperbolic_distances_to,
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


def test_large_norm_saturates_but_stays_in_ball():
    x = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    radius = float(poincare_radius(x, scale=1.0))
    assert radius == pytest.approx(np.tanh(10.0), abs=1e-6)
    assert radius < 1.0
    assert radius > 0.999


def test_zero_vector_projects_to_zero():
    x = np.zeros(5, dtype=np.float32)
    proj = project_to_poincare(x, scale=1.0)
    np.testing.assert_allclose(proj, 0.0, atol=1e-7)
    assert poincare_radius(x, scale=1.0) == 0.0


def test_distance_to_self_is_zero():
    x = np.array([0.3, -0.2, 0.1], dtype=np.float64)
    assert hyperbolic_distance(x, x) == pytest.approx(0.0, abs=1e-9)


def test_distance_is_symmetric():
    u = np.array([0.2, 0.1, -0.3], dtype=np.float64)
    v = np.array([-0.4, 0.5, 0.2], dtype=np.float64)
    assert hyperbolic_distance(u, v) == pytest.approx(hyperbolic_distance(v, u), abs=1e-9)


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
    np.testing.assert_allclose(vec, scalar, rtol=1e-9)


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
    assert boundaries[0][0] == pytest.approx(float(radii.min()), abs=1e-9)
    assert boundaries[-1][1] == pytest.approx(float(radii.max()), abs=1e-9)
    assign = assign_radial_bands(radii, boundaries)
    assert assign.shape == radii.shape
    for band_index in range(len(boundaries)):
        lo, hi = boundaries[band_index]
        members = radii[assign == band_index]
        if members.size:
            assert members.min() >= lo - 1e-9
            assert members.max() <= hi + 1e-9


def test_radial_bands_with_ties_have_no_empty_bands():
    radii = np.array([0.5] * 100, dtype=np.float64)
    boundaries = split_radial_bands(radii, n_bands=3)
    assert len(boundaries) == 1
    assign = assign_radial_bands(radii, boundaries)
    assert set(assign.tolist()) == {0}


def test_radial_bands_empty_input():
    assert split_radial_bands(np.array([]), n_bands=3) == []
