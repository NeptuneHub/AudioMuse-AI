# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Distance metrics, centroid interpolation and signature normalization.

Covers the path_manager math used to walk between embedding centroids: the
euclidean/angular distance functions, linear and SLERP interpolation, and
signature normalization.

Main Features:
* Euclidean and angular distances on known vectors, with None yielding inf
* get_distance selects euclidean by default and angular when configured
* Linear interpolation endpoints/midpoints and SLERP arc on the unit sphere
* SLERP falls back to linear for collinear/zero vectors; signatures normalized
"""

import numpy as np
from tasks.path_manager import (
    get_euclidean_distance,
    get_angular_distance,
    get_distance,
    interpolate_centroids,
    _normalize_signature,
)


class TestEuclideanDistance:
    def test_identical_vectors_return_zero(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.0, 2.0, 3.0])

        dist = get_euclidean_distance(v1, v2)

        assert dist == 0.0

    def test_known_distance_3_4_5_triangle(self):
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([3.0, 4.0, 0.0])

        dist = get_euclidean_distance(v1, v2)

        assert abs(dist - 5.0) < 1e-10

    def test_none_vector_returns_inf(self):
        v1 = np.array([1.0, 2.0])

        assert get_euclidean_distance(None, v1) == float('inf')
        assert get_euclidean_distance(v1, None) == float('inf')
        assert get_euclidean_distance(None, None) == float('inf')

    def test_unit_vectors_on_axes(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        dist = get_euclidean_distance(v1, v2)

        assert abs(dist - np.sqrt(2)) < 1e-10


class TestAngularDistance:
    def test_identical_vectors_return_zero(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.0, 2.0, 3.0])

        dist = get_angular_distance(v1, v2)

        assert abs(dist) < 1e-10

    def test_orthogonal_vectors_return_half(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        dist = get_angular_distance(v1, v2)

        assert abs(dist - 0.5) < 1e-10

    def test_opposite_vectors_return_one(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])

        dist = get_angular_distance(v1, v2)

        assert abs(dist - 1.0) < 1e-10

    def test_scaled_vectors_same_direction(self):
        v1 = np.array([1.0, 1.0])
        v2 = np.array([2.0, 2.0])

        dist = get_angular_distance(v1, v2)

        assert abs(dist) < 1e-7

    def test_zero_vector_returns_inf(self):
        v1 = np.array([1.0, 2.0])
        v_zero = np.array([0.0, 0.0])

        assert get_angular_distance(v_zero, v1) == float('inf')
        assert get_angular_distance(v1, v_zero) == float('inf')

    def test_none_vector_returns_inf(self):
        v1 = np.array([1.0, 2.0])

        assert get_angular_distance(None, v1) == float('inf')
        assert get_angular_distance(v1, None) == float('inf')


class TestGetDistance:
    def test_uses_euclidean_by_default(self, monkeypatch):
        import tasks.path_manager

        monkeypatch.setattr(tasks.path_manager, 'PATH_DISTANCE_METRIC', 'euclidean')

        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])

        dist = get_distance(v1, v2)

        assert abs(dist - 5.0) < 1e-10

    def test_uses_angular_when_configured(self, monkeypatch):
        import tasks.path_manager

        monkeypatch.setattr(tasks.path_manager, 'PATH_DISTANCE_METRIC', 'angular')

        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])

        dist = get_distance(v1, v2)

        assert abs(dist - 0.5) < 1e-10


class TestInterpolateCentroidsLinear:
    def test_linear_interpolation_endpoints(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([10.0, 0.0])

        centroids = interpolate_centroids(v1, v2, num=5, metric="euclidean")

        assert centroids.shape == (5, 2)
        np.testing.assert_array_almost_equal(centroids[0], v1)
        np.testing.assert_array_almost_equal(centroids[-1], v2)

    def test_linear_interpolation_midpoint(self):
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([10.0, 20.0, 30.0])

        centroids = interpolate_centroids(v1, v2, num=3, metric="euclidean")

        expected_midpoint = np.array([5.0, 10.0, 15.0])
        np.testing.assert_array_almost_equal(centroids[1], expected_midpoint)

    def test_linear_interpolation_evenly_spaced(self):
        v1 = np.array([0.0])
        v2 = np.array([4.0])

        centroids = interpolate_centroids(v1, v2, num=5, metric="euclidean")

        expected = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        np.testing.assert_array_almost_equal(centroids, expected)

    def test_linear_single_dimension(self):
        v1 = np.array([0.0])
        v2 = np.array([1.0])

        centroids = interpolate_centroids(v1, v2, num=11, metric="euclidean")

        assert len(centroids) == 11
        assert abs(centroids[0][0] - 0.0) < 1e-10
        assert abs(centroids[-1][0] - 1.0) < 1e-10


class TestInterpolateCentroidsSLERP:
    def test_slerp_endpoints(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        centroids = interpolate_centroids(v1, v2, num=5, metric="angular")

        assert centroids.shape == (5, 3)
        np.testing.assert_array_almost_equal(centroids[0], v1)
        np.testing.assert_array_almost_equal(centroids[-1], v2)

    def test_slerp_preserves_magnitude_variation(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 2.0, 0.0])

        centroids = interpolate_centroids(v1, v2, num=3, metric="angular")

        midpoint_magnitude = np.linalg.norm(centroids[1])
        assert abs(midpoint_magnitude - 1.5) < 1e-10

    def test_slerp_arc_on_unit_sphere(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        centroids = interpolate_centroids(v1, v2, num=5, metric="angular")

        for i, centroid in enumerate(centroids):
            magnitude = np.linalg.norm(centroid)
            assert abs(magnitude - 1.0) < 1e-10, f"Point {i} has magnitude {magnitude}"

    def test_slerp_midpoint_45_degrees(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])

        centroids = interpolate_centroids(v1, v2, num=3, metric="angular")

        expected_direction = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2])
        actual_direction = centroids[1] / np.linalg.norm(centroids[1])
        np.testing.assert_array_almost_equal(actual_direction, expected_direction)

    def test_slerp_fallback_to_linear_for_collinear(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([2.0, 0.0, 0.0])

        centroids = interpolate_centroids(v1, v2, num=3, metric="angular")

        expected = np.array([1.5, 0.0, 0.0])
        np.testing.assert_array_almost_equal(centroids[1], expected)

    def test_slerp_fallback_for_zero_vector(self):
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 1.0, 1.0])

        centroids = interpolate_centroids(v1, v2, num=3, metric="angular")

        expected = np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(centroids[1], expected)

    def test_slerp_high_dimensional(self):
        v1 = np.zeros(128)
        v1[0] = 1.0

        v2 = np.zeros(128)
        v2[1] = 1.0

        centroids = interpolate_centroids(v1, v2, num=5, metric="angular")

        for centroid in centroids:
            magnitude = np.linalg.norm(centroid)
            assert abs(magnitude - 1.0) < 1e-10

    def test_slerp_opposite_vectors(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])

        centroids = interpolate_centroids(v1, v2, num=3, metric="angular")

        np.testing.assert_array_almost_equal(centroids[1], [0.0, 0.0, 0.0])


class TestNormalizeSignature:
    def test_normalizes_case(self):
        sig = _normalize_signature("The Beatles", "Hey Jude")

        assert sig == ("the beatles", "hey jude")

    def test_strips_whitespace(self):
        sig = _normalize_signature("  Artist Name  ", "  Song Title  ")

        assert sig == ("artist name", "song title")

    def test_handles_none_values(self):
        sig = _normalize_signature(None, None)

        assert sig == ("", "")

    def test_handles_empty_strings(self):
        sig = _normalize_signature("", "")

        assert sig == ("", "")

    def test_identical_after_normalization(self):
        sig1 = _normalize_signature("The Beatles", "Hey Jude")
        sig2 = _normalize_signature("THE BEATLES", "HEY JUDE")
        sig3 = _normalize_signature("the beatles", "hey jude")

        assert sig1 == sig2 == sig3

    def test_preserves_special_characters(self):
        sig = _normalize_signature("AC/DC", "Back in Black")

        assert sig == ("ac/dc", "back in black")

    def test_multiple_spaces_collapsed(self):
        sig = _normalize_signature("Pink  Floyd", "Wish You Were   Here")

        assert sig == ("pink  floyd", "wish you were   here")
