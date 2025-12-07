"""Unit tests for tasks/path_manager.py

Tests core path generation logic:
- Distance calculations (Euclidean and Angular)
- Vector interpolation (Linear and SLERP)
- Signature normalization for deduplication
"""
import pytest
import numpy as np
from tasks.path_manager import (
    get_euclidean_distance,
    get_angular_distance,
    get_distance,
    interpolate_centroids,
    _normalize_signature
)


class TestEuclideanDistance:
    """Test Euclidean distance calculation"""

    def test_identical_vectors_return_zero(self):
        """Identical vectors should have distance 0"""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.0, 2.0, 3.0])
        
        dist = get_euclidean_distance(v1, v2)
        
        assert dist == 0.0

    def test_known_distance_3_4_5_triangle(self):
        """Test with known Euclidean distance (3-4-5 triangle)"""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([3.0, 4.0, 0.0])
        
        dist = get_euclidean_distance(v1, v2)
        
        assert abs(dist - 5.0) < 1e-10

    def test_none_vector_returns_inf(self):
        """None vectors should return infinity"""
        v1 = np.array([1.0, 2.0])
        
        assert get_euclidean_distance(None, v1) == float('inf')
        assert get_euclidean_distance(v1, None) == float('inf')
        assert get_euclidean_distance(None, None) == float('inf')

    def test_unit_vectors_on_axes(self):
        """Test distance between orthogonal unit vectors"""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        
        dist = get_euclidean_distance(v1, v2)
        
        # Distance is sqrt(2)
        assert abs(dist - np.sqrt(2)) < 1e-10


class TestAngularDistance:
    """Test Angular distance calculation (derived from cosine similarity)"""

    def test_identical_vectors_return_zero(self):
        """Identical vectors should have angular distance 0"""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.0, 2.0, 3.0])
        
        dist = get_angular_distance(v1, v2)
        
        assert abs(dist) < 1e-10

    def test_orthogonal_vectors_return_half(self):
        """Orthogonal vectors should have angular distance 0.5 (90 degrees / 180)"""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        
        dist = get_angular_distance(v1, v2)
        
        # arccos(0) / pi = pi/2 / pi = 0.5
        assert abs(dist - 0.5) < 1e-10

    def test_opposite_vectors_return_one(self):
        """Opposite vectors should have angular distance 1.0 (180 degrees / 180)"""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        
        dist = get_angular_distance(v1, v2)
        
        # arccos(-1) / pi = pi / pi = 1.0
        assert abs(dist - 1.0) < 1e-10

    def test_scaled_vectors_same_direction(self):
        """Vectors pointing same direction but different magnitudes should have distance 0"""
        v1 = np.array([1.0, 1.0])
        v2 = np.array([2.0, 2.0])
        
        dist = get_angular_distance(v1, v2)
        
        # Same direction, angular distance = 0 (using realistic floating-point tolerance)
        assert abs(dist) < 1e-7

    def test_zero_vector_returns_inf(self):
        """Zero vector should return infinity"""
        v1 = np.array([1.0, 2.0])
        v_zero = np.array([0.0, 0.0])
        
        assert get_angular_distance(v_zero, v1) == float('inf')
        assert get_angular_distance(v1, v_zero) == float('inf')

    def test_none_vector_returns_inf(self):
        """None vectors should return infinity"""
        v1 = np.array([1.0, 2.0])
        
        assert get_angular_distance(None, v1) == float('inf')
        assert get_angular_distance(v1, None) == float('inf')


class TestGetDistance:
    """Test get_distance wrapper with metric configuration"""

    def test_uses_euclidean_by_default(self, monkeypatch):
        """Test that euclidean is used when PATH_DISTANCE_METRIC is 'euclidean'"""
        import tasks.path_manager
        monkeypatch.setattr(tasks.path_manager, 'PATH_DISTANCE_METRIC', 'euclidean')
        
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        
        dist = get_distance(v1, v2)
        
        # Should use Euclidean (3-4-5 triangle)
        assert abs(dist - 5.0) < 1e-10

    def test_uses_angular_when_configured(self, monkeypatch):
        """Test that angular is used when PATH_DISTANCE_METRIC is 'angular'"""
        import tasks.path_manager
        monkeypatch.setattr(tasks.path_manager, 'PATH_DISTANCE_METRIC', 'angular')
        
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        
        dist = get_distance(v1, v2)
        
        # Should use Angular (orthogonal = 0.5)
        assert abs(dist - 0.5) < 1e-10


class TestInterpolateCentroidsLinear:
    """Test linear (Euclidean) interpolation"""

    def test_linear_interpolation_endpoints(self):
        """Test that linear interpolation includes endpoints"""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([10.0, 0.0])
        
        centroids = interpolate_centroids(v1, v2, num=5, metric="euclidean")
        
        assert centroids.shape == (5, 2)
        # First point should be v1
        np.testing.assert_array_almost_equal(centroids[0], v1)
        # Last point should be v2
        np.testing.assert_array_almost_equal(centroids[-1], v2)

    def test_linear_interpolation_midpoint(self):
        """Test that linear interpolation creates correct midpoint"""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([10.0, 20.0, 30.0])
        
        centroids = interpolate_centroids(v1, v2, num=3, metric="euclidean")
        
        # Middle point should be exactly halfway
        expected_midpoint = np.array([5.0, 10.0, 15.0])
        np.testing.assert_array_almost_equal(centroids[1], expected_midpoint)

    def test_linear_interpolation_evenly_spaced(self):
        """Test that linear interpolation creates evenly spaced points"""
        v1 = np.array([0.0])
        v2 = np.array([4.0])
        
        centroids = interpolate_centroids(v1, v2, num=5, metric="euclidean")
        
        # Should be [0, 1, 2, 3, 4]
        expected = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        np.testing.assert_array_almost_equal(centroids, expected)

    def test_linear_single_dimension(self):
        """Test linear interpolation in 1D"""
        v1 = np.array([0.0])
        v2 = np.array([1.0])
        
        centroids = interpolate_centroids(v1, v2, num=11, metric="euclidean")
        
        # Should create 11 points from 0.0 to 1.0
        assert len(centroids) == 11
        assert abs(centroids[0][0] - 0.0) < 1e-10
        assert abs(centroids[-1][0] - 1.0) < 1e-10


class TestInterpolateCentroidsSLERP:
    """Test SLERP (Spherical Linear Interpolation) for angular metric"""

    def test_slerp_endpoints(self):
        """Test that SLERP includes endpoints"""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        
        centroids = interpolate_centroids(v1, v2, num=5, metric="angular")
        
        assert centroids.shape == (5, 3)
        # First and last points should match v1 and v2
        np.testing.assert_array_almost_equal(centroids[0], v1)
        np.testing.assert_array_almost_equal(centroids[-1], v2)

    def test_slerp_preserves_magnitude_variation(self):
        """Test that SLERP linearly interpolates magnitude"""
        # Start with magnitude 1, end with magnitude 2
        v1 = np.array([1.0, 0.0, 0.0])  # magnitude 1
        v2 = np.array([0.0, 2.0, 0.0])  # magnitude 2
        
        centroids = interpolate_centroids(v1, v2, num=3, metric="angular")
        
        # Midpoint should have magnitude 1.5
        midpoint_magnitude = np.linalg.norm(centroids[1])
        assert abs(midpoint_magnitude - 1.5) < 1e-10

    def test_slerp_arc_on_unit_sphere(self):
        """Test that SLERP follows great circle arc on unit sphere"""
        # Two orthogonal unit vectors
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        
        centroids = interpolate_centroids(v1, v2, num=5, metric="angular")
        
        # All points should lie on unit circle (magnitude = 1)
        # because both endpoints are unit vectors
        for i, centroid in enumerate(centroids):
            magnitude = np.linalg.norm(centroid)
            assert abs(magnitude - 1.0) < 1e-10, f"Point {i} has magnitude {magnitude}"

    def test_slerp_midpoint_45_degrees(self):
        """Test that SLERP midpoint between orthogonal vectors is at 45 degrees"""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        
        centroids = interpolate_centroids(v1, v2, num=3, metric="angular")
        
        # Midpoint should be at 45 degrees: [cos(45°), sin(45°)] = [sqrt(2)/2, sqrt(2)/2]
        expected_direction = np.array([np.sqrt(2)/2, np.sqrt(2)/2])
        actual_direction = centroids[1] / np.linalg.norm(centroids[1])
        np.testing.assert_array_almost_equal(actual_direction, expected_direction)

    def test_slerp_fallback_to_linear_for_collinear(self):
        """Test that SLERP falls back to linear for nearly collinear vectors"""
        # Same direction, different magnitudes
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([2.0, 0.0, 0.0])
        
        centroids = interpolate_centroids(v1, v2, num=3, metric="angular")
        
        # Should fall back to linear interpolation
        # Midpoint should be [1.5, 0.0, 0.0]
        expected = np.array([1.5, 0.0, 0.0])
        np.testing.assert_array_almost_equal(centroids[1], expected)

    def test_slerp_fallback_for_zero_vector(self):
        """Test that SLERP falls back to linear for zero vectors"""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 1.0, 1.0])
        
        centroids = interpolate_centroids(v1, v2, num=3, metric="angular")
        
        # Should fall back to linear interpolation
        # Midpoint should be halfway
        expected = np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(centroids[1], expected)

    def test_slerp_high_dimensional(self):
        """Test that SLERP works in high dimensions"""
        # Create two unit vectors in 128 dimensions
        v1 = np.zeros(128)
        v1[0] = 1.0
        
        v2 = np.zeros(128)
        v2[1] = 1.0
        
        centroids = interpolate_centroids(v1, v2, num=5, metric="angular")
        
        # All points should be unit vectors
        for centroid in centroids:
            magnitude = np.linalg.norm(centroid)
            assert abs(magnitude - 1.0) < 1e-10

    def test_slerp_opposite_vectors(self):
        """Test SLERP with opposite vectors (180 degrees)"""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        
        # SLERP is undefined for opposite vectors, should fall back to linear
        centroids = interpolate_centroids(v1, v2, num=3, metric="angular")
        
        # Midpoint should be at origin (linear interpolation)
        np.testing.assert_array_almost_equal(centroids[1], [0.0, 0.0, 0.0])


class TestNormalizeSignature:
    """Test song signature normalization for deduplication"""

    def test_normalizes_case(self):
        """Test that normalization converts to lowercase"""
        sig = _normalize_signature("The Beatles", "Hey Jude")
        
        assert sig == ("the beatles", "hey jude")

    def test_strips_whitespace(self):
        """Test that normalization strips whitespace"""
        sig = _normalize_signature("  Artist Name  ", "  Song Title  ")
        
        assert sig == ("artist name", "song title")

    def test_handles_none_values(self):
        """Test that normalization handles None gracefully"""
        sig = _normalize_signature(None, None)
        
        assert sig == ("", "")

    def test_handles_empty_strings(self):
        """Test that normalization handles empty strings"""
        sig = _normalize_signature("", "")
        
        assert sig == ("", "")

    def test_identical_after_normalization(self):
        """Test that different cases produce same signature"""
        sig1 = _normalize_signature("The Beatles", "Hey Jude")
        sig2 = _normalize_signature("THE BEATLES", "HEY JUDE")
        sig3 = _normalize_signature("the beatles", "hey jude")
        
        assert sig1 == sig2 == sig3

    def test_preserves_special_characters(self):
        """Test that special characters are preserved"""
        sig = _normalize_signature("AC/DC", "Back in Black")
        
        assert sig == ("ac/dc", "back in black")

    def test_multiple_spaces_collapsed(self):
        """Test handling of multiple spaces"""
        # Strip removes leading/trailing, but internal spaces remain
        sig = _normalize_signature("Pink  Floyd", "Wish You Were   Here")
        
        assert sig == ("pink  floyd", "wish you were   here")
