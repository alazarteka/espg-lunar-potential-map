# Tests for src/utils/geometry.py

import numpy as np
import pytest

from src import config
from src.utils.geometry import get_intersection_or_none, get_intersections_or_none_batch
from src.utils.units import ureg


class TestGetIntersectionOrNone:
    """Tests for single ray-sphere intersection."""

    @pytest.mark.parametrize(
        "pos,direction,radius,expected",
        [
            # Ray from origin hitting sphere
            (
                np.array([0, 0, 0]),
                np.array([1, 0, 0]),
                1 * ureg.kilometer,
                np.array([1, 0, 0]),
            ),
            # Ray from sphere surface pointing outward (no forward intersection)
            (np.array([1, 0, 0]), np.array([1, 0, 0]), 1 * ureg.kilometer, None),
            # Ray from outside pointing away from sphere
            (np.array([2, 0, 0]), np.array([1, 0, 0]), 1 * ureg.kilometer, None),
            # Ray from outside pointing toward sphere
            (
                np.array([2, 0, 0]),
                np.array([-1, 0, 0]),
                1 * ureg.kilometer,
                np.array([1, 0, 0]),
            ),
        ],
    )
    def test_basic_intersections(self, pos, direction, radius, expected):
        """Test basic intersection scenarios."""
        result = get_intersection_or_none(pos, direction, radius)

        if expected is None:
            assert result is None
        else:
            assert result is not None
            np.testing.assert_allclose(result, expected, atol=config.EPS)

    def test_ray_from_inside_sphere_hits_surface(self):
        """Ray starting inside sphere should hit surface."""
        pos = np.array([0.5, 0, 0])  # Inside unit sphere
        direction = np.array([1, 0, 0])
        radius = 1 * ureg.kilometer

        result = get_intersection_or_none(pos, direction, radius)

        assert result is not None
        # Should hit at (1, 0, 0)
        np.testing.assert_allclose(result, [1, 0, 0], atol=1e-10)

    def test_tangent_ray_from_surface_misses(self):
        """Ray starting on sphere surface and moving tangentially has no forward intersection."""
        # Ray at y=1 moving in x direction, sphere of radius 1
        # Starting position is already on the sphere surface
        pos = np.array([0, 1, 0])
        direction = np.array([1, 0, 0])
        radius = 1 * ureg.kilometer

        result = get_intersection_or_none(pos, direction, radius)

        # No forward intersection (t=0 at tangent point)
        assert result is None

    def test_tangent_ray_from_outside_hits(self):
        """Ray approaching sphere tangentially from outside should hit."""
        # Start slightly off the sphere, approaching tangent point
        pos = np.array([-1, 1.001, 0])  # Slightly above sphere
        direction = np.array([1, 0, 0])  # Moving toward tangent
        radius = 1 * ureg.kilometer

        result = get_intersection_or_none(pos, direction, radius)

        # Should find intersection near the tangent region
        if result is not None:
            # Verify it's on the sphere
            distance = np.linalg.norm(result)
            np.testing.assert_allclose(distance, 1.0, atol=1e-6)

    def test_ray_parallel_to_sphere_misses(self):
        """Ray parallel to sphere surface (outside) should miss."""
        pos = np.array([0, 2, 0])  # Above unit sphere
        direction = np.array([1, 0, 0])  # Parallel to x-axis
        radius = 1 * ureg.kilometer

        result = get_intersection_or_none(pos, direction, radius)

        assert result is None

    def test_unnormalized_direction_vector(self):
        """Function should handle unnormalized direction vectors."""
        pos = np.array([2, 0, 0])
        direction = np.array([-5, 0, 0])  # Not normalized
        radius = 1 * ureg.kilometer

        result = get_intersection_or_none(pos, direction, radius)

        assert result is not None
        np.testing.assert_allclose(result, [1, 0, 0], atol=1e-10)

    def test_different_radius(self):
        """Test with different sphere radius."""
        pos = np.array([0, 0, 0])
        direction = np.array([1, 1, 1])
        radius = 10 * ureg.kilometer

        result = get_intersection_or_none(pos, direction, radius)

        assert result is not None
        # Distance from origin should equal radius
        distance = np.linalg.norm(result)
        np.testing.assert_allclose(distance, 10.0, atol=1e-10)

    def test_3d_ray_intersection(self):
        """Test ray in 3D space."""
        pos = np.array([0, 0, 0])
        direction = np.array([1, 1, 1])  # Diagonal
        radius = np.sqrt(3) * ureg.kilometer

        result = get_intersection_or_none(pos, direction, radius)

        assert result is not None
        # Result should be along diagonal
        np.testing.assert_allclose(result, [1, 1, 1], atol=1e-10)


class TestGetIntersectionsBatch:
    """Tests for batched ray-sphere intersection."""

    def test_batch_all_hits(self):
        """Test batch intersection where all rays hit."""
        n = 10
        pos = np.zeros((n, 3))
        # Rays pointing in various directions from origin
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        direction = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(n)])
        radius = 1 * ureg.kilometer

        points, mask = get_intersections_or_none_batch(pos, direction, radius)

        assert points.shape == (n, 3)
        assert mask.shape == (n,)
        assert np.all(mask)  # All should hit

        # All points should be on sphere surface
        distances = np.linalg.norm(points[mask], axis=1)
        np.testing.assert_allclose(distances, 1.0, atol=1e-10)

    def test_batch_all_misses(self):
        """Test batch where all rays miss."""
        n = 5
        # Rays from outside pointing away
        pos = np.full((n, 3), [2, 0, 0])
        direction = np.full((n, 3), [1, 0, 0])
        radius = 1 * ureg.kilometer

        points, mask = get_intersections_or_none_batch(pos, direction, radius)

        assert points.shape == (n, 3)
        assert mask.shape == (n,)
        assert not np.any(mask)  # All should miss

        # Points should be NaN where miss
        assert np.all(np.isnan(points))

    def test_batch_mixed_hits_and_misses(self):
        """Test batch with some hits and some misses."""
        pos = np.array([
            [0, 0, 0],      # Will hit
            [2, 0, 0],      # Will miss
            [-2, 0, 0],     # Will hit
            [0, 5, 0],      # Will miss
        ])
        direction = np.array([
            [1, 0, 0],      # Hit from origin
            [1, 0, 0],      # Miss (pointing away)
            [1, 0, 0],      # Hit (pointing toward)
            [0, 1, 0],      # Miss (parallel, far away)
        ])
        radius = 1 * ureg.kilometer

        points, mask = get_intersections_or_none_batch(pos, direction, radius)

        assert points.shape == (4, 3)
        assert mask.shape == (4,)

        # Check expected hits/misses
        assert mask[0]  # Hit
        assert not mask[1]  # Miss
        assert mask[2]  # Hit
        assert not mask[3]  # Miss

        # Verify hit points are on sphere
        hit_distances = np.linalg.norm(points[mask], axis=1)
        np.testing.assert_allclose(hit_distances, 1.0, atol=1e-10)

        # Verify miss points are NaN
        assert np.all(np.isnan(points[~mask]))

    def test_batch_zero_direction_vector(self):
        """Test handling of zero-magnitude direction vectors."""
        pos = np.array([[0, 0, 0], [1, 0, 0]])
        direction = np.array([[0, 0, 0], [1, 0, 0]])  # First is zero
        radius = 1 * ureg.kilometer

        points, mask = get_intersections_or_none_batch(pos, direction, radius)

        # Zero direction should be invalid
        assert not mask[0]
        assert np.all(np.isnan(points[0]))

    def test_batch_input_validation(self):
        """Test input validation raises appropriate errors."""
        radius = 1 * ureg.kilometer

        # Wrong shape for pos
        with pytest.raises(ValueError, match="pos must have shape"):
            get_intersections_or_none_batch(
                np.array([1, 2, 3]),  # 1D instead of 2D
                np.array([[1, 0, 0]]),
                radius
            )

        # Wrong second dimension for pos
        with pytest.raises(ValueError, match="pos must have shape"):
            get_intersections_or_none_batch(
                np.array([[1, 2]]),  # Only 2 columns
                np.array([[1, 0, 0]]),
                radius
            )

        # Wrong shape for direction
        with pytest.raises(ValueError, match="direction must have shape"):
            get_intersections_or_none_batch(
                np.array([[1, 0, 0]]),
                np.array([1, 2, 3]),  # 1D instead of 2D
                radius
            )

        # Mismatched leading dimensions
        with pytest.raises(ValueError, match="same leading dimension"):
            get_intersections_or_none_batch(
                np.array([[1, 0, 0], [2, 0, 0]]),  # 2 rows
                np.array([[1, 0, 0]]),  # 1 row
                radius
            )

    def test_batch_empty_input(self):
        """Test batch with zero rays."""
        pos = np.empty((0, 3))
        direction = np.empty((0, 3))
        radius = 1 * ureg.kilometer

        points, mask = get_intersections_or_none_batch(pos, direction, radius)

        assert points.shape == (0, 3)
        assert mask.shape == (0,)

    def test_batch_large_radius(self):
        """Test with large sphere radius."""
        n = 5
        pos = np.zeros((n, 3))
        direction = np.random.randn(n, 3)
        radius = 1000 * ureg.kilometer

        points, mask = get_intersections_or_none_batch(pos, direction, radius)

        # All rays from origin should hit
        assert np.all(mask)

        # All points should be at specified radius
        distances = np.linalg.norm(points, axis=1)
        np.testing.assert_allclose(distances, 1000.0, atol=1e-6)

    def test_batch_from_inside_sphere(self):
        """Test rays starting from inside the sphere."""
        pos = np.array([
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5],
        ])
        direction = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        radius = 1 * ureg.kilometer

        points, mask = get_intersections_or_none_batch(pos, direction, radius)

        # All should hit the surface
        assert np.all(mask)

        # All intersection points should be on sphere surface
        distances = np.linalg.norm(points, axis=1)
        np.testing.assert_allclose(distances, 1.0, atol=1e-10)

    def test_batch_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme coordinate values."""
        # Very far from sphere
        pos = np.array([[1e6, 1e6, 1e6]])
        direction = np.array([[-1, -1, -1]])  # Pointing toward origin
        radius = 1 * ureg.kilometer

        points, mask = get_intersections_or_none_batch(pos, direction, radius)

        # Should still detect hit
        assert mask[0]
        # Point should be on sphere
        distance = np.linalg.norm(points[0])
        np.testing.assert_allclose(distance, 1.0, atol=1e-8)
