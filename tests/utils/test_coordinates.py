"""Tests for coordinate-system construction helpers."""

import numpy as np
import pytest

from src.utils.coordinates import build_scd_to_j2000, build_scd_to_reference


def test_build_scd_to_reference_identity_axes() -> None:
    spin = np.array([[0.0, 0.0, 1.0]])
    sun = np.array([[1.0, 0.0, 0.0]])

    actual = build_scd_to_reference(spin, sun)

    np.testing.assert_allclose(actual[0], np.eye(3), atol=1e-15)


def test_build_scd_to_reference_is_right_handed_and_orthonormal() -> None:
    spin = np.array([[1.0, 2.0, 3.0], [-2.0, 1.0, 4.0]])
    sun = np.array([[4.0, -1.0, 2.0], [1.0, 5.0, -2.0]])

    matrices = build_scd_to_reference(spin, sun)

    products = np.einsum("nji,njk->nik", matrices, matrices)
    np.testing.assert_allclose(products, np.tile(np.eye(3), (2, 1, 1)), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(matrices), np.ones(2), atol=1e-14)


@pytest.mark.parametrize(
    "spin,sun",
    [
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 2.0]),
    ],
)
def test_build_scd_to_reference_rejects_degenerate_axes(spin, sun) -> None:
    matrix = build_scd_to_reference(
        np.asarray([spin], dtype=float), np.asarray([sun], dtype=float)
    )

    assert np.isnan(matrix).all()


def test_build_scd_to_reference_validates_shapes() -> None:
    with pytest.raises(ValueError, match="spin_vecs"):
        build_scd_to_reference(np.ones(3), np.ones((1, 3)))
    with pytest.raises(ValueError, match="sun_vecs"):
        build_scd_to_reference(np.ones((2, 3)), np.ones((1, 3)))


def test_j2000_compatibility_wrapper() -> None:
    spin = np.array([[0.0, 0.0, 1.0]])
    sun = np.array([[1.0, 0.0, 0.0]])

    np.testing.assert_array_equal(
        build_scd_to_j2000(spin, sun), build_scd_to_reference(spin, sun)
    )
