# Tests for src/utils/geometry.py

import numpy as np
import pytest

from src import config
from src.utils.geometry import (
    get_connection_and_polarity,
    get_connections_and_polarity_batch,
    get_intersection_or_none,
)
from src.utils.units import ureg


@pytest.mark.parametrize(
    "pos,direction,radius,expected",
    [
        (
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            1 * ureg.kilometer,
            np.array([1, 0, 0]),
        ),
        (np.array([1, 0, 0]), np.array([1, 0, 0]), 1 * ureg.kilometer, None),
        (np.array([2, 0, 0]), np.array([1, 0, 0]), 1 * ureg.kilometer, None),
        (
            np.array([2, 0, 0]),
            np.array([-1, 0, 0]),
            1 * ureg.kilometer,
            np.array([1, 0, 0]),
        ),
    ],
)
def test_get_intersection_or_none(pos, direction, radius, expected):
    """
    Test the get_intersection_or_none function with various positions and directions.
    """

    result = get_intersection_or_none(pos, direction, radius)

    if expected is None:
        assert result is None
    else:
        assert result is not None
        np.testing.assert_allclose(result, expected, atol=config.EPS)


@pytest.mark.parametrize(
    "pos,direction,radius,expected_connected,expected_polarity,expected_point",
    [
        (
            np.array([2.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            1 * ureg.kilometer,
            True,
            -1,
            np.array([1.0, 0.0, 0.0]),
        ),
        (
            np.array([2.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            1 * ureg.kilometer,
            True,
            1,
            np.array([1.0, 0.0, 0.0]),
        ),
        (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            1 * ureg.kilometer,
            False,
            0,
            None,
        ),
        (
            np.array([2.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            1 * ureg.kilometer,
            False,
            0,
            None,
        ),
    ],
)
def test_get_connection_and_polarity(
    pos, direction, radius, expected_connected, expected_polarity, expected_point
):
    connected, polarity, point = get_connection_and_polarity(pos, direction, radius)

    assert connected is expected_connected
    assert polarity == expected_polarity
    if expected_point is None:
        assert point is None
    else:
        np.testing.assert_allclose(point, expected_point, atol=config.EPS)


def test_get_connections_and_polarity_batch():
    pos = np.array(
        [
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    direction = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    points, mask, polarity = get_connections_and_polarity_batch(
        pos, direction, 1 * ureg.kilometer
    )

    assert mask.tolist() == [True, True, False, False]
    assert polarity.tolist() == [-1, 1, 0, 0]
    np.testing.assert_allclose(points[0], [1.0, 0.0, 0.0], atol=config.EPS)
    np.testing.assert_allclose(points[1], [1.0, 0.0, 0.0], atol=config.EPS)
    assert np.isnan(points[2]).all()
    assert np.isnan(points[3]).all()
