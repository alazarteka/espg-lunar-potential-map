# Tests for src/utils/geometry.py

import pytest

import numpy as np
from src.utils.geometry import *
from src import config

@pytest.mark.parametrize(
        "pos,direction,radius,expected",

        [
            (np.array([0, 0, 0]), np.array([1, 0, 0]), 1, np.array([1, 0, 0])),  
            (np.array([1, 0, 0]), np.array([1, 0, 0]), 1, None),
            (np.array([2, 0, 0]), np.array([1, 0, 0]), 1, None),
            (np.array([2, 0, 0]), np.array([-1, 0, 0]), 1, np.array([1, 0, 0])),
        ]
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