# Tests for src/utils/geometry.py

import numpy as np
import pytest

from src import config
from src.utils.geometry import get_intersection_or_none
from src.utils.units import ureg, Length


@pytest.mark.parametrize(
    "pos,direction,radius,expected",
    [
        (np.array([0, 0, 0]), np.array([1, 0, 0]), 1 * ureg.kilometer, np.array([1, 0, 0])),
        (np.array([1, 0, 0]), np.array([1, 0, 0]), 1 * ureg.kilometer, None),
        (np.array([2, 0, 0]), np.array([1, 0, 0]), 1 * ureg.kilometer, None),
        (np.array([2, 0, 0]), np.array([-1, 0, 0]), 1 * ureg.kilometer, np.array([1, 0, 0])),
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
