import numpy as np
import pytest

from utils import get_p2i

@pytest.mark.parametrize("data,expected", [
    # Basic test with 3 patients
    (
        np.array([
            [1, 10],
            [1, 11],
            [2, 20],
            [2, 21],
            [2, 22],
            [3, 30],
        ]),
        np.array([
            [0, 2],
            [2, 3],
            [5, 1],
        ])
    ),

    # One patient only
    (
        np.array([
            [4, 100],
            [4, 101],
            [4, 102],
        ]),
        np.array([
            [0, 3],
        ])
    ),

    # Each patient with a single row
    (
        np.array([
            [10, 0],
            [11, 1],
            [12, 2],
        ]),
        np.array([
            [0, 1],
            [1, 1],
            [2, 1],
        ])
    ),
])
def test_get_p2i(data, expected):
    result = get_p2i(data)
    assert np.array_equal(result, expected)
