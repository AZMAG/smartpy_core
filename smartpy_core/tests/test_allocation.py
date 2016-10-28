import pandas as pd
from ..allocation import *


def test_simple_allocation():
    amount = 101
    weights = pd.Series([85, 50, 10, 0])
    assert get_simple_allocation(amount, weights).sum() == amount


def test_constrained_allocation():
    amount = 103
    weights = pd.Series([10, 25, 1, 0])
    cap = pd.Series([5, 10, 50, 100])
    res = get_constrained_allocation(amount, weights, cap)
    assert res.sum() == amount
    assert (res <= cap).all()


def test_simple_segmented_allocation1():
    """
    This test assumes a 1:m between amounts and targets. In
    this case the index on the amount links to the target segments.

    """

    amounts = pd.Series([101, 201, 80], index=pd.Index(['z', 'a', 'g']))
    target_idx = pd.Index([400, 300, 200, 100, 10, 11, 4, 2])
    target_weights = pd.Series([25, 100, 1, 50, 0, 0, 100, 500], index=target_idx)
    target_segments = pd.Series(['z', 'a', 'a', 'z', 'g', 'g', 'f', 'f'], index=target_idx)

    res = get_segmented_allocation(amounts, target_weights, target_segments)

    return res

    # TODO: do the assertions
