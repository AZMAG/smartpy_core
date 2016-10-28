import numpy as np
import pandas as pd

from .choice import *
from .sampling import seeded_call


def test_binary_choice():
    p = pd.Series([0, 0.1, 0.9])
    expected_result = pd.Series([False, False, True])

    # 1st test with a provided threshold
    t = 0.5
    c1 = binary_choice(p, t)
    assert (c1 == expected_result).all()

    # next test with randoms (the default)
    # note the randoms produced with this seed should be
    # [0.69646919,  0.28613933,  0.22685145]
    seed = 123
    c2 = seeded_call(seed, binary_choice, p)
    assert (c2 == expected_result).all()


def test_rate_based_binary_choice():
    # expected randoms should be
    # [ 0.69646919  0.28613933  0.22685145  0.55131477]
    seed = 123

    rates = pd.DataFrame({
        'grp1': ['a', 'a', 'b'],
        'grp2': [1, 2, 1],
        'rate': [0.1, 0.6, 0.3]
    })

    agents = pd.DataFrame({
        'grp1': ['a', 'a', 'b', 'b'],
        'grp2': [1, 2, 1, 3]
    })

    expected_result = pd.Series([False, True, True, False])
    c = seeded_call(
        123,
        rate_based_binary_choice,
        rates, 'rate', agents, ['grp1', 'grp2']
    )
    assert (c == expected_result).all()