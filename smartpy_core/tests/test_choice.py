import numpy as np
import pandas as pd
import pytest

from ..choice import *
from ..sampling import seeded_call


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
        seed,
        rate_based_binary_choice,
        rates, 'rate', agents, ['grp1', 'grp2']
    )
    assert (c == expected_result).all()


def test_logit_binary_choice():
    seed = 123
    agents = pd.DataFrame({
        'do_it': [1, 1, 1, 1, 0, 0, 0],
        'var1': [1, 1, 1, 0, 0, 0, 0],
        'var2': np.arange(7, 0, -1),
        'intercept': np.ones(7)
    })
    coeff = pd.Series(
        [-2, 3, .5],
        index=pd.Index(['intercept', 'var1', 'var2'])
    )
    u, p, c = seeded_call(seed, logit_binary_choice, coeff, agents)

    expected_u = pd.Series([
        90.0171313, 54.59815003, 33.11545196, 1.,
        0.60653066, 0.36787944, 0.22313016
    ])
    assert (u.round(5) == expected_u.round(5)).all()

    expected_p = pd.Series([
        0.98901306, 0.98201379, 0.97068777, 0.5,
        0.37754067, 0.26894142, 0.18242552
    ])
    assert (p.round(5) == expected_p.round(5)).all()

    # expected randoms should be
    # [ 0.69646919  0.28613933  0.22685145  0.55131477
    #   0.71946897  0.42310646 0.9807642 ]
    expected_c = pd.Series([
        True, True, True, False, False, False, False
    ])
    assert (c == expected_c).all()


##########################
# WEIGHTED MULTIPLE CHOICE
##########################


@pytest.fixture()
def agents():
    return pd.DataFrame({'col': ['a', 'b', 'c']})


@pytest.fixture()
def alts():
    return pd.DataFrame({
        'w': [1, 5, 10],
        'c': [3, 2, 1]
    })


def test_weighted_choice_noW_noCap(agents, alts):
    c = seeded_call(
        123, weighted_choice,
        agents, alts
    )
    expected_c = pd.Series([2, 1, 2])
    assert (c == expected_c).all()


def test_weighted_choice_noCap(agents, alts):
    c = seeded_call(
        123, weighted_choice,
        agents, alts, 'w'
    )
    expected_c = pd.Series([2, 1, 1])
    assert (c == expected_c).all()


def test_weighted_choice_noW(agents, alts):
    c = seeded_call(
        123, weighted_choice,
        agents, alts, cap_col='c'
    )
    expected_c = pd.Series([0, 1, 1])
    assert (c == expected_c).all()


def test_weighted_choice(agents, alts):
    c = seeded_call(
        123, weighted_choice,
        agents, alts, w_col='w', cap_col='c'
    )
    expected_c = pd.Series([2, 1, 1])
    assert (c == expected_c).all()


def test_weighted_choice_notEnoughCap(agents, alts):
    with pytest.raises(ValueError):
        weighted_choice(agents, alts.loc[2], w_col='w', cap_col='c')
