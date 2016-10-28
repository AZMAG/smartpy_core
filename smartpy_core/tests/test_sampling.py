import numpy as np
import pandas as pd
import pytest

from ..sampling import *


def test_get_probs():
    # 1st test the typical case
    s = pd.Series([8, 6, 4, 2, np.nan])
    p = get_probs(s)
    assert (p.values == [.4, .3, .2, .1, 0]).all()

    # now test a case with a zero weight sum
    s = pd.Series(np.zeros(4))
    p = get_probs(s)
    assert (p.values == 0.25).all()


def test_get_segmented_probs():
    df = pd.DataFrame({
        'grp': ['a', 'a', 'a', 'b', 'b', 'c', 'c'],
        'w': [1, 3, 6, 1, np.nan, 0, 0]
    })
    p = get_segmented_probs(df, 'w', 'grp')
    assert (p.values == [.1, .3, .6, 1, 0, 0.5, 0.5]).all()


def test_randomize_probs():
    # the random seed to apply
    seed = 123

    # get the function results
    w = pd.Series([2, 4, 6])
    p = get_probs(w)
    r_p = seeded_call(seed, randomize_probs, p)

    # make sure we can reproduce these
    def get_r(count):
        return np.random.rand(count)

    r = seeded_call(seed, get_r, len(p))
    res = np.power(r, 1.0 / p)
    assert (r_p.values == res).all()


def test_sample2d():
    seed = 123
    num_agents = 5   # rows
    num_alts = 20    # choiceset
    sample_size = 5  # colums
    alts = pd.Series(np.random.rand(num_alts))

    sample = seeded_call(seed,
                         sample2d,
                         alts.index.values, num_agents, sample_size)
    row_dups = np.hstack([
        np.full((num_agents), False, dtype=bool).reshape(num_agents, 1),
        sample[:, 1:] == sample[:, :-1]
    ])
    assert np.sum(row_dups) == 0
    assert sample.shape == (num_agents, sample_size)


def test_sample2d_not_enough_alts():
    num_agents = 5   # rows
    num_alts = 3     # choiceset
    sample_size = 5  # colums
    alts = pd.Series(np.random.rand(num_alts))

    # 1st test without replacement, this should raise
    with pytest.raises(ValueError):
        sample2d(alts.values, num_agents, sample_size)

    # then test with replacement -- this is fine
    sample = sample2d(alts.index.values, num_agents, sample_size, True)
    assert sample.shape == (num_agents, sample_size)
