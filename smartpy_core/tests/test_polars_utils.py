"""
Units tests for polars_utils.py

"""

import numpy as np
import pandas as pd
import polars as pl 
import pytest

from ..polars_utils import *

####################
# sampling
####################


def test_segmented_sampling_with_replace():
    """
    Test to ensure we're re-producing the original distributions.

    """
    # number of sampling iterations
    num_iter = 100
    
    # sample counts
    test_counts = pl.DataFrame({
        'grp': ['red', 'green'],
        'amount': [15_000, 12_000]
    }).lazy()

    # rows to sample
    test_df = pl.DataFrame({
        'grp': ['red', 'red', 'red', 'red', 'green', 'green', 'green'],
        'id': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
        'w':   [500, 200, 100, 200, 0, 400, 600]
    }).lazy()

    # do the sampling
    to_concat = []
    for i in range(0, num_iter):
        sampled = (
            segmented_sample_with_replace(test_df, test_counts, 'amount', 'grp', 'w')
            .with_columns(iter=pl.lit(i))
        )
        to_concat.append(sampled)
    sampled_all = pl.concat(to_concat)

    # check resulting distributions
    # ...we'll check the original probabilities against the mean iter probability
    tolerance = 0.001
    sample_mean_p = (
        sampled_all
        .group_by(['grp', 'id', 'iter'])
        .len()
        .with_columns(sampled_p=pl.col('len').smart.probs().over(['grp', 'iter']))
        .group_by('id')
        .agg(pl.mean('sampled_p'))   
    )
    compare = (
        test_df
        .with_columns(p=pl.col('w').smart.probs().over('grp'))
        .join(sample_mean_p, 'id', 'left')
        .with_columns(
            pl.col('sampled_p').fill_null(0)
        )
        .with_columns(
            p_diff = (pl.col('p') - pl.col('sampled_p')).abs()
        )
        .filter(pl.col('p_diff') > tolerance)
    ).collect()
    assert compare.height == 0
