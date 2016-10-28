"""
Contains functions and utilities related to sampling items from
numpy and pandas.

"""
import numpy as np
import pandas as pd

from .wrangling import broadcast, handle_nulls


def get_probs(weights, drop_na=False):
    """
    Returns probabilities for a series of weights.

    Parameters:
    -----------
    weights: pandas.Series
        Series to get probabilities for.
    drop_na: optional, default False
        If False, nulls become 0s. Otherwise
        they are removed from the series.

    """
    w = handle_nulls(weights, drop_na)

    w_sum = w.sum()
    if w_sum == 0:
        probs = pd.Series(np.ones(len(w)) / len(w), index=w.index)
    else:
        probs = w / w_sum

    return probs


def get_segmented_probs(df, w_col, segment_cols, drop_na=False):
    """
    Converts a series of weights into probabilities across multiple
    segments.

    Parameters:
    -----------
    df: pandas.DataFrame
        Data frame containing weights.
    w_col: string
        Name of column containing weights.
    segment_cols: str, list of str, series, index, ...
        Defines  segments to generate probabilties for.
    drop_na: optional, defualt None
        If provided, the data frame will have null values in the weights column
        replaced with this value.

    Returns:
    --------
    pandas.Series with probabilties

    """

    # get the weight series, handle nulls if needed
    w = handle_nulls(df[w_col], drop_na)

    # get the probabilities
    w_sums = broadcast(w.groupby(df[segment_cols]).sum(), df, segment_cols)
    probs = w / w_sums

    # handle cases where all weights are 0
    z_sums = w_sums == 0
    if z_sums.any():
        w_cnts = broadcast(df.groupby(segment_cols).size(), df[segment_cols])
        probs[z_sums] = 1 / w_cnts[z_sums]

    return probs


def seeded_call(seed, func, *args, **kwargs):
    """
    Executes a function with the provided seed. Reverts
    the PRNG back to the previous state after the function
    call. Allows for reproducing results for functions
    with random dependencies.

    Parameters:
    -----------
    seed: numeric
        The seed to provide the function.
    func: callable
        The function to execute.

    """
    old_state = np.random.get_state()
    np.random.seed(seed)
    results = func(*args, **kwargs)
    np.random.set_state(old_state)
    return results


def randomize_probs(p):
    """
    Randomizes probabilities to be used in weighted sorted sampling.
    The probabilities will no longer sum to 1.

    Parameters:
    -----------
    p: pandas.Series
        Series containing probablilties to randomize.

    Returns:
    --------
    pandas.Series

    """
    return np.power(np.random.rand(len(p)), 1.0 / p)


def sample2d(arr, num_rows, num_cols, replace=False, max_iter=100):
    """
    Samples from the provided array into a two-dimensional
    array.

    Parameters:
    -----------
    arr: numpy.array
        Numpy array containing the values to sample.
    num_rows: int
        Number of rows in the resulting matrix.
    num_cols: int
        Number of columns in the resulting matrix.
    replace: bool, optional, default False
        If true, the same item can be duplicated row-wise.
        If false, there will be no duplicates row-wise.
        In both cases, there can be column-level duplicates. If both
        row and column uniqueness is needed then use np.random.choice
        with replacement and reshape.
    max_iter: int
        Maximum number of iterations to apply when eliminating row-level
        duplicates.

    Returns:
    --------
    numpy.array with shape (num_rows, num_cols)

    TODO:
        - Do we need to ensure the IDs in the array are unique?
        - Allow sampling weights
        - See if we can avoid sorting each iteration.
        - The ordering within a row follows the IDs and is not random.
    """

    # do the initial sample with replacement
    sample = np.random.choice(arr, size=num_rows * num_cols).reshape(num_rows, num_cols)
    if replace:
        # if sampling with replacement we are done
        return sample

    # if no replacement, replace duplicates
    if len(arr) < num_cols:
        raise ValueError('not enough elements')

    if len(arr) == num_cols:
        return np.tile(arr, num_rows).reshape(num_rows, num_cols)

    curr_iter = 0
    while curr_iter < max_iter:

        # sort the sample IDs
        sample.sort()

        # get duplicates
        is_dup = np.hstack([
            np.full((num_rows), False, dtype=bool).reshape(num_rows, 1),
            sample[:, 1:] == sample[:, :-1]
        ])

        num_dups = np.sum(is_dup)
        if num_dups == 0:
            break

        # resample
        new_samples = np.random.choice(arr, size=num_dups)
        sample[is_dup] = new_samples
        curr_iter += 1

    return sample
