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


def get_segmented_probs(df, w_col, segment_cols):
    """
    Converts a series of weights into probabilities across multiple
    segments. Null values are treated as 0s. Segments containing
    all nulls and/or 0s will have equal probabilities.

    Parameters:
    -----------
    df: pandas.DataFrame
        Data frame containing weights.
    w_col: string
        Name of column containing weights.
    segment_cols: str, list of str, series, index, ...
        Defines segments to generate probabilties for.

    Returns:
    --------
    pandas.Series with probabilties

    """

    # get probabilities
    w_sums = broadcast(df.groupby(segment_cols)[w_col].sum(), df, segment_cols)
    probs = df[w_col] / w_sums

    # handle nulls
    w_sums = handle_nulls(w_sums)
    probs = handle_nulls(probs)

    # handle cases where all weights in a segment are 0
    z_sums = w_sums == 0
    if z_sums.any():
        w_cnts = broadcast(df.groupby(segment_cols).size(), df, segment_cols)
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


def segmented_sample_no_replace(amounts, data, segment_cols, w_col=None):
    """
    Returns samples without replacement in a segmented manner. Use
    the weights column to control probabilities.

    Parameters:
    -----------
    amounts: pandas.Series
        Amounts to sample. Should be indexed by the segment.
    data: pandas.DataFrame
        Data to sample from.
    segment_cols: str or list of str
        Columns defining segments on the data. Should
        match the index of the amounts.
    w_col: string, optional, default None
        If provided, defines sampling weights. If None
        the sample is random.


    Returns:
    --------
    numpy.array of the indexes of the sampled rows.

    """

    if not isinstance(segment_cols, list):
        segment_cols = [segment_cols]

    # sort the data frame
    if w_col is None:
        # totally random
        data = data[segment_cols].reindex(np.random.permutation(data.index))
    else:
        # apply weights to get randomized probabilities
        probs = get_segmented_probs(data, w_col, segment_cols)
        data = data[segment_cols].copy()
        data['ran_p'] = randomize_probs(probs)
        data.sort('ran_p', ascending=False, inplace=True)

    # broadcast the amounts
    amounts = broadcast(amounts, data, segment_cols)

    # return the top n rows
    cc = data.groupby(segment_cols).cumcount() + 1
    sampled = cc <= amounts
    return data[sampled].index.values


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
        without replacement and reshape.
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
