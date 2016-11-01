"""
Contains methods for making choices.

"""

import numpy as np
import pandas as pd
from patsy import dmatrix

from .wrangling import broadcast, explode
from .sampling import get_probs


def binary_choice(p, t=None):
    """
    Performs a binary choice from a series of probabilities.

    Paramters:
    ---------
    p: pandas.Series
        Series of probabilities.
    t: numeric or array-like
        Threshold value to test against. If not provided
        a random number will be generated.
    Returns:
    --------
    boolean pandas.Series

    """
    if t is None:
        t = np.random.rand(len(p))
    return p > t


def rate_based_binary_choice(rates, rate_col, agents, segment_cols, set_rate_index=True):
    """
    Performs a binary choice using a segmented rates table.
    The rates imply probabilities and should range from 0 - 1.

    Parameters:
    -----------
    rates: pandas.DataFrame
        Data frame containing rates to use as probabilities.
    rates_col: string
        Column in rates table containing rates/probabilities.
    agents: pandas.DataFrame
        Data frame containing agents to choose from.
    segment_cols: string
        List of columns names to link rates to agents.
    set_rate_index: bool, optional default True
        If true, sets the index on the rates to match the segments.

    Returns:
    --------
    boolean pandas.Series

    """
    r = rates
    if set_rate_index:
        r = rates.set_index(segment_cols)
    p = broadcast(r[rate_col], agents, segment_cols)
    p.fillna(0)
    return binary_choice(p)


def logit_binary_choice(coeff, data):
    """
    Performs a binary choice using a logit model.

    Parameters:
    -----------
    coeff: pandas.Series
        Table containing coefficients. Index is the variable
        name, the value the coefficient.
    data: pandas.DataFrame
        Table containing data to choose from. Should have
        columns for all the coefficents.

    SCOTT TODO: how to best allow custom functions in the dmatrix
    evaluation?? Need to figure out how to import these.

    Returns:
    --------
    u - pandas.Series of utilities
    p - pandas.Series of probabilities
    c - pandas.Series of boolean choices

    """
    # get the design matrix
    if 'intercept' not in data.columns:
        data['intercept'] = 1  # should I be copying this first?
    coeff_cols = list(coeff.index.values)
    model_design = dmatrix(data[coeff_cols], return_type='dataframe')

    # get utilties and probabilities
    u = np.exp(np.dot(model_design.values, coeff.values.T))
    p = u / (1 + u)

    # make the choice and return the results
    return u, p, binary_choice(p)


def weighted_choice(agents, alternatives, w_col=None, cap_col=None):
    """
    Makes choices based on scaling weights.

    Parameters:
    -----------
    agents: pandas.DataFrame or pandas.Series
        Agents to make choices.
    alternatives: pandas.DataFrame
        Choice set of alternatives.
    w_col: string, optional, default None.
        Column to serve as weights for the choice set.
    cap_col: string
        Column to serve as capacities for the choice set.

    Returns:
    --------
    pandas.Series of the chosen indexes, aligned to the agents.

    """
    if cap_col is None:
        # unconstrained choice
        if w_col is None:
            probs = None
        else:
            probs = get_probs(alternatives[w_col]).values
        choice_idx = np.random.choice(alternatives.index.values, len(agents), p=probs)
    else:
        # capcity limited choice
        if w_col is None:
            e = explode(alternatives[[cap_col]], cap_col, 'old_idx')
            choice_idx = np.random.choice(e['old_idx'].values, len(agents), replace=False)
        else:
            # make sure we have enough
            if len(agents) > alternatives[cap_col].sum():
                raise ValueError('Not enough capacity for agents')

            # get a row for each unit of capacity
            e = explode(alternatives[[w_col, cap_col]], cap_col, 'old_idx')

            # normalize w/in group weights
            w_sums = broadcast(e.groupby('old_idx')[w_col].sum(), e['old_idx'])
            print w_sums
            gtz = w_sums > 0
            e.loc[gtz, w_col] = e[w_col] / w_sums

            # make the choice
            probs = get_probs(e[w_col])
            print probs
            print e
            choice_idx = np.random.choice(
                e['old_idx'].values, len(agents), p=probs.values, replace=False)

    return pd.Series(choice_idx, index=agents.index)
