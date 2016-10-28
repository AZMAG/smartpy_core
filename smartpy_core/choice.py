"""
Contains methods for making choices.

"""

import numpy as np
import pandas as pd
from patsy import dmatrix

from .wrangling import broadcast


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
        print t
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
    boolean pandas.Series

    """
    # get the design matrix
    if lower('intercept') not in data.columns:
        data['intercept'] = 1 # should I be copying this first?
    coeff_cols = list(coeff.index.values)
    model_design = dmatrix(data[coeff_cols], return_type='dataframe')

    # get utilties and probabilities
    u = np.exp(np.dot(model_design.values, coeff.values.T))
    print u
    p = u / (1 + u)
    print p

    # make the choice
    return binary_choice(p)


def multiple_choice():
    """

    """
    return
