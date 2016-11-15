"""
Contains methods for making choices.

"""

import numpy as np
import pandas as pd
from patsy import dmatrix

from .wrangling import broadcast, explode
from .sampling import get_probs, get_segmented_probs, randomize_probs, sample2d


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
        Series containing coefficients. Index is the variable
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


def weighted_choice(agents, alternatives, w_col=None, cap_col=None, return_probs=False):
    """
    Makes choices based on weights previously assinged to the alternatives.

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
    return_probs: bool, optional, default False
        If True, probabilities will also be returned.

    Returns:
    --------
    pandas.Series of the chosen indexes, aligned to the agents.

    """
    probs = None

    if cap_col is None:
        # unconstrained choice
        if w_col is not None:
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

            # make the choice
            probs = get_probs(e[w_col] / e[cap_col])
            choice_idx = np.random.choice(
                e['old_idx'].values, len(agents), p=probs.values, replace=False)

    # return the results
    choices = pd.Series(choice_idx, index=agents.index)
    if return_probs:
        return choices, probs  # SCOTT, need to add a test for this
    else:
        return choices


def get_interaction_data(choosers, alternatives, sample_size, sample_replace=True):
    """
    Returns an interaction dataset with attributes of both choosers and alternatives,
    with the number of alternatives per chooser defined by a sample size.

    choosers: pandas.DataFrame
        Data frame of agents making choices.
    alternatives: pandas.DataFrame
        Data frame of alternatives to choose from.
    sample_size: int, optional, default 50
        Number of alternatives to sample for each agent.
    sample_replace: bool, optional, default True
        If True, sampled alternatives and choices can be shared across multiple choosers,
        If False, this will generate a non-overlapping choiceset.

    Returns:
    --------
    interaction_data: pandas.DataFrame
        Data frame with 1 row for each chooser and sampled alternative. Index is a
        multi-index with level 0 containing the chooser IDs and level 1 containing
        the alternative IDs.
    sample_size: int
        Sample size used in the sample. This may be smaller than the provided sample
        size if the number of alternatives is less than the desired sample size.

    """
    num_alts = len(alternatives)
    num_choosers = len(choosers)

    # sample from the alternatives
    if sample_replace:
        # allow the same alternative to be sampled across agents
        sample_size = min(sample_size, num_alts)
        sampled_alt_idx = sample2d(alternatives.index.values, num_choosers, sample_size).ravel()
    else:
        # non-overlapping choice-set
        if num_alts < num_choosers:
            raise ValueError("Less alternatives than choosers!")
        sample_size = min(sample_size, num_alts / num_choosers)
        sampled_alt_idx = np.random.choice(
            alternatives.index.values, sample_size * num_choosers, replace=False)

    # align samples to match choosers
    sampled_alts = alternatives.reindex(sampled_alt_idx)
    alt_idx_name = sampled_alts.index.name
    if alt_idx_name is None:
        alt_idx_name = 'alternative_id'
        sampled_alts.index.name = alt_idx_name
    sampled_alts.reset_index(inplace=True)

    # link choosers w/ sampled alternatives
    choosers_r = choosers.reindex(choosers.index.repeat(sample_size))
    chooser_idx_name = choosers_r.index.name
    if chooser_idx_name is None:
        chooser_idx_name = 'chooser_id'
        choosers_r.index.name = chooser_idx_name
    sampled_alts.index = choosers_r.index
    interaction_data = pd.concat([choosers_r, sampled_alts], axis=1)
    interaction_data.set_index(alt_idx_name, append=True, inplace=True)

    return interaction_data, sample_size


def choice_with_sampling(choosers,
                         alternatives,
                         probs_callback,
                         sample_size=50,
                         sample_replace=True,
                         verbose=False,
                         **prob_kwargs):
    """
    Performs a weighted choice while sampling alternatives. Supports
    attributes on both the chooser and the alternatives.

    Parameters:
    -----------
    choosers: pandas.DataFrame
        Data frame of agents making choices.
    alternatives: pandas.DataFrame
        Data frame of alternatives to choose from.
    probs_callback: function
        - Function used to generate probabilities
          from the sampled interaction data.
        - Should return a numpy matrix with the shape
          (number of choosers, sample size).
        - The probabilities for each row must sum to 1.
        - The following arguments will be passed in to the callback:
            - interaction_data
            - num_choosers
            - sample_size
            - additional keyword args (see **prob_kwargs)
    sample_size: int, optional, default 50
        Number of alternatives to sample for each agent.
    sample_replace: bool, optional, default True
        If True, sampled alternatives and choices can be shared across multiple choosers,
        If False, this will generate a non-overlapping choiceset.
    verbose: bool, optional, default False
        If true, an additional data frame is returned containing
        the choice matrix. This has the columns:
            - chooser_id: index of the chooser
            - alternative_id: index of the alternative
            - prob: the probability
    **prob_kwargs:
        Additional key word arguments to pass to the probabilities
        callback.

    Returns:
    --------
    - pandas.DataFrame of the choices, indexed to the chooses, with columns:
        - alternative_id: index of the chosen alternative
        - prob: probability of the chosen alternative

    - optionally, data frame of all samples (see verbose parameter above)

    """
    num_choosers = len(choosers)

    # get sampled interaction data
    interaction_data, sample_size = get_interaction_data(
        choosers, alternatives, sample_size, sample_replace)
    chooser_idx = interaction_data.index.get_level_values(0).values
    alt_idx = interaction_data.index.get_level_values(1).values

    print 'chooser idx:'
    print chooser_idx

    print ''
    print 'alt idx:'
    print alt_idx

    # assign weights/probabiltities to the alternatives
    # the result should a 2d numpy array with dim num choosers (rows) X num alts (cols)
    probs = probs_callback(
        interaction_data=interaction_data,
        num_choosers=num_choosers,
        sample_size=sample_size,
        **prob_kwargs)
    assert probs.shape == (num_choosers, sample_size)
    assert round(probs.sum(), 0) == num_choosers  # fix this per Jacob's suggestion?

    # make choices for each agent
    cs = np.cumsum(probs, axis=1)
    r = np.random.rand(num_choosers).reshape(num_choosers, 1)
    chosen_rel_idx = np.argmax(r < cs, axis=1)
    chosen_abs_idx = chosen_rel_idx + (np.arange(num_choosers) * sample_size)

    print ''
    print 'chosen abs idx:'
    print chosen_abs_idx

    curr_choices = pd.DataFrame(
        {
            'alternative_id': alt_idx[chosen_abs_idx],
            'prob': probs.ravel()[chosen_abs_idx],
        },
        index=pd.Index(choosers.index)
    )

    # return the results
    if verbose:
        curr_samples = pd.DataFrame({
            'chooser_id': chooser_idx,
            'alternative_id': alt_idx,
            'prob': probs.ravel()
        })
        return curr_choices, curr_samples
    else:
        return curr_choices


def capacity_choice_with_sampling(choosers,
                                  alternatives,
                                  cap_col,
                                  probs_callback,
                                  sample_size=50,
                                  max_iterations=10,
                                  verbose=False,
                                  **prob_kwargs):
    """
    Performs a weighted choice while sampling alternatives and
    respecting alternative capacities.

    Parameters:
    -----------
    choosers: pandas.DataFrame
        Data frame of agents making choices.
    alternatives: pandas.DataFrame
        Data frame of alternatives to choose from.
    cap_col: string
        Column on the alternatives data frame to provide capacities.
    probs_callback: function
        - Function used to generate probabilities
          from the sampled interaction data.
        - Should return a numpy matrix with the shape
          (number of choosers, sample size).
        - The probabilities for each row must sum to 1.
        - The following arguments will be passed in to the callback:
            - interaction_data
            - num_choosers
            - sample_size
            - additional keyword args (see **prob_kwargs)
    sample_size: int, optional, default 50
        Number of alternatives to sample for each agent.
    max_iterations: integer, optional, default 10
        Number of iterations to apply.
    verbose: bool, optional, default False
        If true, an additional data frame is returned containing
        the choice matrix. This has the columns:
            - chooser_id: index of the chooser
            - alternative_id: index of the alternative
            - prob: the probability
        **** NOT IMPLEMENTED RIGHT NOW ******
    **prob_kwargs:
        Additional key word arguments to pass to the probabilities
        callback.

    Returns:
    --------
    choices: pandas.Series
        Series of chosen alternative IDs, indexed to the agents.

    capacity: pandas.Series
        Series containing updated capacities after making choices. Indexed to alternatives.

    """

    # initialize the choice results w/ null values
    choices = pd.Series(index=choosers.index)

    # get alternative capacities
    capacity = alternatives[cap_col].copy()

    for i in range(1, max_iterations + 1):

        print i

        # filter out choosers who have already chosen
        curr_choosers = choosers[choices.isnull()]
        num_choosers = len(curr_choosers)
        if num_choosers == 0:
            break

        # filter out unavailable alternatives
        has_cap = capacity > 0
        curr_cap = capacity[has_cap]
        if len(curr_cap) == 0:
            break
        curr_alts = alternatives[has_cap]

        # put sampling weight stuff here -- leave out for now.

        # handle the last iteration a litle differently
        sample_replace = True
        if i == max_iterations:
            cap_rep = curr_cap.index.repeat(curr_cap.astype(int))
            curr_alts = curr_alts.reindex(cap_rep)
            sample_replace = False

        # get the current choices
        curr_choices = choice_with_sampling(
            curr_choosers,
            curr_alts,
            probs_callback,
            sample_size,
            sample_replace,
            verbose,
            **prob_kwargs
        )

        # handle choices chosen by multiple agents
        # prefer agents with higher probabilities, we can think of this as an inverse choice
        cap_reindex = broadcast(capacity, curr_choices['alternative_id'])
        curr_choices['inv_prob'] = get_segmented_probs(curr_choices, 'prob', 'alternative_id')
        curr_choices['r_inv_prob'] = randomize_probs(curr_choices['inv_prob'])
        curr_choices.sort('r_inv_prob', ascending=False, inplace=True)
        cc = curr_choices.groupby('alternative_id').cumcount() + 1
        chosen = curr_choices['alternative_id'][cc <= cap_reindex]
        choices.loc[chosen.index] = chosen

        # update capacities
        capacity -= chosen.value_counts().reindex(capacity.index).fillna(0)

    return choices, capacity
