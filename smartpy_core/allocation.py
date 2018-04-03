"""
Contains functions for allocating quantities to rows using a deterministic pro-rating
or weighted proportional approach.

"""

import numpy as np
import pandas as pd

from .wrangling import broadcast


def get_rounded(amounts, unrounded, segments=None):
    """
    Rounds while attempting to match a control amount. This should be
    called after applying a pro-rating style of allocation to integerize
    the results.

    SCOTT TODO: randomize this by shuffling the unrounded series first?
        Note as this is currently implemented it's pretty much a bucket rounder.
        But we could do something to make it more stochastic?

    SCOTT TODO: this is row-based, generalize so this could be column based?
        or perhaps that is another method?

    Parameters:
    ----------
    amounts: int or pandas.Series
       Series of amounts we want the rounded results to sum to.
    unrounded: pandas.Series
       Series of floating values that will be rounded.
    segments: pandas.Series or list of pandas.Series, optional
        Links unrounded values to amounts. If not provided, it is
        assumed that the amount is a single scalar. Should
        be aligned with the unrounded series.

    """
    # round everything down and get the fractional components
    f = np.floor(unrounded)

    # find shortages from the floors
    if segments is None:
        shortages = amounts - f.sum()
        if shortages > len(unrounded):
            raise ValueError("Amount exceeds elements available for rounding!")
    else:
        # for the segmented case, we will compute a shortage for each segment
        # and align this back to the unrounded results
        allocated = f.groupby(segments).sum().reindex(amounts.index).fillna(0)
        shortages = broadcast(amounts - allocated, segments)

    # rank based on fractional components -- use cum count instead of rank?
    fract = unrounded - f
    if segments is None:
        rank = fract.rank(ascending=False, method='first')
    else:
        rank = fract.groupby(segments).rank(ascending=False, method='first')

    # round up where necessary to meet the shortages
    to_round_up = rank <= shortages
    f[to_round_up] += 1
    return f


def get_simple_allocation(amount, weights):
    """
    Simplest possible allocation: distributes a single
    quantity/amount to rows based on weights. This allocation
    is un-constrained.

    Parameters:
    ----------
    amount: numeric
        The amount to allocate
    weights: pandas.Series
        Series containing weights to allocate to.

    Returns:
    --------
    Pandas series of resulting allocation. Amounts
    are indexed to the weights series.

    """

    if amount % 1 != 0:
        raise ValueError('Amount must be a whole number')

    # compute the initial, un-rounded allocation
    if (weights == 0).all() or (weights.isnull()).all():
        weights = pd.Series(np.ones(len(weights)), index=weights.index)
    unrounded = amount * (weights / weights.sum())

    # round the results
    return get_rounded(amount, unrounded)


def get_constrained_allocation(amount, weights, capacities):
    """
    Distributes a single amount to rows based on weights
    while respecting capacities.

    Parameters:
    ----------
    amount: numeric
        The amount to allocate
    weights: pandas.Series
        Series containing weights to allocate to.
    capacities: pandas.Series
        Series containing capacities to respect. Should be
        aligned with the weights series.

    Returns:
    --------
    Pandas series of resulting allocation. Amounts
    are indexed to the weights series.

    """
    # check amounts
    if amount % 1 != 0:
        raise ValueError('Amount must be a whole number')

    if amount > capacities.sum():
        raise ValueError('Amount is larger than available capacities')

    # perform the initial allocation
    a = pd.Series(np.zeros(len(weights)), index=weights.index)
    w = weights
    while True:
        curr_amount = amount - a.sum()
        have_cap = a < capacities
        if have_cap.any():
            w_sum = w[have_cap].sum()
            if w_sum != 0:
                a[have_cap] = a + (curr_amount * (w / w_sum))
            else:
                a[have_cap] = a + (curr_amount / len(a[have_cap]))

        # set allocation result to capacity for overages
        over = a > capacities
        if over.any():
            a[over] = capacities
        else:
            break

    # round the results
    return get_rounded(amount, a)


def get_segmented_allocation(amounts, target_weights, target_segments, do_round=True):
    """
    Allocates multiple amounts to target rows using weights. This
    allocation is un-constrained. This assumes there is a 1:m relationship
    between amounts and targets (i.e. the amounts index is unique).

    Parameters:
    ----------
    amounts: pandas.Series
        The amounts to allocate.
    target_weights: pandas.Series
        Series from targets data frame of weights for allocating quantities.
    target_segments: pandas.Series
        Foreign key link to the amounts index. Should be aligned to the weights series.
    do_round: optional, bool, default True
        Indicates if allocated amount will be rounded.

    Returns:
    --------
    Pandas series of resulting allocation.

    """
    if do_round:
        if (amounts % 1 != 0).any():
            raise ValueError('Amounts must be  whole numbers')

    if not amounts.index.is_unique:
        raise ValueError('Amounts index must be unique')

    # join amounts to targets
    t = pd.merge(
        pd.DataFrame({'w': target_weights, 's': target_segments}),
        pd.DataFrame({'amount': amounts}),
        left_on='s',
        right_index=True
    )

    # convert weights to shares, groups w zero sums will have equal shares
    w_sums = broadcast(t.groupby('s')['w'].sum(), t.s)
    shares = t.w / w_sums
    z_sums = w_sums <= 0
    if z_sums.any():
        w_cnts = broadcast(t.groupby('s').size(), t.s)
        shares[z_sums] = 1 / w_cnts[z_sums]

    # do the unrounded allocation
    unrounded = t.amount * shares

    # round
    if do_round:
        rounded = get_rounded(amounts, unrounded, t.s)
    else:
        rounded = unrounded

    return rounded.reindex(target_weights.index).fillna(0)


def get_segmented_allocation_mn(amounts, amount_col, targets, weight_col, segment_cols):
    """
    Similar to get_segmented_allocation but takes in amounts and targets
    as data frames. This allows multiple columns of segmentation and allows m:n
    relationship between amounts and targets.

    Parameters:
    ----------
    amounts: pandas.DataFrame
        The amounts to allocate.
    amount_col: string
        Name of column in amounts containing quantities to allocate.
    targets: pandas.DataFrame
        The destination rows to allocate to.
    weight_col: string
        Name of column in targets to weight the allocation.
    segment_cols: list of strings
        Names of columns to serve as segmentation. Should exist in both
        data frames.

    Returns:
    --------
    Pandas series of resulting allocation, aligned to the targets.

    """

    # group the amounts by the segments
    amount_sums = amounts.groupby(segment_cols)[amount_col].sum().to_frame(amount_col)

    # assign a unique value to the amount groups and broadcast to targets
    amount_sums['amount_idx'] = np.arange(len(amount_sums))
    m = pd.merge(
        targets[[weight_col] + segment_cols],
        amount_sums[['amount_idx']],
        left_on=segment_cols,
        right_index=True
    )
    amount_sums.set_index('amount_idx', inplace=True)

    # now we have a simpler 1:m realtionship, just use the simpler allocation method
    results = get_segmented_allocation(amount_sums[amount_col], m[weight_col], m['amount_idx'])

    # align the results with the input
    return results.reindex(targets.index).fillna(0)


def hierarchical_segmented_allocation(amounts, amount_col, targets, weight_col, segment_cols):
    """
    Similar to get_segmented_allocation_mn, except that amounts for segments are allocated
    independently, over multiple iterations. Use this for cases where the allocation needs
    to 'step up' from more detailed segments and where there may not be targets available
    to each detailed segment. This allocation is unconstrained.

    Parameters:
    ----------
    amounts: pandas.DataFrame
        The amounts to allocate.
    amount_col: string
        Name of column in amounts containing quantities to allocate.
    targets: pandas.DataFrame
        The destination rows to allocate to.
    weight_col: string
        Name of column in targets to weight the allocation.
    segment_cols: list of strings
        Names of columns to serve as segmentation. Should exist in both
        data frames.

    Returns:
    --------
    Pandas series of resulting allocation, aligned to the targets.

    """
    # initialize a series of 0s, indexed to the targets
    result = pd.Series(np.zeros(len(targets)), index=targets.index)

    # keep track of the amounts left to allocate
    amounts = amounts.copy()

    # allocate each segment independently
    num_segs = len(segment_cols)
    for i in range(0, num_segs + 1):

        if i == num_segs:
            # no more segments left, everything is elgible.
            curr_res = get_simple_allocation(amounts[amount_col].sum(), targets[weight_col])
            result.loc[curr_res.index.values] += curr_res
        else:
            # allocate for the current segment
            curr_seg = segment_cols[i]
            curr_amounts = amounts[amounts[curr_seg].isin(targets[curr_seg])]

            # set up the current allocation
            amount_sums = curr_amounts.groupby(curr_seg)[amount_col].sum().to_frame(amount_col)
            amount_sums['amount_idx'] = np.arange(len(amount_sums))
            m = pd.merge(
                targets[[weight_col] + segment_cols],
                amount_sums[['amount_idx']],
                left_on=curr_seg,
                right_index=True
            )
            amount_sums.set_index('amount_idx', inplace=True)

            # update the results
            curr_res = get_segmented_allocation(
                amount_sums[amount_col],
                m[weight_col],
                m['amount_idx'])
            result.loc[curr_res.index.values] += curr_res
            amounts.drop(curr_amounts.index.values, inplace=True)
            if len(amounts) == 0:
                break

    return result
