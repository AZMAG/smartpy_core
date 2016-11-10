"""
Contains some utility functions for doing general
data wrangling operations with Pandas and Numpy.

"""

import numpy as np
import pandas as pd


def broadcast(right, left, left_fk=None):
    """
    Re-indexes a series or data frame (right) to align with
    another (left) series or data frame via foreign key relationship.
    The index of the right must be unique.

    Parameters:
    -----------
    right: pandas.DataFrame or pandas.Series
        Columns or set of columns to re-project(broadcast).
    left: pandas.Series or pandas.DataFrame
        Foreign keys to join on. If a series is provided,
        then the series values are used. To broadcast on
        multiple columns, the left will be a DataFrame
        and the left_fk must be provided.
    left_fk: str or list of str
        Use this when broadcasting across multiple
        columns. The right index in the case should
        be a multindex.

    Returns:
    --------
    pandas.Series or pandas.DataFrame with column(s) from
    right aligned with the left.

    """

    # ensure that we can align correctly
    if not right.index.is_unique:
        raise ValueError("The right's index must be unique!")

    # simpler case:
    # if the left is a single series then just re-index
    # this is the old way we were doing this
    if isinstance(left_fk, str):
        left = left[left_fk]

    if isinstance(left, pd.Series):
        a = right.reindex(left)
        a.index = left.index
        return a

    # when broadcasting multiple columns
    # i.e. the right has a multindex
    # if a series for the right provided, convert to a data frame
    if isinstance(right, pd.Series):
        right = right.to_frame('right')
        right_cols = 'right'
    else:
        right_cols = right.columns

    # do the merge
    return pd.merge(
        left=left,
        right=right,
        left_on=left_fk,
        right_index=True,
        how='left'
    )[right_cols]


def handle_nulls(s, drop_na=False, fill_value=0):
    """
    Handles null values in a series.

    Parameters:
    -----------
    s: pandas.Series
        Series to evaluate
    drop_na: bool, default False
        If true, null values are dropped. If false
        nulls are replaced by the fill_value
    fill_value: optional, default 0
        The value to fill nas with. Ignored if nulls
        are being dropped.

    Returns:
    -------
    pandas.Series

    """
    if drop_na:
        return s.dropna()
    else:
        return s.fillna(fill_value)


def rename_columns(df, prefix='', suffix='', cols=None):
    """
    Renames all columns in a table according to a given suffix and
    prefix. This is done in place and doesn't return anything.

    """
    if suffix or prefix:
        new_names = {}
        if cols is None:
            cols = df.columns
        for c in cols:
            new_names[c] = prefix + str(c) + suffix
        df.rename(columns=new_names, inplace=True)


def explode(df, amount_col, index_col='index', keep_index=False):
    """
    Given a dataframe with an capacity column,
    generate a single row for each available unit.
    The new data frame has a unique index with the old
    index preserved in the specified column.

    Parameters:
    -----------
    df: pandas.DataFrame
        Rows to explode.
    amount_col: str
        Column with amounts to explode/repeat.
    index_col: str, optional, default 'index'
        Name of column that will hold the previous
        index.
    keep_index: optional, defualt False
        If True, the original index is preserved.
        If False, the index is copied to a column.
    """
    candidates = df[df[amount_col] > 0]
    repeat_idx = candidates.index.repeat(candidates[amount_col].astype(int))
    exploded = candidates.reindex(repeat_idx)

    if not keep_index:
        exploded[index_col] = exploded.index.values
        exploded.reset_index(drop=True, inplace=True)

    return exploded


def get_2d_pivot(df, rows_col, cols_col, prefix='', suffix='', sum_col=None):
    """
    Returns simple 2-dimensional pivot table. Also takes care of the aggregation.
    Optionally does some renaming of the columns. Right now this
    only works on counts and sums. TODO: allow for all aggregates.

    Parameters:
    -----------
    df: pandas.DataFrame
        Data to pivot.
    rows_col: string
        Name of coumn to serve as the new index (i.e. rows).
    cols_col: string
        Name of column whose unique values will generate columns.
    prefix: string, optional
        String to prefix the column name.
    suffix: string, optional
        String to suffix the column name.
    sum_col: string, optional
        If provided, the column to sum on. Otherwise a count is produced.

    Returns:
    --------
    pandas.DataFrame with aggregated and pivoted counts.

    """

    if sum_col is None:
        piv = df.groupby([rows_col, cols_col]).size().unstack().fillna(0)
    else:
        piv = df.groupby([rows_col, cols_col])[sum_col].sum().unstack().fillna(0)

    rename_columns(piv, prefix, suffix)

    return piv


def hierarchy_aggregate(target_df, source_df, agg_col, segment_cols, agg_f='mean'):
    """
    Aggregates in a manner so that 0 values will not be be represented in
    the aggregate. Useful for things like medians and averages where want
    everything in our target to be represented by something, for example
    when adding buildings to zones to previously didn't have any.

    Typically used for aggregating by a set of nested locations,
    for example aggregating median household income to: TAZ-->RAZ-->MPA.

    Parameters:
    -----------
    target_df: pandas.DataFrame
        Data frame to aggregate values to.
    source_df: pandas.DataFrame
        Data frame to aggregate values from.
    agg_col: string
        Name of column in source data frame to aggregate.
    segment_cols: list of strings
        Columns to segment by:
            - The ordering is from left to right (e.g. TAZ, MPA, RAZ).
            - All of the columns should be present in the target and source,
              EXCEPT the first item. This assumes the first item is NOT in the
              target and instead points to the target index.
    agg_f: string or dictionary or ?
        Bascically anything that can be passed into pandas aggregate method.

    Returns:
    --------
    pandas.Series of aggregated results, aligned to the target_df data frame.

    """

    # initialize a series of 0s, indexed to the target
    result = pd.Series(np.zeros(len(target_df)), index=target_df.index)

    num_segs = len(segment_cols)
    for i in range(0, num_segs + 1):

        if i == num_segs:
            # no more segments left, just take the global aggregate
            seg = pd.Series(np.ones(len(target_df)), index=target_df.index)
            result[result == 0] = source_df.groupby(seg)[agg_col].agg(agg_f)
            break

        # perform the aggregation
        seg = segment_cols[i]
        curr_agg = source_df.groupby(seg)[agg_col].agg(agg_f)

        # reindex to the target
        if i == 0:
            curr_agg = curr_agg.reindex(target_df.index).fillna(0)
        else:
            curr_agg = broadcast(curr_agg, target_df[seg]).fillna(0)

        # update the result
        is_null = result == 0
        result[is_null] = curr_agg[is_null]
        if not is_null.any():
            break

    return result


def hierarchy_aggregate_groups(target_df, source_df, agg_col, segment_cols, group_col, agg_f='mean'):
    """
    Extends hierarchy_aggregate to the case where want to do an additional grouping on the
    source, for example, grouping by location segments (TAZ, RAZ, MPA) and building type.

    Parameters:
    -----------
    target_df: pandas.DataFrame
        Data frame to aggregate values to.
    source_df: pandas.DataFrame
        Data frame to aggregate values from.
    agg_col: string
        Name of column in source data frame to aggregate.
    segment_cols: list of strings
        Columns to segment by:
            - The ordering is from left to right (e.g. TAZ, MPA, RAZ).
            - All of the columns should be present in the target and source,
              EXCEPT the first item. This assumes the first item is NOT in the
              target and instead points to the target index.
    group_col: string
        Name of column in source to do additional grouping with.
    agg_f: string or dictionary or ?
        Bascically anything that can be passed into pandas aggregate method.

    Returns:
    --------
    pandas.DataFrame of aggregated results. Columns are added for the segmentation,
    grouping and aggregatation columns. A column is also added to identify the
    level of segmentation applied.

    """

    # get unique values for the groups
    group_vals = source_df[group_col].unique()

    # create a cross tab dataframe that has all combinations of the target index
    # and the group values in the source
    new_df = target_df[segment_cols[1:]].reindex(target_df.index.repeat(len(group_vals)))
    new_df.index.name = segment_cols[0]
    new_df[group_col] = np.tile(group_vals, len(target_df))
    new_df.reset_index(inplace=True)
    new_df['segment'] = pd.Series()

    num_segs = len(segment_cols)
    null_rows = None
    for i in range(0, num_segs + 1):

        # get the segmentation scheme
        if i == num_segs:
            # no more segments left, just use the group
            seg = [group_col]
        else:
            # combine the group and the current segment
            seg = [segment_cols[i], group_col]

        # do the aggregation
        curr_agg = source_df.groupby(seg).aggregate(agg_f)

        if i == 0:
            # 1st iteration, take all the results
            m = pd.merge(
                left=new_df,
                right=pd.DataFrame(curr_agg),
                left_on=seg,
                right_index=True,
                how='left'
            )
            new_df[agg_col] = m[agg_col].fillna(0)
            new_df['segment'] = str(seg)
        else:
            # assign current level to remaining nulls
            null_rows_merge = pd.merge(
                left=null_rows[seg],
                right=pd.DataFrame(curr_agg),
                left_on=seg,
                right_index=True,
                how='left'
            )
            new_df[agg_col].loc[null_rows.index] = null_rows_merge[agg_col].fillna(0)
            new_df['segment'].loc[null_rows.index] = str(seg)

        # get remaining nulls and move on if we're done
        null_rows = new_df[new_df[agg_col] == 0]
        if len(null_rows) == 0:
            break

    return new_df


def hierarchy_pivot(target_df, source_df, agg_col, segment_cols, group_col, agg_f='mean',
                    prefix='', suffix=''):
    """
    Contructs both a hierarchical aggregation and pivots so that the results
    are aligned with the target data frame and the aggregated values coming from the
    groups are columns. Optionally, renames the columns.

    """

    # do the aggregation
    ha = hierarchy_aggregate_groups(
        target_df,
        source_df,
        agg_col,
        segment_cols,
        group_col,
        agg_f)

    # pivot the results
    piv = ha.pivot(segment_cols[0], group_col, agg_col)

    # rename if desired
    rename_columns(piv, prefix, suffix)

    return piv


def stochastic_round(unrounded):
    """
    Randomly rounds using the fractional component as
    a weight.

    Parameters:
    -----------
    unrounded: pandas.Series
        Floating point series to be rounded.

    Returns:
    --------
    pandas.Series with rounded values.

    """

    # round everything down
    f = np.floor(unrounded)

    # get the fractional component
    fract = unrounded - f

    # round up if the fraction is bigger than a random
    ran = np.random.rand(len(unrounded))
    to_round_up = fract > ran
    f[to_round_up] += 1
    return f


def categorize(series, breaks, labels=None, break_adj=0):
    """
    Given a qualitative series, and a set of breaks,
    returns the group the row falls within.

    Parameters:
    -----------
    series: pandas.DataFrame
        Series with numeric values.
    breaks: list or arraylike
        Breaks to apply.
            - Breaks must be in increasing order.
            - Each break (except the 1st) marks the upper bound for that category.
            - Use np.nan in the 1st or last break to indicate
                unbound breaks.
    labels: list, optional default None
        Categories to assign. Should be 1 less than the number
        of breaks. If not provided, the breaks will be used as the
        labels, excluding the final break value.
    break_adj: numeric, optional, default 0
        Used to define how to classify rows at breaks. Positive values
        will ... (SCOTT add a better descriptions here)

    Returns:
    --------
    pandas.Series of categorized values. Rows falling
    outside the break defintions will have nulls.

    """

    # handle undefined lower bounds, undefined upper is handled by cut
    if np.isnan(breaks[0]):
        breaks = list(breaks)
        s_min = series.min()
        if s_min < breaks[1]:
            breaks[0] = s_min

    # if no labels are provided, use all but the last break value
    if labels is None:
        labels = breaks[:-1]

    # get a categorical and convert this to a series
    return pd.Series(
        pd.cut(
            series + break_adj,
            bins=breaks,
            labels=labels,
            include_lowest=True
        ),
        index=series.index
    )
