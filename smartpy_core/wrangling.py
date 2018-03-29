"""
Contains some utility functions for doing general
data wrangling operations with Pandas and Numpy.

"""

import numpy as np
import pandas as pd


def broadcast(right, left, left_fk=None, right_pk=None, keep_right_index=False):
    """
    Re-indexes a series or data frame (right) to align with
    another (left) series or data frame via foreign key relationship.
    The index or keys on the right must be unique (i.e. this only supports
    1:1 or 1:m relationhips between the right and left).

    Parameters:
    -----------
    right: pandas.DataFrame or pandas.Series
        Columns or set of columns to re-project(broadcast) from.
    left: pandas.Series, pandas.Index or pandas.DataFrame
        Object to align to.
            if panadas.Series:
                Series values are used as the foreign keys.
            if pandas.Index:
                The index will be used as the foreign keys.
            if pandas.DataFrame
                Use the 'left_fk` argument to specify one
                or more columns to serve as the foreign keys.
    left_fk: str or list of str
        Only applicable if 'left' is a dataframe.
        Column or list of columns to serve as foreign keys.
        If not provided the `left's` index will be used.
    right_pk: str or list of str, default None
        Column or list of columns that uniquely define each row
        in the the `right`. If not provided, the `right's` index will be
        used.
    keep_right_index: bool, optional, default False
        If True, and the `right` is a data frame, and a `right_pk` arg is provided,
        then column(s) containing the `right's` index values will be
        appended to the result.

    Returns:
    --------
    pandas.Series or pandas.DataFrame with column(s) from
    right aligned with the left.

    """
    update_index = True

    # if right primary keys are explicitly provided
    if right_pk:
        if keep_right_index:
            right = right.reset_index()
            right.set_index(right_pk, inplace=True)
        else:
            right = right.set_index(right_pk)

    # ensure that we can align correctly
    if not right.index.is_unique:
        raise ValueError("The right's index must be unique!")

    # decide how to broadcast based on the type of left provided
    if isinstance(left, pd.Index):
        update_index = False

    # for cases where a left_fk is provided as a list with a single element
    if left_fk:
        if isinstance(left_fk, list):
            if len(left_fk) == 1:
                left_fk = left_fk[0]

    if isinstance(left, pd.DataFrame):
        if left_fk:
            left = left[left_fk]
        else:
            left = left.index
            update_index = False

    # reindex
    a = right.reindex(left)

    # update the index if necessary
    if update_index:
        a.index = left.index.copy()

    return a


def fill_nulls(data, fill_value=0, inplace=False):
    """
    Fill nans and inf in 1 shot.

    Parameters:
    -----------
    data: pandas.DataFrame or panda.Series
        The data containing values to fill.
    fill_value: optional, defualt 0
        The value to fill nulls with.
    inplace: bool, optional, default False
        If True, the data is modified in place, nothing is returned.
        If False, a copy of the data frame with filled values is returned.

    """
    if inplace:
        data.replace([np.inf, -np.inf, np.nan], fill_value, inplace=True)
    else:
        return data.replace([np.inf, -np.inf, np.nan], fill_value)


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
        return fill_nulls(s, fill_value)


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


def get_2d_pivot(df, rows_col, cols_col, prefix='', suffix='', sum_col=None, agg_f='sum'):
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
        piv = df.groupby([rows_col, cols_col])[sum_col].agg(agg_f).unstack().fillna(0)

    rename_columns(piv, prefix, suffix)

    return piv


def impute(df, cols, segment_cols, min_size=1, size_col=None, agg_f='mean'):
    """
    Imputes missing (null, nan) values in a data frame.

    Parameters:
    -----------
    df: pandas.DataFrame
        Data frame to impute.
    cols: str or list of str
        Columns in the data frame to impute. If multiple columns are provided,
        ALL the provided columns will be imputed if ANY of them  have nulls.
    segment_cols: str or list of str
        Columns that will guide the imputation. Items should be ordered to refelct
        decreasing detail.
    min_size: int, optional, default 1
        Threshold for determining if a given level of aggregation is sufficient for
        imputing.
    size_col: str, optional defualt None
        If provided, the given columns sums are checked against the minimum threshold.
        If not provided, row counts are used to check against the minimum threshold
    agg_f: str or callable, optional, default `mean`
        The aggregate function to use for imputing.

    Returns:
    --------
    pandas.DataFrame containing the imputed columns,
    pandas.Index of rows that have been updated

    """
    if isinstance(cols, str):
        cols = [cols]

    if isinstance(segment_cols, str):
        segment_cols = [segment_cols]

    # get imputed values for rows with nulls
    impute_rows = df[df[cols].isnull().any(axis=1)]
    impute_grps = impute_rows.groupby(segment_cols).size().to_frame('cnt')
    impute_agg = agg_to(
        df, impute_grps, cols, segment_cols, min_size, size_col, agg_f)
    impute_agg_broad = broadcast(impute_agg, impute_rows, segment_cols)

    # assign
    results = df[cols].copy()
    results.loc[impute_rows.index, cols] = impute_agg_broad
    return results, impute_agg_broad.index


def agg_to(from_df, to_df, val_cols, segment_cols, min_size=0, size_col=None, agg_f='mean'):
    """
    Aggregates a data frame to align with another, while meeting minimum size
    thresholds. The aggregations are applied iteratively, checking the row counts or sums
    against the threshold at each iteration: only rows meeting the threshold will be
    assigned the aggregation at that iteration. Each subsequent iteration removes the left-most
    segment from the segment list. Rows not meeting the minimum threshold at any
    aggregation level will be assigned the global values.

    Parameters:
    -----------
    from_df: pandas.DataFrame
        Data frame aggregating from.
    to_df: pandas.DataFrame
        Data frame aggregating to.
    val_cols: str or list of str
        List of columns in the `from_df` to aggregate.
    segment_cols: str or list of str
        Group by columns. Should be ordered from more to less detail.
    min_size: int, optional, default 0
        Defines the threshold a given aggregation must meet.
    size_col: str, optional default None
        If provided, uses a column (and the resulting sums) check against the threshold.
        If not provided, row counts are used to check against the threshold.
    agg_f: str or callable, optional, default `mean`
        The aggregate function to apply.

    Returns:
    --------
    pandas.DataFrame

    """

    if isinstance(val_cols, str):
        val_cols = [val_cols]

    if isinstance(segment_cols, str):
        segment_cols = [segment_cols]

    # init results with nans
    results = pd.DataFrame(
        columns={x: [] for x in val_cols},
        index=to_df.index.copy()
    )

    # remove rows with nulls
    from_cols = val_cols + segment_cols
    if size_col:
        from_cols.append(size_col)
    from_df = from_df[from_cols].dropna(subset=val_cols)

    # get columns for broacasting/reindexing the agg results back to the to_df
    # assume that if a given column is not in the table, it exists as a level in the index
    reidx_by = []
    for c in segment_cols:
        if c in to_df.columns:
            reidx_by.append(to_df[c])
        elif c in to_df.index.names:
            reidx_by.append(to_df.index.get_level_values(c))
        else:
            raise ValueError('Column {}, is missing from the to dataframe'.format(c))

    # iteratively aggregate, each time with one less segment, until all thresholds are met
    for i in range(0, len(segment_cols) + 1):

        # identify rows still needing a value
        is_null = results.isnull().any(axis=1)

        # exit if every row has been assigned
        if not is_null.any():
            break

        # if we've exhausted all segments, take global values
        if i == len(segment_cols):
            for c in val_cols:
                results.loc[is_null, c] = from_df[c].agg(agg_f)
            break

        # get current aggregation
        curr_segs = segment_cols[i:]
        curr_reidx = reidx_by[i:]
        if len(curr_reidx) == 1:
            # for some reason, can't reindex with a list if the list has a single item
            curr_reidx = curr_reidx[0]

        curr_grps = from_df.groupby(curr_segs)
        curr_agg = curr_grps[val_cols].agg(agg_f)
        curr_agg = curr_agg.reindex(curr_reidx)
        curr_agg.index = to_df.index

        # assign aggregate values where the minimum thresholds are met
        if min_size == 0:
            return curr_agg

        if size_col:
            # compare threshold with sums from a column
            curr_sizes = curr_grps[size_col].sum()
        else:
            # compare thresholds with record counts
            curr_sizes = curr_grps.size()

        curr_sizes = curr_sizes.reindex(curr_reidx)
        curr_sizes.index = to_df.index
        curr_sizes.fillna(0, inplace=True)

        gt_thresh = curr_sizes >= min_size
        to_assign = is_null & gt_thresh
        results[to_assign] = curr_agg[to_assign]

    return results


def hierarchy_aggregate(target_df, source_df, agg_col, segment_cols, agg_f='mean'):
    """

    OLD -- use `to_agg` method above instead

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
            # seg = pd.Series(np.ones(len(target_df)), index=target_df.index)
            # result[result == 0] = source_df.groupby(seg)[agg_col].agg(agg_f)
            result[result == 0] = source_df[agg_col].agg(agg_f)
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


def location_quotients(df, level=0):
    """
    Returns location quotients.

    Parameters:
    -----------
    df: pandas.DataFrame
        Data to compute location quotients for. Will compute a quotient
        for each row and column
    level: int, optional, default 0
        Only applicable if the data frame has multi-level columns.
        Defines the level of columns to use, each group
        within the level will be computed independently.

    Returns:
    --------
    pandas.DataFrame

    """

    col_sums = df.sum()

    if isinstance(df.columns, pd.MultiIndex):
        col_shares = col_sums.divide(col_sums.groupby(level=level).sum())
        row_shares = df.divide(df.groupby(level=level, axis=1).sum())
        return row_shares / col_shares
    else:
        return df.divide(df.sum(axis=1), axis=0).divide(col_sums / col_sums.sum())


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
