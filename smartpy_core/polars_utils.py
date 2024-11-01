"""
Utilities for working w/ polars.

"""
import numpy as np
import pandas as pd
import polars as pl


#########################
# funcs for map batches
#########################


def np_rand(s: pl.Series) -> pl.Series:
    """ 
    Wrapper for `numpy.random.rand`, for
    use with `pl.Expr.map_batches`.

    Sample usage:
    -------------
    df = pl.DataFrame({'a': [1, 2, 3]})
    df.select(pl.first().map_batches(np_rand))

    """
    return pl.Series(
        values=np.random.rand(s.len()),
        dtype=pl.Float64
    )


def stochastic_round(s: pl.Series, out_dtype: pl.DataType=pl.Int64) -> pl.Series:
    """
    Rounds a float randomly proportional to the remainder.

    Sample usage:
    -------------
    df = pl.DataFrame({'a': [0.3, 2.1, -10.8]})
    df.select(pl.col('a').map_batches(stochastic_round))

    """
    r = pl.Series(np.random.rand(s.len()))
    frac = s % 1
    return (s.floor() + (frac > r).cast(pl.Int8)).cast(out_dtype)


#########################
# polars api extensions
#########################


@pl.api.register_expr_namespace('smart')
class ExprUtils(object):
    """
    Custom functions that work off expressions. 
    Registered under the `smart` namespace.

    """
    def __init__(self, expr):
        self._expr = expr

    def probs(self):
        """
        Returns column-level probabilities for the provided 
        expression. 

        Sample usage:
        -------------
        df = pl.DataFrame({'a': [1, 2, 3], 'grp': ['a', 'a', 'b']})
        df.select(pl.col('a').smart.probs())

        # to generate probabilities by segments/groups
        df.select(pl.col('a').smart.probs().over('grp'))

        """
        return self._expr / self._expr.sum()
    
    def rand(self) -> pl.Expr:
        """
        Wrapper for numpy.random.rand.

        Sample usage:
        -------------
        df = pl.DataFrame({'a': [1, 2, 3]})
        df.select(pl.first().smart.rand())

        """
        return self._expr.map_batches(np_rand)

    def stochastic_round(self) -> pl.Expr:
        """
        Rounds a float randomly proportional to the remainder.

        Sample usage:
        -------------
        df = pl.DataFrame({'a': [0.3, 2.1, -10.8]})
        df.select(pl.col('a').smart.stochastic_round)

        """
        return self._expr.map_batches(stochastic_round)


@pl.api.register_lazyframe_namespace('smart')
class LazyFrameUtils(object):
    """
    Custom operations that operate on a polars lazy frame.
    Registered under the `smart` namespace.
    
    """
    def __init__(self, ldf: pl.LazyFrame):
        self._ldf = ldf

    def rand(self, name: str=None) -> pl.Series:
        """
        Returns a series containing values from `numpy.random.rand`.

        """
        if name is None:
            name = 'rand'
        return pl.first().smart.rand().alias(name)

    def with_rand(self, name: str=None) -> pl.DataFrame:
        """
        Returns lazy frame appended with a column of `numpy.rand.rand`
        values. 

        Parameters:
        -----------
        str: name
            Name to assign the column.

        """
        return self._ldf.with_columns(self.rand(name))

    def shuffle(self, w_col: str=None) -> pl.LazyFrame:
        """
        Shuffles a lazy frame, optionally using weights.
    
        Parameters:
        -----------
        w_col: str, optional, default None
            Name of the weights column.

        """
        return shuffle(self._ldf, w_col)

    def sel(self, *args, **kwargs):
        """
        Custom select that...

        """
        pass


@pl.api.register_dataframe_namespace('smart')
class DataFrameUtils(object):
    """
    Custom operations that operate on a polars data frame.
    Registered under the `smart` namespace.
    
    """
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def rand(self, name: str=None) -> pl.Series:
        """
        Returns a series containing values from `numpy.random.rand`.

        """
        if name is None:
            name = 'rand'
        return pl.first().smart.rand().alias(name)

    def with_rand(self, name: str=None) -> pl.DataFrame:
        """
        Returns lazy frame appeneded with a column of `numpy.rand.rand`
        values. 

        Parameters:
        -----------
        str: name
            Name to assign the column.

        """
        return self._df.with_columns(self.rand(name))
    
    def shuffle(self, w_col: str=None) -> pl.DataFrame:
        """
        Shuffles a data frame, optionally using weights.
    
        Parameters:
        -----------
        w_col: str, optional, default None
            Name of the weights column.

        """
        return shuffle(self._df, w_col)


#######################
# sampling methods 
#######################


def shuffle(df: pl.LazyFrame | pl.DataFrame, w_col: str=None, rand_col: str=None) -> pl.LazyFrame | pl.DataFrame:
    """
    Shuffles a frame, optionally using weights.
    
    Parameters:
    -----------
    df: pl.LazyFrame or pl.DataFrame
        The data frame to shuffle.
    w_col: str
        Name of the weights column.
    rand_col: str, optional, default None
        If provided, names the temporary random column
        and retains it in the results.
        If None, the random colum is dropped in the returned result
        
    """
    if rand_col is None:
        rand_col = '__r'
        drop_rand = True
    else:
        drop_rand = False

    df = df.smart.with_rand(rand_col)
    if w_col is not None:
        df = df.with_columns(
           (pl.col(rand_col) ** (1 / pl.col(w_col))).alias(rand_col)
        )
    
    df = df.sort(rand_col, descending=True)
    if drop_rand:
        df = df.select(pl.exclude(rand_col))
    return df


def segmented_sample(
        df: pl.LazyFrame | pl.DataFrame, 
        counts: pl.LazyFrame | pl.DataFrame,
        counts_col: str,
        segment_col: str | list[str], 
        replace: bool,
        weights_col: str=None) -> pl.LazyFrame | pl.DataFrame:
    """
    Segmented sampling.

    Parameters:
    -----------
    df:  pl.LazyFrame or pl.DataFrame
        Data frame to sample from.
    counts: pl.LazyFrame or pl.DataFrame
        Data frame containing segments and counts.
        Should match frame type (data or lazy) of the sample frame.
    counts_col: str
        Column in the counts data frame containing
        the counts.
    segment_col: str or list of str
        Column(s) containing the segmentation. These
        columns must exist in both data frames and 
        the combination of column values should be unique.
    replace: bool
        If True, samples WITH replacement.
        If False, samples WITHOUT replacement
    weights_col: str, optional, default None
        If provided, column to serve as weights. 

    Returns:
    --------
    pl.LazyFrame or pl.DataFrame

    """
    if replace:
        return segmented_sample_with_replace(
            df, counts, counts_col, segment_col, weights_col
        )
    else:
        return segmented_sample_no_replace(
            df, counts, counts_col, segment_col, weights_col
        )


def segmented_sample_no_replace(
        df: pl.LazyFrame, 
        counts: pl.LazyFrame,
        counts_col: str,
        segment_col: str | list[str], 
        weights_col: str=None) -> pl.LazyFrame:
    """
    Segmented sampling without replacement.

    Parameters:
    -----------
    df: pl.LazyFrame
        Data frame to sample from.
    counts: pl.LazyFrame
        Data frame containing segments and counts.
    counts_col: str
        Column in the counts data frame containing
        the counts.
    segment_col: str or list of str
        Column(s) containing the segmentation. These
        columns must exist in both data frames and 
        the combination of column values should be unique.
    weights_col: str, optional, default None
        If provided, column to serve as weights. 

    """
    # retain the original schema
    df_cols = df.collect_schema().names()

    # randomize
    df = shuffle(df, weights_col, '__r')
    
    # get the sample by taking the top n rows from each segment
    return (
        df
        .join(counts, on=segment_col, how='left')
        .with_columns(__rank=pl.col('__r').cum_count().over(segment_col))
        .filter(pl.col('__rank') <= pl.col(counts_col))
        .select(df_cols)
    )


def segmented_sample_with_replace(
        df: pl.LazyFrame, 
        counts: pl.LazyFrame,
        counts_col: str,
        segment_col: str | list[str], 
        weights_col:str=None) -> pl.LazyFrame:
    """
    Segmented sampling with replacement.

    Parameters:
    -----------
    df: pl.LazyFrame
        Data frame to sample from.
    counts: pl.LazyFrame
        Data frame containing segments and counts.
    counts_col: str
        Column in the counts data frame containing
        the counts.
    segment_col: str or list of str
        Column(s) containing the segmentation. These
        columns must exist in both data frames and 
        the combination of column values should be unique.
    weights_col: str, optional, default None
        If provided, column to serve as weights. 

    """
    # columns we'll retain in the result
    cols = df.collect_schema().names()

    # work off segments as a list
    if not isinstance(segment_col, list):
        segment_col = [segment_col]

    # explode so we have a random for each count item
    idx_col = '__idx'
    repeat_col = '__repeat'
    rand_col = '__r'
    counts_explode = (
        counts
        .select(counts_col, *segment_col)
        .with_row_index(idx_col)
        .with_columns(
             pl.col(idx_col).repeat_by(counts_col).alias(repeat_col)
        )
        .explode(repeat_col)
        .smart.with_rand(rand_col)
        .select(pl.exclude(idx_col, repeat_col, counts_col))
        .sort(rand_col)
    )

    # get sorted probabilities by segment
    if weights_col is None:
        weights_col = '__w'
        df = df.with_columns(pl.lit(1).alias(weights_col))
    cp = (
        df
        .with_columns(__p=pl.col(weights_col).smart.probs().over(segment_col))
        .with_columns(__cp=pl.col('__p').cum_sum().over(segment_col))
        .sort('__cp')
    )

    # sample by joining rands to cum prob bins
    # ...note: both dfs neeed to be sorted
    return (
        counts_explode
        .join_asof(
            cp,
            left_on=rand_col,
            right_on='__cp',
            by=segment_col,
            strategy='forward'
        )
        .select(cols)
    )


def segmented_sampling_with_accounting_replace(
        df: pl.LazyFrame, 
        amounts: pl.LazyFrame,
        accounting_col: str,
        amount_col: str,
        segment_col: str | list[str], 
        weights_col:str=None,
        max_iter: int=100,
        debug: bool=False) -> pl.LazyFrame:
    """
    Performs segmented sampling while matching
    prescribed accounting totals. Sample use case
    is sampling households while controlas are
    prescribed as population.

    **WITH REPLACEMENT**

    Parameters:
    -----------
    df: pl.LazyFrame
        Data frame to sample from.
    amounts: pl.LazyFrame
        Data frame containing segments and amount/counts
        i.e control totals.
    accounting_col: str
        Name of the column in sampling frame containing
        amounts, e.g. persons.
    counts_col: str
        Column in the amounts data frame containing
        the amounts to match.
    segment_col: str or list of str
        Column(s) containing the segmentation. These
        columns must exist in both data frames and 
        the combination of column values should be unique.
    weights_col: str, optional, default None
        If provided, column to serve as weights. 
    max_iter: int, optional, default 100
        Maximum number of sampling iterations. 
    debug: bool, optional, default False
        If True, prints a message of the whether or
        not all amounts were matched exactly and the 
        number of iterations it took. 
        
    """
    # retain the original schema
    df_cols = df.collect_schema().names()

    # estimate avg accounting amount per sample (e.g. persons per household)
    # ...we'll increase this by 5% to ensure we're sampling enough
    per_sample = (
        amounts
        .join(
            df.group_by(segment_col).agg(
                pl.col(accounting_col).sum().alias('__accounting_sum'),
                pl.col(accounting_col).len().alias('__row_cnt')
            ),
            on=segment_col,
            how='left'
        )
        .with_columns(
            __per_sample=(1.05) * (pl.col('__accounting_sum') / pl.col('__row_cnt'))
        )
    )

    # track the remaining amount needed
    remaining = per_sample.select(
        pl.col(amount_col).alias('__remaining'),
        pl.col('__per_sample'),
        *segment_col
    )

    # iteratively sample until we reach the control total
    to_concat = []
    done = False

    for i in range(0, max_iter):
        # break out of the loop if we've matched all amounts
        if done:
            print('finished at {} iterations'.format(i))
            break
        
        # number of samples to draw in this iteration
        num_samples = (
            remaining
            .with_columns(
                __num_samples = (pl.col('__remaining') / pl.col('__per_sample')).ceil()
            )
        )

        # get current sample
        curr_sample = (
            # get the sample
            segmented_sample(
                df,
                num_samples,
                '__num_samples',
                segment_col,
                True,
                weights_col
            )
            # keep those not exceeding the amount total
            .with_columns(_cs=pl.col(accounting_col).cum_sum().over(segment_col))
            .join(
                remaining.select('__remaining', *segment_col),
                on=segment_col
            )
            .filter(pl.col('_cs') <= pl.col('__remaining'))
        )

        if not curr_sample.is_empty():
            to_concat.append(curr_sample)
        
            # update remaining amounts
            remaining = (
                remaining
                .join(
                    curr_sample
                        .group_by(segment_col)
                        .agg(pl.col(accounting_col).sum().alias('__sampled_amount')
                    ),
                    on=segment_col, 
                    how='left'
                )
                .fill_null(0)
                .select(
                    (pl.col('__remaining') - pl.col('__sampled_amount')).alias('__remaining'),
                    pl.col('__per_sample'),
                    *segment_col
                )
                .filter(pl.col('__remaining') > 0)
            )
            
            # are we done yet?
            if remaining.is_empty():
                done = True
                break

    # finish up
    # TODO: convert this to logging
    if debug:
        if done:
            print('finished in {} iterations'.format(i))
        else:
            print('**did not match all amouns exactly**')
            print(remaining)

    return (
        pl.concat(to_concat, how='vertical_relaxed')
        .select(df_cols)
    )


def segmented_sampling_with_accounting_no_replace():
    """
    TODO...

    """
    pass


#####################
# transition
#####################


def get_starting_id(df: pl.LazyFrame | pl.DataFrame, col: str) -> int:
    """
    Returns a starting ID that ensures unique-ness when adding
    new rows.

    Parameters:
    -----------
    df: pl.DataFrame or pl.LazyFrame
        Frame to add to.
    col: str
        Name of ID column in the frame.

    Returns:
    --------
    int
    
    """
    if isinstance(df, pl.DataFrame):
        return 1 + df[col].max()
    elif isinstance(df, pl.LazyFrame):
        return 1 + df.select(pl.col(col).max()).collect().item()
    else:
        raise ValueError("'df' must be pl.LazyFrame or pl.DataFrame")


def transition_agents(targets,
                      target_col,
                      segment_cols,
                      agents, 
                      agent_id_col) -> pl.LazyFrame | pl.DataFrame:
    """
    In progress.

    Simple transtion model, so far doesn't handle accounting column
    or linked tables. 

    So suitable for emp but not hh pop.

    Also doesn't do the sampling threhold hierarchy stuff
    but we don't really use that.

    Also, what do we do with agents that fall outside the segmentation scheme in
    the controls
    ...by default in urbansim these are removed
    ...right now this would be keeping them

    """
    # starting ID for adding new agents
    starting_id = get_starting_id(agents, agent_id_col)

    # determine the amounts we need to add/remove
    amounts = (
        targets
        .join(
            agents.group_by(segment_cols).agg(__cnt=pl.len()),
            on=segment_cols,
            how='left',
        )
        .with_columns(__diff=pl.col(target_col) - pl.col('__cnt'))
    )

    # add agents where the controls are lower than the existing count
    adds = amounts.filter(pl.col('__diff') > 0)
    added_agents = (
        segmented_sample(
            agents, adds, '__diff', segment_cols,True
        )
        .rename({agent_id_col: 'src_{}'.format(agent_id_col)})
        .with_row_index(agent_id_col, starting_id)
    )
    
    # remove agents where the controls are higher than the existing count
    removals = (
        amounts
        .filter(pl.col('__diff') < 0)
        .with_columns(__diff=pl.col('__diff').abs())
    )
    removed_agents = segmented_sample(agents, removals, '__diff', segment_cols, False)

    # combine
    return pl.concat(
        [
            agents.join(removed_agents, on=agent_id_col, how='anti'),
            added_agents.select(agents.columns)
        ],
        how='vertical_relaxed'
    )
