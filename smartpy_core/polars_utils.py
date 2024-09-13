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


def stochastic_round(s: pl.Series, out_dtype:pl.DataType=pl.Int64) -> pl.Series:
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
        return self._ldf.with_columns(self.rand(name))

    def sel(self, *args, **kwargs):
        """
        Custom select that...

        """
        pass


#######################
# sampling methods 
#######################


def segmented_sample_no_replace(
        df: pl.LazyFrame, 
        counts: pl.LazyFrame,
        counts_col: str,
        segment_col: str | list[str], 
        weights_col:str=None) -> pl.LazyFrame:
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
    # randomize
    ran_col = '__r'
    df = df.smart.with_rand(ran_col)
    if weights_col is not None:
       df = df.with_columns(
          ran_col=pl.col(ran_col) ** (1 / pl.col(weights_col))
        )
    
    # get the sample by taking the top n rows from each segment
    rank_col = '__rank'
    return (
        df
        .join(counts, on=segment_col, how='left')
        .with_columns(pl.col(ran_col).rank().over(segment_col).alias(rank_col))
        .filter(pl.col(rank_col) <= pl.col(counts_col))
        .select(pl.exclude(counts_col, ran_col, rank_col))
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

    # explode so we have a random for each count item
    idx_col = '__idx'
    repeat_col = '__repeat'
    rand_col = '__r'
    counts_explode = (
        counts
        .with_row_index(idx_col)
        .select(
            pl.all(),
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
