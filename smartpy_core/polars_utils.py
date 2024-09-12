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