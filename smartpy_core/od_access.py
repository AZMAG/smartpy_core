"""
Use for acessibility queries with travel times represented
in an OD matrix (i.e. SKIMS). The matrix consists
of an origin zone, a destination zone, and one or more impedances
(e.g. AM travel time) representing different measures of access.

"""

import pandas as pd


class OdAccess(object):
    """
    ...

    """

    def __init__(self, od_df, from_col, to_col):

        self.od_df = od_df
        self.from_col = from_col
        self.to_col = to_col

        # caches??

    def __call__(self, within, zonal_counts, impedance_col, to=True,
                 agg_func=None, **agg_args):
        """

        Parameters:
        ----------
        within: numeric
            Maximum impedance to consider, usually expressed
            in time (minutes).
        zonal_counts: pandas.DataFrame
            Data frame containing zonal opportunities, e.g. number of jobs.
            Should be indexed by zone id.
        impedance_col: string
           Series in OD matrix indicating the impedance, e.g. 'time'.
        to: bool, optional, default True
            Indicates direction for query. True indicates movements towards
            the zone, False indicates movements away from (leaving)
            the zone.
        to: bool, optional, default True
            Indicates direction for query. True indicates movements towards
            the zone, False indicates movements away from (leaving)
            the zone.
        agg_func: optional, default None
            If provided then this will be used to aggregate results. If None then
            counts will be generated for all columns in the data frame.

        """

        # determine the direction of travel
        if to:
            # towards the zone
            # indicates how easy it is for agents in other zones
            # to reach this zone
            join_col = self.from_col
            grp_col = self.to_col

        else:
            # away from the zone
            # indicates how easy it is for agents in the zone to reach
            # opportunities in other zones
            join_col = self.to_col
            grp_col = self.from_col

        # get od rows within the given time
        is_within = self.od_df[impedance_col] <= within

        # join to opportunities
        a = pd.merge(
            self.od_df[is_within],
            zonal_counts,
            left_on=join_col,
            right_index=True
        )

        # perform the aggregation
        grps = a.groupby(grp_col)
        if agg_func:
            if isinstance(agg_func, str):
                aggs = grps.aggregate(agg_func)
            else:
                agg_args['impedance_col'] = impedance_col
                agg_args['within'] = within
                aggs = grps.apply(agg_func, **agg_args)
        else:
            aggs = grps[zonal_counts.columns].sum()

        return aggs.reindex(zonal_counts.index)
