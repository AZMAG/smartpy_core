"""
This module (will) contain methods and utilities for working w/ GIS files and
functions totally indendent of ESRI/arcpy. Instead will rely on a combination
of GDAL, Fiona and Geopandas.

"""

import pandas as pd
import fiona
import geopandas as gpd


def open_gdb_table(gdb, table, flds=None, index_fld=None):
    """
    Opens a table within a file geodatadase and retuns a
    pandas.DataFrame.

    Parameters:
    -----------
    gdb: str
        Full path to the gdb file
    table: str
        Name of the table w/in the file gdb
    flds: str or dict
        Fields to pull. If a dict is provided
        they will also be re-named.
    index_fld: str, optional, default None
        Name of the column to serve as the index.

    Returns:
    --------
    pandas.DataFrame

    """

    with fiona.open(gdb, layer=table) as t:
        df = pd.DataFrame([row['properties'] for row in t])

    if index_fld is not None:
        df.set_index(index_fld, inplace=True)

    return df
