"""
This is basically poached from

urbansim.scripts.cache_to_hdf5.py

But it makes it easier to load a few tables into a notebook without
having to copy/convert the entire cache into a h5 file.

"""


import glob
import os

import numpy as np
import pandas as pd


def opus_cache_to_df(dir_path):
    """
    Convert a directory of binary array data files to a Pandas DataFrame.

    The typical usage is to load in legacy opus cache data.

    Parameters
    ----------
    dir_path : str
        Full path to the table directory.

    Returns:
    --------
    pandas.DataFrame

    """
    table = {}
    for attrib in glob.glob(os.path.join(dir_path, '*')):
        attrib_name, attrib_ext = os.path.splitext(os.path.basename(attrib))
        if attrib_ext == '.lf8':
            attrib_data = np.fromfile(attrib, np.float64)
            table[attrib_name] = attrib_data

        elif attrib_ext == '.lf4':
            attrib_data = np.fromfile(attrib, np.float32)
            table[attrib_name] = attrib_data

        elif attrib_ext == '.li2':
            attrib_data = np.fromfile(attrib, np.int16)
            table[attrib_name] = attrib_data

        elif attrib_ext == '.li4':
            attrib_data = np.fromfile(attrib, np.int32)
            table[attrib_name] = attrib_data

        elif attrib_ext == '.li8':
            attrib_data = np.fromfile(attrib, np.int64)
            table[attrib_name] = attrib_data

        elif attrib_ext == '.ib1':
            attrib_data = np.fromfile(attrib, np.bool_)
            table[attrib_name] = attrib_data

        elif attrib_ext.startswith('.iS'):
            length_string = int(attrib_ext[3:])
            attrib_data = np.fromfile(attrib, ('a' + str(length_string)))
            table[attrib_name] = attrib_data

        else:
            print('Array {} is not a recognized data type'.format(attrib))

    df = pd.DataFrame(table)
    return df
