"""
Contains IO operations for SQL Server.

"""

import pandas.io.sql as pd_sql
import pyodbc


def get_db_connection(server, db):
    """
    Utility method to get a database connection.

    Parameters:
    -----------
    server: string
        Name of the SQL Server instance.
    db: string:
        Name of the database.

    Returns:
    --------
    Connection

    """
    conn_string = """DRIVER={SQL Server};
                     SERVER={};
                     DATABASE={};
                     Trusted_Connection=yes"""

    return pyodbc.connect(conn_string.format(server, db))


# use this function to load a data frame from sql
def sql_to_pandas(query, server, db, index_fld=None):
    """
    Queries and loads data from a SQL Server table into a data frame.

    Parameters:
    -----------
    query: string
        SQL query to apply.
    server: string
        Name of the SQL Server instance.
    db: string:
        Name of the database.
    index_fld: string, optional, default None
        Name of field to populate the data frame index.

    Returns:
    --------
    pandas.DataFrame

    """
    conn = get_db_connection(server, db)
    if index_fld is None:
        df = pd_sql.read_frame(query, conn)
    else:
        df = pd_sql.read_frame(query, conn, index_fld)
    conn.close()
    return df
