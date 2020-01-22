"""
Contains IO operations for SQL Server.

"""

import pandas as pd
import pyodbc
import sqlalchemy

try:
    from urllib.parse import quote_plus # py3
except ImportError:
    from urllib import quote_plus # py27


def get_db_connection(server, db, driver='SQL Server'):
    """
    Utility method to get a database connection.

    Parameters:
    -----------
    server: string
        Name of the SQL Server instance.
    db: string:
        Name of the database.
    driver: string, optional defualt SQL Server
        Driver for connection.

    Returns:
    --------
    Connection

    """
    conn_string = """DRIVER={};
                     SERVER={};
                     DATABASE={};
                     Trusted_Connection=yes"""

    return pyodbc.connect(conn_string.format(driver, server, db))


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
    df = pd.read_sql(query, conn, index_fld)
    conn.close()
    return df


def pandas_to_sql(df, server, db, table, index=True, index_label=None, if_exists='fail', chunksize=50000):
    """
    Writes data from a pandas.DataFrame to sql server table.

    Parameters:
    -----------
    df: pandas.DataFrame
        Data to write out.
    server: str
        Name of sql server instance.
    db:
        Name of the sql server database.
    table: str
        Name of the output table.
    index: bool, default True
        Whether or not to write out the index.
    index_label: str, default None
        Name to apply to index. If not provided, index names will be used.
    if_exists: str, {'fail', 'replace', 'append'}, default 'append'
        What to do if the table already exists.
        - fail: Raise a ValueError.
        - replace: Drop the table before inserting new values.
        - append: Insert new values to the existing table.
    chunksize: int, default 50000
        Number of rows to insert at a given time.

    """
    # get the sql connection
    db_para = 'DRIVER={SQL SERVER};SERVER=' + server + ';DATABASE=' + db + ';Trusted_Connection=yes'
    conn_string = quote_plus(db_para)
    engine = sqlalchemy.create_engine(
        'mssql+pyodbc:///?odbc_connect={}'.format(conn_string)
    )

    @sqlalchemy.event.listens_for(engine, "before_cursor_execute")
    def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
        """
        The presence of this method allows faster inserts.

        """
        if executemany:
            cursor.fast_executemany = True

    # for str columns, determine the output field lengths
    out_types = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            max_chars = df[col].str.len().max()
            out_types[col] = sqlalchemy.VARCHAR(int(max_chars))

    # write the results
    df.to_sql(
        table,
        engine,
        if_exists=if_exists,
        index=index,
        index_label=index_label,
        chunksize=chunksize,
        dtype=out_types,
    )
