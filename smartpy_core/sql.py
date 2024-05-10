"""
Contains IO operations for SQL Server.

Note: this was updated on 05.10.2024 to make greater use
of sqlalchemy engine and limit the use of pyodbc.connect. 

Some references: 
https://docs.sqlalchemy.org/en/13/core/connections.html

"""

import pandas as pd
import sqlalchemy

try:
    from urllib.parse import quote_plus # py3
except ImportError:
    from urllib import quote_plus # py27


# sql alchemy engines are intended to be application level
# ...so we'll maintain a global list
_engines = {}


def get_engine(server, db):
    """
    Returns a sql alchemy engine for sql server.

    Parameters:
    -----------
    server: string
        Name of the SQL Server instance.
    db: string:
        Name of the database.

    """
    key = (server, db)
    if key not in _engines:
        db_para = 'DRIVER={SQL SERVER};SERVER=' + server + ';DATABASE=' + db + ';Trusted_Connection=yes'
        conn_string = quote_plus(db_para)
        _engines[key] = sqlalchemy.create_engine('mssql+pyodbc:///?odbc_connect={}'.format(conn_string))
    return _engines[key]


class SqlServerConn(object):
    """
    *** DEPRECATED**
    Use 
        `sql_execute` method to execute sql statements.
        `get_records` method to query a recordset into a dict. 
        `sql_to_pandas` method to query into a datafrmae.
        `pandas_to_sql` method to write a dataframe to a sql table.
    """

    def __init__(self, server, db):
        raise Exception( 
            """
            *** DEPRECATED**
            Use 
                `sql_execute` method to execute sql statements.
                `get_records` method to query a recordset into a dict. 
                `sql_to_pandas` method to query into a datafrmae.
                `pandas_to_sql` method to write a dataframe to a sql table.
            """
        )


def sql_execute(statements, server, db):
    """
    Executes one or more sql statements.

    Parameters:
    -----------
    statements: str or list of str
        SQL statement(s) to execute.
    server: string
        Name of the SQL Server instance.
    db: string:
        Name of the database.
    
    Sample usage:
    -------------
    sql_execute(
        'select mpa, median_age into __blah_blah from _scott_temp_median_test_mpa', 
        'sql', 
        'ACS2022'
    )
    """
    engine = get_engine(server, db)
    if not isinstance(statements, list):
        statements = [statements]
    with engine.begin() as c:
        for s in statements:
            c.execute(s)


def get_records(query, server, db):
    """
    Returns a recordset for the provided query. 

    Parameters:
    -----------
    query: str
        SQL query to apply.
    server: string
        Name of the SQL Server instance.
    db: string:
        Name of the database.
        
    Returns:
    -------
    list of dictionaries, each dictionary 
    represents a row w/ the keys the column names.
    
    """
    engine = get_engine(server, db)
    with engine.connect() as c:
        return c.execute(query).mappings().all()


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
    engine = get_engine(server, db)
    return pd.read_sql(query, engine, index_fld)


def pandas_to_sql(df, server, db, table, index=True, index_label=None, if_exists='fail', chunksize=50000, dtypes=None):
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
    dtypes: dict, optional, default None
        Data types to override. Keys are column names, values are
        sql alchemy data types.
        see: https://docs.sqlalchemy.org/en/13/core/type_basics.html

    """
    engine = get_engine(server, db)

    @sqlalchemy.event.listens_for(engine, "before_cursor_execute")
    def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
        """
        The presence of this method allows faster inserts.

        """
        if executemany:
            cursor.fast_executemany = True

    # infer sql data types, respect any user provided dtypes
    if dtypes is not None:
        out_types = dtypes.copy()
    else:
        out_types = {}

    def get_int_fld_type(col):
        """
        Determines the sql int type based on the series value range.

        """
        min_val = df[col].min()
        max_val = df[col].max()

        if min_val >= -32767 and max_val <= 32767:
            return sqlalchemy.SMALLINT()
        elif min_val >= -2147483647 and max_val <= 2147483647:
            return sqlalchemy.INT()
        else:
            return sqlalchemy.BIGINT()

    # loop through the columns and assign types
    for col in df.columns:
        if col in out_types:
            continue

        curr_dtype = str(df[col].dtype)

        if curr_dtype == 'object':
            # for str cols figure out the max characters needed
            max_chars = df[col].str.len().max()
            out_types[col] = sqlalchemy.VARCHAR(int(max_chars))

        elif curr_dtype.startswith('float'):
            # check for cases where everything implies an int (no decimal values)
            # this typically occurs when the data is int but nulls are present
            if (df[col].fillna(0) % 1 == 0).all():
                out_types[col] = get_int_fld_type(col)

        elif curr_dtype.startswith('int'):
            out_types[col] = get_int_fld_type(col)

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
