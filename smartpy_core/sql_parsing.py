"""
Use this module to parse query components (columms, tables used)
from a sql query.

In progress. Still testing. May be some edge cases that cause problems.

"""

import re
import pandas as pd
from smartpy_core.wrangling import cache


@cache
def get_query_sections(q: str) -> dict[str, str]:
    """
    Given a sql query, returns a dictionary of the
    statement sections.
    
    Currently supports:
    select, into, from, where, group by, having, order by

    Parameters:
    ----------
    q: str
        The query to parse.

    Returns:
    --------
    dict[str, str]: keys are statements/clauses, values are the 
        extracted str sections. If the section does not exist 
        it will have an entry of None.
    
    """
    # normalize
    q = ' ' + re.sub(r'\s+', ' ', q.strip())

    # list of statements to check
    statements = [
        'select',
        'into',
        'from',
        'where',
        'group by',
        'having',
        'order by'
    ]

    # parse out within section parts
    res = {s: None for s in statements}
    s_curr = statements[0]
    for i in range(1, len(statements)):
        s_next = statements[i]        
        m = re.search(r' {} (.*?) {} '.format(s_curr, s_next), q, flags=re.IGNORECASE)
        if m:
            res[s_curr] = m.group(1)
            s_curr = s_next
    
    # parse out the last section
    m = re.search('(?<= {} )(.*)'.format(s_curr), q, flags=re.IGNORECASE)
    if m:
        res[s_curr] = m.group(1)

    return res


@cache
def get_from_parts(q: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses the components of the `from` section of the query.

    Parameters:
    ----------
    q: str
        The query to parse.
    
    Returns:
    --------
    tables: pandas.DataFrame
        Tables used in `from`, along with the source table
        and a flag indicating if the table is using an alias.
    columns: pandas.DataFrame
        Columns used in join operations. The table and 
        the source table. 

    """
    f = get_query_sections(q)['from']
    if f is None:
        return None, None

    # break apart by join
    join_pattern = r'join|inner|left|right|full|outer'
    j_split = re.split(join_pattern, f, flags=re.IGNORECASE)
    j_split = [j.strip() for j in j_split if j.strip() != '']

    # break out tables and join columns
    tabs = {}
    cols = {}

    for j in j_split:
        j2 = re.sub(r' as ', ' ', j, flags=re.IGNORECASE)
        on_sp = re.split(r' on ', j2, flags=re.IGNORECASE)
        
        # table parts -- name and potentially alias
        tab = on_sp[0]
        alias = None
        tab_key = tab
        
        tab_sp = tab.split(' ')
        if len(tab_sp) == 2:
            tab = tab_sp[0]
            alias = tab_sp[1]
            tab_key = alias        
        
        tabs[tab_key] = {'src': tab, 'is_alias': int(tab != alias)}

        # column parts - column name and potential table name/alias
        # ...this may need more consideration for more complex join 
        # ...operators
        if len(on_sp) > 1:
            operator_pattern = r"[+\-*/=]" 
            op_parts = re.split(operator_pattern, on_sp[1])
            for p in op_parts:
                p = p.strip()

                # by default assume no table, just the column
                col = p
                col_tab = None
                col_src_tab = None

                # separate table from column if present
                p_pos = p.rfind('.')
                if p_pos != -1:
                    col = p[p_pos + 1:]
                    col_tab = p[0:p_pos]
                    col_src_tab = tabs[col_tab]['src']

                cols[(col, col_tab)] = col_src_tab
    
    # return as data frames
    tabs_df = pd.DataFrame.from_dict(tabs, orient='index')
    tabs_df.index.name = 'table'
    
    cols_df = pd.Series(cols)
    cols_df.index.names = ['column', 'table']
    cols_df.name = 'src table'
    cols_df = cols_df.reset_index()
    
    return tabs_df, cols_df    


def get_cols_in_expr(expr: str) -> list[str]:
    """
    Returns a list of column names in a sql 
    where expression or a single line of a select statement.

    Parameters:
    -----------
    expr: str
        The expression to evaluate.

    Returns:
    --------
    list[str]: 
        List of unique column names in the expression.
        If a table name prefixes the column this will be preserved.

    Sample usage:
    -------------
    cols = get_cols_in_expr("city in ('Chandler', 'Tempe', 'Phoenix')")
    >>> ['city']

    cols = get_cols_in_expr("city is not null or county = 'Maricopa'"))
    >>> ['county', 'city']

    cols = get_cols_in_expr(  
        '''
        IIF(isnull(B01001003, 0) + isnull(B01001004, 0) + isnull(B01001005, 0) + isnull(B01001006, 0) + isnull(B01001027, 0) + 
        isnull(B01001028, 0) + isnull(B01001029, 0) + isnull(B01001030, 0) <> 0, (cast(isnull(B10001001, 0) as float)) 
        / (isnull(B01001003, 0) + isnull(B01001004, 0) + isnull(B01001005, 0) + isnull(B01001006, 0) + isnull(B01001027, 0) + 
        isnull(B01001028, 0) + isnull(B01001029, 0) + isnull(B01001030, 0)) * 100, 0)
        '''
    )
    >>> ['B01001005', 'B01001003', 'B01001006','B01001029','B01001030','B01001027','B01001004','B01001028','B10001001']

    """
    # reserverd words
    reserved = [ 'and', 'or', 'is', 'not', 'null', 'in', 'like', 'between']

    # remove dtypes casting 
    cast_pattern = rf"{re.escape(' as ')}(.*?){re.escape(')')}"
    expr = re.sub(cast_pattern, '', expr, flags=re.IGNORECASE)

    # remove data inside quote, e.g. 'uno'
    expr = re.sub(rf"\S+{re.escape('(')}", '', expr)

    # remove function names e.g. isnull(my_col, 10)
    expr = re.sub(r'\'(.*?)\'', '', expr)

    # extract words/parts -- retain periods for cases of table.col 
    period_delim = '____period____'
    expr = expr.replace('.', period_delim)
    parts = re.findall(r'[a-zA-Z]\w*', expr)
    parts = [p.replace(period_delim, '.') for p in parts]
    
    # remove reserved keywords
    parts = list(set([p for p in parts if p.lower() not in reserved]))

    return parts


@cache
def get_sel_parts(q: str) -> pd.DataFrame:
    """
    Parse the `select` section of a sql query 
    into a dataframe/ 

    Parameters:
    -----------
    q: str
        The query to parse.

    Returns:
    ---------
    pandas.DataFrame with columns"
    - column: the output column name
    - expr: the sql expression

    """
    # get the select section
    parts = get_query_sections(q)
    sel = parts.get('select')
    if sel is None:
        return {}
    
    # also get the table parts
    # not using this right now, see table comments below
    #f_tabs, f_cols = get_from_parts(q)

    # parse so we have an item for each column
    LP_DELIM = '____L_PARENTH____'
    RP_DELIM = '____R_PARENTH____'
    COMMA_DELIM = '____COMMA____'
    AS_DELIM = '___AS___'
    
    def replace_callback(match):
        """
        For each group, replace commas w/in a function (.e.g isnull(col, 0))
        with a different delimeter so we can split into lines.

        """
        # replace commas and parenthesis
        r1 = (
            match.group(0)
            .replace(',', COMMA_DELIM)
            .replace('(', LP_DELIM)
            .replace(')', RP_DELIM)
        )
        return re.sub(r' as ', AS_DELIM, r1, flags=re.IGNORECASE)

    # recursively replace commas inside function so we can 
    # ..distinguish from commas separating columns
    while '(' in sel:
        sel = re.sub(r'\([^()]*\)', replace_callback, sel)

    # split into lines
    sel_lines = sel.split(',')

    # parse components for each line (column)
    res = {}
    for s in sel_lines:

        # replace tokens back
        s = (
            re.sub(r' as ', ' ', s, flags=re.IGNORECASE)
            .replace(COMMA_DELIM, ',')
            .replace(LP_DELIM, '(')
            .replace(RP_DELIM, ')')
            .replace(AS_DELIM, ' as ')
            .strip()
        )

        # break-out columns vs. expressions
        # ...by default assume column name is the same as the expression
        col = s
        expr = s
        src_tab = None

        if s.endswith(']'):
            # bracketed cols
            col_start_pos = s.rfind('[')
            col = s[col_start_pos + 1: -1]
            expr = s[0:col_start_pos].strip()
        else:
            # look for aliases
            # ...not these are denoted by the last space
            # ...we've already replace 'as' above
            pos = s.rfind(' ')
            if pos != -1:
                col_start_pos = pos
                col = s[col_start_pos + 1:]
                expr = s[0:col_start_pos].strip()

        # pull out table name if needed
        # **NOT DOING THIS RIGHT NOW **
        # **COULD BE MULTIPLE TABLES SO THIS DOESNT TOTALLY MAKE SENSE
        # ...if the column has it specified
        # ...note right now this is just the alias
        #tab = None
        #p_pos = col.rfind('.')
        #if p_pos != -1:
        #    tab = col[0:p_pos]
        #    col = col[p_pos + 1:]
        #    
        #    # get the underlying source table
        #    src_tab = f_cols.get((col, tab))

        res[col] = {
            #'tab': tab,
            'expr': expr,
            #'src_tab': src_tab
        }

    # return results as data frame
    df = pd.DataFrame.from_dict(res, orient='index')
    df.index.name = 'columns'
    return df
