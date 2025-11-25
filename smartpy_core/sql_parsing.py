"""
Use this module to parse query components (columms, tables used)
from a sql query.

In progress.

"""

import re
from dataclasses import dataclass


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
        extract str sections
    
    """
    q = ' ' + q

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
    else:
        print(':(')

    return res


def get_sel_parts(q: str) -> list[str]:
    """
    Parse the `select` section of a sql query 
    into a list w/ an entry for each column.

    Parameters:
    -----------
    q: str
        The query to parse.

    Returns:
    ---------
        

    """
    # get the select section
    parts = get_query_sections(q)
    sel = parts.get('select')
    if sel is None:
        return {}
    
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
        # ...if the column has it specified
        # ...note right now this is just the alias
        tab = None
        if '.' in col:
            col_split = col.split('.')
            assert len(col_split) == 2
            tab = col_split[0]
            col = col_split[1]
        
        res[col] = {
            'tab': tab,
            'expr': expr
        }

    return res
