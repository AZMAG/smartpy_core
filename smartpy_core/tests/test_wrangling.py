import random
import pandas as pd
import pytest

from ..wrangling import *


######################
# BROADCASTING
######################


@pytest.fixture()
def right_df():
    return pd.DataFrame(
        {
            'col1': [100, 200, 50],
            'col2': [1, 2, 3]
        },
        index=pd.Index(['g', 'b', 'z'])
    )


@pytest.fixture()
def left_df():
    return pd.DataFrame({
        'some_val': [10, 9, 8, 7, 6],
        'fk': ['z', 'g', 'g', 'b', 't'],
        'grp': ['r', 'g', 'r', 'g', 'r']
    })


@pytest.fixture()
def right_df2(right_df):
    df = pd.concat([right_df, right_df * -1])
    df['fk'] = df.index
    df['grp'] = ['r', 'r', 'r', 'g', 'g', 'g']
    df.set_index(['fk', 'grp'], inplace=True)
    return df


def test_broadcast_right_not_unique(right_df, left_df):
    with pytest.raises(ValueError):
        s = right_df.col1
        broadcast(s.append(s), left_df.fk)


def test_series_broadcast(right_df, left_df):
    b = broadcast(right_df.col1, left_df.fk).fillna(-1)
    assert (b.values == [50, 100, 100, 200, -1]).all()


def assert_df_broadcast(b):
    assert (b.col1.values == [50, 100, 100, 200, -1]).all()
    assert (b.col2.values == [3, 1, 1, 2, -1]).all()


def test_df_broadcast(right_df, left_df):
    b = broadcast(right_df, left_df.fk).fillna(-1)
    assert_df_broadcast(b)


def test_broadcast_with_fk_col(right_df, left_df):
    b = broadcast(right_df, left_df, 'fk').fillna(-1)
    assert_df_broadcast(b)


def test_series_multi_col_broadcast(right_df2, left_df):
    b = broadcast(right_df2.col1, left_df, ['fk', 'grp']).fillna(-9999)
    assert (b.values == [50, -100, 100, -200, -9999]).all()


def test_df_multi_col_broadcast(right_df2, left_df):
    b = broadcast(right_df2, left_df, ['fk', 'grp']).fillna(-9999)
    assert (b.col1.values == [50, -100, 100, -200, -9999]).all()
    assert (b.col2.values == [3, -1, 1, -2, -9999]).all()


######################
# NULLS
######################


def test_handle_nulls():
    s = pd.Series([8, 6, np.nan])
    f1 = handle_nulls(s)
    f2 = handle_nulls(s, drop_na=True)

    assert (f1.values == [8, 6, 0]).all()
    assert (f2.values == [8, 6]).all()


######################
# RENAME COLUMNS
######################


@pytest.fixture()
def a_df():
    return pd.DataFrame({
        'col1': np.arange(3),
        'col2': np.arange(3),
        'col3': np.arange(3)
    })


def test_rename_columns_all(a_df):
    rename_columns(a_df, 'blah_')
    assert list(a_df.columns) == ['blah_col1', 'blah_col2', 'blah_col3']


def test_rename_columns_some(a_df):
    rename_columns(a_df, suffix='_blah', cols=['col1'])
    assert list(a_df.columns) == ['col1_blah', 'col2', 'col3']


##################################
# EXPLODE QUANTITIES / DO REPEATS
##################################


def test_explode():
    df = pd.DataFrame(
        {
            'col1': [10, 20, 30, 40],
            'units': [3, 2, 1, 0]
        },
        index=pd.Index(['z', 'y', 'x', 'w'])
    )
    e = explode(df, 'units', 'idx')

    assert (e.columns == ['col1', 'units', 'idx']).all()
    assert len(e) == 6
    assert (e['idx'] == ['z', 'z', 'z', 'y', 'y', 'x']).all()
    assert e.index.unique


######################
# CATEGORIZE
######################

def test_categorize():
    breaks = [np.nan, 3, 5, np.nan]
    labels = ['3 and under',
              '4 to 5',
              'more than 5']
    s = pd.Series(np.arange(8))

    cats = pd.Series([
        '3 and under',
        '3 and under',
        '3 and under',
        '3 and under',
        '4 to 5',
        '4 to 5',
        'more than 5',
        'more than 5'
    ])
    assert (cats == categorize(s, breaks, labels)).all()


#####################
# SEEDED CALLS
#####################

def test_seeded():

    # init some seeded objects
    s1 = Seeded(1)
    s2 = Seeded(2)
    s3 = Seeded(1)  # should yield the same results as s1

    # 1st calls
    with s1:
        r1_1 = random.random()
        r1_1_np = np.random.rand(10)

    with s2:
        r2_1 = random.random()
        r2_1_np = np.random.rand(10)

    with s3:
        r3_1 = random.random()
        r3_1_np = np.random.rand(10)

    def_1 = random.random()  # default random env

    # 2nd calls
    with s1:
        r1_2 = random.random()
        r1_2_np = np.random.rand(10)

    with s2:
        r2_2 = random.random()
        r2_2_np = np.random.rand(10)

    with s3:
        r3_2 = random.random()
        r3_2_np = np.random.rand(10)

    def_2 = random.random()  # default random env

    # assertions for python randoms
    assert r1_1 == r3_1 and r1_2 == r3_2
    assert r1_1 != r2_1 or r1_2 != r2_2
    assert r1_1 != def_1 or r1_2 != def_2

    # assertions for numpy randoms
    assert (np.append(r1_1_np, r1_2_np) == np.append(r3_1_np, r3_2_np)).all()
    assert (np.append(r1_1_np, r1_2_np) != np.append(r2_1_np, r2_2_np)).any()

    # test using a call instead of a with block
    def get_rand(size):
        return np.random.rand(size)

    s_call = Seeded(1)
    s_cal1_r1 = s_call(get_rand, 10)
    s_call_r2 = s_call(get_rand, 10)
    assert (np.append(r1_1_np, r1_2_np) == np.append(s_cal1_r1, s_call_r2)).all()
