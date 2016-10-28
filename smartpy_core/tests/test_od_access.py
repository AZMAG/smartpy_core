import numpy as np
import pandas as pd
import pytest

from ..od_access import *


@pytest.fixture()
def zones():
    z = pd.Series(np.arange(5)).to_frame('zone')
    z['counter'] = 1
    return z


@pytest.fixture()
def oda(zones):
    od = pd.merge(
        zones,
        zones,
        on='counter',
        suffixes=['_from', '_to']
    )

    od.drop('counter', axis=1, inplace=True)
    od['impedance'] = 10
    od.loc[od['zone_from'] == od['zone_to'], 'impedance'] = 1
    od.loc[od['zone_from'] > od['zone_to'], 'impedance'] = 5

    return OdAccess(od, 'zone_from', 'zone_to')


def test_oda_to(oda, zones):
    a = oda(5, zones, 'impedance')
    assert (a['counter'].values == [5, 4, 3, 2, 1]).all()


def test_oda_from(oda, zones):
    a = oda(5, zones, 'impedance', to=False)
    assert (a.counter.values == np.arange(1, 6)).all()


def test_oda_to_with_custom_agg(oda, zones):
    a = oda(5, zones, 'impedance', to=False, agg_func='mean')
    assert (a['counter'].values == np.ones(5)).all()
