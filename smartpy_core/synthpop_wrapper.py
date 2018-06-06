"""
Attempts to make it easier to run synthpop with custom data.

Essentially a re-implementation of the `multiprocess_synthesize` method in
`synthpop.zone_synthesizer.py` with some custom functionality.

Marginal/category configurations should be specified as a  python dictionary
    - First key is the marginal group (or category).
    - Second key is the name of the marginal.
    - The final dictionary should contain entry of `pums`, this is
        the pandas query that should be applied to the pums records to
        include them in the marginal.
    - There maybe additional configuration items in the dictionary that
        are used elsewhere in synth pipeline but those are presently
        ignored here.

For example:

{
    'hh_income': {
        'income_high':  {'pums': 'HINCP < 25000' },
        'income_low':   {'pums': 'HINCP >= 25000 and HINCP < 75000'},
        'income_high':  {'pums': 'HINCP >= 75000'}
    },
    'hh_size': {
        'size_1':     {'pums': 'NP == 1' },
        'size_2':     {'pums': 'NP == 2' },
        'size_3plus': {'pums': 'NP >= 3' }
    }
}

"""

import multiprocessing
from functools import partial
import pandas as pd

import synthpop.categorizer as synth_cat
import synthpop.ipf.ipf as ipf
import synthpop.ipu.ipu as ipu
import synthpop.draw as draw

from .wrangling import broadcast


def _do_synth(h_pums, p_pums,
              h_marg, p_marg,
              h_cat_ids, p_cat_ids,
              jd_zero_sub,
              ipf_tolerance, ipf_max_iterations,
              ipu_tolerance, ipu_max_iterations,
              num_draws):
    """
    Performs the synthesis.
    """

    # get joint distributions
    def get_jd(cat_ids, agents):
        jd = cat_ids.copy()
        jd['frequency'] = agents.groupby(jd.index.names).size().reindex(jd.index).fillna(0)
        jd['frequency'].replace(0, jd_zero_sub, inplace=True)  # handle zero-cells
        return jd

    h_jd = get_jd(h_cat_ids, h_pums)
    p_jd = get_jd(p_cat_ids, p_pums)

    # ipf for households
    h_constraint, h_iter = ipf.calculate_constraints(h_marg,
                                                     h_jd['frequency'],
                                                     ipf_tolerance,
                                                     ipf_max_iterations)
    h_constraint.index = h_jd.cat_id

    # ipf for persons
    p_constraint, p_iter = ipf.calculate_constraints(p_marg,
                                                     p_jd['frequency'],
                                                     ipf_tolerance,
                                                     ipf_max_iterations)
    p_constraint.index = p_jd.cat_id

    # make frequency tables that the ipu expects
    household_freq, person_freq = synth_cat.frequency_tables(p_pums,
                                                             h_pums,
                                                             p_jd['cat_id'],
                                                             h_jd['cat_id'])

    # do the ipu to match person marginals
    best_weights, fit_quality, iterations = ipu.household_weights(household_freq,
                                                                  person_freq,
                                                                  h_constraint,
                                                                  p_constraint,
                                                                  ipu_tolerance,
                                                                  ipu_max_iterations)

    # draw households
    num_households = int(h_marg.groupby(level=0).sum().mean())
    best_hh, best_pers, best_chisq, best_p = draw.draw_households(
        num_households,
        h_pums,
        p_pums,
        household_freq,
        h_constraint,
        p_constraint,
        best_weights,
        hh_index_start=0,
        num_draws=num_draws)

    # ? just return hh ids and we can sample them later?
    # return list(best_hh.index.values)
    # return list(best_hh['serialno'])
    diagnostics = {
        'ipf_hh_iter': h_iter,
        'ipf_pers_iter': p_iter,
        'ipu_iter': iterations,
        'ipu_fit_quality': fit_quality,
        'chi_sqaure': best_chisq,
        'p_score': best_p
    }
    return list(best_hh['serialno']), diagnostics


def _synth_a_zone(zone_tuple,
                  h_pums, p_pums,
                  h_marg, p_marg,
                  h_cat_ids, p_cat_ids,
                  jd_zero_sub,
                  ipf_tolerance, ipf_max_iterations,
                  ipu_tolerance, ipu_max_iterations,
                  num_draws):

    """
    Remember, we want re-run using larger universe of samples
    if we can't converge.

    """

    # get the current marginals
    zone_id = zone_tuple[0]
    puma = zone_tuple[1]
    h_marg = h_marg.loc[zone_id]
    p_marg = p_marg.loc[zone_id]

    print 'start: {}'.format(zone_id)

    # if there are 0 households needed we're done
    if h_marg.sum() < 1:
        print 'no hh needed: {}'.format(zone_id)
        return zone_id, [], {'status': 'no hh needed'}

    # get the current agents
    curr_hh = h_pums.query("puma == '{}'".format(puma))
    curr_pers = p_pums[p_pums['serialno'].isin(curr_hh['serialno'])]

    try:
        # 1st preference, restrict to agents in the puma
        hh_ids, diagnostics = _do_synth(curr_hh, curr_pers,
                                        h_marg, p_marg,
                                        h_cat_ids, p_cat_ids,
                                        jd_zero_sub,
                                        ipf_tolerance, ipf_max_iterations,
                                        ipu_tolerance, ipu_max_iterations,
                                        num_draws)
        diagnostics['status'] = 'puma level'
        print 'end: {}'.format(zone_id)
        return zone_id, hh_ids, diagnostics

    except RuntimeError:
        # this mean we ran out of iterations on the ipf or ipu
        # so retry using the entire sample
        print 'retrying: {}'.format(zone_id)

        try:

            hh_ids, diagnostics = _do_synth(h_pums, p_pums,
                                            h_marg, p_marg,
                                            h_cat_ids, p_cat_ids,
                                            jd_zero_sub,
                                            ipf_tolerance, ipf_max_iterations,
                                            ipu_tolerance, ipu_max_iterations,
                                            num_draws)
            diagnostics['status'] = 'county level'
            print 'end: {}'.format(zone_id)
            return zone_id, hh_ids, diagnostics

        except Exception as e:
            print '{}:{}'.format(str(e), zone_id)
            return zone_id, [], {'status': 'exception encountered'}

    except Exception as e:
        # ? not sure what to do here?
        # need to somehow log a problem?
        print '{}:{}'.format(str(e), zone_id)
        return zone_id, []


def synth_all_mp(zones,
                 h_pums, p_pums,
                 h_marg, p_marg,
                 h_cat_config, p_cat_config,
                 cores=None,
                 pums_geo_col='PUMA',
                 marginal_zero_sub=1e-5,
                 jd_zero_sub=1e-5,
                 ipf_tolerance=1e-3,
                 ipf_max_iterations=10000,
                 ipu_tolerance=1e-4,
                 ipu_max_iterations=50000,
                 num_draws=100):

    """
    ...
    """

    # the list of zones to process
    # this will be a list of tuples in the form (zone, sample geography)
    assert (h_marg.index.values == p_marg.index.values).all()
    assert zones.index.is_unique
    zone_ids = zip(zones.index, zones)

    # classify agents into bins that march the marginal defs
    def classify(cats, agents):

        s_all = {}

        for cat_name, cat in cats.items():
            s = pd.Series(index=agents.index)
            for marg_name, marg_config in cat.items():
                in_marg = agents.query(marg_config['pums']).index.values
                s[in_marg] = marg_name
            s_all[cat_name] = s

        return pd.concat(s_all, axis=1)

    h_pums2 = classify(h_cat_config, h_pums)
    h_cls_cols = h_pums2.columns
    h_pums2['serialno'] = h_pums['serialno']
    h_pums2['puma'] = h_pums[pums_geo_col]

    p_pums2 = classify(p_cat_config, p_pums)
    p_cls_cols = p_pums2.columns
    p_pums2['serialno'] = p_pums['serialno']

    # get a unique ID for each marginal grouping, the IPU will work off of this
    h_cat_ids = synth_cat.category_combinations(h_marg.columns)
    p_starting_cat_id = h_cat_ids['cat_id'].max() + 1
    p_cat_ids = synth_cat.category_combinations(p_marg.columns)
    p_cat_ids['cat_id'] += p_starting_cat_id

    h_pums2['cat_id'] = broadcast(h_cat_ids, h_pums2, left_fk=h_cat_ids.index.names)
    p_pums2['cat_id'] = broadcast(p_cat_ids, p_pums2, left_fk=p_cat_ids.index.names)

    # if we have nulls, we have problems
    if h_marg.isnull().any().any():
        raise ValueError('household marginals have nulls')
    if p_marg.isnull().any().any():
        raise ValueError('person marginals have nulls')
    if h_pums2.isnull().any().any():
        raise ValueError('problem formmating household pums')
    if p_pums2.isnull().any().any():
        raise ValueError('problem formmating persons pums')

    # handle zero-cells on marginals
    h_marg = h_marg.replace(0, marginal_zero_sub)
    p_marg = p_marg.replace(0, marginal_zero_sub)

    # set up multiprocessing
    print 'handing off to mp...'

    worker = partial(_synth_a_zone,
                     h_pums=h_pums2,
                     p_pums=p_pums2,
                     h_marg=h_marg,
                     p_marg=p_marg,
                     h_cat_ids=h_cat_ids,
                     p_cat_ids=p_cat_ids,
                     jd_zero_sub=jd_zero_sub,
                     ipf_tolerance=ipf_tolerance,
                     ipf_max_iterations=ipf_max_iterations,
                     ipu_tolerance=ipu_tolerance,
                     ipu_max_iterations=ipu_max_iterations,
                     num_draws=num_draws)

    cores = cores if cores else (multiprocessing.cpu_count() - 1)
    p = multiprocessing.Pool(cores)
    results = p.map(worker, zone_ids)
    p.close()
    p.join()

    # results are a list of tuples in the form zone_id, [hh_ids]
    # turn these to a data frame and link to households and persons
    print 'compiling...'
    sampled_zone_ids = []
    sampled_hh_ids = []
    diagnostics = []

    try:

        for row in results:
            curr_z = row[0]
            curr_hh_ids = row[1]

            d = row[2]
            d['pbg_id'] = curr_z
            diagnostics.append(d)

            for hh_id in curr_hh_ids:
                sampled_zone_ids.append(curr_z)
                sampled_hh_ids.append(hh_id)

        sampled_df = pd.DataFrame({'pbg_id': sampled_zone_ids, 'serialno': sampled_hh_ids})
        sampled_df['household_id'] = sampled_df.index

        # households
        final_hh = pd.merge(
            sampled_df,
            pd.concat([h_pums, h_pums2[h_cls_cols]], axis=1),
            on='serialno'
        )
        final_hh.set_index('household_id', inplace=True)

        # persons
        final_pers = pd.merge(
            sampled_df,
            pd.concat([p_pums, p_pums2[p_cls_cols]], axis=1),
            on='serialno'
        )
        final_pers.index.name = 'person_id'

        # diagnostics
        final_diag = pd.DataFrame(diagnostics).set_index('pbg_id')

        return final_hh, final_pers, final_diag

    except Exception as e:
        print e
        return results, str(e), ':('
