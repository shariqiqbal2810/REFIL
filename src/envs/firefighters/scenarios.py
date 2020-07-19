import numpy as np
from functools import partial
from itertools import combinations_with_replacement
from numpy.random import RandomState
from .firefighters import AGENT_TYPES, BUILDING_TYPES


def get_all_unique_teams(all_types, min_len, max_len):
    all_uniq = []
    for i in range(min_len, max_len + 1):
        all_uniq += list(combinations_with_replacement(all_types, i))
    return all_uniq


def generate_scenarios(min_n_agents, max_n_agents, rs):
    # we split scenarios by unique sets of agents
    # (environment will generate all possible building configurations)
    uniq_agent_teams = get_all_unique_teams(sorted(AGENT_TYPES.keys()), min_n_agents, max_n_agents)
    rs.shuffle(uniq_agent_teams)

    scenario_list = []
    for i in range(len(uniq_agent_teams)):
        agent_scenario = list(uniq_agent_teams[i])
        n_agents = len(agent_scenario)

        min_n_blds = max(1, n_agents - 1)
        max_n_blds = min(max_n_agents, n_agents + 1)
        scenario_list.append((agent_scenario, (min_n_blds, max_n_blds)))
    return scenario_list


def scen_similarity(scen1, scen2):
    """
    Return overlap coefficient between two scenarios
    """
    scen1_charcount = {}
    scen2_charcount = {}
    for char1 in scen1:
        if char1 not in scen1_charcount:
            scen1_charcount[char1] = 1
        else:
            scen1_charcount[char1] += 1
    for char2 in scen2:
        if char2 not in scen2_charcount:
            scen2_charcount[char2] = 1
        else:
            scen2_charcount[char2] += 1

    intersect = 0
    # union = 0
    for char in set(list(scen1_charcount.keys()) + list(scen2_charcount.keys())):
        intersect += min(scen1_charcount.get(char, 0), scen2_charcount.get(char, 0))
        # union += max(scen1_charcount.get(char, 0), scen2_charcount.get(char, 0))
    # return intersect / union
    return intersect / min(len(scen1), len(scen2))


def rank_mean_similarity(compare_scenarios, rank_scenarios):
    """
    Return indices of rank_scenarios in order of mean similarity to
    compare_scenarios
    """
    comp_scen_strs = [''.join(scen[0]) for scen in compare_scenarios]
    rank_scen_strs = [''.join(scen[0]) for scen in rank_scenarios]

    comp_scen_sims = [np.mean([scen_similarity(rank_sc, comp_sc)
                               for comp_sc in comp_scen_strs])
                      for rank_sc in rank_scen_strs]

    return np.argsort(comp_scen_sims)


def generate_scen_dict(min_n_agents=2, max_n_agents=8, test_ratio=0.15,
                       train_ratio=0.25, bld_spacing=7, rs=None):
    assert bld_spacing >= 3, "Buildings must be at least 3 spaces apart"
    if rs is None:
        rs = RandomState()

    all_scenarios = generate_scenarios(min_n_agents, max_n_agents, rs)
    # test scenarios are always fixed if ratio is same
    n_test = int(test_ratio * len(all_scenarios))
    test_scenarios = all_scenarios[:n_test]
    if train_ratio is None:
        train_scenarios = all_scenarios
    else:
        n_train = int(train_ratio * len(all_scenarios))
        train_scenarios = all_scenarios[n_test:n_test + n_train]
    scenario_dict = {'train_scenarios': train_scenarios,
                     'test_scenarios': test_scenarios,
                     'max_n_agents': max_n_agents,
                     'max_n_buildings': max_n_agents,
                     'bld_spacing': bld_spacing}
    return scenario_dict


def generate_scen_dict_sim(min_n_agents=2, max_n_agents=8, test_ratio=0.15,
                           train_pct_range=(0.0, 0.25), bld_spacing=7, rs=None):
    assert bld_spacing >= 3, "Buildings must be at least 3 spaces apart"
    if rs is None:
        rs = RandomState()

    all_scenarios = generate_scenarios(min_n_agents, max_n_agents, rs)
    # test scenarios are always fixed if ratio is same
    n_test = int(test_ratio * len(all_scenarios))
    test_scenarios = all_scenarios[:n_test]
    other_scenarios = all_scenarios[n_test:]

    other_incr_sim_inds = rank_mean_similarity(test_scenarios, other_scenarios)
    min_rank_ind = int(train_pct_range[0] * len(other_scenarios))
    max_rank_ind = int(train_pct_range[1] * len(other_scenarios))
    train_scen_inds = other_incr_sim_inds[min_rank_ind:max_rank_ind]
    train_scenarios = [other_scenarios[i] for i in train_scen_inds]

    scenario_dict = {'train_scenarios': train_scenarios,
                     'test_scenarios': test_scenarios,
                     'max_n_agents': max_n_agents,
                     'max_n_buildings': max_n_agents,
                     'bld_spacing': bld_spacing}
    return scenario_dict


scenarios = {
    '2-8a_2-8b_05': partial(generate_scen_dict,
                            min_n_agents=2, max_n_agents=8,
                            test_ratio=0.15, train_ratio=0.05),
    '2-8a_2-8b_25': partial(generate_scen_dict,
                            min_n_agents=2, max_n_agents=8,
                            test_ratio=0.15, train_ratio=0.25),
    '2-8a_2-8b_45': partial(generate_scen_dict,
                            min_n_agents=2, max_n_agents=8,
                            test_ratio=0.15, train_ratio=0.45),
    '2-8a_2-8b_65': partial(generate_scen_dict,
                            min_n_agents=2, max_n_agents=8,
                            test_ratio=0.15, train_ratio=0.65),
    '2-8a_2-8b_85': partial(generate_scen_dict,
                            min_n_agents=2, max_n_agents=8,
                            test_ratio=0.15, train_ratio=0.85),
    '2-8a_2-8b_sim_Q1': partial(generate_scen_dict_sim,
                                min_n_agents=2, max_n_agents=8,
                                test_ratio=0.15, train_pct_range=(0.0, 0.25)),
    '2-8a_2-8b_sim_Q2': partial(generate_scen_dict_sim,
                                min_n_agents=2, max_n_agents=8,
                                test_ratio=0.15, train_pct_range=(0.25, 0.5)),
    '2-8a_2-8b_sim_Q3': partial(generate_scen_dict_sim,
                                min_n_agents=2, max_n_agents=8,
                                test_ratio=0.15, train_pct_range=(0.5, 0.75)),
    '2-8a_2-8b_sim_Q4': partial(generate_scen_dict_sim,
                                min_n_agents=2, max_n_agents=8,
                                test_ratio=0.15, train_pct_range=(0.75, 1.0))
}
