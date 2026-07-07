import os
import time
import random
import joblib
import numpy as np
from joblib import Parallel, delayed
from itertools import product

from operators.context import OperatorContext
from operators.utils import OperatorParams, OperatorType
from boltzmann_sas.boltzmann_sas import BoltzmannSAS
from boltzmann_sas.globals import DEFER
from bamcp.bamcp import BAMCPSolver
from bamcp.history import History
from belief.belief import FreqBelief, JointGridBelief
from belief.observation import Observation
from simulations.approach import Approach
from simulations.utils import get_results_path, get_operators

MAX_DEPTH = 10


def unpack_action(bamdp, action):
    """
    NOTE: this duplicates BAMCPSolver.unpack_action (algorithms/bamcp.py) but operates
    on a raw BoltzmannSAS instance rather than a solver. Check whether this standalone
    version is still called anywhere, or whether it's a leftover from before
    BAMCPSolver.unpack_action existed.
    """
    is_defer, is_advice, is_auto = 0, 0, 0

    opidx, advice = action
    if opidx not in bamdp.boltzmann_operator_indices:
        is_auto = 1
    else:
        if advice == DEFER:
            is_defer = 1
        else:
            is_advice = 1

    return is_advice, is_defer, is_auto


def build_sas(domain, operator_contexts):
    operators = get_operators(operator_contexts)
    return BoltzmannSAS(domain, operators)


# ============================================================
# Standard BAMCP
# ============================================================

def _run_std_bamcp_one_sim(sim, domain, operator_contexts, auto_op_contexts, config,
                            true_beta, true_alpha, seed, is_toy, fn_app, save_results):
    """ Runs a single standard-BAMCP simulation. Top-level (picklable) for joblib/loky. """
    sim_seed = seed * 10_000 + sim
    random.seed(sim_seed)
    np.random.seed(sim_seed)

    print(f"Sim: {sim + 1}")
    fn = get_results_path(
        domain_name=domain.domain_name, domain_tag=domain.id_tag(), num_humans=1,
        num_autos=len(auto_op_contexts), approach=Approach.BAMCP,
        true_beta=true_beta, true_alpha=true_alpha, seed=seed, sim=sim + 1, is_toy=is_toy, fn_app=fn_app,
    )
    if os.path.exists(fn):
        print("Results already exist!")
        return

    bamdp = build_sas(domain, operator_contexts)
    solver = BAMCPSolver(bamdp, max_depth=MAX_DEPTH)
    start = time.perf_counter()
    results = solver.run()
    total_time = time.perf_counter() - start
    results["total_time"] = total_time
    results["config"] = config
    if save_results:
        joblib.dump(results, fn)
    print(f"Sim {sim + 1} done in {total_time:.1f}s (total_reward={results['total_reward']:.2f})")


def run_standard_bamcp(config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, seed, num_sims,
                        auto_op_contexts, beta_grid, alpha_grid, is_toy=False, fn_app="", save_results=True, n_jobs=-1, debug=False):
    print("\nApproach: Standard BAMCP")

    if debug: 
        n = len(beta_grid) * len(alpha_grid)
        p_params = {(b, a): 0 for b, a in product(beta_grid, alpha_grid)}
        p_params[(beta_grid[0], alpha_grid[0])] = 1
        init_belief = JointGridBelief(p_params)
    else: 
        init_belief = JointGridBelief.uniform(beta_values=beta_grid, alpha_values=alpha_grid)

    bhuman1 = OperatorContext(
        category=OperatorType.BHUMAN,
        n=1,
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=domain.T_det,
        cost_nominals=cost_nominals,
        params=OperatorParams(beta=true_beta, alpha=true_alpha),
        init_belief=init_belief,
        nom_scoring=Phi_nom,
    )

    operator_contexts = [bhuman1] + list(auto_op_contexts)

    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_std_bamcp_one_sim)(
            sim, domain, operator_contexts, auto_op_contexts, config,
            true_beta, true_alpha, seed, is_toy, fn_app, save_results,
        )
        for sim in range(num_sims)
    )



# ============================================================
# Naive (frequentist) warm-start
# ============================================================

def _run_naive_freq_warmstart_one_sim(sim, domain, operator_contexts, auto_op_contexts, config,
                                  true_beta, true_alpha, seed, n_warmstart, is_toy, fn_app, save_results):
    """ Runs a single naive-warmstart simulation. Top-level (picklable) for joblib/loky. """
    sim_seed = seed * 10_000 + sim
    random.seed(sim_seed)
    np.random.seed(sim_seed)

    print(f"Sim: {sim + 1}")
    fn = get_results_path(
        domain_name=domain.domain_name, domain_tag=domain.id_tag(), num_humans=1,
        num_autos=len(auto_op_contexts), approach=Approach.NAIVE_FREQ_WARMSTART, n_warmstart=n_warmstart,
        true_beta=true_beta, true_alpha=true_alpha, seed=seed, sim=sim + 1, is_toy=is_toy, fn_app=fn_app,
    )
    if os.path.exists(fn):
        print("Results already exist!")
        return

    bamdp = build_sas(domain, operator_contexts)
    state = bamdp.s0
    history = History(items=(state,))
    total_reward = 0
    total_steps = 0
    is_defer_vec, is_advice_vec, is_auto_vec = [], [], []
    belief_vec, belief_stats, rewards, cum_rewards = [], [], [], []
    is_advice = is_auto = 0
    is_defer = 1

    defer_action = (0, DEFER)
    start = time.perf_counter()
    for _ in range(n_warmstart):
        next_state, reward, executed_action = bamdp.step(state, defer_action)
        obs = Observation(
            domain_state=state[0], op_state=state[1][0], operator=bamdp.boltzmann_operators[0],
            executed_domain_action=executed_action[1], issued_domain_action=DEFER,
        )
        is_advice_vec.append(is_advice)
        is_defer_vec.append(is_defer)
        is_auto_vec.append(is_auto)
        rewards.append(reward)

        bamdp.update_belief(obs)
        belief_vec.append(bamdp.get_belief())
        belief_stats.append(bamdp.get_belief_stats())

        history = history.add_entry(defer_action, executed_action, next_state)
        state = next_state
        total_reward += reward
        total_steps += 1
        cum_rewards.append(total_reward)

    bamcp = BAMCPSolver(bamdp, max_depth=MAX_DEPTH)
    results = bamcp.run(
        s0=state, total_reward=total_reward, total_steps=total_steps, is_defer_vec=is_defer_vec,
        is_advice_vec=is_advice_vec, is_auto_vec=is_auto_vec, belief_vec=belief_vec, belief_stats=belief_stats,
        rewards=rewards, cum_rewards=cum_rewards,
    )
    total_time = time.perf_counter() - start

    results["total_time"] = total_time
    results["history"] = History(items=history.items + results["history"].items[1:])
    results["config"] = config

    if save_results:
        joblib.dump(results, fn)
    print(f"Sim {sim + 1} done in {total_time:.1f}s (total_reward={results['total_reward']:.2f})")



def run_naive_freq_warmstart(config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, seed, num_sims,
                         n_warmstart, auto_op_contexts, alpha_grid, is_toy=False, fn_app="", save_results=True, n_jobs=-1):
    print("\nApproach: Naive warmstart")
    bhuman1 = OperatorContext(
        category=OperatorType.BHUMAN,
        n=1,
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=domain.T_det,
        cost_nominals=cost_nominals,
        params=OperatorParams(beta=true_beta, alpha=true_alpha),
        init_belief=FreqBelief.uniform(alpha_values=alpha_grid),
        nom_scoring=Phi_nom,
    )

    config["n_warmstart"] = n_warmstart

    operator_contexts = [bhuman1] + list(auto_op_contexts)

    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_naive_freq_warmstart_one_sim)(
            sim, domain, operator_contexts, auto_op_contexts, config,
            true_beta, true_alpha, seed, n_warmstart, is_toy, fn_app, save_results,
        )
        for sim in range(num_sims)
    )


# ============================================================
# Bayesian warm-start
# ============================================================

def _run_bayesian_warmstart_one_sim(sim, domain, operator_contexts, auto_op_contexts, config,
                                     true_beta, true_alpha, seed, n_warmstart, is_toy, fn_app, save_results):
    """ Runs a single Bayesian-warmstart simulation. Top-level (picklable) for joblib/loky. """
    sim_seed = seed * 10_000 + sim
    random.seed(sim_seed)
    np.random.seed(sim_seed)

    print(f"Sim: {sim + 1}")
    fn = get_results_path(
        domain_name=domain.domain_name, domain_tag=domain.id_tag(), num_humans=1,
        num_autos=len(auto_op_contexts), approach=Approach.NAIVE_BAYESIAN_WARMSTART, n_warmstart=n_warmstart,
        true_beta=true_beta, true_alpha=true_alpha, seed=seed, sim=sim + 1, is_toy=is_toy, fn_app=fn_app,
    )
    if os.path.exists(fn):
        print("Results already exist!")
        return

    bamdp = build_sas(domain, operator_contexts)
    state = bamdp.s0
    history = History(items=(state,))
    total_reward = 0
    total_steps = 0
    is_defer_vec, is_advice_vec, is_auto_vec = [], [], []
    belief_vec, belief_stats, rewards, cum_rewards = [], [], [], []
    is_advice = is_auto = 0
    is_defer = 1

    belief_vec.append(bamdp.get_belief())
    belief_stats.append(bamdp.get_belief_stats())

    defer_action = (0, DEFER)
    start = time.perf_counter()
    for _ in range(n_warmstart):
        next_state, reward, executed_action = bamdp.step(state, defer_action)
        obs = Observation(
            domain_state=state[0], op_state=state[1][0], operator=bamdp.boltzmann_operators[0],
            executed_domain_action=executed_action[1], issued_domain_action=DEFER,
        )
        is_advice_vec.append(is_advice)
        is_defer_vec.append(is_defer)
        is_auto_vec.append(is_auto)
        rewards.append(reward)

        bamdp.update_belief(obs)
        belief_vec.append(bamdp.get_belief())
        belief_stats.append(bamdp.get_belief_stats())

        history = history.add_entry(defer_action, executed_action, next_state)
        state = next_state
        total_reward += reward
        total_steps += 1
        cum_rewards.append(total_reward)

    bamcp = BAMCPSolver(bamdp, max_depth=MAX_DEPTH)
    results = bamcp.run(
        s0=state, total_reward=total_reward, total_steps=total_steps, is_defer_vec=is_defer_vec,
        is_advice_vec=is_advice_vec, is_auto_vec=is_auto_vec, belief_vec=belief_vec, belief_stats=belief_stats,
        rewards=rewards, cum_rewards=cum_rewards,
    )
    total_time = time.perf_counter() - start

    results["total_time"] = total_time
    results["history"] = History(items=history.items + results["history"].items[1:])
    results["config"] = config

    if save_results:
        joblib.dump(results, fn)
    print(f"Sim {sim + 1} done in {total_time:.1f}s (total_reward={results['total_reward']:.2f})")


def run_bayesian_naive_warmstart(config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, seed, num_sims,
                                  n_warmstart, auto_op_contexts, beta_grid, alpha_grid, is_toy=False, fn_app="",
                                  save_results=True, n_jobs=-1, debug=False):
    print("\nApproach: Bayesian naive warmstart")

    if debug: 
        n = len(beta_grid) * len(alpha_grid)
        p_params = {(b, a): 0 for b, a in product(beta_grid, alpha_grid)}
        p_params[(beta_grid[0], alpha_grid[0])] = 1
        init_belief = JointGridBelief(p_params)
    else: 
        init_belief = JointGridBelief.uniform(beta_values=beta_grid, alpha_values=alpha_grid)

    bhuman1 = OperatorContext(
        category=OperatorType.BHUMAN,
        n=1,
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=domain.T_det,
        cost_nominals=cost_nominals,
        params=OperatorParams(beta=true_beta, alpha=true_alpha),
        init_belief=init_belief,
        nom_scoring=Phi_nom,
    )

    config["n_warmstart"] = n_warmstart

    operator_contexts = [bhuman1] + list(auto_op_contexts)

    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_bayesian_warmstart_one_sim)(
            sim, domain, operator_contexts, auto_op_contexts, config,
            true_beta, true_alpha, seed, n_warmstart, is_toy, fn_app, save_results,
        )
        for sim in range(num_sims)
    )