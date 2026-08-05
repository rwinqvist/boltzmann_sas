import os
import time
import random
import joblib
import numpy as np
from joblib import Parallel, delayed
from itertools import product

from operators.context import OperatorContext
from operators.utils import OperatorParams, OperatorType
from boltzmann_sas.boltzmann_bamdp import BoltzmannBAMDP
from boltzmann_sas.boltzmann_mdp import BoltzmannMDP
from boltzmann_sas.globals import DEFER, MDP, BAMDP
from bamcp.bamcp import BAMCPSolver, EarlyStoppingBAMCPSolver
from bamcp.history import History
from bamcp.utils import unpack_action
from algorithms.value_iteration import value_iteration
from belief.belief import FreqBelief, JointGridBelief
from belief.observation import Observation
from simulations.approach import Approach
from simulations.utils import get_results_path, get_operators, get_vi_policy_path, load_all_sims

MAX_DEPTH = 50


def build_sas(domain, operator_contexts, model_type):
    operators = get_operators(operator_contexts)
    if model_type == MDP:
        return BoltzmannMDP(domain, operators)
    elif model_type == BAMDP:
        return BoltzmannBAMDP(domain, operators)


# ============================================================
# Standard BAMCP
# ============================================================

def _run_std_bamcp_one_sim(sim, domain, operator_contexts, auto_op_contexts, config,
                            true_beta, true_alpha, seed, is_toy, fn_app, grid_tag, save_results):
    """ Runs a single standard-BAMCP simulation. Top-level (picklable) for joblib/loky. """
    sim_seed = seed * 10_000 + sim
    random.seed(sim_seed)
    np.random.seed(sim_seed)

    print(f"Sim: {sim + 1}")
    fn = get_results_path(
        domain_name=domain.domain_name, domain_tag=domain.id_tag(), num_humans=1,
        num_autos=len(auto_op_contexts), approach=Approach.BAMCP,
        true_beta=true_beta, true_alpha=true_alpha, seed=seed, sim=sim + 1, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag, config=config
    )
    if os.path.exists(fn):
        print("Results already exist!")
        return

    bamdp = build_sas(domain, operator_contexts, model_type=BAMDP)
    solver = BAMCPSolver(bamdp, max_depth=config["max_depth"])
    start = time.perf_counter()
    results = solver.run()
    total_time = time.perf_counter() - start
    results["total_time"] = total_time
    results["config"] = config
    if save_results:
        joblib.dump(results, fn)
    print(f"Sim {sim + 1} done in {total_time:.1f}s (total_reward={results['total_reward']:.2f})")


def run_standard_bamcp(config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, seed, num_sims,
                        auto_op_contexts, beta_grid, alpha_grid, is_toy=False, fn_app="", grid_tag="", save_results=True, n_jobs=-1, debug=False):
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
            true_beta, true_alpha, seed, is_toy, fn_app, grid_tag, save_results,
        )
        for sim in range(num_sims)
    )
    
    #print("remove this exit!!!")
    #exit()

    all_results = load_all_sims(
        domain_name=domain.domain_name, domain_tag=domain.id_tag(), approach=Approach.BAMCP,
        true_beta=true_beta, true_alpha=true_alpha, seed=seed, num_sims=num_sims,
        num_autos=len(auto_op_contexts), is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag, config=config,
    )

    return all_results



# ============================================================
# Early-stopping BAMCP
# ============================================================

def _run_early_stopping_bamcp_one_sim(sim, domain, operator_contexts, auto_op_contexts, config,
                            true_beta, true_alpha, seed, is_toy, fn_app, grid_tag, save_results):
    """ Runs a single standard-BAMCP simulation. Top-level (picklable) for joblib/loky. """
    sim_seed = seed * 10_000 + sim
    random.seed(sim_seed)
    np.random.seed(sim_seed)

    print(f"Sim: {sim + 1}")
    fn = get_results_path(
        domain_name=domain.domain_name, domain_tag=domain.id_tag(), num_humans=1,
        num_autos=len(auto_op_contexts), approach=Approach.BAMCP_ES,
        true_beta=true_beta, true_alpha=true_alpha, seed=seed, sim=sim + 1, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag, config=config
    )
    if os.path.exists(fn):
        print("Results already exist!")
        return

    bamdp = build_sas(domain, operator_contexts, model_type=BAMDP)
    solver = EarlyStoppingBAMCPSolver(bamdp, max_depth=config["max_depth"])
    start = time.perf_counter()
    results = solver.run()
    total_time = time.perf_counter() - start
    results["total_time"] = total_time
    results["config"] = config
    if save_results:
        joblib.dump(results, fn)
    print(f"Sim {sim + 1} done in {total_time:.1f}s (total_reward={results['total_reward']:.2f})")


def run_early_stopping_bamcp(config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, seed, num_sims,
                        auto_op_contexts, beta_grid, alpha_grid, is_toy=False, fn_app="", grid_tag="", save_results=True, n_jobs=-1, debug=False):
    print("\nApproach: Early-stopping BAMCP")

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
        delayed(_run_early_stopping_bamcp_one_sim)(
            sim, domain, operator_contexts, auto_op_contexts, config,
            true_beta, true_alpha, seed, is_toy, fn_app, grid_tag, save_results,
        )
        for sim in range(num_sims)
    )

    all_results = load_all_sims(
        domain_name=domain.domain_name, domain_tag=domain.id_tag(), approach=Approach.BAMCP_ES,
        true_beta=true_beta, true_alpha=true_alpha, seed=seed, num_sims=num_sims,
        num_autos=len(auto_op_contexts), is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag, config=config,
    )

    return all_results





# ------- Standard VI --------- # 
def _run_std_vi_one_sim(sim, domain, mdp:BoltzmannMDP, policy_data, auto_op_contexts, config,
                        true_beta, true_alpha, planning_beta, planning_alpha, seed, is_toy, fn_app, save_results):
    """ Runs a single Bayesian-warmstart simulation. Top-level (picklable) for joblib/loky. """
    sim_seed = seed * 10_000 + sim
    random.seed(sim_seed)
    np.random.seed(sim_seed)

    print(f"Sim: {sim + 1}")
    fn = get_results_path(
        domain_name=domain.domain_name, domain_tag=domain.id_tag(), num_humans=1,
        num_autos=len(auto_op_contexts), approach=Approach.VI,
        true_beta=true_beta, true_alpha=true_alpha, planning_beta=planning_beta, planning_alpha=planning_alpha, seed=seed, sim=sim + 1, is_toy=is_toy, fn_app=fn_app, config=config
    )

    if os.path.exists(fn):
        print("Results already exist!")
        return 

    # simulate policy
    
    results = policy_data.copy()
    policy = policy_data["policy"]
    state = mdp.s0 
    total_reward = 0
    total_steps = 0 
    rewards = []
    cum_rewards = []
    is_defer_vec, is_advice_vec, is_auto_vec = [], [], [] 
    followed_advice = []
    

    while not mdp.is_terminal(state) and not mdp.is_goal(state):
        issued_action = policy[state]

        is_advice, is_defer, is_auto = unpack_action(mdp, issued_action)
        is_advice_vec.append(is_advice)
        is_defer_vec.append(is_defer)
        is_auto_vec.append(is_auto)
 
        next_state, reward, executed_action = mdp.step(
            state, issued_action,
            op_parametrizations={0: OperatorParams(beta=true_beta, alpha=true_alpha)},
        )

        if issued_action[1] != DEFER and executed_action[1] == issued_action[1]:
            followed_advice.append(1)
        else:
            followed_advice.append(0)
 
        state = next_state
        total_reward += reward 
        total_steps += total_steps
        rewards.append(reward)
        cum_rewards.append(total_reward)

    results["cum_rewards"] = cum_rewards
    results["rewards"] = rewards
    results["total_reward"] = total_reward
    results["total_steps"] = total_steps
    results["is_defer"] = is_defer_vec 
    results["is_advice"] = is_advice_vec 
    results["is_auto"] = is_auto_vec
    results["followed_advice"] = followed_advice

    if save_results:
        joblib.dump(results, fn)



def run_standard_vi(config, domain, Phi_nom, true_beta, true_alpha, planning_beta, planning_alpha, cost_nominals, seed, num_sims,
                        auto_op_contexts, is_toy=False, fn_app="", save_results=True, n_jobs=-1, debug=False):
    print("\nApproach: Standard VI")

    bhuman1 = OperatorContext(
        category=OperatorType.BHUMAN,
        n=1,
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=domain.T_det,
        cost_nominals=cost_nominals,
        params=OperatorParams(beta=planning_beta, alpha=planning_alpha),
        nom_scoring=Phi_nom,
    )

    operator_contexts = [bhuman1] + list(auto_op_contexts)

    mdp = build_sas(domain, operator_contexts, model_type=MDP)
    policy_fn = get_vi_policy_path(domain_name=domain.domain_name, domain_tag=domain.id_tag(), num_humans=1, num_autos=len(auto_op_contexts), 
                                   true_beta=true_beta, true_alpha=true_alpha, planning_beta=planning_beta, planning_alpha=planning_alpha, seed=seed, is_toy=is_toy, fn_app=fn_app)
    
    if os.path.exists(policy_fn):
        print("Policy already exists. Loading it from file")
        policy_data = joblib.load(policy_fn)
    else:
        policy_data = {}
        start = time.perf_counter()
        Q, V, policy = value_iteration(model=mdp)
        total_time = time.perf_counter() - start 
        policy_data["Q"] = Q 
        policy_data["V"] = V
        policy_data["policy"] = policy 
        policy_data["total_time"] = total_time
        print(f"VI done in {total_time:.1f}s")
        joblib.dump(policy_data, policy_fn)
    
    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_std_vi_one_sim)(sim, domain=domain, mdp=mdp, policy_data=policy_data, auto_op_contexts=auto_op_contexts, config=config,
                        true_beta=true_beta, true_alpha=true_alpha, planning_beta=planning_beta, planning_alpha=planning_alpha, seed=seed, is_toy=is_toy, fn_app=fn_app, save_results=save_results)
        for sim in range(num_sims)
    )

    all_results = load_all_sims(domain_name=domain.domain_name, domain_tag=domain.id_tag(), approach=Approach.VI,
                                true_beta=true_beta, true_alpha=true_alpha, planning_beta=planning_beta, planning_alpha=planning_alpha, seed=seed, num_sims=num_sims,
                                num_autos=len(auto_op_contexts), config=config, is_toy=is_toy, fn_app=fn_app)
    
    return all_results

