import numpy as np
from domains.layered_mdp.layered_mdp import LayeredMDP
from simulations.approach import Approach
from simulations.scoring_functions import generate_scoring_function
from simulations.utils import load_all_sims
from simulations.plotting import plot_results, plot_belief_heatmaps, plot_reward_and_belief_heatmaps
from simulations.runners import run_standard_bamcp, run_bayesian_naive_warmstart, run_collapse_aware_bamcp
from simulations.analysis import advice_accuracy, advice_vs_actual_reward
from bamcp.bamcp import BAMCPSolver
from bamcp.history import History
from belief.belief import FreqBelief, JointGridBelief
from simulations.utils import get_results_path, get_operators
from boltzmann_sas.boltzmann_bamdp import BoltzmannBAMDP
from operators.context import OperatorContext
from operators.utils import OperatorParams, OperatorType
from boltzmann_sas.boltzmann_bamdp import BoltzmannBAMDP
from boltzmann_sas.globals import DEFER
from bamcp.bamcp import BAMCPSolver
from bamcp.history import History
from belief.belief import FreqBelief, JointGridBelief
from belief.collapse_monitor import AlphaCollapseMonitor
from belief.observation import Observation
from simulations.approach import Approach
from simulations.utils import get_results_path, get_operators
from simulations.analysis import compare_policy_across_alpha
from simulations.domain_transitions import build_auto_domain_transitions


import simulations.plotting as plotting
import inspect


def make_grid(lb, ub, res, decimals=6):
    n = round((ub - lb) / res) + 1
    return [round(lb + i * res, decimals) for i in range(n)]

def get_grid_tag(lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res):
    """
    Short, self-describing string identifying a belief grid configuration —
    the full set of numbers needed to reconstruct it (both bounds, both
    parameters, and resolution), not just resolution alone. Two runs with
    the same resolution but different bounds get different tags, so their
    result paths won't collide.
    """
    return f"beta{lb_beta}-{ub_beta}_alpha{lb_alpha}-{ub_alpha}_res{grid_res}"
 

def main():
    SEED = 5
    MAX_DEPTH = 50 

    depth = 100
    num_actions = 3
    true_beta = 50
    true_alpha = 0.7
    n_warmstart = 30
    num_sims = 20
    lb_beta, ub_beta = 0, 70
    lb_alpha, ub_alpha = 0, 5
    grid_res = 0.1
    beta_grid = make_grid(lb_beta, ub_beta, grid_res)
    alpha_grid = make_grid(lb_alpha, ub_alpha, grid_res)
    grid_tag = get_grid_tag(lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res)

    if n_warmstart > depth: 
        raise ValueError("Warm start phase longer than depth of MDP. Exiting program!")

    # n_jobs=-1 uses all available cores. Drop to e.g. os.cpu_count() - 1
    # if you want to keep a core free while these run in the background.
    n_jobs = -1

    fn_app = ""
    save_results = True
    is_toy = True
    debug = False 
    manual_debug = False
    if not save_results:
        fn_app ="test"
        print("WARNING! NOT SAVING RESULTS!")
    if is_toy:
        print("WARNING! You're saving under TOY")
    if manual_debug:
        print("WARNING! Debug mode on")
        fn_app = "debug"
        save_results = False
        num_sims = 1
        n_jobs = 1

    config = {
        "seed": SEED,
        "max_depth": MAX_DEPTH,
        "depth": depth,
        "num_actions": num_actions,
        "true_beta": true_beta,
        "true_alpha": true_alpha,
        "num_sims": num_sims,
    }


    domain = LayeredMDP.generate_layered_mdp(depth=depth, num_actions=num_actions, seed=SEED)
    cost_nominals = {a: 0 for a in domain.actions}

    # generate nominal human scoring function 
    Phi_nom = generate_scoring_function(domain, seed=SEED)


    p_success = 0.75   # tune this to create a genuine crossover with the human, per earlier discussion
    fn_app += f"_{p_success}"
    auto_domain_transitions = build_auto_domain_transitions(domain, p_success)

    auto_context = OperatorContext(
        category=OperatorType.AUTO,
        n=1,                              # single performance state, matches the working code path
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=auto_domain_transitions,
        cost_nominals=cost_nominals,       # only takes effect once you apply the from_context fix above
    )

    auto_op_contexts = []

    run_standard_bamcp(
        config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, SEED, num_sims,
        auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag,
        save_results=save_results, n_jobs=n_jobs, debug=debug,
    )

    # run_bayesian_naive_warmstart(
    #     config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, SEED, num_sims, n_warmstart,
    #     auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag,
    #     save_results=save_results, n_jobs=n_jobs, debug=debug
    # )

    collapse_kwargs = {
        "min_advice_obs": 15,
        "check_every": 10,
        "tolerance": 0.05,
        "confidence": 0.1,
        "required_consecutive": 2,
    }

    # run_collapse_aware_bamcp(
    #     config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, SEED, num_sims,
    #     auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag,
    #     save_results=save_results, n_jobs=n_jobs, collapse_kwargs=collapse_kwargs,
    # )

    results_bamcp = load_all_sims(
        domain_name=domain.domain_name, domain_tag=domain.id_tag(), approach=Approach.BAMCP,
        true_beta=true_beta, true_alpha=true_alpha, seed=SEED, num_sims=num_sims,
        num_autos=len(auto_op_contexts), is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag, config=config,
    )
    # results_bayesian_warmstart = load_all_sims(
    #     domain_name=domain.domain_name, domain_tag=domain.id_tag(), approach=Approach.NAIVE_BAYESIAN_WARMSTART,
    #     true_beta=true_beta, true_alpha=true_alpha, seed=SEED, num_sims=num_sims,
    #     num_autos=len(auto_op_contexts), n_warmstart=n_warmstart, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag, config=config,
    # )

    # results_collapse_aware = load_all_sims(
    #     domain_name=domain.domain_name, domain_tag=domain.id_tag(), approach=Approach.BAMCP_ALPHA_COLLAPSE,
    #     true_beta=true_beta, true_alpha=true_alpha, seed=SEED, num_sims=num_sims,
    #     num_autos=len(auto_op_contexts), is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag, config=config,
    # )

    results_by_approach = {
        Approach.BAMCP: results_bamcp,
    }


    #plot_results(results_by_approach=results_by_approach, config=config, n_warmstart=n_warmstart)
    plot_reward_and_belief_heatmaps(results_by_approach, config)

    for approach, results_list in results_by_approach.items():
        all_followed = [sum(r["followed_advice"]) for r in results_list]
        all_advice = [sum(r["is_advice"]) for r in results_list]

        rates = [f / a if a > 0 else float("nan") for f, a in zip(all_followed, all_advice)]

        print(
            f"{approach.value}: compliance rate = {np.nanmean(rates):.3f} "
            f"(std={np.nanstd(rates):.3f}) "
            f"| total advice steps = {np.mean(all_advice):.1f} "
            f"| total followed = {np.mean(all_followed):.1f}"
        )

    # print("\n=== Reward comparison across approaches ===")
    # print("(NOTE: alpha-collapse is a decision-quality/variance claim, not a wall-clock")
    # print(" speed claim, since BAMCP's search runs on a fixed time budget -- see discussion)")
    # for approach, results_list in results_by_approach.items():
    #     total_rewards = [r["total_reward"] for r in results_list]
    #     print(f"{approach.value}: mean reward = {np.mean(total_rewards):.2f}, "
    #           f"std = {np.std(total_rewards):.2f} (n={len(total_rewards)})")
 
    # print("\n=== Alpha-collapse diagnostics ===")
    # collapse_steps = [r.get("alpha_collapse_step") for r in results_collapse_aware]
    # n_collapsed = sum(1 for s in collapse_steps if s is not None)
    # print(f"Collapsed in {n_collapsed}/{len(results_collapse_aware)} sims")
    # if n_collapsed > 0:
    #     avg_step = np.mean([s for s in collapse_steps if s is not None])
    #     print(f"Average collapse step (of those that did collapse): {avg_step:.1f} / {depth}")
    #     collapse_values = [r.get("alpha_collapse_value") for r in results_collapse_aware if r.get("alpha_collapse_value") is not None]
    #     print(f"Frozen alpha values: {collapse_values}")


    # paired_diffs = [
    # results_collapse_aware[i]["total_reward"] - results_bamcp[i]["total_reward"]
    # for i in range(len(results_bamcp))
    # ]
    # print(f"Paired reward diff: mean={np.mean(paired_diffs):.2f}, std={np.std(paired_diffs):.2f}")
    
    # for r in results_bamcp:  # or whichever results list you have
    #     followed_rewards = [rew for rew, adv, fol in zip(r["rewards"], r["is_advice"], r["followed_advice"]) if adv and fol]
    #     deviated_rewards = [rew for rew, adv, fol in zip(r["rewards"], r["is_advice"], r["followed_advice"]) if adv and not fol]
    #     print(f"followed advice: mean reward = {np.mean(followed_rewards):.3f} (n={len(followed_rewards)})")
    #     print(f"deviated from advice: mean reward = {np.mean(deviated_rewards):.3f} (n={len(deviated_rewards)})")

    # acc, n = advice_accuracy(results_bamcp, domain)
    # print(f"\n\nAdvice accuracy: {acc:.2%} (n={n} advice steps)")

    # advised_r, actual_r, followed = advice_vs_actual_reward(results_bamcp, domain)
    # deviated = ~followed
    # print(f"\n\nOn deviated steps: mean advised-action reward = {advised_r[deviated].mean():.3f}, "
    #     f"mean actual reward = {actual_r[deviated].mean():.3f}")



def run_analysis():
    SEED = 5
    MAX_DEPTH = 50 

    depth = 100
    num_actions = 3
    true_beta = 0.5
    true_alpha = 1.2
    n_warmstart = 30
    num_sims = 20
    lb_beta, ub_beta = 0, 2.5
    lb_alpha, ub_alpha = 0,2
    grid_res = 0.1
    beta_grid = make_grid(lb_beta, ub_beta, grid_res)
    alpha_grid = make_grid(lb_alpha, ub_alpha, grid_res)
    grid_tag = get_grid_tag(lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res)

    if n_warmstart > depth: 
        raise ValueError("Warm start phase longer than depth of MDP. Exiting program!")

    # n_jobs=-1 uses all available cores. Drop to e.g. os.cpu_count() - 1
    # if you want to keep a core free while these run in the background.
    n_jobs = -1

    fn_app = ""
    save_results = True
    is_toy = True
    debug = False 
    manual_debug = True
    if not save_results:
        fn_app ="test"
        print("WARNING! NOT SAVING RESULTS!")
    if is_toy:
        print("WARNING! You're saving under TOY")
    if manual_debug:
        print("WARNING! Debug mode on")
        fn_app = "debug"
        save_results = False
        num_sims = 1
        n_jobs = 1

    config = {
        "seed": SEED,
        "max_depth": MAX_DEPTH,
        "depth": depth,
        "num_actions": num_actions,
        "true_beta": true_beta,
        "true_alpha": true_alpha,
        "num_sims": num_sims,
    }


    domain = LayeredMDP.generate_layered_mdp(depth=depth, num_actions=num_actions, seed=SEED)
    cost_nominals = {a: 0 for a in domain.actions}

    # generate nominal human scoring function 
    Phi_nom = generate_scoring_function(domain, seed=SEED)

    auto_op_contexts = []
    
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
    bamdp = build_sas(domain, operator_contexts)
    solver = BAMCPSolver(bamdp, max_depth=config["max_depth"])

    state = bamdp.s0  # some real state from a trajectory, e.g. bamdp.s0 or a state pulled from a saved History
    compare_policy_across_alpha(
        bamdp, state, beta_fixed=1.1, alpha_values=[0.0, 0.5, 1.0, 1.5, 2.0],
        max_depth=config["max_depth"],
    )

def build_sas(domain, operator_contexts):
    operators = get_operators(operator_contexts)
    return BoltzmannBAMDP(domain, operators)


if __name__ == "__main__":
    main()