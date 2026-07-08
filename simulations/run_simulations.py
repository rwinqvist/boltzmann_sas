import numpy as np
from domains.layered_mdp.layered_mdp import LayeredMDP
from simulations.approach import Approach
from simulations.scoring_functions import generate_scoring_function
from simulations.utils import load_all_sims
from simulations.plotting import plot_results, plot_belief_heatmaps, plot_reward_and_belief_heatmaps
from simulations.runners import run_standard_bamcp, run_bayesian_naive_warmstart

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
    depth = 100
    num_actions = 3
    true_beta = 0.5
    true_alpha = 1.5
    n_warmstart = 30
    num_sims = 1
    lb_beta, ub_beta = 0, 2.5
    lb_alpha, ub_alpha = 0,2
    grid_res = 0.1
    beta_grid = make_grid(lb_beta, ub_beta, grid_res)
    alpha_grid = make_grid(lb_alpha, ub_alpha, grid_res)
    grid_tag = get_grid_tag(lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res)

    if n_warmstart > depth: 
        raise ValueError("Warm start phase longer than depth of MDP. Exiting program!")


    config = {
        "seed": SEED,
        "depth": depth,
        "num_actions": num_actions,
        "true_beta": true_beta,
        "true_alpha": true_alpha,
        "num_sims": num_sims,
    }

    fn_app = ""

    domain = LayeredMDP.generate_layered_mdp(depth=depth, num_actions=num_actions, seed=SEED)
    cost_nominals = {a: 0 for a in domain.actions}

    # generate nominal human scoring function 
    Phi_nom = generate_scoring_function(domain, seed=SEED)

    auto_op_contexts = []

    save_results = True
    is_toy = True
    debug = False 
    if not save_results:
        print("WARNING! NOT SAVING RESULTS!")
    if is_toy:
        print("WARNING! You're saying under TOY")
    if debug:
        print("WARNING! Debug mode on")
        fn_app = "debug"

    # n_jobs=-1 uses all available cores. Drop to e.g. os.cpu_count() - 1
    # if you want to keep a core free while these run in the background.
    n_jobs = -1

    run_standard_bamcp(
        config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, SEED, num_sims,
        auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag,
        save_results=save_results, n_jobs=n_jobs, debug=debug,
    )

    run_bayesian_naive_warmstart(
        config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, SEED, num_sims, n_warmstart,
        auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag,
        save_results=save_results, n_jobs=n_jobs, debug=debug
    )

    results_bamcp = load_all_sims(
        domain_name=domain.domain_name, domain_tag=domain.id_tag(), approach=Approach.BAMCP,
        true_beta=true_beta, true_alpha=true_alpha, seed=SEED, num_sims=num_sims,
        num_autos=len(auto_op_contexts), is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag,
    )
    results_bayesian_warmstart = load_all_sims(
        domain_name=domain.domain_name, domain_tag=domain.id_tag(), approach=Approach.NAIVE_BAYESIAN_WARMSTART,
        true_beta=true_beta, true_alpha=true_alpha, seed=SEED, num_sims=num_sims,
        num_autos=len(auto_op_contexts), n_warmstart=n_warmstart, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag,
    )

    results_by_approach = {
        Approach.BAMCP: results_bamcp,
        Approach.NAIVE_BAYESIAN_WARMSTART: results_bayesian_warmstart,
    }

    #plot_results(results_by_approach=results_by_approach, config=config, n_warmstart=n_warmstart)
    plot_reward_and_belief_heatmaps(results_by_approach, config)
    print(inspect.signature(plotting.plot_belief_heatmap))
    
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


if __name__ == "__main__":
    main()