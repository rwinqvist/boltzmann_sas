import argparse
import numpy as np
from domains.layered_mdp.layered_mdp import LayeredMDP
from operators.context import OperatorContext
from operators.utils import OperatorType
from simulations.approach import Approach
from simulations.scoring_functions import generate_scoring_function
from simulations.utils import get_grid_tag, make_grid
from simulations.runners import run_standard_bamcp, run_early_stopping_bamcp
from simulations.domain_transitions import build_auto_domain_transitions



def build_domain_and_contexts(depth, num_actions, p_success, seed, utility_scale=1):
    """ Build the LayeredMDP domain + autonomous-operator context for one depth. """
    domain = LayeredMDP.generate_layered_mdp(depth=depth, num_actions=num_actions, seed=seed)
    cost_nominals = {a: 0 for a in domain.actions}
    Phi_nom = generate_scoring_function(domain, scale=utility_scale, seed=seed)

    auto_domain_transitions = build_auto_domain_transitions(domain, p_success)
    auto_context = OperatorContext(
        category=OperatorType.AUTO,
        n=1,  # single performance state, matches the working code path
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=auto_domain_transitions,
        cost_nominals=cost_nominals,  # only takes effect once you apply the from_context fix
    )

    return domain, Phi_nom, cost_nominals, [auto_context]


def run_both_bamcp_variants(config, domain, Phi_nom, cost_nominals, auto_op_contexts,
                             true_beta, true_alpha, seed, num_sims, beta_grid, alpha_grid,
                             window, tol, is_toy, is_cluster, fn_app, grid_tag, save_results, n_jobs, debug, return_results=False):
    """ Run standard + early-stopping BAMCP for one (depth, true_beta, true_alpha) point. """
    results_bamcp = run_standard_bamcp(
        config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, seed, num_sims,
        auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, is_cluster=is_cluster,
        fn_app=fn_app, grid_tag=grid_tag, save_results=save_results, n_jobs=n_jobs, debug=debug, return_results=return_results
    )


    es_fn_app = fn_app + f"_window{window}_eps{tol}"
    results_bamcp_es = run_early_stopping_bamcp(
        config=config, domain=domain, Phi_nom=Phi_nom, true_beta=true_beta, true_alpha=true_alpha,
        cost_nominals=cost_nominals, seed=seed, num_sims=num_sims, auto_op_contexts=auto_op_contexts,
        beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, is_cluster=is_cluster, fn_app=es_fn_app, grid_tag=grid_tag,
        save_results=save_results, n_jobs=n_jobs, debug=debug, window=window, tol=tol, return_results=return_results
    )

    if not return_results:
        return None 
    
    return {Approach.BAMCP: results_bamcp, Approach.BAMCP_ES: results_bamcp_es}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--layout-idx", type=int, required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--alpha", type=float, required=True)
    args = p.parse_args()

    depth = args.depth
    lidx = args.layout_idx
    true_beta = args.beta
    true_alpha = args.alpha

    num_actions = 3
    num_sims = 10
    p_success = 0.75
    SEED = 5
    num_trials = 1000
    window = 20
    tol = 0.05
    utility_scale = 1
    n_jobs = 2
    is_toy = False
    is_cluster = True
    debug = False
    save_results = True

    MAX_DEPTH = 200  # keep in sync manually

    if utility_scale > 1:
        base_fn_app = f"_uscale{utility_scale}_psuccess{p_success}"
    else:
        base_fn_app = f"_{p_success}"

    lb_beta, ub_beta = 0, 10
    lb_alpha, ub_alpha = 0, 5
    grid_res = 0.1
    beta_grid = make_grid(lb_beta, ub_beta, grid_res)
    alpha_grid = make_grid(lb_alpha, ub_alpha, grid_res)
    grid_tag = get_grid_tag(lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res)

    layout_seed = SEED + 1000 * lidx
    fn_app = base_fn_app + f"_L{lidx + 1}"

    print(f"[start] depth={depth} layout={lidx+1} beta={true_beta} alpha={true_alpha}", flush=True)

    domain, Phi_nom, cost_nominals, auto_op_contexts = build_domain_and_contexts(
        depth, num_actions, p_success, utility_scale=utility_scale, seed=layout_seed
    )

    config = {
        "layout_seed": layout_seed,
        "seed": SEED,
        "max_depth": MAX_DEPTH,
        "depth": depth,
        "num_actions": num_actions,
        "true_beta": true_beta,
        "true_alpha": true_alpha,
        "num_sims": num_sims,
        "num_trials": num_trials,
    }

    # save_results=True writes to disk inside this call -- we deliberately
    # do NOT keep the return value around, so nothing accumulates in memory
    # across combos, and this process exits (freeing everything) right after.
    run_both_bamcp_variants(
        config=config, domain=domain, Phi_nom=Phi_nom, cost_nominals=cost_nominals,
        auto_op_contexts=auto_op_contexts, true_beta=true_beta, true_alpha=true_alpha,
        seed=SEED, num_sims=num_sims, beta_grid=beta_grid, alpha_grid=alpha_grid,
        window=window, tol=tol, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag,
        save_results=save_results, n_jobs=n_jobs, debug=debug, is_cluster=is_cluster, return_results=False
    )

    print(f"[done] depth={depth} layout={lidx+1} beta={true_beta} alpha={true_alpha}", flush=True)


if __name__ == "__main__":
    main()