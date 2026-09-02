import argparse
import numpy as np

from simulations.approach import Approach
from simulations.utils import make_grid, get_grid_tag
from simulations.layered_mdp_simulations.run_paper_simulations import (
    build_domain_and_contexts, run_both_bamcp_variants, set_paper_style,
)


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
    num_sims = 20
    p_success = 0.75
    SEED = 5
    num_trials = 1000
    window = 20
    tol = 0.05
    utility_scale = 1
    n_jobs = 10
    is_toy = False
    debug = False
    save_results = True

    MAX_DEPTH = 200  # matches max(depths) in your sweep -- keep in sync manually

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

    set_paper_style()

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
        save_results=save_results, n_jobs=n_jobs, debug=debug,
    )

    print(f"[done] depth={depth} layout={lidx+1} beta={true_beta} alpha={true_alpha}", flush=True)


if __name__ == "__main__":
    main()