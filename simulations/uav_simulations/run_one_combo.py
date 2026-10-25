import argparse
from domains.uav.uav_domain import UAVDomain
from domains.uav.utils import get_layouts
from operators.context import OperatorContext
from operators.utils import OperatorType
from simulations.approach import Approach
from simulations.scoring_functions import generate_progression_biased_uav_scoring_function
from simulations.utils import get_grid_tag, make_grid
from simulations.runners import run_standard_bamcp, run_early_stopping_bamcp
from simulations.domain_transitions import build_auto_domain_transitions
from simulations.uav_simulations import sweep_config as cfg

# NOTE: restrict_to_forward is NOT currently implemented in uav_domain.py's
# UAVDomain.__init__/from_map (run_uav_simulations.py calls it with a
# restrict_to_forward kwarg that the class doesn't accept). We don't pass it
# to from_map below, and we don't pull cfg.RESTRICT_TO_FORWARD in here at all.


def build_domain_and_contexts(size, p_obs, layout, p_success, layout_seed, utility_scale=1):
    """ Build the UAV domain + human/auto operator contexts for one layout. """
    domain = UAVDomain.from_map(size=size, p_obs=p_obs, layout=layout)
    cost_nominals = {a: 0 for a in domain.actions}
    Phi_nom = generate_progression_biased_uav_scoring_function(domain, scale=utility_scale, seed=layout_seed)

    auto_domain_transitions = build_auto_domain_transitions(domain, p_success)
    auto_context = OperatorContext(
        category=OperatorType.AUTO,
        n=1,  # single performance state
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=auto_domain_transitions,
        cost_nominals=cost_nominals,
    )

    return domain, Phi_nom, cost_nominals, [auto_context]


def run_both_bamcp_variants(config, domain, Phi_nom, cost_nominals, auto_op_contexts,
                             true_beta, true_alpha, seed, num_sims, beta_grid, alpha_grid,
                             window, tol, is_toy, is_cluster, fn_app, grid_tag, save_results, n_jobs, debug, return_results=False):
    """ Run standard + early-stopping BAMCP for one (layout, true_beta, true_alpha) point. """
    # results_bamcp = run_standard_bamcp(
    #     config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, seed, num_sims,
    #     auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, is_cluster=is_cluster,
    #     fn_app=fn_app, grid_tag=grid_tag, save_results=save_results, n_jobs=n_jobs, debug=debug, return_results=return_results
    # )

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
    p.add_argument("--layout-idx", type=int, required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--alpha", type=float, required=True)
    args = p.parse_args()

    lidx = args.layout_idx
    true_beta = args.beta
    true_alpha = args.alpha

    # -- fixed UAV domain params, pulled from the single source of truth so
    # this script can't drift from what build_summary_df.py / make_plots.py
    # assume ran --
    SIZE = cfg.SIZE
    P_OBS = cfg.P_OBS
    NUM_LAYOUTS = cfg.NUM_LAYOUTS

    UTILITY_SCALE = cfg.UTILITY_SCALE
    p_success = cfg.p_success
    SEED = cfg.SEED
    num_trials = cfg.num_trials
    num_sims = cfg.num_sims
    window = cfg.window
    tol = cfg.tol
    is_toy = cfg.is_toy
    is_cluster = cfg.is_cluster
    MAX_DEPTH = cfg.MAX_DEPTH
    base_fn_app = cfg.build_base_fn_app(num_trials=num_trials, num_sims=num_sims)

    n_jobs = 2  # cluster-local, not part of the shared config (doesn't affect saved results)
    debug = False
    save_results = True

    beta_grid = make_grid(cfg.lb_beta, cfg.ub_beta, cfg.grid_res)
    alpha_grid = make_grid(cfg.lb_alpha, cfg.ub_alpha, cfg.grid_res)
    grid_tag = get_grid_tag(cfg.lb_beta, cfg.ub_beta, cfg.lb_alpha, cfg.ub_alpha, cfg.grid_res)

    layout_seed = SEED + 1000 * lidx
    fn_app = base_fn_app + f"_L{lidx + 1}"

    print(f"[start] layout={lidx+1} beta={true_beta} alpha={true_alpha}", flush=True)

    # Reads the pre-generated layout cache (see SLURM script's warm-up step --
    # this call assumes the cache file already exists, so it never hits the
    # generate-and-dump branch here and can't race with sibling combo jobs).
    layouts = get_layouts(size=SIZE, p_obs=P_OBS, num_layouts=NUM_LAYOUTS)
    layout = layouts[lidx]

    domain, Phi_nom, cost_nominals, auto_op_contexts = build_domain_and_contexts(
        size=SIZE, p_obs=P_OBS, layout=layout, p_success=p_success,
        layout_seed=layout_seed, utility_scale=UTILITY_SCALE,
    )

    config = {
        "layout_seed": layout_seed,
        "seed": SEED,
        "max_depth": MAX_DEPTH,
        "size": SIZE,
        "p_obs": P_OBS,
        "true_beta": true_beta,
        "true_alpha": true_alpha,
        "num_sims": num_sims,
        "num_trials": num_trials,
        "utility_scale": UTILITY_SCALE,
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

    print(f"[done] layout={lidx+1} beta={true_beta} alpha={true_alpha}", flush=True)


if __name__ == "__main__":
    main()