"""
UAV domain simulations -- cluster version.

Single job, internal loop over (true_beta, true_alpha) x layouts, both
human + autonomous operators, full 4-action set. Designed to run headless
(no display) and to survive individual run failures without losing the
whole job's results.
"""
import os
import sys
import json
import time
import traceback

from domains.uav.uav_domain import UAVDomain
from domains.uav.utils import get_layouts
from operators.context import OperatorContext
from operators.utils import OperatorType
from simulations.utils import get_grid_tag, make_grid
from simulations.approach import Approach
from simulations.scoring_functions import generate_progression_biased_uav_scoring_function
from simulations.domain_transitions import build_auto_domain_transitions
from simulations.runners import run_early_stopping_bamcp, run_standard_bamcp
from simulations.uav_simulations import sweep_config as cfg


RESULTS_LOG_PATH = "simulations/uav_simulations/run_log.txt"
MANIFEST_PATH = "simulations/uav_simulations/run_manifest.jsonl"


def ensure_output_dirs():
    os.makedirs(os.path.dirname(RESULTS_LOG_PATH), exist_ok=True)


def log(msg):
    """ Timestamped, flushed print -- so cluster log tailing shows real-time progress. """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def flatten_layouts_uav(results_by_approach):
    """
    {approach: {lidx: [trial_dict, ...]}} -> {approach: [trial_dict, ...]}
    (one level shallower than the layered-MDP flatten_layouts, since this
    script doesn't sweep an outer "depth"/"size" dimension.)
    """
    return {
        approach: [trial for layout_results in per_layout.values() for trial in layout_results]
        for approach, per_layout in results_by_approach.items()
    }


def build_domain_and_contexts(size, p_obs, layout, layout_seed, p_success, utility_scale=1,
                               restrict_to_forward=False):
    domain = UAVDomain.from_map(size=size, p_obs=p_obs, layout=layout, restrict_to_forward=restrict_to_forward)
    cost_nominals = {a: 0 for a in domain.actions}
    Phi_nom = generate_progression_biased_uav_scoring_function(domain, scale=utility_scale, seed=layout_seed)

    auto_domain_transitions = build_auto_domain_transitions(domain, p_success)
    auto_context = OperatorContext(
        category=OperatorType.AUTO,
        n=1,
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=auto_domain_transitions,
        cost_nominals=cost_nominals,
    )

    auto_op_contexts = [auto_context]  # human + auto, both operators active

    return domain, Phi_nom, cost_nominals, auto_op_contexts


def run_both_bamcp_variants(config, domain, Phi_nom, cost_nominals, auto_op_contexts,
                             true_beta, true_alpha, seed, num_sims, beta_grid, alpha_grid,
                             window, tol, is_toy, is_cluster, fn_app, grid_tag, save_results, n_jobs, debug,
                             return_results=False):
    results_bamcp = run_standard_bamcp(
        config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, seed, num_sims,
        auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, is_cluster=is_cluster,
        fn_app=fn_app, grid_tag=grid_tag, save_results=save_results, n_jobs=n_jobs, debug=debug,
        return_results=return_results,
    )

    es_fn_app = fn_app + f"_window{window}_eps{tol}"
    results_bamcp_es = run_early_stopping_bamcp(
        config=config, domain=domain, Phi_nom=Phi_nom, true_beta=true_beta, true_alpha=true_alpha,
        cost_nominals=cost_nominals, seed=seed, num_sims=num_sims, auto_op_contexts=auto_op_contexts,
        beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, is_cluster=is_cluster, fn_app=es_fn_app, grid_tag=grid_tag,
        save_results=save_results, n_jobs=n_jobs, debug=debug, window=window, tol=tol,
        return_results=return_results,
    )
    return {Approach.BAMCP: results_bamcp, Approach.BAMCP_ES: results_bamcp_es}


def main():
    ensure_output_dirs()
    t_start = time.time()

    # --- uav domain info (from sweep_config -- single source of truth) ---
    size = cfg.SIZE
    p_obs = cfg.P_OBS
    restrict_to_forward = cfg.RESTRICT_TO_FORWARD

    # --- BAMCP and simulation configs ---
    SEED = cfg.SEED
    MAX_DEPTH = cfg.MAX_DEPTH
    num_trials = cfg.num_trials
    num_sims = cfg.num_sims
    utility_scale = cfg.UTILITY_SCALE
    num_layouts = cfg.NUM_LAYOUTS
    window = cfg.window
    tol = cfg.tol
    n_jobs = 10  # cluster-local, not part of the shared config (doesn't affect saved results)

    # --- belief grids ---
    beta_grid = make_grid(cfg.lb_beta, cfg.ub_beta, cfg.grid_res)
    alpha_grid = make_grid(cfg.lb_alpha, cfg.ub_alpha, cfg.grid_res)
    grid_tag = get_grid_tag(cfg.lb_beta, cfg.ub_beta, cfg.lb_alpha, cfg.ub_alpha, cfg.grid_res)

    parameter_pairs = cfg.PARAM_PAIRS
    p_success = cfg.p_success

    layouts = get_layouts(size=size, p_obs=p_obs, num_layouts=num_layouts)

    save_results = True
    is_toy = cfg.is_toy
    is_cluster = cfg.is_cluster
    debug = False

    base_fn_app = cfg.base_fn_app

    log(f"Starting UAV sweep: size={size} p_obs={p_obs} restrict_to_forward={restrict_to_forward} "
        f"parameter_pairs={parameter_pairs} num_layouts={num_layouts} num_trials={num_trials}")

    for pair_idx, (true_beta, true_alpha) in enumerate(parameter_pairs):
        log(f"=== parameter pair {pair_idx+1}/{len(parameter_pairs)}: "
            f"beta={true_beta}, alpha={true_alpha} ===")

        n_layouts_ok = 0

        for lidx, layout in enumerate(layouts):
            layout_seed = SEED + 1000 * lidx
            log(f"  layout {lidx+1}/{len(layouts)} (seed={layout_seed})")
            t_layout_start = time.time()

            try:
                domain, Phi_nom, cost_nominals, auto_op_contexts = build_domain_and_contexts(
                    size=size, p_obs=p_obs, layout=layout, layout_seed=layout_seed,
                    p_success=p_success, utility_scale=utility_scale,
                    restrict_to_forward=restrict_to_forward,
                )

                fn_app = base_fn_app + f"_L{lidx+1}"

                config = {
                    "layout_seed": layout_seed,
                    "seed": SEED,
                    "max_depth": MAX_DEPTH,
                    "size": size,
                    "true_beta": true_beta,
                    "true_alpha": true_alpha,
                    "num_sims": num_sims,
                    "num_trials": num_trials,
                    "utility_scale": utility_scale,
                    "restrict_to_forward": restrict_to_forward,
                }

                # return_results=False -- results are already persisted to disk via
                # save_results=True; no need to hold/reload them in memory here.
                run_both_bamcp_variants(
                    config=config, domain=domain, Phi_nom=Phi_nom, cost_nominals=cost_nominals,
                    auto_op_contexts=auto_op_contexts, true_beta=true_beta, true_alpha=true_alpha,
                    seed=SEED, num_sims=num_sims, beta_grid=beta_grid, alpha_grid=alpha_grid,
                    window=window, tol=tol, is_toy=is_toy, is_cluster=is_cluster, fn_app=fn_app, grid_tag=grid_tag,
                    save_results=save_results, n_jobs=n_jobs, debug=debug, return_results=False,
                )

                n_layouts_ok += 1
                log(f"  layout {lidx+1} done in {time.time()-t_layout_start:.1f}s")

            except Exception:
                # don't let one bad layout/parameter combo kill the whole cluster job --
                # log it and keep going, so partial results are still usable.
                log(f"  !! layout {lidx+1} FAILED after {time.time()-t_layout_start:.1f}s:")
                log(traceback.format_exc())
                with open(RESULTS_LOG_PATH, "a") as f:
                    f.write(f"FAILED: beta={true_beta} alpha={true_alpha} layout={lidx+1} "
                            f"seed={layout_seed}\n{traceback.format_exc()}\n")
                continue

        n_ok = n_layouts_ok
        if n_ok == 0:
            log(f"  no successful layouts for beta={true_beta}, alpha={true_alpha}")
            continue

        tag = (f"depths{size}_pobs{p_obs}_forward{restrict_to_forward}_b{true_beta}_a{true_alpha}"
               f"_uscale{utility_scale}_psuccess{p_success}_n{num_sims}_window{window}_eps{tol}"
               f"_num_trials{num_trials}_algdepth_{MAX_DEPTH}_numlayouts{num_layouts}")

        # write one manifest line per completed (beta, alpha) point -- individual sim
        # results are already persisted to disk by run_standard_bamcp/run_early_stopping_bamcp
        # (via joblib.dump, see get_results_path); this manifest just records which
        # (config, tag) combinations completed, for the local plotting script to pick up.
        with open(MANIFEST_PATH, "a") as f:
            f.write(json.dumps({
                "tag": tag, "true_beta": true_beta, "true_alpha": true_alpha,
                "size": size, "p_obs": p_obs, "restrict_to_forward": restrict_to_forward,
                "utility_scale": utility_scale, "p_success": p_success, "num_sims": num_sims,
                "window": window, "tol": tol, "num_trials": num_trials, "max_depth": MAX_DEPTH,
                "num_layouts": num_layouts, "seed": SEED,
                "n_layouts_completed": n_ok,
            }) + "\n")

        log(f"  saved {n_ok} layout-results for beta={true_beta}, alpha={true_alpha} (tag={tag})")

    log(f"Done. Total time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()