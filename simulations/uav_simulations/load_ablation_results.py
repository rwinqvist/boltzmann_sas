"""
Local script -- run after syncing cluster results down.

Loads BAMCP and FIHP results for every num_trials value in
sweep_config.NUM_TRIALS_ABLATION_VALUES, at every (beta, alpha) in
sweep_config.PARAM_PAIRS, across all layouts, and prints mean/std
total_reward and total_time for each -- lets you see directly whether
BAMCP's reward approaches FIHP's as num_trials grows, and at what time cost.
"""
import os
import joblib
import numpy as np

from domains.uav.uav_domain import UAVDomain
from domains.uav.utils import get_layouts
from simulations.utils import get_grid_tag, make_grid, get_results_path
from simulations.approach import Approach
from simulations.uav_simulations import sweep_config as cfg

NUM_AUTOS = 1  # always exactly one autonomous operator in this setup


def load_results(domain, grid_tag, true_beta, true_alpha, num_trials, num_sims, lidx, approach, es_suffix=""):
    """ Load all sims for one (beta, alpha, num_trials, layout, approach) combo. """
    config = {
        "layout_seed": cfg.SEED + 1000 * lidx, "seed": cfg.SEED, "max_depth": cfg.MAX_DEPTH,
        "size": cfg.SIZE, "true_beta": true_beta, "true_alpha": true_alpha, "num_sims": num_sims,
        "num_trials": num_trials, "utility_scale": cfg.UTILITY_SCALE,
        "restrict_to_forward": cfg.RESTRICT_TO_FORWARD,
    }
    fn_app = cfg.build_base_fn_app(num_trials=num_trials, num_sims=num_sims) + f"_L{lidx+1}" + es_suffix

    results = []
    for sim in range(1, num_sims + 1):
        fn = get_results_path(
            domain_name=domain.domain_name, domain_tag=domain.id_tag, num_humans=1,
            num_autos=NUM_AUTOS, approach=approach,
            true_beta=true_beta, true_alpha=true_alpha, seed=cfg.SEED, sim=sim,
            is_toy=cfg.is_toy, fn_app=fn_app, grid_tag=grid_tag, config=config, is_cluster=cfg.is_cluster,
        )
        if not os.path.exists(fn):
            print(f"    [!!] missing: {fn}")
            return None  # missing file -- caller reports this
        results.append(joblib.load(fn))
    return results


def summarize(results, label):
    if not results:
        print(f"    {label:>28}: NO DATA FOUND")
        return
    rewards = np.array([r["total_reward"] for r in results])
    times = np.array([r["total_time"] for r in results])
    print(f"    {label:>28}: n={len(results):>4}  "
          f"reward={rewards.mean():>8.1f} (+/-{rewards.std():>6.1f})  "
          f"time={times.mean():>7.1f}s (+/-{times.std():>6.1f})")


def main():
    beta_grid = make_grid(cfg.lb_beta, cfg.ub_beta, cfg.grid_res)
    alpha_grid = make_grid(cfg.lb_alpha, cfg.ub_alpha, cfg.grid_res)
    grid_tag = get_grid_tag(cfg.lb_beta, cfg.ub_beta, cfg.lb_alpha, cfg.ub_alpha, cfg.grid_res)

    layouts = get_layouts(size=cfg.SIZE, p_obs=cfg.P_OBS, num_layouts=cfg.NUM_LAYOUTS)
    es_suffix = f"_window{cfg.window}_eps{cfg.tol}"

    for (true_beta, true_alpha) in cfg.PARAM_PAIRS:
        # PARAM_PAIRS may store plain ints (e.g. (5, 2)), but the cluster
        # combo script parses --beta/--alpha via argparse(type=float), so
        # what actually got saved used floats (5.0, 2.0). Match that here,
        # or get_results_path builds "true_b5_a2" instead of "true_b5.0_a2.0".
        true_beta, true_alpha = float(true_beta), float(true_alpha)
        print(f"\n=== beta={true_beta}, alpha={true_alpha} ===")

        for num_trials in cfg.NUM_TRIALS_ABLATION_VALUES:
            print(f"  num_trials={num_trials}:")

            bamcp_all, fihp_all = [], []
            for lidx, layout in enumerate(layouts):
                domain = UAVDomain.from_map(size=cfg.SIZE, p_obs=cfg.P_OBS, layout=layout,
                                            restrict_to_forward=cfg.RESTRICT_TO_FORWARD)

                bamcp_res = load_results(domain, grid_tag, true_beta, true_alpha, num_trials,
                                          cfg.num_sims, lidx, Approach.BAMCP)
                fihp_res = load_results(domain, grid_tag, true_beta, true_alpha, num_trials,
                                         cfg.num_sims, lidx, Approach.BAMCP_ES, es_suffix=es_suffix)

                if bamcp_res is None:
                    print(f"    [!!] missing BAMCP results for layout {lidx+1}")
                else:
                    bamcp_all.extend(bamcp_res)

                if fihp_res is None:
                    print(f"    [!!] missing FIHP results for layout {lidx+1}")
                else:
                    fihp_all.extend(fihp_res)

            summarize(bamcp_all, "BAMCP")
            summarize(fihp_all, "FIHP")

    print("\nInterpretation:")
    print("  - if BAMCP's mean reward climbs toward FIHP's as num_trials increases, while")
    print("    BAMCP's mean time climbs past FIHP's, that's the expected sampling-noise")
    print("    story: BAMCP needs more trials to approach FIHP's (near-exact) reward.")
    print("  - if BAMCP's reward plateaus meaningfully BELOW FIHP's even at the largest")
    print("    num_trials tested, that suggests something beyond sampling noise --")
    print("    e.g. residual MCTS/tree-reuse issues -- worth investigating further.")


if __name__ == "__main__":
    main()