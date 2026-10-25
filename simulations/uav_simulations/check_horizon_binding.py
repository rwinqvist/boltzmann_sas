"""
Local script -- run on your machine (not the cluster) once ablation
results are synced down.

Checks whether MAX_DEPTH is actually binding for the saved BAMCP results
at the (beta=5, alpha=2, size=70) ablation point, by reloading the
already-computed results (no recomputation -- every sim file already
exists on disk, so run_standard_bamcp just hits its
`if os.path.exists(fn): return` cache path per sim) and comparing
total_steps against MAX_DEPTH.

If total_steps is consistently well below MAX_DEPTH, depth isn't the
constraint and the num_trials ablation is the one worth pursuing further.
If total_steps is bunching up near MAX_DEPTH, that's evidence trials are
getting cut off before reaching the goal, and depth is worth increasing.
"""
import os
import joblib
import numpy as np

from domains.uav.uav_domain import UAVDomain
from domains.uav.utils import get_layouts
from simulations.utils import get_grid_tag, make_grid, get_results_path
from simulations.approach import Approach
from simulations.uav_simulations import sweep_config as cfg


FIXED_TRUE_BETA = 5.0
FIXED_TRUE_ALPHA = 2.0
FIXED_SIZE = 70
NUM_TRIALS_VALUES = [1000, 2000]  # whatever you've actually run so far
NUM_AUTOS = 1  # always exactly one autonomous operator in this setup -- known constant,
               # no need to build an OperatorContext just to count it


def main():
    size = FIXED_SIZE
    p_obs = cfg.P_OBS
    restrict_to_forward = cfg.RESTRICT_TO_FORWARD
    manhattan_dist = (size - 1) + (size - 1)
    MAX_DEPTH = 2 * manhattan_dist

    SEED = cfg.SEED
    num_sims = cfg.num_sims
    utility_scale = cfg.UTILITY_SCALE
    is_toy = cfg.is_toy
    is_cluster = cfg.is_cluster

    beta_grid = make_grid(cfg.lb_beta, cfg.ub_beta, cfg.grid_res)
    alpha_grid = make_grid(cfg.lb_alpha, cfg.ub_alpha, cfg.grid_res)
    grid_tag = get_grid_tag(cfg.lb_beta, cfg.ub_beta, cfg.lb_alpha, cfg.ub_alpha, cfg.grid_res)

    num_layouts = cfg.NUM_LAYOUTS
    layouts = get_layouts(size=size, p_obs=p_obs, num_layouts=num_layouts)

    print(f"MAX_DEPTH = {MAX_DEPTH} (manhattan_dist = {manhattan_dist})")
    print(f"{'num_trials':>10} | {'n':>4} | {'mean_steps':>10} | {'max_steps':>9} | "
          f"{'frac > 90% MAX_DEPTH':>20} | {'frac == MAX_DEPTH (cutoff)':>26}")
    print("-" * 90)

    for num_trials in NUM_TRIALS_VALUES:
        all_total_steps = []

        for lidx, layout in enumerate(layouts):
            layout_seed = SEED + 1000 * lidx

            domain = UAVDomain.from_map(size=size, p_obs=p_obs, layout=layout,
                                        restrict_to_forward=restrict_to_forward)

            # NOTE: the file that actually exists on disk has NO
            # "_numtrialsN_ablation" suffix (confirmed via the probe) --
            # meaning whichever run produced it did NOT include num_trials
            # in fn_app. If num_trials=1000 and num_trials=2000 were both
            # run this way, they map to the IDENTICAL path, so the second
            # run's `if os.path.exists(fn): return` cache check would have
            # silently skipped recomputation and reused the first run's
            # results. That would fully explain "2000 trials didn't do
            # much" -- it may never have actually run.
            fn_app_on_disk = cfg.base_fn_app + f"_L{lidx+1}"  # matches what's really saved, no suffix

            config = {
                "layout_seed": layout_seed, "seed": SEED, "max_depth": MAX_DEPTH, "size": size,
                "true_beta": FIXED_TRUE_BETA, "true_alpha": FIXED_TRUE_ALPHA, "num_sims": num_sims,
                "num_trials": num_trials, "utility_scale": utility_scale,
                "restrict_to_forward": restrict_to_forward,
            }

            probe_fn = get_results_path(
                domain_name=domain.domain_name, domain_tag=domain.id_tag, num_humans=1,
                num_autos=NUM_AUTOS, approach=Approach.BAMCP,
                true_beta=FIXED_TRUE_BETA, true_alpha=FIXED_TRUE_ALPHA, seed=SEED, sim=1,
                is_toy=is_toy, fn_app=fn_app_on_disk, grid_tag=grid_tag, config=config, is_cluster=is_cluster,
            )
            print(f"    [probe] sim=1 path exists={os.path.exists(probe_fn)}: {probe_fn}")

            # load_all_sims doesn't support is_cluster at all (confirmed via
            # TypeError) -- so load each sim's file directly via joblib,
            # using get_results_path (which DOES support is_cluster) to
            # build each per-sim path ourselves. Pure data loading, no
            # BAMCP simulation, no domain/scoring/transition construction.
            results = []
            for sim in range(1, num_sims + 1):
                fn = get_results_path(
                    domain_name=domain.domain_name, domain_tag=domain.id_tag, num_humans=1,
                    num_autos=NUM_AUTOS, approach=Approach.BAMCP,
                    true_beta=FIXED_TRUE_BETA, true_alpha=FIXED_TRUE_ALPHA, seed=SEED, sim=sim,
                    is_toy=is_toy, fn_app=fn_app_on_disk, grid_tag=grid_tag, config=config, is_cluster=is_cluster,
                )
                if not os.path.exists(fn):
                    print(f"    [!!] missing sim={sim} for num_trials={num_trials}, layout {lidx+1}: {fn}")
                    continue
                results.append(joblib.load(fn))

            if results:
                all_total_steps.extend(r["total_steps"] for r in results)

        if not all_total_steps:
            print(f"{num_trials:>10} | no results found -- check fn_app/tag matches what the ablation run saved")
            continue

        steps = np.array(all_total_steps)
        frac_near_cutoff = np.mean(steps > 0.9 * MAX_DEPTH)
        frac_at_cutoff = np.mean(steps >= MAX_DEPTH)

        print(f"{num_trials:>10} | {len(steps):>4} | {steps.mean():>10.1f} | {steps.max():>9} | "
              f"{frac_near_cutoff:>20.1%} | {frac_at_cutoff:>26.1%}")

    print("\nInterpretation:")
    print("  - if frac_near_cutoff / frac_at_cutoff are ~0 across the board, MAX_DEPTH is")
    print("    NOT binding -- increasing it further is unlikely to change reward, and the")
    print("    num_trials ablation (larger jumps, e.g. 10k-20k) is the one worth pursuing.")
    print("  - if a meaningful fraction of episodes are hitting/near MAX_DEPTH, depth IS")
    print("    plausibly constraining performance and is worth increasing directly.")
    print("\n  !! IMPORTANT: fn_app here does NOT include num_trials (matching what's")
    print("     actually on disk). If your num_trials=1000 and num_trials=2000 runs both")
    print("     used this same fn_app convention, they map to the SAME saved file --")
    print("     meaning the second run's cache check may have silently reused the first")
    print("     run's results without recomputing. Check whether the rows above for 1000")
    print("     and 2000 are IDENTICAL (same mean/max steps) -- if so, that would fully")
    print("     explain \"2000 trials didn't do much\": it likely never actually ran.")


if __name__ == "__main__":
    main()