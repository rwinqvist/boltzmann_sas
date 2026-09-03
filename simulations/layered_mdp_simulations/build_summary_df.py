# build_summary_df.py
import joblib
import pandas as pd
from joblib import Parallel, delayed

from simulations.approach import Approach
from simulations.utils import get_grid_tag, get_results_path
from simulations.layered_mdp_simulations.sweep_config import (
    DEPTHS, NUM_LAYOUTS, PARAM_PAIRS, num_actions, num_sims, SEED, num_trials,
    window, tol, is_toy, is_cluster, MAX_DEPTH, base_fn_app,
    lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res, CACHE_PATH, cached_fn
)

grid_tag = get_grid_tag(lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res)


def extract_row(result_dict, depth, layout_idx, beta, alpha, approach):
    stop_info = result_dict.get("stop_info") or {}
    is_auto = result_dict.get("is_auto")
    stop_step = stop_info.get("stop_step")
    human_obs = sum(1 for a in is_auto[:stop_step] if not a) if (is_auto and stop_step) else None

    return {
        "depth": depth, "layout_idx": layout_idx, "beta": beta, "alpha": alpha,
        "approach": approach.value,
        "total_reward": result_dict.get("total_reward"),
        "total_time": result_dict.get("total_time"),
        "total_steps": result_dict.get("total_steps"),
        "stopped_early": stop_info.get("stopped_early"),
        "stop_step": stop_step,
        "trigger_param": stop_info.get("trigger_param"),
        "human_obs_at_switch": human_obs,
    }


def build_file_list():
    file_list = []
    domain_name = "layered_mdp"

    for true_beta, true_alpha in PARAM_PAIRS:
        for depth in DEPTHS:
            domain_tag = f"d{depth}_a{num_actions}"
            for lidx in range(NUM_LAYOUTS):
                config = {"max_depth": MAX_DEPTH, "depth": depth, "true_beta": true_beta,
                          "true_alpha": true_alpha, "num_sims": num_sims, "num_trials": num_trials}

                bamcp_fn_app = base_fn_app + f"_L{lidx + 1}"
                es_fn_app = bamcp_fn_app + f"_window{window}_eps{tol}"

                for approach, fn_app in [(Approach.BAMCP, bamcp_fn_app), (Approach.BAMCP_ES, es_fn_app)]:
                    for sim in range(1, num_sims + 1):
                        fn = get_results_path(
                            domain_name=domain_name, domain_tag=domain_tag,
                            num_humans=1, num_autos=1, approach=approach,
                            true_beta=true_beta, true_alpha=true_alpha, seed=SEED, sim=sim,
                            config=config, is_toy=is_toy, is_cluster=is_cluster,
                            fn_app=fn_app, grid_tag=grid_tag,
                        )
                        file_list.append((fn, depth, lidx, true_beta, true_alpha, approach))
    return file_list


def load_one(fn, depth, lidx, beta, alpha, approach):
    try:
        result_dict = joblib.load(fn)
    except Exception:
        return None
    row = extract_row(result_dict, depth, lidx, beta, alpha, approach)
    del result_dict
    return row


def build_summary_df(n_jobs=20):
    file_list = build_file_list()
    print(f"Loading {len(file_list)} files with n_jobs={n_jobs}...")

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(load_one)(fn, depth, lidx, beta, alpha, approach)
        for fn, depth, lidx, beta, alpha, approach in file_list
    )

    rows = [r for r in results if r is not None]
    n_missing = len(results) - len(rows)

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows ({n_missing} missing/failed)")
    return df


if __name__ == "__main__":
    df = build_summary_df(n_jobs=20)
    df.to_pickle(cached_fn)
    print(f"Saved to {cached_fn}")