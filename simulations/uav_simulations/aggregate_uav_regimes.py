"""
Generalized analog of build_summary_df_uav.py -- but instead of assuming a
single fixed cfg.SIZE and sweeping cfg.PARAM_PAIRS, this varies (size,
beta, alpha, num_trials) together across an arbitrary REGIMES list, so
results from different domain sizes / trial budgets / control conditions
can all be cached in one dataframe and sliced later without re-touching
any joblib files.

REGIMES is the single source of truth for what to load -- both
plot_regime_comparison_2xN.py and build_results_table.py should read
their data from the cached pickle this produces (via cfg.CACHE_PATH +
REGIMES_CACHE_NAME) rather than loading raw files themselves.
"""
import os
import joblib
import pandas as pd
from joblib import Parallel, delayed

from simulations.approach import Approach
from simulations.utils import get_grid_tag, get_results_path
from simulations.uav_simulations import sweep_config as cfg

grid_tag = get_grid_tag(cfg.lb_beta, cfg.ub_beta, cfg.lb_alpha, cfg.ub_alpha, cfg.grid_res)

REGIMES_CACHE_NAME = "cached_regimes_summary.pkl"
REGIMES_CACHE_PATH = f"{cfg.CACHE_PATH}/{REGIMES_CACHE_NAME}"

# --- single source of truth for which (size, beta, alpha, num_trials)
# points to load -- add/remove regimes here; both the table and the 2xN
# plot script read from the cached df this produces.
REGIMES = [
    {"label": r"$d=30$, $\beta_0$=0.1",
     "size": 30, "true_beta": 0.1, "true_alpha": 2.0, "num_trials": 1000},
    {"label": r"$d=30$, $\beta_0$=5",
     "size": 30, "true_beta": 5.0, "true_alpha": 2.0, "num_trials": 1000},
    {"label": r"$d=70$, $\beta_0$=5",
     "size": 70, "true_beta": 5.0, "true_alpha": 2.0, "num_trials": 1000},
    {"label": r"$d=70$, $\beta_0$=8",
    "size": 70, "true_beta": 8.0, "true_alpha": 2.0, "num_trials": 1000},
    {"label": r"$d=100$, $\beta_0$=5",
     "size": 100, "true_beta": 5.0, "true_alpha": 2.0, "num_trials": 1000},
    {"label": r"$d=100$, $\beta_0$=8",
    "size": 100, "true_beta": 8.0, "true_alpha": 2.0, "num_trials": 1000},
]


def regime_key(regime):
    """
    Derives a stable matching key directly from the regime's actual
    parameters -- never hand-maintained, so it can't drift out of sync
    the way a separately-typed "key" string could if someone edits size/
    beta/alpha/num_trials but forgets to update a parallel key field.
    Editing "label" (display text) never changes this.
    """
    return (f"d{regime['size']}_b{regime['true_beta']}"
            f"_a{regime['true_alpha']}_t{regime['num_trials']}")


def extract_row(result_dict, regime_key, size, layout_idx, beta, alpha, num_trials, approach):
    stop_info = result_dict.get("stop_info") or {}
    is_auto = result_dict.get("is_auto")
    stop_step = stop_info.get("stop_step")
    human_obs = sum(1 for a in is_auto[:stop_step] if not a) if (is_auto and stop_step) else None

    return {
        "regime_key": regime_key, "size": size, "layout_idx": layout_idx,
        "beta": beta, "alpha": alpha, "num_trials": num_trials,
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

    for regime in REGIMES:
        key = regime_key(regime)
        size = regime["size"]
        true_beta, true_alpha = float(regime["true_beta"]), float(regime["true_alpha"])
        num_trials = regime["num_trials"]
        num_sims = cfg.num_sims

        domain_tag = f"size{size}_pobs{cfg.P_OBS}"  # matches UAVDomain.id_tag
        fn_app_base = cfg.build_base_fn_app(num_trials=num_trials, num_sims=num_sims)

        for lidx in range(cfg.NUM_LAYOUTS):
            layout_seed = cfg.SEED + 1000 * lidx

            config = {
                "layout_seed": layout_seed, "seed": cfg.SEED, "max_depth": 2 * ((size - 1) * 2),
                "size": size, "p_obs": cfg.P_OBS, "true_beta": true_beta, "true_alpha": true_alpha,
                "num_sims": num_sims, "num_trials": num_trials, "utility_scale": cfg.UTILITY_SCALE,
            }

            bamcp_fn_app = fn_app_base + f"_L{lidx + 1}"
            es_fn_app = bamcp_fn_app + f"_window{cfg.window}_eps{cfg.tol}"

            for approach, fn_app in [(Approach.BAMCP, bamcp_fn_app), (Approach.BAMCP_ES, es_fn_app)]:
                for sim in range(1, num_sims + 1):
                    fn = get_results_path(
                        domain_name="uav", domain_tag=domain_tag,
                        num_humans=1, num_autos=1, approach=approach,
                        true_beta=true_beta, true_alpha=true_alpha, seed=cfg.SEED, sim=sim,
                        config=config, is_toy=cfg.is_toy, is_cluster=cfg.is_cluster,
                        fn_app=fn_app, grid_tag=grid_tag,
                    )
                    file_list.append((fn, key, size, lidx, true_beta, true_alpha, num_trials, approach))
    return file_list


def load_one(fn, key, size, lidx, beta, alpha, num_trials, approach):
    try:
        result_dict = joblib.load(fn)
    except Exception:
        return None
    row = extract_row(result_dict, key, size, lidx, beta, alpha, num_trials, approach)
    del result_dict  # never keep the heavy per-step belief trace around
    return row


def build_regimes_summary_df(n_jobs=20):
    file_list = build_file_list()

    # check files are unique
    #for item in file_list:
     #   fn, key, size, lidx, beta, alpha, num_trials, approach = item

      #  if size == 70 and lidx == 0 and approach == Approach.BAMCP_ES:
        #    print(f"beta={beta}: {fn}")
            
    print(f"Loading {len(file_list)} files with n_jobs={n_jobs}...")

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(load_one)(fn, key, size, lidx, beta, alpha, num_trials, approach)
        for fn, key, size, lidx, beta, alpha, num_trials, approach in file_list
    )

    rows = [r for r in results if r is not None]
    n_missing = len(results) - len(rows)

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows ({n_missing} missing/failed)")
    return df


if __name__ == "__main__":
    df = build_regimes_summary_df(n_jobs=20)
    print(REGIMES_CACHE_PATH)
    df.to_pickle(REGIMES_CACHE_PATH)
    print(f"Saved to {REGIMES_CACHE_PATH}")