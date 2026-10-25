"""
Local script -- builds the BAMCP-vs-FIHP results table across regimes,
loading directly from saved results (no hand-typed numbers). Includes a
paired significance test on reward (BAMCP and FIHP share seeds/layouts,
so they're paired observations -- Wilcoxon signed-rank, nonparametric,
appropriate given the reward distributions aren't obviously normal).

Fill in REGIMES below with your actual (size, beta, alpha, num_trials)
combinations -- the size=70/beta=5 row is already filled in from your
num_trials ablation; replace the size=30 placeholders with your real values.
"""
import os
import joblib
import numpy as np
from scipy.stats import wilcoxon

from simulations.utils import get_grid_tag, make_grid, get_results_path
from simulations.approach import Approach
from simulations.uav_simulations import sweep_config as cfg

NUM_AUTOS = 1

# --- FILL IN YOUR ACTUAL REGIMES HERE ---
REGIMES = [
    {"label": r"size=30, $\beta$=0.1, $\alpha$=2 (low $\beta$)",
     "size": 30, "true_beta": 0.1, "true_alpha": 2.0, "num_trials": 1000},
    {"label": r"size=30, $\beta$=5, $\alpha$=2 (no time to collapse)",
     "size": 30, "true_beta": 5.0, "true_alpha": 2.0, "num_trials": 1000},
    {"label": r"size=70, $\beta$=5, $\alpha$=2 (num\_trials=1000)",
     "size": 70, "true_beta": 5.0, "true_alpha": 2.0, "num_trials": 1000},
    {"label": r"size=70, $\beta$=5, $\alpha$=2 (num\_trials=4000)",
     "size": 70, "true_beta": 5.0, "true_alpha": 2.0, "num_trials": 4000},
]


def load_results(domain_tag, grid_tag, size, true_beta, true_alpha, num_trials, num_sims, lidx, approach, es_suffix=""):
    config = {
        "layout_seed": cfg.SEED + 1000 * lidx, "seed": cfg.SEED, "max_depth": 2 * ((size - 1) * 2),
        "size": size, "true_beta": true_beta, "true_alpha": true_alpha, "num_sims": num_sims,
        "num_trials": num_trials, "utility_scale": cfg.UTILITY_SCALE,
        "restrict_to_forward": cfg.RESTRICT_TO_FORWARD,
    }
    fn_app = cfg.build_base_fn_app(num_trials=num_trials, num_sims=num_sims) + f"_L{lidx+1}" + es_suffix

    results = []
    for sim in range(1, num_sims + 1):
        fn = get_results_path(
            domain_name="uav", domain_tag=domain_tag, num_humans=1,
            num_autos=NUM_AUTOS, approach=approach,
            true_beta=true_beta, true_alpha=true_alpha, seed=cfg.SEED, sim=sim,
            is_toy=cfg.is_toy, fn_app=fn_app, grid_tag=grid_tag, config=config, is_cluster=cfg.is_cluster,
        )
        if not os.path.exists(fn):
            print(f"    [!!] missing: {fn}")
            return None
        raw = joblib.load(fn)
        # strip to just what we need -- saved result dicts also carry a
        # full per-step belief trace (beta_grid x alpha_grid, logged every
        # step), which is heavy enough to OOM when accumulated across many
        # sims/layouts/regimes. Discard it immediately rather than keeping
        # the full dict in memory.
        results.append({"total_reward": raw["total_reward"], "total_time": raw["total_time"]})
        del raw
    return results


def collect_regime(regime):
    size = regime["size"]
    true_beta, true_alpha = regime["true_beta"], regime["true_alpha"]
    num_trials = regime["num_trials"]
    num_sims = cfg.num_sims

    beta_grid = make_grid(cfg.lb_beta, cfg.ub_beta, cfg.grid_res)
    alpha_grid = make_grid(cfg.lb_alpha, cfg.ub_alpha, cfg.grid_res)
    grid_tag = get_grid_tag(cfg.lb_beta, cfg.ub_beta, cfg.lb_alpha, cfg.ub_alpha, cfg.grid_res)
    es_suffix = f"_window{cfg.window}_eps{cfg.tol}"

    # domain_tag matches UAVDomain.id_tag's format directly (confirmed by
    # belief_plots_uav.py's load_raw_sims) -- no need to build a real domain
    # or load real layouts just to read back saved results.
    domain_tag = f"size{size}_pobs{cfg.P_OBS}"

    bamcp_all, fihp_all = [], []
    for lidx in range(cfg.NUM_LAYOUTS):
        bamcp_res = load_results(domain_tag, grid_tag, size, true_beta, true_alpha, num_trials,
                                  num_sims, lidx, Approach.BAMCP)
        fihp_res = load_results(domain_tag, grid_tag, size, true_beta, true_alpha, num_trials,
                                 num_sims, lidx, Approach.BAMCP_ES, es_suffix=es_suffix)

        if bamcp_res is not None:
            bamcp_all.extend(bamcp_res)
        if fihp_res is not None:
            fihp_all.extend(fihp_res)

    return bamcp_all, fihp_all


def main():
    rows = []

    for regime in REGIMES:
        print(f"Loading: {regime['label']}...")
        bamcp_all, fihp_all = collect_regime(regime)

        if not bamcp_all or not fihp_all:
            print(f"  [!!] incomplete data for this regime -- skipping row")
            continue

        bamcp_reward = np.array([r["total_reward"] for r in bamcp_all])
        fihp_reward = np.array([r["total_reward"] for r in fihp_all])
        bamcp_time = np.array([r["total_time"] for r in bamcp_all])
        fihp_time = np.array([r["total_time"] for r in fihp_all])

        # paired test requires equal-length, aligned samples (same seeds/layouts)
        if len(bamcp_reward) == len(fihp_reward):
            try:
                _, p_value = wilcoxon(bamcp_reward, fihp_reward)
            except ValueError:
                p_value = float("nan")  # e.g. all differences are zero
        else:
            print(f"  [!!] BAMCP (n={len(bamcp_reward)}) and FIHP (n={len(fihp_reward)}) "
                  f"sample sizes differ -- paired test skipped")
            p_value = float("nan")

        rows.append({
            "label": regime["label"],
            "bamcp_reward": bamcp_reward, "fihp_reward": fihp_reward,
            "bamcp_time": bamcp_time, "fihp_time": fihp_time,
            "p_value": p_value,
        })

    # --- write LaTeX table to file ---
    LATEX_TABLES_DIR = f"{cfg.CACHE_PATH}/latex_tables"
    os.makedirs(LATEX_TABLES_DIR, exist_ok=True)

    table_name = f"bamcp_vs_fihp_reward_time_uav_n{cfg.num_sims}_L{cfg.NUM_LAYOUTS}_{len(rows)}regimes.tex"
    table_path = os.path.join(LATEX_TABLES_DIR, table_name)

    lines = []
    lines.append("\\begin{tabular}{llrrr}")
    lines.append("\\toprule")
    lines.append("Domain & Approach & Reward & Time (s) & $p$ (paired) \\\\")
    lines.append("\\midrule")
    for row in rows:
        p_str = "$<0.001$" if row["p_value"] < 0.001 else f"${row['p_value']:.3f}$"
        lines.append(f"\\multirow{{2}}{{*}}{{{row['label']}}} "
                      f"& BAMCP & {row['bamcp_reward'].mean():.1f} $\\pm$ {row['bamcp_reward'].std():.1f} "
                      f"& {row['bamcp_time'].mean():.1f} $\\pm$ {row['bamcp_time'].std():.1f} & --- \\\\")
        lines.append(f" & FIHP & {row['fihp_reward'].mean():.1f} $\\pm$ {row['fihp_reward'].std():.1f} "
                      f"& {row['fihp_time'].mean():.1f} $\\pm$ {row['fihp_time'].std():.1f} & {p_str} \\\\")
        lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(table_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nWrote LaTeX table to: {table_path}")
    print("(needs \\usepackage{multirow} in your preamble)")


if __name__ == "__main__":
    main()