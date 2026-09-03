# make_plots.py
import os
import numpy as np
import pandas as pd
from simulations.approach import Approach
from simulations.layered_mdp_simulations.sweep_config import PARAM_PAIRS, CACHE_PATH, cached_fn, fn
from simulations.paper_plots import (
    plot_metric_vs_length, plot_switch_step_vs_length,
    plot_switch_step_vs_human_obs, set_paper_style, plot_reward_and_time_combined_boxplot
)
from simulations.layered_mdp_simulations.belief_plots import plot_beliefs, plot_beliefs_both

PAPER_PLOTS_DIR = f"{CACHE_PATH}/paper_plots"
PRESENTATION_PLOTS_DIR = f"{CACHE_PATH}/presentation_plots"
LATEX_TABLES_DIR = f"{CACHE_PATH}/latex_tables"
os.makedirs(PAPER_PLOTS_DIR, exist_ok=True)
os.makedirs(PRESENTATION_PLOTS_DIR, exist_ok=True)
os.makedirs(LATEX_TABLES_DIR, exist_ok=True)

tag = fn

def df_to_nested(df, beta, alpha):
    sub = df[(df.beta == beta) & (df.alpha == alpha)]
    out = {}
    for depth, depth_group in sub.groupby("depth"):
        out[depth] = {}
        for approach_str, approach_group in depth_group.groupby("approach"):
            approach = Approach.BAMCP if approach_str == "bamcp" else Approach.BAMCP_ES
            out[depth][approach] = approach_group.to_dict("records")
    return out


def make_plots(df):

    for true_beta, true_alpha in PARAM_PAIRS:
        flat_results = df_to_nested(df, true_beta, true_alpha)
        if not flat_results:
            print(f"No data for beta={true_beta}, alpha={true_alpha} -- skipping")
            continue

        #tag = f"b{true_beta}_a{true_alpha}"

        plot_metric_vs_length(
            flat_results, metric="total_reward", ylabel="Total reward",
            pdf_save_path=f"{PAPER_PLOTS_DIR}/fig_reward_vs_depth{tag}.pdf",
            png_save_path=f"{PRESENTATION_PLOTS_DIR}/fig_reward_vs_depth{tag}.png",
        )
        plot_metric_vs_length(
            flat_results, metric="total_time", ylabel="Wall-clock time (s)",
            pdf_save_path=f"{PAPER_PLOTS_DIR}/fig_time_vs_depth{tag}.pdf",
            png_save_path=f"{PRESENTATION_PLOTS_DIR}/fig_time_vs_depth{tag}.png",
        )
        plot_switch_step_vs_length(
            flat_results, ylabel="Switch step",
            pdf_save_path=f"{PAPER_PLOTS_DIR}/fig_switchstep_vs_depth{tag}.pdf",
            png_save_path=f"{PRESENTATION_PLOTS_DIR}/fig_switchstep_vs_depth{tag}.png",
        )

        depths = sorted(flat_results.keys())
        plot_reward_and_time_combined_boxplot(
            flat_results, depths,
            pdf_save_path=f"{PAPER_PLOTS_DIR}/fig_reward_time_combined{tag}.pdf",
            png_save_path=f"{PRESENTATION_PLOTS_DIR}/fig_reward_time_combined{tag}.png",
        )

        summary_df = summarize_switch_stats(flat_results)
        print(f"\n=== beta={true_beta}, alpha={true_alpha} ===")
        print(summary_df.round(1))
        summary_df.round(1).to_latex(f"{LATEX_TABLES_DIR}/switch_stats_{tag}.tex", index=False)



def summarize_switch_stats(results_by_depth_and_approach, fihp_approach=Approach.BAMCP_ES,
                            bamcp_approach=Approach.BAMCP):
    """
    Version matched to the flat schema produced by build_summary_df.py's
    extract_row -- fields live at the top level (stopped_early, stop_step,
    human_obs_at_switch), not nested under "stop_info" / full-length
    "is_auto" vectors like the raw result dicts this was originally written
    against in paper_plots.py.
    """
    rows = []

    for depth, per_approach in results_by_depth_and_approach.items():
        fihp_results = per_approach.get(fihp_approach, [])
        bamcp_results = per_approach.get(bamcp_approach, [])
        bamcp_total_steps = [r.get("total_steps") for r in bamcp_results]

        switch_steps, human_obs, fracs = [], [], []
        n_total = len(fihp_results)
        n_stopped = 0

        for i, r in enumerate(fihp_results):
            if not r.get("stopped_early"):
                continue
            n_stopped += 1

            stop_step = r.get("stop_step")
            if stop_step is not None:
                switch_steps.append(stop_step)

            hobs = r.get("human_obs_at_switch")
            if hobs is not None:
                human_obs.append(hobs)

            if i < len(bamcp_total_steps) and bamcp_total_steps[i]:
                fracs.append(stop_step / bamcp_total_steps[i])

        def _mean_median(vals):
            arr = np.asarray(vals, dtype=float)
            return (np.nan, np.nan) if len(arr) == 0 else (arr.mean(), np.median(arr))

        switch_mean, switch_median = _mean_median(switch_steps)
        obs_mean, obs_median = _mean_median(human_obs)
        frac_mean, frac_median = _mean_median(fracs)

        rows.append({
            "depth": depth, "n_stopped": n_stopped, "n_total": n_total,
            "switch_step_mean": switch_mean, "switch_step_median": switch_median,
            "human_obs_mean": obs_mean, "human_obs_median": obs_median,
            "switch_frac_mean": frac_mean, "switch_frac_median": frac_median,
        })

    return pd.DataFrame(rows).sort_values("depth").reset_index(drop=True)




if __name__ == "__main__":
    set_paper_style()
    df = pd.read_pickle(cached_fn)
    #print(f"Loaded {len(df)} rows from {cached_fn}")
    make_plots(df)
    #plot_beliefs(PAPER_PLOTS_DIR, PRESENTATION_PLOTS_DIR, tag, depth=150, layout_idx=0, sims=range(1, 2))
    #plot_beliefs_both(PAPER_PLOTS_DIR, PRESENTATION_PLOTS_DIR, tag, depth=150, layout_idx=0, sims=range(1, 2))