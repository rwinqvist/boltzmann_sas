# make_plots_uav.py
"""
UAV analog of make_plots.py.

The DAG sweep's plots are all "metric vs. depth" (plot_metric_vs_length,
plot_switch_step_vs_length, ...), grouped by approach. The UAV sweep has
no depth axis -- it fixes SIZE=40 and instead varies (beta, alpha) -- so
the natural x-axis is PARAM_PAIRS, one figure per metric, instead of one
figure per (beta, alpha) with depth on the x-axis.

plot_metric_vs_params already does exactly this and is reused as-is. The
switch-step plot, combined reward/time boxplot, and switch-stats table
don't have a params-keyed version in paper_plots.py (only depth-keyed),
so small local versions are defined below rather than editing that shared
module.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matploblib as mpl
# uncomment for LCSS
#mpl.rcParams['pdf.fonttype'] = 42
#mpl.rcParams['ps.fonttype'] = 42

from simulations.approach import Approach
from simulations.uav_simulations import sweep_config as cfg
from simulations.paper_plots import (
    set_paper_style, plot_metric_vs_params, format_beta_alpha_label,
    grouped_boxplot, DEFAULT_APPROACH_COLORS, DEFAULT_APPROACH_LABELS,
)
from simulations.uav_simulations.belief_plots import plot_beliefs, plot_beliefs_both


PAPER_PLOTS_DIR = f"{cfg.CACHE_PATH}/paper_plots"
PRESENTATION_PLOTS_DIR = f"{cfg.CACHE_PATH}/presentation_plots"
LATEX_TABLES_DIR = f"{cfg.CACHE_PATH}/latex_tables"
os.makedirs(PAPER_PLOTS_DIR, exist_ok=True)
os.makedirs(PRESENTATION_PLOTS_DIR, exist_ok=True)
os.makedirs(LATEX_TABLES_DIR, exist_ok=True)

tag = cfg.fn


def df_to_nested_params(df):
    """
    {beta,alpha,approach,...} rows -> {(beta, alpha): {approach: [row_dict, ...]}}
    pooling across layouts (there's no depth axis here, so layout is the
    only thing to pool over within a given (beta, alpha) point).
    """
    out = {}
    for (beta, alpha), group in df.groupby(["beta", "alpha"]):
        out[(beta, alpha)] = {}
        for approach_str, approach_group in group.groupby("approach"):
            approach = Approach.BAMCP if approach_str == "bamcp" else Approach.BAMCP_ES
            out[(beta, alpha)][approach] = approach_group.to_dict("records")
    # preserve PARAM_PAIRS order rather than groupby's sorted order
    return {pair: out[pair] for pair in cfg.PARAM_PAIRS if pair in out}


def plot_switch_step_vs_params(results_by_params_and_approach, approach=Approach.BAMCP_ES,
                                figsize=(8, 5), pdf_save_path=None, png_save_path=None):
    """ Params-axis analog of paper_plots.plot_switch_step_vs_length. """
    params = list(results_by_params_and_approach.keys())
    data = {approach: []}
    for beta, alpha in params:
        rows = results_by_params_and_approach[(beta, alpha)].get(approach, [])
        steps = [r["stop_step"] for r in rows if r.get("stopped_early") and r.get("stop_step") is not None]
        data[approach].append(np.array(steps, dtype=float))

    x_labels = [format_beta_alpha_label(b, a) for b, a in params]
    fig, ax = plt.subplots(figsize=figsize)
    grouped_boxplot(ax, data, x_labels, [approach])
    ax.set_xlabel(r"Human parameters ($\beta$, $\alpha$)")
    ax.set_ylabel("Switch step")

    fig.tight_layout()
    if pdf_save_path:
        fig.savefig(pdf_save_path)
        print(f"Saved: {pdf_save_path}")
    if png_save_path:
        fig.savefig(png_save_path)
        print(f"Saved: {png_save_path}")
    return fig, ax


def plot_reward_and_time_combined_boxplot_params(results_by_params_and_approach, params,
                                                   pdf_save_path=None, png_save_path=None,
                                                   colors=None, figsize=(8, 4)):
    """ Params-axis analog of paper_plots.plot_reward_and_time_combined_boxplot. """
    fig, (ax_reward, ax_time) = plt.subplots(1, 2, figsize=figsize, sharex=False)
    fig.subplots_adjust(wspace=0.02)
    ax_time.yaxis.tick_right()
    ax_time.yaxis.set_label_position("right")

    default_colors = {Approach.BAMCP: "steelblue", Approach.BAMCP_ES: "darkorange"}
    colors = colors or default_colors
    labels = {Approach.BAMCP: "BAMCP", Approach.BAMCP_ES: "FIHP"}
    approaches = [Approach.BAMCP, Approach.BAMCP_ES]

    reward_data = {a: [] for a in approaches}
    time_data = {a: [] for a in approaches}
    for beta, alpha in params:
        per_approach = results_by_params_and_approach[(beta, alpha)]
        for a in approaches:
            rows = per_approach.get(a, [])
            reward_data[a].append(np.array([r["total_reward"] for r in rows]))
            time_data[a].append(np.array([r["total_time"] for r in rows]))

    x_labels = [format_beta_alpha_label(b, a) for b, a in params]
    grouped_boxplot(ax_reward, reward_data, x_labels, approaches,
                     colors={a.value: colors[a] for a in approaches})
    grouped_boxplot(ax_time, time_data, x_labels, approaches,
                     colors={a.value: colors[a] for a in approaches})

    ax_reward.set_ylabel("Total reward")
    ax_time.set_ylabel("Wall-clock time (s)")
    for ax in (ax_reward, ax_time):
        ax.set_xlabel("")
        ax.grid(alpha=0.3, axis="y")
        ax.get_legend().remove() if ax.get_legend() else None

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[a]) for a in approaches]
    fig.legend(handles, [labels[a] for a in approaches], loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1), frameon=True)

    fig.supxlabel(r"Human parameters ($\beta$, $\alpha$)", fontsize=12, y=0.02)
    fig.subplots_adjust(bottom=0.25)
    fig.subplots_adjust(wspace=0)

    if pdf_save_path:
        fig.savefig(pdf_save_path)
        print(f"Saved: {pdf_save_path}")
    if png_save_path:
        fig.savefig(png_save_path, dpi=150)
        print(f"Saved: {png_save_path}")
    return fig, (ax_reward, ax_time)


def summarize_switch_stats_params(results_by_params_and_approach, fihp_approach=Approach.BAMCP_ES,
                                   bamcp_approach=Approach.BAMCP):
    """ Params-axis analog of make_plots.summarize_switch_stats. """
    rows = []
    for (beta, alpha), per_approach in results_by_params_and_approach.items():
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
            "beta": beta, "alpha": alpha, "n_stopped": n_stopped, "n_total": n_total,
            "switch_step_mean": switch_mean, "switch_step_median": switch_median,
            "human_obs_mean": obs_mean, "human_obs_median": obs_median,
            "switch_frac_mean": frac_mean, "switch_frac_median": frac_median,
        })

    return pd.DataFrame(rows)


def make_plots(df):
    flat_results = df_to_nested_params(df)
    if not flat_results:
        print("No data found -- did build_summary_df_uav.py run first?")
        return

    plot_metric_vs_params(
        flat_results, metric="total_reward", ylabel="Total reward",
        save_path=f"{PAPER_PLOTS_DIR}/fig_reward_vs_params{tag}.pdf",
    )
    plot_metric_vs_params(
        flat_results, metric="total_time", ylabel="Wall-clock time (s)",
        save_path=f"{PAPER_PLOTS_DIR}/fig_time_vs_params{tag}.pdf",
    )
    plot_switch_step_vs_params(
        flat_results,
        pdf_save_path=f"{PAPER_PLOTS_DIR}/fig_switchstep_vs_params{tag}.pdf",
        png_save_path=f"{PRESENTATION_PLOTS_DIR}/fig_switchstep_vs_params{tag}.png",
    )

    params = list(flat_results.keys())
    plot_reward_and_time_combined_boxplot_params(
        flat_results, params,
        pdf_save_path=f"{PAPER_PLOTS_DIR}/fig_reward_time_combined{tag}.pdf",
        png_save_path=f"{PRESENTATION_PLOTS_DIR}/fig_reward_time_combined{tag}.png",
    )

    summary_df = summarize_switch_stats_params(flat_results)
    print("\n=== UAV switch stats ===")
    print(summary_df.round(1))
    summary_df.round(1).to_latex(f"{LATEX_TABLES_DIR}/switch_stats_uav{tag}.tex", index=False)


if __name__ == "__main__":
    set_paper_style(base_size=14)
    df = pd.read_pickle(cfg.cached_fn)
    make_plots(df)
    plot_beliefs(PAPER_PLOTS_DIR, PRESENTATION_PLOTS_DIR, tag, layout_idx=0, sims=range(1, 2))
    plot_beliefs_both(PAPER_PLOTS_DIR, PRESENTATION_PLOTS_DIR, tag, layout_idx=0, sims=range(1, 2))