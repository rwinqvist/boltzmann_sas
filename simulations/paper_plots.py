from simulations.utils import Approach

"""
paper_plots.py

Clean, publication-style plotting utilities for the paper / thesis chapter.

This is deliberately separate from `simulations/plotting.py` (your existing,
exploratory "everything in one big figure" module, which is still fine for
day-to-day debugging). This module produces ONE focused figure per function,
each answering a single question, sized and captioned for individual
placement in LaTeX.

Figures provided:
    1. plot_metric_vs_length      -- reward (or time) vs. MDP depth, grouped
                                      by approach, fixed (beta, alpha)
    2. plot_metric_vs_params      -- reward (or time) vs. (beta, alpha)
                                      setting, grouped by approach, fixed depth
    3. plot_belief_regime_comparison -- belief-concentration heatmaps side by
                                      side across human-rationality regimes
                                      (e.g. low beta vs. high beta)

Call `set_paper_style()` once, before creating any figures.

Usage sketch (fill in with your actual sweep):

    from simulations.paper_plots import (
        set_paper_style, plot_metric_vs_length, plot_metric_vs_params,
        plot_belief_regime_comparison,
    )
    from simulations.approach import Approach
    from simulations.utils import load_all_sims

    set_paper_style()

    # --- reward / time vs. depth, fixed (beta, alpha) ---
    results_by_depth_and_approach = {}
    for depth in [10, 25, 50, 100]:
        domain_tag = f"d{depth}_a3"   # match LayeredMDP.id_tag
        results_by_depth_and_approach[depth] = {
            Approach.BAMCP: load_all_sims(
                domain_name="layered_mdp", domain_tag=domain_tag, approach=Approach.BAMCP,
                true_beta=5, true_alpha=1.2, seed=5, num_sims=20, num_autos=0,
                config={"max_depth": 50}, is_toy=True, grid_tag=my_grid_tag,
            ),
            Approach.BAMCP_ES: load_all_sims(
                domain_name="layered_mdp", domain_tag=domain_tag, approach=Approach.BAMCP_ES,
                true_beta=5, true_alpha=1.2, seed=5, num_sims=20, num_autos=0,
                config={"max_depth": 50}, is_toy=True, grid_tag=my_grid_tag,
            ),
        }

    plot_metric_vs_length(
        results_by_depth_and_approach, metric="total_reward",
        ylabel="Total reward", save_path="fig_reward_vs_length.pdf",
    )
    plot_metric_vs_length(
        results_by_depth_and_approach, metric="total_time",
        ylabel="Wall-clock time (s)", save_path="fig_time_vs_length.pdf",
    )

    # --- reward / time vs. (beta, alpha), fixed depth ---
    results_by_params_and_approach = {
        (2, 1): {Approach.BAMCP: [...], Approach.BAMCP_ES: [...]},
        (2, 4): {Approach.BAMCP: [...], Approach.BAMCP_ES: [...]},
        (5, 0.1): {Approach.BAMCP: [...], Approach.BAMCP_ES: [...]},
    }
    plot_metric_vs_params(
        results_by_params_and_approach, metric="total_reward",
        ylabel="Total reward", save_path="fig_reward_vs_params.pdf",
    )

    # --- belief comparison across beta regimes ---
    plot_belief_regime_comparison(
        {r"Low $\beta=1$": results_low_beta, r"High $\beta=8$": results_high_beta},
        param_index=ALPHA_INDEX, true_value=1.2,
        save_path="fig_belief_regimes.pdf",
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from simulations.plotting import (
    BETA_INDEX, ALPHA_INDEX,
    average_marginal_matrices, coarsen_marginal, plot_belief_heatmaps,
)


# ============================================================
# Style
# ============================================================

def set_paper_style(base_size=14, use_tex=False):
    """
    Set matplotlib rcParams for paper/thesis-quality figures: LaTeX
    (Computer Modern) rendering and a larger base font size than
    matplotlib's default. Call this once, before creating any figures.

    :param base_size: base font size in points. Titles/labels are scaled up
        from this; tick labels/legend scaled down slightly, so the hierarchy
        stays readable at typical LaTeX \\includegraphics widths (e.g. half
        the text width).
    :param use_tex: if True, try to render text with a real LaTeX
        installation (requires `latex` + `dvipng`/`ghostscript` on PATH,
        matching what your .tex build already needs). If unavailable, this
        catches the failure and falls back to matplotlib's built-in
        "mathtext" mode with a Computer-Modern-like serif font instead of
        raising -- so a machine without LaTeX still gets close-to-LaTeX
        figures rather than a crash.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": base_size,
        "axes.titlesize": base_size + 2,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 2,
        "ytick.labelsize": base_size - 2,
        "legend.fontsize": base_size - 2,
        "figure.titlesize": base_size + 4,
        "lines.linewidth": 2,
        #"axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    if use_tex:
        try:
            plt.rcParams.update({
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            })
            # Force a render now, rather than discovering LaTeX is missing
            # on the first savefig() deep inside a long sweep script.
            fig = plt.figure()
            fig.text(0.5, 0.5, r"$\beta$")
            fig.canvas.draw()
            plt.close(fig)
            return
        except Exception as e:
            print(f"[paper_plots] LaTeX rendering unavailable ({e!r}); "
                  f"falling back to mathtext (no LaTeX install required).")

    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "axes.formatter.use_mathtext": True,
    })


# ============================================================
# Generic grouped box plot (shared building block)
# ============================================================

DEFAULT_APPROACH_COLORS = {
    "bamcp": "steelblue",
    "early_stopping_bamcp": "darkorange",
    "frequentist_warmstart": "firebrick",
    "bayesian_warmstart": "seagreen",
    "bamcp_alpha_collapse": "purple",
    "VI": "gray",
}

DEFAULT_APPROACH_LABELS = {
    "bamcp": "BAMCP",
    "early_stopping_bamcp": "FIHP",
    "frequentist_warmstart": "Freq. warm-start",
    "bayesian_warmstart": "Bayesian warm-start",
    "bamcp_alpha_collapse": r"BAMCP ($\alpha$-collapse)",
    "VI": "VI",
}


def _approach_key(approach):
    return approach.value if hasattr(approach, "value") else approach


def grouped_boxplot(ax, data, x_labels, approaches, colors=None, labels=None,
                     group_gap=0.15, showfliers=False):
    """
    Draw a grouped ("clustered") box plot: one cluster of boxes per
    x-position, one box per approach within each cluster, colored by
    approach. This is the building block behind both
    plot_metric_vs_length and plot_metric_vs_params -- use it directly if
    you want a custom x-axis.

    :param data: {approach: [array_at_x0, array_at_x1, ...]} -- for each
        approach, a list (same order/length as x_labels) of 1D arrays
        holding the per-sim metric values at that x position. A missing
        (approach, x) combination can be an empty array -- that box is
        simply skipped, so sparse sweeps (not every approach run at every
        setting) are fine.
    :param x_labels: tick labels for the x positions (e.g. depths, or
        "beta=.., alpha=.." tags)
    :param approaches: ordered list of approach keys/enums to include, and
        their left-to-right order within each cluster
    :param colors, labels: optional dicts overriding
        DEFAULT_APPROACH_COLORS / DEFAULT_APPROACH_LABELS for approaches
        not already covered by the defaults (e.g. a new Approach you add
        later), merged on top of the defaults
    :param group_gap: fraction of each x-slot left empty between adjacent
        depth/param clusters (visually separates one setting from the next)
    :param showfliers: whether to draw outlier points beyond the whiskers.
        Off by default -- with typically-small per-setting sample counts in
        these sweeps, individual outliers can look more dramatic than they
        are; turn on if you want to inspect them.
    """
    colors = {**DEFAULT_APPROACH_COLORS, **(colors or {})}
    labels = {**DEFAULT_APPROACH_LABELS, **(labels or {})}

    n_groups = len(approaches)
    n_x = len(x_labels)
    box_width = (1.0 - group_gap) / n_groups

    x_positions = np.arange(n_x)

    for gi, approach in enumerate(approaches):
        key = _approach_key(approach)
        color = colors.get(key, "gray")
        series = data.get(approach, data.get(key, None))
        if series is None:
            continue

        offset = (gi - (n_groups - 1) / 2) * box_width
        positions = x_positions + offset

        # keep only positions that actually have data (sparse sweeps)
        valid_positions, valid_series = [], []
        for pos, vals in zip(positions, series):
            vals = np.asarray(vals, dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                valid_positions.append(pos)
                valid_series.append(vals)

        if not valid_series:
            continue

        bp = ax.boxplot(
            valid_series, positions=valid_positions, widths=box_width * 0.9,
            patch_artist=True, showfliers=showfliers, manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor(color)
        for element in ("whiskers", "caps"):
            for line in bp[element]:
                line.set_color(color)
        for median in bp["medians"]:
            median.set_color("black")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)

    legend_handles = [
        Patch(facecolor=colors.get(_approach_key(a), "gray"), alpha=0.6,
              edgecolor=colors.get(_approach_key(a), "gray"),
              label=labels.get(_approach_key(a), _approach_key(a)))
        for a in approaches
    ]
    ax.legend(handles=legend_handles)


def _extract_metric_grid(results_by_x_and_approach, metric):
    """
    Turn {x_val: {approach: [result_dict, ...]}} into the (x_values,
    approaches, data) shape grouped_boxplot expects, pulling `metric` out
    of each result dict (e.g. "total_reward", "total_time").
    """
    x_values = list(results_by_x_and_approach.keys())
    approaches = sorted(
        {a for per_x in results_by_x_and_approach.values() for a in per_x.keys()},
        key=_approach_key,
    )

    data = {a: [] for a in approaches}
    for x in x_values:
        per_x = results_by_x_and_approach[x]
        for a in approaches:
            results_list = per_x.get(a, [])
            vals = [r[metric] for r in results_list if metric in r]
            data[a].append(np.array(vals, dtype=float))

    return x_values, approaches, data


# ============================================================
# Figure: metric vs. MDP length
# ============================================================

def plot_metric_vs_length(results_by_depth_and_approach, metric="total_reward",
                           ylabel=None, title=None, colors=None, labels=None,
                           figsize=(8, 5), pdf_save_path=None, png_save_path=None):
    """
    Box plot of `metric` (e.g. "total_reward" or "total_time") across sims,
    grouped by MDP depth on the x-axis and by approach within each depth
    cluster.

    :param results_by_depth_and_approach: {depth: {approach: [result_dict, ...]}}
        one entry per (depth, approach) sweep point; result_dict is a single
        simulation's result (e.g. one element of load_all_sims(...)), so
        result_dict["total_reward"] / result_dict["total_time"] must exist.
    """
    depths, approaches, data = _extract_metric_grid(results_by_depth_and_approach, metric)
    x_labels = [str(d) for d in depths]

    fig, ax = plt.subplots(figsize=figsize)
    grouped_boxplot(ax, data, x_labels, approaches, colors=colors, labels=labels)

    ax.set_xlabel("DAG depth")
    ax.set_ylabel(ylabel or metric.replace("_", " ").capitalize())
    if title:
        ax.set_title(title)

    fig.tight_layout()
    if pdf_save_path:
        fig.savefig(pdf_save_path)
        print(f"Saved: {pdf_save_path}")
    if png_save_path:
        fig.savefig(png_save_path)
        print(f"Saved: {png_save_path}")
    return fig, ax


# ============================================================
# Figure: metric vs. (beta, alpha)
# ============================================================

def format_beta_alpha_label(beta, alpha):
    return rf"$\beta{{=}}{beta:g}$" "\n" rf"$\alpha{{=}}{alpha:g}$"


def plot_metric_vs_params(results_by_params_and_approach, metric="total_reward",
                           ylabel=None, title=None, colors=None, labels=None,
                           figsize=(10, 5), save_path=None, param_label_fn=None):
    """
    Box plot of `metric` across sims, grouped by (beta, alpha) setting on
    the x-axis (one cluster per setting) and by approach within each
    cluster -- this is the plot that lets you compare standard BAMCP vs.
    early-stopping BAMCP at each human-parameter setting side by side,
    instead of needing separate figures per approach.

    :param results_by_params_and_approach: {(beta, alpha): {approach: [result_dict, ...]}}
    :param param_label_fn: optional custom function (beta, alpha) -> str
        for x-tick labels. Defaults to a two-line "beta=.. / alpha=.." label.
    """
    param_label_fn = param_label_fn or format_beta_alpha_label
    params, approaches, data = _extract_metric_grid(results_by_params_and_approach, metric)
    x_labels = [param_label_fn(b, a) for (b, a) in params]

    fig, ax = plt.subplots(figsize=figsize)
    grouped_boxplot(ax, data, x_labels, approaches, colors=colors, labels=labels)

    ax.set_xlabel(r"Human parameters ($\beta$, $\alpha$)")
    ax.set_ylabel(ylabel or metric.replace("_", " ").capitalize())
    if title:
        ax.set_title(title)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    return fig, ax


# ============================================================
# Belief comparison: e.g. small-beta vs. large-beta regime
# ============================================================

def plot_belief_regime_comparison(regime_results, param_index=ALPHA_INDEX, true_value=None,
                                   opidx=0, belief_key="belief", display_res=None,
                                   figsize=None, pdf_save_path=None, png_save_path=None,
                                   cmap="viridis", integer_ticks_only=True, true_value_label=None):
    regimes = list(regime_results.items()) if isinstance(regime_results, dict) else list(regime_results)
    n = len(regimes)
    figsize = figsize or (5.5 * n, 4.5)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    param_name = r"$b(\beta)$" if param_index == BETA_INDEX else r"$b(\alpha)$"
    ims = []

    for ax, (regime_label, results_list) in zip(axes, regimes):
        values, matrix = average_marginal_matrices(results_list, param_index, opidx=opidx, belief_key=belief_key)
        if display_res is not None:
            values, matrix = coarsen_marginal(values, matrix, display_res)

        im = plot_belief_heatmaps(
            ax, values, matrix, true_value=true_value,
            title=regime_label, ylabel=param_name, cmap=cmap,
            integer_ticks_only=integer_ticks_only, true_value_label=true_value_label,
        )
        ims.append(im)

    fig.colorbar(ims[-1], ax=list(axes), fraction=0.03, pad=0.02)
    fig.suptitle(f"{param_name} belief concentration across human-rationality regimes")

    if pdf_save_path:
        fig.savefig(pdf_save_path)
        print(f"Saved: {pdf_save_path}")
    if png_save_path:
        fig.savefig(png_save_path, dpi=150)
        print(f"Saved: {png_save_path}")

    return fig, axes


def plot_belief_regime_comparison_both(regimes, opidx=0, belief_key="belief",
                                        display_res_beta=None, display_res_alpha=None,
                                        share_vmax_per_row=True, figsize=None,
                                        pdf_save_path=None, png_save_path=None,
                                        cmap="viridis", integer_ticks_only=True,
                                        hspace=0.15, wspace=0.15, height_ratios=(1, 1)):
    """
    ...(existing docstring)...

    :param height_ratios: relative height of (beta row, alpha row). Default
        (2, 1) matches beta's [0,10] grid vs alpha's [0,5] grid, so both
        rows show the same visual scale (pixels per unit) rather than being
        stretched to equal height regardless of their actual value range.
        Adjust if your grid bounds change.
    """
    n = len(regimes)
    figsize = figsize or (3 * n, 3)

    fig, axes = plt.subplots(
        2, n, figsize=figsize, squeeze=False, sharex=True, sharey='row',
        gridspec_kw={"height_ratios": list(height_ratios)},
    )
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    beta_data, alpha_data = [], []
    for regime in regimes:
        results_list = regime["results"]
        beta_values, beta_matrix = average_marginal_matrices(results_list, BETA_INDEX, opidx=opidx, belief_key=belief_key)
        alpha_values, alpha_matrix = average_marginal_matrices(results_list, ALPHA_INDEX, opidx=opidx, belief_key=belief_key)

        if display_res_beta is not None:
            beta_values, beta_matrix = coarsen_marginal(beta_values, beta_matrix, display_res_beta)
        if display_res_alpha is not None:
            alpha_values, alpha_matrix = coarsen_marginal(alpha_values, alpha_matrix, display_res_alpha)

        beta_data.append((beta_values, beta_matrix))
        alpha_data.append((alpha_values, alpha_matrix))

    beta_vmax = max(m.max() for _, m in beta_data) if share_vmax_per_row else None
    alpha_vmax = max(m.max() for _, m in alpha_data) if share_vmax_per_row else None

    beta_ims, alpha_ims = [], []
    for col, (regime, (beta_values, beta_matrix), (alpha_values, alpha_matrix)) in enumerate(
        zip(regimes, beta_data, alpha_data)
    ):
        beta_true = regime.get("true_beta")
        beta_label = fr"$\beta_0={beta_true}$" if beta_true is not None else None
        im_b = plot_belief_heatmaps(
        axes[0][col], beta_values, beta_matrix, true_value=beta_true,
        title=regime["label"], ylabel=r"$b(\beta)$" if col == 0 else None,
        cmap=cmap, vmax=beta_vmax,
        integer_ticks_only=integer_ticks_only, tick_step=2, true_value_label=beta_label,
        )
        axes[0][col].set_xlabel("")
        beta_ims.append(im_b)

        alpha_true = regime.get("true_alpha")
        alpha_label = fr"$\alpha_0={alpha_true}$" if alpha_true is not None else None
        im_a = plot_belief_heatmaps(
            axes[1][col], alpha_values, alpha_matrix, true_value=alpha_true,
            title=None, ylabel=r"$b(\alpha)$" if col == 0 else None,
            cmap=cmap, vmax=alpha_vmax,
            integer_ticks_only=integer_ticks_only, true_value_label=alpha_label,
        )
        axes[1][col].set_xlabel("")  # suppress per-panel label; one shared label added below instead
        alpha_ims.append(im_a)

    #fig.colorbar(beta_ims[-1], ax=list(axes[0]), fraction=0.03, pad=0.02)
    #fig.colorbar(alpha_ims[-1], ax=list(axes[1]), fraction=0.03, pad=0.02)
    fig.colorbar(beta_ims[-1], ax=list(axes[0]), fraction=0.025, pad=0.015)
    fig.colorbar(alpha_ims[-1], ax=list(axes[1]), fraction=0.025, pad=0.015)

    for col in range(n):
        axes[1][col].tick_params(axis='x', rotation=30)

    fig.supxlabel("Step", fontsize=12, y=-0.05)  # one centered label for the whole figure instead
    #fig.suptitle(r"$\beta$ and $\alpha$ belief concentration across regimes")

    if pdf_save_path:
        fig.savefig(pdf_save_path)
        print(f"Saved: {pdf_save_path}")
    if png_save_path:
        fig.savefig(png_save_path, dpi=150)
        print(f"Saved: {png_save_path}")

    return fig, axes
# ============================================================
# Figure: FIHP switch step vs. MDP length
# ============================================================

def _extract_stop_step_by_depth(results_by_depth_and_approach, approach=Approach.BAMCP_ES):
    """
    Pull FIHP's switch timestep (results["stop_info"]["stop_step"]) out of
    {depth: {approach: [result_dict, ...]}}, keeping only trials that
    actually triggered early stopping (stop_info["stopped_early"] is True).
    Trials that ran to completion without switching don't have a real
    "switch step" and are silently excluded (with a printed count) rather
    than polluting the plot with their final step count.
    """

    depths = list(results_by_depth_and_approach.keys())
    data = {approach: []}

    for depth in depths: 
        results_list = results_by_depth_and_approach[depth].get(approach, [])
        steps, n_never_switched = [], 0
        for r in results_list:
            stop_info = r.get("stop_info")
            if stop_info is None: 
                continue 
            if stop_info.get("stopped_early"):
                steps.append(stop_info["stop_step"])
            else:
                n_never_switched += 1

        data[approach].append(np.array(steps, dtype=float))

    return depths, data


def plot_switch_step_vs_length(results_by_depth_and_approach, approach=Approach.BAMCP_ES,
                                ylabel=None, title=None, color=None, figsize=(8, 5),
                                pdf_save_path=None, png_save_path=None):
    """
    Box plot of FIHP's switch timestep (when the Fisher-information
    stopping criterion triggered) vs. MDP depth. Same visual style/x-axis
    as plot_metric_vs_length(metric="total_time").

    :param results_by_depth_and_approach: {depth: {approach: [result_dict, ...]}}
        only the FIHP entries are used (standard BAMCP has no stop_info).
    :param approach: which approach's stop_info to read; defaults to FIHP
        (Approach.BAMCP_ES), since that's the only one with stop_info.
    """

    depths, data = _extract_stop_step_by_depth(results_by_depth_and_approach, approach=approach)
    x_labels = [str(d) for d in depths]

    fig, ax = plt.subplots(figsize=figsize)
    grouped_boxplot(ax, data, x_labels, [approach], colors=({_approach_key(approach): color} if color else None))

    ax.set_xlabel("MDP depth")
    ax.set_ylabel(ylabel or "Switch step")
    if title:
        ax.set_title(title)

    fig.tight_layout()
    if pdf_save_path:
        fig.savefig(pdf_save_path)
        print(f"Saved: {pdf_save_path}")
    if png_save_path:
        fig.savefig(png_save_path)
        print(f"Saved: {png_save_path}")
    return fig, ax


# ============================================================
# Figure: FIHP human-observation count at switch vs. MDP length
# ============================================================

def _extract_human_obs_by_depth(results_by_depth_and_approach, approach=Approach.BAMCP_ES):
    """
    Like _extract_stop_step_by_depth, but counts only the human-operator
    steps up to the switch, rather than total environment steps.

    stop_info["stop_step"] counts *every* environment step (human-operator
    AND autonomous-operator), but the Fisher information -- and therefore
    the stopping criterion -- only accumulates on steps where the human
    operator acted. If the policy routes more steps through autonomy at
    one depth/parameter setting than another, stop_step is inflated for
    reasons unrelated to how fast information actually accumulates. This
    reconstructs the "real" count: how many human-operator observations
    had actually come in by the time FIHP switched.

    Uses is_auto_vec, which is already saved in every result dict
    (no rerun needed) -- unpack_action() sets is_auto=1 exactly on steps
    issued to the autonomous operator, so summing (1 - is_auto) up to
    stop_step gives the human-observation count.
    """
    depths = list(results_by_depth_and_approach.keys())
    data = {approach: []}

    for depth in depths:
        results_list = results_by_depth_and_approach[depth].get(approach, [])
        counts, n_skipped = [], 0
        for r in results_list:
            stop_info = r.get("stop_info")
            is_auto_vec = r.get("is_auto")
            if stop_info is None or is_auto_vec is None or not stop_info.get("stopped_early"):
                n_skipped += 1
                continue
            stop_step = stop_info["stop_step"]
            human_obs = sum(1 for a in is_auto_vec[:stop_step] if not a)
            counts.append(human_obs)
        if n_skipped:
            print(f"[human_obs] depth={depth}: {n_skipped}/{len(results_list)} "
                  f"trials skipped (never switched, or missing fields)")
        data[approach].append(np.array(counts, dtype=float))

    return depths, data


def plot_switch_step_vs_human_obs(results_by_depth_and_approach, approach=Approach.BAMCP_ES,
                                   figsize=(10, 5), pdf_save_path=None, png_save_path=None):
    """
    Side-by-side comparison: switch step (total env steps, includes
    autonomous-operator steps) vs. human observations at switch (the
    information-relevant count), both grouped by depth, sharing a y-axis
    scale so you can see directly how much of the depth trend in
    stop_step survives once autonomous steps are excluded.
    """
    depths, step_data = _extract_stop_step_by_depth(results_by_depth_and_approach, approach=approach)
    _, obs_data = _extract_human_obs_by_depth(results_by_depth_and_approach, approach=approach)
    x_labels = [str(d) for d in depths]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    grouped_boxplot(axes[0], step_data, x_labels, [approach])
    axes[0].set_xlabel("MDP depth")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Switch step (total env steps)")

    grouped_boxplot(axes[1], obs_data, x_labels, [approach])
    axes[1].set_xlabel("MDP depth")
    axes[1].set_title("Human observations at switch")

    fig.tight_layout()
    if pdf_save_path:
        fig.savefig(pdf_save_path)
        print(f"Saved: {pdf_save_path}")
    if png_save_path:
        fig.savefig(png_save_path)
        print(f"Saved: {png_save_path}")
    return fig, axes


def plot_reward_and_time_combined_boxplot(flat_results, depths, pdf_save_path=None, png_save_path=None,
                                           colors=None, figsize=(8, 4)):
    fig, (ax_reward, ax_time) = plt.subplots(1, 2, figsize=figsize, sharex=False)
    fig.subplots_adjust(wspace=0.02)  # push panels together

    ...  # unchanged boxplot-drawing code

    # move the time panel's y-axis to the right side, so it reads left-to-right
    # as "reward axis | shared plot area | time axis" rather than two separate boxes
    ax_time.yaxis.tick_right()
    ax_time.yaxis.set_label_position("right")

    default_colors = {Approach.BAMCP: "steelblue", Approach.BAMCP_ES: "darkorange"}
    colors = colors or default_colors
    labels = {Approach.BAMCP: "BAMCP", Approach.BAMCP_ES: "FIHP"}
    approaches = [Approach.BAMCP, Approach.BAMCP_ES]

    reward_data = {a: [] for a in approaches}
    time_data = {a: [] for a in approaches}
    for d in depths:
        for a in approaches:
            reward_data[a].append(np.array([r["total_reward"] for r in flat_results[d][a]]))
            time_data[a].append(np.array([r["total_time"] for r in flat_results[d][a]]))

    x_labels = [str(d) for d in depths]
    grouped_boxplot(ax_reward, reward_data, x_labels, approaches,
                     colors={a.value: colors[a] for a in approaches})
    grouped_boxplot(ax_time, time_data, x_labels, approaches,
                     colors={a.value: colors[a] for a in approaches})

    #ax_reward.set_title("Reward")
    ax_reward.set_ylabel("Total reward")
    #ax_time.set_title("Wall-clock time")
    ax_time.set_ylabel("Wall-clock time (s)")

    for ax in (ax_reward, ax_time):
        ax.set_xlabel("")  # suppress per-panel label -- shared label added below instead
        ax.grid(alpha=0.3, axis="y")
        ax.get_legend().remove() if ax.get_legend() else None  # remove any per-panel legend

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[a]) for a in approaches]
    fig.legend(handles, [labels[a] for a in approaches], loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1), frameon=True)

    fig.supxlabel("DAG depth", fontsize=12, y=0.12)
    #fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.subplots_adjust(bottom=0.25)  # leave more space for shared xlabel
    fig.subplots_adjust(wspace=0)  # small horizontal spacing

    if pdf_save_path:
        fig.savefig(pdf_save_path)
        print(f"Saved: {pdf_save_path}")
    if png_save_path:
        fig.savefig(png_save_path, dpi=150)
        print(f"Saved: {png_save_path}")
    return fig, (ax_reward, ax_time)

