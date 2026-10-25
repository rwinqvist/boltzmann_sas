"""
Combined regime-comparison figure: ONE reward panel + ONE time panel
(side by side), with each regime (size, beta, alpha, num_trials) as an
x-axis group rather than a separate subplot. This reuses the same
grouped_boxplot pattern as make_plots_uav.py's
plot_reward_and_time_combined_boxplot_params, just generalized to let
each x-position carry a full regime instead of only (beta, alpha) at one
fixed size -- avoids the font/width-sizing fragility of faceting into a
2xN grid of small subplots.

Reads from the cached dataframe built by aggregate_uav_regimes_df.py.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# uncomment for LCSS 
#mpl.rcParams['pdf.fonttype'] = 42
#mpl.rcParams['ps.fonttype'] = 42

from simulations.approach import Approach
from simulations.uav_simulations import sweep_config as cfg
from simulations.uav_simulations.aggregate_uav_regimes import REGIMES, REGIMES_CACHE_PATH, regime_key
from simulations.paper_plots import grouped_boxplot, set_paper_style

PAPER_PLOTS_DIR = f"{cfg.CACHE_PATH}/paper_plots"
PRESENTATION_PLOTS_DIR = f"{cfg.CACHE_PATH}/presentation_plots"
os.makedirs(PAPER_PLOTS_DIR, exist_ok=True)
os.makedirs(PRESENTATION_PLOTS_DIR, exist_ok=True)

APPROACHES = [Approach.BAMCP, Approach.BAMCP_ES]
APPROACH_COLORS = {Approach.BAMCP: "steelblue", Approach.BAMCP_ES: "darkorange"}
APPROACH_LABELS = {Approach.BAMCP: "BAMCP", Approach.BAMCP_ES: "FIHP"}


def main():
    if not os.path.exists(REGIMES_CACHE_PATH):
        print(f"No cache found at {REGIMES_CACHE_PATH} -- run aggregate_uav_regimes_df.py first.")
        return
    df = pd.read_pickle(REGIMES_CACHE_PATH)

    width_scale = 0.5

    # group regimes by domain size -- each size gets its own panel with its
    # own y-axis scale, since reward/time magnitudes aren't comparable
    # across different sizes for reasons unrelated to beta/FIHP. Regimes
    # sharing a size (e.g. the two num_trials points, or the two beta
    # values) go together as x-tick groups within that size's panel.
    sizes = sorted({regime["size"] for regime in REGIMES})
    n_sizes = len(sizes)

    fig, axes = plt.subplots(2, n_sizes, figsize=(3 * n_sizes, 5))
    #fig, axes = plt.subplots(2, n_sizes, figsize=(8, 4))
    if n_sizes == 1:
        axes = axes.reshape(2, 1)

    for col_idx, size in enumerate(sizes):
        size_regimes = [r for r in REGIMES if r["size"] == size]
        # build x-tick labels directly from the actual parameter value --
        # no string parsing/stripping of the display "label" needed, and
        # no risk of "d=" or other regime-descriptive text leaking in.
        x_labels = [fr"$\beta_0={r['true_beta']}$" for r in size_regimes]

        reward_data = {a: [] for a in APPROACHES}
        time_data = {a: [] for a in APPROACHES}

        for regime in size_regimes:
            regime_df = df[df["regime_key"] == regime_key(regime)]
            for approach in APPROACHES:
                rows = regime_df[regime_df["approach"] == approach.value]
                reward_data[approach].append(rows["total_reward"].dropna().to_numpy())
                time_data[approach].append(rows["total_time"].dropna().to_numpy())

        ax_reward, ax_time = axes[0, col_idx], axes[1, col_idx]

        grouped_boxplot(ax_reward, reward_data, x_labels, APPROACHES,
                         colors={a.value: APPROACH_COLORS[a] for a in APPROACHES}, width_scale=width_scale)
        grouped_boxplot(ax_time, time_data, x_labels, APPROACHES,
                         colors={a.value: APPROACH_COLORS[a] for a in APPROACHES}, width_scale=width_scale)

        ax_reward.tick_params(labelbottom=False)  # only the bottom row needs x-tick labels

        ax_reward.set_title(f"$d={size}$")
        for ax in (ax_reward, ax_time):
            ax.tick_params(axis="x")
            if ax.get_legend():
                ax.get_legend().remove()

    axes[0, 0].set_ylabel("Total reward")
    axes[1, 0].set_ylabel("Wall-clock time (s)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=APPROACH_COLORS[a]) for a in APPROACHES]
    fig.legend(handles, [APPROACH_LABELS[a] for a in APPROACHES], loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig.subplots_adjust(wspace=0.25, hspace=0.1)

    fig.align_ylabels(axes[:, 0])

    pdf_path = f"{PAPER_PLOTS_DIR}/fig_regime_comparison_combined.pdf"
    png_path = f"{PRESENTATION_PLOTS_DIR}/fig_regime_comparison_combined.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    set_paper_style(base_size=14)
    main()