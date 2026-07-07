import numpy as np
import matplotlib.pyplot as plt
from simulations.approach import Approach

BETA_INDEX = 0
ALPHA_INDEX = 1

def extract_belief_series(results, param, opidx=0, clip=None):
    """ Extract a single belief stat as a list over steps. """
    series = [step[opidx][param] for step in results["belief_stats"]]
    if clip is not None:
        series = [np.clip(v, clip[0], clip[1]) for v in series]
    return series


def extract_beta_error(results, true_beta, opidx=0, clip=None):
    series = [abs(step[opidx]["beta_mean"] - true_beta)
              for step in results["belief_stats"]]
    if clip is not None:
        series = [np.clip(v, clip[0], clip[1]) for v in series]
    return series


def pad_and_stack(list_of_lists):
    """ Pad lists to same length and stack into 2D array (n_sims x n_steps). """
    max_len = max(len(s) for s in list_of_lists)
    padded = [s + [s[-1]] * (max_len - len(s)) for s in list_of_lists]
    return np.array(padded, dtype=float)


def get_grid_values(belief_history, param_index, opidx=0):
    """ Sorted list of distinct grid values for one parameter (beta or alpha), taken from step 0. """
    first_belief = belief_history[0][opidx]
    return sorted({key[param_index] for key in first_belief.keys()})

def average_marginal_matrices(results_list, param_index, opidx=0, belief_key="belief"):
    """
    Average marginal-belief heatmaps across multiple sims (a single approach).
    Sims of different lengths are padded by repeating each sim's final column,
    matching the convention used by pad_and_stack() elsewhere in this module.
 
    :param belief_key: key in each results dict holding the full belief history
        (list of {opidx: {(beta, alpha): p}} per step). Defaults to "belief" —
        NOT "belief_stats", which holds collapsed summary statistics only.
    """
    values = get_grid_values(results_list[0][belief_key], param_index, opidx=opidx)
 
    matrices = [
        compute_marginal_matrix(r[belief_key], param_index, opidx=opidx, values=values)[1]
        for r in results_list
    ]
 
    max_steps = max(m.shape[1] for m in matrices)
    padded = []
    for m in matrices:
        if m.shape[1] < max_steps:
            pad_width = max_steps - m.shape[1]
            last_col = m[:, -1:].repeat(pad_width, axis=1)
            m = np.hstack([m, last_col])
        padded.append(m)
 
    mean_matrix = np.mean(padded, axis=0)
    return values, mean_matrix

def plot_belief_heatmap(ax, values, matrix, true_value=None, title=None, ylabel=None, cmap="viridis"):
    """ Plot one parameter's belief-over-time heatmap on a given axis. """
    im = ax.imshow(
        matrix, aspect="auto", origin="lower", cmap=cmap, vmin=0.0, vmax=1.0,
        extent=[0, matrix.shape[1], -0.5, len(values) - 0.5],
    )
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(values)
    ax.set_xlabel("Step", fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=12)
 
    if true_value is not None and true_value in values:
        row = values.index(true_value)
        ax.axhline(row, color="red", linestyle="--", linewidth=1.5, label=f"True value={true_value}")
        ax.legend(fontsize=9, loc="upper right")
 
    return im
 
 
def plot_belief_heatmaps(results_by_approach, config, opidx=0, belief_key="belief", save_path=None):
    """
    Plot beta and alpha marginal-belief heatmaps (probability mass by value, over
    simulation step), one row per approach, averaged across that approach's sims.
 
    :param belief_key: key in each results dict holding the full belief history.
        Must be the full grid distribution (e.g. "belief"), not "belief_stats"
        (which holds collapsed summary statistics only, not the distribution).
    """
    true_beta = config["true_beta"]
    true_alpha = config["true_alpha"]
 
    approaches = list(results_by_approach.keys())
    n_rows = len(approaches)
 
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 4.5 * n_rows), squeeze=False)
 
    for row, approach in enumerate(approaches):
        results_list = results_by_approach[approach]
        label = approach.value if hasattr(approach, "value") else approach
 
        beta_values, beta_matrix = average_marginal_matrices(results_list, BETA_INDEX, opidx=opidx, belief_key=belief_key)
        alpha_values, alpha_matrix = average_marginal_matrices(results_list, ALPHA_INDEX, opidx=opidx, belief_key=belief_key)
 
        im_b = plot_belief_heatmap(
            axes[row][0], beta_values, beta_matrix, true_value=true_beta,
            title=f"{label}: β belief over time", ylabel="β",
        )
        im_a = plot_belief_heatmap(
            axes[row][1], alpha_values, alpha_matrix, true_value=true_alpha,
            title=f"{label}: α belief over time", ylabel="α",
        )
        fig.colorbar(im_b, ax=axes[row][0], label="P(value)")
        fig.colorbar(im_a, ax=axes[row][1], label="P(value)")
 
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_reward_and_belief_heatmaps(results_by_approach, config, opidx=0, belief_key="belief", colors=None, save_path=None):
    """
    Plot layout:
        Row 0:    [cumulative reward, all approaches overlaid | DEFER rate, all approaches overlaid]
        Row 1..n: [β heatmap | α heatmap] for each approach
    """
    true_beta = config["true_beta"]
    true_alpha = config["true_alpha"]
 
    approaches = list(results_by_approach.keys())
    n_rows = len(approaches)
 
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if colors is None:
        colors = {}
 
    fig, axes = plt.subplots(n_rows + 1, 2, figsize=(13, 4.5 * (n_rows + 1)))
 
    # --- row 0: shared reward + DEFER rate comparison across approaches ---
    for i, approach in enumerate(approaches):
        results_list = results_by_approach[approach]
        label = approach.value if hasattr(approach, "value") else approach
        color = colors.get(label, default_cycle[i % len(default_cycle)])
 
        cum_rewards = pad_and_stack([r["cum_rewards"] for r in results_list])
        plot_mean_std(axes[0][0], cum_rewards, label, color)
 
        defer_series = pad_and_stack([r["is_defer"] for r in results_list])
        plot_mean_std(axes[0][1], defer_series, label, color, plot_std=False)
 
    axes[0][0].set_xlabel("Step", fontsize=12)
    axes[0][0].set_ylabel("Cumulative reward", fontsize=12)
    axes[0][0].set_title("Cumulative reward", fontsize=12)
    axes[0][0].legend(fontsize=10)
    axes[0][0].grid(alpha=0.3)
 
    axes[0][1].set_xlabel("Step", fontsize=12)
    axes[0][1].set_ylabel("DEFER rate", fontsize=12)
    axes[0][1].set_title("DEFER rate over steps", fontsize=12)
    axes[0][1].legend(fontsize=10)
    axes[0][1].grid(alpha=0.3)
    axes[0][1].set_ylim(-0.05, 1.05)
 
    # --- rows 1..n: per-approach belief heatmaps ---
    for i, approach in enumerate(approaches):
        results_list = results_by_approach[approach]
        label = approach.value if hasattr(approach, "value") else approach
 
        beta_values, beta_matrix = average_marginal_matrices(results_list, BETA_INDEX, opidx=opidx, belief_key=belief_key)
        alpha_values, alpha_matrix = average_marginal_matrices(results_list, ALPHA_INDEX, opidx=opidx, belief_key=belief_key)
 
        im_b = plot_belief_heatmap(
            axes[i + 1][0], beta_values, beta_matrix, true_value=true_beta,
            title=f"{label}: β belief over time", ylabel="β",
        )
        im_a = plot_belief_heatmap(
            axes[i + 1][1], alpha_values, alpha_matrix, true_value=true_alpha,
            title=f"{label}: α belief over time", ylabel="α",
        )
        fig.colorbar(im_b, ax=axes[i + 1][0], label="P(value)")
        fig.colorbar(im_a, ax=axes[i + 1][1], label="P(value)")
 
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def compute_marginal_matrix(belief_history, param_index, opidx=0, values=None):
    """
    Build a (n_values x n_steps) matrix of marginal probability mass over time
    for one parameter (beta or alpha), from a single sim's full belief history.
 
    :param belief_history: list over steps of {opidx: {(beta, alpha): p}} —
        i.e. results["belief"], the full grid distribution at each step.
        NOT results["belief_stats"], which only holds collapsed summary
        statistics (mean/VaR/CVaR/optimistic/pessimistic), not the distribution.
    :param param_index: BETA_INDEX or ALPHA_INDEX
    """
    if values is None:
        values = get_grid_values(belief_history, param_index, opidx=opidx)
    value_to_row = {v: i for i, v in enumerate(values)}
 
    n_steps = len(belief_history)
    matrix = np.zeros((len(values), n_steps))
 
    for t, step_belief in enumerate(belief_history):
        belief = step_belief[opidx]
        marginal = np.zeros(len(values))
        for key, p in belief.items():
            marginal[value_to_row[key[param_index]]] += p
        matrix[:, t] = marginal
 
    return values, matrix


def plot_mean_std(ax, data_2d, label, color, alpha=0.2, plot_std=True):
    """ Plot mean ± 1 std across rows of data_2d. """
    mean = np.mean(data_2d, axis=0)
    std = np.std(data_2d, axis=0)
    x = np.arange(len(mean))
    ax.plot(x, mean, label=label, color=color, linewidth=2)
    if plot_std:
        ax.fill_between(x, mean - std, mean + std, alpha=alpha, color=color)


def plot_results(results_by_approach, config, n_warmstart, colors=None, labels=None, save_path=None):
    """
    Plot 4 panels:
        1. Cumulative reward over steps
        2. Beta estimate (mean) over steps
        3. Alpha estimate (mean) over steps
        4. DEFER rate over steps
    """
    true_beta = config["true_beta"]
    true_alpha = config["true_alpha"]

    colors = {
            Approach.BAMCP: "steelblue",
            Approach.NAIVE_BAYESIAN_WARMSTART: "darkorange",
            Approach.NAIVE_FREQ_WARMSTART: "red",
        }

    labels = {
            Approach.BAMCP: "Standard BAMCP",
            Approach.NAIVE_BAYESIAN_WARMSTART: f"Naive warm-start (n={n_warmstart})",
            Approach.NAIVE_FREQ_WARMSTART: f"Naive freq warm-start (n={n_warmstart})",
        }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for approach, results_list in results_by_approach.items():
        key = approach
        color = colors.get(key, "gray")
        label = labels.get(key, key)

        # --- cumulative reward ---
        cum_rewards = pad_and_stack([r["cum_rewards"] for r in results_list])
        plot_mean_std(axes[0], cum_rewards, label, color)

        # --- beta — mean with posterior std as shading ---
        beta_mean = pad_and_stack([extract_belief_series(r, "beta_mean", clip=(0, 20)) for r in results_list])
        beta_post_std = pad_and_stack([extract_belief_series(r, "beta_std", clip=(0, 20)) for r in results_list])
        mean_b = np.mean(beta_mean, axis=0)
        shade_b = np.mean(beta_post_std, axis=0)
        x = np.arange(len(mean_b))
        axes[1].plot(x, mean_b, label=label, color=color, linewidth=2)
        axes[1].fill_between(x, mean_b - shade_b, mean_b + shade_b, alpha=0.2, color=color)

        # --- alpha — mean with posterior std as shading ---
        alpha_mean = pad_and_stack([extract_belief_series(r, "alpha_mean") for r in results_list])
        alpha_post_std = pad_and_stack([extract_belief_series(r, "alpha_std") for r in results_list])
        mean_a = np.mean(alpha_mean, axis=0)
        shade_a = np.mean(alpha_post_std, axis=0)
        axes[2].plot(x, mean_a, label=label, color=color, linewidth=2)
        axes[2].fill_between(x, mean_a - shade_a, mean_a + shade_a, alpha=0.2, color=color)

        # --- defer rate ---
        defer_series = pad_and_stack([r["is_defer"] for r in results_list])
        plot_mean_std(axes[3], defer_series, label, color, plot_std=False)

    # --- formatting ---
    axes[0].set_xlabel("Step", fontsize=12)
    axes[0].set_ylabel("Cumulative reward", fontsize=12)
    axes[0].set_title("Cumulative reward", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(true_beta, color="black", linestyle="--", linewidth=1.5, label=f"True β={true_beta}")
    axes[1].set_xlabel("Step", fontsize=12)
    axes[1].set_ylabel("β estimate", fontsize=12)
    axes[1].set_title("β belief mean", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    axes[2].axhline(true_alpha, color="black", linestyle="--", linewidth=1.5, label=f"True α={true_alpha}")
    axes[2].set_xlabel("Step", fontsize=12)
    axes[2].set_ylabel("α estimate", fontsize=12)
    axes[2].set_title("α belief mean", fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(alpha=0.3)

    if n_warmstart > 0:
        axes[3].axvline(n_warmstart, color="darkorange", linestyle=":", linewidth=1.5, label="Warm-start end")
    axes[3].set_xlabel("Step", fontsize=12)
    axes[3].set_ylabel("DEFER rate", fontsize=12)
    axes[3].set_title("DEFER rate over steps", fontsize=12)
    axes[3].legend(fontsize=10)
    axes[3].grid(alpha=0.3)
    axes[3].set_ylim(-0.05, 1.05)

    plt.suptitle(
        f"True β={true_beta}, α={true_alpha} | warm-start n={n_warmstart} | "
        f"depth={config['depth']}, seed={config['seed']}",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_results_with_error(results_by_approach, config, n_warmstart, colors=None, labels=None, save_path=None):
    """
    Plot 5 panels:
        1. Cumulative reward over steps
        2. Beta estimate (mean) over steps
        3. Beta absolute error over steps
        4. Alpha estimate (mean) over steps
        5. DEFER rate over steps
    """
    true_beta = config["true_beta"]
    true_alpha = config["true_alpha"]

    colors = {
            Approach.BAMCP: "steelblue",
            Approach.NAIVE_BAYESIAN_WARMSTART: "darkorange",
            Approach.NAIVE_FREQ_WARMSTART: "red",
        }

    labels = {
            Approach.BAMCP: "Standard BAMCP",
            Approach.NAIVE_BAYESIAN_WARMSTART: f"Naive warm-start (n={n_warmstart})",
            Approach.NAIVE_FREQ_WARMSTART: f"Naive freq warm-start (n={n_warmstart})",
        }

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()

    for approach, results_list in results_by_approach.items():
        key = approach.value
        color = colors.get(key, "gray")
        label = labels.get(key, key)

        cum_rewards = pad_and_stack([r["cum_rewards"] for r in results_list])
        plot_mean_std(axes[0], cum_rewards, label, color)

        beta_series = pad_and_stack([extract_belief_series(r, "beta_mean", clip=(0, 10)) for r in results_list])
        plot_mean_std(axes[1], beta_series, label, color)

        beta_error = pad_and_stack([extract_beta_error(r, true_beta, clip=(0, 10)) for r in results_list])
        plot_mean_std(axes[2], beta_error, label, color)

        alpha_series = pad_and_stack([extract_belief_series(r, "alpha_mean") for r in results_list])
        plot_mean_std(axes[3], alpha_series, label, color)

        defer_series = pad_and_stack([r["is_defer"] for r in results_list])
        plot_mean_std(axes[4], defer_series, label, color)

    axes[0].set_xlabel("Step", fontsize=12)
    axes[0].set_ylabel("Cumulative reward", fontsize=12)
    axes[0].set_title("Cumulative reward", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(true_beta, color="black", linestyle="--", linewidth=1.5, label=f"True β={true_beta}")
    axes[1].set_xlabel("Step", fontsize=12)
    axes[1].set_ylabel("β estimate", fontsize=12)
    axes[1].set_title("β belief mean", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    axes[2].axhline(0, color="black", linestyle="--", linewidth=1.5, label="Zero error")
    axes[2].set_xlabel("Step", fontsize=12)
    axes[2].set_ylabel("|β_hat - β_true|", fontsize=12)
    axes[2].set_title("β absolute error", fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(alpha=0.3)
    if n_warmstart > 0:
        axes[2].axvline(n_warmstart, color="darkorange", linestyle=":", linewidth=1.5, label="Warm-start end")
        axes[2].legend(fontsize=10)

    axes[3].axhline(true_alpha, color="black", linestyle="--", linewidth=1.5, label=f"True α={true_alpha}")
    axes[3].set_xlabel("Step", fontsize=12)
    axes[3].set_ylabel("α estimate", fontsize=12)
    axes[3].set_title("α belief mean", fontsize=12)
    axes[3].legend(fontsize=10)
    axes[3].grid(alpha=0.3)

    if n_warmstart > 0:
        axes[4].axvline(n_warmstart, color="darkorange", linestyle=":", linewidth=1.5, label="Warm-start end")
    axes[4].set_xlabel("Step", fontsize=12)
    axes[4].set_ylabel("DEFER rate", fontsize=12)
    axes[4].set_title("DEFER rate over steps", fontsize=12)
    axes[4].legend(fontsize=10)
    axes[4].grid(alpha=0.3)
    axes[4].set_ylim(-0.05, 1.05)

    axes[5].set_visible(False)  # hide unused 6th panel

    plt.suptitle(
        f"True β={true_beta}, α={true_alpha} | warm-start n={n_warmstart} | "
        f"depth={config['depth']}, seed={config['seed']}",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()