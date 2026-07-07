import numpy as np
import matplotlib.pyplot as plt
from simulations.approach import Approach


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