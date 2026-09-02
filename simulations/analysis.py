import numpy as np
from boltzmann_sas.globals import DEFER
from bamcp.bamcp import BAMCPSolver
from bamcp.history import History
from belief.belief import FreqBelief, JointGridBelief
from simulations.utils import Approach
import pandas as pd

def extract_step_records(history):
    """
    Unpack a History object's flat (s0, (issued,executed), s1, (issued,executed), s2, ...)
    structure into a list of (state, issued_action, executed_action, next_state) tuples,
    one per real step.
    """
    items = history.items
    records = []
    # items[0] = s0, items[1] = (issued, executed), items[2] = s1, items[3] = (issued, executed), ...
    for i in range(0, len(items) - 1, 2):
        state = items[i]
        issued_action, executed_action = items[i + 1]
        next_state = items[i + 2]
        records.append((state, issued_action, executed_action, next_state))
    return records


def get_best_reward_action(domain, state):
    """
    The enabled action at `state` leading to the highest-reward next state.

    NOTE: this assumes deterministic transitions (true for LayeredMDP) and uses
    domain.get_next_states(state, action) + domain.state_rewards to score each
    option. Double-check these two calls match your current LayeredMDP API --
    I'm working from an earlier version of model.py in this conversation and
    can't confirm it hasn't since changed.
    """
    best_action, best_reward = None, -np.inf
    for action in domain.enabled_actions[state]:
        next_state = domain.get_next_state(state, action)
        reward = domain.state_rewards.get(next_state, 0)
        if reward > best_reward:
            best_reward = reward
            best_action = action
    return best_action


def advice_accuracy(results_list, domain):
    """
    Fraction of ADVICE steps (across all sims in results_list) where the
    issued advice matched the objectively best-reward action at that state --
    i.e. how good BAMCP's advice actually is, independent of whether the
    operator complied with it.

    Returns (accuracy, n_advice_steps).
    """
    correct = 0
    total = 0

    for r in results_list:
        records = extract_step_records(r["history"])
        for state, issued_action, executed_action, next_state in records:
            opidx, advice = issued_action
            if advice == DEFER:   # adjust to match your actual DEFER sentinel if different
                continue
            domain_state = state[0]
            best_action = get_best_reward_action(domain, domain_state)
            total += 1
            if advice == best_action:
                correct += 1

    accuracy = correct / total if total > 0 else float("nan")
    return accuracy, total


def advice_vs_actual_reward(results_list, domain):
    """
    For each ADVICE step: reward of the ADVISED action (what you'd get if
    followed) vs the ACTUAL reward obtained (whatever was executed). This
    isolates advice quality's direct contribution to the followed-vs-deviated
    reward gap, without going through a proxy like "was advice the objectively
    best action."
    """
    advised_rewards, actual_rewards, followed_flags = [], [], []

    for r in results_list:
        records = extract_step_records(r["history"])
        for state, issued_action, executed_action, next_state in records:
            opidx, advice = issued_action
            if advice == DEFER:
                continue
            domain_state = state[0]

            advised_probs = domain.T_det[0][(domain_state, advice)]
            advised_next = max(advised_probs, key=advised_probs.get)
            advised_reward = domain.state_rewards.get(advised_next, 0)

            actual_reward = domain.state_rewards.get(next_state[0], 0)

            advised_rewards.append(advised_reward)
            actual_rewards.append(actual_reward)
            followed_flags.append(advice == executed_action[1])

    return np.array(advised_rewards), np.array(actual_rewards), np.array(followed_flags)

def compare_policy_across_alpha(bamdp, state, beta_fixed, alpha_values, max_depth, t=2000, num_trials=0):
    """
    For a fixed state and beta, temporarily pin alpha's belief to a point mass
    at each candidate value, run a FRESH BAMCP search (new solver, new tree,
    root-only history so get_next_action takes its fresh-tree branch rather
    than reusing an old tree from a previous alpha), and record which action
    the REAL search-based policy actually chooses.
 
    This replaces calling solver.rollout() directly, which only tests the
    random rollout policy used deep in the tree for value backup -- not
    BAMCP's actual UCB-driven decision (see conversation).
 
    NOTE: mutates bamdp.belief temporarily, restores it afterward. Run this
    interactively/standalone, not inside a parallel joblib worker, to avoid
    any risk of another process reading bamdp.belief mid-swap.
    """
    operator = bamdp.boltzmann_operators[0]   # assumes a single Boltzmann human operator, matching the rest of this codebase
    bidx = bamdp.boltzmann_operator_indices[operator]
    original_belief = bamdp.belief[bidx]
 
    results = {}
    try:
        for alpha_test in alpha_values:
            bamdp.belief[bidx] = JointGridBelief({(beta_fixed, alpha_test): 1.0})
            solver = BAMCPSolver(bamdp, max_depth=max_depth, t=t, num_trials=num_trials)
            fresh_history = History(items=(state,))  # length 1 -> forces get_next_action's fresh-tree branch
            action = solver.get_next_action(fresh_history)
            results[alpha_test] = action
            print(f"alpha={alpha_test}: chosen action = {action}")
    finally:
        bamdp.belief[bidx] = original_belief  # always restore, even if a call above raises
 
    return results

if __name__ == "__main__":
    # Example usage -- adjust imports/paths to match how you already load results elsewhere
    #
    # from simulations.utils import load_all_sims
    # from simulations.approach import Approach
    #
    # results_bamcp = load_all_sims(..., approach=Approach.BAMCP, ...)
    # acc, n = advice_accuracy(results_bamcp, domain)
    # print(f"Advice accuracy: {acc:.2%} (n={n} advice steps)")
    pass


def summarize_switch_stats(results_by_depth_and_approach, fihp_approach=Approach.BAMCP_ES,
                            bamcp_approach=Approach.BAMCP):
    """
    Table-friendly summary of FIHP's switch behavior, for reporting in the
    paper as a table rather than a boxplot (see plot_switch_step_vs_length /
    plot_human_obs_vs_length for the exploratory distributional versions).

    For each depth, computes across FIHP trials that triggered early
    stopping:
      - mean / median switch_step        (total env steps at switch)
      - mean / median human_obs_at_switch (human-operator steps only,
        excludes autonomous-operator steps -- see _extract_human_obs_by_depth)
      - mean / median switch_frac        (switch_step / matched standard-BAMCP
        total_steps for the same trial index, i.e. same seed/domain
        realization -- "FIHP switched after X% of the steps plain BAMCP
        needed to finish the same scenario"). Requires BAMCP and BAMCP_ES to
        share the sim_seed = seed*10_000 + sim scheme (true for
        runners.py's current implementation), so trial i of each approach
        is directly comparable.
      - n_stopped / n_total              how many trials actually triggered
        early stopping vs. ran to completion (a low ratio here means most
        of the "switch" statistics below are based on a small, possibly
        unrepresentative subset of trials -- worth checking before reading
        too much into the means)

    :param results_by_depth_and_approach: {depth: {approach: [result_dict, ...]}},
        the same dict used throughout paper_plots.py. Must contain both
        fihp_approach and bamcp_approach entries for the fraction column;
        if bamcp_approach is missing at a depth, switch_frac is left as NaN
        for that row rather than raising.
    :return: pandas DataFrame, one row per depth, ready for
        df.to_latex(...) or df.to_csv(...).
    """
    rows = []

    for depth, per_approach in results_by_depth_and_approach.items():
        fihp_results = per_approach.get(fihp_approach, [])
        bamcp_results = per_approach.get(bamcp_approach, [])
        # index-aligned with fihp_results, since both approaches use the
        # same sim_seed = seed*10_000 + sim scheme for a given sim index
        bamcp_total_steps = [r.get("total_steps") for r in bamcp_results]

        switch_steps, human_obs, fracs = [], [], []
        n_total = len(fihp_results)
        n_stopped = 0

        for i, r in enumerate(fihp_results):
            stop_info = r.get("stop_info")
            is_auto_vec = r.get("is_auto")
            if stop_info is None or not stop_info.get("stopped_early"):
                continue
            n_stopped += 1

            stop_step = stop_info["stop_step"]
            switch_steps.append(stop_step)

            if is_auto_vec is not None:
                human_obs.append(sum(1 for a in is_auto_vec[:stop_step] if not a))

            if i < len(bamcp_total_steps) and bamcp_total_steps[i]:
                fracs.append(stop_step / bamcp_total_steps[i])

        def _mean_median(vals):
            arr = np.asarray(vals, dtype=float)
            return (np.nan, np.nan) if len(arr) == 0 else (arr.mean(), np.median(arr))

        switch_mean, switch_median = _mean_median(switch_steps)
        obs_mean, obs_median = _mean_median(human_obs)
        frac_mean, frac_median = _mean_median(fracs)

        rows.append({
            "depth": depth,
            "n_stopped": n_stopped,
            "n_total": n_total,
            "switch_step_mean": switch_mean,
            "switch_step_median": switch_median,
            "human_obs_mean": obs_mean,
            "human_obs_median": obs_median,
            "switch_frac_mean": frac_mean,
            "switch_frac_median": frac_median,
        })

    df = pd.DataFrame(rows).sort_values("depth").reset_index(drop=True)
    return df


def summarize_switch_stats_by_beta(results_by_params_and_approach, fihp_approach=Approach.BAMCP_ES,
                                    bamcp_approach=Approach.BAMCP):
    """
    Same as summarize_switch_stats, but grouped by (beta, alpha) setting
    rather than depth -- use with results_by_params_and_approach
    ({(beta, alpha): {approach: [...]}}), matching plot_metric_vs_params.
    """
    rows = []
    for (beta, alpha), per_approach in results_by_params_and_approach.items():
        # reuse the depth-keyed summarizer with a dummy "depth" slot
        sub = summarize_switch_stats({(beta, alpha): per_approach},
                                      fihp_approach=fihp_approach, bamcp_approach=bamcp_approach)
        sub = sub.rename(columns={"depth": "beta_alpha"})
        sub["beta"] = beta
        sub["alpha"] = alpha
        rows.append(sub.drop(columns=["beta_alpha"]))

    df = pd.concat(rows, ignore_index=True).sort_values(["beta", "alpha"]).reset_index(drop=True)
    return df