import numpy as np
from domains.layered_mdp.layered_mdp import LayeredMDP
from operators.context import OperatorContext
from operators.utils import OperatorParams, OperatorType
from simulations.approach import Approach
from simulations.scoring_functions import generate_scoring_function
from simulations.utils import load_all_sims, get_grid_tag, make_grid
from simulations.plotting import plot_reward_and_belief_heatmaps, plot_final_reward_time_and_belief_heatmaps
from simulations.runners import run_standard_bamcp, run_standard_vi, run_early_stopping_bamcp
from simulations.domain_transitions import build_auto_domain_transitions
from simulations.paper_plots import (
    set_paper_style, plot_metric_vs_length, plot_metric_vs_params,
    plot_belief_regime_comparison_both,
)


# ============================================================
# Shared low-level helpers -- same simulation pipeline either way.
# Everything about WHICH sweep runs, and when, lives in the two
# functions below instead; these two are just "build a domain" and
# "run both BAMCP variants once", nothing sweep-specific.
# ============================================================

def build_domain_and_contexts(depth, num_actions, p_success, seed, utility_scale=1):
    """ Build the LayeredMDP domain + autonomous-operator context for one depth. """
    domain = LayeredMDP.generate_layered_mdp(depth=depth, num_actions=num_actions, seed=seed)
    cost_nominals = {a: 0 for a in domain.actions}
    Phi_nom = generate_scoring_function(domain, scale=utility_scale, seed=seed)

    auto_domain_transitions = build_auto_domain_transitions(domain, p_success)
    auto_context = OperatorContext(
        category=OperatorType.AUTO,
        n=1,  # single performance state, matches the working code path
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=auto_domain_transitions,
        cost_nominals=cost_nominals,  # only takes effect once you apply the from_context fix
    )

    return domain, Phi_nom, cost_nominals, [auto_context]


def run_both_bamcp_variants(config, domain, Phi_nom, cost_nominals, auto_op_contexts,
                             true_beta, true_alpha, seed, num_sims, beta_grid, alpha_grid,
                             is_toy, fn_app, grid_tag, save_results, n_jobs, debug):
    """ Run standard + early-stopping BAMCP for one (depth, true_beta, true_alpha) point. """
    results_bamcp = run_standard_bamcp(
        config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, seed, num_sims,
        auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy,
        fn_app=fn_app, grid_tag=grid_tag, save_results=save_results, n_jobs=n_jobs, debug=debug,
    )
    results_bamcp_es = run_early_stopping_bamcp(
        config=config, domain=domain, Phi_nom=Phi_nom, true_beta=true_beta, true_alpha=true_alpha,
        cost_nominals=cost_nominals, seed=seed, num_sims=num_sims, auto_op_contexts=auto_op_contexts,
        beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag,
        save_results=save_results, n_jobs=n_jobs, debug=debug,
    )
    return {Approach.BAMCP: results_bamcp, Approach.BAMCP_ES: results_bamcp_es}


# ============================================================
# Sweep 1: reward / time vs. MDP depth, fixed (true_beta, true_alpha).
# Produces fig_reward_vs_length.pdf and fig_time_vs_length.pdf.
# ============================================================

def run_depth_sweep(parameter_pairs):
    SEED = 5
    MAX_DEPTH = 50

    num_actions = 3
    num_sims = 20
    # p_success is the baseline success rate for the autonomous operator
    p_success = 0.75

    depths = [100, 150, 200]

    lb_beta, ub_beta = 0, 10
    lb_alpha, ub_alpha = 0, 5
    grid_res = 0.1
    beta_grid = make_grid(lb_beta, ub_beta, grid_res)
    alpha_grid = make_grid(lb_alpha, ub_alpha, grid_res)
    grid_tag = get_grid_tag(lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res)

    n_jobs = -1
    save_results = True
    is_toy = False
    debug = False
    utility_scale=1
    if utility_scale > 1:
        fn_app = f"_uscale{utility_scale}_psuccess{p_success}"
    else:
        fn_app = f"_{p_success}"


    set_paper_style()

    for true_beta, true_alpha in parameter_pairs:
        results_by_depth_and_approach = {}
        for depth in depths:
            domain, Phi_nom, cost_nominals, auto_op_contexts = build_domain_and_contexts(
                depth, num_actions, p_success, SEED, utility_scale=utility_scale
            )
            config = {
                "seed": SEED, "max_depth": MAX_DEPTH, "depth": depth,
                "num_actions": num_actions, "true_beta": true_beta, "true_alpha": true_alpha,
                "num_sims": num_sims,
            }
            results_by_depth_and_approach[depth] = run_both_bamcp_variants(
                config, domain, Phi_nom, cost_nominals, auto_op_contexts,
                true_beta, true_alpha, SEED, num_sims, beta_grid, alpha_grid,
                is_toy, fn_app, grid_tag, save_results, n_jobs, debug
            )

        depths_tag = "-".join(str(d) for d in depths)
        tag = f"depths{depths_tag}_b{true_beta}_a{true_alpha}_uscale{utility_scale}_psuccess{p_success}_n{num_sims}"

        plot_metric_vs_length(
            results_by_depth_and_approach, metric="total_reward",
            ylabel="Total reward", save_path=f"simulations/layered_mdp_simulations/paper_plots/fig_reward_vs_depth_{tag}.pdf",
        )
        plot_metric_vs_length(
            results_by_depth_and_approach, metric="total_time",
            ylabel="Wall-clock time (s)", save_path=f"simulations/layered_mdp_simulations/paper_plots/fig_time_vs_depth_{tag}.pdf",
        )


# ============================================================
# Sweep 2: reward / time vs. TRUE (beta, alpha), fixed depth.
# Produces fig_reward_vs_params.pdf, fig_time_vs_params.pdf, and
# fig_belief_regimes.pdf (which just reuses two of this sweep's points).
# ============================================================

def run_params_sweep(param_values):
    SEED = 5
    MAX_DEPTH = 50

    num_actions = 3
    num_sims = 20
    p_success = 0.75

    depth = 100

    lb_beta, ub_beta = 0, 10
    lb_alpha, ub_alpha = 0, 5
    grid_res = 0.1
    beta_grid = make_grid(lb_beta, ub_beta, grid_res)
    alpha_grid = make_grid(lb_alpha, ub_alpha, grid_res)
    grid_tag = get_grid_tag(lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res)

    n_jobs = -1
    save_results = True
    is_toy = False
    debug = False
    utility_scale=5
    if utility_scale > 1:
        fn_app = f"_uscale{utility_scale}_psuccess{p_success}"
    else:
        fn_app = f"_{p_success}"
    fn_app = f"_{p_success}"

    set_paper_style()

    domain, Phi_nom, cost_nominals, auto_op_contexts = build_domain_and_contexts(
        depth, num_actions, p_success, SEED,
    )

    results_by_params_and_approach = {}
    for (beta_val, alpha_val) in param_values:
        config = {
            "seed": SEED, "max_depth": MAX_DEPTH, "depth": depth,
            "num_actions": num_actions, "true_beta": beta_val, "true_alpha": alpha_val,
            "num_sims": num_sims,
        }
        results_by_params_and_approach[(beta_val, alpha_val)] = run_both_bamcp_variants(
            config, domain, Phi_nom, cost_nominals, auto_op_contexts,
            beta_val, alpha_val, SEED, num_sims, beta_grid, alpha_grid,
            is_toy, fn_app, grid_tag, save_results, n_jobs, debug,
        )

    tag = f"depth{depth}_uscale{utility_scale}_psuccess{p_success}_n{num_sims}"

    plot_metric_vs_params(
        results_by_params_and_approach, metric="total_reward",
        ylabel="Total reward", save_path=f"simulations/layered_mdp_simulations/paper_plots/fig_reward_vs_params_{tag}.pdf",
    )

    # plot_belief_regime_comparison_both(
    #     [
    #         {
    #             "label": r"Low $\beta$ ($\beta=2,\alpha=1$)",
    #             "results": results_by_params_and_approach[(2, 1)][Approach.BAMCP],
    #             "true_beta": 2, "true_alpha": 1,
    #         },
    #         {
    #             "label": r"High $\beta$ ($\beta=5,\alpha=0.1$)",
    #             "results": results_by_params_and_approach[(5, 0.1)][Approach.BAMCP],
    #             "true_beta": 5, "true_alpha": 0.1,
    #         },
    #     ],
    #     save_path="simulations/layered_mdp_simulations/paper_plots/fig_belief_regimes_{tag}.pdf",
    # )

    return results_by_params_and_approach


if __name__ == "__main__":
    parameter_pairs = [(0.5, 1.2), (1, 1.2), (1.5, 1.2), (2, 1.2), (2.5, 1.2), (3, 1.2), (3.5, 1.2), (4, 1.2), (4.5, 1.2), (5,1.2)]
    #run_depth_sweep(parameter_pairs)
    run_params_sweep(parameter_pairs)