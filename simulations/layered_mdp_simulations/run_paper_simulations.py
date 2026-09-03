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
    plot_belief_regime_comparison_both, plot_switch_step_vs_length, plot_switch_step_vs_human_obs
)
from simulations.analysis import summarize_switch_stats


def flatten_layouts(results_by_depth_and_approach):
    """
    Convert {depth: {approach: {lidx: [trial_dict, ...]}}} into the flat
    {depth: {approach: [trial_dict, ...]}} shape the existing paper_plots
    functions expect, by concatenating across layouts. Use this right
    before calling plot_metric_vs_length / plot_switch_step_vs_length /
    summarize_switch_stats etc. -- keep the nested version around
    separately if you want to do a per-layout consistency check first.
    """
    flat = {}
    for depth, per_approach in results_by_depth_and_approach.items():
        flat[depth] = {}
        for approach, per_layout in per_approach.items():
            flat[depth][approach] = [
                trial for layout_results in per_layout.values() for trial in layout_results
            ]
    return flat

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
        n=1,  # single performance state
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=auto_domain_transitions,
        cost_nominals=cost_nominals,  # only takes effect once you apply the from_context fix
    )

    return domain, Phi_nom, cost_nominals, [auto_context]

def run_both_bamcp_variants(config, domain, Phi_nom, cost_nominals, auto_op_contexts,
                             true_beta, true_alpha, seed, num_sims, beta_grid, alpha_grid,
                             window, tol, is_toy, is_cluster, fn_app, grid_tag, save_results, n_jobs, debug, return_results=False):
    """ Run standard + early-stopping BAMCP for one (depth, true_beta, true_alpha) point. """
    results_bamcp = run_standard_bamcp(
        config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, seed, num_sims,
        auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, is_cluster=is_cluster,
        fn_app=fn_app, grid_tag=grid_tag, save_results=save_results, n_jobs=n_jobs, debug=debug, return_results=return_results
    )


    es_fn_app = fn_app + f"_window{window}_eps{tol}"
    results_bamcp_es = run_early_stopping_bamcp(
        config=config, domain=domain, Phi_nom=Phi_nom, true_beta=true_beta, true_alpha=true_alpha,
        cost_nominals=cost_nominals, seed=seed, num_sims=num_sims, auto_op_contexts=auto_op_contexts,
        beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, is_cluster=is_cluster, fn_app=es_fn_app, grid_tag=grid_tag,
        save_results=save_results, n_jobs=n_jobs, debug=debug, window=window, tol=tol, return_results=return_results
    )

    if not return_results:
        return None 
    
    return {Approach.BAMCP: results_bamcp, Approach.BAMCP_ES: results_bamcp_es}


# ============================================================
# Sweep 1: reward / time vs. MDP depth, fixed (true_beta, true_alpha).
# Produces fig_reward_vs_length.pdf and fig_time_vs_length.pdf.
# ============================================================

def run_depth_sweep(parameter_pairs):
    #MAX_DEPTH = 50

    depths = [100, 150, 200]
    MAX_DEPTH = max(depths)
    depths = [100, 150, 200]
    num_actions = 3
    num_sims = 20
    num_layouts = 10
    # p_success is the baseline success rate for the autonomous operator
    p_success = 0.75

    
    SEED = 5
    num_trials = 1000
    window = 20 
    tol = 0.05

    lb_beta, ub_beta = 0, 10
    lb_alpha, ub_alpha = 0, 5
    grid_res = 0.1
    beta_grid = make_grid(lb_beta, ub_beta, grid_res)
    alpha_grid = make_grid(lb_alpha, ub_alpha, grid_res)
    grid_tag = get_grid_tag(lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res)

    n_jobs = 20
    save_results = True
    is_toy = False
    debug = False
    is_cluster = False
    utility_scale=1
    if utility_scale > 1:
        fn_app = f"_uscale{utility_scale}_psuccess{p_success}"
    else:
        fn_app = f"_{p_success}"

    base_fn_app = fn_app

    set_paper_style()

    layout_seeds = [SEED + 1000 * i for i in range(num_layouts)]

    for true_beta, true_alpha in parameter_pairs:
        results_by_depth_and_approach = {}
        for depth in depths:
            results_by_depth_and_approach[depth] = {
                Approach.BAMCP: {},
                Approach.BAMCP_ES: {},
            }

            for (lidx, layout_seed) in enumerate(layout_seeds):
                print("Layout: ", lidx+1)
                domain, Phi_nom, cost_nominals, auto_op_contexts = build_domain_and_contexts(depth, num_actions, p_success, utility_scale=utility_scale, seed=layout_seed)
                                                                                            
                fn_app = base_fn_app + f"_L{lidx+1}"

                config = {
                "layout_seed": layout_seed,
                "seed": SEED, 
                "max_depth": MAX_DEPTH, 
                "depth": depth,
                "num_actions": num_actions, 
                "true_beta": true_beta, 
                "true_alpha": true_alpha,
                "num_sims": num_sims,
                "num_trials": num_trials,
                }

                layout_results = run_both_bamcp_variants(
                        config=config, domain=domain, Phi_nom=Phi_nom, cost_nominals=cost_nominals,
                        auto_op_contexts=auto_op_contexts, true_beta=true_beta, true_alpha=true_alpha,
                        seed=SEED, num_sims=num_sims, beta_grid=beta_grid, alpha_grid=alpha_grid,
                        window=window, tol=tol, is_toy=is_toy, is_cluster=is_cluster, fn_app=fn_app, grid_tag=grid_tag,
                        save_results=save_results, n_jobs=n_jobs, debug=debug,
                    )

                #results_by_depth_and_approach[depth][Approach.BAMCP][lidx] = layout_results[Approach.BAMCP]
                #results_by_depth_and_approach[depth][Approach.BAMCP_ES][lidx] = layout_results[Approach.BAMCP_ES]

        #depths_tag = "-".join(str(d) for d in depths)
        #tag = f"depths{depths_tag}_b{true_beta}_a{true_alpha}_uscale{utility_scale}_psuccess{p_success}_n{num_sims}_window{window}_eps{tol}_num_trials{config["num_trials"]}_algdepth_{config["max_depth"]}_numlayouts{num_layouts}"

        #flat_results = flatten_layouts(results_by_depth_and_approach)

        # plot_metric_vs_length(flat_results, metric="total_reward", ylabel="Total reward", 
        #                       pdf_save_path=f"simulations/layered_mdp_simulations/paper_plots/fig_reward_vs_depth_{tag}.pdf",
        #                       png_save_path=f"simulations/layered_mdp_simulations/presentation_plots/fig_reward_vs_depth_{tag}.png")

        # plot_metric_vs_length(flat_results, metric="total_time",
        #      ylabel="Wall-clock time (s)", pdf_save_path=f"simulations/layered_mdp_simulations/paper_plots/fig_time_vs_depth_{tag}.pdf",
        #      png_save_path=f"simulations/layered_mdp_simulations/presentation_plots/fig_time_vs_depth_{tag}.png",
        # )

        # plot_switch_step_vs_length(
        #     flat_results, 
        #     ylabel="Switch step",
        #     pdf_save_path=f"simulations/layered_mdp_simulations/paper_plots/fig_switchstep_vs_depth_{tag}.pdf",
        #     png_save_path=f"simulations/layered_mdp_simulations/presentation_plots/fig_switchstep_vs_depth_{tag}.png",
        # )

        # plot_switch_step_vs_human_obs(
        #    flat_results,  # e.g. your b=0.5 dict
        #    pdf_save_path=f"simulations/layered_mdp_simulations/paper_plots/fig_switchcompare_vs_depth_{tag}.pdf",
        #    png_save_path=f"simulations/layered_mdp_simulations/presentation_plots/fig_switchcompare_vs_depth_{tag}.png",
        # )

        # df = summarize_switch_stats(flat_results)  # e.g. your b=0.5 dict
        # print(df.round(1))
        # # or straight to a LaTeX table:
        # df.round(1).to_latex(f"simulations/layered_mdp_simulations/latex_tables/switch_stats_table_{tag}.tex", index=False)

        for depth in depths:
            for approach in [Approach.BAMCP, Approach.BAMCP_ES]:
                print(f"depth={depth}, {approach}")
                for lidx, layout_trials in results_by_depth_and_approach[depth][approach].items():
                    rewards = [r["total_reward"] for r in layout_trials]
                    print(f"  layout {lidx}: mean reward = {np.mean(rewards):.1f}  (n={len(rewards)})")
            #input("Next depth...")

        # plot_metric_vs_length(
        #     results_by_depth_and_approach, metric="total_reward",
        #     ylabel="Total reward", pdf_save_path=f"simulations/layered_mdp_simulations/paper_plots/fig_reward_vs_depth_{tag}.pdf",
        #     png_save_path=f"simulations/layered_mdp_simulations/presentation_plots/fig_reward_vs_depth_{tag}.png",
        # )
        # plot_metric_vs_length(
        #     results_by_depth_and_approach, metric="total_time",
        #     ylabel="Wall-clock time (s)", pdf_save_path=f"simulations/layered_mdp_simulations/paper_plots/fig_time_vs_depth_{tag}.pdf",
        #     png_save_path=f"simulations/layered_mdp_simulations/presentation_plots/fig_time_vs_depth_{tag}.png",
        # )
        # plot_switch_step_vs_length(
        #     results_by_depth_and_approach, 
        #     ylabel="Switch step",
        #     pdf_save_path=f"simulations/layered_mdp_simulations/paper_plots/fig_switchstep_vs_depth_{tag}.pdf",
        #     png_save_path=f"simulations/layered_mdp_simulations/presentation_plots/fig_switchstep_vs_depth_{tag}.png",
        # )
        #plot_switch_step_vs_human_obs(
        #    results_by_depth_and_approach,  # e.g. your b=0.5 dict
        #    pdf_save_path=f"simulations/layered_mdp_simulations/paper_plots/fig_switchcompare_vs_depth_{tag}.pdf",
        #    png_save_path=f"simulations/layered_mdp_simulations/presentation_plots/fig_switchcompare_vs_depth_{tag}.png",
        #)

        # df = summarize_switch_stats(results_by_depth_and_approach)  # e.g. your b=0.5 dict
        # print(df.round(1))
        # # or straight to a LaTeX table:
        # df.round(1).to_latex(f"simulations/layered_mdp_simulations/latex_tables/switch_stats_table_{tag}.tex", index=False)



# ============================================================
# Sweep 2: reward / time vs. TRUE (beta, alpha), fixed depth.
# Produces fig_reward_vs_params.pdf, fig_time_vs_params.pdf, and
# fig_belief_regimes.pdf (which just reuses two of this sweep's points).
# ============================================================

def run_params_sweep(param_values):
    num_actions = 3
    num_sims = 20
    p_success = 0.75

    depth = 100

    MAX_DEPTH = depth
    SEED = 5
    num_trials = 1000
    window = 20 
    tol = 0.05

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
            "seed": SEED, 
            "max_depth": MAX_DEPTH, 
            "depth": depth,
            "num_actions": num_actions, 
            "true_beta": beta_val, 
            "true_alpha": alpha_val,
            "num_sims": num_sims,
            "num_trials": num_trials
        }
        results_by_params_and_approach[(beta_val, alpha_val)] = run_both_bamcp_variants(
            config, domain, Phi_nom, cost_nominals, auto_op_contexts,
            beta_val, alpha_val, SEED, num_sims, beta_grid, alpha_grid,
            is_toy, fn_app, grid_tag, save_results, n_jobs, debug, window=window, tol=tol
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
    #parameter_pairs = [(0.5, 1.2), (5, 1.2)]
    #parameter_pairs = [(0.5, 1.2), (5, 1.2)]
    parameter_pairs = [(0.5, 1.2), (1.5, 1.2), (5,1.2)]
    #parameter_pairs = [(0.5, 1.2)]
    run_depth_sweep(parameter_pairs)
    #run_params_sweep(parameter_pairs)