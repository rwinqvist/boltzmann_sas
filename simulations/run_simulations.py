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
    plot_belief_regime_comparison,
)
from simulations.approach import Approach
from simulations.utils import load_all_sims


def main():
    SEED = 5
    MAX_DEPTH = 50 

    # layered-mdp domain info
    depth = 100
    num_actions = 3
    true_beta = 5
    true_alpha = 1.2
    num_sims = 20

    # belief representation
    lb_beta, ub_beta = 0, 10
    lb_alpha, ub_alpha = 0, 5
    grid_res = 0.1
    beta_grid = make_grid(lb_beta, ub_beta, grid_res)
    alpha_grid = make_grid(lb_alpha, ub_alpha, grid_res)
    grid_tag = get_grid_tag(lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res)


    # MDP (used params)
    planning_beta = 10
    planning_alpha = 0.1
    used_params = [(10, 0.1), (5, 0.1), (3, 0.1), (2, 1), (2, 4), (2, 5)]
    #used_params = [(2, 1)]

    # simulation config
    config = {
        "seed": SEED,
        "max_depth": MAX_DEPTH,
        "depth": depth,
        "num_actions": num_actions,
        "true_beta": true_beta,
        "true_alpha": true_alpha,
        "num_sims": num_sims,
    }

    # n_jobs=-1 uses all available cores. Drop to e.g. os.cpu_count() - 1
    # if you want to keep a core free while these run in the background.
    n_jobs = -1

    fn_app = ""
    save_results = True
    is_toy = True
    debug = False 
    manual_debug = False
    if not save_results:
        fn_app ="test"
        print("WARNING! NOT SAVING RESULTS!")
    if is_toy:
        print("WARNING! You're saving under TOY")
    if manual_debug:
        print("WARNING! Debug mode on")
        fn_app = "debug"
        save_results = False
        num_sims = 1
        n_jobs = 1


    domain = LayeredMDP.generate_layered_mdp(depth=depth, num_actions=num_actions, seed=SEED)
    cost_nominals = {a: 0 for a in domain.actions}

    # generate nominal human scoring function 
    Phi_nom = generate_scoring_function(domain, seed=SEED)

    # build autonomous operator
    p_success = 0.75   # tune this to create a different autonomous operator
    fn_app += f"_{p_success}"
    auto_domain_transitions = build_auto_domain_transitions(domain, p_success)

    auto_context = OperatorContext(
        category=OperatorType.AUTO,
        n=1,                              # single performance state, matches the working code path
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=auto_domain_transitions,
        cost_nominals=cost_nominals,       # only takes effect once you apply the from_context fix above
    )

    auto_op_contexts = []

    # # run standard bamcp
    results_bamcp = run_standard_bamcp(
        config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, SEED, num_sims,
        auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag,
        save_results=save_results, n_jobs=n_jobs, debug=debug,
    )

    # # run early stopping bamcp 
    results_bamcp_es = run_early_stopping_bamcp(config=config, domain=domain, Phi_nom=Phi_nom, true_beta=true_beta, true_alpha=true_alpha,
                                                cost_nominals=cost_nominals, seed=SEED, num_sims=num_sims,
                                                auto_op_contexts=auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid,
                                                is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag, save_results=save_results,
                                                n_jobs=n_jobs, debug=debug)

    results_by_approach = {
        Approach.BAMCP.value: results_bamcp,
        Approach.BAMCP_ES.value: results_bamcp_es
    }


    # for planning_beta, planning_alpha in used_params:
    #     results_vi = run_standard_vi(config=config, domain=domain, Phi_nom=Phi_nom, true_beta=true_beta,
    #                                 true_alpha=true_alpha, planning_beta=planning_beta, planning_alpha=planning_alpha, 
    #                                 cost_nominals=cost_nominals, seed=SEED, num_sims=num_sims, auto_op_contexts=auto_op_contexts, 
    #                                 is_toy=is_toy, fn_app=fn_app, save_results=save_results, n_jobs=n_jobs, debug=debug)
    #     results_by_approach[f"Approach.VI: b={planning_beta}, a={planning_alpha}"] = results_vi
        
   

    #plot_reward_and_belief_heatmaps(results_by_approach, config)
    plot_final_reward_time_and_belief_heatmaps(results_by_approach, config)


def run_layered_mdp_paper_sims():

    SEED = 5
    MAX_DEPTH = 50 

    # layered-mdp domain info
    #depth = 100
    num_actions = 3
    true_beta = 5
    true_alpha = 1.2
    num_sims = 20

    # belief representation
    lb_beta, ub_beta = 0, 10
    lb_alpha, ub_alpha = 0, 5
    grid_res = 0.1
    beta_grid = make_grid(lb_beta, ub_beta, grid_res)
    alpha_grid = make_grid(lb_alpha, ub_alpha, grid_res)
    grid_tag = get_grid_tag(lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res)


    # MDP (used params)
    planning_beta = 10
    planning_alpha = 0.1
    used_params = [(10, 0.1), (5, 0.1), (3, 0.1), (2, 1), (2, 4), (2, 5)]
    #used_params = [(2, 1)]


    # n_jobs=-1 uses all available cores. Drop to e.g. os.cpu_count() - 1
    # if you want to keep a core free while these run in the background.
    n_jobs = -1

    fn_app = ""
    save_results = True
    is_toy = False
    debug = False 
    manual_debug = False
    if not save_results:
        fn_app ="test"
        print("WARNING! NOT SAVING RESULTS!")
    if is_toy:
        print("WARNING! You're saving under TOY")
    if manual_debug:
        print("WARNING! Debug mode on")
        fn_app = "debug"
        save_results = False
        num_sims = 1
        n_jobs = 1

    set_paper_style()

    param_pairs = [(5, 1.2)]
    depths = [50, 100, 150, 200]
    depths = [100, 150]

    # ============================================================
    # Sweep 1: reward / time vs. MDP depth
    # One figure per (true_beta, true_alpha) setting in param_pairs -- right
    # now that's just the one default setting, but the loop already supports
    # adding more later (each gets its own tagged save_path so they don't
    # overwrite each other).
    # ============================================================
    for true_beta, true_alpha in param_pairs:
        results_by_depth_and_approach = {}

        for depth in depths:
            # simulation config
            config = {
                "seed": SEED,
                "max_depth": MAX_DEPTH,
                "depth": depth,
                "num_actions": num_actions,
                "true_beta": true_beta,
                "true_alpha": true_alpha,
                "num_sims": num_sims,
            }

            domain = LayeredMDP.generate_layered_mdp(depth=depth, num_actions=num_actions, seed=SEED)
            cost_nominals = {a: 0 for a in domain.actions}

            # generate nominal human scoring function 
            Phi_nom = generate_scoring_function(domain, seed=SEED)

            # build autonomous operator
            p_success = 0.75   # tune this to create a different autonomous operator
            fn_app += f"_{p_success}"
            auto_domain_transitions = build_auto_domain_transitions(domain, p_success)

            auto_context = OperatorContext(
                category=OperatorType.AUTO,
                n=1,                              # single performance state, matches the working code path
                actions=domain.actions,
                enabled_actions=domain.enabled_actions,
                domain_transitions=auto_domain_transitions,
                cost_nominals=cost_nominals,       # only takes effect once you apply the from_context fix above
            )

            auto_op_contexts = [auto_context]

            # # run standard bamcp
            results_bamcp = run_standard_bamcp(
                config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, SEED, num_sims,
                auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag,
                save_results=save_results, n_jobs=n_jobs, debug=debug,
            )

            # # run early stopping bamcp 
            results_bamcp_es = run_early_stopping_bamcp(config=config, domain=domain, Phi_nom=Phi_nom, true_beta=true_beta, true_alpha=true_alpha,
                                                        cost_nominals=cost_nominals, seed=SEED, num_sims=num_sims,
                                                        auto_op_contexts=auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid,
                                                        is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag, save_results=save_results,
                                                        n_jobs=n_jobs, debug=debug)

            results_by_depth_and_approach[depth] = {
                Approach.BAMCP: results_bamcp,
                Approach.BAMCP_ES: results_bamcp_es
            }

        tag = f"b{true_beta}_a{true_alpha}"
        plot_metric_vs_length(
            results_by_depth_and_approach, metric="total_reward",
            ylabel="Total reward", save_path=f"fig_reward_vs_length_{tag}.pdf",
        )
        plot_metric_vs_length(
            results_by_depth_and_approach, metric="total_time",
            ylabel="Wall-clock time (s)", save_path=f"fig_time_vs_length_{tag}.pdf",
        )


if __name__ == "__main__":
    run_layered_mdp_paper_sims()