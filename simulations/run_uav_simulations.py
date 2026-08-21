from domains.uav.uav_domain import UAVDomain
from domains.uav.utils import get_layouts
from operators.context import OperatorContext
from operators.utils import OperatorType
from simulations.utils import get_grid_tag, make_grid
from simulations.approach import Approach
from simulations.scoring_functions import generate_uav_scoring_function
from simulations.domain_transitions import build_auto_domain_transitions
from simulations.runners import run_early_stopping_bamcp, run_standard_bamcp
from simulations.plotting import plot_reward_and_belief_heatmaps
from domains.uav.utils import get_layouts
from itertools import product
from simulations.paper_plots import (
    set_paper_style, plot_metric_vs_length, plot_metric_vs_params,
    plot_belief_regime_comparison,
)

N_JOBS = -1

def run_params_sweep(layout, idx, size, p_obs, beta_vals, alpha_vals, num_sims, utility_scale, p_success, seed, max_depth, debug, is_toy, save_results):
    # set belief grids 
    lb_beta, ub_beta = 0, max(beta_vals) * 2      
    lb_alpha, ub_alpha = 0, max(alpha_vals) * 2

    grid_res = 0.1
    beta_grid = make_grid(lb_beta, ub_beta, grid_res)
    alpha_grid = make_grid(lb_alpha, ub_alpha, grid_res)
    grid_tag = get_grid_tag(lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res)

    # build domain 
    domain = UAVDomain.from_map(size=size, p_obs=p_obs, layout=layout)
    cost_nominals = {a: 0 for a in domain.actions}
    Phi_nom = generate_uav_scoring_function(domain, scale=utility_scale, seed=seed)

    auto_domain_transitions = build_auto_domain_transitions(domain, p_success)
    auto_context = OperatorContext(
        category=OperatorType.AUTO,
        n=1,                              # single performance state
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=auto_domain_transitions,
        cost_nominals=cost_nominals,   
    )

    auto_op_contexts = [auto_context]

    tag = f"layout{idx+1}_psuccess_{p_success}_numsims{num_sims}"
    fn_app = tag 

    results_by_params_and_approach = {}

    for true_beta, true_alpha in product(beta_vals, alpha_vals):
        # simulation config
        if true_alpha > 0.1:
            continue
        config = {
            "seed": seed,
            "max_depth": max_depth,
            "size": size,
            "p_obs": p_obs,
            "true_beta": true_beta,
            "true_alpha": true_alpha,
            "num_sims": num_sims,
            "utility_scale": utility_scale,
        }

        # # run standard bamcp
        results_bamcp = run_standard_bamcp(
            config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, seed, num_sims,
            auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag,
            save_results=save_results, n_jobs=N_JOBS, debug=debug,
        )

        # # run early stopping bamcp 
        results_bamcp_es = run_early_stopping_bamcp(config=config, domain=domain, Phi_nom=Phi_nom, true_beta=true_beta, true_alpha=true_alpha,
                                                    cost_nominals=cost_nominals, seed=seed, num_sims=num_sims,
                                                    auto_op_contexts=auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid,
                                                    is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag, save_results=save_results,
                                                    n_jobs=N_JOBS, debug=debug)


        results_by_params_and_approach[(true_beta, true_alpha)] = {
                Approach.BAMCP: results_bamcp,
                Approach.BAMCP_ES: results_bamcp_es
            }

        for r in results_by_params_and_approach[(true_beta, true_alpha)][Approach.BAMCP]:
            print(r["total_steps"])
        
    plot_metric_vs_params(
        results_by_params_and_approach, metric="total_reward",
        ylabel="Total reward", save_path=f"fig_uav_reward_vs_params__{tag}.pdf",
    )
    plot_metric_vs_params(
        results_by_params_and_approach, metric="total_time",
        ylabel="Wall-clock time (s)", save_path=f"fig_uav_time_vs_params__{tag}.pdf",
    )


            



def main():
    # BAMCP and simulation configs 
    SEED = 10
    MAX_DEPTH = 50
    num_sims = 20 
    utility_scale = 1
    num_layouts = 1
    
    # uav domain info 
    size = 5
    p_obs = 0.2

    beta_vals = [0.25, 1, 4]
    alpha_vals = [0.1, 1, 4]
    alpha_vals = [0.1, 1, 4]
    #beta_vals = [4]
    #alpha_vals = [4]

    # success rate of autonomous operator 
    p_success = 0.75

    layouts = get_layouts(size=size, p_obs=p_obs, num_layouts=num_layouts)

    save_results = True
    is_toy = True
    debug = False

    for (idx, layout) in enumerate(layouts): 
        run_params_sweep(layout, idx, size, p_obs, beta_vals, alpha_vals, num_sims, utility_scale, p_success, SEED, MAX_DEPTH, debug, is_toy, save_results)





if __name__ == "__main__":
    main()