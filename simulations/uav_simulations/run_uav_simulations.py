from domains.uav.uav_domain import UAVDomain
from domains.uav.utils import get_layouts
from operators.context import OperatorContext
from operators.utils import OperatorType
from simulations.utils import get_grid_tag, make_grid
from simulations.approach import Approach
from simulations.scoring_functions import generate_random_uav_scoring_function, generate_progression_biased_uav_scoring_function
from simulations.domain_transitions import build_auto_domain_transitions
from simulations.runners import run_early_stopping_bamcp, run_standard_bamcp
from simulations.plotting import plot_reward_and_belief_heatmaps
from domains.uav.utils import get_layouts
from itertools import product
from simulations.paper_plots import (
    set_paper_style, plot_metric_vs_length, plot_metric_vs_params,
    plot_belief_regime_comparison_both, plot_switch_step_vs_length, plot_switch_step_vs_human_obs
)
from simulations.analysis import summarize_switch_stats


def flatten_layouts(results_by_approach):
    """
    Convert {approach: {lidx: [trial_dict, ...]}} into {approach: [trial_dict, ...]}
    by concatenating trials across layouts.
    """
    return {
        approach: [trial for layout_results in per_layout.values() for trial in layout_results]
        for approach, per_layout in results_by_approach.items()
    }


def build_domain_and_contexts(size, p_obs, layout, p_success, seed, utility_scale=1):
    """ Build the LayeredMDP domain + autonomous-operator context for one depth. """

    # build domain 
    domain = UAVDomain.from_map(size=size, p_obs=p_obs, layout=layout)
    cost_nominals = {a: 0 for a in domain.actions}
    Phi_nom = generate_progression_biased_uav_scoring_function(domain, scale=utility_scale, seed=seed)

    auto_domain_transitions = build_auto_domain_transitions(domain, p_success)
    auto_context = OperatorContext(
        category=OperatorType.AUTO,
        n=1,                              # single performance state
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=auto_domain_transitions,
        cost_nominals=cost_nominals,   
    )

    #auto_op_contexts = [auto_context]
    auto_op_contexts = []

    return domain, Phi_nom, cost_nominals, auto_op_contexts



def run_both_bamcp_variants(config, domain, Phi_nom, cost_nominals, auto_op_contexts,
                             true_beta, true_alpha, seed, num_sims, beta_grid, alpha_grid,
                             window, tol, is_toy, fn_app, grid_tag, save_results, n_jobs, debug):
    """ Run standard + FIHP for one (depth, true_beta, true_alpha) point. """
    results_bamcp = run_standard_bamcp(
        config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, seed, num_sims,
        auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy,
        fn_app=fn_app, grid_tag=grid_tag, save_results=save_results, n_jobs=n_jobs, debug=debug,
    )


    es_fn_app = fn_app + f"_window{window}_eps{tol}"
    results_bamcp_es = run_early_stopping_bamcp(
        config=config, domain=domain, Phi_nom=Phi_nom, true_beta=true_beta, true_alpha=true_alpha,
        cost_nominals=cost_nominals, seed=seed, num_sims=num_sims, auto_op_contexts=auto_op_contexts,
        beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, fn_app=es_fn_app, grid_tag=grid_tag,
        save_results=save_results, n_jobs=n_jobs, debug=debug, window=window, tol=tol
    )
    return {Approach.BAMCP: results_bamcp, Approach.BAMCP_ES: results_bamcp_es}


def main(): 
    # uav domain info 
    size = 40
    p_obs = 0.2
    manhattan_dist = (size-1)+(size-1)

    # BAMCP and simulation configs 
    SEED = 10
    MAX_DEPTH = 2*manhattan_dist
    num_trials = 1000
    num_sims = 10 
    utility_scale = 5
    num_layouts = 1
    window = 20
    tol = 0.1
    n_jobs = 10 
    
    # set belief grids 
    lb_beta, ub_beta = 0, 10    
    lb_alpha, ub_alpha = 0, 5

    grid_res = 0.1
    beta_grid = make_grid(lb_beta, ub_beta, grid_res)
    alpha_grid = make_grid(lb_alpha, ub_alpha, grid_res)
    grid_tag = get_grid_tag(lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res)

    alpha = max(1, utility_scale-1)
    parameter_pairs = [(5, alpha)]

    # success rate of autonomous operator 
    p_success = 0.75

    layouts = get_layouts(size=size, p_obs=p_obs, num_layouts=num_layouts)

    save_results = True
    is_toy = False
    debug = False

    if utility_scale > 1:
        fn_app = f"_uscale{utility_scale}_psuccess{p_success}"
    else:
        fn_app = f"_{p_success}"

    base_fn_app = fn_app

    for true_beta, true_alpha in parameter_pairs:
        results_by_approach = {
                Approach.BAMCP: {},
                Approach.BAMCP_ES: {},
            }

        for (lidx, layout) in enumerate(layouts): 
            layout_seed = SEED + 1000 * lidx

            print("Layout: ", lidx+1)
            domain, Phi_nom, cost_nominals, auto_op_contexts = build_domain_and_contexts(size=size, p_obs=p_obs, layout=layout, p_success=p_success, utility_scale=utility_scale, seed=layout_seed)
                        
            fn_app = base_fn_app + f"_L{lidx+1}"

            config = {
                "layout_seed": layout_seed,
                "seed": SEED, 
                "max_depth": MAX_DEPTH, 
                "size": size,
                "true_beta": true_beta, 
                "true_alpha": true_alpha,
                "num_sims": num_sims,
                "num_trials": num_trials,
                "utility_scale": utility_scale,
            }

            layout_results = run_both_bamcp_variants(
                        config=config, domain=domain, Phi_nom=Phi_nom, cost_nominals=cost_nominals,
                        auto_op_contexts=auto_op_contexts, true_beta=true_beta, true_alpha=true_alpha,
                        seed=SEED, num_sims=num_sims, beta_grid=beta_grid, alpha_grid=alpha_grid,
                        window=window, tol=tol, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag,
                        save_results=save_results, n_jobs=n_jobs, debug=debug,
                    )

            results_by_approach[Approach.BAMCP][lidx] = layout_results[Approach.BAMCP]
            results_by_approach[Approach.BAMCP_ES][lidx] = layout_results[Approach.BAMCP_ES]

        tag = f"depths{size}_pobs{p_obs}_b{true_beta}_a{true_alpha}_uscale{utility_scale}_psuccess{p_success}_n{num_sims}_window{window}_eps{tol}_num_trials{config["num_trials"]}_algdepth_{config["max_depth"]}_numlayouts{num_layouts}"

        flat_results = flatten_layouts(results_by_approach)

        plot_metric_vs_length(flat_results, metric="total_reward", ylabel="Total reward", 
                              pdf_save_path=f"simulations/uav_simulations/paper_plots/fig_reward{tag}.pdf",
                              png_save_path=f"simulations/uav_simulations/presentation_plots/fig_reward{tag}.png")

        plot_metric_vs_length(flat_results, metric="total_time",
             ylabel="Wall-clock time (s)", pdf_save_path=f"simulations/uav_simulations/paper_plots/fig_time{tag}.pdf",
             png_save_path=f"simulations/uav_simulations/presentation_plots/fig_time{tag}.png",
        )

        plot_switch_step_vs_length(
            flat_results, 
            ylabel="Switch step",
            pdf_save_path=f"simulations/uav_simulations/paper_plots/fig_switchstep{tag}.pdf",
            png_save_path=f"simulations/uav_simulations/presentation_plots/fig_switchstep{tag}.png",
        )

        plot_switch_step_vs_human_obs(
           flat_results,  # e.g. your b=0.5 dict
           pdf_save_path=f"simulations/uav_simulations/paper_plots/fig_switchcompare{tag}.pdf",
           png_save_path=f"simulations/uav_simulations/presentation_plots/fig_switchcompare{tag}.png",
        )

        df = summarize_switch_stats(flat_results)  # e.g. your b=0.5 dict
        print(df.round(1))
        # or straight to a LaTeX table:
        df.round(1).to_latex(f"simulations/uav_simulations/latex_tables/switch_stats_table_{tag}.tex", index=False)



if __name__ == "__main__":
    main()