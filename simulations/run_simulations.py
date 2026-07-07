import numpy as np
from domains.layered_mdp.layered_mdp import LayeredMDP
from simulations.approach import Approach
from simulations.scoring_functions import generate_scoring_function
from simulations.utils import load_all_sims
from simulations.plotting import plot_results
from simulations.runners import run_standard_bamcp, run_bayesian_naive_warmstart


def main():
    SEED = 5
    depth = 100
    num_actions = 3
    true_beta = 0.5
    true_alpha = 1
    n_warmstart = 30
    num_sims = 8
    beta_grid = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    alpha_grid = [0, 0.5, 1.0, 1.5, 2.0]

    config = {
        "seed": SEED,
        "depth": depth,
        "num_actions": num_actions,
        "true_beta": true_beta,
        "true_alpha": true_alpha,
        "num_sims": num_sims,
    }

    fn_app = ""

    domain = LayeredMDP.generate_layered_mdp(depth=depth, num_actions=num_actions, seed=SEED)
    cost_nominals = {a: 0 for a in domain.actions}

    # generate nominal human scoring function 
    Phi_nom = generate_scoring_function(domain, seed=SEED)

    auto_op_contexts = []

    save_results = True
    is_toy = True
    if not save_results:
        print("WARNING! NOT SAVING RESULTS!")
    if is_toy:
        print("WARNING! You're saying under TOY")

    # n_jobs=-1 uses all available cores. Drop to e.g. os.cpu_count() - 1
    # if you want to keep a core free while these run in the background.
    n_jobs = -1

    run_standard_bamcp(
        config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, SEED, num_sims,
        auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, fn_app=fn_app,
        save_results=save_results, n_jobs=n_jobs,
    )

    run_bayesian_naive_warmstart(
        config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, SEED, num_sims, n_warmstart,
        auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, fn_app=fn_app,
        save_results=save_results, n_jobs=n_jobs,
    )

    results_bamcp = load_all_sims(
        domain_name=domain.domain_name, domain_tag=domain.id_tag(), approach=Approach.BAMCP,
        true_beta=true_beta, true_alpha=true_alpha, seed=SEED, num_sims=num_sims,
        num_autos=len(auto_op_contexts), is_toy=is_toy, fn_app=fn_app,
    )
    results_bayesian_warmstart = load_all_sims(
        domain_name=domain.domain_name, domain_tag=domain.id_tag(), approach=Approach.NAIVE_BAYESIAN_WARMSTART,
        true_beta=true_beta, true_alpha=true_alpha, seed=SEED, num_sims=num_sims,
        num_autos=len(auto_op_contexts), n_warmstart=n_warmstart, is_toy=is_toy, fn_app=fn_app,
    )

    results_by_approach = {
        Approach.BAMCP: results_bamcp,
        Approach.NAIVE_BAYESIAN_WARMSTART: results_bayesian_warmstart,
    }

    plot_results(results_by_approach=results_by_approach, config=config, n_warmstart=n_warmstart)

    for approach, results_list in results_by_approach.items():
        all_followed = [sum(r["followed_advice"]) for r in results_list]
        all_advice = [sum(r["is_advice"]) for r in results_list]

        rates = [f / a if a > 0 else float("nan") for f, a in zip(all_followed, all_advice)]

        print(
            f"{approach.value}: compliance rate = {np.nanmean(rates):.3f} "
            f"(std={np.nanstd(rates):.3f}) "
            f"| total advice steps = {np.mean(all_advice):.1f} "
            f"| total followed = {np.mean(all_followed):.1f}"
        )


if __name__ == "__main__":
    main()