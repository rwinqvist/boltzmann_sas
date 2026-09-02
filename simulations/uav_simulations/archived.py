def run_params_sweep(layout, idx, size, p_obs, beta_vals, alpha_vals, num_sims, utility_scale, p_success, seed, layout_seed, max_depth, num_trials, debug, is_toy, save_results, window=20, tol=0.05):

    # build domain 
    domain = UAVDomain.from_map(size=size, p_obs=p_obs, layout=layout)
    cost_nominals = {a: 0 for a in domain.actions}
    #Phi_nom = generate_random_uav_scoring_function(domain, scale=utility_scale, seed=layout_seed)
    Phi_nom = generate_progression_biased_uav_scoring_function(domain, scale=utility_scale, seed=layout_seed)

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

    tag = f"layout{idx+1}_psuccess_{p_success}_numsims{num_sims}"
    es_tag = tag + f"_window{window}_eps{tol}"
    fn_app = tag 
    es_fn_app = es_tag

    results_by_params_and_approach = {}

    for true_beta, true_alpha in product(beta_vals, alpha_vals):
        config = {
            "seed": seed,
            "max_depth": max_depth,
            "num_trials": num_trials,
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
                                                    is_toy=is_toy, fn_app=es_fn_app, grid_tag=grid_tag, save_results=save_results,
                                                    n_jobs=N_JOBS, debug=debug, window=window, tol=tol)


        results_by_params_and_approach[(true_beta, true_alpha)] = {
                Approach.BAMCP: results_bamcp,
                Approach.BAMCP_ES: results_bamcp_es
            }

        for r in results_by_params_and_approach[(true_beta, true_alpha)][Approach.BAMCP]:
            print(r["total_steps"])
        
    plot_metric_vs_params(
        results_by_params_and_approach, metric="total_reward",
        ylabel="Total reward", save_path=f"fig_uav_reward_vs_params_{tag}.pdf",
    )
    plot_metric_vs_params(
        results_by_params_and_approach, metric="total_time",
        ylabel="Wall-clock time (s)", save_path=f"fig_uav_time_vs_params_{tag}.pdf",
    )

