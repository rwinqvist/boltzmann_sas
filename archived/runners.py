# ============================================================
# Naive (frequentist) warm-start
# ============================================================

def _run_naive_freq_warmstart_one_sim(sim, domain, operator_contexts, auto_op_contexts, config,
                                  true_beta, true_alpha, seed, n_warmstart, is_toy, fn_app, save_results):
    """ Runs a single naive-warmstart simulation. Top-level (picklable) for joblib/loky. """
    sim_seed = seed * 10_000 + sim
    random.seed(sim_seed)
    np.random.seed(sim_seed)

    print(f"Sim: {sim + 1}")
    fn = get_results_path(
        domain_name=domain.domain_name, domain_tag=domain.id_tag, num_humans=1,
        num_autos=len(auto_op_contexts), approach=Approach.NAIVE_FREQ_WARMSTART, config=config, n_warmstart=n_warmstart,
        true_beta=true_beta, true_alpha=true_alpha, seed=seed, sim=sim + 1, is_toy=is_toy, fn_app=fn_app,
    )
    if os.path.exists(fn):
        print("Results already exist!")
        return

    bamdp = build_sas(domain, operator_contexts, model_type=BAMDP)
    state = bamdp.s0
    history = History(items=(state,))
    total_reward = 0
    total_steps = 0
    is_defer_vec, is_advice_vec, is_auto_vec = [], [], []
    belief_vec, belief_stats, rewards, cum_rewards = [], [], [], []
    is_advice = is_auto = 0
    is_defer = 1

    defer_action = (0, DEFER)
    start = time.perf_counter()
    for _ in range(n_warmstart):
        next_state, reward, executed_action = bamdp.step(state, defer_action)
        obs = Observation(
            domain_state=state[0], op_state=state[1][0], operator=bamdp.boltzmann_operators[0],
            executed_domain_action=executed_action[1], issued_domain_action=DEFER,
        )
        is_advice_vec.append(is_advice)
        is_defer_vec.append(is_defer)
        is_auto_vec.append(is_auto)
        rewards.append(reward)

        bamdp.update_belief(obs)
        belief_vec.append(bamdp.get_belief())
        belief_stats.append(bamdp.get_belief_stats())

        history = history.add_entry(defer_action, executed_action, next_state)
        state = next_state
        total_reward += reward
        total_steps += 1
        cum_rewards.append(total_reward)

    bamcp = BAMCPSolver(bamdp, max_depth=config["max_depth"])
    results = bamcp.run(
        s0=state, total_reward=total_reward, total_steps=total_steps, is_defer_vec=is_defer_vec,
        is_advice_vec=is_advice_vec, is_auto_vec=is_auto_vec, belief_vec=belief_vec, belief_stats=belief_stats,
        rewards=rewards, cum_rewards=cum_rewards,
    )
    total_time = time.perf_counter() - start

    results["total_time"] = total_time
    results["history"] = History(items=history.items + results["history"].items[1:])
    results["config"] = config

    if save_results:
        joblib.dump(results, fn)
    print(f"Sim {sim + 1} done in {total_time:.1f}s (total_reward={results['total_reward']:.2f})")



def run_naive_freq_warmstart(config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, seed, num_sims,
                         n_warmstart, auto_op_contexts, alpha_grid, is_toy=False, fn_app="", save_results=True, n_jobs=-1):
    print("\nApproach: Naive warmstart")
    bhuman1 = OperatorContext(
        category=OperatorType.BHUMAN,
        n=1,
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=domain.T_det,
        cost_nominals=cost_nominals,
        params=OperatorParams(beta=true_beta, alpha=true_alpha),
        init_belief=FreqBelief.uniform(alpha_values=alpha_grid),
        nom_scoring=Phi_nom,
    )

    config["n_warmstart"] = n_warmstart

    operator_contexts = [bhuman1] + list(auto_op_contexts)

    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_naive_freq_warmstart_one_sim)(
            sim, domain, operator_contexts, auto_op_contexts, config,
            true_beta, true_alpha, seed, n_warmstart, is_toy, fn_app, save_results,
        )
        for sim in range(num_sims)
    )




# ============================================================
# Bayesian warm-start
# ============================================================

def _run_bayesian_warmstart_one_sim(sim, domain, operator_contexts, auto_op_contexts, config,
                                     true_beta, true_alpha, seed, n_warmstart, is_toy, fn_app, grid_tag, save_results):
    """ Runs a single Bayesian-warmstart simulation. Top-level (picklable) for joblib/loky. """
    sim_seed = seed * 10_000 + sim
    random.seed(sim_seed)
    np.random.seed(sim_seed)

    print(f"Sim: {sim + 1}")
    fn = get_results_path(
        domain_name=domain.domain_name, domain_tag=domain.id_tag, num_humans=1,
        num_autos=len(auto_op_contexts), approach=Approach.NAIVE_BAYESIAN_WARMSTART, config=config, n_warmstart=n_warmstart,
        true_beta=true_beta, true_alpha=true_alpha, seed=seed, sim=sim + 1, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag,
    )
    if os.path.exists(fn):
        print("Results already exist!")
        return

    bamdp = build_sas(domain, operator_contexts, model_type=BAMDP)
    state = bamdp.s0
    history = History(items=(state,))
    total_reward = 0
    total_steps = 0
    is_defer_vec, is_advice_vec, is_auto_vec = [], [], []
    belief_vec, belief_stats, rewards, cum_rewards = [], [], [], []
    is_advice = is_auto = 0
    is_defer = 1

    belief_vec.append(bamdp.get_belief())
    belief_stats.append(bamdp.get_belief_stats())

    defer_action = (0, DEFER)
    start = time.perf_counter()
    for _ in range(n_warmstart):
        next_state, reward, executed_action = bamdp.step(state, defer_action)
        obs = Observation(
            domain_state=state[0], op_state=state[1][0], operator=bamdp.boltzmann_operators[0],
            executed_domain_action=executed_action[1], issued_domain_action=DEFER,
        )
        is_advice_vec.append(is_advice)
        is_defer_vec.append(is_defer)
        is_auto_vec.append(is_auto)
        rewards.append(reward)

        bamdp.update_belief(obs)
        belief_vec.append(bamdp.get_belief())
        belief_stats.append(bamdp.get_belief_stats())

        history = history.add_entry(defer_action, executed_action, next_state)
        state = next_state
        total_reward += reward
        total_steps += 1
        cum_rewards.append(total_reward)

    bamcp = BAMCPSolver(bamdp, max_depth=config["max_depth"])
    results = bamcp.run(
        s0=state, total_reward=total_reward, total_steps=total_steps, is_defer_vec=is_defer_vec,
        is_advice_vec=is_advice_vec, is_auto_vec=is_auto_vec, belief_vec=belief_vec, belief_stats=belief_stats,
        rewards=rewards, cum_rewards=cum_rewards,
    )
    total_time = time.perf_counter() - start

    results["total_time"] = total_time
    results["history"] = History(items=history.items + results["history"].items[1:])
    results["config"] = config

    if save_results:
        joblib.dump(results, fn)
    print(f"Sim {sim + 1} done in {total_time:.1f}s (total_reward={results['total_reward']:.2f})")


def run_bayesian_naive_warmstart(config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, seed, num_sims,
                                  n_warmstart, auto_op_contexts, beta_grid, alpha_grid, is_toy=False, fn_app="", grid_tag="",
                                  save_results=True, n_jobs=-1, debug=False):
    print("\nApproach: Bayesian naive warmstart")

    if debug: 
        n = len(beta_grid) * len(alpha_grid)
        p_params = {(b, a): 0 for b, a in product(beta_grid, alpha_grid)}
        p_params[(beta_grid[0], alpha_grid[0])] = 1
        init_belief = JointGridBelief(p_params)
    else: 
        init_belief = JointGridBelief.uniform(beta_values=beta_grid, alpha_values=alpha_grid)

    bhuman1 = OperatorContext(
        category=OperatorType.BHUMAN,
        n=1,
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=domain.T_det,
        cost_nominals=cost_nominals,
        params=OperatorParams(beta=true_beta, alpha=true_alpha),
        init_belief=init_belief,
        nom_scoring=Phi_nom,
    )

    config["n_warmstart"] = n_warmstart

    operator_contexts = [bhuman1] + list(auto_op_contexts)

    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_bayesian_warmstart_one_sim)(
            sim, domain, operator_contexts, auto_op_contexts, config,
            true_beta, true_alpha, seed, n_warmstart, is_toy, fn_app, grid_tag, save_results,
        )
        for sim in range(num_sims)
    )



# ============================================================
# BAMCP with alpha-collapse monitor
#    - Identical to standard BAMCP, except: after every belief update, checks
#      whether the AlphaCollapseMonitor says alpha is no longer identifiable
#      from here (given a pessimistic beta estimate), and if so, freezes
#      alpha's belief and lets planning continue with a reduced belief space.
#    - Written as an explicit step loop (not delegating to solver.run())
#      because we need a hook after every belief update to check the monitor.
# ============================================================

def _run_collapse_aware_bamcp_one_sim(sim, domain, operator_contexts, auto_op_contexts, config, true_beta, true_alpha, seed, is_toy, fn_app, grid_tag, save_results, collapse_kwargs):
    """ Runs a single alpha-collapse-aware BAMCP simulation. Top-level (pickable) for joblib/loky """
    sim_seed = seed * 10_000 + sim 
    random.seed(sim_seed)
    np.random.seed(sim_seed)

    print(f"Sim: {sim + 1}")
    fn = get_results_path(
        domain_name=domain.domain_name, domain_tag=domain.id_tag, num_humans=1,
        num_autos=len(auto_op_contexts), approach=Approach.BAMCP_ALPHA_COLLAPSE,
        true_beta=true_beta, true_alpha=true_alpha, seed=seed, sim=sim + 1, is_toy=is_toy,
        fn_app=fn_app, grid_tag=grid_tag, config=config,
    )
    if os.path.exists(fn):
        print("Results already exist!")
        return

    bamdp = build_sas(domain, operator_contexts, model_type=BAMDP)
    solver = BAMCPSolver(bamdp, max_depth=config["max_depth"])
    monitor = AlphaCollapseMonitor(**(collapse_kwargs or {}))

    state = bamdp.s0
    history = History(items=(state,))
    total_reward = 0
    total_steps = 0
    is_defer_vec, is_advice_vec, is_auto_vec = [], [], []
    belief_vec, belief_stats, rewards, cum_rewards = [], [], [], []
    followed_advice = []
    n_advice_so_far = 0
    collapse_step_record = None   # real step index at which collapse happened, for later analysis
    max_steps = config.get("max_steps", np.inf)

    start = time.perf_counter()
    while not bamdp.is_goal(state) and not bamdp.is_terminal(state) and total_steps < max_steps:
        issued_action = solver.get_next_action(history)
        is_advice, is_defer, is_auto = unpack_action(bamdp, issued_action)
        is_advice_vec.append(is_advice)
        is_defer_vec.append(is_defer)
        is_auto_vec.append(is_auto)
 
        next_state, reward, executed_action = bamdp.step(state, issued_action)
 
        if issued_action[1] != DEFER and executed_action[1] == issued_action[1]:
            followed_advice.append(1)
        else:
            followed_advice.append(0)
 
        rewards.append(reward)
 
        opidx, executed_domain_action = executed_action
        operator = bamdp.operators[opidx]
        _, issued_domain_action = issued_action
        domain_state, joint_op_state = state
        op_state = joint_op_state[opidx]
 
        obs = Observation(
            domain_state=domain_state, op_state=op_state, operator=operator,
            executed_domain_action=executed_domain_action, issued_domain_action=issued_domain_action,
        )
        bamdp.update_belief(obs)
 
        if is_advice:
            n_advice_so_far += 1

        # NOTE: bamdp.belief is keyed by boltzmann_operator_indices[operator],
        # NOT by the action's opidx (which indexes the full operator list,
        # including any autonomous operators) -- these are different indices.
        if not monitor.collapsed and operator in bamdp.boltzmann_operators:
            bidx = bamdp.boltzmann_operator_indices[operator]
            belief = bamdp.belief[bidx]
            n_remaining = max_steps - total_steps - 1 if np.isfinite(max_steps) else domain.depth - total_steps - 1
            triggered = monitor.check(belief, n_advice_so_far, n_remaining)
            if triggered:
                belief.freeze_alpha(monitor.collapse_value)
                collapse_step_record = total_steps + 1
                print(f"Sim {sim+1}: alpha collapsed at step {collapse_step_record} "
                      f"(n_advice={n_advice_so_far}), frozen alpha={monitor.collapse_value}")

        
        belief_vec.append(bamdp.get_belief())
        belief_stats.append(bamdp.get_belief_stats())
 
        history = history.add_entry(issued_action, executed_action, next_state)
        state = next_state
        total_reward += reward
        total_steps += 1
        cum_rewards.append(total_reward)

    total_time = time.perf_counter() - start
     
    results = {
        "cum_rewards": cum_rewards,
        "rewards": rewards,
        "total_reward": total_reward,
        "total_steps": total_steps,
        "is_defer": is_defer_vec,
        "is_advice": is_advice_vec,
        "is_auto": is_auto_vec,
        "belief": belief_vec,
        "belief_stats": belief_stats,
        "history": history,
        "followed_advice": followed_advice,
        "total_time": total_time,
        "config": config,
        "alpha_collapse_step": collapse_step_record,
        "alpha_collapse_value": monitor.collapse_value,
    }

    if save_results:
        joblib.dump(results, fn)

    collapse_msg = f"collapsed at step {collapse_step_record}" if collapse_step_record else "never collapsed"
    print(f"Sim {sim + 1} done in {total_time:.1f}s (total_reward={results['total_reward']:.2f}, alpha {collapse_msg})")
 

def run_collapse_aware_bamcp(config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, seed, num_sims,
                              auto_op_contexts, beta_grid, alpha_grid, is_toy=False, fn_app="", grid_tag="",
                              save_results=True, n_jobs=-1, collapse_kwargs=None):
    """
    :param collapse_kwargs: dict of kwargs passed to AlphaCollapseMonitor, e.g.
        {"min_advice_obs": 15, "check_every": 10, "tolerance": 0.05,
         "confidence": 0.1, "required_consecutive": 2}. Defaults used if None.
    """
    print("\nApproach: BAMCP with alpha-collapse")

    init_belief = JointGridBelief.uniform(beta_values=beta_grid, alpha_values=alpha_grid)
 
    bhuman1 = OperatorContext(
        category=OperatorType.BHUMAN,
        n=1,
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=domain.T_det,
        cost_nominals=cost_nominals,
        params=OperatorParams(beta=true_beta, alpha=true_alpha),
        init_belief=init_belief,
        nom_scoring=Phi_nom,
    )

    operator_contexts = [bhuman1] + list(auto_op_contexts)
 
    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_collapse_aware_bamcp_one_sim)(
            sim, domain, operator_contexts, auto_op_contexts, config,
            true_beta, true_alpha, seed, is_toy, fn_app, grid_tag, save_results, collapse_kwargs,
        )
        for sim in range(num_sims)
    )