from domains.uav.uav_domain import UAVDomain
from domains.uav.utils import get_layouts
from boltzmann_sas.globals import MDP, BAMDP
from boltzmann_sas.boltzmann_mdp import BoltzmannMDP
from boltzmann_sas.boltzmann_bamdp import BoltzmannBAMDP
from operators.context import OperatorContext
from operators.utils import OperatorType, OperatorParams
from simulations.approach import Approach
from simulations.utils import get_results_path, get_operators, get_grid_tag, make_grid
from simulations.scoring_functions import generate_uav_scoring_function
from simulations.domain_transitions import build_auto_domain_transitions
from simulations.runners import run_standard_vi, run_early_stopping_bamcp
from algorithms.value_iteration import value_iteration
from simulations.plotting import plot_reward_and_belief_heatmaps



def build_sas(domain, operator_contexts, model_type):
    operators = get_operators(operator_contexts)
    if model_type == MDP: 
        return BoltzmannMDP(domain, operators)
    elif model_type == BAMDP:
        return BoltzmannBAMDP(domain, operators)


def test_early_stopping_bamcp(config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, SEED, num_sims,
        auto_op_contexts, beta_grid, alpha_grid, is_toy, fn_app, grid_tag,
        save_results, n_jobs, debug):
    
    # # run early stopping bamcp
    results_es_bamcp = run_early_stopping_bamcp(
        config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, SEED, num_sims,
        auto_op_contexts, beta_grid=beta_grid, alpha_grid=alpha_grid, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag,
        save_results=save_results, n_jobs=n_jobs, debug=debug,
    )

    results_by_approach = {
        Approach.BAMCP_ES.value: results_es_bamcp
    }

    plot_reward_and_belief_heatmaps(results_by_approach, config)

    


def test_vi(domain, operator_contexts):
    mdp = build_sas(domain, operator_contexts, model_type=MDP)
    Q, V, policy = value_iteration(model=mdp)


def test_mdp(domain, operator_contexts):
    mdp = build_sas(domain, operator_contexts, model_type=MDP)

    for state in mdp.states:
        if state not in mdp.terminal_states:
            print("\n\nState: ", state)
            for action in mdp.actions:
                print("Action: ", action)
                if (state, action) in mdp.T:
                    possible_transitions = mdp.T[(state, action)]
                    for next_state in possible_transitions:
                        print(f"Next state: {next_state}, r: {mdp.R[state, action, next_state]}")
                        
            input("Next...")

    

def test_dynamics_structure(size, p_obs):
    uav = UAVDomain(size, p_obs)

    for state in uav.states:
        if state not in uav.terminal_states:
            print("\n\nState: ", state)
            for action in uav.actions:
                print("Action: ", action)
                if (state, action) in uav.T_structure:
                    possible_transitions = uav.T_structure[(state, action)]
                    for next_state in possible_transitions:
                        print(f"Next state: {next_state}, r: {uav.get_reward(state, action, next_state)}")
                        
            input("Next...")


def test_load_layouts(size, p_obs, num_layouts):
    for _ in range(2):
        layouts = get_layouts(size, p_obs, num_layouts)
    for layout in layouts: 
        print(layout, "\n")
        


def test_generate_layouts(size, p_obs, num_layouts):
    layouts = get_layouts(size, p_obs, num_layouts)
    for layout in layouts:
        print("\n", layout)


def test_build_domain(size, p_obs):
    domain = UAVDomain(size=size, p_obs=p_obs)


if __name__ == "__main__":
    size = 18 
    p_obs = 0.2

    true_beta = 10
    true_alpha = 1
    num_sims = 1
    SEED = 1
    MAX_DEPTH = 50 

    # belief representation
    lb_beta, ub_beta = 0, 10
    lb_alpha, ub_alpha = 0, 5
    grid_res = 0.1
    beta_grid = make_grid(lb_beta, ub_beta, grid_res)
    alpha_grid = make_grid(lb_alpha, ub_alpha, grid_res)
    grid_tag = get_grid_tag(lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res)


    # simulation config
    config = {
        "seed": SEED,
        "max_depth": MAX_DEPTH,
        "size": size,
        "p_obs": p_obs,
        "true_beta": true_beta,
        "true_alpha": true_alpha,
        "num_sims": num_sims,
    }

    # n_jobs=-1 uses all available cores. Drop to e.g. os.cpu_count() - 1
    # if you want to keep a core free while these run in the background.
    n_jobs = -1

    fn_app = "test_bamcp_es_solver"
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

    #test_build_domain(size, p_obs)
    #test_generate_layouts(size, p_obs, 10)
    #test_load_layouts(size, p_obs, 2)
    #test_dynamics_structure(size, p_obs)

    domain = UAVDomain(size, p_obs)
    cost_nominals = {a: 0 for a in domain.actions}

    p_success = 0.75 
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

    # generate nominal human scoring function 
    Phi_nom = generate_uav_scoring_function(domain)

    bhuman1 = OperatorContext(
        category=OperatorType.BHUMAN,
        n=1,
        actions=domain.actions,
        enabled_actions=domain.enabled_actions,
        domain_transitions=domain.T_det,
        cost_nominals=cost_nominals,
        params=OperatorParams(beta=true_beta, alpha=true_alpha),
        nom_scoring=Phi_nom,
    )    

    operator_contexts = [bhuman1]

    #test_mdp(domain, operator_contexts)
    #test_vi(domain, operator_contexts)

    domain = UAVDomain(size, p_obs)

    test_early_stopping_bamcp(config, domain, Phi_nom, true_beta, true_alpha, cost_nominals, SEED, num_sims,
        auto_op_contexts, beta_grid, alpha_grid, is_toy, fn_app, grid_tag,
        save_results, n_jobs, debug)


    
