import numpy as np
from itertools import product
from collections import defaultdict
from domains.layered_mdp.layered_mdp import LayeredMDP
from boltzmann_sas.boltzmann_bamdp import BoltzmannBAMDP
from boltzmann_sas.boltzmann_mdp import BoltzmannMDP
from operators.context import OperatorContext
from operators.utils import OperatorParams, OperatorType
from simulations.approach import Approach
from simulations.scoring_functions import generate_scoring_function
from simulations.utils import load_all_sims, get_grid_tag, make_grid
from simulations.plotting import plot_reward_and_belief_heatmaps, plot_final_reward_time_and_belief_heatmaps
from simulations.runners import run_standard_bamcp, run_standard_vi, run_early_stopping_bamcp
from simulations.domain_transitions import build_auto_domain_transitions
from belief.belief import FreqBelief, JointGridBelief
from simulations.utils import get_results_path, get_operators, get_vi_policy_path, load_all_sims
from boltzmann_sas.globals import DEFER, MDP, BAMDP
from bamcp.bamcp import BAMCPSolver, EarlyStoppingBAMCPSolver, MIN_BIAS


def instrument_uct_weight(solver: BAMCPSolver):
    """
    Wrap solver.bamcp.uct_weight to record (node_value, tree_level, bound)
    on every call, without changing its behavior. Returns the log list --
    append to it across multiple solver runs to pool statistics.
    """
    log = []
    original_weight_fn = solver.bamcp.uct_weight

    def logging_weight_fn(node):
        value = node.get_value()
        bound = original_weight_fn(node)
        log.append({
            "abs_value": abs(value),
            "bound": bound,
            "floor_bound": bound == MIN_BIAS and abs(value) < MIN_BIAS,
            "tree_level": node.tree_level,
        })
        return bound

    solver.bamcp.uct_weight = logging_weight_fn
    return log


def summarize_floor_binding(log, level_bucket_size=10):
    """
    Print overall floor-bind rate and bind rate by tree-level bucket.
    A high overall rate, concentrated at deep tree levels, supports the
    "reward-to-go shrinks near the episode end" explanation; a high rate
    spread evenly across levels would suggest something else is going on
    (e.g. the reward scale itself, not just horizon, is too small).
    """
    n = len(log)
    if n == 0:
        print("No uct_weight calls logged.")
        return

    n_floor = sum(1 for entry in log if entry["floor_bound"])
    print(f"Overall floor-bind rate: {n_floor}/{n} = {n_floor/n:.1%}")

    by_bucket = defaultdict(lambda: [0, 0])  # bucket -> [n_floor, n_total]
    for entry in log:
        bucket = (entry["tree_level"] // level_bucket_size) * level_bucket_size
        by_bucket[bucket][1] += 1
        if entry["floor_bound"]:
            by_bucket[bucket][0] += 1

    print(f"\n{'Tree level':<15}{'Bind rate':<12}{'n calls'}")
    for bucket in sorted(by_bucket):
        n_floor_b, n_total_b = by_bucket[bucket]
        rate = n_floor_b / n_total_b if n_total_b else float("nan")
        print(f"{bucket:<15}{rate:<12.1%}{n_total_b}")

    values = np.array([e["abs_value"] for e in log])
    print(f"\nabs(node value) distribution: "
          f"min={values.min():.3f}, median={np.median(values):.3f}, "
          f"mean={values.mean():.3f}, max={values.max():.3f}  "
          f"(MIN_BIAS={MIN_BIAS})")



def build_sas(domain, operator_contexts, model_type):
    operators = get_operators(operator_contexts)
    if model_type == MDP:
        return BoltzmannMDP(domain, operators)
    elif model_type == BAMDP:
        return BoltzmannBAMDP(domain, operators)



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




def main():
    #MAX_DEPTH = 50

    depth = 150
    num_actions = 3
    num_sims = 1
    # p_success is the baseline success rate for the autonomous operator
    p_success = 0.75

    MAX_DEPTH = 150
    SEED = 5
    num_trials = 100
    window = 20 
    tol = 0.05
    
    lb_beta, ub_beta = 0, 10
    lb_alpha, ub_alpha = 0, 5
    grid_res = 0.1
    beta_grid = make_grid(lb_beta, ub_beta, grid_res)
    alpha_grid = make_grid(lb_alpha, ub_alpha, grid_res)
    grid_tag = get_grid_tag(lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res)

    n_jobs = 1
    save_results = False
    is_toy = True
    debug = False
    utility_scale=1

    param_values = [(0.5, 1.2)]
    beta_val = 0.5 
    alpha_val = 1.2

    domain, Phi_nom, cost_nominals, auto_op_contexts = build_domain_and_contexts(
        depth, num_actions, p_success, SEED,
    )


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
        params=OperatorParams(beta=beta_val, alpha=alpha_val),
        init_belief=init_belief,
        nom_scoring=Phi_nom,
    )

    operator_contexts = [bhuman1] + list(auto_op_contexts)

    bamdp = build_sas(domain, operator_contexts, model_type=BAMDP)
    log = []
    for sim in range(5):
        print("sim: ", sim)
        solver = BAMCPSolver(bamdp, max_depth=config["max_depth"], num_trials=config["num_trials"])
        sim_log = instrument_uct_weight(solver)
        solver.run()  # however you currently invoke a full trial run
        log.extend(sim_log)

    
    summarize_floor_binding(log)


if __name__ == "__main__":
    main()