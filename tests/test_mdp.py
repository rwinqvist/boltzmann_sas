from domains.layered_mdp.layered_mdp import LayeredMDP
from boltzmann_sas.globals import MDP, BAMDP
from boltzmann_sas.boltzmann_mdp import BoltzmannMDP
from boltzmann_sas.boltzmann_bamdp import BoltzmannBAMDP
from operators.context import OperatorContext
from operators.utils import OperatorType, OperatorParams
from simulations.utils import get_results_path, get_operators
from simulations.scoring_functions import generate_scoring_function
from simulations.domain_transitions import build_auto_domain_transitions
from simulations.runners import run_standard_vi



def main():
    SEED = 5 
    MAX_DEPTH = 50 

    depth = 100
    num_actions = 3
    true_beta = 0.5
    true_alpha = 1
    num_sims = 1

    # n_jobs=-1 uses all available cores. Drop to e.g. os.cpu_count() - 1
    # if you want to keep a core free while these run in the background.
    n_jobs = -1

    fn_app = ""
    save_results = True 
    is_toy = True 
    manual_debug = False 

    if not save_results:
        fn_app ="test"
        print("WARNING! NOT SAVING RESULTS!")
    if is_toy:
        print("WARNING! You're saying under TOY")
    if manual_debug:
        print("WARNING! Debug mode on")
        fn_app = "debug"
        save_results = False
        num_sims = 1
        n_jobs = 1

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

    p_success = 0.75   # tune this to create a genuine crossover with the human, per earlier discussion
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

    # generate nominal human scoring function 
    Phi_nom = generate_scoring_function(domain, seed=SEED)

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

    operator_contexts = [bhuman1] + list(auto_op_contexts)
    results = {}
    
    mdp = build_sas(domain, operator_contexts, model_type=MDP)
    

def build_sas(domain, operator_contexts, model_type):
    operators = get_operators(operator_contexts)
    if model_type == MDP: 
        return BoltzmannMDP(domain, operators)
    elif model_type == BAMDP:
        return BoltzmannBAMDP(domain, operators)


if __name__ == "__main__":
    main()