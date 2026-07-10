import os
import joblib
from global_config import ROOT_DIR
from simulations.approach import Approach
from operators.context import OperatorContext
from operators.utils import OperatorType
from operators.auto import AutonomousOperator
from operators.human import HumanOperator, BoltzmannHumanOperator

def check_fn(fn):
    """Ensure the parent directory of `fn` exists."""
    os.makedirs(os.path.dirname(fn), exist_ok=True)


def get_results_dir(domain_name, is_toy=False):
    subfolder = "toy_results" if is_toy else "results"
    return f"{ROOT_DIR}/simulations/{domain_name}_simulations/{subfolder}"

def get_policy_dir(domain_name, is_toy=False):
    subfolder = "toy_policies" if is_toy else ""
    return f"{ROOT_DIR}/simulations/policies/{domain_name}policies/{subfolder}"

def get_vi_policy_path(domain_name, domain_tag, num_humans, num_autos, true_beta, true_alpha, seed, is_toy=False, fn_app=""):
    policy_dir = get_policy_dir(domain_name, is_toy)
    
    fn = f"{policy_dir}/{domain_tag}_h{num_humans}_a{num_autos}/true_b{true_beta}_a{true_alpha}/s{seed}_{fn_app}.joblib"
    check_fn(fn)
    return fn

def get_results_path(domain_name, domain_tag, num_humans, num_autos, approach, true_beta, true_alpha, seed, sim, config, n_warmstart=0, is_toy=False, fn_app="", grid_tag=""):
    results_dir = get_results_dir(domain_name, is_toy)

    if approach in (Approach.BAMCP, Approach.BAMCP_ALPHA_COLLAPSE):
        approach_tag = approach.value
    elif approach in (Approach.NAIVE_FREQ_WARMSTART, Approach.NAIVE_BAYESIAN_WARMSTART):
        approach_tag = f"{approach.value}_n{n_warmstart}"
    elif approach == Approach.VI:
        approach_tag = approach.value
    else:
        raise ValueError(f"Unhandled approach: {approach}")
    
    grid_suffix = f"_{grid_tag}" if grid_tag != "" else ""

    fn = f"{results_dir}/{domain_tag}_h{num_humans}_a{num_autos}/{approach_tag}{grid_tag}/true_b{true_beta}_a{true_alpha}/s{seed}_sim{sim}_depth{config["max_depth"]}{fn_app}.joblib"
    check_fn(fn)
    return fn


def load_all_sims(domain_name, domain_tag, approach: Approach, true_beta, true_alpha, seed,
                   num_sims, num_autos, config, n_warmstart=0, is_toy=False, fn_app="", grid_tag=""):
    """ Load all sim results for a given approach into a list. """
    all_results = []
    for sim in range(1, num_sims + 1):
        fn = get_results_path(
            domain_name=domain_name, domain_tag=domain_tag, num_humans=1, num_autos=num_autos,
            approach=approach, config=config, n_warmstart=n_warmstart, true_beta=true_beta, true_alpha=true_alpha,
            seed=seed, sim=sim, is_toy=is_toy, fn_app=fn_app, grid_tag=grid_tag,
        )
        results = joblib.load(fn)
        all_results.append(results)
 
    return all_results


def get_operators(operator_contexts: list[OperatorContext]):
    operators = [None for _ in range(len(operator_contexts))]

    for i, operator_context in enumerate(operator_contexts):
        # IMDP is currently not supported so will only consider single-valued transitions 
        if operator_context.category == OperatorType.BHUMAN:
            operator = BoltzmannHumanOperator.from_context(operator_context)
        elif operator_context.category == OperatorType.AUTO:
            operator = AutonomousOperator.from_context(operator_context)
        elif operator_context.category == OperatorType.HUMAN:
            operator = HumanOperator.from_context(operator_context)

        operators[i] = operator

    return operators