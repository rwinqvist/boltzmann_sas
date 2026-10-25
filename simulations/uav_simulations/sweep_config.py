# sweep_config.py
"""
Single source of truth for UAV sweep parameters, shared between
run_uav_simulations.py (cluster) and any local aggregation/plotting
script. Change values here once; both stay in sync automatically --
the cluster script imports from here rather than redefining these,
so there's no way for what actually ran to drift from what a later
plotting script assumes ran.
"""
from global_config import ROOT_DIR

SIZE = 30
P_OBS = 0.2
RESTRICT_TO_FORWARD = False   # full 4-action set (human + auto both active)
NUM_LAYOUTS = 10

UTILITY_SCALE = 3
ALPHA = max(1, UTILITY_SCALE - 1)
PARAM_PAIRS = [(8,ALPHA)]

num_sims = 10
p_success = 0.75
SEED = 5
num_trials = 1000
NUM_TRIALS_ABLATION_VALUES = [1000, 4000]
window = 20
tol = 0.1
is_toy = False
is_cluster = True

MANHATTAN_DIST = (SIZE - 1) + (SIZE - 1)
MAX_DEPTH = 2 * MANHATTAN_DIST  # keep in sync manually if SIZE changes

DEFAULT_NUM_TRIALS = 1000
DEFAULT_NUM_SIMS = 10

def build_base_fn_app(num_trials=None, num_sims=None):
    """
    Same fn_app convention as before, but only appends num_trials/num_sims
    when they differ from the defaults -- so any run using the defaults
    (including everything already saved on disk) gets the EXACT same path
    as before, while a run using a non-default num_trials or num_sims
    (e.g. the num_trials ablation) automatically gets a distinct path
    with no risk of colliding with the baseline or with each other.
    """
    nt = num_trials if num_trials is not None else DEFAULT_NUM_TRIALS
    ns = num_sims if num_sims is not None else DEFAULT_NUM_SIMS
 
    trials_suffix = f"_numtrials{nt}" if nt != DEFAULT_NUM_TRIALS else ""
    sims_suffix = f"_numsims{ns}" if ns != DEFAULT_NUM_SIMS else ""
 
    if UTILITY_SCALE > 1:
        return f"_uscale{UTILITY_SCALE}_psuccess{p_success}{trials_suffix}"
    else:
        return f"_{p_success}{trials_suffix}"

#base_fn_app = f"_uscale{UTILITY_SCALE}_psuccess{p_success}_numtrials{num_trials}_numsims_{num_sims}" if UTILITY_SCALE > 1 else f"_{p_success}"
base_fn_app = build_base_fn_app(num_trials=DEFAULT_NUM_TRIALS, num_sims=DEFAULT_NUM_SIMS)

lb_beta, ub_beta = 0, 10
lb_alpha, ub_alpha = 0, 5
grid_res = 0.1

params_tag = "-".join(f"b{b}a{a}" for b, a in PARAM_PAIRS)

CACHE_PATH = f"{ROOT_DIR}/simulations/uav_simulations/results"

def get_cached_fn(size):
    fn = (f"summary_results_uav_size{size}_pobs{P_OBS}_forward{RESTRICT_TO_FORWARD}"
        f"_{params_tag}_n{num_sims}_L{NUM_LAYOUTS}.pkl")
    return CACHE_PATH + f"/{fn}"

#fn = (f"summary_results_uav_size{SIZE}_pobs{P_OBS}_forward{RESTRICT_TO_FORWARD}"
    #f"_{params_tag}_n{num_sims}_L{NUM_LAYOUTS}.pkl")
    
cached_fn = get_cached_fn(SIZE)


