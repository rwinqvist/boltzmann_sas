# sweep_config.py
"""
Single source of truth for UAV sweep parameters, shared between
run_uav_simulations.py (cluster) and any local aggregation/plotting
script. Change values here once; both stay in sync automatically --
the cluster script imports from here rather than redefining these,
so there's no way for what actually ran to drift from what a later
plotting script assumes ran.
"""

SIZE = 40
P_OBS = 0.2
RESTRICT_TO_FORWARD = False   # full 4-action set (human + auto both active)
NUM_LAYOUTS = 1

UTILITY_SCALE = 5
ALPHA = max(1, UTILITY_SCALE - 1)
PARAM_PAIRS = [(0.05, ALPHA), (8, ALPHA)]


num_sims = 1
p_success = 0.75
SEED = 10
num_trials = 1000
window = 20
tol = 0.1
is_toy = False
is_cluster = True

MANHATTAN_DIST = (SIZE - 1) + (SIZE - 1)
MAX_DEPTH = 2 * MANHATTAN_DIST  # keep in sync manually if SIZE changes

base_fn_app = f"_uscale{UTILITY_SCALE}_psuccess{p_success}" if UTILITY_SCALE > 1 else f"_{p_success}"

lb_beta, ub_beta = 0, 10
lb_alpha, ub_alpha = 0, 5
grid_res = 0.1

params_tag = "-".join(f"b{b}a{a}" for b, a in PARAM_PAIRS)
CACHE_PATH = (
    f"summary_results_uav_size{SIZE}_pobs{P_OBS}_forward{RESTRICT_TO_FORWARD}"
    f"_{params_tag}_n{num_sims}_L{NUM_LAYOUTS}.pkl"
)