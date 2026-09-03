# sweep_config.py
"""
Single source of truth for sweep parameters shared between
build_summary_df.py (cluster) and make_plots.py (local).
Change values here once; both scripts stay in sync automatically.
"""
from global_config import ROOT_DIR

DEPTHS = [100, 150, 200]
NUM_LAYOUTS = 10
PARAM_PAIRS = [(0.5, 1.2), (1.5, 1.2), (5.0, 1.2)]

num_actions = 3
num_sims = 10
p_success = 0.75
SEED = 5
num_trials = 1000
window = 20
tol = 0.05
utility_scale = 1
is_toy = False
is_cluster = True
MAX_DEPTH = 200

base_fn_app = f"_uscale{utility_scale}_psuccess{p_success}" if utility_scale > 1 else f"_{p_success}"

lb_beta, ub_beta = 0, 10
lb_alpha, ub_alpha = 0, 5
grid_res = 0.1

depths_tag = "-".join(str(d) for d in DEPTHS)
params_tag = "-".join(f"b{b}a{a}" for b, a in PARAM_PAIRS)
CACHE_PATH = f"{ROOT_DIR}/simulations/layered_mdp_simulations/results"
fn = f"summary_results_depths{depths_tag}_{params_tag}_n{num_sims}_L{NUM_LAYOUTS}.pkl"
cached_fn = CACHE_PATH + f"/{fn}"


