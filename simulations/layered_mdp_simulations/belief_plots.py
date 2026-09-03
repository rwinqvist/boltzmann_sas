# belief_plots.py
import joblib
from simulations.approach import Approach
from simulations.utils import get_results_path, get_grid_tag
from simulations.paper_plots import plot_belief_regime_comparison, plot_belief_regime_comparison_both, ALPHA_INDEX, BETA_INDEX
from simulations.layered_mdp_simulations.sweep_config import (
    PARAM_PAIRS, num_actions, SEED, num_trials, window, tol, is_toy, is_cluster,
    MAX_DEPTH, base_fn_app, lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res,
)

_grid_tag = get_grid_tag(lb_beta, ub_beta, lb_alpha, ub_alpha, grid_res)


def load_raw_sims(depth, beta, alpha, approach, layout_idx, sims=range(1, 2)):
    """
    Load FULL (unstripped) result dicts -- including "belief" -- for belief
    heatmaps. Unlike summary_results.pkl (scalars only), these are heavy
    (~10MB each), so only load a handful: default is just sim 1 of one
    layout. Pass sims=range(1, 11) to average over all 10 sims of a layout
    if you want a smoother heatmap at the cost of a slower load.
    """
    domain_name = "layered_mdp"
    domain_tag = f"d{depth}_a{num_actions}"
    config = {"max_depth": MAX_DEPTH, "depth": depth, "true_beta": beta,
              "true_alpha": alpha, "num_sims": num_trials, "num_trials": num_trials}

    fn_app = base_fn_app + f"_L{layout_idx + 1}"
    if approach == Approach.BAMCP_ES:
        fn_app += f"_window{window}_eps{tol}"

    results = []
    for sim in sims:
        result_fn = get_results_path(
            domain_name=domain_name, domain_tag=domain_tag,
            num_humans=1, num_autos=1, approach=approach,
            true_beta=beta, true_alpha=alpha, seed=SEED, sim=sim,
            config=config, is_toy=is_toy, is_cluster=is_cluster,
            fn_app=fn_app, grid_tag=_grid_tag,
        )
        try:
            results.append(joblib.load(result_fn))
        except FileNotFoundError:
            print(f"missing: {result_fn}")
    return results


def plot_beliefs(paper_plots_dir, presentation_plots_dir, tag, depth=150, layout_idx=0, sims=range(1, 2),
                  approach=Approach.BAMCP, param_index=ALPHA_INDEX, true_value=1.2):
    regime_results = {}
    for true_beta, true_alpha in PARAM_PAIRS:
        results = load_raw_sims(depth, true_beta, true_alpha, approach, layout_idx, sims=sims)
        if results:
            regime_results[fr"$\beta$={true_beta}"] = results
        else:
            print(f"no results loaded for beta={true_beta}, alpha={true_alpha} -- skipping regime")

    if not regime_results:
        print("No belief data loaded -- nothing to plot")
        return

    param_label = "alpha" if param_index == ALPHA_INDEX else "beta"
    param_symbol = r"\alpha_0" if param_index == ALPHA_INDEX else r"\beta_0"
    true_value_label = fr"${param_symbol}={true_value}$"

    pdf_save_path = f"{paper_plots_dir}/fig_belief_{param_label}_depth{depth}_L{layout_idx+1}{tag}.pdf"
    png_save_path = f"{presentation_plots_dir}/fig_belief_{param_label}_depth{depth}_L{layout_idx+1}{tag}.png"

    plot_belief_regime_comparison(
        regime_results, param_index=param_index, true_value=true_value,
        true_value_label=true_value_label, pdf_save_path=pdf_save_path, png_save_path=png_save_path,
    )


def plot_beliefs_both(paper_plots_dir, presentation_plots_dir, tag, depth=150, layout_idx=0,
                       sims=range(1, 2), approach=Approach.BAMCP,
                       display_res_beta=None, display_res_alpha=None):
    """
    Two-row (beta, alpha) belief-concentration comparison across the beta
    regimes in PARAM_PAIRS, at a single depth/layout. Saves both PDF (for
    the paper) and PNG (for slides) versions.
    """
    regimes = []
    for true_beta, true_alpha in PARAM_PAIRS:
        results = load_raw_sims(depth, true_beta, true_alpha, approach, layout_idx, sims=sims)
        if not results:
            print(f"no results loaded for beta={true_beta}, alpha={true_alpha} -- skipping regime")
            continue
        regimes.append({
            "label": fr"$\beta_0={true_beta}$",
            "results": results,
            "true_beta": true_beta,
            "true_alpha": true_alpha,
        })

    if not regimes:
        print("No belief data loaded -- nothing to plot")
        return

    fname = f"fig_belief_both_depth{depth}_L{layout_idx+1}{tag}"
    pdf_save_path = f"{paper_plots_dir}/{fname}.pdf"
    png_save_path = f"{presentation_plots_dir}/{fname}.png"

    plot_belief_regime_comparison_both(
        regimes, display_res_beta=display_res_beta, display_res_alpha=display_res_alpha,
        pdf_save_path=pdf_save_path, png_save_path=png_save_path,
    )