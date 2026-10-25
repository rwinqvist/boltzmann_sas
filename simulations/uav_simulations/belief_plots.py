# belief_plots_uav.py
"""
UAV analog of belief_plots.py.

plot_belief_regime_comparison / plot_belief_regime_comparison_both (and the
matrix-extraction helpers underneath them in plotting.py) are domain-agnostic
-- they just consume a list of raw result dicts with a "belief" key. The
only DAG-specific part of belief_plots.py is load_raw_sims(), which builds a
depth-based domain_tag/config and compares regimes across PARAM_PAIRS at one
fixed depth. The UAV sweep has no depth axis at all (fixed SIZE=40), so that
dimension is dropped here rather than faked; everything else -- comparing
regimes across the beta values in PARAM_PAIRS -- carries over unchanged.
"""
import joblib
from simulations.approach import Approach
from simulations.utils import get_results_path, get_grid_tag
from simulations.paper_plots import (
    plot_belief_regime_comparison, plot_belief_regime_comparison_both, ALPHA_INDEX, BETA_INDEX,
)
from simulations.uav_simulations import sweep_config as cfg

_grid_tag = get_grid_tag(cfg.lb_beta, cfg.ub_beta, cfg.lb_alpha, cfg.ub_alpha, cfg.grid_res)


def load_raw_sims(beta, alpha, approach, layout_idx=0, sims=None):
    """
    Load FULL (unstripped) result dicts -- including "belief" -- for belief
    heatmaps. These are heavy, so only load a handful; defaults to just
    sim 1 (all there is, given cfg.num_sims=1). If num_sims is raised later,
    pass sims=range(1, cfg.num_sims + 1) to average over all of them.
    """
    beta, alpha = float(beta), float(alpha)  # match run_one_combo.py's argparse(type=float)
    if sims is None:
        sims = range(1, cfg.num_sims + 1)

    domain_name = "uav"
    domain_tag = f"size{cfg.SIZE}_pobs{cfg.P_OBS}"  # matches UAVDomain.id_tag
    layout_seed = cfg.SEED + 1000 * layout_idx

    # same config as build_summary_df_uav.py -- must match what
    # run_one_combo.py passed to get_results_path() when writing
    config = {
        "layout_seed": layout_seed,
        "seed": cfg.SEED,
        "max_depth": cfg.MAX_DEPTH,
        "size": cfg.SIZE,
        "p_obs": cfg.P_OBS,
        "true_beta": beta,
        "true_alpha": alpha,
        "num_sims": cfg.num_sims,
        "num_trials": cfg.num_trials,
        "utility_scale": cfg.UTILITY_SCALE,
    }

    fn_app = cfg.base_fn_app + f"_L{layout_idx + 1}"
    if approach == Approach.BAMCP_ES:
        fn_app += f"_window{cfg.window}_eps{cfg.tol}"

    results = []
    for sim in sims:
        result_fn = get_results_path(
            domain_name=domain_name, domain_tag=domain_tag,
            num_humans=1, num_autos=1, approach=approach,
            true_beta=beta, true_alpha=alpha, seed=cfg.SEED, sim=sim,
            config=config, is_toy=cfg.is_toy, is_cluster=cfg.is_cluster,
            fn_app=fn_app, grid_tag=_grid_tag,
        )
        try:
            results.append(joblib.load(result_fn))
        except FileNotFoundError:
            print(f"missing: {result_fn}")
    return results


def plot_beliefs(paper_plots_dir, presentation_plots_dir, tag, layout_idx=0, sims=None,
                  approach=Approach.BAMCP, param_index=ALPHA_INDEX, true_value=None):
    if true_value is None:
        # PARAM_PAIRS varies the other axis across regimes while holding
        # this one fixed -- pull that fixed value out instead of just
        # grabbing PARAM_PAIRS[0] and assuming the rest match.
        axis_pos = 1 if param_index == ALPHA_INDEX else 0
        values = {pair[axis_pos] for pair in cfg.PARAM_PAIRS}
        if len(values) != 1:
            raise ValueError(
                f"PARAM_PAIRS doesn't hold a single value fixed on the "
                f"{'alpha' if param_index == ALPHA_INDEX else 'beta'} axis "
                "-- pass true_value explicitly."
            )
        true_value = values.pop()


    regime_results = {}
    for true_beta, true_alpha in cfg.PARAM_PAIRS:
        results = load_raw_sims(true_beta, true_alpha, approach, layout_idx, sims=sims)
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

    pdf_save_path = f"{paper_plots_dir}/fig_belief_uav_{param_label}_L{layout_idx+1}{tag}.pdf"
    png_save_path = f"{presentation_plots_dir}/fig_belief_uav_{param_label}_L{layout_idx+1}{tag}.png"

    plot_belief_regime_comparison(
        regime_results, param_index=param_index, true_value=true_value,
        true_value_label=true_value_label, pdf_save_path=pdf_save_path, png_save_path=png_save_path,
    )


def plot_beliefs_both(paper_plots_dir, presentation_plots_dir, tag, layout_idx=0, sims=None,
                       approach=Approach.BAMCP, display_res_beta=None, display_res_alpha=None):
    """
    Two-row (beta, alpha) belief-concentration comparison across the beta
    regimes in cfg.PARAM_PAIRS, at a single layout. Saves both PDF (paper)
    and PNG (slides) versions.
    """
    regimes = []
    for true_beta, true_alpha in cfg.PARAM_PAIRS:
        results = load_raw_sims(true_beta, true_alpha, approach, layout_idx, sims=sims)
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

    fname = f"fig_belief_uav_both_L{layout_idx+1}{tag}"
    pdf_save_path = f"{paper_plots_dir}/{fname}.pdf"
    png_save_path = f"{presentation_plots_dir}/{fname}.png"

    plot_belief_regime_comparison_both(
        regimes, display_res_beta=display_res_beta, display_res_alpha=display_res_alpha,
        pdf_save_path=pdf_save_path, png_save_path=png_save_path,
    )