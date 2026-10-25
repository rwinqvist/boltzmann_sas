"""Generate paper figures from already-aggregated UAV simulation results.

Run from the project root:
    python -m simulations.uav_simulations.generate_paper_results

"""

from pathlib import Path
import pandas as pd

from simulations.approach import Approach

from simulations.uav_simulations import sweep_config as cfg
from simulations.uav_simulations.aggregate_uav_regimes import (
    REGIMES,
    REGIMES_CACHE_PATH,
    regime_key,
)
from simulations.uav_simulations import plot_regime_comparison
from simulations.paper_plots import set_paper_style


def validate_regimes_cache(df):
    """Catch missing or incomplete regime groups before plotting."""
    required = {"regime_key", "approach", "total_reward", "total_time"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in cached results: {sorted(missing)}")

    expected = {regime_key(regime) for regime in REGIMES}
    available = set(df["regime_key"].dropna())
    missing_regimes = expected - available
    if missing_regimes:
        raise ValueError(f"No cached rows for regimes: {sorted(missing_regimes)}")

    print(f"Loaded {len(df)} rows covering {len(expected)} configured regimes.")
    for regime in REGIMES:
        subset = df.loc[df["regime_key"] == regime_key(regime)]
        counts = subset.groupby("approach").size().to_dict()
        print(f"  {regime_key(regime)}: {counts}")


def generate_switching_table(df):
    """Summarize FIHP switching by regime; no raw joblib files needed."""
    required = {"stopped_early", "human_obs_at_switch"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing switching columns: {sorted(missing)}")

    rows = []
    for regime in REGIMES:
        subset = df.loc[
            (df["regime_key"] == regime_key(regime))
            & (df["approach"] == Approach.BAMCP_ES.value)
        ]
        if subset.empty:
            raise ValueError(f"No FIHP results for {regime_key(regime)}")

        switched = subset["stopped_early"].fillna(False).astype(bool)
        obs = pd.to_numeric(
            subset.loc[switched, "human_obs_at_switch"], errors="coerce"
        ).dropna()
        if switched.sum() != len(obs):
            print(f"Warning: missing observation counts for {regime_key(regime)}")

        rows.append({
            "size": regime["size"],
            "beta": regime["true_beta"],
            "alpha": regime["true_alpha"],
            "num_trials": regime["num_trials"],
            "n_runs": len(subset),
            "n_switched": int(switched.sum()),
            "switch_rate_pct": 100 * switched.mean(),
            "human_obs_median": obs.median(),
            "human_obs_q25": obs.quantile(0.25),
            "human_obs_q75": obs.quantile(0.75),
        })

    table = pd.DataFrame(rows)
    output_dir = Path(cfg.CACHE_PATH) / "latex_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "uav_switching_statistics.csv"
    table.to_csv(csv_path, index=False)
    print("\n=== FIHP switching statistics ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"Saved: {csv_path}")
    return table

def main():
    cache_path = Path(REGIMES_CACHE_PATH)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Missing cached results: {cache_path}\n"
            "Run aggregate_uav_regimes.py first."
        )

    df = pd.read_pickle(cache_path)
    validate_regimes_cache(df)

    # Reuse the plotting code that has already reproduced the paper figure.
    # Its main() reads the same cached DataFrame internally.
    set_paper_style(base_size=14)
    plot_regime_comparison.main()

    generate_switching_table(df)

    print("Done: UAV figure and switching table generated.")


if __name__ == "__main__":
    main()
