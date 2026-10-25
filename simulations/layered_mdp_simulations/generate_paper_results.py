"""Generate DAG paper figures and FIHP switching statistics from cached results.

Run from the project root:
    python -m simulations.layered_mdp_simulations.generate_paper_results
"""
from pathlib import Path

import pandas as pd

from simulations.approach import Approach
from simulations.paper_plots import set_paper_style
from simulations.layered_mdp_simulations import sweep_config as cfg
from simulations.layered_mdp_simulations.make_plots import make_plots


def validate_cache(df):
    required = {
        "depth", "beta", "alpha", "approach", "total_reward", "total_time",
        "stopped_early", "human_obs_at_switch", "total_human_obs",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing cached columns: {sorted(missing)}")

    print(f"Loaded {len(df)} cached rows.")
    for depth in cfg.DEPTHS:
        for beta, alpha in cfg.PARAM_PAIRS:
            subset = df.loc[
                (df["depth"] == depth)
                & (df["beta"] == beta)
                & (df["alpha"] == alpha)
            ]
            counts = subset.groupby("approach").size().to_dict()
            print(f"  depth={depth}, beta={beta}, alpha={alpha}: {counts}")
            if not counts.get(Approach.BAMCP_ES.value):
                raise ValueError(f"No FIHP rows for depth={depth}, beta={beta}, alpha={alpha}")


def generate_switching_table(df):
    """Calculate switching statistics by depth and true behavior parameters."""
    fihp = df.loc[df["approach"] == Approach.BAMCP_ES.value].copy()
    fihp["switched"] = fihp["stopped_early"].eq(True)
    fihp["human_obs_at_switch"] = pd.to_numeric(
        fihp["human_obs_at_switch"], errors="coerce"
    )
    fihp["total_human_obs"] = pd.to_numeric(
        fihp["total_human_obs"], errors="coerce"
    )

    rows = []
    for depth in cfg.DEPTHS:
        for beta, alpha in cfg.PARAM_PAIRS:
            subset = fihp.loc[
                (fihp["depth"] == depth)
                & (fihp["beta"] == beta)
                & (fihp["alpha"] == alpha)
            ]
            if subset.empty:
                raise ValueError(f"Missing FIHP group: {depth=}, {beta=}, {alpha=}")
            switched = subset["switched"]
            obs = subset.loc[switched, "human_obs_at_switch"].dropna()
            no_switch_obs = subset.loc[~switched, "total_human_obs"].dropna()
            if len(obs) != int(switched.sum()):
                print(f"Warning: missing switch observation counts for {depth=}, {beta=}")
            if len(no_switch_obs) != int((~switched).sum()):
                print(f"Warning: missing non-switch observation counts for {depth=}, {beta=}")
            rows.append({
                "depth": depth,
                "beta": beta,
                "alpha": alpha,
                "n_runs": len(subset),
                "n_switched": int(switched.sum()),
                "switch_rate_pct": 100 * switched.mean(),
                "human_obs_median": obs.median(),
                "human_obs_mean": obs.mean(),
                "human_obs_std": obs.std(),
                "human_obs_q25": obs.quantile(0.25),
                "human_obs_q75": obs.quantile(0.75),
                "no_switch_obs_median": no_switch_obs.median(),
                "no_switch_obs_mean": no_switch_obs.mean(),
                "no_switch_obs_std": no_switch_obs.std(),
                "no_switch_obs_q25": no_switch_obs.quantile(0.25),
                "no_switch_obs_q75": no_switch_obs.quantile(0.75),
            })

    table = pd.DataFrame(rows)
    output_dir = Path(cfg.CACHE_PATH) / "latex_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dag_switching_statistics.csv"
    tex_path = output_dir / "dag_switching_statistics.tex"
    table.to_csv(csv_path, index=False)

    def fmt(value):
        return "--" if pd.isna(value) else f"{value:g}"

    # Detailed table: group each depth using LaTeX multirow.
    # Add \usepackage{multirow} to the paper preamble.
    latex_rows = []
    for depth in cfg.DEPTHS:
        depth_rows = table.loc[table["depth"] == depth]
        for i, row in enumerate(depth_rows.itertuples(index=False)):
            depth_cell = rf"\multirow{{{len(depth_rows)}}}{{*}}{{{depth}}}" if i == 0 else ""
            latex_rows.append(
                f"{depth_cell} & {fmt(row.beta)} & {fmt(row.switch_rate_pct)} "
                f"& {fmt(row.human_obs_median)} & {fmt(row.no_switch_obs_median)} "
                + "\\" * 2 + "\n"
            )
        latex_rows.append("\\hline\n")

    latex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\caption{FIHP switching statistics across DAG regimes "
        "($\\alpha_0=1.2$). Human-observation statistics are reported "
        "as medians, separately for switched and non-switching runs.}\n"
        "\\label{tab:dag_fihp_switching}\n"
        "\\begin{tabular}{ccccc}\n"
        "\\hline\n"
        r"$d$ & $\beta_0$ & Switched (\%) & Switch obs. & No-switch obs. \\" + "\n"
        "\\hline\n"
        + "".join(latex_rows)
        + "\\end{tabular}\n"
        "\\end{table}\n"
    )
    tex_path.write_text(latex, encoding="utf-8")

    # Compact paper table: one row per depth, one column per beta.
    # Each cell is switching percentage / median human observations at switch.
    compact_path = output_dir / "dag_switching_statistics_compact.tex"
    betas = [beta for beta, _ in cfg.PARAM_PAIRS]
    if len(set(betas)) != len(betas):
        raise ValueError("Compact table requires distinct beta values in PARAM_PAIRS")
    if len({alpha for _, alpha in cfg.PARAM_PAIRS}) != 1:
        raise ValueError("Compact table requires a common alpha across PARAM_PAIRS")

    compact_rows = []
    for depth in cfg.DEPTHS:
        cells = [str(depth)]
        for beta in betas:
            match = table.loc[(table["depth"] == depth) & (table["beta"] == beta)]
            if len(match) != 1:
                raise ValueError(f"Expected one row for depth={depth}, beta={beta}")
            row = match.iloc[0]
            cells.append(f"{fmt(row['switch_rate_pct'])} / {fmt(row['human_obs_median'])}")
        compact_rows.append(" & ".join(cells) + " " + "\\" * 2 + "\n")

    beta_headers = " & ".join(rf"$\beta_0={fmt(beta)}$" for beta in betas)
    alpha = cfg.PARAM_PAIRS[0][1]
    compact_latex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{FIHP switching in the DAG domain ($\\alpha_0=" + fmt(alpha) +
        "$). Entries give switching percentage / median human observations "
        "at switch; medians are over switched runs only.}\n"
        "\\label{tab:dag_fihp_switching_compact}\n"
        "\\begin{tabular}{" + "c" * (len(betas) + 1) + "}\n"
        "\\hline\n"
        r"$d$ & " + beta_headers + " " + "\\" * 2 + "\n"
        "\\hline\n"
        + "".join(compact_rows)
        + "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    compact_path.write_text(compact_latex, encoding="utf-8")
    print(f"Saved: {compact_path}")

    # Grouped paper table: three subcolumns beneath each beta header.
    grouped_path = output_dir / "dag_switching_statistics_grouped.tex"
    grouped_rows = []
    for depth in cfg.DEPTHS:
        cells = [str(depth)]
        for beta in betas:
            match = table.loc[(table["depth"] == depth) & (table["beta"] == beta)]
            if len(match) != 1:
                raise ValueError(f"Expected one row for depth={depth}, beta={beta}")
            row = match.iloc[0]
            cells.extend([
                fmt(row["switch_rate_pct"]),
                fmt(row["human_obs_median"]),
                fmt(row["no_switch_obs_median"]),
            ])
        grouped_rows.append(" & ".join(cells) + " " + "\\" * 2 + "\n")

    group_headers = " & ".join(
        rf"\multicolumn{{3}}{{c}}{{$\beta_0={fmt(beta)}$}}" for beta in betas
    )
    sub_headers = " & ".join(
        [r"Switch (\%) & Sw. obs. & No sw. obs."] * len(betas)
    )
    grouped_latex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{FIHP switching in the DAG domain ($\\alpha_0=" + fmt(alpha) +
        "$). Sw. obs. is the median number of human observations at switching "
        "among switched runs; No sw. obs. is the median total number of human "
        "observations among non-switching runs.}\n"
        "\\label{tab:dag_fihp_switching_grouped}\n"
        "\\begin{tabular}{c" + "ccc" * len(betas) + "}\n"
        "\\hline\n"
        r"$d$ & " + group_headers + " " + "\\" * 2 + "\n"
        " & " + sub_headers + " " + "\\" * 2 + "\n"
        "\\hline\n"
        + "".join(grouped_rows)
        + "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    grouped_path.write_text(grouped_latex, encoding="utf-8")
    print(f"Saved: {grouped_path}")

    print("\n=== DAG FIHP switching statistics ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")
    return table


def main():
    cache_path = Path(cfg.cached_fn)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Missing cached results: {cache_path}\n"
            "Run aggregate_dag_results.py first."
        )
    df = pd.read_pickle(cache_path)
    validate_cache(df)
    generate_switching_table(df)
    set_paper_style(base_size=12)
    make_plots(df)
    print("Done: DAG figures and switching table generated.")


if __name__ == "__main__":
    main()
