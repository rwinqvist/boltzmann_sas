"""
Local UAV sweep driver -- same functionality as run_uav_sweep.sh (prime layout
cache once, then loop over layout-idx x PARAM_PAIRS calling run_one_combo.py
as a subprocess per combo), no lockfile/PID guard, no SLURM directives, no
log files -- output streams straight to the terminal.

Run from the repo root:
    python run_uav_sweep.py
"""
import subprocess
import sys

from simulations.uav_simulations import sweep_config as cfg


def prime_layout_cache():
    """ Generate the layout cache once, serially, before any combo runs. """
    print(f"Priming layout cache (size={cfg.SIZE}, p_obs={cfg.P_OBS}, num_layouts={cfg.NUM_LAYOUTS})...", flush=True)
    from domains.uav.utils import get_layouts
    get_layouts(size=cfg.SIZE, p_obs=cfg.P_OBS, num_layouts=cfg.NUM_LAYOUTS)
    print("Layout cache ready.", flush=True)


def run_combo(lidx, beta, alpha):
    print(f"=== beta={beta} alpha={alpha} layout={lidx + 1} ===", flush=True)

    result = subprocess.run([
        sys.executable, "-m", "simulations.uav_simulations.run_one_combo",
        "--layout-idx", str(lidx), "--beta", str(beta), "--alpha", str(alpha),
    ])

    if result.returncode != 0:
        print(f"!!! FAILED: beta={beta} alpha={alpha} layout={lidx + 1}", flush=True)


def main():
    prime_layout_cache()

    for beta, alpha in cfg.PARAM_PAIRS:
        for lidx in range(cfg.NUM_LAYOUTS):
            run_combo(lidx, beta, alpha)

    print("Sweep complete.", flush=True)


if __name__ == "__main__":
    main()