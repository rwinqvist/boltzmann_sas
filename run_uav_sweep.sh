#!/bin/bash
#SBATCH --job-name=uav-sweep
#SBATCH --output=uav_sweep_logs/slurm_%j.out
#SBATCH --error=uav_sweep_logs/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=<FILL IN: e.g. shared>
#SBATCH --account=<FILL IN: your NAISS project ID>

set -uo pipefail   # NOT -e: one failed combo shouldn't kill the whole run


LOCKFILE="/tmp/run_uav_sweep.lock"
PIDFILE="/tmp/run_uav_sweep.pid"

if [ -e "$LOCKFILE" ]; then
  echo "Sweep already running (lockfile exists at $LOCKFILE, PID $(cat "$PIDFILE" 2>/dev/null)). Exiting."
  exit 1
fi
touch "$LOCKFILE"
echo $$ > "$PIDFILE"
trap "rm -f $LOCKFILE $PIDFILE" EXIT

# -- pulled straight from sweep_config.py, so this script can't drift from
# what run_one_combo.py / build_summary_df.py / make_plots.py assume ran --
eval "$(python -c "
from simulations.uav_simulations import sweep_config as cfg
print(f'SIZE={cfg.SIZE}')
print(f'P_OBS={cfg.P_OBS}')
print(f'NUM_LAYOUTS={cfg.NUM_LAYOUTS}')
print('PARAM_PAIRS=(' + ' '.join(f\"\\\"{b},{a}\\\"\" for b, a in cfg.PARAM_PAIRS) + ')')
")"

if [ -z "${SIZE:-}" ]; then
  echo "!!! Failed to load sweep_config.py values -- aborting."
  exit 1
fi

LOGDIR="uav_sweep_logs"
mkdir -p "$LOGDIR"

# --- prime the layout cache ONCE, serially, before any combo runs.
# get_layouts() only generates+dumps if size{SIZE}_pobs{P_OBS}.joblib doesn't
# already exist; doing this single-threaded up front means every later
# run_one_combo.py call just loads the cached file, so there's no race
# between combos writing it concurrently. ---
echo "Priming layout cache (size=$SIZE, p_obs=$P_OBS, num_layouts=$NUM_LAYOUTS)..."
python -c "
from domains.uav.utils import get_layouts
get_layouts(size=$SIZE, p_obs=$P_OBS, num_layouts=$NUM_LAYOUTS)
print('Layout cache ready.')
"
if [ $? -ne 0 ]; then
  echo "!!! Failed to prime layout cache -- aborting sweep."
  exit 1
fi

for pair in "${PARAM_PAIRS[@]}"; do
  IFS=',' read -r BETA ALPHA <<< "$pair"

  for ((lidx=0; lidx<NUM_LAYOUTS; lidx++)); do
    logfile="$LOGDIR/layout$((lidx+1))_b${BETA}_a${ALPHA}.log"
    echo "=== beta=$BETA alpha=$ALPHA layout=$((lidx+1)) ==="
    python -m simulations.uav_simulations.run_one_combo \
      --layout-idx "$lidx" --beta "$BETA" --alpha "$ALPHA" \
      > "$logfile" 2>&1

    if [ $? -ne 0 ]; then
      echo "!!! FAILED: beta=$BETA alpha=$ALPHA layout=$((lidx+1)) -- see $logfile"
    fi
  done
done

echo "Sweep complete."