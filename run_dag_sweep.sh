#!/bin/bash
set -uo pipefail   # NOT -e: one failed combo shouldn't kill the whole overnight run

LOCKFILE="/tmp/run_dag_sweep.lock"
PIDFILE="/tmp/run_dag_sweep.pid"

if [ -e "$LOCKFILE" ]; then
  echo "Sweep already running (lockfile exists at $LOCKFILE, PID $(cat "$PIDFILE" 2>/dev/null)). Exiting."
  exit 1
fi
touch "$LOCKFILE"
echo $$ > "$PIDFILE"
trap "rm -f $LOCKFILE $PIDFILE" EXIT

DEPTHS=(100 150 200)
NUM_LAYOUTS=10

# beta,alpha pairs -- one "beta,alpha" string per entry, comma-separated
PARAM_PAIRS=("0.5,1.2" "1.5,1.2" "5,1.2")

LOGDIR="dag_sweep_logs"
mkdir -p "$LOGDIR"

for pair in "${PARAM_PAIRS[@]}"; do
  IFS=',' read -r BETA ALPHA <<< "$pair"

  for depth in "${DEPTHS[@]}"; do
    for ((lidx=0; lidx<NUM_LAYOUTS; lidx++)); do
      logfile="$LOGDIR/depth${depth}_layout$((lidx+1))_b${BETA}_a${ALPHA}.log"
      echo "=== beta=$BETA alpha=$ALPHA depth=$depth layout=$((lidx+1)) ==="
      python -m simulations.layered_mdp_simulations.run_one_combo \
        --depth "$depth" --layout-idx "$lidx" --beta "$BETA" --alpha "$ALPHA" \
        > "$logfile" 2>&1

      if [ $? -ne 0 ]; then
        echo "!!! FAILED: beta=$BETA alpha=$ALPHA depth=$depth layout=$((lidx+1)) -- see $logfile"
      fi
    done
  done
done

echo "Sweep complete."