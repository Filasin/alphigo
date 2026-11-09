#!/bin/bash
# sims=800 anchor for separate margin value net.
# Waits for the sims=200 extra run to finish, then runs sims=800 N=30.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

AGENT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/r0_joint_smargin_b19.h5
VALUE=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/value_standalone_margin_b19.h5
CSV=results/sims_sweep_b19_separate_margin_800.csv

echo "[anchor] waiting for sims=200 extra run to finish..."
while pgrep -f "sims200_extra" > /dev/null 2>&1; do
    sleep 120
done
echo "[anchor] sims=200 extra done, starting sims=800 N=30 at $(date)"

"$PY" -u scripts/bench_az_vs_gnugo_mp.py \
    --agent "$AGENT" \
    --value-model "$VALUE" \
    --opponent gnugo:level=5 \
    --num-simulations 800 \
    --board-size 19 \
    --num-games 30 \
    --num-workers 4 \
    --out-csv "$CSV"

echo "[anchor] done at $(date)"
cat "$CSV"
