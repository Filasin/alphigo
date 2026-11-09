#!/bin/bash
# Parallel sims sweep against the seed-2 smargin retrain.
# 4 cells × N=30, mp adapter with 4 workers.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}
AGENT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/r0_joint_smargin_b19_seed2.h5
CSV=results/sims_sweep_b19_smargin_seed2.csv

rm -f "$CSV"
for sims in 1 25 50 200; do
    echo "[sweep] sims=$sims starting $(date +%H:%M:%S)"
    "$PY" -u scripts/bench_az_vs_gnugo_mp.py \
        --agent "$AGENT" \
        --opponent gnugo:level=5 \
        --num-simulations "$sims" \
        --board-size 19 \
        --num-games 30 \
        --num-workers 4 \
        --out-csv "$CSV" \
        > "logs/sweep_seed2_sims${sims}.log" 2>&1
    rc=$?
    echo "[sweep] sims=$sims done rc=$rc at $(date +%H:%M:%S)"
done
echo "[sweep] DONE $(date +%H:%M:%S)"
