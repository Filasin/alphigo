#!/bin/bash
# sims=800 anchor cell for the 19x19 sweep.
# Waits for the current confirmation sweep to finalize sims=200 in the CSV,
# then runs sims=800 (N=25) vs gnugo:level=5 and appends the row.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

MAIN_CSV=results/sims_sweep_b19_allchunks.csv
ANCHOR_CSV=results/sims_sweep_b19_sims800.csv
AGENT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/r0_value_b19_allchunks.h5

echo "[anchor] waiting for sims=200 row in ${MAIN_CSV}..."
while ! grep -q '^200,' "${MAIN_CSV}" 2>/dev/null; do
    sleep 120
done
echo "[anchor] sims=200 row found, starting sims=800 N=25 run at $(date)"

AGENT="${AGENT}" \
OPP=gnugo:level=5 \
N_GAMES=25 \
SIMS_LIST="800" \
BOARD_SIZE=19 \
OUT_CSV="${ANCHOR_CSV}" \
    bash scripts/sims_sweep.sh

echo "[anchor] appending sims=800 row to ${MAIN_CSV}"
tail -n +2 "${ANCHOR_CSV}" >> "${MAIN_CSV}"
echo "[anchor] done at $(date)"
cat "${MAIN_CSV}"
