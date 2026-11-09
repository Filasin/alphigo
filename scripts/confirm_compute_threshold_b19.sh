#!/bin/bash
# Compute-threshold confirmation experiment (19x19).
#
# Hypothesis: MCTS fails at our scale because the value head is
# insufficiently trained, not because the algorithm is wrong.
#
# Procedure:
#   1. Retrain value head on ALL 10 SGF chunks (2M positions, ~10K games)
#      using --max-samples to keep RAM bounded.
#   2. Run sims sweep (1, 25, 50, 200) with the new checkpoint.
#   3. Compare to the 1-chunk result (208K positions):
#        - If sims=25 or 200 now beats pure policy (80%) → threshold found.
#        - If still below → threshold requires >10K games of data.
#   Either outcome is publishable.
#
# Wall time: ~1h value train + ~20h sims sweep.
# RAM: ~12GB peak (--max-samples 30000 per chunk × 10 = 300K samples).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

SRC=${SRC:-runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/round_000.h5}
EXP_GLOB=${EXP_GLOB:-'data/sgf_value_buffer_b19_*.h5'}
DST=${DST:-runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/r0_value_b19_allchunks.h5}
MAX_SAMPLES=${MAX_SAMPLES:-300000}
SWEEP_N=${SWEEP_N:-50}
SWEEP_SIMS=${SWEEP_SIMS:-"1 25 50 200"}
OPP=${OPP:-gnugo:level=5}
BOARD_SIZE=${BOARD_SIZE:-19}
OUT_CSV=${OUT_CSV:-results/sims_sweep_b19_allchunks.csv}

echo
echo "===== Compute-threshold confirmation (19x19) ====="
echo "  src checkpoint: ${SRC}"
echo "  data glob:      ${EXP_GLOB}"
echo "  max samples:    ${MAX_SAMPLES} (spread across all chunks)"
echo "  dst checkpoint: ${DST}"
echo "  sims sweep:     ${SWEEP_SIMS}, N=${SWEEP_N} vs ${OPP}"
echo

# Step 1: retrain value head on all chunks.
echo "--- Step 1: value head training ---"
if ! ${PY} -u scripts/train_value_head_only.py \
    --src "${SRC}" \
    --exp-glob "${EXP_GLOB}" \
    --dst "${DST}" \
    --max-samples "${MAX_SAMPLES}" \
    --epochs 20 \
    --bs 256 \
    --lr 1e-3; then
    echo "ERROR: value head training failed"
    exit 1
fi

echo
echo "--- Step 2: sims sweep ---"
AGENT="${DST}" \
OPP="${OPP}" \
N_GAMES="${SWEEP_N}" \
SIMS_LIST="${SWEEP_SIMS}" \
OUT_CSV="${OUT_CSV}" \
BOARD_SIZE="${BOARD_SIZE}" \
    bash scripts/sims_sweep.sh

echo
echo "===== Done. Results: ${OUT_CSV} ====="
echo "Compare to 1-chunk baseline:"
echo "  sims=1:   24/30 = 80% (pure policy)"
echo "  sims=25:  21/30 = 70% (1-chunk MCTS)"
echo "  sims=200: ~42/100 = 42% (1-chunk MCTS, from current sweep)"
