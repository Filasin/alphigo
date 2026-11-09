#!/bin/bash
# Phase A bootstrap (19x19): AZ from SL warmstart at the data-abundant regime.
#
# Adaptation of phase_a_alphazero_bootstrap.sh for 19x19. The 9x9
# version showed that AZ training cannot beat SL when SL is weak
# (5/30 vs gnugo:level=1 = 17%) because the value head is data-
# bottlenecked and MCTS amplifies its noise (see sims sweep).
#
# 19x19 SL is dramatically stronger thanks to abundant KGS data:
#   supervised_alphazero_alphazero_adam_b19_model_epoch_35.h5
#     val_acc 0.455
#     vs gnugo:level=1  = 82%
#     vs gnugo:level=5  = 76%
#     vs gnugo:level=10 = 62%
#
# The crossover hypothesis: with abundant data, the value head learns
# to discriminate positions, MCTS adds signal, bootstrap+AZ exceeds SL.
#
# Wall budget: per-round ~80 min on 19x19 (~2x of 9x9). 15 rounds ~ 20h.
# WORKERS=3 by default — 19x19 network is ~46MB vs 9x9's 7.6MB, so 4
# workers + main process risks VRAM saturation at 24GB.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

ROUNDS=${ROUNDS:-15}
GAMES=${GAMES:-100}              # smaller than 9x9 due to longer games (4x more moves)
EVAL_GAMES=${EVAL_GAMES:-50}
MIN_WIN_RATE=${MIN_WIN_RATE:-0.55}
SIMS=${SIMS:-25}
LR=${LR:-1e-4}
BS=${BS:-256}
EARLY_STOP=${EARLY_STOP:-15}
WORKERS=${WORKERS:-3}
SEED=${SEED:-1}
BOOTSTRAP_SRC=${BOOTSTRAP_SRC:-checkpoints/supervised_alphazero_alphazero_adam_b19_model_epoch_35.h5}

SLUG="alphazero_alphazero_alphazero_lr${LR}_b19_seed${SEED}"
RUN_DIR="runs/${SLUG}"
ARCHIVE_DIR="${RUN_DIR}_bootstrap_b19_sims${SIMS}"

echo
echo "############# Phase A bootstrap b19: AZ from SL warmstart #############"
echo "  bootstrap source: ${BOOTSTRAP_SRC}"
echo "  board size: 19, sims=${SIMS}, lr=${LR}"
echo "  rounds: ${ROUNDS} x ${GAMES} games, eval=${EVAL_GAMES}, workers=${WORKERS}"
echo "  slug: ${SLUG}"
echo "  archive on completion -> ${ARCHIVE_DIR}"
echo

if [ ! -f "${BOOTSTRAP_SRC}" ]; then
    echo "ERROR: bootstrap source not found: ${BOOTSTRAP_SRC}"
    exit 1
fi

# Resume support.
if [ -d "${ARCHIVE_DIR}" ] && [ ! -d "${RUN_DIR}" ]; then
    echo "  [resume] restoring ${ARCHIVE_DIR} -> ${RUN_DIR}"
    mv "${ARCHIVE_DIR}" "${RUN_DIR}"
fi
if [ -d "${RUN_DIR}/checkpoints" ] && \
   ls "${RUN_DIR}/checkpoints"/round_*.h5 >/dev/null 2>&1; then
    echo "  [resume] continuing from existing checkpoints in ${RUN_DIR}/checkpoints"
fi

if ! ${PY} -u train_rl.py \
    --board-size 19 \
    --encoder alphazero \
    --network alphazero \
    --method alphazero \
    --rounds "${ROUNDS}" \
    --games "${GAMES}" \
    --eval-games "${EVAL_GAMES}" \
    --min-win-rate "${MIN_WIN_RATE}" \
    --mcts-simulations "${SIMS}" \
    --num-workers "${WORKERS}" \
    --lr "${LR}" \
    --bs "${BS}" \
    --seed "${SEED}" \
    --early-stop-rounds "${EARLY_STOP}" \
    --bootstrap "${BOOTSTRAP_SRC}"; then
    echo "  [error] train_rl.py exited non-zero — aborting 19x19 bootstrap"
    exit 1
fi

if [ -d "${RUN_DIR}" ]; then
    echo "  [archive] ${RUN_DIR} -> ${ARCHIVE_DIR}"
    mv "${RUN_DIR}" "${ARCHIVE_DIR}"
fi

echo
echo "##################### Phase A bootstrap b19 done #####################"
echo "  Final agent: ${ARCHIVE_DIR}/checkpoints/round_${ROUNDS}.h5"
echo "  Inspect rounds via results/experiments.jsonl (filter seed=${SEED}, board_size=19)"
echo "  Eval the best round vs gnugo with scripts/eval_az_r9_vs_gnugo.sh"
echo "    (or just run.py benchmark --board-size 19)"
