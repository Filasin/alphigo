#!/bin/bash
# Phase A bootstrap: AZ with SL warmstart (textbook Ch.14 pattern).
#
# Cell 1 (cold-start, seed=1) plateaued at r9 and went 0/30 vs GnuGo
# level=1 — actively *worse* than the alphazero/alphazero SL net it
# could have started from (5/30 = 17% vs the same opponent). This run
# uses bootstrap_from_sl to copy the SL trunk+policy weights into a
# fresh ZeroAgent, then trains as normal.
#
# Bootstrap source: 9x9_alphazero_alphazero_adam_b9_model_epoch_100.h5
#   - val_acc 0.46 (Pumperla MCTS data, 9x9)
#   - 5/30 = 17% wins vs gnugo:level=1 (pure policy, no MCTS)
#
# Hyperparams kept identical to cell 1 (cold-start) for a direct A/B:
#   lr=1e-4, 15 rounds, 200 games/round, eval=50, workers=4
#
# Slug differs by seed (seed=2 here vs seed=1 for cold-start) so the
# checkpoints/jsonl don't collide.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

ROUNDS=${ROUNDS:-15}
GAMES=${GAMES:-200}
EVAL_GAMES=${EVAL_GAMES:-50}
MIN_WIN_RATE=${MIN_WIN_RATE:-0.55}
SIMS=${SIMS:-25}
LR=${LR:-1e-4}
BS=${BS:-256}
EARLY_STOP=${EARLY_STOP:-15}
WORKERS=${WORKERS:-4}
SEED=${SEED:-2}
BOOTSTRAP_SRC=${BOOTSTRAP_SRC:-checkpoints/9x9_alphazero_alphazero_adam_b9_model_epoch_100.h5}

SLUG="alphazero_alphazero_alphazero_lr${LR}_b9_seed${SEED}"
RUN_DIR="runs/${SLUG}"
ARCHIVE_DIR="${RUN_DIR}_bootstrap_sims${SIMS}"

echo
echo "############# Phase A bootstrap: AZ from SL warmstart #############"
echo "  bootstrap source: ${BOOTSTRAP_SRC}"
echo "  hyperparams: lr=${LR}, sims=${SIMS}, ${ROUNDS}r × ${GAMES}g, eval=${EVAL_GAMES}, workers=${WORKERS}"
echo "  slug: ${SLUG}"
echo "  archive on completion -> ${ARCHIVE_DIR}"
echo

if [ ! -f "${BOOTSTRAP_SRC}" ]; then
    echo "ERROR: bootstrap source not found: ${BOOTSTRAP_SRC}"
    exit 1
fi

# Resume support: if the cell-archive exists from a prior run, restore it.
if [ -d "${ARCHIVE_DIR}" ] && [ ! -d "${RUN_DIR}" ]; then
    echo "  [resume] restoring ${ARCHIVE_DIR} -> ${RUN_DIR}"
    mv "${ARCHIVE_DIR}" "${RUN_DIR}"
fi
if [ -d "${RUN_DIR}/checkpoints" ] && \
   ls "${RUN_DIR}/checkpoints"/round_*.h5 >/dev/null 2>&1; then
    echo "  [resume] continuing from existing checkpoints in ${RUN_DIR}/checkpoints"
fi

# Run with parallel-eval (CPU workers) + parallel self-play (GPU workers).
# train_rl.py's --bootstrap is wired to AlphaZeroMethod.bootstrap_from_sl
# which validates the source has encoder='alphazero' and matching board size.
if ! ${PY} -u train_rl.py \
    --board-size 9 \
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
    echo "  [error] train_rl.py exited non-zero — aborting bootstrap run"
    exit 1
fi

# Archive on success so cold-start (seed=1) and bootstrap (seed=2) results
# are both preserved on disk under distinct slugs.
if [ -d "${RUN_DIR}" ]; then
    echo "  [archive] ${RUN_DIR} -> ${ARCHIVE_DIR}"
    mv "${RUN_DIR}" "${ARCHIVE_DIR}"
fi

echo
echo "##################### Phase A bootstrap done #####################"
echo "  Compare to cell 1 (cold-start) by inspecting results/experiments.jsonl"
echo "  for rl_round entries with seed=${SEED} vs seed=1."
