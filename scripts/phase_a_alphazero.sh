#!/bin/bash
# Phase A: AlphaZero headline runs at large budget for thesis Big Comparison.
#
# 3 seeds × 50 rounds × 1000 self-play games.  num_simulations=25 keeps
# wall time under ~20h on RTX 3090 (PUCT search dominates self-play cost
# vs the other methods).
#
# Naming convention: alphazero_alphazero_alphazero_lr1e-4_b9
# (canonical Run.slug; matches phase_a_train.sh's existing convention —
# seed-isolated dirs require the same upstream slug-with-seed work.)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

ROUNDS=${ROUNDS:-50}
GAMES=${GAMES:-1000}
EVAL_GAMES=${EVAL_GAMES:-200}
MIN_WIN_RATE=${MIN_WIN_RATE:-0.55}
MCTS_SIMS=${MCTS_SIMS:-25}
LR=${LR:-1e-4}
EARLY_STOP=${EARLY_STOP:-15}
TAG="big50x1000"

train_one() {
    local seed="$1"
    ${PY} train_rl.py \
        --board-size 9 \
        --encoder alphazero \
        --network alphazero \
        --method alphazero \
        --rounds "${ROUNDS}" \
        --games "${GAMES}" \
        --eval-games "${EVAL_GAMES}" \
        --min-win-rate "${MIN_WIN_RATE}" \
        --mcts-simulations "${MCTS_SIMS}" \
        --lr "${LR}" \
        --seed "${seed}" \
        --early-stop-rounds "${EARLY_STOP}" \
        --bs 256 2>&1 | \
        grep -E 'init|ROUND|Round [0-9]+ done|vs ref|EARLY STOP|Traceback|Error|FAIL|Killed'
}

run_seed() {
    local seed="$1"
    # Run.seed appends `_seed${N}` to the slug, isolating each seed's
    # checkpoints under its own runs/ dir.
    local slug="alphazero_alphazero_alphazero_lr${LR}_b9_seed${seed}"
    local mdir="runs/${slug}/checkpoints"

    echo "================================================================"
    echo "  Phase A: alphazero alphazero+alphazero lr=${LR}"
    echo "  sims=${MCTS_SIMS} seed=${seed}"
    echo "================================================================"

    if [ -d "${mdir}" ] && ls "${mdir}"/round_*.h5 >/dev/null 2>&1; then
        echo "  [resume] continuing from existing checkpoints in ${mdir}"
    fi
    train_one "${seed}"
}

echo
echo "########### AlphaZero alphazero+alphazero lr=${LR} sims=${MCTS_SIMS} ###########"
for seed in ${SEEDS:-1 2 3}; do
    run_seed ${seed}
done

echo
echo "##################### Phase A AlphaZero done #####################"
