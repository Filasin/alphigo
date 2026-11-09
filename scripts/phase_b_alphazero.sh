#!/bin/bash
# Phase B (AlphaZero): learning-rate sensitivity sweep.
#
# Fixes num_simulations at the Phase A value (default 25) and sweeps LR
# across {1e-5, 1e-4, 1e-3, 1e-2}.  Each LR lands in its own Run dir
# because Run.slug embeds the lr value, so this is naturally
# resume/skip-safe via train_rl.py's existing JSONL key check.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

ROUNDS=${ROUNDS:-10}
GAMES=${GAMES:-200}
EVAL_GAMES=${EVAL_GAMES:-30}
MCTS_SIMS=${MCTS_SIMS:-25}
LRS=${LRS:-"1e-5 1e-4 1e-3 1e-2"}

train_one() {
    local lr="$1"
    ${PY} train_rl.py \
        --board-size 9 \
        --encoder alphazero \
        --network alphazero \
        --method alphazero \
        --rounds "${ROUNDS}" \
        --games "${GAMES}" \
        --eval-games "${EVAL_GAMES}" \
        --mcts-simulations "${MCTS_SIMS}" \
        --lr "${lr}" \
        --bs 256 2>&1 | \
        grep -E 'init|ROUND|Round [0-9]+ done|vs ref|Traceback|Error|FAIL|Killed'
}

for lr in ${LRS}; do
    echo "================================================================"
    echo "=== Phase B AZ lr=${lr} sims=${MCTS_SIMS}"
    echo "================================================================"
    train_one "${lr}"
done

echo
echo "##################### Phase B AlphaZero done #####################"
