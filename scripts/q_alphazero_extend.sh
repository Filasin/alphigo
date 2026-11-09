#!/bin/bash
# Extend the two strongest alphazero-encoder runs from 5 to 10 rounds
# at the same hyperparameters (lr=0.1, eps=0.1, 200 games/round).
# Tests the hypothesis: does the rich 17-plane encoder just need more
# gradient steps to convert into stable strength?
#
# Resumes from existing rl_agents/9x9_alphazero_*_q checkpoints.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

COMBOS=(
    "alphazero:book_2layer"
    "alphazero:large_nobn"
)

for combo in "${COMBOS[@]}"; do
    enc="${combo%:*}"
    ext="${combo#*:}"
    echo "================================================================"
    echo "=== EXTEND: encoder=${enc}  extractor=${ext}  rounds=10 (resume)"
    echo "================================================================"

    ${PY} train_rl.py \
        --board-size 9 \
        --encoder "${enc}" \
        --network "${ext}" \
        --method q \
        --rounds 10 \
        --games 200 \
        --epsilon 0.1 \
        --lr 0.1 2>&1 | \
        grep -E 'init|ROUND|SKIP|Promoted|Not promoted|new vs best|Round [0-9]+ done|Traceback|Error|FAIL|Killed'
done
echo "================================================================"
echo "=== alphazero extend done ==="
echo "================================================================"
