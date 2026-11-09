#!/bin/bash
# 6-combo Q-learning sweep (book-faithful two-input).
# Each combo: 5 rounds x 200 games x eps=0.1 x lr=0.1.
# Re-runs combo 1 under the new fixed (512, 256) dense merge so the
# sweep is apples-to-apples across all 6.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

COMBOS=(
    "oneplane:book_2layer"
    "sevenplane:book_2layer"
    "sevenplane:large"
    "alphazero:book_2layer"
    "alphazero:large"
    "alphazero:alphazero"
)

for combo in "${COMBOS[@]}"; do
    enc="${combo%:*}"
    ext="${combo#*:}"
    echo "================================================================"
    echo "=== SWEEP: encoder=${enc}  extractor=${ext}"
    echo "================================================================"

    rm -rf "rl_agents/9x9_${enc}_${ext}_q"
    rm -rf "rl_experience/9x9_${enc}_${ext}_q"

    ${PY} train_rl.py \
        --board-size 9 \
        --encoder "${enc}" \
        --network "${ext}" \
        --method q \
        --rounds 5 \
        --games 200 \
        --epsilon 0.1 \
        --lr 0.1 2>&1 | \
        grep -E 'init|ROUND|Promoted|Not promoted|new vs best|Round [0-9]+ done|Traceback|Error|FAIL|Killed'
done
echo "================================================================"
echo "=== sweep done ==="
echo "================================================================"
