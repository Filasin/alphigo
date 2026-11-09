#!/bin/bash
# Re-train the 4 PG combos at lr=0.1 (matching the book's tuned Q
# hyperparameter) so we can disentangle "Q architecture wins" from
# "Q's lr is just bigger".
#
# Backs up the existing lr=1e-4 PG dirs (which carry the 100-round
# legacy runs) under a *_pg_lr1e4 suffix before training fresh.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

COMBOS=(
    "oneplane:small"
    "sevenplane:large"
    "alphazero:alphazero"
    "elevenplane:large"
)

for combo in "${COMBOS[@]}"; do
    enc="${combo%:*}"; net="${combo#*:}"
    dir="rl_agents/9x9_${enc}_${net}"
    bak="${dir}_pg_lr1e4"
    expdir="rl_experience/9x9_${enc}_${net}"
    expbak="${expdir}_pg_lr1e4"

    echo "================================================================"
    echo "=== PG@lr=0.1: encoder=${enc}  network=${net}"
    echo "================================================================"

    if [ -d "${dir}" ]; then
        if [ -d "${bak}" ]; then rm -rf "${dir}"
        else mv "${dir}" "${bak}"; fi
    fi
    if [ -d "${expdir}" ]; then
        if [ -d "${expbak}" ]; then rm -rf "${expdir}"
        else mv "${expdir}" "${expbak}"; fi
    fi

    ${PY} train_rl.py \
        --board-size 9 \
        --encoder "${enc}" \
        --network "${net}" \
        --method policy_gradient \
        --rounds 5 \
        --games 200 \
        --lr 0.1 2>&1 | \
        grep -E 'init|ROUND|Promoted|Not promoted|Round [0-9]+ done|Traceback|Error|FAIL|Killed'
done
echo "================================================================"
echo "=== pg lr=0.1 sweep done ==="
echo "================================================================"
