#!/bin/bash
# Phase C: architecture sensitivity at moderate budget.
#
# 5 architecture combos × 3 methods × 1 seed × 25r × 500g.  Tests F2/F3
# (encoder × extractor matters across methods?) at a budget large enough
# for AC to start learning but small enough to fit ~15 runs in ~15h.
#
# Naming: 9x9_<enc>_<ext>_<method>_med25x500
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

ROUNDS=25
GAMES=500
EVAL_GAMES=50
TAG="med25x500"

train_one() {
    local method="$1" enc="$2" ext="$3" lr="$4" eps="$5"
    ${PY} train_rl.py \
        --board-size 9 --encoder "${enc}" --network "${ext}" --method "${method}" \
        --rounds "${ROUNDS}" --games "${GAMES}" --eval-games "${EVAL_GAMES}" \
        --epsilon "${eps}" --lr "${lr}" --bs 512 2>&1 | \
        grep -E 'init|ROUND|Round [0-9]+ done|vs ref|Traceback|Error|FAIL|Killed'
}

run_combo() {
    local method="$1" enc="$2" ext="$3" lr="$4" eps="$5"
    local slug="${method}_${enc}_${ext}_lr${lr}_b9_${TAG}"
    local mdir="runs/${slug}/checkpoints"
    local edir="runs/${slug}/experience"

    echo "================================================================"
    echo "  Phase C: ${method} ${enc}+${ext} (25r×500g)"
    echo "================================================================"
    if [ -d "runs/${slug}" ]; then echo "  [skip] runs/${slug} exists"; return; fi
    rm -rf "runs/${slug}"
    train_one "${method}" "${enc}" "${ext}" "${lr}" "${eps}"
}

# Architecture combos: (encoder, extractor)
# All BN-free per F1; AC and Q get the extractor sweep, PG keeps oneplane+small
# (its established working config) plus the cross-method headline points.
COMBOS_QAC=(
    "oneplane book_2layer"
    "oneplane small_nobn"
    "oneplane alphazero_nobn"
    "sevenplane book_2layer"
    "alphazero book_2layer"
)

# ─── Q architecture sweep ───
echo "########### Q architecture sweep ###########"
for combo in "${COMBOS_QAC[@]}"; do
    read -r enc ext <<< "${combo}"
    run_combo q "${enc}" "${ext}" 0.1 0.1
done

# ─── AC architecture sweep ───
echo "########### AC architecture sweep ###########"
for combo in "${COMBOS_QAC[@]}"; do
    read -r enc ext <<< "${combo}"
    run_combo ac "${enc}" "${ext}" 0.01 0.0
done

# ─── PG architecture sweep (limited — small is the canonical PG arch) ───
echo "########### PG architecture sweep ###########"
for combo in "oneplane small" "sevenplane small" "oneplane medium_nobn" \
             "alphazero small" "elevenplane small"; do
    read -r enc ext <<< "${combo}"
    run_combo pg "${enc}" "${ext}" 1e-4 0.0
done

echo
echo "##################### Phase C done #####################"
