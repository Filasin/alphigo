#!/bin/bash
# Phase B: hyperparameter sensitivity at large budget.
#
# Each method × 4 lrs × 1 seed × 50r × 1000g.  Tests F8/F11 at scale —
# does the "disjoint usable lr range" finding hold under more data?
#
# Naming: 9x9_<enc>_<ext>_<method>_big50x1000_lr<value>
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

ROUNDS=50
GAMES=1000
EVAL_GAMES=50
TAG="big50x1000"

train_one() {
    local method="$1" enc="$2" ext="$3" lr="$4" eps="$5"
    ${PY} train_rl.py \
        --board-size 9 --encoder "${enc}" --network "${ext}" --method "${method}" \
        --rounds "${ROUNDS}" --games "${GAMES}" --eval-games "${EVAL_GAMES}" \
        --epsilon "${eps}" --lr "${lr}" --bs 512 2>&1 | \
        grep -E 'init|ROUND|Round [0-9]+ done|vs ref|Traceback|Error|FAIL|Killed'
}

run_lr() {
    local method="$1" enc="$2" ext="$3" lr="$4" eps="$5"
    local slug="${method}_${enc}_${ext}_lr${lr}_b9_${TAG}"
    local mdir="runs/${slug}/checkpoints"
    local edir="runs/${slug}/experience"

    echo "================================================================"
    echo "  Phase B: ${method} ${enc}+${ext} lr=${lr} eps=${eps}"
    echo "================================================================"
    if [ -d "runs/${slug}" ]; then echo "  [skip] runs/${slug} exists"; return; fi
    rm -rf "runs/${slug}"
    train_one "${method}" "${enc}" "${ext}" "${lr}" "${eps}"
}

# ─── Q lr sensitivity (oneplane+book_2layer) ───
echo "########### Q lr sensitivity ###########"
for lr in 1e-3 1e-2 1e-1 0.5; do
    run_lr q oneplane book_2layer "${lr}" 0.1
done

# ─── PG lr sensitivity (oneplane+small) ───
echo "########### PG lr sensitivity ###########"
for lr in 1e-5 1e-4 1e-3 1e-2; do
    run_lr pg oneplane small "${lr}" 0.0
done

# ─── AC lr sensitivity (oneplane+book_2layer) ───
echo "########### AC lr sensitivity ###########"
for lr in 1e-4 1e-3 1e-2 1e-1; do
    run_lr ac oneplane book_2layer "${lr}" 0.0
done

echo
echo "##################### Phase B done #####################"
