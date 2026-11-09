#!/bin/bash
# Phase A: headline runs at large budget for thesis Big Comparison.
#
# 3 methods × 3 seeds × 50 rounds × 1000 games each.  Each method at
# its tuned config (Q lr=0.1, PG lr=1e-4, AC lr=0.01).  ~14h total
# wall clock at ~92 min/run on RTX 3090.
#
# Naming convention: 9x9_<enc>_<ext>_<method>_big50x1000_seed{1,2,3}
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

ROUNDS=${ROUNDS:-50}
GAMES=${GAMES:-1000}
EVAL_GAMES=${EVAL_GAMES:-200}
MIN_WIN_RATE=${MIN_WIN_RATE:-0.55}
EARLY_STOP=${EARLY_STOP:-15}
TAG="big50x1000"

train_one() {
    local method="$1" enc="$2" ext="$3" lr="$4" eps="$5" seed="$6" extra_env="$7"
    eval "${extra_env}" ${PY} train_rl.py \
        --board-size 9 \
        --encoder "${enc}" \
        --network "${ext}" \
        --method "${method}" \
        --rounds "${ROUNDS}" \
        --games "${GAMES}" \
        --eval-games "${EVAL_GAMES}" \
        --min-win-rate "${MIN_WIN_RATE}" \
        --epsilon "${eps}" \
        --lr "${lr}" \
        --seed "${seed}" \
        --early-stop-rounds "${EARLY_STOP}" \
        --bs 512 2>&1 | \
        grep -E 'init|ROUND|Round [0-9]+ done|vs ref|EARLY STOP|Traceback|Error|FAIL|Killed'
}

run_seed() {
    local method="$1" enc="$2" ext="$3" lr="$4" eps="$5" seed="$6" extra_env="${7:-}"
    # Run.seed appends `_seed${N}` to the slug, isolating each seed's
    # checkpoints under its own runs/ dir. (Pre-Run.seed scripts wrote
    # all 3 seeds into the same dir, overwriting each other; this
    # version produces independent runs.)
    local slug="${method}_${enc}_${ext}_lr${lr}_b9_seed${seed}"
    local mdir="runs/${slug}/checkpoints"

    echo "================================================================"
    echo "  Phase A: ${method} ${enc}+${ext} lr=${lr} eps=${eps} seed=${seed}"
    echo "================================================================"

    if [ -d "${mdir}" ] && ls "${mdir}"/round_*.h5 >/dev/null 2>&1; then
        echo "  [resume] continuing from existing checkpoints in ${mdir}"
    fi
    train_one "${method}" "${enc}" "${ext}" "${lr}" "${eps}" "${seed}" "${extra_env}"
}

# ─── Q (book Ch 11) ────────────────────────────────────────────────
echo
echo "########### Q oneplane+book_2layer lr=0.1 ###########"
for seed in ${SEEDS:-1 2 3}; do
    run_seed q oneplane book_2layer 0.1 0.1 ${seed}
done

# ─── PG (REINFORCE, book Ch 9-10) ──────────────────────────────────
echo
echo "########### PG oneplane+small lr=1e-4 ###########"
for seed in ${SEEDS:-1 2 3}; do
    run_seed pg oneplane small 1e-4 0.0 ${seed}
done

# ─── AC (book Ch 12) ───────────────────────────────────────────────
echo
echo "########### AC oneplane+book_2layer lr=0.01 ###########"
for seed in ${SEEDS:-1 2 3}; do
    run_seed ac oneplane book_2layer 0.01 0.0 ${seed}
done

echo
echo "##################### Phase A done #####################"
