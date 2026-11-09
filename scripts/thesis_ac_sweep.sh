#!/bin/bash
# AC sweep for thesis comparison.
#
# Trains:
#   - oneplane + book_2layer × 3 seeds × 5 rounds × 200 games (apples-to-apples
#     vs Q's headline)
#   - oneplane + small_nobn × 1 seed (extractor probe; small_nobn is BN-free
#     to avoid the F1 BN pathology that affects tanh+MSE value heads)
#
# Following book Ch 12: lr=0.01, batch_size=512.  Epsilon=0.0 because the
# softmax policy is already stochastic.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

train_ac() {
    local enc="$1" ext="$2" rounds="$3"
    ${PY} train_rl.py \
        --board-size 9 \
        --encoder "${enc}" \
        --network "${ext}" \
        --method actor_critic \
        --rounds "${rounds}" \
        --games 200 \
        --epsilon 0.0 \
        --lr 0.01 \
        --bs 512 2>&1 | \
        grep -E 'init|ROUND|Round [0-9]+ done|vs ref|Traceback|Error|FAIL|Killed'
}

# ─── E1: oneplane + book_2layer multi-seed (5 rounds) ───
echo "==================== AC: oneplane+book_2layer multi-seed ===================="
for seed in 1 2 3; do
    echo "--- ac oneplane+book_2layer seed ${seed} ---"
    rm -rf "rl_agents/9x9_oneplane_book_2layer_ac" \
           "rl_experience/9x9_oneplane_book_2layer_ac"
    train_ac oneplane book_2layer 5
    [ -d "rl_agents/9x9_oneplane_book_2layer_ac" ] && \
        mv "rl_agents/9x9_oneplane_book_2layer_ac" \
           "rl_agents/9x9_oneplane_book_2layer_ac_seed${seed}"
    [ -d "rl_experience/9x9_oneplane_book_2layer_ac" ] && \
        mv "rl_experience/9x9_oneplane_book_2layer_ac" \
           "rl_experience/9x9_oneplane_book_2layer_ac_seed${seed}"
done

# ─── E2: oneplane + small_nobn extractor probe (1 seed, 5 rounds) ───
echo "==================== AC: oneplane+small_nobn extractor probe ===================="
echo "--- ac oneplane+small_nobn seed 1 ---"
rm -rf "rl_agents/9x9_oneplane_small_nobn_ac" \
       "rl_experience/9x9_oneplane_small_nobn_ac"
train_ac oneplane small_nobn 5
[ -d "rl_agents/9x9_oneplane_small_nobn_ac" ] && \
    mv "rl_agents/9x9_oneplane_small_nobn_ac" \
       "rl_agents/9x9_oneplane_small_nobn_ac_seed1"
[ -d "rl_experience/9x9_oneplane_small_nobn_ac" ] && \
    mv "rl_experience/9x9_oneplane_small_nobn_ac" \
       "rl_experience/9x9_oneplane_small_nobn_ac_seed1"

echo "==================== AC sweep done ===================="
