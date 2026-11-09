#!/bin/bash
# AC hyperparameter sensitivity (single seed, 5 rounds × 200 games).
#
# Probes three axes:
#   - lr:           {1e-3, 1e-2 (book), 5e-2, 1e-1}
#   - epsilon:      {0.0 (book / softmax-stochastic), 0.1 (matches Q regime)}
#   - value loss weight: {0.25, 0.5 (book), 1.0}
#
# 4 + 1 + 2 = 7 extra runs.  Each ~3 min on RTX 3090 → ~25 min total.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

train_ac() {
    local enc="$1" ext="$2" rounds="$3" lr="$4" eps="$5" tag="$6"
    ${PY} train_rl.py \
        --board-size 9 \
        --encoder "${enc}" \
        --network "${ext}" \
        --method actor_critic \
        --rounds "${rounds}" \
        --games 200 \
        --epsilon "${eps}" \
        --lr "${lr}" \
        --bs 512 2>&1 | \
        grep -E 'init|ROUND|Round [0-9]+ done|vs ref|Traceback|Error|FAIL|Killed'
    src="rl_agents/9x9_${enc}_${ext}_ac"
    src_exp="rl_experience/9x9_${enc}_${ext}_ac"
    [ -d "${src}" ] && mv "${src}" "${src}_${tag}"
    [ -d "${src_exp}" ] && mv "${src_exp}" "${src_exp}_${tag}"
}

ENC="oneplane"
EXT="book_2layer"

# ─── LR sensitivity (eps=0, default loss weights) ───
echo "==================== AC lr sensitivity ===================="
for lr in 1e-3 5e-2 1e-1; do
    tag="lr${lr}"
    echo "--- ac ${ENC}+${EXT} ${tag} ---"
    rm -rf "rl_agents/9x9_${ENC}_${EXT}_ac" "rl_experience/9x9_${ENC}_${EXT}_ac"
    train_ac "${ENC}" "${EXT}" 5 "${lr}" 0.0 "${tag}"
done
# lr=1e-2 already covered by seed1 of thesis_ac_sweep.sh

# ─── Epsilon sensitivity (lr=0.01) ───
echo "==================== AC epsilon sensitivity ===================="
echo "--- ac ${ENC}+${EXT} eps0.1 ---"
rm -rf "rl_agents/9x9_${ENC}_${EXT}_ac" "rl_experience/9x9_${ENC}_${EXT}_ac"
train_ac "${ENC}" "${EXT}" 5 0.01 0.1 "eps0.1"
# eps=0.0 covered by seed1.

# ─── Loss-weight sensitivity (lr=0.01, eps=0) ───
# Requires --value-loss-weight CLI arg in train_rl.py — handled below
# via a small inline Python wrapper that bypasses the CLI default.
echo "==================== AC loss-weight sensitivity ===================="
for vw in 0.25 1.0; do
    tag="vw${vw}"
    echo "--- ac ${ENC}+${EXT} ${tag} ---"
    rm -rf "rl_agents/9x9_${ENC}_${EXT}_ac" "rl_experience/9x9_${ENC}_${EXT}_ac"
    AC_VALUE_LOSS_WEIGHT="${vw}" ${PY} train_rl.py \
        --board-size 9 --encoder "${ENC}" --network "${EXT}" \
        --method actor_critic --rounds 5 --games 200 \
        --epsilon 0.0 --lr 0.01 --bs 512 2>&1 | \
        grep -E 'init|ROUND|Round [0-9]+ done|vs ref|Traceback|Error|FAIL|Killed'
    src="rl_agents/9x9_${ENC}_${EXT}_ac"
    src_exp="rl_experience/9x9_${ENC}_${EXT}_ac"
    [ -d "${src}" ] && mv "${src}" "${src}_${tag}"
    [ -d "${src_exp}" ] && mv "${src_exp}" "${src_exp}_${tag}"
done

echo "==================== AC sensitivity done ===================="
