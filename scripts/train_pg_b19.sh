#!/bin/bash
# Policy gradient on 19x19 — REINFORCE (Ch. 9).
#
# Best 9x9 config: oneplane + small, lr=1e-4 (plateau 51-58% vs Random).
# PG is generally weaker than Q at this budget; 19x19 will be harder.
# Using oneplane + small as the minimal viable config to establish the baseline.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

ROUNDS=${ROUNDS:-15}
GAMES=${GAMES:-100}
EVAL_GAMES=${EVAL_GAMES:-50}
LR=${LR:-1e-4}
SEED=${SEED:-1}
ENCODER=${ENCODER:-oneplane}
NETWORK=${NETWORK:-small}

echo
echo "===== Policy Gradient 19x19 ====="
echo "  encoder=${ENCODER}  network=${NETWORK}  lr=${LR}"
echo "  rounds=${ROUNDS} x ${GAMES} games, eval=${EVAL_GAMES}"
echo

${PY} -u train_rl.py \
    --board-size 19 \
    --encoder "${ENCODER}" \
    --network "${NETWORK}" \
    --method pg \
    --rounds "${ROUNDS}" \
    --games "${GAMES}" \
    --eval-games "${EVAL_GAMES}" \
    --lr "${LR}" \
    --seed "${SEED}"

echo
echo "===== PG 19x19 done ====="
