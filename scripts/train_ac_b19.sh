#!/bin/bash
# Actor-Critic on 19x19 — shared conv body, policy + value heads (Ch. 12).
#
# Best 9x9 config: oneplane + book_2layer, lr=0.01, eps=0 (~48% vs Random).
# AC was the weakest RL method at 9x9 at our budget. 19x19 baseline.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

ROUNDS=${ROUNDS:-15}
GAMES=${GAMES:-100}
EVAL_GAMES=${EVAL_GAMES:-50}
LR=${LR:-1e-2}
SEED=${SEED:-1}
ENCODER=${ENCODER:-oneplane}
NETWORK=${NETWORK:-book_2layer}

echo
echo "===== Actor-Critic 19x19 ====="
echo "  encoder=${ENCODER}  network=${NETWORK}  lr=${LR}"
echo "  rounds=${ROUNDS} x ${GAMES} games, eval=${EVAL_GAMES}"
echo

${PY} -u train_rl.py \
    --board-size 19 \
    --encoder "${ENCODER}" \
    --network "${NETWORK}" \
    --method ac \
    --rounds "${ROUNDS}" \
    --games "${GAMES}" \
    --eval-games "${EVAL_GAMES}" \
    --lr "${LR}" \
    --seed "${SEED}"

echo
echo "===== AC 19x19 done ====="
