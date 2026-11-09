#!/bin/bash
# Q-learning on 19x19 — book-faithful two-input formulation (Ch. 11).
#
# Best 9x9 config: oneplane + book_2layer, lr=0.1, eps=0.1 (72-83% vs Random).
# 19x19 uses the same config. The action space grows from 82 to 362, and the
# board is 4x larger, so we expect slower learning per round.
#
# 15 rounds x 100 games (vs 200 on 9x9 — 19x19 games are ~4x longer).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

ROUNDS=${ROUNDS:-15}
GAMES=${GAMES:-100}
EVAL_GAMES=${EVAL_GAMES:-50}
LR=${LR:-0.1}
EPS=${EPS:-0.1}
SEED=${SEED:-1}
ENCODER=${ENCODER:-oneplane}
NETWORK=${NETWORK:-book_2layer}

echo
echo "===== Q-learning 19x19 ====="
echo "  encoder=${ENCODER}  network=${NETWORK}  lr=${LR}  eps=${EPS}"
echo "  rounds=${ROUNDS} x ${GAMES} games, eval=${EVAL_GAMES}"
echo

${PY} -u train_rl.py \
    --board-size 19 \
    --encoder "${ENCODER}" \
    --network "${NETWORK}" \
    --method q \
    --rounds "${ROUNDS}" \
    --games "${GAMES}" \
    --eval-games "${EVAL_GAMES}" \
    --epsilon "${EPS}" \
    --lr "${LR}" \
    --seed "${SEED}"

echo
echo "===== Q 19x19 done ====="
