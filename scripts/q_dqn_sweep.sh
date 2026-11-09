#!/bin/bash
# Q-DQN sweep — train_rl.py only supports network='small' for q-dqn,
# so vary the encoder across 3 options. Same budget: 5 rounds x 200
# games, lr=0.1, eps=0.1.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

ENCODERS=(oneplane sevenplane alphazero)

for enc in "${ENCODERS[@]}"; do
    echo "================================================================"
    echo "=== Q-DQN SWEEP: encoder=${enc}  network=small"
    echo "================================================================"

    rm -rf "rl_agents/9x9_${enc}_small_q_dqn"
    rm -rf "rl_experience/9x9_${enc}_small_q_dqn"

    ${PY} train_rl.py \
        --board-size 9 \
        --encoder "${enc}" \
        --network small \
        --method q-dqn \
        --rounds 5 \
        --games 200 \
        --epsilon 0.1 \
        --lr 0.1 2>&1 | \
        grep -E 'init|ROUND|Promoted|Not promoted|Round [0-9]+ done|Traceback|Error|FAIL|Killed'
done
echo "================================================================"
echo "=== q-dqn sweep done ==="
echo "================================================================"
