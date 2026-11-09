#!/bin/bash
# Queue: run Q, PG, AC on 19x19 sequentially.
# Designed to run after the AZ confirmation experiment completes.
# Each method writes its run to runs/{method}_*/
# After all three finish, evaluates each best checkpoint vs gnugo:level=1 and level=5.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

mkdir -p logs

echo "================================================================"
echo "=== RL 19x19 queue: Q → PG → AC"
echo "=== $(date)"
echo "================================================================"

# Q-learning
echo; echo "--- Q-learning 19x19 ---"
bash scripts/train_q_b19.sh 2>&1 | tee logs/train_q_b19.log
echo "Q done: $(date)"

# Policy Gradient
echo; echo "--- Policy Gradient 19x19 ---"
bash scripts/train_pg_b19.sh 2>&1 | tee logs/train_pg_b19.log
echo "PG done: $(date)"

# Actor-Critic
echo; echo "--- Actor-Critic 19x19 ---"
bash scripts/train_ac_b19.sh 2>&1 | tee logs/train_ac_b19.log
echo "AC done: $(date)"

echo
echo "================================================================"
echo "=== All three done. Evaluating vs GnuGo ==="
echo "================================================================"

# Find best checkpoint for each method (last promoted round or final round).
for method in q pg ac; do
    best=$(ls runs/${method}_*_b19_seed1/checkpoints/round_*.h5 2>/dev/null | sort | tail -1)
    if [ -z "${best}" ]; then
        echo "WARNING: no checkpoint found for ${method} 19x19 seed1"
        continue
    fi
    echo
    echo "--- ${method} best: ${best} ---"
    echo "vs gnugo:level=1 (N=50):"
    ${PY} run.py benchmark \
        --black "dl:agent_file=${best}" \
        --white "gnugo:level=1" \
        --board-size 19 -n 25 2>/dev/null | tail -5
    echo "vs gnugo:level=5 (N=50):"
    ${PY} run.py benchmark \
        --black "dl:agent_file=${best}" \
        --white "gnugo:level=5" \
        --board-size 19 -n 25 2>/dev/null | tail -5
done

echo
echo "================================================================"
echo "=== RL 19x19 queue complete: $(date)"
echo "================================================================"
