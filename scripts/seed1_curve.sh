#!/bin/bash
# Bench Q seed 1's full Phase A learning curve vs Random N=100.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}
DIR="rl_agents/9x9_oneplane_book_2layer_q_big50x1000_seed1"

echo "===== Q seed 1 (50r×1000g) learning curve vs Random N=100 ====="
printf "%-6s %s\n" "round" "win%"
for r in 000 005 010 015 020 025 030 035 040 045 050; do
    ckpt="${DIR}/round_${r}.h5"
    [ -f "${ckpt}" ] || { printf "r%-5s --\n" "${r}"; continue; }
    wins=$(${PY} run.py benchmark --black q:agent_file="${ckpt}" --white random \
        --board-size 9 -n 100 2>/dev/null | grep -oE "wins: +[0-9]+" | head -1 | grep -oE "[0-9]+")
    [ -z "${wins}" ] && wins="err"
    printf "r%-5s %s%%\n" "${r}" "${wins}"
done
