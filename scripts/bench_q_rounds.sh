#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
for r in 000 001 002 003 004 005; do
    echo "=== round_${r} vs random ==="
    ${ALPHIGO_PYTHON:-python} run.py benchmark \
        --black q:agent_file=rl_agents/9x9_oneplane_small_q/round_${r}.h5 \
        --white random --board-size 9 -n 30 2>/dev/null | grep -E 'wins|Elo'
done
