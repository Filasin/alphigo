#!/bin/bash
# Sequential GnuGo L1 benchmarks for the lr=1e-5 epsilon sweep.
set -e
source ~/alphigo-venv/bin/activate
cd "$(dirname "$0")/.."

for run in eps01 eps10; do
    for round in round_001 round_050; do
        ckpt="rl_agents/9x9_agplane_large_lr1e5_${run}/${round}.h5"
        log="logs/bench_lr1e5_${run}_${round}.log"
        echo "=== $(date) BENCH ${run} ${round} ==="
        python run.py benchmark \
            --black "pg:agent_file=${ckpt}" \
            --white gnugo:level=1 \
            -n 50 \
            --board-size 9 \
            > "${log}" 2>&1
        echo "=== $(date) DONE ${run} ${round} ==="
    done
done

echo "=== $(date) ALL BENCHMARKS DONE ==="
