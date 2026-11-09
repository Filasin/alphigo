#!/bin/bash
# Phase B AZ bench: vs Random N=100 per checkpoint across the LR sweep.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}
N_GAMES=${N_GAMES:-100}
MCTS_SIMS=${MCTS_SIMS:-25}
LRS=${LRS:-"1e-5 1e-4 1e-3 1e-2"}

bench_combo() {
    local label="$1" dir="$2"
    printf "%-32s  " "${label}"
    if [ ! -d "${dir}" ]; then printf " (missing)\n"; return; fi
    for r in 000 001 002 003 005 010; do
        ckpt="${dir}/round_${r}.h5"
        if [ ! -f "${ckpt}" ]; then printf "  --   "; continue; fi
        wins=$(${PY} run.py benchmark \
            --black "alphazero:agent_file=${ckpt},num_simulations=${MCTS_SIMS}" \
            --white random \
            --board-size 9 -n "${N_GAMES}" 2>/dev/null \
            | grep -oE "wins: +[0-9]+" | head -1 | grep -oE "[0-9]+")
        if [ -z "${wins}" ]; then printf " err   "
        else pct=$(( 100 * wins / N_GAMES )); printf "%3d%%   " "${pct}"; fi
    done
    printf "\n"
}

printf "Phase B AZ LR sweep (sims=${MCTS_SIMS}) "
for r in 000 001 002 003 005 010; do printf "r%s    " "$r"; done
printf "\n"; printf "%.s-" {1..90}; printf "\n"

for lr in ${LRS}; do
    bench_combo "AZ lr=${lr}" \
        "runs/alphazero_alphazero_alphazero_lr${lr}_b9/checkpoints"
done
