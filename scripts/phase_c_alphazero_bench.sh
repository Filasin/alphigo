#!/bin/bash
# Phase C AZ bench: vs Random N=100 across the num_simulations sweep.
# Reads stashed run dirs (runs/_phase_c_stash_sims*) plus the active one.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}
N_GAMES=${N_GAMES:-100}
BEST_LR=${BEST_LR:-1e-4}
SIMS_LIST=${SIMS_LIST:-"10 25 50 100"}

bench_one() {
    local label="$1" dir="$2" sims="$3"
    printf "%-32s  " "${label}"
    if [ ! -d "${dir}" ]; then printf " (missing)\n"; return; fi
    for r in 000 001 002 003 005 010; do
        ckpt="${dir}/round_${r}.h5"
        if [ ! -f "${ckpt}" ]; then printf "  --   "; continue; fi
        wins=$(${PY} run.py benchmark \
            --black "alphazero:agent_file=${ckpt},num_simulations=${sims}" \
            --white random \
            --board-size 9 -n "${N_GAMES}" 2>/dev/null \
            | grep -oE "wins: +[0-9]+" | head -1 | grep -oE "[0-9]+")
        if [ -z "${wins}" ]; then printf " err   "
        else pct=$(( 100 * wins / N_GAMES )); printf "%3d%%   " "${pct}"; fi
    done
    printf "\n"
}

printf "Phase C AZ sims sweep (lr=${BEST_LR}) "
for r in 000 001 002 003 005 010; do printf "r%s    " "$r"; done
printf "\n"; printf "%.s-" {1..90}; printf "\n"

for sims in ${SIMS_LIST}; do
    home="runs/alphazero_alphazero_alphazero_lr${BEST_LR}_b9/checkpoints"
    stash="runs/_phase_c_stash_sims${sims}/checkpoints"

    # Use whichever dir currently corresponds to this sims value.
    dir=""
    if [ -d "${home}" ] && [ -f "runs/alphazero_alphazero_alphazero_lr${BEST_LR}_b9/_sims" ] && \
       [ "$(cat runs/alphazero_alphazero_alphazero_lr${BEST_LR}_b9/_sims)" = "${sims}" ]; then
        dir="${home}"
    elif [ -d "${stash}" ]; then
        dir="${stash}"
    fi
    bench_one "AZ sims=${sims}" "${dir}" "${sims}"
done
