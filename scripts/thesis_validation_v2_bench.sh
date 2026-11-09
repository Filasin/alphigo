#!/bin/bash
# Bench the multi-seed runs for the two new headline candidates.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}
N_GAMES=${N_GAMES:-100}

bench_dir() {
    local label="$1" dir="$2" rounds="$3"
    printf "%-46s  " "${label}"
    if [ ! -d "${dir}" ]; then printf " (missing)\n"; return; fi
    for r in ${rounds}; do
        ckpt="${dir}/round_${r}.h5"
        if [ ! -f "${ckpt}" ]; then printf "  --   "; continue; fi
        wins=$(${PY} run.py benchmark \
            --black q:agent_file="${ckpt}" --white random \
            --board-size 9 -n "${N_GAMES}" 2>/dev/null \
            | grep -oE "wins: +[0-9]+" | head -1 | grep -oE "[0-9]+")
        if [ -z "${wins}" ]; then printf " err   "
        else pct=$(( 100 * wins / N_GAMES )); printf "%3d%%   " "${pct}"; fi
    done
    printf "\n"
}

echo "===== E4: oneplane+large_nobn multi-seed (5 rounds) ====="
printf "%-46s  " "combo (seed)"
for r in 000 001 002 003 004 005; do printf "r%s    " "$r"; done
printf "\n"; printf "%.s-" {1..120}; printf "\n"
ROUNDS_5="000 001 002 003 004 005"
for seed in 1 2 3; do
    bench_dir "oneplane+large_nobn (seed ${seed})" \
              "rl_agents/9x9_oneplane_large_nobn_q_seed${seed}" "${ROUNDS_5}"
done
echo

echo "===== E5: alphazero+book_2layer long-budget multi-seed (30 rounds) ====="
printf "%-46s  " "combo (seed)"
ROUNDS_30="000 005 010 015 020 025 030"
for r in ${ROUNDS_30}; do printf "r%s    " "$r"; done
printf "\n"; printf "%.s-" {1..130}; printf "\n"
for seed in 1 2 3; do
    bench_dir "alphazero+book_2layer 30r (seed ${seed})" \
              "rl_agents/9x9_alphazero_book_2layer_q_long30_seed${seed}" "${ROUNDS_30}"
done
