#!/bin/bash
# Bench AC sweep checkpoints vs Random at N=100 (default).
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
            --black ac:agent_file="${ckpt}" --white random \
            --board-size 9 -n "${N_GAMES}" 2>/dev/null \
            | grep -oE "wins: +[0-9]+" | head -1 | grep -oE "[0-9]+")
        if [ -z "${wins}" ]; then printf " err   "
        else pct=$(( 100 * wins / N_GAMES )); printf "%3d%%   " "${pct}"; fi
    done
    printf "\n"
}

ROUNDS="000 001 002 003 004 005"

echo "===== AC: oneplane+book_2layer multi-seed (5 rounds) vs Random ====="
printf "%-46s  " "combo (seed)"
for r in ${ROUNDS}; do printf "r%s    " "$r"; done
printf "\n"; printf "%.s-" {1..120}; printf "\n"
for seed in 1 2 3; do
    bench_dir "ac oneplane+book_2layer (seed ${seed})" \
              "rl_agents/9x9_oneplane_book_2layer_ac_seed${seed}" "${ROUNDS}"
done
echo

echo "===== AC: oneplane+small_nobn extractor probe vs Random ====="
printf "%-46s  " "combo (seed)"
for r in ${ROUNDS}; do printf "r%s    " "$r"; done
printf "\n"; printf "%.s-" {1..120}; printf "\n"
bench_dir "ac oneplane+small_nobn (seed 1)" \
          "rl_agents/9x9_oneplane_small_nobn_ac_seed1" "${ROUNDS}"
