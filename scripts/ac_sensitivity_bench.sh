#!/bin/bash
# Bench AC sensitivity sweep checkpoints vs Random at N=100.
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
BASE="rl_agents/9x9_oneplane_book_2layer_ac"

echo "===== AC sensitivity (oneplane+book_2layer) vs Random N=${N_GAMES} ====="
printf "%-46s  " "config"
for r in ${ROUNDS}; do printf "r%s    " "$r"; done
printf "\n"; printf "%.s-" {1..120}; printf "\n"

# lr=1e-2 baseline (already trained as seed1)
bench_dir "lr=1e-2 eps=0.0 vw=0.5 (book; seed1)" \
          "${BASE}_seed1" "${ROUNDS}"

# lr sensitivity
bench_dir "lr=1e-3 eps=0.0 vw=0.5"  "${BASE}_lr1e-3" "${ROUNDS}"
bench_dir "lr=5e-2 eps=0.0 vw=0.5"  "${BASE}_lr5e-2" "${ROUNDS}"
bench_dir "lr=1e-1 eps=0.0 vw=0.5"  "${BASE}_lr1e-1" "${ROUNDS}"

# eps sensitivity
bench_dir "lr=1e-2 eps=0.1 vw=0.5"  "${BASE}_eps0.1" "${ROUNDS}"

# value-loss-weight sensitivity
bench_dir "lr=1e-2 eps=0.0 vw=0.25" "${BASE}_vw0.25" "${ROUNDS}"
bench_dir "lr=1e-2 eps=0.0 vw=1.0"  "${BASE}_vw1.0"  "${ROUNDS}"
