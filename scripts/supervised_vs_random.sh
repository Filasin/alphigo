#!/bin/bash
# Bench the 9x9 supervised baseline (mcts_oneplane_small_b9) vs Random N=100.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}
N_GAMES=${N_GAMES:-100}

echo "===== Supervised mcts_oneplane_small_b9 vs Random (N=${N_GAMES}) ====="
printf "%-50s  %s\n" "checkpoint" "win%"
printf "%.s-" {1..70}; printf "\n"

for ep in 1 2 3 4 5 6; do
    ckpt="checkpoints/mcts_oneplane_small_adam_b9_model_epoch_${ep}.h5"
    printf "%-50s  " "epoch ${ep}"
    if [ ! -f "${ckpt}" ]; then printf "(missing)\n"; continue; fi
    wins=$(${PY} run.py benchmark \
        --black "dl:agent_file=${ckpt}" --white random \
        --board-size 9 -n "${N_GAMES}" 2>/dev/null \
        | grep -oE "wins: +[0-9]+" | head -1 | grep -oE "[0-9]+")
    if [ -z "${wins}" ]; then printf "err\n"
    else pct=$(( 100 * wins / N_GAMES )); printf "%3d%%  (%s wins)\n" "${pct}" "${wins}"; fi
done
