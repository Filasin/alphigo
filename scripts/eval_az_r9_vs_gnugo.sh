#!/bin/bash
# Cell 1 anchor eval: round_009 (final ref) vs GnuGo level=1 and level=10.
#
# Goal: convert the relative vs_ref Elo curve from cell 1 into an absolute
# strength estimate against a published baseline. 30 games per opponent
# (Wilson CI ~±17pp at 50%), color-swapped via run.py benchmark.
#
# GnuGo Elo anchors (alphigo/training/baseline.py REFERENCE_ELOS):
#   level=1  → ~600  EGF
#   level=10 → ~1200 EGF
#
# Time budget (estimate):
#   vs level=1: 30 games × ~11s/game = ~6 min (measured from 2-game smoke)
#
# Level=10 (~1200 EGF) skipped on this pass — the smoke test (0/2 vs
# level=1) suggests AZ r9 is well below 600 EGF, so level=10 would be
# a near-shutout with little additional information. Re-enable by
# adding "gnugo:level=10" to OPPONENTS below.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

CKPT=${CKPT:-runs/alphazero_alphazero_alphazero_lr1e-4_b9_seed1_sims25/checkpoints/round_009.h5}
N_GAMES=${N_GAMES:-30}
SIMS=${SIMS:-25}
OUT=${OUT:-results/az_r9_vs_gnugo.json}

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

# `alphazero:agent_file=...,num_simulations=...,temperature=0.0` — registry spec.
# Temperature 0 = deterministic argmax on visit counts (eval, not self-play).
AZ_SPEC="alphazero:agent_file=${CKPT},num_simulations=${SIMS},temperature=0.0"

mkdir -p "$(dirname "${OUT}")"
echo "{\"checkpoint\": \"${CKPT}\", \"num_simulations\": ${SIMS}, \"num_games\": ${N_GAMES}, \"results\": {" > "${OUT}"
first=1

OPPONENTS=(${OPPONENTS:-"gnugo:level=1"})
for opp in "${OPPONENTS[@]}"; do
    echo "================================================================"
    echo "  AZ round_009 (sims=${SIMS}) vs ${opp}, ${N_GAMES} games"
    echo "================================================================"
    t0=$(date +%s)
    # run.py benchmark prints "  wins:  XX" and similar; capture full stdout.
    log=$(${PY} run.py benchmark \
        --black "${AZ_SPEC}" --white "${opp}" \
        --board-size 9 -n "${N_GAMES}" 2>&1)
    echo "${log}" | tail -25
    t1=$(date +%s)
    wins=$(echo "${log}" | grep -oE "wins:[[:space:]]+[0-9]+" | head -1 | grep -oE "[0-9]+")
    if [ -z "${wins}" ]; then
        wins=null
    fi
    pct=$(echo "${log}" | grep -oE "a_win_pct[\"']?:[[:space:]]+[0-9.]+" | head -1 | grep -oE "[0-9.]+$")
    if [ -z "${pct}" ]; then
        pct=null
    fi
    if [ ${first} -eq 0 ]; then echo "  ," >> "${OUT}"; fi
    first=0
    echo "  \"${opp}\": {\"wins_raw\": ${wins}, \"win_pct\": ${pct}, \"wall_seconds\": $((t1 - t0))}" >> "${OUT}"
    echo "[done] ${opp}: wall=$((t1 - t0))s"
done

echo "}}" >> "${OUT}"
echo
echo "##################### eval done #####################"
echo "  raw results saved to ${OUT}"
echo "  inspect with: cat ${OUT} | python3 -m json.tool"
