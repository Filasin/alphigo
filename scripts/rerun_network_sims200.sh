#!/bin/bash
# Re-run the network sims=200 cell of mcts_correctness_b19 after the
# original cell crashed mid-benchmark (gnugo GTP subprocess died after
# 5 games). Waits for the current correctness sweep to finish, then
# reruns that one cell and patches the CSV in place.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

OUT_CSV=results/mcts_correctness_b19.csv
AGENT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/r0_value_b19_allchunks.h5
N_GAMES=15
SIMS=200
MODE=network
BOARD_SIZE=19
OPP=gnugo:level=5

mkdir -p logs

echo "[rerun] $(date) — waiting for original sweep to finish..."
# The original sweep writes the rollout sims=200 row last. Poll for it.
while ! grep -q '^rollout,200,' "${OUT_CSV}" 2>/dev/null; do
    sleep 300
done

echo "[rerun] $(date) — sweep finished, rerunning ${MODE} sims=${SIMS}"

raw_log="logs/mcts_correctness_rerun_${MODE}_sims${SIMS}.log"
spec="alphazero:agent_file=${AGENT},num_simulations=${SIMS},temperature=0.0,value_mode=${MODE}"

t0=$(date +%s)
${PY} run.py benchmark \
    --black "${spec}" \
    --white "${OPP}" \
    --board-size "${BOARD_SIZE}" -n "${N_GAMES}" \
    > "${raw_log}" 2>&1
rc=$?
t1=$(date +%s)
wall=$((t1 - t0))
wins=$(grep -oE "wins: +[0-9]+" "${raw_log}" | head -1 | grep -oE "[0-9]+$")
wins=${wins:-0}

if [ "${rc}" -ne 0 ]; then
    echo "[rerun] WARNING: benchmark rc=${rc}; check ${raw_log}"
fi

read pct ci_lo ci_hi < <(${PY} -c "
from alphigo.benchmark import wilson_ci
n=${N_GAMES}; w=${wins}
pct = 100.0 * w / n if n else 0.0
lo, hi = wilson_ci(w, n)
print(f'{pct:.1f} {lo*100:.1f} {hi*100:.1f}')
")

new_row="${MODE},${SIMS},${N_GAMES},${wins},${pct},${ci_lo},${ci_hi},${wall}"
echo "[rerun] new row: ${new_row}"

# Patch the CSV: replace the existing network,200,... row with new_row.
${PY} -c "
import csv, sys
path = '${OUT_CSV}'
with open(path) as f:
    rows = list(csv.reader(f))
header = rows[0]
out = [header]
patched = False
for r in rows[1:]:
    if r[0] == '${MODE}' and r[1] == '${SIMS}':
        out.append('${new_row}'.split(','))
        patched = True
    else:
        out.append(r)
if not patched:
    out.append('${new_row}'.split(','))
with open(path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerows(out)
print(f'patched={patched}')
"
echo "[rerun] $(date) — done"
cat "${OUT_CSV}"
