#!/bin/bash
# LR sensitivity sweep on the headline combos:
#   - Q   : oneplane + book_2layer (lr in {1e-4, 1e-3, 1e-2}; lr=0.1 is the canonical run)
#   - PG  : oneplane + small       (lr in {1e-3, 1e-2}; lr=0.1 and 1e-4 already exist)
# Runs are isolated under *_lr<value> directories so the canonical
# results from earlier sweeps are preserved.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

# ---------------- Q lr sensitivity ----------------
Q_LRS=("1e-4" "1e-3" "1e-2")
ENC=oneplane; EXT=book_2layer
canonical="rl_agents/9x9_${ENC}_${EXT}_q"
canonical_exp="rl_experience/9x9_${ENC}_${EXT}_q"
# Preserve the existing lr=0.1 run
if [ -d "${canonical}" ] && [ ! -d "${canonical}_lr0.1" ]; then
    mv "${canonical}" "${canonical}_lr0.1"
fi
if [ -d "${canonical_exp}" ] && [ ! -d "${canonical_exp}_lr0.1" ]; then
    mv "${canonical_exp}" "${canonical_exp}_lr0.1"
fi

for lr in "${Q_LRS[@]}"; do
    echo "================================================================"
    echo "=== Q@lr=${lr}: ${ENC}+${EXT}"
    echo "================================================================"
    rm -rf "${canonical}" "${canonical_exp}"
    ${PY} train_rl.py --board-size 9 --encoder "${ENC}" --network "${EXT}" \
        --method q --rounds 5 --games 200 --epsilon 0.1 --lr "${lr}" 2>&1 | \
        grep -E 'init|ROUND|Round [0-9]+ done|Traceback|Error|FAIL|Killed'
    [ -d "${canonical}" ] && mv "${canonical}" "${canonical}_lr${lr}"
    [ -d "${canonical_exp}" ] && mv "${canonical_exp}" "${canonical_exp}_lr${lr}"
done

# ---------------- PG lr sensitivity ----------------
PG_LRS=("1e-3" "1e-2")
ENC=oneplane; NET=small
canonical="rl_agents/9x9_${ENC}_${NET}"
canonical_exp="rl_experience/9x9_${ENC}_${NET}"
# The current canonical is the lr=0.1 PG run from the previous step
if [ -d "${canonical}" ] && [ ! -d "${canonical}_pg_lr0.1" ]; then
    mv "${canonical}" "${canonical}_pg_lr0.1"
fi
if [ -d "${canonical_exp}" ] && [ ! -d "${canonical_exp}_pg_lr0.1" ]; then
    mv "${canonical_exp}" "${canonical_exp}_pg_lr0.1"
fi

for lr in "${PG_LRS[@]}"; do
    echo "================================================================"
    echo "=== PG@lr=${lr}: ${ENC}+${NET}"
    echo "================================================================"
    rm -rf "${canonical}" "${canonical_exp}"
    ${PY} train_rl.py --board-size 9 --encoder "${ENC}" --network "${NET}" \
        --method policy_gradient --rounds 5 --games 200 --lr "${lr}" 2>&1 | \
        grep -E 'init|ROUND|Round [0-9]+ done|Traceback|Error|FAIL|Killed'
    [ -d "${canonical}" ] && mv "${canonical}" "${canonical}_pg_lr${lr}"
    [ -d "${canonical_exp}" ] && mv "${canonical_exp}" "${canonical_exp}_pg_lr${lr}"
done
echo "================================================================"
echo "=== lr sensitivity sweep done ==="
echo "================================================================"
