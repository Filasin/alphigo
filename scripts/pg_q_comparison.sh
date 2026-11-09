#!/bin/bash
# Train missing PG checkpoints (oneplane_small rounds 2-5) so we have
# a budget-matched PG vs Q comparison on the same encoder.
# Other 3 combos already have rounds 0..100 from prior runs.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

echo "=== Extending oneplane+small PG to 5 rounds (resume) ==="
${PY} train_rl.py \
    --board-size 9 \
    --encoder oneplane \
    --network small \
    --method policy_gradient \
    --rounds 5 \
    --games 200 2>&1 | \
    grep -E 'init|ROUND|Promoted|Not promoted|Round [0-9]+ done|Traceback|Error|FAIL|Killed'

echo "=== PG training (oneplane+small) done ==="
