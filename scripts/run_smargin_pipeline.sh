#!/bin/bash
# Score-margin + joint-training + sims-sweep pipeline (option A+B).
#
# Step 1: re-encode the 19x19 pro corpus with score-margin value labels
#         and one-hot policy targets, into a new buffer set
#         data/sgf_value_buffer_b19_v2_*.h5.
# Step 2: joint policy+value training (body unfrozen) streaming all
#         ~2M positions from disk. Holds out chunk 0009 for val.
# Step 3: sims sweep with the new checkpoint vs gnugo:level=5 at
#         sims=1,25,50,200 (N=30 per cell).
#
# Total wall: ~36-48h on RTX 3090 (re-encode ~2h, train ~16h, sweep ~6-12h).
#
# Each step skips itself if its primary output already exists, so a
# crashed/interrupted run can be resumed by re-invoking the script.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
PY=${ALPHIGO_PYTHON:-python}

# Step knobs
SGF_DIR=${SGF_DIR:-data}
SMARGIN_SCALE=${SMARGIN_SCALE:-20}
CHUNK_GAMES=${CHUNK_GAMES:-10000}
NEW_BUFFER_BASE=data/sgf_value_buffer_b19_v2
NEW_AGENT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/r0_joint_smargin_b19.h5
SRC_CKPT=runs/alphazero_alphazero_alphazero_lr1e-4_b19_seed1/checkpoints/round_000.h5
SWEEP_CSV=results/sims_sweep_b19_smargin.csv
SWEEP_N=${SWEEP_N:-30}
SWEEP_SIMS=${SWEEP_SIMS:-"1 25 50 200"}

EPOCHS=${EPOCHS:-8}
BS=${BS:-256}
LR=${LR:-5e-4}
ALPHA=${ALPHA:-0.5}

mkdir -p logs

echo
echo "===== run_smargin_pipeline.sh @ $(date) ====="
echo "  SGF src:           ${SGF_DIR}"
echo "  smargin scale:     ${SMARGIN_SCALE}"
echo "  new buffer base:   ${NEW_BUFFER_BASE}_*.h5"
echo "  src checkpoint:    ${SRC_CKPT}"
echo "  dst checkpoint:    ${NEW_AGENT}"
echo "  sweep CSV:         ${SWEEP_CSV}"
echo "  train: epochs=${EPOCHS} bs=${BS} lr=${LR} alpha=${ALPHA}"
echo

# ─── Step 1: re-encode ─────────────────────────────────────────────────
if ls ${NEW_BUFFER_BASE}_*.h5 >/dev/null 2>&1; then
    echo "[step1] SKIP: ${NEW_BUFFER_BASE}_*.h5 already exists"
else
    echo "[step1] re-encoding SGFs with score-margin labels..."
    ${PY} -u scripts/encode_sgf_value_data.py \
        --src-dir "${SGF_DIR}" \
        --out "${NEW_BUFFER_BASE}.h5" \
        --board-size 19 \
        --encoder alphazero \
        --chunk-games "${CHUNK_GAMES}" \
        --score-margin-scale "${SMARGIN_SCALE}" \
        --policy-target one-hot \
        2>&1 | tee -a logs/smargin_pipeline_step1.log
    rc=$?
    if [ "${rc}" -ne 0 ]; then
        echo "[step1] ERROR rc=${rc}"; exit 1
    fi
fi

# Sanity: confirm at least one chunk exists.
if ! ls ${NEW_BUFFER_BASE}_*.h5 >/dev/null 2>&1; then
    echo "[step1] ERROR: no chunks produced"; exit 1
fi
echo "[step1] chunks:"
ls -la ${NEW_BUFFER_BASE}_*.h5

# ─── Step 2: joint training ────────────────────────────────────────────
if [ -f "${NEW_AGENT}" ]; then
    echo "[step2] SKIP: ${NEW_AGENT} already exists"
else
    echo "[step2] joint training (streaming, body unfrozen)..."
    # Use the last chunk as val.
    LAST_CHUNK=$(ls ${NEW_BUFFER_BASE}_*.h5 | tail -1)
    ${PY} -u scripts/train_value_joint_b19.py \
        --src "${SRC_CKPT}" \
        --train-glob "${NEW_BUFFER_BASE}_*.h5" \
        --val-chunk "${LAST_CHUNK}" \
        --dst "${NEW_AGENT}" \
        --epochs "${EPOCHS}" --bs "${BS}" --lr "${LR}" --alpha "${ALPHA}" \
        2>&1 | tee -a logs/smargin_pipeline_step2.log
    rc=$?
    if [ "${rc}" -ne 0 ]; then
        echo "[step2] ERROR rc=${rc}"; exit 1
    fi
fi

# ─── Step 3: sims sweep ────────────────────────────────────────────────
if [ -f "${SWEEP_CSV}" ] && \
        [ "$(wc -l < ${SWEEP_CSV})" -gt 1 ]; then
    echo "[step3] SKIP: ${SWEEP_CSV} already has results"
else
    echo "[step3] sims sweep vs gnugo:level=5..."
    AGENT="${NEW_AGENT}" \
    OPP=gnugo:level=5 \
    N_GAMES="${SWEEP_N}" \
    SIMS_LIST="${SWEEP_SIMS}" \
    BOARD_SIZE=19 \
    OUT_CSV="${SWEEP_CSV}" \
        bash scripts/sims_sweep.sh 2>&1 | tee -a logs/smargin_pipeline_step3.log
    rc=$?
    if [ "${rc}" -ne 0 ]; then
        echo "[step3] WARN rc=${rc} — partial results may still be useful"
    fi
fi

echo
echo "===== run_smargin_pipeline.sh DONE @ $(date) ====="
echo "  new V checkpoint: ${NEW_AGENT}"
echo "  new sims CSV:     ${SWEEP_CSV}"
echo
echo "Compare to current curve (88/66/34/30/32 @ sims=1/25/50/200/800):"
cat "${SWEEP_CSV}" 2>/dev/null
