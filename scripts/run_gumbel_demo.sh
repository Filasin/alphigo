#!/bin/bash
# Gumbel AlphaZero demo run: 10 rounds, 2000 games, 8 workers.
# Stripped-down v12 config for thesis demo (no yardstick/pachi).
set -u
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
PY="${ALPHIGO_PYTHON:-python}"

NVIDIA_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/nvidia"
TRT_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/tensorrt_libs"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${TRT_LIB}:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Config (v12 settings, reduced scale) ────────────────────────────
BOARD=9
BLOCKS=8
FILTERS=64
VALUE_FC=64
SIMS=256
GUMBEL_M=16
GAMES_PER_ROUND=2000
WORKERS=8
BATCH_MCTS=128
TRAIN_EPOCHS=1
TRAIN_BS=512
ROUNDS=20
MAX_WINDOW=10
MIN_WINDOW=5
EVAL_EVERY=1
GNUGO_GAMES=20
BN_MOMENTUM=0.9
USE_ONNX=true
WEIGHT_DECAY=1e-5
VALUE_WEIGHT=1.5
OWNERSHIP_WEIGHT=0.5
SCORE_WEIGHT=0.5
PLAYOUT_CAP_FRAC=0.75
HEADS=4
Q_BLEND=0.0
SCORE_MARGIN_SCALE=10
ENCODER=alphazero_komi
KOMI_LO=5.5
KOMI_HI=9.5

RUN_DIR=runs/gumbel_demo
DATA_DIR=${ALPHIGO_DATA_DIR:-data}/gumbel_demo
RESULTS_CSV=results/gumbel_demo.csv
LOG=logs/gumbel_demo.log

mkdir -p "$RUN_DIR/checkpoints" "$DATA_DIR" logs results

log() { echo "[gumbel-demo] $(date +%F\ %H:%M:%S) $*" | tee -a "$LOG"; }

CKPT_DIR="$RUN_DIR/checkpoints"
ROUND0_AGENT="$CKPT_DIR/round_000_agent.h5"

# ── CSV header ──────────────────────────────────────────────────────
if [ ! -f "$RESULTS_CSV" ]; then
    echo "round,gnugo_wins,gnugo_n,policy_ce,value_acc,self_vacc,positions,gen_secs" > "$RESULTS_CSV"
fi

# ── LR schedule ─────────────────────────────────────────────────────
get_lr() {
    echo "0.010000"  # first 10 rounds always at max LR
}

# ── Helpers ─────────────────────────────────────────────────────────
run_gnugo () {
    local out
    out=$("$PY" -u scripts/quick_eval_mcts.py \
        --agent "$1" --sims "$SIMS" --games "$GNUGO_GAMES" \
        --board-size "$BOARD" --search gumbel --gumbel-m "$GUMBEL_M" \
        --opponent gnugo:level=1 2>&1 | tee -a "$LOG")
    GNUGO_WINS=$(echo "$out" | sed -n 's/^Result: \([0-9][0-9]*\)\/.*/\1/p')
}

run_probe () {
    local out
    out=$("$PY" -u scripts/probe_policy_value.py --agent "$1" 2>&1 | tee -a "$LOG")
    POLICY_CE=$(echo "$out" | grep '^PROBE' | sed 's/.*policy_ce=\([0-9.]*\).*/\1/')
    VALUE_ACC=$(echo "$out" | grep '^PROBE' | sed 's/.*value_acc=\([0-9.]*\).*/\1/')
}

# ── Round 0: create fresh random 4-head model ──────────────────────
if ! grep -q '^0,' "$RESULTS_CSV"; then
    log "round 0: creating fresh random 4-head model (encoder=$ENCODER)"
    "$PY" -u - <<PYEOF
import h5py
from alphigo.data.networks import build_policy_scoredist_ownership_network
from alphigo.encoders.base import get_encoder_by_name
from alphigo import kerasutil
import tensorflow as tf
tf.keras.utils.set_random_seed(42)

encoder = get_encoder_by_name('$ENCODER', ($BOARD, $BOARD))
model = build_policy_scoredist_ownership_network(
    encoder.shape(), num_classes=${BOARD}*${BOARD}+1,
    num_blocks=$BLOCKS, filters=$FILTERS,
    value_fc=$VALUE_FC, bn_momentum=$BN_MOMENTUM)
print(f'[init] fresh 4-head model ({encoder.num_planes} planes): '
      f'{model.count_params():,} params')

with h5py.File("$ROUND0_AGENT", 'w') as h5:
    h5.create_group('encoder')
    h5['encoder'].attrs['name'] = encoder.name()
    h5['encoder'].attrs['board_width'] = $BOARD
    h5['encoder'].attrs['board_height'] = $BOARD
    h5.create_group('model')
    kerasutil.save_model_to_hdf5_group(model, h5['model'])
    h5.create_group('agent')
    h5['agent'].attrs['num_simulations'] = $SIMS
    h5['agent'].attrs['c_puct'] = 1.5
print(f'[init] saved {repr("$ROUND0_AGENT")}')
PYEOF
    rc=$?
    if [ $rc -ne 0 ]; then log "round 0: FAILED to create model rc=$rc"; exit 1; fi

    run_gnugo "$ROUND0_AGENT"
    run_probe "$ROUND0_AGENT"
    echo "0,${GNUGO_WINS},${GNUGO_GAMES},${POLICY_CE},${VALUE_ACC},,," >> "$RESULTS_CSV"
    log "round 0: baseline gnugo=${GNUGO_WINS}/${GNUGO_GAMES} ce=${POLICY_CE} vacc=${VALUE_ACC}"
fi

# ── Main loop ───────────────────────────────────────────────────────
for round in $(seq 1 "$ROUNDS"); do
    R=$(printf '%03d' "$round")
    PREV=$(printf '%03d' "$((round - 1))")
    LATEST_AGENT="$CKPT_DIR/round_${PREV}_agent.h5"
    ROUND_MODEL="$CKPT_DIR/round_${R}.h5"
    ROUND_AGENT="$CKPT_DIR/round_${R}_agent.h5"
    ROUND_DATA="$DATA_DIR/round_${R}"

    if grep -q "^${round}," "$RESULTS_CSV"; then
        log "round $round: SKIP (already in CSV)"
        continue
    fi

    TRAIN_LR=$(get_lr "$round")

    # ── Stage 1: self-play generation ───────────────────────────────
    T_GEN0=$(date +%s)
    if ls "$ROUND_DATA"/selfplay_b${BOARD}_*.h5 1>/dev/null 2>&1; then
        log "round $round stage1: SKIP (chunks exist)"
    else
        mkdir -p "$ROUND_DATA"
        log "round $round stage1: $GAMES_PER_ROUND games (gumbel n=$SIMS m=$GUMBEL_M pcr=$PLAYOUT_CAP_FRAC komi=${KOMI_LO}-${KOMI_HI}, agent=round_${PREV})"
        ONNX_FLAG=""
        [ "$USE_ONNX" = "true" ] && ONNX_FLAG="--onnx"
        ORT_INTRA_OP_NUM_THREADS=2 \
        ORT_INTER_OP_NUM_THREADS=1 \
        TF_NUM_INTRAOP_THREADS=1 \
        PLAYOUT_CAP_FRAC="$PLAYOUT_CAP_FRAC" \
        SIMS_LOW=32 \
        GPU_BATCH_WAIT_MS=0 \
        WORKER_GAMES=1 \
        "$PY" -u scripts/generate_selfplay_data_mp.py \
            --agent "$LATEST_AGENT" \
            --num-games "$GAMES_PER_ROUND" \
            --num-workers "$WORKERS" \
            --sims "$SIMS" \
            --search gumbel --gumbel-m "$GUMBEL_M" \
            --value-target margin --score-margin-scale "$SCORE_MARGIN_SCALE" \
            --komi-random "$KOMI_LO" "$KOMI_HI" \
            --board-size "$BOARD" \
            --batch-mcts "$BATCH_MCTS" \
            --chunk-size 250 \
            --seed "$((10000 + round))" \
            $ONNX_FLAG \
            --out-dir "$ROUND_DATA" >> "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then log "round $round stage1: FAILED rc=$rc"; exit 1; fi
    fi
    T_GEN=$(( $(date +%s) - T_GEN0 ))

    # ── Stage 2: training ───────────────────────────────────────────
    if [ -f "$ROUND_AGENT" ]; then
        log "round $round stage2: SKIP (checkpoint exists)"
    else
        TRAIN_GLOBS=()
        WINDOW_SIZE=$(( MIN_WINDOW + round / 2 ))
        [ "$WINDOW_SIZE" -gt "$MAX_WINDOW" ] && WINDOW_SIZE=$MAX_WINDOW
        [ "$WINDOW_SIZE" -gt "$round" ] && WINDOW_SIZE=$round
        FIRST=$(( round - WINDOW_SIZE + 1 ))
        for r in $(seq "$FIRST" "$round"); do
            TRAIN_GLOBS+=("$DATA_DIR/round_$(printf '%03d' "$r")/selfplay_b${BOARD}_*.h5")
        done

        log "round $round stage2: train (init=round_${PREV}, window=${WINDOW_SIZE}, lr=$TRAIN_LR, heads=$HEADS)"
        "$PY" -u scripts/train_distill_b19.py \
            --init-from "$LATEST_AGENT" --full-init \
            --train-glob "${TRAIN_GLOBS[@]}" \
            --dst "$ROUND_MODEL" \
            --num-blocks "$BLOCKS" --filters "$FILTERS" \
            --value-fc "$VALUE_FC" \
            --board-size "$BOARD" --heads "$HEADS" --augment \
            --encoder "$ENCODER" \
            --epochs "$TRAIN_EPOCHS" --bs "$TRAIN_BS" --lr "$TRAIN_LR" \
            --weight-decay "$WEIGHT_DECAY" \
            --bn-momentum "$BN_MOMENTUM" \
            --patience 99 --mega-chunks 8 \
            --policy-weight 1.0 --value-weight "$VALUE_WEIGHT" \
            --ownership-weight "$OWNERSHIP_WEIGHT" \
            --score-weight "$SCORE_WEIGHT" \
            --q-blend "$Q_BLEND" \
            --one-cycle \
            --seed "$((1 + round))" >> "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then log "round $round stage2: FAILED rc=$rc"; exit 1; fi
        if [ ! -f "$ROUND_AGENT" ]; then
            log "round $round stage2: FAILED (no agent checkpoint produced)"
            exit 1
        fi
    fi

    # ── Stage 3: evaluation (gnugo + probe every round) ─────────────
    POSITIONS=$("$PY" - <<EOF
import glob, h5py
n = 0
for p in glob.glob("$ROUND_DATA/selfplay_b${BOARD}_*.h5"):
    with h5py.File(p) as f:
        n += int(f.attrs.get('num_positions', 0))
print(n)
EOF
)

    run_gnugo "$ROUND_AGENT"

    out=$("$PY" -u scripts/probe_policy_value.py --agent "$ROUND_AGENT" 2>&1 | tee -a "$LOG")
    _POLICY_CE=$(echo "$out" | grep '^PROBE' | sed 's/.*policy_ce=\([0-9.]*\).*/\1/')
    _VALUE_ACC=$(echo "$out" | grep '^PROBE' | sed 's/.*value_acc=\([0-9.]*\).*/\1/')

    _SELF_VACC=""
    SELF_CHUNK=$(ls "$ROUND_DATA"/selfplay_b${BOARD}_*.h5 2>/dev/null | tail -1)
    if [ -n "$SELF_CHUNK" ]; then
        out=$("$PY" -u scripts/probe_policy_value.py --agent "$ROUND_AGENT" \
            --probe-chunk "$SELF_CHUNK" 2>&1 | tee -a "$LOG")
        _SELF_VACC=$(echo "$out" | grep '^PROBE' | sed 's/.*value_acc=\([0-9.]*\).*/\1/')
    fi

    log "round $round: COMPLETE (gnugo=${GNUGO_WINS}/${GNUGO_GAMES} ce=${_POLICY_CE} vacc=${_VALUE_ACC} svacc=${_SELF_VACC} gen=${T_GEN}s)"
    echo "${round},${GNUGO_WINS},${GNUGO_GAMES},${_POLICY_CE},${_VALUE_ACC},${_SELF_VACC},${POSITIONS},${T_GEN}" >> "$RESULTS_CSV"
done

log "ALL ROUNDS DONE"
cat "$RESULTS_CSV"
echo ""
echo "Best agent: $CKPT_DIR/round_$(printf '%03d' $ROUNDS)_agent.h5"
echo "Play against it:"
echo "  python run.py play --black human --white gumbel:agent_file=$CKPT_DIR/round_$(printf '%03d' $ROUNDS)_agent.h5,num_simulations=256 --board-size 9"
