#!/bin/bash
# Gumbel AlphaZero self-play loop on 9x9 — v12 (score distribution head).
# Fresh random init with 8b64f (~624K params). Hypothesis: v9/v10 plateau was
# caused by the IPC bug (fixed r107 of v11), not insufficient capacity.
# The 10b128f v11 net (2.5M) already converged to the same ceiling, reinforcing
# the hypothesis. We test whether 8b64f with a clean protocol + score
# distribution head can match v11's ceiling and converge faster.
#
# Changes vs v11:
#   - Network: 8 blocks x 64 filters, value_fc=64 (~624K params)
#   - HEADS=4: policy + value + score_distribution + ownership
#   - score_distribution: 61 bins (margin -30 to +30), CE loss, weight 0.5
#   - All v11 IPC fixes present from round 0 (no flag-2 bug period)
#   - Same encoder, komi range, 1cycle LR, PCR, eval cadence as v11
#
# Usage:
#   bash scripts/run_gumbel_loop_b9_v12.sh [--rounds N] [--games N]
set -u
PROJECT_DIR="${ALPHIGO_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
PY="${ALPHIGO_PYTHON:-python}"

NVIDIA_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/nvidia"
TRT_LIB="${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages/tensorrt_libs"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${TRT_LIB}:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Config ───────────────────────────────────────────────────────────
BOARD=9
BLOCKS=8
FILTERS=64
VALUE_FC=64
SIMS=256
GUMBEL_M=16
GAMES_PER_ROUND=3000
WORKERS=10
BATCH_MCTS=128
TRAIN_EPOCHS=1
TRAIN_BS=512
ROUNDS=250
MAX_WINDOW=10
MIN_WINDOW=5
EVAL_EVERY=1
H2H_GAMES=20
GNUGO_GAMES=20
# Fixed strong yardstick: frozen v11 r132. h2h vs this anchor gives a
# monotone strength curve against a non-moving opponent (GnuGo saturates,
# rolling-prev h2h is a moving target). Fires on the mod-20 rounds.
YARDSTICK_AGENT=runs/gumbel_loop_b9_v11/checkpoints/round_132_agent.h5
YARDSTICK_GAMES=40
# Pachi external eval: GnuGo L1 saturated (~90%) by r30; pachi@8000
# calibrated at r40 to 12/20 (60%) — the only external with resolution
# left in both directions. Runs on the mod-5 rounds alongside GnuGo.
# r62+: bumped to 16000 — @8000 saturating (r55 17/20, r60 16/20);
# 4x playout step ~ 450-500 Elo keeps the ruler two-sided.
PACHI_OPP="pachi:playouts=16000"
PACHI_GAMES=20
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

RUN_DIR=runs/gumbel_loop_b9_v12
DATA_DIR=${ALPHIGO_DATA_DIR:-data}/gumbel_loop_b9_v12
RESULTS_CSV=results/gumbel_loop_b9_v12.csv
LOG=logs/gumbel_loop_b9_v12.log

# ── Parse args ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds) ROUNDS="$2"; shift 2 ;;
        --games) GAMES_PER_ROUND="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RUN_DIR/checkpoints" "$DATA_DIR" logs results

log() { echo "[gumbel-v12] $(date +%F\ %H:%M:%S) $*" | tee -a "$LOG"; }

CKPT_DIR="$RUN_DIR/checkpoints"
ROUND0_AGENT="$CKPT_DIR/round_000_agent.h5"

# ── CSV header ───────────────────────────────────────────────────────
if [ ! -f "$RESULTS_CSV" ]; then
    echo "round,yard132_wins,yard132_n,gnugo_wins,gnugo_n,policy_ce,value_acc,self_vacc,h2hprev_wins,h2hprev_n,positions,gen_secs,gnugo_ph_wins,pachi_wins" > "$RESULTS_CSV"
fi

# ── LR schedule (step decay of the 1cycle base LR) ──────────────────
get_lr() {
    local r=$1
    local lr_hold="$CKPT_DIR/lr_hold.txt"
    if [ -f "$lr_hold" ]; then
        local hold_round hold_lr
        read -r hold_round hold_lr < "$lr_hold"
        if [ "$r" -ge "$hold_round" ]; then
            echo "$hold_lr"
            return
        fi
    fi
    if [ "$r" -lt 40 ]; then
        echo "0.010000"
    elif [ "$r" -lt 80 ]; then
        echo "0.003000"
    elif [ "$r" -lt 120 ]; then
        echo "0.001000"
    else
        echo "0.000300"
    fi
}

# ── Helpers ──────────────────────────────────────────────────────────
run_gnugo () {
    local out
    out=$("$PY" -u scripts/quick_eval_mcts.py \
        --agent "$1" --sims "$SIMS" --games "$GNUGO_GAMES" \
        --board-size "$BOARD" --search gumbel --gumbel-m "$GUMBEL_M" \
        --opponent gnugo:level=1 2>&1 | tee -a "$LOG")
    GNUGO_WINS=$(echo "$out" | sed -n 's/^Result: \([0-9][0-9]*\)\/.*/\1/p')
}

run_gnugo_passhack () {
    local out
    out=$("$PY" -u scripts/quick_eval_mcts.py \
        --agent "$1" --sims "$SIMS" --games "$GNUGO_GAMES" \
        --board-size "$BOARD" --search gumbel --gumbel-m "$GUMBEL_M" \
        --opponent gnugo:level=1 --pass-hack 2>&1 | tee -a "$LOG")
    GNUGO_PH_WINS=$(echo "$out" | sed -n 's/^Result: \([0-9][0-9]*\)\/.*/\1/p')
}

run_probe () {
    local out
    out=$("$PY" -u scripts/probe_policy_value.py --agent "$1" 2>&1 | tee -a "$LOG")
    POLICY_CE=$(echo "$out" | grep '^PROBE' | sed 's/.*policy_ce=\([0-9.]*\).*/\1/')
    VALUE_ACC=$(echo "$out" | grep '^PROBE' | sed 's/.*value_acc=\([0-9.]*\).*/\1/')
}

run_puzzles () {
    log "round $2 stage3: tactical puzzle test v1 (sims=$SIMS)"
    "$PY" -u scripts/_test_tactics.py --agent "$1" --sims "$SIMS" 2>&1 | tee -a "$LOG"
    # edge-v2 = the tactical-value diagnostic v11 scored 0/5 on; the score-dist
    # head is meant to move this. Folded into the same round%20 cadence.
    log "round $2 stage3: edge puzzle test v2 (sims=$SIMS)"
    "$PY" -u scripts/_test_tactics_edge_v2.py --agent "$1" --sims "$SIMS" 2>&1 | tee -a "$LOG"
}

# ── Round 0: create fresh random 4-head model (komi encoder) ────────
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
    run_gnugo_passhack "$ROUND0_AGENT"
    run_probe "$ROUND0_AGENT"
    echo "0,10,${H2H_GAMES},${GNUGO_WINS},${GNUGO_GAMES},${POLICY_CE},${VALUE_ACC},,,,,,${GNUGO_PH_WINS}" >> "$RESULTS_CSV"
    log "round 0: baseline gnugo=${GNUGO_WINS}/${GNUGO_GAMES} ph=${GNUGO_PH_WINS}/${GNUGO_GAMES} ce=${POLICY_CE} vacc=${VALUE_ACC}"
fi

# ── Main loop ────────────────────────────────────────────────────────
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

    # ── Stage 1: self-play generation (randomized komi) ──────────────
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

    # ── Stage 2: training (4-head, margin+scoredist targets, 1cycle) ─
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

        log "round $round stage2: train (init=round_${PREV}, window=${WINDOW_SIZE}, lr=$TRAIN_LR 1cycle, wd=$WEIGHT_DECAY, bs=$TRAIN_BS, vw=$VALUE_WEIGHT, ow=$OWNERSHIP_WEIGHT, sw=$SCORE_WEIGHT, heads=$HEADS, enc=$ENCODER)"
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

    # ── Stage 3: evaluation (sequential) ────────────────────────────────
    POSITIONS=$("$PY" - <<EOF
import glob, h5py
n = 0
for p in glob.glob("$ROUND_DATA/selfplay_b${BOARD}_*.h5"):
    with h5py.File(p) as f:
        n += int(f.attrs.get('num_positions', 0))
print(n)
EOF
)

    if [ $((round % EVAL_EVERY)) -eq 0 ]; then
        _GNUGO_WINS=""
        _GNUGO_PH_WINS=""
        _POLICY_CE=""
        _VALUE_ACC=""
        _SELF_VACC=""
        _H2HPREV_WINS=""
        _H2HPREV_N=""
        _YARD_WINS=""
        _YARD_N=""
        _PACHI_WINS=""

        _H2H_PID=""
        if [ $((round % 10)) -eq 0 ] && [ "$round" -ge 10 ]; then
            REF=$(printf '%03d' "$((round - 10))")
            log "round $round stage3: rolling h2h vs round_${REF} (parallel)"
            "$PY" -u scripts/head_to_head.py \
                --agent-a "$ROUND_AGENT" --agent-b "$CKPT_DIR/round_${REF}_agent.h5" \
                --num-games "$H2H_GAMES" --board-size "$BOARD" \
                --sims "$SIMS" --search gumbel --gumbel-m "$GUMBEL_M" \
                --seed "$((200 + round))" > "/tmp/v12_h2h_r${round}.log" 2>&1 &
            _H2H_PID=$!
        fi

        if [ $((round % 5)) -eq 0 ]; then
            log "round $round stage3: gnugo evals (2x$GNUGO_GAMES games) + pachi ($PACHI_OPP, $PACHI_GAMES games), parallel"
            "$PY" -u scripts/quick_eval_mcts.py \
                --agent "$ROUND_AGENT" --sims "$SIMS" --games "$PACHI_GAMES" \
                --board-size "$BOARD" --search gumbel --gumbel-m "$GUMBEL_M" \
                --opponent "$PACHI_OPP" > "/tmp/v12_pachi_r${round}.log" 2>&1 &
            _PID_PACHI=$!
            "$PY" -u scripts/quick_eval_mcts.py \
                --agent "$ROUND_AGENT" --sims "$SIMS" --games "$GNUGO_GAMES" \
                --board-size "$BOARD" --search gumbel --gumbel-m "$GUMBEL_M" \
                --opponent gnugo:level=1 > "/tmp/v12_gnugo_plain_r${round}.log" 2>&1 &
            _PID_PLAIN=$!
            "$PY" -u scripts/quick_eval_mcts.py \
                --agent "$ROUND_AGENT" --sims "$SIMS" --games "$GNUGO_GAMES" \
                --board-size "$BOARD" --search gumbel --gumbel-m "$GUMBEL_M" \
                --opponent gnugo:level=1 --pass-hack > "/tmp/v12_gnugo_ph_r${round}.log" 2>&1 &
            _PID_PH=$!
            wait "$_PID_PLAIN" "$_PID_PH" "$_PID_PACHI"
            cat "/tmp/v12_gnugo_plain_r${round}.log" "/tmp/v12_gnugo_ph_r${round}.log" "/tmp/v12_pachi_r${round}.log" >> "$LOG"
            _GNUGO_WINS=$(sed -n 's/^Result: \([0-9][0-9]*\)\/.*/\1/p' "/tmp/v12_gnugo_plain_r${round}.log")
            _GNUGO_PH_WINS=$(sed -n 's/^Result: \([0-9][0-9]*\)\/.*/\1/p' "/tmp/v12_gnugo_ph_r${round}.log")
            _PACHI_WINS=$(sed -n 's/^Result: \([0-9][0-9]*\)\/.*/\1/p' "/tmp/v12_pachi_r${round}.log")
            log "round $round stage3: gnugo ${_GNUGO_WINS}/${GNUGO_GAMES} ph ${_GNUGO_PH_WINS}/${GNUGO_GAMES} pachi ${_PACHI_WINS}/${PACHI_GAMES}"
            rm -f "/tmp/v12_gnugo_plain_r${round}.log" "/tmp/v12_gnugo_ph_r${round}.log" "/tmp/v12_pachi_r${round}.log"
        fi

        out=$("$PY" -u scripts/probe_policy_value.py --agent "$ROUND_AGENT" 2>&1 | tee -a "$LOG")
        _POLICY_CE=$(echo "$out" | grep '^PROBE' | sed 's/.*policy_ce=\([0-9.]*\).*/\1/')
        _VALUE_ACC=$(echo "$out" | grep '^PROBE' | sed 's/.*value_acc=\([0-9.]*\).*/\1/')

        SELF_CHUNK=$(ls "$ROUND_DATA"/selfplay_b${BOARD}_*.h5 2>/dev/null | tail -1)
        if [ -n "$SELF_CHUNK" ]; then
            out=$("$PY" -u scripts/probe_policy_value.py --agent "$ROUND_AGENT" \
                --probe-chunk "$SELF_CHUNK" 2>&1 | tee -a "$LOG")
            _SELF_VACC=$(echo "$out" | grep '^PROBE' | sed 's/.*value_acc=\([0-9.]*\).*/\1/')
        fi

        if [ -n "$_H2H_PID" ]; then
            wait "$_H2H_PID"
            cat "/tmp/v12_h2h_r${round}.log" >> "$LOG"
            _H2HPREV_WINS=$(sed -n 's/.*A wins: \([0-9][0-9]*\).*/\1/p' "/tmp/v12_h2h_r${round}.log")
            _H2HPREV_N=$H2H_GAMES
            rm -f "/tmp/v12_h2h_r${round}.log"
        fi

        if [ $((round % 20)) -eq 0 ]; then
            if [ -f "$YARDSTICK_AGENT" ]; then
                log "round $round stage3: yardstick h2h vs v11 r132 ($YARDSTICK_GAMES games)"
                "$PY" -u scripts/head_to_head.py \
                    --agent-a "$ROUND_AGENT" --agent-b "$YARDSTICK_AGENT" \
                    --num-games "$YARDSTICK_GAMES" --board-size "$BOARD" \
                    --sims "$SIMS" --search gumbel --gumbel-m "$GUMBEL_M" \
                    --seed "$((900 + round))" > "/tmp/v12_yard_r${round}.log" 2>&1
                cat "/tmp/v12_yard_r${round}.log" >> "$LOG"
                _YARD_WINS=$(sed -n 's/.*A wins: \([0-9][0-9]*\).*/\1/p' "/tmp/v12_yard_r${round}.log")
                _YARD_N=$YARDSTICK_GAMES
                log "round $round stage3: yardstick vs r132 = ${_YARD_WINS}/${YARDSTICK_GAMES}"
                rm -f "/tmp/v12_yard_r${round}.log"
            fi
            run_puzzles "$ROUND_AGENT" "$round"
        fi

        log "round $round: COMPLETE (gnugo=${_GNUGO_WINS}/${GNUGO_GAMES} ph=${_GNUGO_PH_WINS}/${GNUGO_GAMES} pachi=${_PACHI_WINS}/${PACHI_GAMES} ce=${_POLICY_CE} vacc=${_VALUE_ACC} svacc=${_SELF_VACC} lr=${TRAIN_LR} wd=${WEIGHT_DECAY})"
        echo "${round},${_YARD_WINS},${_YARD_N},${_GNUGO_WINS},${GNUGO_GAMES},${_POLICY_CE},${_VALUE_ACC},${_SELF_VACC},${_H2HPREV_WINS},${_H2HPREV_N},${POSITIONS},${T_GEN},${_GNUGO_PH_WINS},${_PACHI_WINS}" >> "$RESULTS_CSV"
    else
        log "round $round: COMPLETE (train-only, lr=${TRAIN_LR})"
        echo "${round},,,,,,,,,,${POSITIONS},${T_GEN},," >> "$RESULTS_CSV"
    fi
done

log "ALL ROUNDS DONE"
"$PY" -u scripts/plot_gumbel_loop.py --csv "$RESULTS_CSV" \
    --out results/gumbel_loop_b9_v12.png >> "$LOG" 2>&1 || true
cat "$RESULTS_CSV"
