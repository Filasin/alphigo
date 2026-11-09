#!/bin/bash
# GPU wrapper for control validation.
# Waits for the selfplay experiment to finish, then runs control checks.
source "${VIRTUAL_ENV:-venv}/bin/activate"
SITE=${VIRTUAL_ENV:-venv}/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/nccl/lib:$SITE/nvidia/nvjitlink/lib:/usr/lib/wsl/lib"
export ALPHIGO_PYTHON=$(which python)
export ALPHIGO_PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$(dirname "$0")/.."

echo "[control-gpu] $(date) Waiting for selfplay experiment to finish..."
while pgrep -f "bench_az_vs_gnugo_mp" > /dev/null 2>&1; do
    sleep 60
done
echo "[control-gpu] $(date) Selfplay experiment done, starting control validation"

bash scripts/run_control_validation.sh

echo "[control-gpu] $(date) ALL DONE"
