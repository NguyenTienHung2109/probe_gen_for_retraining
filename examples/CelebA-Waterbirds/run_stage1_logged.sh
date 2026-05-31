#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data/waterbirds/waterbird_complete95_forest2water2}"
OUTPUT_PATH="${OUTPUT_PATH:-./checkpoints/deit_small_moe_waterbirds_rerun_logged.pth}"
LOG_DIR="${LOG_DIR:-./logs/deit_small_moe_waterbirds_stage1}"

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_PATH}")"
export PYTHONPATH="${PYTHONPATH:-.}"

python examples/CelebA-Waterbirds/finetune_deit_waterbirds_logged.py \
  --data-root "${DATA_ROOT}" \
  --output-path "${OUTPUT_PATH}" \
  --log-dir "${LOG_DIR}" \
  --num-experts "${NUM_EXPERTS:-4}" \
  --epochs "${EPOCHS:-10}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr "${LR:-1e-5}" \
  --weight-decay "${WEIGHT_DECAY:-0.05}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --seed "${SEED:-0}" \
  --log-every "${LOG_EVERY:-20}" \
  2>&1 | tee "${LOG_DIR}/stdout.log"
