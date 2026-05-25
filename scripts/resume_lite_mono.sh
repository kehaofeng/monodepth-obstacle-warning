#!/bin/bash
# Resume Lite-Mono training from checkpoint
# Usage: bash scripts/resume_lite_mono.sh

set -e

PYTHON=python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LITE_MONO_DIR="${PROJECT_DIR}/lite-mono"
DATA_PATH="${PROJECT_DIR}/data/kitti"
SPLIT_NAME="kitti_subset"

# V1 (7ep) + V2 (7ep) = 14 epochs done. Resume from v2 weights_6 for 16 remaining.
CHECKPOINT="tmp/lite-mono-v2/models/weights_6"
MODEL_NAME="lite-mono-v3"
REMAINING_EPOCHS=16

echo "============================================"
echo " Resume Lite-Mono Training"
echo " Checkpoint:  ${CHECKPOINT}"
echo " Model name:  ${MODEL_NAME}"
echo " Epochs:      ${REMAINING_EPOCHS}"
echo " Data:        ${DATA_PATH}"
echo "============================================"

cd "${LITE_MONO_DIR}"

PYTHONUNBUFFERED=1 ${PYTHON} -u train.py \
  --data_path "${DATA_PATH}" \
  --split ${SPLIT_NAME} \
  --model_name ${MODEL_NAME} \
  --model lite-mono \
  --png \
  --batch_size 12 \
  --height 192 \
  --width 640 \
  --num_epochs ${REMAINING_EPOCHS} \
  --num_workers 1 \
  --drop_path 0.2 \
  --weight_decay 0.01 \
  --disparity_smoothness 0.001 \
  --load_weights_folder "${CHECKPOINT}" \
  --log_dir ./tmp \
  --save_frequency 1

echo ""
echo "Training complete. Logs saved to tmp/${MODEL_NAME}/"
