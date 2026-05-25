#!/bin/bash
# Lite-Mono training pipeline for KITTI subset
# Usage: bash scripts/start_lite_mono_training.sh [epochs]
#
# Prerequisites:
#   1. KITTI raw data extracted to data/kitti/
#   2. timm, thop, pytorch-linear-warmup-cosine-annealing-warm-restarts-weight-decay installed
#   3. monodepth conda env activated

set -e

EPOCHS=${1:-30}

PYTHON=python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LITE_MONO_DIR="${PROJECT_DIR}/lite-mono"
DATA_PATH="${PROJECT_DIR}/data/kitti"
SPLIT_NAME="kitti_subset"
MODEL_NAME="lite-mono-v1"

echo "============================================"
echo " Lite-Mono Training Pipeline"
echo " Epochs:      ${EPOCHS}"
echo " Data:        ${DATA_PATH}"
echo " Logs:        ${LITE_MONO_DIR}/tmp/${MODEL_NAME}"
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
  --num_epochs ${EPOCHS} \
  --num_workers 2 \
  --drop_path 0.2 \
  --weight_decay 0.01 \
  --disparity_smoothness 0.001 \
  --mypretrain pretrained/lite-mono_pretrained.pth \
  --log_dir ./tmp \
  --save_frequency 1

echo ""
echo "Training complete. Logs saved to tmp/${MODEL_NAME}/"
