#!/bin/bash
# Monodepth2 training pipeline for KITTI subset
# Usage: bash scripts/start_training.sh [epochs] [batch_size]
#
# Prerequisites:
#   1. KITTI raw data extracted to data/kitti/
#   2. Python with PyTorch + torchvision installed

set -e

EPOCHS=${1:-40}
BATCH_SIZE=${2:-8}

PYTHON=python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MD2_DIR="${PROJECT_DIR}/monodepth2"
DATA_PATH="${PROJECT_DIR}/data/kitti"
LOG_DIR="${PROJECT_DIR}/logs"
SPLIT_NAME="kitti_subset"
MODEL_NAME="kitti_subset_v2"

echo "============================================"
echo " Monodepth2 Training Pipeline"
echo " Epochs:      ${EPOCHS}"
echo " Batch size:  ${BATCH_SIZE}"
echo " Data:        ${DATA_PATH}"
echo " Logs:        ${LOG_DIR}"
echo "============================================"

# Step 1: Convert split files to Monodepth2 format
echo ""
echo "[1/2] Converting split files to Monodepth2 format..."
${PYTHON} "${PROJECT_DIR}/scripts/convert_to_monodepth2.py"
echo "       Done."

# Step 2: Start training
echo ""
echo "[2/2] Starting training..."
cd "${MD2_DIR}"

PYTHONUNBUFFERED=1 ${PYTHON} -u train.py \
  --data_path "${DATA_PATH}" \
  --split ${SPLIT_NAME} \
  --model_name ${MODEL_NAME} \
  --png \
  --batch_size ${BATCH_SIZE} \
  --height 192 \
  --width 640 \
  --num_epochs ${EPOCHS} \
  --num_layers 18 \
  --num_workers 2 \
  --scheduler_step_size 15 \
  --disparity_smoothness 0.0015 \
  --log_dir "${LOG_DIR}" \
  --save_frequency 5

echo ""
echo "Training complete. Logs saved to ${LOG_DIR}/${MODEL_NAME}/"
