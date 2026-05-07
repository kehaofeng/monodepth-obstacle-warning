#!/bin/bash
# Monodepth2 training pipeline for KITTI subset
# Usage: bash scripts/start_training.sh [epochs] [batch_size]
#
# Prerequisites:
#   1. KITTI raw data at E:\monodepth-obstacle-warning\data\kitti\
#   2. Conda env "monodepth" with PyTorch + torchvision

set -e

EPOCHS=${1:-20}
BATCH_SIZE=${2:-4}

PYTHON="D:/anaconda3/envs/monodepth/python.exe"
PROJECT_DIR="E:/monodepth-obstacle-warning"
MD2_DIR="${PROJECT_DIR}/monodepth2"
DATA_PATH="E:/monodepth_project/data/kitti"
LOG_DIR="${PROJECT_DIR}/logs"
SPLIT_NAME="kitti_subset"
MODEL_NAME="kitti_subset_model"

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
  --log_dir "${LOG_DIR}" \
  --save_frequency 5

echo ""
echo "Training complete. Logs saved to ${LOG_DIR}/${MODEL_NAME}/"
