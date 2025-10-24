#!/bin/bash
# Simple Inference Script for notebook-trained models

# Default paths
MODEL_PATH="../experiments/nutrition5k_experiments/inceptionv3_image_volume_20251024_140149/best_model.pth"
TEST_ROOT="/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/Nutrition5K/test"
OUTPUT_PATH="../submission_inceptionv2.csv"
BATCH_SIZE=16
IMG_SIZE=256
DEVICE="cpu"

echo "=========================================="
echo "Simple Test Set Inference (Notebook Model Compatible)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Test data: $TEST_ROOT"
echo "Output: $OUTPUT_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "=========================================="

cd src

python inference.py \
    --model_path "$MODEL_PATH" \
    --test_root "$TEST_ROOT" \
    --output_path "$OUTPUT_PATH" \
    --batch_size "$BATCH_SIZE" \
    --img_size "$IMG_SIZE" \
    --device "$DEVICE"
