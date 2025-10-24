#!/bin/bash
# Test Set Inference Script

# Default paths
MODEL_PATH="/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/experiments/exp5_resnet18_inception_no_aug_20251023_223907/best_model.pth"
TEST_ROOT="/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/Nutrition5K/test"
OUTPUT_PATH="../submission.csv"
BATCH_SIZE=32
IMG_SIZE=224
DEVICE="cuda"

echo "=========================================="
echo "Test Set Inference"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Test data: $TEST_ROOT"
echo "Output: $OUTPUT_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "=========================================="

cd src

python test_inference.py \
    --model_path "$MODEL_PATH" \
    --test_root "$TEST_ROOT" \
    --output_path "$OUTPUT_PATH" \
    --batch_size "$BATCH_SIZE" \
    --img_size "$IMG_SIZE" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Inference completed successfully!"
    echo "Submission file: $OUTPUT_PATH"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Inference failed!"
    echo "=========================================="
    exit 1
fi
