#!/bin/bash
# Ensemble Test Set Inference Script

# Predefined ensemble model paths (best performing models)
MODEL_PATHS=(
    "/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/outputs/experiments/exp14_ensemble_lr_variation/best_model.pth"
    "/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/outputs/experiments/exp14_ensemble_dropout_variation_0.3/best_model.pth"
    "/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/outputs/experiments/exp14_ensemble_dropout_variation_0.2/best_model.pth"
    "/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/outputs/experiments/exp14_ensemble_dropout_variation_0.3_weight_decay_1e-5/best_model.pth"
    "/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/outputs/experiments/exp14_ensemble_seed_123/best_model.pth"
)

# Default paths
TEST_ROOT="/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/Nutrition5K/Nutrition5K/test"
OUTPUT_PATH="../5_ensemble_submission.csv"
BATCH_SIZE=32
IMG_SIZE=224
DEVICE="cuda"

echo "=========================================="
echo "Ensemble Test Set Inference"
echo "=========================================="
echo "MODE: ENSEMBLE (${#MODEL_PATHS[@]} models)"
for i in "${!MODEL_PATHS[@]}"; do
    echo "  Model $((i+1)): ${MODEL_PATHS[$i]}"
done
echo "Test data: $TEST_ROOT"
echo "Output: $OUTPUT_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "=========================================="

cd src

python test_inference.py \
    --model_paths "${MODEL_PATHS[@]}" \
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
