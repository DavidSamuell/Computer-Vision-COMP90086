#!/bin/bash
# Ensemble Inference Helper Script
# Automatically uses the best models from exp14 ensemble experiments

# Default ensemble models (update these paths after exp14 completes)
ENSEMBLE_MODELS=(
    "/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/outputs/experiments/exp14_ensemble_seed_42/best_model.pth"
    "/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/outputs/experiments/exp14_ensemble_seed_123/best_model.pth"
    "/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/outputs/experiments/exp14_ensemble_lr_variation/best_model.pth"
    "/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/outputs/experiments/exp14_ensemble_dropout_variation/best_model.pth"
)

# Default settings
TEST_ROOT="/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/Nutrition5K/Nutrition5K/test"
OUTPUT_PATH="../ensemble_submission.csv"
BATCH_SIZE=32
DEVICE="cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --output_path PATH    Output CSV file (default: ../ensemble_submission.csv)"
            echo "  --batch_size SIZE     Batch size (default: 32)"
            echo "  --device DEVICE       Device to use (default: cuda)"
            echo "  --help               Show this help message"
            echo ""
            echo "This script automatically uses the 4 best models from exp14 ensemble training."
            echo "Make sure to run exp14_ensemble_and_tta.sh first to train the ensemble models."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "ENSEMBLE INFERENCE - BEST 4 MODELS"
echo "=========================================="
echo "Models to ensemble:"
for i in "${!ENSEMBLE_MODELS[@]}"; do
    model_name=$(basename "$(dirname "${ENSEMBLE_MODELS[$i]}")")
    if [[ -f "${ENSEMBLE_MODELS[$i]}" ]]; then
        echo "  ✓ Model $((i+1)): $model_name"
    else
        echo "  ✗ Model $((i+1)): $model_name (NOT FOUND)"
        echo ""
        echo "ERROR: Model not found: ${ENSEMBLE_MODELS[$i]}"
        echo "Please run 'bash exp14_ensemble_and_tta.sh' first to train ensemble models."
        exit 1
    fi
done
echo ""
echo "Output: $OUTPUT_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "=========================================="

# Run ensemble inference
./run_test_inference.sh \
    --model_paths "${ENSEMBLE_MODELS[@]}" \
    --output_path "$OUTPUT_PATH" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "🎉 ENSEMBLE INFERENCE COMPLETE!"
    echo "=========================================="
    echo "Ensemble submission file: $OUTPUT_PATH"
    echo "Used ${#ENSEMBLE_MODELS[@]} models for averaging"
    echo "Expected improvement: 2-5% over single model"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ ENSEMBLE INFERENCE FAILED!"
    echo "=========================================="
    exit 1
fi
