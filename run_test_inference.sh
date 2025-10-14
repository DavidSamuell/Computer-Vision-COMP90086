#!/bin/bash
# Test Set Inference Script

# Default paths
MODEL_PATH=""
TEST_ROOT="../Nutrition5K/test"
OUTPUT_PATH="submission.csv"
BATCH_SIZE=32
IMG_SIZE=224
DEVICE="cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --test_root)
            TEST_ROOT="$2"
            shift 2
            ;;
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
        -h|--help)
            echo "Usage: $0 --model_path <path_to_model.pth> [options]"
            echo ""
            echo "Required:"
            echo "  --model_path PATH     Path to trained model checkpoint"
            echo ""
            echo "Optional:"
            echo "  --test_root PATH      Path to test data directory (default: ../Nutrition5K/test)"
            echo "  --output_path PATH    Output submission file (default: submission.csv)"
            echo "  --batch_size SIZE     Batch size for inference (default: 32)"
            echo "  --device DEVICE       Device to use: cuda or cpu (default: cuda)"
            echo ""
            echo "Example:"
            echo "  $0 --model_path ../outputs/my_experiment/best_model.pth"
            echo "  $0 --model_path ../outputs/my_experiment/best_model.pth --output_path my_submission.csv"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path is required!"
    echo "Usage: $0 --model_path <path_to_model.pth>"
    echo "Use --help for more information"
    exit 1
fi

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Check if test directory exists
if [ ! -d "$TEST_ROOT" ]; then
    echo "Error: Test directory not found: $TEST_ROOT"
    exit 1
fi

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
