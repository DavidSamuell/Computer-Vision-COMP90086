#!/bin/bash
# Training script for Nutrition5K Calorie Prediction

# Default configuration
DATA_ROOT="../Nutrition5K/train"
CSV_PATH="../Nutrition5K/nutrition5k_train.csv"
BATCH_SIZE=64
NUM_EPOCHS=500
LR=0.0005
DROPOUT=0.4
WEIGHT_DECAY=1e-4
EARLY_STOPPING=15
CALORIE_WEIGHT=1.0
SEG_WEIGHT=0.3
VAL_RATIO=0.15
IMG_SIZE=224
NUM_WORKERS=4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create experiment name if not provided
if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME=$(date +"%Y%m%d_%H%M%S")
fi

echo "=========================================="
echo "Nutrition5K Calorie Prediction Training"
echo "=========================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Epochs: $NUM_EPOCHS"
echo "=========================================="

cd src

python train.py \
    --data_root "$DATA_ROOT" \
    --csv_path "$CSV_PATH" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --lr "$LR" \
    --dropout "$DROPOUT" \
    --weight_decay "$WEIGHT_DECAY" \
    --early_stopping_patience "$EARLY_STOPPING" \
    --calorie_weight "$CALORIE_WEIGHT" \
    --seg_weight "$SEG_WEIGHT" \
    --val_ratio "$VAL_RATIO" \
    --img_size "$IMG_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --experiment_name "$EXPERIMENT_NAME"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Results saved to: ../outputs/$EXPERIMENT_NAME"
echo "=========================================="

