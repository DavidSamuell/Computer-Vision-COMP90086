#!/bin/bash
# Quick start training script for ResNet-34 with best hyperparameters
# Based on best_config.json from grid search

# Best hyperparameters from grid search
BEST_LR=0.001
BEST_DROPOUT=0.3
BEST_WEIGHT_DECAY=0.0001
BEST_BATCH_SIZE=64
BEST_CALORIE_WEIGHT=1.0
BEST_SEG_WEIGHT=0.5

# Model architecture - NOW USING RESNET-34
ENCODER="resnet34"
FUSION="middle"
REGRESSION_HEAD="standard"
SEGMENTATION_HEAD="standard"

# Data paths
DATA_ROOT="../Nutrition5K/Nutrition5K/train"
CSV_PATH="../Nutrition5K/Nutrition5K/nutrition5k_train.csv"

# Training settings
NUM_EPOCHS=70
EARLY_STOPPING_PATIENCE=5

# Output
OUTPUT_DIR="../outputs/resnet34_v2"
EXPERIMENT_NAME="resnet34_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Training ResNet-34 with Best Config"
echo "=========================================="
echo "Encoder: $ENCODER"
echo "Fusion: $FUSION"
echo "Learning Rate: $BEST_LR"
echo "Dropout: $BEST_DROPOUT"
echo "Weight Decay: $BEST_WEIGHT_DECAY"
echo "Batch Size: $BEST_BATCH_SIZE"
echo "=========================================="

cd src

python train.py \
    --val_ratio 0.15 \
    --data_root "$DATA_ROOT" \
    --csv_path "$CSV_PATH" \
    --encoder "$ENCODER" \
    --fusion "$FUSION" \
    --regression_head "$REGRESSION_HEAD" \
    --segmentation_head "$SEGMENTATION_HEAD" \
    --lr $BEST_LR \
    --dropout $BEST_DROPOUT \
    --weight_decay $BEST_WEIGHT_DECAY \
    --batch_size $BEST_BATCH_SIZE \
    --calorie_weight $BEST_CALORIE_WEIGHT \
    --seg_weight $BEST_SEG_WEIGHT \
    --num_epochs $NUM_EPOCHS \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "$EXPERIMENT_NAME" \
    --num_workers 4

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Model saved to: $OUTPUT_DIR/$EXPERIMENT_NAME"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed!"
    echo "=========================================="
    exit 1
fi

