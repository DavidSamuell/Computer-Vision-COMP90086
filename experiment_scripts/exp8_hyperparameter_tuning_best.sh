#!/bin/bash
# Experiment 8: Hyperparameter Tuning of Best Config
# Based on your top result: early_fusion_resnet18 with MAE 63.78
# Fine-tune learning rate, dropout, and weight decay

cd ../src

echo "Starting hyperparameter tuning for best configuration..."
echo "Based on: early_fusion_resnet18, MAE: 63.78"

# Test different learning rates around the current best (Inception fusion)
for lr in 5e-5 8e-5 1e-4 1.2e-4 1.5e-4; do
    echo "Testing Inception Fusion LR: $lr"
    python train.py \
        --data_root ../Nutrition5K/Nutrition5K/train \
        --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
        --encoder resnet18 \
        --fusion inception \
        --regression_head standard \
        --segmentation_head standard \
        --lr $lr \
        --scheduler reduce_on_plateau \
        --dropout 0.4 \
        --weight_decay 1e-6 \
        --batch_size 32 \
        --calorie_weight 1.0 \
        --num_epochs 40 \
        --early_stopping_patience 7 \
        --output_dir ../outputs/experiments \
        --experiment_name exp8_inception_lr_${lr}_tuning \
        --num_workers 4 \
        --no_augment
done

echo ""
echo "=========================================="
echo "Experiment 8 Complete!"
echo "Hyperparameter tuning for best config"
echo "Check which LR gives lowest validation loss"
echo "=========================================="
