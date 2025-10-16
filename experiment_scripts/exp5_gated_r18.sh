#!/bin/bash
# Experiment 5: Gated Fusion
# Phase 2 - Medium Priority
# Try ONLY if Phase 1 successful and no overfitting
# Expected: 3-7% improvement, good for noisy modalities

cd ../src

python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion gated \
    --regression_head standard \
    --segmentation_head fpn \
    --lr 1e-4 \
    --scheduler reduce_on_plateau \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --seg_weight 0.5 \
    --num_epochs 40 \
    --early_stopping_patience 5 \
    --output_dir ../outputs/experiments \
    --experiment_name exp5_gated_r18_seg_weight_0.5 \
    --num_workers 4 \
    --use_segmentation \
    --no_augment

echo ""
echo "=========================================="
echo "Experiment 5 Complete!"
echo "Gated fusion adapts per sample"
echo "=========================================="

