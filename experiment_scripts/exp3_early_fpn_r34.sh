#!/bin/bash
# Experiment 3: Early Fusion + FPN (Best of Exp 1 & 2)
# Phase 1 - High Priority
# Expected: 20-30% total improvement

cd ../src

python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder early_fusion_resnet18 \
    --fusion middle \
    --regression_head standard \
    --segmentation_head fpn \
    --lr 1e-4 \
    --scheduler reduce_on_plateau \
    --dropout 0.5 \
    --weight_decay 1e-7 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --seg_weight 0.4 \
    --num_epochs 70 \
    --early_stopping_patience 5 \
    --output_dir ../outputs/experiments \
    --experiment_name exp3_early_fpn_r18_seg_weight_0.4 \
    --num_workers 4 \
    --no_augment \
    --use_segmentation

echo ""
echo "=========================================="
echo "Experiment 3 Complete!"
echo "Best combo: Early Fusion + FPN"
echo "Fewest params + best segmentation"
echo "=========================================="

