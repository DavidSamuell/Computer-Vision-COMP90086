#!/bin/bash
# Experiment 1: Early Fusion ResNet-34
# Phase 1 - High Priority
# Expected: 10-15% improvement, ~50% fewer parameters

cd ../src

python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder early_fusion_resnet18 \
    --fusion middle \
    --regression_head standard \
    --segmentation_head standard \
    --lr 0.0001 \
    --scheduler reduce_on_plateau \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --seg_weight 0.4 \
    --num_epochs 40 \
    --early_stopping_patience 7 \
    --output_dir ../outputs/experiments \
    --experiment_name exp1_early_fusion_r18_no_aug_seg_weight_0.4 \
    --num_workers 4 \
    --no_augment

echo ""
echo "=========================================="
echo "Experiment 1 Complete!"
echo "Compare with baseline ResNet-34 results"
echo "=========================================="

