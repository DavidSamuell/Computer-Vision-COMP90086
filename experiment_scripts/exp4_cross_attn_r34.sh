#!/bin/bash
# Experiment 4: Cross-Modal Attention Fusion
# Phase 2 - Medium Priority
# Try ONLY if Phase 1 successful and no overfitting
# Expected: 5-10% improvement over middle fusion

cd ../src

python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion cross_modal_attention \
    --regression_head standard \
    --segmentation_head fpn \
    --lr 5-e5 \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --seg_weight 0.5 \
    --num_epochs 40 \
    --early_stopping_patience 7 \
    --output_dir ../outputs/experiments \
    --experiment_name exp4_cross_attn_18_seg_weight_0.5 \
    --num_workers 4 \
    --use_segmentation \
    --no_augment

echo ""
echo "=========================================="
echo "Experiment 4 Complete!"
echo "Cross-modal attention learns RGB/Depth importance"
echo "=========================================="

