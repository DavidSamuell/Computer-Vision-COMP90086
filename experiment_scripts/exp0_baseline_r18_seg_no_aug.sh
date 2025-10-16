#!/bin/bash
# Experiment 0: Baseline ResNet-18 (No Segmentation)
# Simplest baseline - calorie prediction only
# Expected: Establishes baseline performance

cd ../src

python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion middle \
    --regression_head standard \
    --segmentation_head standard \
    --lr 0.0001 \
    --scheduler reduce_on_plateau \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --seg_weight 0.5 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp0_baseline_r18_seg_no_aug_seg_weight_0.5 \
    --num_workers 4 \
    --no_augment \
    --use_segmentation

echo ""
echo "=========================================="
echo "Experiment 0 Complete!"
echo "Baseline ResNet-18 (No Segmentation)"
echo "Use this as comparison for other experiments"
echo "=========================================="


