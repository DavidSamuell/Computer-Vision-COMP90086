#!/bin/bash
# Experiment 6b: Inception Fusion ResNet-18 (No Segmentation)
# Multi-scale fusion, calorie prediction only
# Expected: Better feature mixing for calorie prediction

cd ../src

python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion inception \
    --regression_head standard \
    --segmentation_head standard \
    --lr 0.0001 \
    --scheduler reduce_on_plateau \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp6_inception_fusion_r18_no_seg \
    --num_workers 4 \
    --no_augment

echo ""
echo "=========================================="
echo "Experiment 6b Complete!"
echo "Inception Fusion (No Segmentation)"
echo "Compare with early fusion baseline"
echo "=========================================="
