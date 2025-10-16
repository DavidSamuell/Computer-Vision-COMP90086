#!/bin/bash
# Experiment 6: Inception Fusion ResNet-18
# Multi-scale fusion with parallel convolution branches
# Expected: Better feature mixing, may improve over simple middle fusion

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
    --seg_weight 0.5 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp6_inception_fusion_r18_seg_weight_0.5 \
    --num_workers 4 \
    --no_augment \
    --use_segmentation

echo ""
echo "=========================================="
echo "Experiment 6 Complete!"
echo "Inception Fusion: Multi-scale feature mixing"
echo "Compare with middle fusion baseline"
echo "=========================================="
