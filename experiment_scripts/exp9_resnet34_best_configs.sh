#!/bin/bash
# Experiment 9: ResNet-34 with Best Configurations
# Test your top 3 configs with ResNet-34 for better capacity

cd ../src

echo "Testing ResNet-34 with best configurations..."

# Config 1: Inception fusion ResNet-34 (your new best)
echo "1. Inception Fusion ResNet-34..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet34 \
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
    --early_stopping_patience 7 \
    --output_dir ../outputs/experiments \
    --experiment_name exp9_inception_resnet34_no_seg \
    --num_workers 4 \
    --no_augment

# Config 2: Baseline with segmentation (your #1)
echo "2. Baseline ResNet-34 with segmentation..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet34 \
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
    --experiment_name exp9_baseline_resnet34_seg_weight_0.5 \
    --num_workers 4 \
    --no_augment \
    --use_segmentation

# Config 3: FPN segmentation
echo "3. ResNet-34 with FPN segmentation..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet34 \
    --fusion middle \
    --regression_head standard \
    --segmentation_head fpn \
    --lr 0.0001 \
    --scheduler reduce_on_plateau \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --seg_weight 0.4 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp9_fpn_seg_resnet34_seg_weight_0.4 \
    --num_workers 4 \
    --no_augment \
    --use_segmentation

echo ""
echo "=========================================="
echo "Experiment 9 Complete!"
echo "ResNet-34 should provide better feature extraction"
echo "Compare with ResNet-18 baselines"
echo "=========================================="
