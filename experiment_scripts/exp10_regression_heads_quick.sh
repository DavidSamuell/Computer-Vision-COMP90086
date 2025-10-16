#!/bin/bash
# Experiment 10: Quick Regression Head Test (Current Best Config)
# Test different regression heads on early_fusion_resnet18 (your best: 63.78 MAE)
# Quick test to see if regression heads matter before full fusion experiments

cd ../src

echo "Testing regression heads on current best config (early_fusion_resnet18)..."

# Test 1: Minimal head (ultra-lightweight)
echo "1. Testing minimal regression head..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder early_fusion_resnet18 \
    --fusion middle \
    --regression_head minimal \
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
    --experiment_name exp10_early_fusion_minimal_head \
    --num_workers 4 \
    --no_augment

# Test 2: Light head (recommended for limited data)
echo "2. Testing light regression head..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder early_fusion_resnet18 \
    --fusion middle \
    --regression_head light \
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
    --experiment_name exp10_early_fusion_light_head \
    --num_workers 4 \
    --no_augment

# Test 3: SE head (attention-based)
echo "3. Testing SE regression head..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder early_fusion_resnet18 \
    --fusion middle \
    --regression_head se \
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
    --experiment_name exp10_early_fusion_se_head \
    --num_workers 4 \
    --no_augment

echo ""
echo "=========================================="
echo "Experiment 10 Complete!"
echo "Quick regression head test on current best config"
echo "Compare MAE: minimal vs light vs se vs standard (63.78)"
echo "Use best head for future fusion experiments"
echo "=========================================="
