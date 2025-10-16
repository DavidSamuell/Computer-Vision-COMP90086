#!/bin/bash
# Experiment 13: Fine-tune Best Configuration (MSE: 8412.55)
# Micro-adjustments around your best performing model
# exp6_inception_fusion_r18_no_seg: ResNet-18 + Inception + No Segmentation

cd ../src

echo "Fine-tuning best configuration: ResNet-18 + Inception Fusion (MSE: 8412.55)"
echo "Testing micro-adjustments around optimal parameters..."

# Test 1: Slightly lower learning rate
echo "1. Testing LR 9e-5 (slightly lower than 1e-4)..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion inception \
    --regression_head standard \
    --segmentation_head standard \
    --lr 9e-5 \
    --scheduler reduce_on_plateau \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp13_inception_lr_9e5_fine_tune \
    --num_workers 4 \
    --no_augment

# Test 2: Slightly higher learning rate
echo "2. Testing LR 1.1e-4 (slightly higher than 1e-4)..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion inception \
    --regression_head standard \
    --segmentation_head standard \
    --lr 1.1e-4 \
    --scheduler reduce_on_plateau \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp13_inception_lr_1.1e4_fine_tune \
    --num_workers 4 \
    --no_augment

# Test 3: Optimal weight decay (between 1e-6 and 5e-6)
echo "3. Testing weight decay 2e-6..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion inception \
    --regression_head standard \
    --segmentation_head standard \
    --lr 1e-4 \
    --scheduler reduce_on_plateau \
    --dropout 0.4 \
    --weight_decay 2e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp13_inception_wd_2e6_optimal \
    --num_workers 4 \
    --no_augment

# Test 4: Slightly lower dropout
echo "4. Testing dropout 0.35..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion inception \
    --regression_head standard \
    --segmentation_head standard \
    --lr 1e-4 \
    --scheduler reduce_on_plateau \
    --dropout 0.35 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp13_inception_dropout_0.35 \
    --num_workers 4 \
    --no_augment

echo ""
echo "=========================================="
echo "Experiment 13 Complete!"
echo "Fine-tuning around best configuration"
echo "Target: Beat MSE 8412.55"
echo "=========================================="
