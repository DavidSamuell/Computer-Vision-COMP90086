#!/bin/bash
# Experiment 15: Advanced Scheduler Optimization
# Test different learning rate schedules with best config
# Focus on schedules that work well with limited data

cd ../src

echo "Testing advanced schedulers with best configuration..."
echo "ResNet-18 + Inception Fusion + Different LR Schedules"

# Test 1: Linear scheduler with warmup
echo "1. Testing Linear scheduler..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion inception \
    --regression_head standard \
    --segmentation_head standard \
    --lr 1e-4 \
    --scheduler linear \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp15_inception_linear_scheduler \
    --num_workers 4 \
    --no_augment

# Test 2: OneCycle with lower max LR
echo "2. Testing OneCycle with conservative max LR..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion inception \
    --regression_head standard \
    --segmentation_head standard \
    --lr 0.001 \
    --scheduler one_cycle \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp15_inception_onecycle_conservative \
    --num_workers 4 \
    --no_augment

# Test 3: Cosine annealing with restarts
echo "3. Testing Cosine annealing with warm restarts..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion inception \
    --regression_head standard \
    --segmentation_head standard \
    --lr 1e-4 \
    --scheduler cosine_warm_restarts \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 50 \
    --early_stopping_patience 15 \
    --output_dir ../outputs/experiments \
    --experiment_name exp15_inception_cosine_restarts \
    --num_workers 4 \
    --no_augment

# Test 4: No scheduler (constant LR) - sometimes best for small datasets
echo "4. Testing constant LR (no scheduler)..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion inception \
    --regression_head standard \
    --segmentation_head standard \
    --lr 8e-5 \
    --scheduler none \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp15_inception_constant_lr \
    --num_workers 4 \
    --no_augment

echo ""
echo "=========================================="
echo "Experiment 15 Complete!"
echo "Advanced scheduler optimization"
echo "May find better convergence pattern"
echo "=========================================="
