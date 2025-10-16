#!/bin/bash
# Experiment 14: Ensemble and Test-Time Augmentation
# Train multiple versions of best config for ensemble
# Different seeds and slight variations

cd ../src

echo "Training ensemble models based on best configuration..."
echo "Multiple seeds + slight variations for robust ensemble"

# Model 1: Best config with seed 42 (baseline)
echo "1. Training ensemble model 1 (seed 42)..."
# python train.py \
#     --data_root ../Nutrition5K/Nutrition5K/train \
#     --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
#     --encoder resnet18 \
#     --fusion inception \
#     --regression_head standard \
#     --segmentation_head standard \
#     --lr 1e-4 \
#     --scheduler reduce_on_plateau \
#     --dropout 0.4 \
#     --weight_decay 1e-6 \
#     --batch_size 32 \
#     --calorie_weight 1.0 \
#     --num_epochs 40 \
#     --early_stopping_patience 10 \
#     --seed 42 \
#     --output_dir ../outputs/experiments \
#     --experiment_name exp14_ensemble_seed_42 \
#     --num_workers 4 \
#     --no_augment

# # Model 2: Different seed
# echo "2. Training ensemble model 2 (seed 123)..."
# python train.py \
#     --data_root ../Nutrition5K/Nutrition5K/train \
#     --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
#     --encoder resnet18 \
#     --fusion inception \
#     --regression_head standard \
#     --segmentation_head standard \
#     --lr 1e-4 \
#     --scheduler reduce_on_plateau \
#     --dropout 0.4 \
#     --weight_decay 1e-6 \
#     --batch_size 32 \
#     --calorie_weight 1.0 \
#     --num_epochs 40 \
#     --early_stopping_patience 10 \
#     --seed 123 \
#     --output_dir ../outputs/experiments \
#     --experiment_name exp14_ensemble_seed_123 \
#     --num_workers 4 \
#     --no_augment

# Model 3: Slight LR variation
# echo "3. Training ensemble model 3 (LR variation)..."
# python train.py \
#     --data_root ../Nutrition5K/Nutrition5K/train \
#     --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
#     --encoder resnet18 \
#     --fusion inception \
#     --regression_head standard \
#     --segmentation_head standard \
#     --lr 9.5e-5 \
#     --scheduler reduce_on_plateau \
#     --dropout 0.4 \
#     --weight_decay 1e-6 \
#     --batch_size 32 \
#     --calorie_weight 1.0 \
#     --num_epochs 40 \
#     --early_stopping_patience 10 \
#     --seed 456 \
#     --output_dir ../outputs/experiments \
#     --experiment_name exp14_ensemble_lr_variation \
#     --num_workers 4 \
#     --no_augment

# Model 4: Slight dropout variation
echo "4. Training ensemble model 4 (dropout variation)..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion inception \
    --regression_head standard \
    --segmentation_head standard \
    --lr 3e-5 \
    --scheduler reduce_on_plateau \
    --dropout 0.4 \
    --weight_decay 1e-5 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 50 \
    --early_stopping_patience 7 \
    --seed 789 \
    --output_dir ../outputs/experiments \
    --experiment_name exp14_ensemble_dropout_variation_0.3_weight_decay_1e-5 \
    --num_workers 4 \
    --no_augment

echo ""
echo "=========================================="
echo "Experiment 14 Complete!"
echo "Ensemble models trained - use for averaging predictions"
echo "Expected: 2-5% improvement through ensemble"
echo "=========================================="
