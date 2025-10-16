#!/bin/bash
# Experiment 11: Full Regression Head Experiments (Run AFTER fusion experiments)
# Test all regression heads on the best fusion method from Exp 6-7
# 
# INSTRUCTIONS:
# 1. First run Exp 6-7 (inception and late fusion)
# 2. Identify best fusion method
# 3. Edit this script to use best fusion
# 4. Then run this experiment

cd ../src

# Updated with your best fusion results (MAE: 62.99)
BEST_ENCODER="resnet18"                 # Best from Exp 6: Inception fusion
BEST_FUSION="inception"                 # Winner: Inception fusion
BEST_SEG_HEAD="standard"                # Standard works fine

echo "Testing all regression heads on best fusion method..."
echo "Current config: $BEST_ENCODER + $BEST_FUSION"
echo "If this is wrong, edit the script with your best fusion results!"

# Test 1: Minimal head (512 → 1, ~512 params)
echo "1. Testing minimal regression head..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder $BEST_ENCODER \
    --fusion $BEST_FUSION \
    --regression_head minimal \
    --segmentation_head $BEST_SEG_HEAD \
    --lr 0.0001 \
    --scheduler reduce_on_plateau \
    --dropout 0.3 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --seg_weight 0.4 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp11_${BEST_ENCODER}_${BEST_FUSION}_minimal_head \
    --num_workers 4 \
    --no_augment

# Test 2: Light head (512 → 128 → 1, ~66K params)
echo "2. Testing light regression head..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder $BEST_ENCODER \
    --fusion $BEST_FUSION \
    --regression_head light \
    --segmentation_head $BEST_SEG_HEAD \
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
    --experiment_name exp11_${BEST_ENCODER}_${BEST_FUSION}_light_head \
    --num_workers 4 \
    --no_augment

# Test 3: Standard head (512 → 256 → 128 → 1, ~165K params) - Current baseline
echo "3. Testing standard regression head (baseline)..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder $BEST_ENCODER \
    --fusion $BEST_FUSION \
    --regression_head standard \
    --segmentation_head $BEST_SEG_HEAD \
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
    --experiment_name exp11_${BEST_ENCODER}_${BEST_FUSION}_standard_head \
    --num_workers 4 \
    --no_augment

# Test 4: SE head (standard + squeeze-excitation, ~170K params)
echo "4. Testing SE regression head..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder $BEST_ENCODER \
    --fusion $BEST_FUSION \
    --regression_head se \
    --segmentation_head $BEST_SEG_HEAD \
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
    --experiment_name exp11_${BEST_ENCODER}_${BEST_FUSION}_se_head \
    --num_workers 4 \
    --no_augment

# Test 5: Deep head (5-layer MLP, ~450K params) - Only if not overfitting
echo "5. Testing deep regression head (high capacity)..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder $BEST_ENCODER \
    --fusion $BEST_FUSION \
    --regression_head deep \
    --segmentation_head $BEST_SEG_HEAD \
    --lr 0.0001 \
    --scheduler reduce_on_plateau \
    --dropout 0.5 \
    --weight_decay 5e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --seg_weight 0.4 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp11_${BEST_ENCODER}_${BEST_FUSION}_deep_head \
    --num_workers 4 \
    --no_augment

echo ""
echo "=========================================="
echo "Experiment 11 Complete!"
echo "Full regression head comparison on best fusion method"
echo "Expected ranking for 3K data: light > minimal > standard > se > deep"
echo "=========================================="
