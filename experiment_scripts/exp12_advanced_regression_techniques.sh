#!/bin/bash
# Experiment 12: Advanced Regression Techniques
# Test advanced regression approaches on best config
# Run AFTER finding best fusion + regression head combination

cd ../src

# Updated with your best results (MAE: 62.99)
BEST_ENCODER="resnet18"                 # Best: ResNet-18 with Inception
BEST_FUSION="inception"                 # Best: Inception fusion
BEST_REG_HEAD="standard"                # Standard beat light head
BEST_SEG_HEAD="standard"                # Standard works fine

echo "Testing advanced regression techniques..."
echo "Config: $BEST_ENCODER + $BEST_FUSION + $BEST_REG_HEAD"

# Test 1: Different schedulers - OneCycle (often better for limited data)
echo "1. Testing OneCycle scheduler..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder $BEST_ENCODER \
    --fusion $BEST_FUSION \
    --regression_head $BEST_REG_HEAD \
    --segmentation_head $BEST_SEG_HEAD \
    --lr 0.005 \
    --scheduler one_cycle \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp12_${BEST_ENCODER}_${BEST_FUSION}_onecycle \
    --num_workers 4 \
    --no_augment

# Test 2: Higher dropout for better regularization
echo "2. Testing higher dropout (0.5)..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder $BEST_ENCODER \
    --fusion $BEST_FUSION \
    --regression_head $BEST_REG_HEAD \
    --segmentation_head $BEST_SEG_HEAD \
    --lr 0.0001 \
    --scheduler reduce_on_plateau \
    --dropout 0.5 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp12_${BEST_ENCODER}_${BEST_FUSION}_dropout_0.5 \
    --num_workers 4 \
    --no_augment

# Test 3: Higher weight decay for better regularization
echo "3. Testing higher weight decay (5e-6)..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder $BEST_ENCODER \
    --fusion $BEST_FUSION \
    --regression_head $BEST_REG_HEAD \
    --segmentation_head $BEST_SEG_HEAD \
    --lr 0.0001 \
    --scheduler reduce_on_plateau \
    --dropout 0.4 \
    --weight_decay 5e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp12_${BEST_ENCODER}_${BEST_FUSION}_wd_5e6 \
    --num_workers 4 \
    --no_augment

# Test 4: Different batch size (larger for more stable gradients)
echo "4. Testing larger batch size (64)..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder $BEST_ENCODER \
    --fusion $BEST_FUSION \
    --regression_head $BEST_REG_HEAD \
    --segmentation_head $BEST_SEG_HEAD \
    --lr 0.0001 \
    --scheduler reduce_on_plateau \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 64 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp12_${BEST_ENCODER}_${BEST_FUSION}_bs64 \
    --num_workers 4 \
    --no_augment

# Test 5: Cosine annealing scheduler
echo "5. Testing cosine annealing scheduler..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder $BEST_ENCODER \
    --fusion $BEST_FUSION \
    --regression_head $BEST_REG_HEAD \
    --segmentation_head $BEST_SEG_HEAD \
    --lr 0.0001 \
    --scheduler cosine_warm_restarts \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp12_${BEST_ENCODER}_${BEST_FUSION}_cosine \
    --num_workers 4 \
    --no_augment

echo ""
echo "=========================================="
echo "Experiment 12 Complete!"
echo "Advanced regression techniques tested"
echo "Note: Some flags may not exist yet - implement if promising"
echo "=========================================="
