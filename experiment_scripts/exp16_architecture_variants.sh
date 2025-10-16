#!/bin/bash
# Experiment 16: Architecture Variants of Best Config
# Test slight architectural modifications to inception fusion
# Different fusion channels and regression head combinations

cd ../src

echo "Testing architectural variants of best configuration..."
echo "ResNet-18 + Inception Fusion + Architecture Modifications"

# Test 1: Different fusion channels (256 instead of 512)
echo "1. Testing fusion_channels=256..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion inception \
    --regression_head standard \
    --segmentation_head standard \
    --fusion_channels 256 \
    --lr 1e-4 \
    --scheduler reduce_on_plateau \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp16_inception_fusion_channels_256 \
    --num_workers 4 \
    --no_augment

# Test 2: Different fusion channels (768)
echo "2. Testing fusion_channels=768..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion inception \
    --regression_head standard \
    --segmentation_head standard \
    --fusion_channels 768 \
    --lr 1e-4 \
    --scheduler reduce_on_plateau \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp16_inception_fusion_channels_768 \
    --num_workers 4 \
    --no_augment

# Test 3: Inception + Light regression head
echo "3. Testing Inception + Light regression head..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion inception \
    --regression_head light \
    --segmentation_head standard \
    --lr 1e-4 \
    --scheduler reduce_on_plateau \
    --dropout 0.3 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp16_inception_light_head_optimized \
    --num_workers 4 \
    --no_augment

# Test 4: Inception + SE regression head
echo "4. Testing Inception + SE regression head..."
python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion inception \
    --regression_head se \
    --segmentation_head standard \
    --lr 1e-4 \
    --scheduler reduce_on_plateau \
    --dropout 0.4 \
    --weight_decay 1e-6 \
    --batch_size 32 \
    --calorie_weight 1.0 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --output_dir ../outputs/experiments \
    --experiment_name exp16_inception_se_head_optimized \
    --num_workers 4 \
    --no_augment

echo ""
echo "=========================================="
echo "Experiment 16 Complete!"
echo "Architecture variants tested"
echo "May find better architecture configuration"
echo "=========================================="
