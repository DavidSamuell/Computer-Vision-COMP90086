#!/bin/bash
# Experiment 7b: Late Fusion Average ResNet-18
# Separate RGB and Depth streams with simple averaging
# Expected: Simple ensemble, baseline for late fusion

cd ../src

python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion late_average \
    --regression_head standard \
    --segmentation_head standard \
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
    --experiment_name exp7_late_fusion_average_r18_seg_weight_0.4 \
    --num_workers 4 \
    --no_augment \
    --use_segmentation

echo ""
echo "=========================================="
echo "Experiment 7b Complete!"
echo "Late Fusion: Simple averaging baseline"
echo "Compare with weighted late fusion"
echo "=========================================="
