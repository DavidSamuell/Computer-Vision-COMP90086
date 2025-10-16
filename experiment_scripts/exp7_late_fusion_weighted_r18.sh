#!/bin/bash
# Experiment 7: Late Fusion Weighted ResNet-18
# Separate RGB and Depth streams with learned prediction weighting
# Expected: Ensemble-like behavior, may be more robust

cd ../src

python train.py \
    --data_root ../Nutrition5K/Nutrition5K/train \
    --csv_path ../Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet18 \
    --fusion late_weighted \
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
    --experiment_name exp7_late_fusion_weighted_r18_seg_weight_0.4 \
    --num_workers 4 \
    --no_augment \
    --use_segmentation

echo ""
echo "=========================================="
echo "Experiment 7 Complete!"
echo "Late Fusion: Separate streams + learned weighting"
echo "Should be more robust to noisy modalities"
echo "=========================================="
