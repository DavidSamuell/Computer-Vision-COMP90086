#!/bin/bash
# Grid Search for Hyperparameter Tuning

echo "=========================================="
echo "Hyperparameter Grid Search"
echo "=========================================="
echo ""
echo "This will run an exhaustive grid search to find"
echo "the best hyperparameter configuration."
echo ""
echo "Default grid will test:"
echo "  - Learning rates: [1e-5, 5e-5, 1e-4, 5e-4]"
echo "  - Dropout: [0.3, 0.4, 0.5]"
echo "  - Weight decay: [1e-5, 1e-4, 5e-4]"
echo "  - Batch size: [8, 16, 32]"
echo "  - Segmentation weight: [0.3, 0.5, 0.7]"
echo ""
echo "Total: 4 × 3 × 3 × 3 × 3 = 324 combinations"
echo "=========================================="
echo ""


python src/hyperparameter_search.py \
    --data_root ./Nutrition5K/train \
    --csv_path ./Nutrition5K/nutrition5k_train.csv \
    --max_epochs 50 \
    --early_stopping_patience 10 \
    --output_dir ./grid_search_results \
    --num_workers 4 \
    --device cuda \
    --custom_grid ./quick_grid_search.json

echo ""
echo "=========================================="
echo "Grid Search Complete!"
echo "=========================================="

