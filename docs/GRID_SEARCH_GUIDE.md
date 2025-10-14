# Grid Search Hyperparameter Tuning Guide

## Overview

The grid search script systematically tests different hyperparameter combinations to find the optimal configuration for your calorie prediction model.

## Features

✅ **Exhaustive Search**: Tests all combinations of specified hyperparameters
✅ **Resume Support**: Can resume interrupted grid searches
✅ **Result Tracking**: Saves all results to CSV for analysis
✅ **Best Model Selection**: Automatically identifies the best configuration
✅ **Early Stopping**: Each configuration uses early stopping to save time
✅ **TensorBoard Logs**: Each run gets its own TensorBoard logs

## Quick Start

### Option 1: Use Default Grid (Easiest)

```bash
./run_grid_search.sh
```

This tests 324 combinations:

- Learning rates: [1e-5, 5e-5, 1e-4, 5e-4]
- Dropout: [0.3, 0.4, 0.5]
- Weight decay: [1e-5, 1e-4, 5e-4]
- Batch size: [8, 16, 32]
- Segmentation weight: [0.3, 0.5, 0.7]

**Estimated time**: 100-250 hours (4-10 days on GPU)

### Option 2: Quick Grid Search (Faster)

For faster testing with fewer combinations:

```bash
cd src
python hyperparameter_search.py \
    --data_root ../Nutrition5K/train \
    --csv_path ../Nutrition5K/nutrition5k_train.csv \
    --custom_grid ../quick_grid_search.json \
    --max_epochs 30 \
    --output_dir ../quick_grid_results
```

This tests only 6 combinations (6-12 hours).

### Option 3: Custom Grid

1. Copy and edit `custom_grid_template.json`:

```bash
cp custom_grid_template.json my_grid.json
# Edit my_grid.json with your preferred values
```

2. Run with custom grid:

```bash
cd src
python hyperparameter_search.py \
    --data_root ../Nutrition5K/train \
    --csv_path ../Nutrition5K/nutrition5k_train.csv \
    --custom_grid ../my_grid.json \
    --output_dir ../my_grid_results
```

## Understanding the Results

### Output Files

After running grid search, you'll find:

```
grid_search_results/
├── grid_search_results.csv    # All results in CSV format
├── best_config.json            # Best configuration found
└── run_XXXX/                   # Individual run directories
    ├── best_model.pth          # Best model for this run
    ├── config.json             # Configuration for this run
    └── tensorboard/            # TensorBoard logs
```

### Results CSV Columns

- `run_id`: Unique identifier for each run
- `best_val_loss`: Best validation loss achieved
- `lr`: Learning rate used
- `dropout`: Dropout rate used
- `weight_decay`: Weight decay used
- `batch_size`: Batch size used
- `calorie_weight`: Calorie loss weight
- `seg_weight`: Segmentation loss weight

### Best Configuration

The best configuration is automatically saved to `best_config.json`:

```json
{
  "run_id": 42,
  "best_val_loss": 45.23,
  "lr": 0.0001,
  "dropout": 0.4,
  "weight_decay": 0.0001,
  "batch_size": 16,
  "calorie_weight": 1.0,
  "seg_weight": 0.5
}
```

## Analyzing Results

### View Results in Python

```python
import pandas as pd

# Load results
df = pd.read_csv('grid_search_results/grid_search_results.csv')

# Sort by validation loss
df_sorted = df.sort_values('best_val_loss')

# View top 10 configurations
print(df_sorted.head(10))

# Analyze effect of learning rate
df.groupby('lr')['best_val_loss'].mean()

# Plot results
import matplotlib.pyplot as plt
plt.scatter(df['lr'], df['best_val_loss'])
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')
plt.xscale('log')
plt.show()
```

### Compare with TensorBoard

```bash
# View all runs simultaneously
tensorboard --logdir grid_search_results/
```

This lets you compare training curves across all configurations!

## Resume Interrupted Search

If your grid search is interrupted, simply run the same command again. The script will:

1. Load existing results from CSV
2. Skip already-completed configurations
3. Continue with remaining combinations

## Customize Grid Search

### Define Your Own Grid

Edit the parameter grid in the JSON file:

```json
{
  "lr": [1e-4, 5e-4], // Test 2 learning rates
  "dropout": [0.3, 0.4, 0.5], // Test 3 dropout values
  "weight_decay": [1e-4], // Fix weight decay
  "batch_size": [16], // Fix batch size
  "calorie_weight": [1.0], // Fix calorie weight
  "seg_weight": [0.3, 0.5, 0.7] // Test 3 seg weights
}
```

Total combinations: 2 × 3 × 1 × 1 × 1 × 3 = 18

### Adjust Training Duration

```bash
# Shorter runs (faster but less accurate)
python hyperparameter_search.py \
    --max_epochs 30 \
    --early_stopping_patience 7

# Longer runs (slower but more accurate)
python hyperparameter_search.py \
    --max_epochs 80 \
    --early_stopping_patience 15
```

## Tips for Efficient Grid Search

### 1. Start Small

Begin with a coarse grid to identify promising regions:

```json
{
  "lr": [1e-5, 1e-4, 1e-3],
  "dropout": [0.3, 0.5],
  "weight_decay": [1e-5, 1e-4],
  "batch_size": [16],
  "calorie_weight": [1.0],
  "seg_weight": [0.5]
}
```

Then refine around the best values.

### 2. Fix Less Important Parameters

If you know certain parameters work well, fix them:

```json
{
  "batch_size": [16], // Fixed
  "calorie_weight": [1.0], // Fixed
  "lr": [5e-5, 1e-4, 5e-4], // Variable
  "dropout": [0.3, 0.4, 0.5], // Variable
  "weight_decay": [1e-4, 5e-4], // Variable
  "seg_weight": [0.5] // Fixed
}
```

### 3. Use Quick Early Stopping

For initial searches, use aggressive early stopping:

```bash
--max_epochs 30 --early_stopping_patience 7
```

### 4. Prioritize Important Hyperparameters

Focus on parameters that typically have the largest impact:

1. **Learning rate** (most important)
2. **Dropout** (regularization)
3. **Weight decay** (regularization)
4. **Loss weights** (task balance)
5. Batch size (less critical, affects training speed)

### 5. Run Overnight/Weekend

Grid search takes time. Start it before you leave:

```bash
nohup ./run_grid_search.sh > grid_search.log 2>&1 &
```

Check progress:

```bash
tail -f grid_search.log
```

## Common Parameter Ranges

### Learning Rate

- **Very small**: 1e-5 to 5e-5 (very stable, slow)
- **Small**: 1e-4 to 5e-4 (good default range)
- **Medium**: 1e-3 (may be unstable)
- **Large**: >1e-3 (usually too large for this task)

### Dropout

- **Light**: 0.2-0.3 (less regularization)
- **Medium**: 0.4-0.5 (good default range)
- **Heavy**: 0.6-0.7 (strong regularization, may hurt performance)

### Weight Decay

- **Very light**: 1e-6 to 1e-5
- **Light**: 1e-5 to 5e-5
- **Medium**: 1e-4 to 5e-4 (good default range)
- **Heavy**: 1e-3 (strong regularization)

### Segmentation Weight

- **Low**: 0.1-0.3 (focus on calories)
- **Medium**: 0.4-0.6 (balanced)
- **High**: 0.7-1.0 (emphasize segmentation)

## Advanced Usage

### Parallel Grid Search

If you have multiple GPUs, run different sections in parallel:

**Terminal 1 (GPU 0):**

```bash
CUDA_VISIBLE_DEVICES=0 python hyperparameter_search.py \
    --custom_grid grid_part1.json \
    --output_dir ../grid_results
```

**Terminal 2 (GPU 1):**

```bash
CUDA_VISIBLE_DEVICES=1 python hyperparameter_search.py \
    --custom_grid grid_part2.json \
    --output_dir ../grid_results
```

### Merge Results

```python
import pandas as pd

df1 = pd.read_csv('grid_results_1/grid_search_results.csv')
df2 = pd.read_csv('grid_results_2/grid_search_results.csv')

df_merged = pd.concat([df1, df2], ignore_index=True)
df_merged.to_csv('all_results.csv', index=False)
```

## Troubleshooting

**Problem**: Grid search is too slow
**Solution**: Reduce `--max_epochs` or use fewer parameter combinations

**Problem**: Out of memory
**Solution**: Reduce batch sizes in the grid: `"batch_size": [8, 16]`

**Problem**: Want to test different architectures
**Solution**: This script tests hyperparameters only. For architecture search, modify the model creation in `_train_single_config()`

**Problem**: Need to stop and resume
**Solution**: Just Ctrl+C and run again - progress is saved in CSV

## Example Workflows

### Workflow 1: Quick Initial Search

```bash
# 1. Test coarse grid (6-12 hours)
python hyperparameter_search.py --custom_grid quick_grid_search.json

# 2. Identify best learning rate and dropout
# 3. Refine around those values
```

### Workflow 2: Comprehensive Search

```bash
# 1. Run full grid search (4-10 days)
./run_grid_search.sh

# 2. Analyze results
# 3. Train final model with best configuration
python train.py <best_hyperparameters>
```

### Workflow 3: Targeted Search

```bash
# 1. Fix most parameters
# 2. Focus on 2-3 key parameters
# 3. Quick iterations (1-2 days)
```

## After Grid Search

Once you've found the best configuration:

```bash
# Train final model with best hyperparameters
cd src
python train.py \
    --lr 0.0001 \
    --dropout 0.4 \
    --weight_decay 0.0001 \
    --batch_size 16 \
    --seg_weight 0.5 \
    --num_epochs 100 \
    --experiment_name final_model
```

Use the full dataset and more epochs for the final model!
