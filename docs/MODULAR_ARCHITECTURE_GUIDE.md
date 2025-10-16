# Modular Architecture Guide

This guide explains how to use the modular architecture system to easily experiment with different model components.

## Overview

The model has been refactored into modular components:
- **Encoders**: Feature extraction backbones (ResNet-18, ResNet-34, ResNet-50)
- **Fusion Modules**: Multi-modal fusion strategies (Middle, Middle+Attention, Additive)
- **Heads**: Task-specific prediction heads (Standard/Deep Regression, Standard/Light Segmentation)

## Quick Start

### 1. Train with ResNet-34 (using best config from grid search)

```bash
bash train_resnet34.sh
```

This script uses the best hyperparameters from your grid search with ResNet-34.

### 2. Train with custom architecture via command line

```bash
cd src
python train.py \
    --encoder resnet34 \
    --fusion middle \
    --regression_head light \
    --segmentation_head standard \
    --lr 0.005 \
    --scheduler one_cycle \
    --dropout 0.4 \
    --weight_decay 0.0001 \
    --batch_size 64 \
    --num_epochs 100 \
    --early_stopping_patience 15 \
    --calorie_weight 1.0 \
    --seg_weight 0.3 \
    --use_segmentation
```

**Note:** Segmentation is disabled by default. Add `--use_segmentation` to enable auxiliary segmentation task.

### 3. Build models programmatically

```python
from model import build_model, list_available_components

# List all available components
list_available_components()

# Build ResNet-34 with middle fusion
model = build_model(
    encoder='resnet34',
    fusion='middle',
    dropout_rate=0.4
)

# Build ResNet-50 with attention fusion and deep regression head
model = build_model(
    encoder='resnet50',
    fusion='middle_attention',
    regression_head='deep',
    dropout_rate=0.4
)
```

## Available Components

### Encoders (`src/encoders.py`)

#### Standard Dual-Encoder Architecture
- **resnet18**: ResNet-18 (11M params per encoder, 512 output channels)
  - Fastest training
  - Good baseline
  - ~22M total parameters (dual encoder)
- **resnet34**: ResNet-34 (21M params per encoder, 512 output channels)
  - Deeper than ResNet-18
  - Better feature extraction
  - **RECOMMENDED** for your dataset
  - ~42M total parameters (dual encoder)
- **resnet50**: ResNet-50 (23M params per encoder, 2048 output channels)
  - Bottleneck architecture
  - Most expressive
  - Slower training
  - ~46M total parameters (dual encoder)

#### Early Fusion Architecture (Single Encoder)
- **early_fusion_resnet18**: ResNet-18 with 4-channel input (RGB+Depth)
  - 50% fewer parameters than dual ResNet-18
  - Fuses at input level
  - Good for limited data
  - ~11M total parameters
- **early_fusion_resnet34**: ResNet-34 with 4-channel input (RGB+Depth)
  - 50% fewer parameters than dual ResNet-34
  - Deeper feature extraction with early fusion
  - **RECOMMENDED** for limited data scenarios
  - ~21M total parameters

### Fusion Modules (`src/fusion_modules.py`)

**Note:** Fusion modules are only used with standard dual-encoder architectures. Early fusion encoders don't need a fusion module.

- **middle**: Simple concatenation + 1x1 conv
  - Fast and efficient
  - Good default choice
  - Minimal extra parameters (~500K)
- **middle_attention**: Middle fusion with channel attention
  - Learned feature weighting via SE blocks
  - May improve performance
  - Extra parameters (~550K)
- **additive**: Element-wise addition
  - Fastest fusion
  - Requires same channel dimensions
  - Minimal extra parameters (~250K)
- **cross_modal_attention**: Cross-attention between RGB and Depth
  - RGB and Depth attend to each other
  - Best for complementary modalities
  - More parameters (~1M)
- **gated**: Adaptive gating mechanism
  - Learns per-sample fusion weights
  - Handles noisy modalities well
  - Extra parameters (~800K)
- **late_average**: Average predictions from separate streams
  - Each modality has its own head
  - Simple ensemble approach
  - No fusion module parameters
- **late_weighted**: Learned weighted combination of predictions
  - Learnable fusion weights
  - Better than simple averaging
  - Minimal extra parameters (~2)

### Regression Heads (`src/heads.py`)

- **minimal** ⭐ NEW: Direct linear mapping (512→1)
  - Ultra lightweight (~512 params)
  - Best for: Very limited data, strong encoder
  - **RECOMMENDED** for 3K samples with ResNet-34+
  
- **light** ⭐ NEW: Single hidden layer (512→128→1)
  - Lightweight (~66K params)
  - Best for: Limited data scenarios (3K samples)
  - **RECOMMENDED** for your dataset size
  
- **standard**: 3-layer MLP (512→256→128→1)
  - Balanced capacity (~165K params)
  - Good default choice
  - Current baseline
  
- **se**: Standard + Squeeze-and-Excitation attention
  - Channel attention before pooling (~170K params)
  - Best for: When feature weighting helps
  
- **deep**: 5-layer MLP (512→512→256→128→64→1)
  - High capacity (~450K params)
  - Best for: Large datasets (10K+ samples)
  - Risk of overfitting on 3K samples

### Segmentation Heads (`src/heads.py`)

**Note:** Segmentation is disabled by default. Use `--use_segmentation` flag to enable.

- **standard**: Transposed convolutions for upsampling
  - Learned upsampling (5 transpose conv layers)
  - Better quality masks
  - More parameters (~5M)
  
- **light**: Bilinear upsampling + convolutions
  - Fixed bilinear upsampling
  - Faster training
  - Fewer parameters (~300K)
  
- **fpn**: Feature Pyramid Network style head
  - Progressive upsampling with refinement
  - Best segmentation quality
  - Good for limited data
  - Medium parameters (~1M)

## Training Hyperparameters

### Learning Rate Schedulers

- **one_cycle** (RECOMMENDED for OneCycle paper method)
  - Uses OneCycleLR with max_lr = lr * 10
  - Fast convergence
  - Great for limited data
  - Automatically handles warmup and decay
  - **Use higher base LR** (e.g., 0.005 instead of 0.0005)
  
- **linear**: Linear decay from initial LR to 0
  - Simple and predictable
  - Use lower base LR (e.g., 0.0001-0.001)
  
- **cosine_warm_restarts**: Cosine annealing with restarts
  - Periodic restarts help escape local minima
  - Good for long training runs
  
- **reduce_on_plateau**: Reduce LR when validation plateaus
  - Conservative and adaptive
  - More patient (patience=10)
  
- **none**: No scheduler (constant LR)
  - Simplest option

### Augmentation Options

- **Default (enabled)**: Use `--augment` or don't specify (augmentation ON by default)
- **Disable**: Use `--no_augment` to disable all augmentation
  - Useful for debugging or when data is already diverse

### Multi-Task Learning Options

- **Calorie-only (RECOMMENDED)**: Don't add `--use_segmentation` flag
  - Focus solely on calorie prediction
  - Simpler and often better results
  
- **Multi-task**: Add `--use_segmentation` flag
  - Segmentation as auxiliary task
  - May improve feature learning
  - Recommended weights: `--calorie_weight 1.0 --seg_weight 0.1-0.3`

### Complete Hyperparameter Reference

```bash
# Model Architecture
--encoder {resnet18, resnet34, resnet50, early_fusion_resnet18, early_fusion_resnet34}
--fusion {middle, middle_attention, additive, cross_modal_attention, gated}
--regression_head {minimal, light, standard, se, deep}
--segmentation_head {standard, light, fpn}

# Training
--lr 0.005                      # Learning rate (use higher for one_cycle)
--scheduler {one_cycle, linear, cosine_warm_restarts, reduce_on_plateau, none}
--batch_size 64                 # Batch size
--num_epochs 100               # Maximum epochs
--early_stopping_patience 15   # Early stopping patience

# Regularization
--dropout 0.4                  # Dropout rate
--weight_decay 0.0001          # L2 regularization

# Loss Weights
--calorie_weight 1.0           # Calorie loss weight
--seg_weight 0.3               # Segmentation loss weight (if using segmentation)

# Data
--augment / --no_augment       # Enable/disable augmentation (default: enabled)
--use_segmentation             # Enable segmentation task (default: disabled)
--img_size 224                 # Image size

# System
--num_workers 4                # Data loading workers
--seed 42                      # Random seed
```

## Running Experiments

### Experiment 1: Compare Encoders
```bash
# ResNet-18 (baseline)
python train.py --encoder resnet18 --experiment_name resnet18_baseline

# ResNet-34 (deeper)
python train.py --encoder resnet34 --experiment_name resnet34_experiment

# ResNet-50 (largest)
python train.py --encoder resnet50 --experiment_name resnet50_experiment
```

### Experiment 2: Compare Fusion Strategies
```bash
# Middle fusion (baseline)
python train.py --encoder resnet34 --fusion middle --experiment_name fusion_middle

# Attention fusion
python train.py --encoder resnet34 --fusion middle_attention --experiment_name fusion_attention

# Additive fusion
python train.py --encoder resnet34 --fusion additive --experiment_name fusion_additive
```

### Experiment 3: Grid Search with ResNet-34
```bash
cd src
python hyperparameter_search.py \
    --encoder resnet34 \
    --output_dir ../grid_search_resnet34
```

## Architecture Comparison

### Encoder Comparison

| Encoder            | Type        | Total Params | Output Channels | Speed  | Capacity | For 3K Data |
|--------------------|-------------|--------------|-----------------|--------|----------|-------------|
| resnet18           | Dual        | ~22M         | 512             | Fast   | Good     | ✓ Good      |
| resnet34           | Dual        | ~42M         | 512             | Medium | Better   | ✓✓ Best     |
| resnet50           | Dual        | ~46M         | 2048            | Slow   | Best     | ⚠ May overfit |
| early_fusion_r18   | Single      | ~11M         | 512             | Faster | Lower    | ✓✓ Great    |
| early_fusion_r34   | Single      | ~21M         | 512             | Fast   | Good     | ✓✓ Great    |

### Fusion Comparison

| Fusion              | Extra Params | Compute | Description                         | Best Use Case |
|---------------------|--------------|---------|-------------------------------------|---------------|
| middle              | ~500K        | Low     | Simple concatenation                | Default       |
| middle_attention    | ~550K        | Medium  | Channel attention                   | Better fusion |
| additive            | ~250K        | Low     | Element-wise addition               | Fast          |
| cross_modal_attention| ~1M         | High    | RGB-Depth cross-attention          | Complementary |
| gated               | ~800K        | Medium  | Adaptive per-sample gating          | Noisy data    |
| late_average        | ~0           | Low     | Average predictions                 | Ensemble      |
| late_weighted       | ~2           | Low     | Learned prediction weighting        | Better ensemble|

### Regression Head Comparison

| Head     | Params  | Architecture          | Best For                    | Risk for 3K Data |
|----------|---------|----------------------|-----------------------------|--------------------|
| minimal  | ~512    | 512→1                | Very limited data, strong encoder | Low          |
| light    | ~66K    | 512→128→1            | Limited data (3K samples)   | Low                |
| standard | ~165K   | 512→256→128→1        | Balanced                    | Medium             |
| se       | ~170K   | standard + attention | Feature weighting           | Medium             |
| deep     | ~450K   | 512→512→256→128→64→1 | Large datasets (10K+)       | High               |

## Recommendations for 3K Dataset

Based on your dataset (Nutrition5K with 3,000 training samples):

### Best Starting Configuration (Limited Data)

```bash
python train.py \
    --encoder resnet34 \
    --fusion middle \
    --regression_head light \
    --lr 0.005 \
    --scheduler one_cycle \
    --dropout 0.4 \
    --weight_decay 0.0001 \
    --batch_size 64 \
    --num_epochs 70 \
    --early_stopping_patience 5 \
    --no_augment
```

**Why these choices:**
- ResNet-34: Good balance (not too deep)
- Light head: Prevents overfitting with limited data
- OneCycle: Fast convergence for small datasets
- No augmentation initially: Cleaner signal to debug

### If Underfitting (Loss Still High)

Try these progressively:

1. **More capacity in head:**
   ```bash
   --regression_head standard  # or se
   ```

2. **Deeper encoder:**
   ```bash
   --encoder resnet50
   ```

3. **Better fusion:**
   ```bash
   --fusion middle_attention  # or gated
   ```

4. **More data with augmentation:**
   ```bash
   --augment  # Enable data augmentation
   ```

### If Overfitting (Train Loss << Val Loss)

Try these progressively:

1. **Lighter regression head:**
   ```bash
   --regression_head minimal
   ```

2. **Early fusion (fewer parameters):**
   ```bash
   --encoder early_fusion_resnet34
   --fusion middle  # Will be ignored for early fusion
   ```

3. **Stronger regularization:**
   ```bash
   --dropout 0.5
   --weight_decay 0.0005
   ```

4. **Enable augmentation:**
   ```bash
   --augment
   ```

5. **Smaller encoder:**
   ```bash
   --encoder resnet18
   ```

### For Fastest Training/Iteration

```bash
python train.py \
    --encoder resnet18 \
    --regression_head minimal \
    --batch_size 128 \
    --num_epochs 50
```

### For Best Performance (If You Have Time)

```bash
python train.py \
    --encoder resnet34 \
    --fusion gated \
    --regression_head se \
    --lr 0.005 \
    --scheduler one_cycle \
    --dropout 0.4 \
    --batch_size 64 \
    --num_epochs 100 \
    --augment
```

## Testing Models

Test any architecture:
```bash
cd src
python model.py
```

This will:
1. List all available components
2. Test the original ResNet-18 model
3. Test multiple modular configurations

## Backward Compatibility

The original `MultiStreamCaloriePredictor` class still exists and works exactly as before:

```python
from model import MultiStreamCaloriePredictor

# This still works!
model = MultiStreamCaloriePredictor(
    pretrained=False,
    dropout_rate=0.4,
    fusion_channels=512
)
```

All existing scripts (predict.py, test_inference.py) continue to work with saved models.

## File Organization

```
src/
├── encoders.py          # Encoder architectures
├── fusion_modules.py    # Fusion strategies  
├── heads.py             # Prediction heads
├── model.py             # Model factory + original classes
├── train.py             # Training script (now supports --encoder, --fusion flags)
├── hyperparameter_search.py  # Grid search (now supports architecture search)
├── dataset.py           # Dataset (unchanged)
├── predict.py           # Inference (unchanged)
└── test_inference.py    # Test set inference (unchanged)
```

## Next Steps

1. **Train ResNet-34 with best config**: `bash train_resnet34.sh`
2. **Compare with ResNet-18**: Check validation loss improvements
3. **Try attention fusion**: May improve multi-modal fusion
4. **Grid search with ResNet-34**: Find optimal hyperparameters for the new architecture

## Tips

- ResNet-34 has ~2x parameters of ResNet-18 but only ~1.5x slower
- **New heads available:** `light` and `minimal` for limited data scenarios
- Start with same hyperparameters as best config from ResNet-18
- Monitor for overfitting (your dataset is relatively small)
- Use TensorBoard to compare experiments: `tensorboard --logdir outputs/`
- **OneCycle scheduler requires higher LR** (0.005 vs 0.0005 for linear)

## Quick Command Reference

### Minimal Example (Fastest)
```bash
python train.py --encoder resnet18 --regression_head minimal
```

### Recommended for 3K Data
```bash
python train.py \
    --encoder resnet34 \
    --regression_head light \
    --lr 0.005 \
    --scheduler one_cycle \
    --no_augment
```

### Full Featured (Best Performance)
```bash
python train.py \
    --encoder resnet34 \
    --fusion gated \
    --regression_head standard \
    --segmentation_head fpn \
    --lr 0.005 \
    --scheduler one_cycle \
    --dropout 0.4 \
    --weight_decay 0.0001 \
    --batch_size 64 \
    --num_epochs 100 \
    --early_stopping_patience 15 \
    --calorie_weight 1.0 \
    --seg_weight 0.2 \
    --use_segmentation \
    --augment \
    --output_dir ../outputs/experiments \
    --experiment_name my_experiment
```

### Early Fusion (Parameter Efficient)
```bash
python train.py \
    --encoder early_fusion_resnet34 \
    --regression_head light \
    --lr 0.005 \
    --scheduler one_cycle
```

## Outputs

After training, you'll find in your output directory:
- `best_model.pth` - Best checkpoint by validation loss
- `best_metrics.json` - Best training/validation metrics (MAE, MSE, MAPE)
- `config.json` - Full configuration used for training
- `tensorboard/` - TensorBoard logs
- `final_model.pth` - Final epoch checkpoint

