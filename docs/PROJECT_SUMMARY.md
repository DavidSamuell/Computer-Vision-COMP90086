# Nutrition5K Calorie Prediction - Project Summary

## Overview

Complete implementation of a Multi-Stream CNN with Middle Fusion for predicting food calories from RGB and depth images, following a data-efficient, regularization-heavy approach.

## Architecture Details

### 1. Multi-Stream CNN with Middle Fusion

- **RGB Encoder**: ResNet-18 (3 input channels) → 512 feature channels
- **Depth Encoder**: ResNet-18 (1 input channel) → 512 feature channels
- **Middle Fusion**: Concatenate (1024 channels) → 1x1 Conv → 512 channels
- **Regression Head**: Global pooling → FC layers (256 → 128 → 1) with dropout
- **Segmentation Head**: Transposed convolutions for upsampling to full resolution

### 2. Model Size

- **Total Parameters**: ~11-15M (small enough to prevent overfitting)
- **Backbone**: ResNet-18 (trained from scratch, no pretrained weights)
- **Design Philosophy**: Moderate size for data efficiency

## Key Features Implemented

### ✅ Data Augmentation (Fighting Overfitting)

**Geometric Augmentations** (applied consistently to RGB, depth, and mask):

- Random horizontal flip (50% probability)
- Random rotation (±15 degrees, 50% probability)
- Random resized crop (scale 0.8-1.0, 50% probability)

**Color Augmentations** (applied to RGB only):

- Random brightness (0.7-1.3x)
- Random contrast (0.7-1.3x)
- Random saturation (0.7-1.3x)
- Random hue shift (±0.1)

### ✅ Multi-Task Learning

**Primary Task**: Calorie prediction (regression, MSE loss)
**Auxiliary Task**: Food segmentation (binary segmentation, BCE loss)

**Combined Loss**:

```
Total Loss = α × Calorie Loss + β × Segmentation Loss
```

Default weights: α=1.0, β=0.5

**Why It Works**: The segmentation task forces the model to learn precise spatial understanding of food, preventing shortcuts and improving calorie prediction accuracy.

### ✅ Strong Regularization

1. **Dropout (0.4)**: Applied in regression head to prevent overfitting
2. **Weight Decay (1e-4)**: L2 regularization in AdamW optimizer
3. **Early Stopping (patience=15)**: Stops when validation loss plateaus
4. **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.5, patience=5)
5. **Gradient Clipping (max_norm=1.0)**: Prevents exploding gradients

### ✅ Robust Error Handling

- **Pre-validation**: All images validated at dataset initialization
- **Corrupt Image Handling**: Automatically skips corrupt/missing images
- **Fallback Mechanism**: Returns random valid sample if loading fails
- **Warnings**: Logs all problematic samples for debugging

## File Structure

```
compvis/
├── Nutrition5K/              # Dataset directory
│   ├── nutrition5k_train.csv
│   └── train/
│       ├── color/
│       └── depth_raw/
├── src/                      # Source code
│   ├── __init__.py
│   ├── dataset.py           # Dataset loader with augmentation
│   ├── model.py             # Multi-Stream CNN architecture
│   ├── train.py             # Training script
│   └── predict.py           # Inference script
├── requirements.txt          # Python dependencies
├── run_training.sh          # Shell script for easy training
├── README.md                # Full documentation
├── QUICKSTART.md            # Quick start guide
└── PROJECT_SUMMARY.md       # This file
```

## Components

### 1. Dataset (`dataset.py`)

- **Class**: `Nutrition5KDataset`
- **Features**:
  - Loads RGB images, depth images, and generates segmentation masks
  - Applies synchronized geometric augmentation
  - Applies color augmentation to RGB only
  - Pre-validates all images
  - Handles corrupt images gracefully
  - Automatic train/val split

### 2. Model (`model.py`)

- **Main Class**: `MultiStreamCaloriePredictor`
- **Components**:
  - `ResNetEncoder`: Feature extraction backbone
  - `MiddleFusionModule`: 1x1 conv fusion
  - `RegressionHead`: Calorie prediction with dropout
  - `SegmentationHead`: Upsampling for mask prediction

### 3. Training (`train.py`)

- **Main Class**: `Trainer`
- **Features**:
  - Multi-task loss calculation
  - Training and validation loops
  - Early stopping mechanism
  - Model checkpointing
  - TensorBoard logging
  - Metrics tracking (MAE, RMSE, MAPE)

### 4. Inference (`predict.py`)

- **Main Class**: `CaloriePredictor`
- **Features**:
  - Single image prediction
  - Batch prediction
  - Segmentation mask output
  - Easy-to-use API

## Usage

### Training

**Quick start**:

```bash
./run_training.sh
```

**Full control**:

```bash
cd src
python train.py \
    --data_root ../Nutrition5K/train \
    --csv_path ../Nutrition5K/nutrition5k_train.csv \
    --batch_size 16 \
    --num_epochs 100 \
    --lr 1e-4 \
    --dropout 0.4 \
    --weight_decay 1e-4 \
    --early_stopping_patience 15 \
    --calorie_weight 1.0 \
    --seg_weight 0.5
```

### Inference

**Single prediction**:

```bash
python predict.py \
    --checkpoint ../outputs/my_experiment/best_model.pth \
    --rgb /path/to/rgb.png \
    --depth /path/to/depth_raw.png \
    --save_mask output_mask.png
```

## Training Strategy

### Phase 1: Initial Training

- Start with default hyperparameters
- Monitor both train and validation losses
- Watch for overfitting signs (train loss << val loss)

### Phase 2: Hyperparameter Tuning (if needed)

- **If overfitting**: Increase dropout (0.5-0.6), increase weight decay (5e-4)
- **If underfitting**: Decrease dropout (0.2-0.3), decrease weight decay (1e-5)
- **If unstable**: Reduce learning rate, increase batch size

### Phase 3: Loss Balancing

- Adjust `--calorie_weight` and `--seg_weight` based on task priorities
- If segmentation is too hard, reduce `--seg_weight` to 0.3
- If calorie prediction is poor, increase `--calorie_weight` to 1.5

## Expected Results

With proper training on Nutrition5k dataset:

- **MAE**: 50-100 kcal
- **RMSE**: 80-150 kcal
- **MAPE**: 15-30%
- **Training Time**: 30-60 epochs (3-10 hours on GPU)

## Design Rationale

### Why ResNet-18?

- Data-efficient: Strong inductive bias from convolutional structure
- Moderate size: ~11M parameters prevents overfitting on small datasets
- Proven architecture: Well-established and stable

### Why Middle Fusion?

- **Better than early fusion**: Allows each modality to learn specialized features first
- **Better than late fusion**: Enables interaction between modalities at feature level
- **Efficient**: 1x1 convolution learns optimal mixing with minimal parameters

### Why Multi-Task Learning?

- **Regularization**: Forces model to learn robust representations
- **Spatial awareness**: Segmentation task prevents relying on spurious correlations
- **Better features**: Shared features benefit both tasks

### Why No Pretrained Weights?

- **Constraint**: Following the requirement to train from scratch only
- **Small dataset**: Pretrained weights might dominate, preventing adaptation
- **Domain gap**: Food images differ from ImageNet

### Why These Augmentations?

- **Geometric**: Preserves spatial relationships across modalities
- **Color**: Only RGB (depth is metric, should not be color-augmented)
- **Conservative**: Not too aggressive to prevent destroying semantic content

## Monitoring Training

Launch TensorBoard:

```bash
tensorboard --logdir outputs/<experiment_name>/tensorboard
```

**Key metrics to watch**:

1. **Train vs Val Loss**: Should be similar (not too far apart)
2. **MAE/RMSE**: Lower is better, watch for validation metrics
3. **Learning Rate**: Should decrease when validation loss plateaus
4. **Segmentation Loss**: Should converge to ~0.3-0.5

## Troubleshooting

| Issue             | Solution                                       |
| ----------------- | ---------------------------------------------- |
| Out of memory     | Reduce `--batch_size` to 8 or 4                |
| Training too slow | Increase `--batch_size` or reduce `--img_size` |
| Overfitting       | Increase `--dropout` and `--weight_decay`      |
| Underfitting      | Decrease regularization or train longer        |
| Corrupt images    | They're automatically skipped with warnings    |
| NaN losses        | Reduce learning rate or check data loading     |

## Future Improvements

If you want to enhance the model further:

1. **Add more modalities**: Include food metadata if available
2. **Ensemble models**: Train multiple models and average predictions
3. **Better segmentation**: Use real segmentation masks if available
4. **Test-time augmentation**: Average predictions over multiple augmented versions
5. **Advanced architectures**: Try EfficientNet or ConvNeXt backbones
6. **Loss functions**: Try Huber loss or quantile regression for robustness

## References

- ResNet: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- AdamW: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
- Multi-Task Learning: "An Overview of Multi-Task Learning in Deep Neural Networks" (Ruder, 2017)

## License

This code is for educational/research purposes as part of the COMP 90086 course.
