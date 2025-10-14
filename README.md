# Nutrition5K Calorie Prediction

Multi-stream CNN with middle fusion for predicting calories from RGB and depth images using multi-task learning.

## Architecture

- **Dual Encoders**: Two ResNet-18 encoders (one for RGB, one for depth)
- **Middle Fusion**: Concatenation + 1x1 convolution to merge features
- **Multi-Task Heads**:
  - Regression head for calorie prediction
  - Segmentation head for food mask prediction

## Features

- ✅ Data-efficient ResNet-18 architecture (trained from scratch)
- ✅ Aggressive data augmentation (geometric + color)
- ✅ Multi-task learning (calories + segmentation)
- ✅ Strong regularization (dropout, weight decay, early stopping)
- ✅ Robust error handling for corrupt images
- ✅ TensorBoard logging
- ✅ Automatic train/val splitting

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Structure

Your dataset should be organized as:

```
Nutrition5K/
├── nutrition5k_train.csv
└── train/
    ├── color/
    │   └── dish_XXXX/
    │       └── rgb.png
    └── depth_raw/
        └── dish_XXXX/
            └── depth_raw.png
```

## Training

### Basic Training

```bash
cd src
python train.py \
    --data_root ../Nutrition5K/train \
    --csv_path ../Nutrition5K/nutrition5k_train.csv \
    --batch_size 16 \
    --num_epochs 100 \
    --lr 1e-4
```

### Advanced Options

```bash
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
    --seg_weight 0.5 \
    --val_ratio 0.15 \
    --img_size 224 \
    --experiment_name my_experiment
```

### Key Arguments

- `--data_root`: Path to training data directory
- `--csv_path`: Path to CSV file with calorie labels
- `--batch_size`: Batch size (default: 16)
- `--num_epochs`: Maximum number of epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--dropout`: Dropout rate for regularization (default: 0.4)
- `--weight_decay`: L2 regularization strength (default: 1e-4)
- `--early_stopping_patience`: Patience for early stopping (default: 15)
- `--calorie_weight`: Weight for calorie loss (default: 1.0)
- `--seg_weight`: Weight for segmentation loss (default: 0.5)
- `--val_ratio`: Validation split ratio (default: 0.15)
- `--no_augment`: Disable data augmentation
- `--output_dir`: Output directory for checkpoints (default: ../outputs)

## Data Augmentation

The training pipeline applies:

**Geometric Augmentations** (applied to RGB, depth, and mask):

- Random horizontal flip
- Random rotation (±15°)
- Random resized crop (scale 0.8-1.0)

**Color Augmentations** (applied to RGB only):

- Random brightness (0.7-1.3x)
- Random contrast (0.7-1.3x)
- Random saturation (0.7-1.3x)
- Random hue shift (±0.1)

## Regularization Techniques

1. **Dropout** (0.4): Applied in regression head
2. **Weight Decay** (1e-4): L2 regularization in AdamW optimizer
3. **Early Stopping** (patience=15): Stops when validation loss plateaus
4. **Learning Rate Scheduling**: ReduceLROnPlateau with factor=0.5
5. **Gradient Clipping** (max_norm=1.0): Prevents exploding gradients

## Multi-Task Learning

The model is trained to simultaneously:

1. **Predict calories** (regression task, MSE loss)
2. **Predict food segmentation** (segmentation task, BCE loss)

Total Loss = Calorie Loss + α × Segmentation Loss

The segmentation task acts as an auxiliary task that forces the model to learn better spatial representations of food, improving calorie prediction.

## Output

Training outputs are saved to `../outputs/<experiment_name>/`:

- `best_model.pth`: Best model checkpoint (lowest validation loss)
- `checkpoint_epoch_X.pth`: Regular checkpoints every 5 epochs
- `final_model.pth`: Final model after training
- `config.json`: Training configuration
- `tensorboard/`: TensorBoard logs

## Monitoring Training

```bash
tensorboard --logdir ../outputs/<experiment_name>/tensorboard
```

Then open http://localhost:6006 in your browser.

## Testing the Components

Test the dataset:

```bash
python dataset.py
```

Test the model:

```bash
python model.py
```

## Error Handling

The dataset class includes robust error handling:

- Pre-validates all images during initialization
- Skips corrupt or missing images
- Logs warnings for problematic samples
- Falls back to random valid samples if loading fails

## Model Size

ResNet-18 based architecture: ~11-15M trainable parameters (small enough to prevent overfitting)

## Expected Performance

With proper training, you should achieve:

- MAE: 50-100 kcal
- RMSE: 80-150 kcal
- MAPE: 15-30%

(These are rough estimates - actual performance depends on data quality and hyperparameters)

## Tips for Better Performance

1. **Tune loss weights**: Adjust `--calorie_weight` and `--seg_weight` based on your priorities
2. **Increase dropout**: If overfitting, increase `--dropout` to 0.5-0.6
3. **Adjust learning rate**: Try values between 1e-5 and 5e-4
4. **Batch size**: Larger batches (32-64) can stabilize training if you have enough GPU memory
5. **Image size**: Larger images (256, 320) may capture more details but require more memory
6. **Weight decay**: Increase to 5e-4 or 1e-3 for stronger regularization

## Troubleshooting

**Out of memory**: Reduce batch size or image size
**Overfitting**: Increase dropout, weight decay, or reduce model size
**Underfitting**: Decrease regularization, increase model capacity, or train longer
**Unstable training**: Reduce learning rate or increase batch size
