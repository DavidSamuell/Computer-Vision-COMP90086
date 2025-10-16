# Quick Start: Training with ResNet-34

## What's New?

The model architecture is now **modular**! You can easily switch between:
- **Encoders**: ResNet-18, ResNet-34, ResNet-50
- **Fusion strategies**: Middle, Middle+Attention, Additive
- **Prediction heads**: Standard/Deep (regression), Standard/Light (segmentation)

The old code is preserved for documentation - everything is backward compatible!

## Train ResNet-34 with Best Config

The simplest way to train with ResNet-34 using your best hyperparameters from the grid search:

```bash
bash train_resnet34.sh
```

This script automatically uses:
- **Encoder**: ResNet-34 (deeper than ResNet-18)
- **Best hyperparameters** from `best_config.json`:
  - Learning rate: 0.0005
  - Dropout: 0.4
  - Weight decay: 0.0001
  - Batch size: 64
  - Calorie weight: 1.0
  - Segmentation weight: 0.5

## Manual Training Command

If you want more control:

```bash
cd src
python train.py \
    --data_root ./Nutrition5K/Nutrition5K/train \
    --csv_path ./Nutrition5K/Nutrition5K/nutrition5k_train.csv \
    --encoder resnet34 \
    --fusion middle \
    --lr 0.0005 \
    --dropout 0.4 \
    --weight_decay 0.0001 \
    --batch_size 64 \
    --calorie_weight 1.0 \
    --seg_weight 0.5 \
    --num_epochs 50 \
    --experiment_name my_resnet34_run
```

## Try Different Architectures

### ResNet-50 (larger model)
```bash
cd src
python train.py --encoder resnet50 --experiment_name resnet50_test
```

### With Attention Fusion
```bash
cd src
python train.py --encoder resnet34 --fusion middle_attention --experiment_name resnet34_attention
```

### With Deep Regression Head
```bash
cd src
python train.py --encoder resnet34 --regression_head deep --experiment_name resnet34_deep
```

## Grid Search with ResNet-34

Run hyperparameter search with ResNet-34:

```bash
cd src
python hyperparameter_search.py \
    --encoder resnet34 \
    --output_dir ../grid_search_resnet34 \
    --max_epochs 50
```

Or use a custom parameter grid JSON file:

```bash
python hyperparameter_search.py \
    --encoder resnet34 \
    --custom_grid ../my_param_grid.json \
    --output_dir ../grid_search_custom
```

## Check Available Components

```bash
cd src
python model.py
```

This will:
1. List all available components (encoders, fusion modules, heads)
2. Test the original ResNet-18 model
3. Test multiple ResNet-34 and ResNet-50 configurations

## Python API

Use in your own scripts:

```python
from model import build_model

# Build ResNet-34 model
model = build_model(
    encoder='resnet34',
    fusion='middle',
    regression_head='standard',
    segmentation_head='standard',
    pretrained=False,
    dropout_rate=0.4,
    fusion_channels=512
)

print(f"Model has {model.get_num_parameters():,} parameters")
print(f"Config: {model.get_config()}")
```

## Expected Improvements with ResNet-34

Compared to ResNet-18:
- **~2x more parameters** (21M vs 11M per encoder)
- **Deeper architecture** (34 vs 18 layers)
- **Better feature extraction** (more residual blocks)
- **~1.5x slower training** (worth it for better performance)
- **Expected validation loss** improvement of 5-15%

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir outputs/
```

Then open http://localhost:6006 in your browser.

## File Structure

```
src/
├── encoders.py          # NEW: Different encoder architectures
├── fusion_modules.py    # NEW: Different fusion strategies
├── heads.py             # NEW: Different prediction heads
├── model.py             # UPDATED: Now includes modular factory functions
├── train.py             # UPDATED: Supports --encoder, --fusion flags
├── hyperparameter_search.py  # UPDATED: Supports architecture search
└── [other files unchanged]

train_resnet34.sh        # NEW: Quick start script for ResNet-34
docs/
└── MODULAR_ARCHITECTURE_GUIDE.md  # NEW: Detailed guide
```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're in the `src/` directory or have it in your PYTHONPATH.

### CUDA Out of Memory
If you run out of GPU memory with ResNet-34 or ResNet-50:
- Reduce batch size: `--batch_size 32` or `--batch_size 16`
- Use gradient checkpointing (would need to implement)
- Use a smaller image size: `--img_size 192` (instead of 224)

### Model Not Improving
- Check tensorboard logs
- Try different learning rates
- Ensure data augmentation is working
- Verify data paths are correct

## Next Steps

1. **✅ Train ResNet-34**: `bash train_resnet34.sh`
2. Compare validation loss with ResNet-18 baseline
3. Try attention fusion if results are promising
4. Run grid search with ResNet-34 to optimize hyperparameters
5. Generate test set predictions with best model

## Questions?

See the detailed guide: `docs/MODULAR_ARCHITECTURE_GUIDE.md`

