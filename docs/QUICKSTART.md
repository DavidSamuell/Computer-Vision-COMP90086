# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Verify Dataset Structure

Make sure your dataset is organized like this:

```
compvis/
├── Nutrition5K/
│   ├── nutrition5k_train.csv
│   └── train/
│       ├── color/
│       │   ├── dish_0000/
│       │   │   └── rgb.png
│       │   ├── dish_0001/
│       │   │   └── rgb.png
│       │   └── ...
│       └── depth_raw/
│           ├── dish_0000/
│           │   └── depth_raw.png
│           ├── dish_0001/
│           │   └── depth_raw.png
│           └── ...
└── src/
    ├── dataset.py
    ├── model.py
    └── train.py
```

## 3. Test the Components (Optional)

Test the dataset loader:

```bash
cd src
python dataset.py
```

Test the model architecture:

```bash
python model.py
```

## 4. Start Training

### Option A: Using the shell script (easiest)

```bash
./run_training.sh
```

Or with custom parameters:

```bash
./run_training.sh --batch_size 32 --lr 5e-4 --epochs 50 --experiment_name my_first_run
```

### Option B: Direct Python command

```bash
cd src
python train.py \
    --data_root ../Nutrition5K/train \
    --csv_path ../Nutrition5K/nutrition5k_train.csv \
    --batch_size 16 \
    --num_epochs 100 \
    --lr 1e-4 \
    --experiment_name my_experiment
```

## 5. Monitor Training

Open a new terminal and run:

```bash
tensorboard --logdir outputs/<your_experiment_name>/tensorboard
```

Then open http://localhost:6006 in your browser.

## 6. Understand the Output

Training creates:

- `outputs/<experiment_name>/best_model.pth` - Best model (use this for inference)
- `outputs/<experiment_name>/config.json` - Training configuration
- `outputs/<experiment_name>/tensorboard/` - Training logs

## Key Features

✅ **Handles corrupt images automatically** - Validates all images at startup and skips bad ones
✅ **Multi-task learning** - Learns both calorie prediction AND food segmentation
✅ **Smart regularization** - Dropout, weight decay, and early stopping prevent overfitting
✅ **Data augmentation** - Aggressive augmentation to expand limited training data
✅ **Middle fusion** - Efficiently combines RGB and depth information

## Troubleshooting

**"No module named torch"**: Run `pip install -r requirements.txt`

**"Directory does not exist"**: Check that your dataset paths match the structure above

**"Out of memory"**: Reduce `--batch_size` to 8 or 4

**Training is too slow**: Increase `--batch_size` or reduce `--img_size`

**Model is overfitting** (train loss << val loss):

- Increase `--dropout` to 0.5 or 0.6
- Increase `--weight_decay` to 5e-4 or 1e-3
- Reduce `--seg_weight` to focus more on the main task

**Model is underfitting** (both losses high):

- Decrease `--dropout` to 0.2 or 0.3
- Decrease `--weight_decay` to 1e-5
- Train longer (increase `--num_epochs`)
- Check that images are loading correctly

## Expected Training Time

- **CPU**: ~2-4 hours per epoch (not recommended)
- **GPU (RTX 3060)**: ~5-10 minutes per epoch
- **GPU (RTX 4090)**: ~2-3 minutes per epoch

With early stopping, training typically completes in 30-60 epochs.

## Next Steps

After training completes:

1. Check TensorBoard for training curves
2. Find your best model at `outputs/<experiment_name>/best_model.pth`
3. Use the model for inference on test data
4. Experiment with different hyperparameters if performance isn't satisfactory

## Tips for Best Results

1. **Start with defaults** - The default hyperparameters are well-tuned
2. **Monitor validation loss** - If it plateaus early, your model might need more capacity
3. **Check augmentation** - Run `python dataset.py` to visualize augmented samples
4. **Tune loss weights** - If segmentation is too hard, decrease `--seg_weight`
5. **Be patient** - Training from scratch takes time, but early stopping will prevent wasting compute
