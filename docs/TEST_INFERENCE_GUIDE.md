# Test Set Inference Guide

## Overview

The test inference script runs your trained model on the test set and generates a submission file in the exact format required by the competition.

## Features

✅ **Batch Inference**: Efficient batch processing of test samples
✅ **Error Handling**: Robust handling of corrupt/missing images
✅ **Complete Submissions**: Ensures all test samples are included
✅ **Validation**: Pre-validates all test images before inference
✅ **Statistics**: Provides prediction statistics and sample output
✅ **Flexible Output**: Customizable submission file name and location

## Quick Start

### Method 1: Using the Shell Script (Easiest)

```bash
# Basic usage with best model
./run_test_inference.sh --model_path outputs/my_experiment/best_model.pth

# Custom output file
./run_test_inference.sh \
    --model_path outputs/my_experiment/best_model.pth \
    --output_path my_submission.csv

# With custom batch size
./run_test_inference.sh \
    --model_path outputs/my_experiment/best_model.pth \
    --batch_size 64 \
    --output_path submission_batch64.csv
```

### Method 2: Direct Python Command

```bash
cd src
python test_inference.py \
    --model_path ../outputs/my_experiment/best_model.pth \
    --test_root ../Nutrition5K/test \
    --output_path ../submission.csv \
    --batch_size 32
```

## Command Line Arguments

### Required

- `--model_path`: Path to your trained model checkpoint (.pth file)

### Optional

- `--test_root`: Path to test data directory (default: `../Nutrition5K/test`)
- `--output_path`: Output submission file name (default: `submission.csv`)
- `--batch_size`: Batch size for inference (default: 32)
- `--img_size`: Image resize dimension (default: 224)
- `--num_workers`: Data loading workers (default: 4)
- `--device`: Device to use - cuda or cpu (default: cuda)

## Expected Output

### Console Output

```
================================================================================
TEST SET INFERENCE - NUTRITION5K CALORIE PREDICTION
================================================================================
Model: outputs/my_experiment/best_model.pth
Test data: ../Nutrition5K/test
Output: submission.csv
Batch size: 32
Image size: 224
================================================================================

Loading test dataset...
Found 189 test samples
Valid test samples: 189

Initializing inference engine...
Model loaded from: outputs/my_experiment/best_model.pth
Loaded model from epoch 45
Best validation loss: 234.5678

Running inference on 189 test samples...
Batch size: 32
------------------------------------------------------------
Inference: 100%|████████████| 6/6 [00:15<00:00,  2.58s/it]

Inference complete!
Successfully predicted: 189
Failed samples: 0

Creating submission file: submission.csv
Submission file saved: submission.csv
Total predictions: 189

Sample of submission file:
        ID   Value
0  dish_3301  245.67
1  dish_3302  189.23
2  dish_3303  567.89
3  dish_3304  123.45
4  dish_3305  432.10
...

Prediction Statistics:
  Min: 45.23
  Max: 789.56
  Mean: 234.56
  Std: 123.45
  Median: 198.76

================================================================================
INFERENCE COMPLETE!
================================================================================
Submission file ready: submission.csv
================================================================================
```

### Output Files

The script generates a CSV file in the exact competition format:

```csv
ID,Value
dish_3301,245.67
dish_3302,189.23
dish_3303,567.89
dish_3304,123.45
...
```

## Understanding the Process

### 1. Data Loading

- Loads all test images (RGB + depth) from the test directory
- Pre-validates images to identify corrupt/missing files
- Creates batches for efficient processing

### 2. Model Loading

- Loads your trained model checkpoint
- Sets model to evaluation mode (disables dropout)
- Displays model information (epoch, validation loss)

### 3. Inference

- Processes test images in batches
- Applies same preprocessing as training (resize, normalize)
- Generates calorie predictions (ignores segmentation output)
- Handles any failed samples gracefully

### 4. Submission File Creation

- Formats predictions in competition format
- Ensures all test samples are included
- Provides prediction statistics
- Saves to specified output file

## Test Data Structure

Your test data should be organized as:

```
Nutrition5K/test/
├── color/
│   ├── dish_3301/
│   │   └── rgb.png
│   ├── dish_3302/
│   │   └── rgb.png
│   └── ...
└── depth_raw/
    ├── dish_3301/
    │   └── depth_raw.png
    ├── dish_3302/
    │   └── depth_raw.png
    └── ...
```

## Different Model Types

### Use Best Model (Recommended)

```bash
./run_test_inference.sh --model_path outputs/my_experiment/best_model.pth
```

### Use Final Model

```bash
./run_test_inference.sh --model_path outputs/my_experiment/final_model.pth
```

### Use Specific Checkpoint

```bash
./run_test_inference.sh --model_path outputs/my_experiment/checkpoint_epoch_50.pth
```

## Batch Size Optimization

### For Speed (if you have sufficient GPU memory)

```bash
./run_test_inference.sh \
    --model_path outputs/my_experiment/best_model.pth \
    --batch_size 64
```

### For Memory Efficiency (if running out of GPU memory)

```bash
./run_test_inference.sh \
    --model_path outputs/my_experiment/best_model.pth \
    --batch_size 16
```

### For CPU Inference

```bash
./run_test_inference.sh \
    --model_path outputs/my_experiment/best_model.pth \
    --device cpu \
    --batch_size 8
```

## Error Handling

The script handles various error conditions:

### Missing Images

- **Issue**: Some test samples have missing RGB or depth images
- **Handling**: Logs warning, sets prediction to 0.0, continues processing

### Corrupt Images

- **Issue**: Images are corrupted and can't be loaded
- **Handling**: Logs warning, sets prediction to 0.0, continues processing

### Model Loading Errors

- **Issue**: Model checkpoint is incompatible or corrupted
- **Solution**: Check model path, ensure it's from the same architecture

### Out of Memory

- **Issue**: GPU runs out of memory during inference
- **Solution**: Reduce `--batch_size` or use `--device cpu`

## Quality Checks

### Check Submission Format

```bash
head -10 submission.csv
# Should show: ID,Value format with dish_XXXX IDs
```

### Check Number of Predictions

```bash
wc -l submission.csv
# Should show: 190 (189 test samples + 1 header)
```

### Check for Missing Predictions

```bash
grep -c "0.0" submission.csv
# High count might indicate many failed samples
```

### Validate Prediction Range

Look at the statistics printed by the script:

- **Reasonable range**: 50-800 calories is typical for food
- **Suspicious values**: <10 or >2000 calories might indicate issues

## Multiple Model Ensemble

For better performance, you can create predictions from multiple models:

```bash
# Model 1
./run_test_inference.sh \
    --model_path outputs/experiment1/best_model.pth \
    --output_path submission1.csv

# Model 2
./run_test_inference.sh \
    --model_path outputs/experiment2/best_model.pth \
    --output_path submission2.csv

# Model 3
./run_test_inference.sh \
    --model_path outputs/experiment3/best_model.pth \
    --output_path submission3.csv
```

Then average the predictions:

```python
import pandas as pd

# Load predictions
df1 = pd.read_csv('submission1.csv')
df2 = pd.read_csv('submission2.csv')
df3 = pd.read_csv('submission3.csv')

# Average predictions
df_ensemble = df1.copy()
df_ensemble['Value'] = (df1['Value'] + df2['Value'] + df3['Value']) / 3

# Save ensemble
df_ensemble.to_csv('submission_ensemble.csv', index=False)
```

## Troubleshooting

### Problem: "Model not found"

**Solution**: Check the model path exists:

```bash
ls -la outputs/my_experiment/best_model.pth
```

### Problem: "Test directory not found"

**Solution**: Verify test data location:

```bash
ls -la Nutrition5K/test/color/ | head
```

### Problem: "CUDA out of memory"

**Solution**: Reduce batch size:

```bash
./run_test_inference.sh \
    --model_path outputs/my_experiment/best_model.pth \
    --batch_size 8
```

### Problem: Predictions all seem too low/high

**Solution**:

1. Check if you're using the right model (best_model.pth vs others)
2. Verify the model was trained properly (check training logs)
3. Compare predictions with training set statistics

### Problem: Many samples have 0.0 predictions

**Solution**: Check for corrupt test images:

```bash
# The script will print warnings about corrupt images
# Look for lines like "Corrupt image for dish_XXXX"
```

## Final Submission Checklist

Before submitting:

✅ **Format Check**: CSV has "ID,Value" header
✅ **Sample Count**: 189 test samples (plus header = 190 lines)
✅ **ID Format**: All IDs are "dish_XXXX" format  
✅ **Value Range**: Predictions in reasonable calorie range (50-800)
✅ **No Missing**: All test dish IDs are present
✅ **No Errors**: No obvious prediction errors (like all zeros)

```bash
# Quick validation
echo "Lines in submission: $(wc -l < submission.csv)"
echo "Unique IDs: $(tail -n +2 submission.csv | cut -d, -f1 | sort -u | wc -l)"
echo "Min prediction: $(tail -n +2 submission.csv | cut -d, -f2 | sort -n | head -1)"
echo "Max prediction: $(tail -n +2 submission.csv | cut -d, -f2 | sort -n | tail -1)"
```

## Performance Tips

### For Fastest Inference

- Use GPU with large batch size (64-128)
- Use multiple workers (8-16)
- Use best_model.pth (usually fastest to load)

### For Most Accurate Predictions

- Use the model with lowest validation loss
- Consider ensemble of multiple models
- Ensure test preprocessing matches training exactly

### For Memory-Constrained Environments

- Use smaller batch sizes (8-16)
- Use CPU if GPU memory is insufficient
- Reduce number of workers if system RAM is limited

Your test inference system is now ready to generate competition submissions! 🚀
