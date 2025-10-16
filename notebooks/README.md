# Jupyter Notebooks

Interactive notebooks for training and experimentation.

## Available Notebooks

### 1. `train_resnet34.ipynb` - Quick Training with ResNet-34

**Purpose:** Train models interactively with different architectures.

**Features:**
- Quick configuration changes
- Live training progress
- Visualization of data samples
- Instant architecture experimentation
- Integrated with modular architecture system

**Usage:**
```bash
cd notebooks
jupyter notebook train_resnet34.ipynb
```

Then just run all cells to train ResNet-34 with best config!

**Quick Experiments:**

Modify cell 2 to try different architectures:

```python
# Try ResNet-50
ENCODER = 'resnet50'

# Try attention fusion
FUSION = 'middle_attention'

# Try deep regression head
REGRESSION_HEAD = 'deep'
```

Then rerun from cell 2 onwards.

## Notebook Structure

1. **Setup** - Import libraries
2. **Configuration** - Easy-to-modify settings
3. **Show Components** - List available architectures
4. **Load Data** - Create datasets and visualize samples
5. **Create Loaders** - Setup data loaders
6. **Build Model** - Construct model with modular system
7. **Setup Training** - Loss, optimizer, scheduler
8. **Train** - Run training with progress bars
9. **View Results** - Summary and TensorBoard info
10. **Test Model** - Quick inference test

## Tips

- **Fast iteration**: Only rerun from configuration cell when changing architecture
- **Monitor training**: Use TensorBoard for detailed curves
- **Save experiments**: Each run creates a timestamped output directory
- **Compare models**: Train multiple configurations and compare results

## Requirements

Make sure you have Jupyter installed in your conda environment:

```bash
conda activate compvis
conda install jupyter notebook ipywidgets
```

Or:

```bash
pip install jupyter notebook ipywidgets
```

## Starting Jupyter

From the project root:

```bash
jupyter notebook notebooks/
```

Or from the notebooks directory:

```bash
cd notebooks
jupyter notebook
```

## Output

Each training run saves to:
```
../outputs/notebook_{encoder}_{fusion}_{timestamp}/
├── best_model.pth
├── final_model.pth
├── checkpoint_epoch_*.pth
├── config.json
└── tensorboard/
```

## Next Steps

After training:
1. View results in TensorBoard
2. Generate test predictions with `test_inference.py`
3. Compare with other models using `compare_architectures.py`

