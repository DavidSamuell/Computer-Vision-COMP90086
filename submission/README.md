# Nutrition5K Calorie Prediction - Setup and Execution Guide

This repository contains three Jupyter notebooks for calorie prediction experiments on the Nutrition5K dataset.

## Prerequisites

- Anaconda or Miniconda installed
- Kaggle API credentials configured (for dataset download)
- CUDA-compatible GPU (recommended)

## Setup Instructions

### 1. Create Conda Environment

```bash
conda create -n compvis python=3.10 -y
conda activate compvis
```

### 2. Install Required Packages

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install additional dependencies
pip install pandas matplotlib pillow tqdm tensorboard kaggle
```

**Required packages:**
- `torch` (with CUDA support)
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `pillow` (PIL)
- `tqdm`
- `tensorboard`
- `kaggle` (for dataset download)

### 3. Download Dataset

Navigate to the submission directory and download the Nutrition5K dataset from Kaggle:

```bash
cd /path/to/Computer-Vision-COMP90086/submission
kaggle competitions download -c comp-90086-nutrition-5-k
unzip comp-90086-nutrition-5-k.zip -d ./Nutrition5K
```

**Expected directory structure:**
```

baseline.ipynb
resnet_experiments.ipynb
inception_v3_experiments.ipynb
Nutrition5K/
└── Nutrition5K/
    ├── train/
    │   ├── color/
    │   └── depth_raw/
    ├── test/
    │   ├── color/
    │   └── depth_raw/
    ├── nutrition5k_train.csv
    └── nutrition5k_test.csv
```



## Running the Notebooks

1. Open Jupyter Lab or Jupyter Notebook:
   ```bash
   conda activate compvis
   jupyter lab
   # or
   jupyter notebook
   ```

2. Navigate to the notebook you want to run

3. **Select the kernel:** Kernel → Change Kernel → `compvis`

4. **Run all cells:** Cell → Run All

## Notes

- Training requires a CUDA-compatible GPU (runs on CPU but significantly slower)
- Each notebook creates an `experiments/` directory to save model checkpoints and logs
- Models automatically use early stopping to prevent overfitting
- The InceptionV3 notebook includes test inference code that generates `submission.csv`

## Expected Outputs
Each experiment saves:
- `best_model.pth` - Model checkpoint with best validation performance
- `tensorboard/` - TensorBoard logs for training visualization
- `config.json` - Experiment configuration (InceptionV3 experiments only)

