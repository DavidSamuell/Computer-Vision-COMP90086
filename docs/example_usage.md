# Example Usage - Complete Workflow

## 1. Training a Model

```bash
# Basic training
./run_training.sh

# Or with custom parameters
./run_training.sh --batch_size 32 --lr 5e-4 --epochs 50 --experiment_name my_best_model
```

This creates: `outputs/my_best_model/best_model.pth`

## 2. Running Test Inference

```bash
# Generate submission file
./run_test_inference.sh --model_path outputs/my_best_model/best_model.pth

# Or with custom output name
./run_test_inference.sh \
    --model_path outputs/my_best_model/best_model.pth \
    --output_path final_submission.csv
```

This creates: `final_submission.csv` in competition format

## 3. Grid Search (Optional)

```bash
# Quick grid search (6-12 hours)
cd src
python hyperparameter_search.py \
    --custom_grid ../quick_grid_search.json \
    --output_dir ../quick_grid_results \
    --max_epochs 30

# Full grid search (4-10 days)
./run_grid_search.sh
```

## 4. Analyze Grid Search Results

```bash
cd src
python analyze_grid_search.py \
    --results_file ../grid_search_results/grid_search_results.csv \
    --output_dir ../grid_analysis
```

## 5. Train Final Model with Best Hyperparameters

```bash
# Use best hyperparameters from grid search
cd src
python train.py \
    --lr 0.0001 \
    --dropout 0.4 \
    --weight_decay 0.0001 \
    --batch_size 16 \
    --seg_weight 0.5 \
    --num_epochs 100 \
    --experiment_name final_optimized_model
```

## 6. Generate Final Submission

```bash
./run_test_inference.sh \
    --model_path outputs/final_optimized_model/best_model.pth \
    --output_path competition_submission.csv
```

## Expected File Structure After Full Workflow

```
compvis/
├── outputs/
│   ├── my_best_model/
│   │   ├── best_model.pth
│   │   ├── config.json
│   │   └── tensorboard/
│   └── final_optimized_model/
│       ├── best_model.pth
│       └── ...
├── grid_search_results/
│   ├── grid_search_results.csv
│   └── run_XXXX/
└── competition_submission.csv    ← Final submission file
```
