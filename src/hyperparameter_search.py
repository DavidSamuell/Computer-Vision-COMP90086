"""
Hyperparameter Tuning with Grid Search for Calorie Prediction Model

This script performs exhaustive grid search over hyperparameter combinations
to find the optimal configuration for the calorie prediction model.
"""

import os
import json
import argparse
import itertools
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import Nutrition5KDataset, create_train_val_split
from model import MultiStreamCaloriePredictor, build_model
from train import MultiTaskLoss, EarlyStopping, Trainer


class GridSearcher:
    """Grid search for hyperparameter tuning"""
    
    def __init__(
        self,
        param_grid,
        data_root,
        csv_path,
        output_dir,
        device='cuda',
        val_ratio=0.15,
        img_size=224,
        num_workers=4,
        seed=42,
        max_epochs=50,
        early_stopping_patience=10
    ):
        """
        Args:
            param_grid: Dictionary of hyperparameters to search
            data_root: Path to training data
            csv_path: Path to training CSV
            output_dir: Output directory for results
            device: Device to use (cuda/cpu)
            val_ratio: Validation split ratio
            img_size: Image size
            num_workers: Number of data loading workers
            seed: Random seed
            max_epochs: Maximum epochs per configuration
            early_stopping_patience: Early stopping patience
        """
        self.param_grid = param_grid
        self.data_root = data_root
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.val_ratio = val_ratio
        self.img_size = img_size
        self.num_workers = num_workers
        self.seed = seed
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Results tracking
        self.results_file = os.path.join(output_dir, 'grid_search_results.csv')
        self.best_config_file = os.path.join(output_dir, 'best_config.json')
        
        # Load existing results if resuming
        self.results_df = self._load_existing_results()
        
        # Generate all parameter combinations
        self.param_combinations = self._generate_combinations()
        
        print(f"Grid Search Configuration:")
        print(f"  Total combinations: {len(self.param_combinations)}")
        print(f"  Max epochs per run: {max_epochs}")
        print(f"  Output directory: {output_dir}")
        print(f"  Device: {self.device}")
        print(f"  Already completed: {len(self.results_df)}")
        print()
    
    def _load_existing_results(self):
        """Load existing results if resuming"""
        if os.path.exists(self.results_file):
            return pd.read_csv(self.results_file)
        else:
            return pd.DataFrame()
    
    def _generate_combinations(self):
        """Generate all hyperparameter combinations"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _is_completed(self, params):
        """Check if this configuration has already been run"""
        if self.results_df.empty:
            return False
        
        # Create a string representation of params for comparison
        for _, row in self.results_df.iterrows():
            match = True
            for key, value in params.items():
                if key in row and row[key] != value:
                    match = False
                    break
            if match:
                return True
        
        return False
    
    def _prepare_data(self):
        """Prepare data loaders"""
        # Create train/val split (only once)
        train_csv, val_csv = create_train_val_split(
            self.csv_path,
            val_ratio=self.val_ratio,
            random_seed=self.seed
        )
        
        # Create datasets
        train_dataset = Nutrition5KDataset(
            csv_path=train_csv,
            data_root=self.data_root,
            split='train',
            augment=True,
            img_size=self.img_size
        )
        
        val_dataset = Nutrition5KDataset(
            csv_path=val_csv,
            data_root=self.data_root,
            split='val',
            augment=False,
            img_size=self.img_size
        )
        
        return train_dataset, val_dataset
    
    def _train_single_config(self, params, run_id):
        """Train a single configuration"""
        print(f"\n{'='*80}")
        print(f"Configuration {run_id + 1}/{len(self.param_combinations)}")
        print(f"{'='*80}")
        print("Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print(f"{'='*80}\n")
        
        # Set random seed
        torch.manual_seed(self.seed)
        
        # Prepare data
        train_dataset, val_dataset = self._prepare_data()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Create model (use modular system if encoder is specified)
        if 'encoder' in params:
            model = build_model(
                encoder=params.get('encoder', 'resnet18'),
                fusion=params.get('fusion', 'middle'),
                regression_head=params.get('regression_head', 'standard'),
                segmentation_head=params.get('segmentation_head', 'standard'),
                pretrained=False,
                dropout_rate=params['dropout'],
                fusion_channels=params.get('fusion_channels', 512)
            )
        else:
            # Fallback to original model for backward compatibility
            model = MultiStreamCaloriePredictor(
                pretrained=False,
                dropout_rate=params['dropout'],
                fusion_channels=512
            )
        model = model.to(self.device)
        
        # Create loss function
        criterion = MultiTaskLoss(
            calorie_weight=params['calorie_weight'],
            seg_weight=params['seg_weight']
        )
        
        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )
        
        # Create scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Create output directory for this run
        run_dir = os.path.join(self.output_dir, f'run_{run_id:04d}')
        os.makedirs(run_dir, exist_ok=True)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            output_dir=run_dir,
            early_stopping_patience=self.early_stopping_patience
        )
        
        # Train
        try:
            trainer.train(self.max_epochs)
            
            # Get best validation loss
            best_val_loss = trainer.best_val_loss
            
            # Record results
            result = {
                'run_id': run_id,
                'best_val_loss': best_val_loss,
                **params
            }
            
            return result, True
            
        except Exception as e:
            print(f"\n[ERROR] Training failed: {e}")
            result = {
                'run_id': run_id,
                'best_val_loss': float('inf'),
                'error': str(e),
                **params
            }
            return result, False
    
    def search(self):
        """Run grid search"""
        print(f"\n{'='*80}")
        print(f"Starting Grid Search")
        print(f"{'='*80}\n")
        
        start_time = datetime.now()
        
        for run_id, params in enumerate(self.param_combinations):
            # Skip if already completed
            if self._is_completed(params):
                print(f"Skipping run {run_id + 1}/{len(self.param_combinations)} (already completed)")
                continue
            
            # Train this configuration
            result, success = self._train_single_config(params, run_id)
            
            # Save result
            self._save_result(result)
            
            # Update best configuration
            self._update_best_config()
        
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"Grid Search Complete!")
        print(f"{'='*80}")
        print(f"Total time: {elapsed}")
        print(f"Results saved to: {self.results_file}")
        print(f"Best configuration saved to: {self.best_config_file}")
        print(f"{'='*80}\n")
        
        self._print_summary()
    
    def _save_result(self, result):
        """Save a single result"""
        # Append to results
        result_df = pd.DataFrame([result])
        
        if os.path.exists(self.results_file):
            result_df.to_csv(self.results_file, mode='a', header=False, index=False)
        else:
            result_df.to_csv(self.results_file, index=False)
        
        # Reload results
        self.results_df = pd.read_csv(self.results_file)
    
    def _update_best_config(self):
        """Update best configuration"""
        if self.results_df.empty:
            return
        
        # Find best configuration
        best_row = self.results_df.loc[self.results_df['best_val_loss'].idxmin()]
        
        # Save best config
        best_config = best_row.to_dict()
        with open(self.best_config_file, 'w') as f:
            json.dump(best_config, f, indent=4)
    
    def _print_summary(self):
        """Print summary of results"""
        if self.results_df.empty:
            print("No results to summarize.")
            return
        
        print("Top 5 Configurations:")
        print("-" * 80)
        
        # Sort by validation loss
        sorted_df = self.results_df.sort_values('best_val_loss')
        
        for idx, row in sorted_df.head(5).iterrows():
            print(f"\nRank {idx + 1}:")
            print(f"  Val Loss: {row['best_val_loss']:.4f}")
            print(f"  LR: {row['lr']}, Dropout: {row['dropout']}, Weight Decay: {row['weight_decay']}")
            print(f"  Batch Size: {row['batch_size']}")
            print(f"  Loss Weights - Calorie: {row['calorie_weight']}, Seg: {row['seg_weight']}")
        
        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Grid Search')
    
    # Data
    parser.add_argument('--data_root', type=str, default='../Nutrition5K/train',
                        help='Path to training data directory')
    parser.add_argument('--csv_path', type=str, default='../Nutrition5K/nutrition5k_train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation split ratio')
    
    # Grid search settings
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum epochs per configuration')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='../grid_search_results',
                        help='Output directory for grid search results')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    # Custom grid (optional - will use defaults if not specified)
    parser.add_argument('--custom_grid', type=str, default=None,
                        help='Path to JSON file with custom parameter grid')
    
    # Model architecture (optional - for testing different architectures)
    parser.add_argument('--encoder', type=str, default=None,
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Encoder architecture (overrides grid if specified)')
    parser.add_argument('--fusion', type=str, default=None,
                        choices=['middle', 'middle_attention', 'additive'],
                        help='Fusion strategy (overrides grid if specified)')
    
    args = parser.parse_args()
    
    # Define parameter grid
    if args.custom_grid and os.path.exists(args.custom_grid):
        with open(args.custom_grid, 'r') as f:
            param_grid = json.load(f)
        # Filter out comment fields (keys starting with _)
        param_grid = {k: v for k, v in param_grid.items() if not k.startswith('_')}
        print(f"Loaded custom parameter grid from {args.custom_grid}")
    else:
        # Default parameter grid
        param_grid = {
            'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
            'dropout': [0.3, 0.4, 0.5],
            'weight_decay': [1e-5, 1e-4, 5e-4],
            'batch_size': [8, 16, 32],
            'calorie_weight': [1.0],
            'seg_weight': [0.3, 0.5, 0.7]
        }
    
    # Add architecture parameters if specified via command line
    if args.encoder is not None:
        param_grid['encoder'] = [args.encoder]
    if args.fusion is not None:
        param_grid['fusion'] = [args.fusion]
    
    print("\nParameter Grid:")
    print("-" * 80)
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    print("-" * 80)
    
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    
    print(f"\nTotal combinations to test: {total_combinations}")
    print(f"Estimated time: {total_combinations * 0.5:.1f} - {total_combinations * 1.5:.1f} hours")
    print("(assuming 30-90 minutes per configuration with early stopping)")
    
    response = input("\nProceed with grid search? (yes/no): ")
    if response.lower() != 'yes':
        print("Grid search cancelled.")
        return
    
    # Create grid searcher
    searcher = GridSearcher(
        param_grid=param_grid,
        data_root=args.data_root,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        device=args.device,
        val_ratio=args.val_ratio,
        img_size=224,
        num_workers=args.num_workers,
        seed=args.seed,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Run search
    searcher.search()


if __name__ == '__main__':
    main()

