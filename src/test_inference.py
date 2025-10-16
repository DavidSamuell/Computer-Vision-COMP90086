"""
Test Set Inference Script for Nutrition5K Calorie Prediction
Generates submission file in the required format
"""

import os
import argparse
import pandas as pd
import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import Normalize
from tqdm import tqdm
import warnings

from model import MultiStreamCaloriePredictor, build_model


class TestDataset(Dataset):
    """Dataset class for test set inference"""
    
    def __init__(self, test_root, img_size=224):
        """
        Args:
            test_root: Root directory containing test data (e.g., '../Nutrition5K/test')
            img_size: Target image size for resizing
        """
        self.test_root = test_root
        self.img_size = img_size
        
        # Paths to subdirectories
        self.color_dir = os.path.join(test_root, 'color')
        self.depth_raw_dir = os.path.join(test_root, 'depth_raw')
        
        # Get all dish IDs from color directory
        self.dish_ids = []
        if os.path.exists(self.color_dir):
            self.dish_ids = sorted([d for d in os.listdir(self.color_dir) 
                                  if os.path.isdir(os.path.join(self.color_dir, d))])
        else:
            raise FileNotFoundError(f"Test color directory not found: {self.color_dir}")
        
        print(f"Found {len(self.dish_ids)} test samples")
        
        # Color normalization (same as training)
        self.color_normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Validate test data
        self.valid_indices = self._validate_test_data()
        print(f"Valid test samples: {len(self.valid_indices)}")
    
    def _validate_test_data(self):
        """Validate test data and return valid indices"""
        valid_indices = []
        
        for idx, dish_id in enumerate(self.dish_ids):
            rgb_path = os.path.join(self.color_dir, dish_id, 'rgb.png')
            depth_path = os.path.join(self.depth_raw_dir, dish_id, 'depth_raw.png')
            
            # Check if files exist
            if not os.path.exists(rgb_path):
                warnings.warn(f"Missing RGB image: {rgb_path}")
                continue
            if not os.path.exists(depth_path):
                warnings.warn(f"Missing depth image: {depth_path}")
                continue
            
            # Try to load images to check for corruption
            try:
                with Image.open(rgb_path) as img:
                    img.verify()
                with Image.open(depth_path) as img:
                    img.verify()
                valid_indices.append(idx)
            except Exception as e:
                warnings.warn(f"Corrupt image for {dish_id}: {e}")
                continue
        
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def _load_image_safe(self, path, mode='RGB'):
        """Safely load an image, return None if corrupt"""
        try:
            img = Image.open(path).convert(mode)
            return img
        except Exception as e:
            warnings.warn(f"Failed to load image {path}: {e}")
            return None
    
    def __getitem__(self, idx):
        """
        Returns:
            rgb_tensor: (3, H, W) RGB image
            depth_tensor: (1, H, W) raw depth image
            dish_id: string - dish ID for this sample
        """
        actual_idx = self.valid_indices[idx]
        dish_id = self.dish_ids[actual_idx]
        
        # Load images
        rgb_path = os.path.join(self.color_dir, dish_id, 'rgb.png')
        depth_path = os.path.join(self.depth_raw_dir, dish_id, 'depth_raw.png')
        
        rgb_img = self._load_image_safe(rgb_path, mode='RGB')
        depth_img = self._load_image_safe(depth_path, mode='L')
        
        # Handle corrupt images (shouldn't happen after validation, but just in case)
        if rgb_img is None or depth_img is None:
            # Return zeros - will be flagged in results
            rgb_tensor = torch.zeros(3, self.img_size, self.img_size)
            depth_tensor = torch.zeros(1, self.img_size, self.img_size)
            return {
                'rgb': rgb_tensor,
                'depth': depth_tensor,
                'dish_id': dish_id,
                'is_valid': False
            }
        
        # Resize images (no augmentation for test)
        rgb_img = TF.resize(rgb_img, (self.img_size, self.img_size))
        depth_img = TF.resize(depth_img, (self.img_size, self.img_size))
        
        # Convert to tensors
        rgb_tensor = TF.to_tensor(rgb_img)  # (3, H, W)
        depth_tensor = TF.to_tensor(depth_img)  # (1, H, W)
        
        # Normalize RGB
        rgb_tensor = self.color_normalize(rgb_tensor)
        
        # Normalize depth to [0, 1] range
        depth_tensor = depth_tensor / 65535.0 if depth_tensor.max() > 1.0 else depth_tensor
        
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'dish_id': dish_id,
            'is_valid': True
        }
    
    def get_all_dish_ids(self):
        """Get all dish IDs (including invalid ones) for complete submission"""
        return self.dish_ids


class TestInference:
    """Test inference manager"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use for inference
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
        print(f"Using device: {self.device}")
    
    def _load_model(self):
        """Load trained model from checkpoint"""
        # Try to load config file from the model directory
        model_dir = os.path.dirname(self.model_path)
        config_path = os.path.join(model_dir, 'config.json')
        
        if os.path.exists(config_path):
            print(f"Loading model config from: {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create model using the build_model function with saved config
            model = build_model(
                encoder=config.get('encoder', 'resnet18'),
                fusion=config.get('fusion', 'middle'),
                regression_head=config.get('regression_head', 'standard'),
                segmentation_head=config.get('segmentation_head', 'standard'),
                pretrained=False,
                dropout_rate=config.get('dropout', 0.4),
                fusion_channels=config.get('fusion_channels', 512)
            )
            print(f"Model architecture: {config.get('encoder', 'resnet18')} + {config.get('fusion', 'middle')} fusion")
        else:
            print(f"Warning: Config file not found at {config_path}")
            print("Using default ResNet-18 architecture")
            # Fallback to default model
            model = MultiStreamCaloriePredictor(
                pretrained=False,
                dropout_rate=0.4
            )
        
        # Load checkpoint
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'best_val_loss' in checkpoint:
                print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        return model.to(self.device)
    
    @torch.no_grad()
    def predict_test_set(self, test_dataset, batch_size=32, num_workers=4):
        """
        Run inference on entire test set
        
        Args:
            test_dataset: TestDataset object
            batch_size: Batch size for inference
            num_workers: Number of data loading workers
        
        Returns:
            predictions: Dictionary mapping dish_id to predicted calories
        """
        print(f"\nRunning inference on {len(test_dataset)} test samples...")
        print(f"Batch size: {batch_size}")
        print("-" * 60)
        
        # Create data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        predictions = {}
        failed_samples = []
        
        # Run inference
        for batch in tqdm(test_loader, desc="Inference"):
            # Move to device
            rgb = batch['rgb'].to(self.device)
            depth = batch['depth'].to(self.device)
            dish_ids = batch['dish_id']
            is_valid = batch['is_valid']
            
            # Forward pass (only need calorie predictions, ignore segmentation)
            calorie_pred, _ = self.model(rgb, depth)
            
            # Convert predictions to CPU and store
            calorie_pred = calorie_pred.cpu().numpy().flatten()
            
            for i, dish_id in enumerate(dish_ids):
                if is_valid[i]:
                    predictions[dish_id] = float(calorie_pred[i])
                else:
                    # Flag failed samples
                    failed_samples.append(dish_id)
                    predictions[dish_id] = 0.0  # Default prediction for failed samples
        
        if failed_samples:
            print(f"\nWarning: {len(failed_samples)} samples failed to load properly")
            print("These samples will have prediction = 0.0")
        
        print(f"\nInference complete!")
        print(f"Successfully predicted: {len(predictions) - len(failed_samples)}")
        print(f"Failed samples: {len(failed_samples)}")
        
        return predictions, failed_samples
    
    def create_submission_file(self, predictions, all_dish_ids, output_path):
        """
        Create submission CSV file in the required format
        
        Args:
            predictions: Dictionary of predictions
            all_dish_ids: List of all dish IDs (to ensure completeness)
            output_path: Path to save submission file
        """
        print(f"\nCreating submission file: {output_path}")
        
        # Prepare submission data
        submission_data = []
        missing_predictions = []
        
        for dish_id in sorted(all_dish_ids):
            if dish_id in predictions:
                prediction = predictions[dish_id]
            else:
                # Handle missing predictions
                prediction = 0.0
                missing_predictions.append(dish_id)
            
            submission_data.append({
                'ID': dish_id,
                'Value': prediction
            })
        
        # Create DataFrame
        submission_df = pd.DataFrame(submission_data)
        
        # Save to CSV
        submission_df.to_csv(output_path, index=False)
        
        print(f"Submission file saved: {output_path}")
        print(f"Total predictions: {len(submission_data)}")
        if missing_predictions:
            print(f"Missing predictions (set to 0.0): {len(missing_predictions)}")
        
        # Print sample of submission
        print(f"\nSample of submission file:")
        print(submission_df.head(10))
        
        # Print statistics
        values = submission_df['Value'].values
        print(f"\nPrediction Statistics:")
        print(f"  Min: {values.min():.2f}")
        print(f"  Max: {values.max():.2f}")
        print(f"  Mean: {values.mean():.2f}")
        print(f"  Std: {values.std():.2f}")
        print(f"  Median: {np.median(values):.2f}")
        
        return submission_df


class EnsembleInference:
    """Ensemble inference engine for multiple models"""
    
    def __init__(self, model_paths, device='cuda'):
        """
        Initialize ensemble with multiple models
        
        Args:
            model_paths: List of paths to model checkpoints
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.model_paths = model_paths
        
        print(f"Loading {len(model_paths)} models for ensemble...")
        
        for i, model_path in enumerate(model_paths, 1):
            print(f"  Loading model {i}/{len(model_paths)}: {os.path.basename(model_path)}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model config
            if 'config' in checkpoint:
                config = checkpoint['config']
                print(f"    Config: encoder={config.get('encoder', 'resnet18')}, fusion={config.get('fusion', 'middle')}, use_seg={config.get('use_segmentation', False)}")
                model = build_model(
                    encoder=config.get('encoder', 'resnet18'),
                    fusion=config.get('fusion', 'middle'),
                    regression_head=config.get('regression_head', 'standard'),
                    segmentation_head=config.get('segmentation_head', 'standard'),
                    pretrained=False,
                    dropout_rate=config.get('dropout', 0.4),  # Fix: use 'dropout' not 'dropout_rate'
                    fusion_channels=config.get('fusion_channels', 512),
                    use_segmentation=config.get('use_segmentation', False)
                )
            else:
                # Try to infer from checkpoint keys
                state_dict_keys = list(checkpoint['model_state_dict'].keys())
                has_segmentation = any('segmentation_head' in key for key in state_dict_keys)
                has_inception = any('branch1x1' in key or 'branch3x3' in key for key in state_dict_keys)
                
                print(f"    Warning: No config found, inferring from state_dict")
                print(f"    Detected: segmentation={has_segmentation}, inception_fusion={has_inception}")
                
                # Build model based on detected architecture
                fusion_type = 'inception' if has_inception else 'middle'
                model = build_model(
                    encoder='resnet18',
                    fusion=fusion_type,
                    regression_head='standard',
                    segmentation_head='standard',
                    pretrained=False,
                    dropout_rate=0.4,
                    fusion_channels=512,
                    use_segmentation=has_segmentation
                )
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            self.models.append(model)
        
        print(f"✓ Ensemble ready with {len(self.models)} models")
    
    def predict_batch(self, rgb_batch, depth_batch):
        """
        Predict on a batch using ensemble averaging
        
        Args:
            rgb_batch: RGB images tensor (B, 3, H, W) - already on device
            depth_batch: Depth images tensor (B, 1, H, W) - already on device
        
        Returns:
            Ensemble averaged predictions (B, 1)
        """
        
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                # Get prediction from this model
                if hasattr(model, 'forward'):
                    calorie_pred, _ = model(rgb_batch, depth_batch)
                else:
                    calorie_pred = model(rgb_batch, depth_batch)
                
                predictions.append(calorie_pred)
        
        # Average predictions across all models
        ensemble_pred = torch.stack(predictions, dim=0).mean(dim=0)
        
        return ensemble_pred
    
    def predict_test_set(self, test_dataset, batch_size=32, num_workers=4):
        """
        Run ensemble inference on entire test set
        
        Args:
            test_dataset: TestDataset instance
            batch_size: Batch size for inference
            num_workers: Number of data loading workers
        
        Returns:
            predictions: Dict mapping dish_id to predicted calories
            failed_samples: List of dish_ids that failed
        """
        dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        predictions = {}
        failed_samples = []
        
        print(f"\nRunning ensemble inference on {len(test_dataset)} samples...")
        print(f"Using {len(self.models)} models for averaging")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Ensemble Inference")):
                try:
                    # Move to device
                    rgb_batch = batch['rgb'].to(self.device)
                    depth_batch = batch['depth'].to(self.device)
                    dish_ids = batch['dish_id']
                    is_valid = batch['is_valid']
                    
                    # Get ensemble predictions
                    pred_batch = self.predict_batch(rgb_batch, depth_batch)
                    
                    # Convert predictions to CPU and store
                    pred_batch = pred_batch.cpu().numpy().flatten()
                    
                    # Store predictions
                    for i, dish_id in enumerate(dish_ids):
                        if is_valid[i]:
                            predictions[dish_id] = float(pred_batch[i])
                        else:
                            # Flag failed samples
                            failed_samples.append(dish_id)
                            predictions[dish_id] = 0.0  # Default prediction for failed samples
                        
                except Exception as e:
                    print(f"\nWarning: Batch {batch_idx} failed: {e}")
                    # Add failed samples with default prediction (if dish_ids is available)
                    if 'dish_id' in batch:
                        for dish_id in batch['dish_id']:
                            predictions[dish_id] = 0.0
                            failed_samples.append(dish_id)
        
        print(f"✓ Ensemble inference complete: {len(predictions)} predictions")
        return predictions, failed_samples
    
    def create_submission_file(self, predictions, all_dish_ids, output_path):
        """Create submission CSV file from ensemble predictions"""
        submission_data = []
        
        for dish_id in all_dish_ids:
            calorie_pred = predictions.get(dish_id, 0.0)
            submission_data.append({
                'ID': dish_id,
                'Value': max(0.0, calorie_pred)  # Ensure non-negative
            })
        
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(output_path, index=False)
        
        print(f"✓ Ensemble submission saved: {output_path}")
        print(f"  Samples: {len(submission_df)}")
        print(f"  Avg prediction: {submission_df['Value'].mean():.1f} kcal")
        print(f"  Min prediction: {submission_df['Value'].min():.1f} kcal")
        print(f"  Max prediction: {submission_df['Value'].max():.1f} kcal")
        
        return submission_df


def main():
    parser = argparse.ArgumentParser(description='Test Set Inference for Calorie Prediction')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=False,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--model_paths', type=str, nargs='+', required=False,
                        help='Paths to multiple model checkpoints for ensemble inference')
    parser.add_argument('--test_root', type=str, required=True,
                        help='Path to test data directory (containing color/ and depth_raw/)')
    
    # Output
    parser.add_argument('--output_path', type=str, default='submission.csv',
                        help='Path to save submission CSV file')
    
    # Inference settings
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for resizing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_path and not args.model_paths:
        parser.error("Either --model_path or --model_paths must be provided")
    if args.model_path and args.model_paths:
        parser.error("Cannot use both --model_path and --model_paths simultaneously")
    
    # Determine if ensemble or single model
    if args.model_paths:
        model_paths = args.model_paths
        is_ensemble = True
    else:
        model_paths = [args.model_path]
        is_ensemble = False
    
    print("="*80)
    print("TEST SET INFERENCE - NUTRITION5K CALORIE PREDICTION")
    print("="*80)
    if is_ensemble:
        print(f"ENSEMBLE MODE: {len(model_paths)} models")
        for i, path in enumerate(model_paths, 1):
            print(f"  Model {i}: {path}")
    else:
        print(f"SINGLE MODEL: {model_paths[0]}")
    print(f"Test data: {args.test_root}")
    print(f"Output: {args.output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print("="*80)
    
    # Validate inputs
    for model_path in model_paths:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not os.path.exists(args.test_root):
        raise FileNotFoundError(f"Test data not found: {args.test_root}")
    
    # Create test dataset
    print("\nLoading test dataset...")
    test_dataset = TestDataset(
        test_root=args.test_root,
        img_size=args.img_size
    )
    
    if is_ensemble:
        # Ensemble inference
        print(f"\nInitializing ensemble with {len(model_paths)} models...")
        ensemble_inference = EnsembleInference(
            model_paths=model_paths,
            device=args.device
        )
        
        # Run ensemble inference
        predictions, failed_samples = ensemble_inference.predict_test_set(
            test_dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Create submission file
        all_dish_ids = test_dataset.get_all_dish_ids()
        submission_df = ensemble_inference.create_submission_file(
            predictions=predictions,
            all_dish_ids=all_dish_ids,
            output_path=args.output_path
        )
    else:
        # Single model inference
        print("\nInitializing single model inference...")
        inference = TestInference(
            model_path=model_paths[0],
            device=args.device
        )
        
        # Run inference
        predictions, failed_samples = inference.predict_test_set(
            test_dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Create submission file
        all_dish_ids = test_dataset.get_all_dish_ids()
        submission_df = inference.create_submission_file(
            predictions=predictions,
            all_dish_ids=all_dish_ids,
            output_path=args.output_path
        )
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    print(f"Submission file ready: {args.output_path}")
    print(f"Total samples: {len(submission_df)}")
    if failed_samples:
        print(f"Failed samples: {len(failed_samples)}")
    if is_ensemble:
        print(f"Ensemble averaging: {len(model_paths)} models")
    print("="*80)


if __name__ == '__main__':
    main()
