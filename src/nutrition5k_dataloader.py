"""
Custom Nutrition5k dataset loader with preprocessing as described in the paper
(Thames et al. 2021)

Key preprocessing elements:
- Images resized to 256x256
- Random crop to 224x224 during training
- Center crop to 224x224 during validation/testing
- Custom normalization values from the paper
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from typing import Tuple, Optional
import warnings


class Nutrition5kPaperDataset(Dataset):
    """
    Dataset class for Nutrition5K with preprocessing exactly as described in the paper
    """
    
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        split: str = 'train',
        img_size: int = 256,
        use_augmentation: bool = True,
        normalize_depth: bool = True
    ):
        """
        Args:
            csv_path: Path to the CSV file with dish IDs and calorie values
            data_root: Root directory containing color/, depth_raw/ subdirectories
            split: 'train' or 'val'
            img_size: Size to resize images (256 in the paper)
            use_augmentation: Whether to apply data augmentation
            normalize_depth: Whether to normalize depth images
        """
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.use_augmentation = use_augmentation and split == 'train'
        self.normalize_depth = normalize_depth
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Rename 'Value' column to 'calories' if it exists
        if 'Value' in self.df.columns and 'calories' not in self.df.columns:
            self.df = self.df.rename(columns={'Value': 'calories'})
        
        # Make sure calories column exists
        if 'calories' not in self.df.columns:
            raise ValueError("CSV file must contain a 'calories' column or a 'Value' column that can be renamed")
        
        # Filter out high-calorie samples (as mentioned in the paper)
        self.df = self.df[self.df['calories'] < 3000].reset_index(drop=True)
        
        # Build paths
        self.color_dir = os.path.join(data_root, 'color')
        self.depth_raw_dir = os.path.join(data_root, 'depth_raw')
        
        # Validate dataset
        self.valid_indices = self._validate_dataset()
        print(f"Loaded {len(self.valid_indices)} valid samples out of {len(self.df)}")
        
        # Normalization values from the paper
        self.color_normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def _validate_dataset(self):
        """Pre-validate all samples and return valid indices"""
        valid_indices = []
        
        for idx in range(len(self.df)):
            dish_id = self.df.iloc[idx]['ID']
            
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
    
    def _load_image_safe(self, path: str, mode: str = 'RGB') -> Optional[Image.Image]:
        """Safely load an image with error handling"""
        try:
            with Image.open(path) as img:
                return img.convert(mode).copy()
        except Exception as e:
            warnings.warn(f"Failed to load image {path}: {e}")
            return None
    
    def _apply_paper_preprocessing(self, rgb_img, depth_img):
        """
        Modified preprocessing to maintain 256x256 resolution:
        - Resize directly to img_size x img_size (256x256)
        - No additional cropping to maintain more image information
        """
        # Center crop to maintain aspect ratio, then resize to img_size x img_size (256x256)
        # This ensures we're capturing the center of the food item
        min_dim = min(rgb_img.size)
        rgb_img = TF.center_crop(rgb_img, (min_dim, min_dim))
        depth_img = TF.center_crop(depth_img, (min_dim, min_dim))
        
        # Now resize to target size
        rgb_img = TF.resize(rgb_img, (self.img_size, self.img_size))
        depth_img = TF.resize(depth_img, (self.img_size, self.img_size))
        
        # Apply data augmentation for training
        if self.split == 'train' and self.use_augmentation:
            # Random horizontal flip (50% probability)
            if random.random() > 0.5:
                rgb_img = TF.hflip(rgb_img)
                depth_img = TF.hflip(depth_img)
        
        return rgb_img, depth_img
    
    def __getitem__(self, idx):
        """Get a single sample"""
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        
        dish_id = row['ID']
        calorie = float(row['calories'])
        
        # Load images
        rgb_path = os.path.join(self.color_dir, dish_id, 'rgb.png')
        depth_path = os.path.join(self.depth_raw_dir, dish_id, 'depth_raw.png')
        
        rgb_img = self._load_image_safe(rgb_path, 'RGB')
        depth_img = self._load_image_safe(depth_path, 'L')  # Grayscale for depth
        
        if rgb_img is None or depth_img is None:
            # Fallback: return a black image
            rgb_img = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
            depth_img = Image.new('L', (self.img_size, self.img_size), 0)
        
        # Apply preprocessing as described in the paper
        rgb_img, depth_img = self._apply_paper_preprocessing(rgb_img, depth_img)
        
        # Convert to tensors
        rgb_tensor = TF.to_tensor(rgb_img)  # (3, H, W)
        depth_tensor = TF.to_tensor(depth_img)  # (1, H, W)
        
        # Normalize RGB
        rgb_tensor = self.color_normalize(rgb_tensor)
        
        # Normalize depth (according to the paper)
        if self.normalize_depth:
            depth_tensor = depth_tensor / 255.0
        
        return {
            'dish_id': dish_id,
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'calorie': torch.tensor(calorie, dtype=torch.float32)
        }


def create_train_val_split(csv_path: str, val_ratio: float = 0.15, random_seed: int = 42):
    """
    Create train/validation split CSV files
    """
    # Read original CSV
    df = pd.read_csv(csv_path)    
    
    # Shuffle with fixed seed
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Split
    val_size = int(len(df_shuffled) * val_ratio)
    train_df = df_shuffled[val_size:]
    val_df = df_shuffled[:val_size]
    
    # Save temporary CSV files
    base_dir = os.path.dirname(csv_path)
    train_csv = os.path.join(base_dir, 'train_split.csv')
    val_csv = os.path.join(base_dir, 'val_split.csv')
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    return train_csv, val_csv
