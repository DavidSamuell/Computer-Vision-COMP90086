"""
Nutrition5K Dataset Class with Augmentation and Robust Error Handling
Handles RGB images, raw depth images, and segmentation masks
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


class Nutrition5KDataset(Dataset):
    """
    Dataset class for Nutrition5K with multi-modal inputs (RGB + Depth)
    and multi-task outputs (Calories + Segmentation)
    """
    
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        split: str = 'train',
        augment: bool = True,
        img_size: int = 224
    ):
        """
        Args:
            csv_path: Path to the CSV file with dish IDs and calorie values
            data_root: Root directory containing color/, depth_raw/ subdirectories
            split: 'train' or 'val'
            augment: Whether to apply data augmentation
            img_size: Target image size for resizing
        """
        self.data_root = data_root
        self.split = split
        self.augment = augment
        self.img_size = img_size
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Build paths
        self.color_dir = os.path.join(data_root, 'color')
        self.depth_raw_dir = os.path.join(data_root, 'depth_raw')
        
        # Validate dataset
        self.valid_indices = self._validate_dataset()
        print(f"Loaded {len(self.valid_indices)} valid samples out of {len(self.df)}")
        
        # Color normalization (ImageNet stats as baseline)
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
        """Safely load an image, return None if corrupt"""
        try:
            img = Image.open(path).convert(mode)
            return img
        except Exception as e:
            warnings.warn(f"Failed to load image {path}: {e}")
            return None
    
    def _create_segmentation_mask(self, rgb_image: Image.Image) -> Image.Image:
        """
        Create a simple segmentation mask from RGB image.
        Since the dataset doesn't include masks, we create a simple one
        by detecting non-background pixels (simple thresholding).
        
        This is a placeholder - ideally you'd have real segmentation masks.
        For training, we'll assume the food is generally centered and non-white.
        """
        # Convert to numpy array
        img_array = np.array(rgb_image)
        
        # Simple background detection: assume white/light background
        # Create mask where any channel is below a threshold (indicating food)
        mask = np.any(img_array < 240, axis=2).astype(np.uint8) * 255
        
        # Convert to PIL Image
        mask_img = Image.fromarray(mask, mode='L')
        
        return mask_img
    
    def _apply_geometric_augmentation(
        self,
        rgb: Image.Image,
        depth: Image.Image,
        mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """
        Apply the SAME geometric augmentation to all three images.
        This is critical for maintaining spatial correspondence.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            rgb = TF.hflip(rgb)
            depth = TF.hflip(depth)
            mask = TF.hflip(mask)
        
        # Random rotation (-15 to +15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            rgb = TF.rotate(rgb, angle)
            depth = TF.rotate(depth, angle)
            mask = TF.rotate(mask, angle)
        
        # Random crop and resize
        if random.random() > 0.5:
            i, j, h, w = T.RandomResizedCrop.get_params(
                rgb, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            rgb = TF.resized_crop(rgb, i, j, h, w, (self.img_size, self.img_size))
            depth = TF.resized_crop(depth, i, j, h, w, (self.img_size, self.img_size))
            mask = TF.resized_crop(mask, i, j, h, w, (self.img_size, self.img_size))
        else:
            rgb = TF.resize(rgb, (self.img_size, self.img_size))
            depth = TF.resize(depth, (self.img_size, self.img_size))
            mask = TF.resize(mask, (self.img_size, self.img_size))
        
        return rgb, depth, mask
    
    def _apply_color_augmentation(self, rgb: Image.Image) -> Image.Image:
        """
        Apply color augmentation ONLY to RGB image.
        Do NOT apply to depth image.
        """
        # Random brightness
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.7, 1.3)
            rgb = TF.adjust_brightness(rgb, brightness_factor)
        
        # Random contrast
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.7, 1.3)
            rgb = TF.adjust_contrast(rgb, contrast_factor)
        
        # Random saturation
        if random.random() > 0.5:
            saturation_factor = random.uniform(0.7, 1.3)
            rgb = TF.adjust_saturation(rgb, saturation_factor)
        
        # Random hue
        if random.random() > 0.5:
            hue_factor = random.uniform(-0.1, 0.1)
            rgb = TF.adjust_hue(rgb, hue_factor)
        
        return rgb
    
    def __getitem__(self, idx):
        """
        Returns:
            rgb_tensor: (3, H, W) RGB image
            depth_tensor: (1, H, W) raw depth image
            mask_tensor: (1, H, W) segmentation mask
            calorie: float - calorie value
        """
        actual_idx = self.valid_indices[idx]
        
        # Get dish info
        dish_id = self.df.iloc[actual_idx]['ID']
        calorie = self.df.iloc[actual_idx]['Value']
        
        # Load images
        rgb_path = os.path.join(self.color_dir, dish_id, 'rgb.png')
        depth_path = os.path.join(self.depth_raw_dir, dish_id, 'depth_raw.png')
        
        rgb_img = self._load_image_safe(rgb_path, mode='RGB')
        depth_img = self._load_image_safe(depth_path, mode='L')  # Grayscale
        
        # Fallback for corrupt images (shouldn't happen after validation, but just in case)
        if rgb_img is None or depth_img is None:
            # Return a random valid sample instead
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Create segmentation mask
        mask_img = self._create_segmentation_mask(rgb_img)
        
        # Apply augmentation if training
        if self.augment and self.split == 'train':
            # 1. Apply geometric augmentation (same for all)
            rgb_img, depth_img, mask_img = self._apply_geometric_augmentation(
                rgb_img, depth_img, mask_img
            )
            
            # 2. Apply color augmentation (only to RGB)
            rgb_img = self._apply_color_augmentation(rgb_img)
        else:
            # Just resize for validation
            rgb_img = TF.resize(rgb_img, (self.img_size, self.img_size))
            depth_img = TF.resize(depth_img, (self.img_size, self.img_size))
            mask_img = TF.resize(mask_img, (self.img_size, self.img_size))
        
        # Convert to tensors
        rgb_tensor = TF.to_tensor(rgb_img)  # (3, H, W)
        depth_tensor = TF.to_tensor(depth_img)  # (1, H, W)
        mask_tensor = TF.to_tensor(mask_img)  # (1, H, W)
        
        # Normalize RGB
        rgb_tensor = self.color_normalize(rgb_tensor)
        
        # Normalize depth to [0, 1] range (assuming 16-bit depth)
        depth_tensor = depth_tensor / 65535.0 if depth_tensor.max() > 1.0 else depth_tensor
        
        # Convert mask to binary (0 or 1)
        mask_tensor = (mask_tensor > 0.5).float()
        
        # Convert calorie to tensor
        calorie_tensor = torch.tensor(calorie, dtype=torch.float32)
        
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'mask': mask_tensor,
            'calorie': calorie_tensor,
            'dish_id': dish_id
        }


def create_train_val_split(csv_path: str, val_ratio: float = 0.15, random_seed: int = 42):
    """
    Split the training data into train and validation sets
    
    Args:
        csv_path: Path to nutrition5k_train.csv
        val_ratio: Ratio of validation data
        random_seed: Random seed for reproducibility
    
    Returns:
        train_df, val_df: Two DataFrames for training and validation
    """
    df = pd.read_csv(csv_path)
    
    # Shuffle
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Split
    val_size = int(len(df) * val_ratio)
    val_df = df[:val_size]
    train_df = df[val_size:]
    
    # Save to temporary files
    train_csv_path = csv_path.replace('.csv', '_train_split.csv')
    val_csv_path = csv_path.replace('.csv', '_val_split.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    return train_csv_path, val_csv_path


if __name__ == '__main__':
    # Test the dataset
    csv_path = '../Nutrition5K/nutrition5k_train.csv'
    data_root = '../Nutrition5K/train'
    
    dataset = Nutrition5KDataset(csv_path, data_root, split='train', augment=True)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"RGB shape: {sample['rgb'].shape}")
    print(f"Depth shape: {sample['depth'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Calorie: {sample['calorie'].item()}")
    print(f"Dish ID: {sample['dish_id']}")

