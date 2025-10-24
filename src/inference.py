"""
Simplified Test Set Inference Script for Nutrition5K Calorie Prediction
Compatible with InceptionV2, InceptionV3, and Volume Estimation models
"""

import os
import argparse
import pandas as pd
import torch
import numpy as np
import torch.serialization
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import Normalize
from tqdm import tqdm
import warnings
import traceback
import json
import torchvision.models as models

# Define model components for standalone inference
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    """Basic convolution module for InceptionV2: Conv2d + BatchNorm + ReLU"""
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionModule(nn.Module):
    """InceptionV2 module with BatchNorm"""
    
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()
        
        # 1x1 branch
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        
        # 3x3 branch
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 branch (implemented as two 3x3 convs)
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
            BasicConv2d(ch5x5, ch5x5, kernel_size=3, padding=1)
        )
        
        # Pool branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class InceptionV2Encoder(nn.Module):
    """InceptionV2 encoder backbone"""
    
    def __init__(self, pretrained=False, in_channels=3):
        super().__init__()
        
        self.out_channels = 1024
        
        # Initial layers
        self.conv1 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicConv2d(64, 64, kernel_size=1),
            BasicConv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Inception blocks
        self.inception3a = InceptionModule(192, 64, 64, 64, 64, 96, 32)
        self.inception3b = InceptionModule(256, 64, 64, 96, 64, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # More Inception blocks
        self.inception4a = InceptionModule(320, 224, 64, 96, 96, 128, 128)
        self.inception4b = InceptionModule(576, 192, 96, 128, 96, 128, 128)
        self.inception4c = InceptionModule(576, 160, 128, 160, 128, 160, 96)
        self.inception4d = InceptionModule(576, 96, 128, 192, 160, 192, 96)
        self.inception4e = InceptionModule(576, 128, 128, 192, 192, 256, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Final Inception blocks
        self.inception5a = InceptionModule(704, 352, 192, 320, 160, 224, 128)
        self.inception5b = InceptionModule(1024, 352, 192, 320, 192, 224, 128)
    
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        return x


class RegressionHead(nn.Module):
    """
    Regression head for calorie prediction
    Modified architecture with smaller FC layers for non-pretrained model
    """
    
    def __init__(self, in_channels=1024, fc_dim=1024, dropout_rate=0.4):
        super().__init__()
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers with reduced dimensions for non-pretrained model
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_dim, fc_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_dim // 2, 1)
        )
    
    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc_layers(x)
        return x


class CalorieInceptionV2Model(nn.Module):
    """Complete InceptionV2 model for calorie prediction"""
    
    def __init__(self, pretrained=False, dropout_rate=0.4, fc_dim=1024):
        super().__init__()
        
        # InceptionV2 encoder
        self.encoder = InceptionV2Encoder(pretrained=pretrained, in_channels=3)
        
        # Regression head for calorie prediction
        self.regression_head = RegressionHead(
            in_channels=self.encoder.out_channels,
            fc_dim=fc_dim,
            dropout_rate=dropout_rate
        )
    
    def forward(self, x):
        features = self.encoder(x)
        calories = self.regression_head(features)
        return calories
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# InceptionV3 Model Components
class InceptionV3Encoder(nn.Module):
    """InceptionV3 encoder as used in the original Nutrition5k paper"""
    
    def __init__(self, pretrained: bool = False, in_channels: int = 3):
        super().__init__()
        
        # Load InceptionV3 model
        inception = models.inception_v3(pretrained=pretrained, aux_logits=False)
        
        # The output of InceptionV3 features is 2048 channels
        self.out_channels = 2048
        
        # Modify first conv if we have different input channels (e.g., 1 for depth)
        if in_channels != 3:
            self.Conv2d_1a_3x3 = nn.Conv2d(
                in_channels, 32, kernel_size=3, stride=2, bias=False
            )
        else:
            self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        
        # Copy all other layers from InceptionV3
        # First block
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = inception.maxpool1
        
        # Second block
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = inception.maxpool2
        
        # Inception blocks
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Feature map (B, 2048, H/32, W/32)
        """
        # First block
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        
        # Second block
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        
        # Inception blocks
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        
        return x


class VolumeEstimator(nn.Module):
    """
    Food volume estimation from overhead depth images following the Nutrition5k paper.
    
    Given:
    - Distance between camera and capture plane: 35.9 cm
    - Per-pixel surface area at this distance: 5.957 × 10^-3 cm²
    
    The volume is calculated by:
    1. Computing per-pixel volume (depth × surface_area)
    2. Summing over all food pixels (using binary threshold segmentation)
    """
    
    def __init__(self, 
                 camera_distance: float = 35.9,  # cm
                 pixel_surface_area: float = 5.957e-3,  # cm²
                 depth_threshold: float = 0.1):  # Threshold for simple segmentation
        super().__init__()
        
        self.camera_distance = camera_distance
        self.pixel_surface_area = pixel_surface_area
        self.depth_threshold = depth_threshold
    
    def forward(self, depth_images):
        """
        Args:
            depth_images: Depth images (B, 1, H, W), normalized to [0, 1] range
        
        Returns:
            volume_estimates: Volume in cm³ for each image (B, 1)
        """
        # Simple threshold-based segmentation for foreground/background
        segmentation_mask = (depth_images > self.depth_threshold).float()
        
        # Convert normalized depth back to actual depth values
        depth_cm = depth_images * self.camera_distance
        
        # Calculate per-pixel volume: depth × surface_area
        per_pixel_volume = depth_cm * self.pixel_surface_area  # (B, 1, H, W)
        
        # Apply segmentation mask to consider only food pixels
        masked_volume = per_pixel_volume * segmentation_mask
        
        # Sum over all pixels to get total volume
        volume_estimates = masked_volume.sum(dim=[2, 3])  # (B, 1)
        
        return volume_estimates


class RegressionHeadV3(nn.Module):
    """InceptionV3 regression head"""
    def __init__(self, in_channels: int = 2048, dropout_rate: float = 0.4):
        super().__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        x = self.avgpool(x)  # (B, C, 1, 1)
        x = self.fc_layers(x)  # (B, 1)
        return x


class RegressionHeadWithVolume(nn.Module):
    """
    Regression head that concatenates volume estimate to InceptionV3 features.
    """
    
    def __init__(self, in_channels: int = 2048, dropout_rate: float = 0.4):
        super().__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Two FC layers as described in the paper (2048+1 -> 64 -> 1)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels + 1, 64),  # +1 for volume
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
    
    def forward(self, features, volume):
        """
        Args:
            features: Feature maps from backbone (B, 2048, H, W)
            volume: Volume estimates (B, 1)
        
        Returns:
            Predicted calories (B, 1)
        """
        # Global average pooling
        x = self.avgpool(features)  # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)  # (B, 2048)
        
        # Concatenate volume estimate
        x = torch.cat([x, volume], dim=1)  # (B, 2049)
        
        # Predict calories
        x = self.fc_layers(x)  # (B, 1)
        
        return x


class Nutrition5kModel(nn.Module):
    """
    Implementation of the dual-stream architecture used in the original Nutrition5k paper
    Uses InceptionV3 as the backbone with different fusion types
    """
    
    def __init__(
        self,
        fusion: str = 'middle',
        fusion_channels: int = 2048,
        dropout_rate: float = 0.4,
        pretrained: bool = False,
        use_volume: bool = False
    ):
        super().__init__()
        
        self.use_volume = use_volume
        self.fusion = fusion
        
        if fusion == 'image_only':
            # Image-only variant: only RGB is used
            self.rgb_encoder = InceptionV3Encoder(pretrained=pretrained, in_channels=3)
            
            # Volume estimator (if enabled)
            if use_volume:
                self.volume_estimator = VolumeEstimator()
                self.regression_head = RegressionHeadWithVolume(
                    in_channels=self.rgb_encoder.out_channels,
                    dropout_rate=dropout_rate
                )
            else:
                self.regression_head = RegressionHeadV3(
                    in_channels=self.rgb_encoder.out_channels,
                    dropout_rate=dropout_rate
                )
        elif fusion == 'image_volume':
            # Image+Volume variant: RGB encoder + volume as additional signal
            self.rgb_encoder = InceptionV3Encoder(pretrained=pretrained, in_channels=3)
            self.volume_estimator = VolumeEstimator()
            self.regression_head = RegressionHeadWithVolume(
                in_channels=self.rgb_encoder.out_channels,
                dropout_rate=dropout_rate
            )
            self.use_volume = True  # Always use volume for this variant
        else:  # middle fusion
            # RGB and Depth encoders using InceptionV3
            self.rgb_encoder = InceptionV3Encoder(pretrained=pretrained, in_channels=3)
            self.depth_encoder = InceptionV3Encoder(pretrained=pretrained, in_channels=1)
            
            # Create middle fusion module
            from_channels = self.rgb_encoder.out_channels + self.depth_encoder.out_channels
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(from_channels, fusion_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            )
            
            # Volume estimator (if enabled)
            if use_volume:
                self.volume_estimator = VolumeEstimator()
                self.regression_head = RegressionHeadWithVolume(
                    in_channels=fusion_channels,
                    dropout_rate=dropout_rate
                )
            else:
                self.regression_head = RegressionHeadV3(
                    in_channels=fusion_channels,
                    dropout_rate=dropout_rate
                )
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: RGB images (B, 3, H, W)
            depth: Depth images (B, 1, H, W)
        
        Returns:
            calorie_pred: Predicted calories (B, 1)
        """
        # Calculate volume estimate if enabled
        volume = None
        if self.use_volume and hasattr(self, 'volume_estimator'):
            volume = self.volume_estimator(depth)  # (B, 1)
        
        # Image-only or Image+Volume variant
        if hasattr(self, 'rgb_encoder') and not hasattr(self, 'depth_encoder'):
            rgb_features = self.rgb_encoder(rgb)  # (B, 2048, H/32, W/32)
            
            if volume is not None:
                calorie_pred = self.regression_head(rgb_features, volume)
            else:
                calorie_pred = self.regression_head(rgb_features)
            
            return calorie_pred
        
        # Extract features from both streams
        rgb_features = self.rgb_encoder(rgb)      # (B, 2048, H/32, W/32)
        depth_features = self.depth_encoder(depth)  # (B, 2048, H/32, W/32)
        
        # Middle fusion - concatenate and apply 1x1 conv
        fused = torch.cat([rgb_features, depth_features], dim=1)  # (B, 4096, H/32, W/32)
        fused = self.fusion_conv(fused)  # (B, 2048, H/32, W/32)
        
        # Predict calories (with or without volume)
        if volume is not None:
            calorie_pred = self.regression_head(fused, volume)
        else:
            calorie_pred = self.regression_head(fused)
        
        return calorie_pred
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class TestDataset(Dataset):
    """Dataset class for test set inference"""
    
    def __init__(self, test_root, img_size=256):
        """
        Args:
            test_root: Root directory containing test data
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
    
    def _resize_and_center_crop(self, img, target_size: int = 256):
        """
        Resize and center crop image to target_size x target_size
        Matches the preprocessing in the Nutrition5k paper
        
        Args:
            img: PIL Image
            target_size: Target size (default 256x256 as per paper)
        
        Returns:
            Cropped PIL Image
        """
        # Get original dimensions
        width, height = img.size
        
        # Resize so the shorter side is target_size
        if width < height:
            new_width = target_size
            new_height = int(target_size * height / width)
        else:
            new_height = target_size
            new_width = int(target_size * width / height)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Center crop to target_size x target_size
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        
        img = img.crop((left, top, right, bottom))
        
        return img
    
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
        
        # Resize and center crop to match training preprocessing
        rgb_img = self._resize_and_center_crop(rgb_img, target_size=self.img_size)
        depth_img = self._resize_and_center_crop(depth_img, target_size=self.img_size)
        
        # Convert to tensors
        rgb_tensor = TF.to_tensor(rgb_img)  # (3, H, W)
        depth_tensor = TF.to_tensor(depth_img)  # (1, H, W)
        
        # Normalize RGB
        rgb_tensor = self.color_normalize(rgb_tensor)
        
        # Normalize depth to [0, 1] range
        depth_tensor = depth_tensor / 255.0
        
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'dish_id': dish_id,
            'is_valid': True
        }
    
    def get_all_dish_ids(self):
        """Get all dish IDs (including invalid ones) for complete submission"""
        return self.dish_ids

class SimpleInference:
    """Simple inference manager compatible with models from comp90086.ipynb"""
    
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
        model_dir = os.path.dirname(self.model_path)
        
        # Look for config file
        config_path = os.path.join(model_dir, 'config.json')
        config = None
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded config from: {config_path}")
        else:
            print("No config file found, using default settings")
        
        # Determine model type and configuration
        if config and 'fusion' in config:
            # InceptionV3 model with fusion
            model_type = 'inceptionv3'
            fusion = config.get('fusion', 'middle')
            use_volume = config.get('use_volume', False)
            dropout_rate = config.get('dropout_rate', 0.4)
            fusion_channels = config.get('fusion_channels', 2048)
            
            print(f"Using InceptionV3 model configuration:")
            print(f"  Fusion type: {fusion}")
            print(f"  Use volume: {use_volume}")
            print(f"  Dropout rate: {dropout_rate}")
            print(f"  Fusion channels: {fusion_channels}")
            
            # Create InceptionV3 model
            model = Nutrition5kModel(
                fusion=fusion,
                fusion_channels=fusion_channels,
                dropout_rate=dropout_rate,
                pretrained=False,
                use_volume=use_volume
            )
        else:
            # Default to InceptionV2 model for backward compatibility
            model_type = 'inceptionv2'
            dropout_rate = 0.4
            fc_dim = 1024
            
            print(f"Using InceptionV2 model configuration:")
            print(f"  FC dimensions: {fc_dim}")
            print(f"  Dropout rate: {dropout_rate}")
            
            # Create InceptionV2 model
            model = CalorieInceptionV2Model(
                pretrained=False,
                dropout_rate=dropout_rate,
                fc_dim=fc_dim
            )
        
        # Load checkpoint
        try:
            # Add numpy globals to the safe list
            torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
            
            # Try loading with explicitly setting weights_only=False
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            print("Successfully loaded checkpoint with weights_only=False")
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                if 'val_loss' in checkpoint:
                    print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
            else:
                # Direct state dict
                model.load_state_dict(checkpoint)
                print("Loaded model state dict directly")
            
        except Exception as e:
            print("\nDetailed error information:")
            import traceback
            traceback.print_exc()
            
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
            
            # Forward pass (handle both model types)
            try:
                # Try InceptionV3 model with depth input
                calorie_pred = self.model(rgb, depth)
            except TypeError:
                # Try InceptionV2 model with RGB only
                calorie_pred = self.model(rgb)
            
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

def main():
    parser = argparse.ArgumentParser(description='Simple Test Inference for Notebook Models')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--test_root', type=str, required=True,
                        help='Path to test data directory (containing color/ and depth_raw/)')
    
    # Output
    parser.add_argument('--output_path', type=str, default='submission.csv',
                        help='Path to save submission CSV file')
    
    # Inference settings
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for resizing (256 for InceptionV3, 224 for InceptionV2)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("NUTRITION5K INFERENCE - INCEPTIONV2/V3 + VOLUME ESTIMATION")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Test data: {args.test_root}")
    print(f"Output: {args.output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print("="*80)
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    if not os.path.exists(args.test_root):
        raise FileNotFoundError(f"Test data not found: {args.test_root}")
    
    # Create test dataset
    print("\nLoading test dataset...")
    test_dataset = TestDataset(
        test_root=args.test_root,
        img_size=args.img_size
    )
    
    # Initialize inference
    print("\nInitializing inference...")
    inference = SimpleInference(
        model_path=args.model_path,
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
    print(f"Submission file saved: {args.output_path}")

if __name__ == '__main__':
    main()