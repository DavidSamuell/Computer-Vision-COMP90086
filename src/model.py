"""
Multi-Stream CNN with Middle Fusion for Calorie Prediction
Architecture: Dual ResNet-18 encoders with middle fusion + multi-task heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetEncoder(nn.Module):
    """
    ResNet-18 encoder that extracts feature maps
    Removes the final average pooling and fully connected layers
    """
    
    def __init__(self, pretrained: bool = False, in_channels: int = 3):
        super().__init__()
        
        # Load ResNet-18 (pretrained=False as per constraints)
        resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv if we have different input channels (e.g., 1 for depth)
        if in_channels != 3:
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = resnet.conv1
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # ResNet layers
        self.layer1 = resnet.layer1  # Output: 64 channels
        self.layer2 = resnet.layer2  # Output: 128 channels
        self.layer3 = resnet.layer3  # Output: 256 channels
        self.layer4 = resnet.layer4  # Output: 512 channels
        
        # Remove avgpool and fc
        self.out_channels = 512
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Feature map (B, 512, H/32, W/32)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


class MiddleFusionModule(nn.Module):
    """
    Middle fusion: Concatenate RGB and Depth features, then merge with 1x1 conv
    """
    
    def __init__(self, rgb_channels: int = 512, depth_channels: int = 512, output_channels: int = 512):
        super().__init__()
        
        # 1x1 convolution to merge features
        self.fusion_conv = nn.Conv2d(
            rgb_channels + depth_channels,
            output_channels,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, rgb_features, depth_features):
        """
        Args:
            rgb_features: (B, 512, H, W)
            depth_features: (B, 512, H, W)
        Returns:
            Fused features: (B, 512, H, W)
        """
        # Concatenate along channel dimension
        concatenated = torch.cat([rgb_features, depth_features], dim=1)  # (B, 1024, H, W)
        
        # Merge and reduce channels
        fused = self.fusion_conv(concatenated)  # (B, 512, H, W)
        fused = self.bn(fused)
        fused = self.relu(fused)
        
        return fused


class RegressionHead(nn.Module):
    """
    Regression head for calorie prediction
    Includes dropout for regularization
    """
    
    def __init__(self, in_channels: int = 512, dropout_rate: float = 0.4):
        super().__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Feature map (B, C, H, W)
        Returns:
            Calorie prediction (B, 1)
        """
        x = self.avgpool(x)  # (B, C, 1, 1)
        x = self.fc_layers(x)  # (B, 1)
        
        return x


class SegmentationHead(nn.Module):
    """
    Segmentation head for food mask prediction
    Uses transposed convolutions to upsample
    """
    
    def __init__(self, in_channels: int = 512, num_classes: int = 1):
        super().__init__()
        
        # Upsampling path
        self.upsample = nn.Sequential(
            # Upsample by 2x
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Upsample by 2x
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsample by 2x
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsample by 2x
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Upsample by 2x to match input size (32x upsampling total)
            nn.ConvTranspose2d(32, num_classes, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, x):
        """
        Args:
            x: Feature map (B, 512, H/32, W/32)
        Returns:
            Segmentation mask (B, 1, H, W)
        """
        x = self.upsample(x)
        return x


class MultiStreamCaloriePredictor(nn.Module):
    """
    Complete Multi-Stream CNN with Middle Fusion and Multi-Task Learning
    
    Architecture:
        - RGB Encoder (ResNet-18)
        - Depth Encoder (ResNet-18)
        - Middle Fusion Module
        - Regression Head (Calorie Prediction)
        - Segmentation Head (Food Mask Prediction)
    """
    
    def __init__(
        self,
        pretrained: bool = False,
        dropout_rate: float = 0.4,
        fusion_channels: int = 512
    ):
        super().__init__()
        
        # Encoders
        self.rgb_encoder = ResNetEncoder(pretrained=pretrained, in_channels=3)
        self.depth_encoder = ResNetEncoder(pretrained=pretrained, in_channels=1)
        
        # Fusion
        self.fusion = MiddleFusionModule(
            rgb_channels=512,
            depth_channels=512,
            output_channels=fusion_channels
        )
        
        # Heads
        self.regression_head = RegressionHead(
            in_channels=fusion_channels,
            dropout_rate=dropout_rate
        )
        self.segmentation_head = SegmentationHead(
            in_channels=fusion_channels,
            num_classes=1
        )
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: RGB image (B, 3, H, W)
            depth: Depth image (B, 1, H, W)
        
        Returns:
            calorie_pred: Predicted calories (B, 1)
            seg_pred: Predicted segmentation mask (B, 1, H, W)
        """
        # Extract features
        rgb_features = self.rgb_encoder(rgb)  # (B, 512, H/32, W/32)
        depth_features = self.depth_encoder(depth)  # (B, 512, H/32, W/32)
        
        # Fuse features
        fused_features = self.fusion(rgb_features, depth_features)  # (B, 512, H/32, W/32)
        
        # Multi-task predictions
        calorie_pred = self.regression_head(fused_features)  # (B, 1)
        seg_pred = self.segmentation_head(fused_features)  # (B, 1, H, W)
        
        return calorie_pred, seg_pred
    
    def get_num_parameters(self):
        """Return the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """Test the model architecture"""
    model = MultiStreamCaloriePredictor(pretrained=False, dropout_rate=0.4)
    
    # Create dummy inputs
    batch_size = 4
    rgb = torch.randn(batch_size, 3, 224, 224)
    depth = torch.randn(batch_size, 1, 224, 224)
    
    # Forward pass
    calorie_pred, seg_pred = model(rgb, depth)
    
    print(f"Model has {model.get_num_parameters():,} parameters")
    print(f"Input RGB shape: {rgb.shape}")
    print(f"Input Depth shape: {depth.shape}")
    print(f"Output Calorie shape: {calorie_pred.shape}")
    print(f"Output Segmentation shape: {seg_pred.shape}")
    
    assert calorie_pred.shape == (batch_size, 1)
    assert seg_pred.shape == (batch_size, 1, 224, 224)
    
    print("✓ Model test passed!")


if __name__ == '__main__':
    test_model()

