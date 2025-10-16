"""
Multi-Stream CNN with Middle Fusion for Calorie Prediction
Architecture: Dual ResNet encoders with middle fusion + multi-task heads

MODULAR VERSION: Use build_model() factory function to construct models
with different encoder/fusion/head combinations.

For backward compatibility, MultiStreamCaloriePredictor still exists
and uses ResNet-18 by default.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Import modular components
from encoders import build_encoder, get_encoder_out_channels, ENCODER_REGISTRY
from fusion_modules import build_fusion, FUSION_REGISTRY
from heads import build_regression_head, build_segmentation_head, REGRESSION_HEAD_REGISTRY, SEGMENTATION_HEAD_REGISTRY


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


class ModularCaloriePredictor(nn.Module):
    """
    Modular Multi-Stream CNN - Easily configure different components
    
    Use this class for experiments with different architectures:
    - Different encoders: resnet18, resnet34, resnet50, early_fusion_resnet18, early_fusion_resnet34
    - Different fusion strategies: 
        * Middle fusion: middle, middle_attention, additive, cross_modal_attention, gated
        * Early fusion: Use early_fusion_resnet* encoders
        * Late fusion: late_average, late_weighted
    - Different heads: standard, deep, se (regression), standard, light, fpn (segmentation)
    
    Example:
        # Middle fusion (default)
        model = ModularCaloriePredictor(
            encoder_name='resnet34',
            fusion_name='middle_attention',
            regression_head='deep',
            segmentation_head='fpn'
        )
        
        # Early fusion (single encoder)
        model = ModularCaloriePredictor(
            encoder_name='early_fusion_resnet34',
            fusion_name='none',  # No fusion needed with early fusion
        )
        
        # Late fusion (combine predictions)
        model = ModularCaloriePredictor(
            encoder_name='resnet34',
            fusion_name='late_average',  # or 'late_weighted'
        )
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet18',
        fusion_name: str = 'middle',
        regression_head_name: str = 'standard',
        segmentation_head_name: str = 'standard',
        pretrained: bool = False,
        dropout_rate: float = 0.4,
        fusion_channels: int = 512,
        use_segmentation: bool = False
    ):
        super().__init__()
        
        # Check fusion type
        self.is_early_fusion = 'early_fusion' in encoder_name
        self.is_late_fusion = 'late' in fusion_name
        self.use_segmentation = use_segmentation
        
        # Store config for reference
        self.config = {
            'encoder': encoder_name,
            'fusion': fusion_name if not self.is_early_fusion else 'none',
            'regression_head': regression_head_name,
            'segmentation_head': segmentation_head_name if use_segmentation else 'none',
            'pretrained': pretrained,
            'dropout_rate': dropout_rate,
            'fusion_channels': fusion_channels,
            'is_early_fusion': self.is_early_fusion,
            'is_late_fusion': self.is_late_fusion,
            'use_segmentation': use_segmentation
        }
        
        if self.is_early_fusion:
            # Early fusion: single encoder with 4-channel input
            self.encoder = build_encoder(encoder_name, pretrained=pretrained, in_channels=4)
            encoder_out_channels = self.encoder.out_channels
            fusion_output_channels = encoder_out_channels
            
            # Build single set of heads
            self.regression_head = build_regression_head(
                regression_head_name,
                in_channels=fusion_output_channels,
                dropout_rate=dropout_rate
            )
            if self.use_segmentation:
                self.segmentation_head = build_segmentation_head(
                    segmentation_head_name,
                    in_channels=fusion_output_channels,
                    num_classes=1
                )
            else:
                self.segmentation_head = None
                
        elif self.is_late_fusion:
            # Late fusion: separate encoders and separate heads for each modality
            self.rgb_encoder = build_encoder(encoder_name, pretrained=pretrained, in_channels=3)
            self.depth_encoder = build_encoder(encoder_name, pretrained=pretrained, in_channels=1)
            encoder_out_channels = self.rgb_encoder.out_channels
            
            # Separate prediction heads for RGB and Depth
            self.rgb_regression_head = build_regression_head(
                regression_head_name,
                in_channels=encoder_out_channels,
                dropout_rate=dropout_rate
            )
            self.depth_regression_head = build_regression_head(
                regression_head_name,
                in_channels=encoder_out_channels,
                dropout_rate=dropout_rate
            )
            
            # Fusion weights for late fusion
            if fusion_name == 'late_weighted':
                # Learnable weights for combining predictions
                self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
            else:  # late_average
                # Fixed equal weights
                self.register_buffer('fusion_weights', torch.tensor([0.5, 0.5]))
            
            # Segmentation: use combined features (average of RGB and depth features)
            if self.use_segmentation:
                self.segmentation_head = build_segmentation_head(
                    segmentation_head_name,
                    in_channels=encoder_out_channels,
                    num_classes=1
                )
            else:
                self.segmentation_head = None
                
        else:
            # Middle fusion: separate encoders, fusion module, shared heads
            self.rgb_encoder = build_encoder(encoder_name, pretrained=pretrained, in_channels=3)
            self.depth_encoder = build_encoder(encoder_name, pretrained=pretrained, in_channels=1)
            encoder_out_channels = self.rgb_encoder.out_channels
            
            # Build fusion module
            self.fusion = build_fusion(
                fusion_name,
                rgb_channels=encoder_out_channels,
                depth_channels=encoder_out_channels,
                output_channels=fusion_channels
            )
            fusion_output_channels = fusion_channels
            
            # Build prediction heads
            self.regression_head = build_regression_head(
                regression_head_name,
                in_channels=fusion_output_channels,
                dropout_rate=dropout_rate
            )
            if self.use_segmentation:
                self.segmentation_head = build_segmentation_head(
                    segmentation_head_name,
                    in_channels=fusion_output_channels,
                    num_classes=1
                )
            else:
                self.segmentation_head = None
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: RGB image (B, 3, H, W)
            depth: Depth image (B, 1, H, W)
        
        Returns:
            calorie_pred: Predicted calories (B, 1)
            seg_pred: Predicted segmentation mask (B, 1, H, W) or None if segmentation disabled
        """
        if self.is_early_fusion:
            # Early fusion: concatenate RGB and Depth at input
            rgbd = torch.cat([rgb, depth], dim=1)  # (B, 4, H, W)
            fused_features = self.encoder(rgbd)
            
            # Calorie prediction
            calorie_pred = self.regression_head(fused_features)
            
            # Segmentation prediction (only if enabled)
            if self.use_segmentation:
                seg_pred = self.segmentation_head(fused_features)
            else:
                seg_pred = None
                
        elif self.is_late_fusion:
            # Late fusion: process streams independently and combine predictions
            # Extract features separately
            rgb_features = self.rgb_encoder(rgb)
            depth_features = self.depth_encoder(depth)
            
            # Separate predictions
            rgb_calorie_pred = self.rgb_regression_head(rgb_features)
            depth_calorie_pred = self.depth_regression_head(depth_features)
            
            # Normalize weights (softmax for learned weights)
            if isinstance(self.fusion_weights, nn.Parameter):
                weights = F.softmax(self.fusion_weights, dim=0)
            else:
                weights = self.fusion_weights
            
            # Combine predictions with weighted average
            calorie_pred = weights[0] * rgb_calorie_pred + weights[1] * depth_calorie_pred
            
            # Segmentation: use average of features
            if self.use_segmentation:
                avg_features = (rgb_features + depth_features) / 2.0
                seg_pred = self.segmentation_head(avg_features)
            else:
                seg_pred = None
                
        else:
            # Middle fusion: fuse features then predict
            # Extract features separately
            rgb_features = self.rgb_encoder(rgb)
            depth_features = self.depth_encoder(depth)
            
            # Fuse features
            fused_features = self.fusion(rgb_features, depth_features)
            
            # Calorie prediction
            calorie_pred = self.regression_head(fused_features)
            
            # Segmentation prediction (only if enabled)
            if self.use_segmentation:
                seg_pred = self.segmentation_head(fused_features)
            else:
                seg_pred = None
        
        return calorie_pred, seg_pred
    
    def get_num_parameters(self):
        """Return the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self):
        """Return model configuration"""
        return self.config


def build_model(
    encoder: str = 'resnet18',
    fusion: str = 'middle',
    regression_head: str = 'standard',
    segmentation_head: str = 'standard',
    pretrained: bool = False,
    dropout_rate: float = 0.4,
    fusion_channels: int = 512,
    use_segmentation: bool = False
):
    """
    Factory function to build a model with specified components
    
    Args:
        encoder: Encoder architecture ('resnet18', 'resnet34', 'resnet50')
        fusion: Fusion strategy ('middle', 'middle_attention', 'additive')
        regression_head: Regression head type ('standard', 'deep')
        segmentation_head: Segmentation head type ('standard', 'light')
        pretrained: Use pretrained weights
        dropout_rate: Dropout rate for regularization
        fusion_channels: Number of channels after fusion
        use_segmentation: Whether to build segmentation head (default: False)
    
    Returns:
        ModularCaloriePredictor model
    
    Example:
        # Try ResNet-34 with attention fusion
        model = build_model(encoder='resnet34', fusion='middle_attention')
        
        # Baseline without segmentation
        model = build_model(encoder='resnet18', use_segmentation=False)
    """
    return ModularCaloriePredictor(
        encoder_name=encoder,
        fusion_name=fusion,
        regression_head_name=regression_head,
        segmentation_head_name=segmentation_head,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        fusion_channels=fusion_channels,
        use_segmentation=use_segmentation
    )


def list_available_components():
    """Print all available model components"""
    print("\n" + "="*60)
    print("AVAILABLE MODEL COMPONENTS")
    print("="*60)
    
    print("\nEncoders:")
    for name in ENCODER_REGISTRY.keys():
        print(f"  - {name}")
    
    print("\nFusion Modules:")
    for name in FUSION_REGISTRY.keys():
        print(f"  - {name}")
    
    print("\nRegression Heads:")
    for name in REGRESSION_HEAD_REGISTRY.keys():
        print(f"  - {name}")
    
    print("\nSegmentation Heads:")
    for name in SEGMENTATION_HEAD_REGISTRY.keys():
        print(f"  - {name}")
    
    print("\n" + "="*60 + "\n")


def test_model():
    """Test the model architecture"""
    print("="*60)
    print("TESTING ORIGINAL MODEL")
    print("="*60)
    
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


def test_modular_models():
    """Test different model configurations"""
    print("\n" + "="*60)
    print("TESTING MODULAR MODELS")
    print("="*60)
    
    # Test configurations
    configs = [
        {'encoder': 'resnet18', 'fusion': 'middle', 'name': 'ResNet-18 + Middle Fusion'},
        {'encoder': 'resnet34', 'fusion': 'middle', 'name': 'ResNet-34 + Middle Fusion'},
        {'encoder': 'resnet34', 'fusion': 'middle_attention', 'name': 'ResNet-34 + Attention Fusion'},
        {'encoder': 'resnet50', 'fusion': 'middle', 'name': 'ResNet-50 + Middle Fusion (Large)'},
    ]
    
    batch_size = 2
    rgb = torch.randn(batch_size, 3, 224, 224)
    depth = torch.randn(batch_size, 1, 224, 224)
    
    for config in configs:
        print(f"\n{config['name']}:")
        print("-" * 60)
        
        model = build_model(
            encoder=config['encoder'],
            fusion=config['fusion'],
            pretrained=False,
            dropout_rate=0.4
        )
        
        # Forward pass
        calorie_pred, seg_pred = model(rgb, depth)
        
        print(f"  Parameters: {model.get_num_parameters():,}")
        print(f"  Output Calorie: {calorie_pred.shape}")
        print(f"  Output Segmentation: {seg_pred.shape}")
        
        assert calorie_pred.shape == (batch_size, 1)
        assert seg_pred.shape == (batch_size, 1, 224, 224)
        
        print(f"  ✓ Test passed!")
    
    print("\n" + "="*60)
    print("✓ All modular models tested successfully!")
    print("="*60 + "\n")


if __name__ == '__main__':
    # List available components
    list_available_components()
    
    # Test original model
    test_model()
    
    # Test modular models
    test_modular_models()

