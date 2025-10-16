"""
Encoder modules for feature extraction
Supports different backbone architectures (ResNet-18, ResNet-34, ResNet-50, etc.)
"""

import torch
import torch.nn as nn
from torchvision import models


class BaseEncoder(nn.Module):
    """Base class for all encoders"""
    
    def __init__(self):
        super().__init__()
        self.out_channels = 512  # Default, will be overridden
    
    def forward(self, x):
        raise NotImplementedError


class ResNet18Encoder(nn.Module):
    """
    ResNet-18 encoder that extracts feature maps
    Removes the final average pooling and fully connected layers
    """
    
    def __init__(self, pretrained: bool = False, in_channels: int = 3):
        super().__init__()
        
        # Load ResNet-18
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


class ResNet34Encoder(nn.Module):
    """
    ResNet-34 encoder that extracts feature maps
    Deeper than ResNet-18 with more layers in each block
    """
    
    def __init__(self, pretrained: bool = False, in_channels: int = 3):
        super().__init__()
        
        # Load ResNet-34
        resnet = models.resnet34(pretrained=pretrained)
        
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
        
        # ResNet layers (more layers than ResNet-18)
        self.layer1 = resnet.layer1  # Output: 64 channels
        self.layer2 = resnet.layer2  # Output: 128 channels
        self.layer3 = resnet.layer3  # Output: 256 channels
        self.layer4 = resnet.layer4  # Output: 512 channels
        
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


class ResNet50Encoder(nn.Module):
    """
    ResNet-50 encoder that extracts feature maps
    Uses bottleneck blocks for deeper architecture
    """
    
    def __init__(self, pretrained: bool = False, in_channels: int = 3):
        super().__init__()
        
        # Load ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first conv if we have different input channels
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
        self.layer1 = resnet.layer1  # Output: 256 channels
        self.layer2 = resnet.layer2  # Output: 512 channels
        self.layer3 = resnet.layer3  # Output: 1024 channels
        self.layer4 = resnet.layer4  # Output: 2048 channels
        
        self.out_channels = 2048  # Note: ResNet-50 outputs 2048 channels
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Feature map (B, 2048, H/32, W/32)
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


class EarlyFusionResNet18Encoder(nn.Module):
    """
    Early Fusion ResNet-18: Single encoder with 4-channel input (RGB+Depth)
    Reduces parameters by ~50% compared to dual encoder approach
    Good for limited data scenarios
    """
    
    def __init__(self, pretrained: bool = False, in_channels: int = 4):
        super().__init__()
        
        # Load ResNet-18 and modify for 4-channel input
        resnet = models.resnet18(pretrained=False)  # Can't use pretrained with 4 channels
        
        # Modify first conv for 4 channels (RGB=3 + Depth=1)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # ResNet layers
        self.layer1 = resnet.layer1  # Output: 64 channels
        self.layer2 = resnet.layer2  # Output: 128 channels
        self.layer3 = resnet.layer3  # Output: 256 channels
        self.layer4 = resnet.layer4  # Output: 512 channels
        
        self.out_channels = 512
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 4, H, W) - Concatenated RGB+Depth
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


class EarlyFusionResNet34Encoder(nn.Module):
    """
    Early Fusion ResNet-34: Single encoder with 4-channel input (RGB+Depth)
    Deeper than ResNet-18 but still uses early fusion for parameter efficiency
    """
    
    def __init__(self, pretrained: bool = False, in_channels: int = 4):
        super().__init__()
        
        # Load ResNet-34 and modify for 4-channel input
        resnet = models.resnet34(pretrained=False)
        
        # Modify first conv for 4 channels
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # ResNet layers
        self.layer1 = resnet.layer1  # Output: 64 channels
        self.layer2 = resnet.layer2  # Output: 128 channels
        self.layer3 = resnet.layer3  # Output: 256 channels
        self.layer4 = resnet.layer4  # Output: 512 channels
        
        self.out_channels = 512
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 4, H, W) - Concatenated RGB+Depth
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


# Encoder factory
ENCODER_REGISTRY = {
    'resnet18': ResNet18Encoder,
    'resnet34': ResNet34Encoder,
    'resnet50': ResNet50Encoder,
    'early_fusion_resnet18': EarlyFusionResNet18Encoder,
    'early_fusion_resnet34': EarlyFusionResNet34Encoder,
}


def build_encoder(encoder_name: str, pretrained: bool = False, in_channels: int = 3):
    """
    Factory function to build encoder
    
    Args:
        encoder_name: Name of encoder ('resnet18', 'resnet34', 'resnet50')
        pretrained: Whether to use pretrained weights
        in_channels: Number of input channels
    
    Returns:
        Encoder module
    """
    if encoder_name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder: {encoder_name}. Available: {list(ENCODER_REGISTRY.keys())}")
    
    encoder_class = ENCODER_REGISTRY[encoder_name]
    return encoder_class(pretrained=pretrained, in_channels=in_channels)


def get_encoder_out_channels(encoder_name: str) -> int:
    """Get output channels for a given encoder"""
    if encoder_name in ['resnet18', 'resnet34', 'early_fusion_resnet18', 'early_fusion_resnet34']:
        return 512
    elif encoder_name == 'resnet50':
        return 2048
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")


if __name__ == '__main__':
    # Test encoders
    print("Testing encoders...")
    
    for encoder_name in ENCODER_REGISTRY.keys():
        print(f"\n{encoder_name.upper()}:")
        
        # RGB encoder
        rgb_encoder = build_encoder(encoder_name, pretrained=False, in_channels=3)
        print(f"  Out channels: {rgb_encoder.out_channels}")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        out = rgb_encoder(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        
        # Count parameters
        params = sum(p.numel() for p in rgb_encoder.parameters())
        print(f"  Parameters: {params:,}")
    
    print("\n✓ All encoders tested successfully!")

