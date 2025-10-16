"""
Prediction heads for different tasks
Includes regression head for calories and segmentation head for masks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DeepRegressionHead(nn.Module):
    """
    Deeper regression head for calorie prediction
    More layers for better capacity
    """
    
    def __init__(self, in_channels: int = 512, dropout_rate: float = 0.4):
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
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
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


class LightRegressionHead(nn.Module):
    """
    Light regression head with single hidden layer
    Good for limited data scenarios (3K samples)
    Reduces overfitting risk with fewer parameters
    """
    
    def __init__(self, in_channels: int = 512, dropout_rate: float = 0.4):
        super().__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
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


class MinimalRegressionHead(nn.Module):
    """
    Minimal regression head - direct mapping from features to prediction
    Most parameter-efficient option
    Best for very limited data or when encoder is already strong
    """
    
    def __init__(self, in_channels: int = 512, dropout_rate: float = 0.3):
        super().__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),  # Only dropout before final layer
            nn.Linear(in_channels, 1)
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
            x: Feature map (B, C, H/32, W/32)
        Returns:
            Segmentation mask (B, num_classes, H, W)
        """
        x = self.upsample(x)
        return x


class LightSegmentationHead(nn.Module):
    """
    Lighter segmentation head with fewer parameters
    Uses bilinear upsampling instead of learned transposed convs
    """
    
    def __init__(self, in_channels: int = 512, num_classes: int = 1):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )
    
    def forward(self, x):
        """
        Args:
            x: Feature map (B, C, H/32, W/32)
        Returns:
            Segmentation mask (B, num_classes, H, W)
        """
        x = self.conv_layers(x)
        # Upsample to original size
        x = F.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False)
        return x


class FPNSegmentationHead(nn.Module):
    """
    Feature Pyramid Network (FPN) Segmentation Head
    Uses multi-scale features for better segmentation quality
    Particularly good for limited data scenarios
    
    Note: Requires encoder to return intermediate features
    For now, uses only final features (compatible with current encoders)
    """
    
    def __init__(self, in_channels: int = 512, num_classes: int = 1):
        super().__init__()
        
        # Lateral connections (for multi-scale, currently single scale)
        self.lateral = nn.Conv2d(in_channels, 256, kernel_size=1)
        
        # FPN upsampling path with skip connections
        self.fpn_conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_conv4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final prediction
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: Feature map (B, C, H/32, W/32)
        Returns:
            Segmentation mask (B, num_classes, H, W)
        """
        # Lateral connection
        x = self.lateral(x)  # (B, 256, H/32, W/32)
        
        # Progressive upsampling with convolutions
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.fpn_conv1(x)  # (B, 256, H/16, W/16)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.fpn_conv2(x)  # (B, 128, H/8, W/8)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.fpn_conv3(x)  # (B, 64, H/4, W/4)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.fpn_conv4(x)  # (B, 32, H/2, W/2)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.final_conv(x)  # (B, num_classes, H, W)
        
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Adds channel attention to emphasize important features
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Feature map (B, C, H, W)
        Returns:
            Attention-weighted features (B, C, H, W)
        """
        B, C, _, _ = x.shape
        
        # Squeeze
        y = self.global_pool(x).view(B, C)
        
        # Excitation
        y = self.fc(y).view(B, C, 1, 1)
        
        # Scale
        return x * y


class RegressionHeadWithSE(nn.Module):
    """
    Regression head with Squeeze-and-Excitation attention
    Adds channel attention before global pooling
    """
    
    def __init__(self, in_channels: int = 512, dropout_rate: float = 0.4):
        super().__init__()
        
        # SE block before pooling
        self.se = SEBlock(in_channels)
        
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
        x = self.se(x)  # Apply channel attention
        x = self.avgpool(x)
        x = self.fc_layers(x)
        
        return x


# Head factories
REGRESSION_HEAD_REGISTRY = {
    'standard': RegressionHead,
    'deep': DeepRegressionHead,
    'light': LightRegressionHead,
    'minimal': MinimalRegressionHead,
    'se': RegressionHeadWithSE,
}

SEGMENTATION_HEAD_REGISTRY = {
    'standard': SegmentationHead,
    'light': LightSegmentationHead,
    'fpn': FPNSegmentationHead,
}


def build_regression_head(head_name: str, in_channels: int, dropout_rate: float):
    """Build regression head"""
    if head_name not in REGRESSION_HEAD_REGISTRY:
        raise ValueError(f"Unknown regression head: {head_name}. Available: {list(REGRESSION_HEAD_REGISTRY.keys())}")
    
    head_class = REGRESSION_HEAD_REGISTRY[head_name]
    return head_class(in_channels=in_channels, dropout_rate=dropout_rate)


def build_segmentation_head(head_name: str, in_channels: int, num_classes: int = 1):
    """Build segmentation head"""
    if head_name not in SEGMENTATION_HEAD_REGISTRY:
        raise ValueError(f"Unknown segmentation head: {head_name}. Available: {list(SEGMENTATION_HEAD_REGISTRY.keys())}")
    
    head_class = SEGMENTATION_HEAD_REGISTRY[head_name]
    return head_class(in_channels=in_channels, num_classes=num_classes)


if __name__ == '__main__':
    # Test heads
    print("Testing prediction heads...")
    
    print("\nREGRESSION HEADS:")
    for head_name in REGRESSION_HEAD_REGISTRY.keys():
        print(f"\n{head_name.upper()}:")
        
        head = build_regression_head(head_name, in_channels=512, dropout_rate=0.4)
        
        # Test forward pass
        x = torch.randn(2, 512, 7, 7)
        out = head(x)
        
        print(f"  Input: {x.shape}")
        print(f"  Output: {out.shape}")
        
        # Count parameters
        params = sum(p.numel() for p in head.parameters())
        print(f"  Parameters: {params:,}")
    
    print("\n" + "="*60)
    print("SEGMENTATION HEADS:")
    for head_name in SEGMENTATION_HEAD_REGISTRY.keys():
        print(f"\n{head_name.upper()}:")
        
        head = build_segmentation_head(head_name, in_channels=512, num_classes=1)
        
        # Test forward pass
        x = torch.randn(2, 512, 7, 7)
        out = head(x)
        
        print(f"  Input: {x.shape}")
        print(f"  Output: {out.shape}")
        
        # Count parameters
        params = sum(p.numel() for p in head.parameters())
        print(f"  Parameters: {params:,}")
    
    print("\n✓ All heads tested successfully!")

