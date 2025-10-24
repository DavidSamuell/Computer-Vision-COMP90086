import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models

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

# Early Fusion Module (RGB + Depth fused at input level)
class EarlyFusion(nn.Module):
    """
    Early Fusion: Combine RGB and Depth channels at the input level
    before processing through the network
    """
    
    def __init__(self, pretrained: bool = False, fusion_channels: int = 2048, dropout_rate: float = 0.4):
        super().__init__()
        
        # Create a single encoder with 4 input channels (3 RGB + 1 Depth)
        self.encoder = InceptionV3Encoder(pretrained=pretrained, in_channels=4)
        
        # Regression head for calorie prediction
        self.regression_head = RegressionHead(
            in_channels=self.encoder.out_channels,
            dropout_rate=dropout_rate
        )
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: RGB images (B, 3, H, W)
            depth: Depth images (B, 1, H, W)
        
        Returns:
            Predicted calories (B, 1)
        """
        # Concatenate RGB and depth along channel dimension
        x = torch.cat([rgb, depth], dim=1)  # (B, 4, H, W)
        
        # Process through the encoder
        features = self.encoder(x)
        
        # Predict calories
        calories = self.regression_head(features)
        
        return calories

# Late Fusion Module (RGB + Depth processed separately and fused at regression level)
class LateFusion(nn.Module):
    """
    Late Fusion: Process RGB and Depth streams independently, then fuse at the regression head level
    """
    
    def __init__(self, pretrained: bool = False, fusion_channels: int = 2048, dropout_rate: float = 0.4):
        super().__init__()
        
        # RGB and Depth encoders
        self.rgb_encoder = InceptionV3Encoder(pretrained=pretrained, in_channels=3)
        self.depth_encoder = InceptionV3Encoder(pretrained=pretrained, in_channels=1)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fusion at the feature vector level
        in_features = self.rgb_encoder.out_channels + self.depth_encoder.out_channels
        
        # Fully connected layers for regression
        self.regression_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: RGB images (B, 3, H, W)
            depth: Depth images (B, 1, H, W)
        
        Returns:
            Predicted calories (B, 1)
        """
        # Extract features from both streams
        rgb_features = self.rgb_encoder(rgb)    # (B, 2048, H/32, W/32)
        depth_features = self.depth_encoder(depth)  # (B, 2048, H/32, W/32)
        
        # Apply global average pooling
        rgb_features = self.avgpool(rgb_features)    # (B, 2048, 1, 1)
        depth_features = self.avgpool(depth_features)  # (B, 2048, 1, 1)
        
        # Concatenate feature vectors
        fused = torch.cat([rgb_features, depth_features], dim=1)  # (B, 4096, 1, 1)
        
        # Predict calories
        calories = self.regression_layers(fused)
        
        return calories

class RegressionHead(nn.Module):
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
        # Assuming depth is normalized to [0, 1] and represents distance from camera
        # For simplicity, we assume the depth represents actual distance in cm scaled to [0, 1]
        depth_cm = depth_images * self.camera_distance
        
        # Calculate per-pixel volume: depth × surface_area
        per_pixel_volume = depth_cm * self.pixel_surface_area  # (B, 1, H, W)
        
        # Apply segmentation mask to consider only food pixels
        masked_volume = per_pixel_volume * segmentation_mask
        
        # Sum over all pixels to get total volume
        volume_estimates = masked_volume.sum(dim=[2, 3])  # (B, 1)
        
        return volume_estimates


class RegressionHeadWithVolume(nn.Module):
    """
    Regression head that concatenates volume estimate to InceptionV3 features.
    
    According to the paper: "concatenating the volume estimation value to the output 
    of the InceptionV3 backbone, before the following two fully connected layers"
    with FC layers of 64 and 1 dimension.
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
    Uses InceptionV3 as the backbone and middle fusion
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
        
        if fusion == 'early':
            self.model = EarlyFusion(
                pretrained=pretrained,
                fusion_channels=fusion_channels,
                dropout_rate=dropout_rate
            )
        elif fusion == 'late':
            self.model = LateFusion(
                pretrained=pretrained,
                fusion_channels=fusion_channels,
                dropout_rate=dropout_rate
            )
        elif fusion == 'image_only':
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
                self.regression_head = RegressionHead(
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
                self.regression_head = RegressionHead(
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
        if hasattr(self, 'model'):
            return self.model(rgb, depth)
        
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

# Factory function to build Nutrition5k models with different fusion types
def build_nutrition5k_model(fusion='middle', pretrained=False, dropout_rate=0.4, fusion_channels=2048, 
                           use_volume=False, **kwargs):
    """
    Factory function to build models using the Nutrition5k paper architecture (InceptionV3 backbone)
    
    Args:
        fusion: Fusion type ('early', 'middle', 'late', 'image_only', or 'image_volume')
        pretrained: Whether to use pretrained weights for InceptionV3
        dropout_rate: Dropout rate for regression head
        fusion_channels: Number of channels after fusion
        use_volume: Whether to use volume estimation as additional signal (uses simple threshold-based segmentation)
    
    Returns:
        Nutrition5k model with specified configuration
    """
    return Nutrition5kModel(
        fusion=fusion,
        fusion_channels=fusion_channels,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
        use_volume=use_volume
    )
