"""
Fusion modules for combining RGB and Depth features
Supports different fusion strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiddleFusion(nn.Module):
    """
    Middle fusion: Concatenate RGB and Depth features, then merge with 1x1 conv
    This is the original fusion strategy
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
            rgb_features: (B, C1, H, W)
            depth_features: (B, C2, H, W)
        Returns:
            Fused features: (B, output_channels, H, W)
        """
        # Concatenate along channel dimension
        concatenated = torch.cat([rgb_features, depth_features], dim=1)
        
        # Merge and reduce channels
        fused = self.fusion_conv(concatenated)
        fused = self.bn(fused)
        fused = self.relu(fused)
        
        return fused


class MiddleFusionWithAttention(nn.Module):
    """
    Middle fusion with channel attention
    Uses squeeze-and-excitation style attention to weight features
    """
    
    def __init__(self, rgb_channels: int = 512, depth_channels: int = 512, output_channels: int = 512):
        super().__init__()
        
        # Concatenation
        concat_channels = rgb_channels + depth_channels
        
        # Channel attention
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(concat_channels, concat_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(concat_channels // 16, concat_channels),
            nn.Sigmoid()
        )
        
        # Fusion conv
        self.fusion_conv = nn.Conv2d(
            concat_channels,
            output_channels,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, rgb_features, depth_features):
        """
        Args:
            rgb_features: (B, C1, H, W)
            depth_features: (B, C2, H, W)
        Returns:
            Fused features: (B, output_channels, H, W)
        """
        # Concatenate
        concatenated = torch.cat([rgb_features, depth_features], dim=1)
        B, C, H, W = concatenated.shape
        
        # Channel attention
        pooled = self.global_pool(concatenated).view(B, C)
        attention_weights = self.attention(pooled).view(B, C, 1, 1)
        
        # Apply attention
        attended = concatenated * attention_weights
        
        # Fusion
        fused = self.fusion_conv(attended)
        fused = self.bn(fused)
        fused = self.relu(fused)
        
        return fused


class AdditiveFusion(nn.Module):
    """
    Additive fusion: Element-wise addition of RGB and Depth features
    Requires both features to have same number of channels
    """
    
    def __init__(self, rgb_channels: int = 512, depth_channels: int = 512, output_channels: int = 512):
        super().__init__()
        
        # Project to output channels if needed
        self.rgb_proj = None
        self.depth_proj = None
        
        if rgb_channels != output_channels:
            self.rgb_proj = nn.Sequential(
                nn.Conv2d(rgb_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_channels)
            )
        
        if depth_channels != output_channels:
            self.depth_proj = nn.Sequential(
                nn.Conv2d(depth_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, rgb_features, depth_features):
        """
        Args:
            rgb_features: (B, C1, H, W)
            depth_features: (B, C2, H, W)
        Returns:
            Fused features: (B, output_channels, H, W)
        """
        # Project if needed
        if self.rgb_proj is not None:
            rgb_features = self.rgb_proj(rgb_features)
        
        if self.depth_proj is not None:
            depth_features = self.depth_proj(depth_features)
        
        # Add
        fused = rgb_features + depth_features
        fused = self.relu(fused)
        
        return fused


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-Modal Attention Fusion: RGB and Depth attend to each other
    Learns which modality to trust for each spatial region
    """
    
    def __init__(self, rgb_channels: int = 512, depth_channels: int = 512, output_channels: int = 512):
        super().__init__()
        
        # Attention mechanism (simplified cross-attention)
        self.rgb_query = nn.Conv2d(rgb_channels, rgb_channels // 8, kernel_size=1)
        self.depth_key = nn.Conv2d(depth_channels, depth_channels // 8, kernel_size=1)
        self.depth_value = nn.Conv2d(depth_channels, depth_channels, kernel_size=1)
        
        self.depth_query = nn.Conv2d(depth_channels, depth_channels // 8, kernel_size=1)
        self.rgb_key = nn.Conv2d(rgb_channels, rgb_channels // 8, kernel_size=1)
        self.rgb_value = nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1)
        
        # Fusion conv
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
            rgb_features: (B, C1, H, W)
            depth_features: (B, C2, H, W)
        Returns:
            Fused features: (B, output_channels, H, W)
        """
        B, C, H, W = rgb_features.shape
        
        # RGB queries Depth
        q_rgb = self.rgb_query(rgb_features).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C')
        k_depth = self.depth_key(depth_features).view(B, -1, H * W)  # (B, C', HW)
        v_depth = self.depth_value(depth_features).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C)
        
        attn_rgb_to_depth = torch.softmax(torch.bmm(q_rgb, k_depth) / (C ** 0.5), dim=-1)  # (B, HW, HW)
        rgb_attended = torch.bmm(attn_rgb_to_depth, v_depth).permute(0, 2, 1).view(B, C, H, W)
        
        # Depth queries RGB
        q_depth = self.depth_query(depth_features).view(B, -1, H * W).permute(0, 2, 1)
        k_rgb = self.rgb_key(rgb_features).view(B, -1, H * W)
        v_rgb = self.rgb_value(rgb_features).view(B, -1, H * W).permute(0, 2, 1)
        
        attn_depth_to_rgb = torch.softmax(torch.bmm(q_depth, k_rgb) / (C ** 0.5), dim=-1)
        depth_attended = torch.bmm(attn_depth_to_rgb, v_rgb).permute(0, 2, 1).view(B, C, H, W)
        
        # Concatenate and fuse
        concatenated = torch.cat([rgb_attended, depth_attended], dim=1)
        fused = self.fusion_conv(concatenated)
        fused = self.bn(fused)
        fused = self.relu(fused)
        
        return fused


class GatedFusion(nn.Module):
    """
    Gated Fusion: Learn adaptive fusion weights per sample
    Handles cases where one modality is noisy or less informative
    """
    
    def __init__(self, rgb_channels: int = 512, depth_channels: int = 512, output_channels: int = 512):
        super().__init__()
        
        # Project to same channels if needed
        self.rgb_proj = None
        self.depth_proj = None
        
        if rgb_channels != output_channels:
            self.rgb_proj = nn.Sequential(
                nn.Conv2d(rgb_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_channels)
            )
        
        if depth_channels != output_channels:
            self.depth_proj = nn.Sequential(
                nn.Conv2d(depth_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_channels)
            )
        
        # Gate network
        self.gate_conv = nn.Sequential(
            nn.Conv2d(output_channels * 2, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, rgb_features, depth_features):
        """
        Args:
            rgb_features: (B, C1, H, W)
            depth_features: (B, C2, H, W)
        Returns:
            Fused features: (B, output_channels, H, W)
        """
        # Project if needed
        if self.rgb_proj is not None:
            rgb_features = self.rgb_proj(rgb_features)
        
        if self.depth_proj is not None:
            depth_features = self.depth_proj(depth_features)
        
        # Compute gating weights
        concatenated = torch.cat([rgb_features, depth_features], dim=1)
        gate = self.gate_conv(concatenated)
        
        # Gated fusion
        fused = gate * rgb_features + (1 - gate) * depth_features
        fused = self.relu(fused)
        
        return fused


class LateFusionAverage(nn.Module):
    """
    Late Fusion - Average: Average predictions from two separate streams
    Each stream processes independently with its own heads
    
    Note: This is a placeholder module. Late fusion is handled at the model level
    since it requires separate prediction heads for each modality.
    """
    
    def __init__(self, rgb_channels: int = 512, depth_channels: int = 512, output_channels: int = 512):
        super().__init__()
        # Late fusion doesn't combine features - it combines predictions
        # This module exists for API compatibility
        pass
    
    def forward(self, rgb_features, depth_features):
        """
        For late fusion, features are not combined here.
        This method should not be called in late fusion mode.
        """
        raise NotImplementedError(
            "Late fusion does not combine features. "
            "Use LateFusionModel or set is_late_fusion=True in ModularCaloriePredictor."
        )


class LateFusionWeighted(nn.Module):
    """
    Late Fusion - Weighted: Learned weighted combination of predictions
    
    Note: This is a placeholder module. Late fusion is handled at the model level.
    """
    
    def __init__(self, rgb_channels: int = 512, depth_channels: int = 512, output_channels: int = 512):
        super().__init__()
        pass
    
    def forward(self, rgb_features, depth_features):
        raise NotImplementedError(
            "Late fusion does not combine features. "
            "Use LateFusionModel or set is_late_fusion=True in ModularCaloriePredictor."
        )


class InceptionFusion(nn.Module):
    """
    Inception-style Fusion: Multi-scale processing after concatenation
    Applies different kernel sizes in parallel like Inception modules
    """
    
    def __init__(self, rgb_channels: int = 512, depth_channels: int = 512, output_channels: int = 512):
        super().__init__()
        
        concat_channels = rgb_channels + depth_channels
        branch_channels = output_channels // 4
        
        # Branch 1: 1x1 conv
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(concat_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 1x1 -> 3x3 conv
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(concat_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 1x1 -> 5x5 conv (using two 3x3)
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(concat_channels, branch_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels // 2, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 3x3 max pool -> 1x1 conv
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(concat_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final adjustment if needed
        total_out_channels = branch_channels * 4
        if total_out_channels != output_channels:
            self.final_conv = nn.Sequential(
                nn.Conv2d(total_out_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.final_conv = None
    
    def forward(self, rgb_features, depth_features):
        """
        Args:
            rgb_features: (B, C1, H, W)
            depth_features: (B, C2, H, W)
        Returns:
            Fused features: (B, output_channels, H, W)
        """
        # Concatenate first
        concatenated = torch.cat([rgb_features, depth_features], dim=1)
        
        # Apply parallel branches
        branch1 = self.branch1x1(concatenated)
        branch2 = self.branch3x3(concatenated)
        branch3 = self.branch5x5(concatenated)
        branch4 = self.branch_pool(concatenated)
        
        # Concatenate all branches
        fused = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        
        # Final adjustment if needed
        if self.final_conv is not None:
            fused = self.final_conv(fused)
        
        return fused


# Fusion factory
FUSION_REGISTRY = {
    'middle': MiddleFusion,
    'middle_attention': MiddleFusionWithAttention,
    'additive': AdditiveFusion,
    'cross_modal_attention': CrossModalAttentionFusion,
    'gated': GatedFusion,
    'late_average': LateFusionAverage,
    'late_weighted': LateFusionWeighted,
    'inception': InceptionFusion,
}


def build_fusion(fusion_name: str, rgb_channels: int, depth_channels: int, output_channels: int):
    """
    Factory function to build fusion module
    
    Args:
        fusion_name: Name of fusion strategy
        rgb_channels: Number of RGB feature channels
        depth_channels: Number of depth feature channels
        output_channels: Number of output channels
    
    Returns:
        Fusion module
    """
    if fusion_name not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion: {fusion_name}. Available: {list(FUSION_REGISTRY.keys())}")
    
    fusion_class = FUSION_REGISTRY[fusion_name]
    return fusion_class(rgb_channels=rgb_channels, depth_channels=depth_channels, output_channels=output_channels)


if __name__ == '__main__':
    # Test fusion modules
    print("Testing fusion modules...")
    
    for fusion_name in FUSION_REGISTRY.keys():
        print(f"\n{fusion_name.upper()}:")
        
        fusion = build_fusion(fusion_name, rgb_channels=512, depth_channels=512, output_channels=512)
        
        # Test forward pass
        rgb_feat = torch.randn(2, 512, 7, 7)
        depth_feat = torch.randn(2, 512, 7, 7)
        
        out = fusion(rgb_feat, depth_feat)
        
        print(f"  RGB input: {rgb_feat.shape}")
        print(f"  Depth input: {depth_feat.shape}")
        print(f"  Output: {out.shape}")
        
        # Count parameters
        params = sum(p.numel() for p in fusion.parameters())
        print(f"  Parameters: {params:,}")
    
    print("\n✓ All fusion modules tested successfully!")

