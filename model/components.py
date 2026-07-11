"""
File usage for: model/components.py
This file contains various building blocks and components for constructing neural network models.
These components include standard convolutional layers, attention mechanisms, and custom-designed modules
for specific computer vision tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvBNReLU(nn.Module):
    """
    A standard convolutional block consisting of a 2D convolution,
    batch normalization, and a GELU activation function.
    """
    def __init__(self, n_channels, out_ch, k=3, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = (k - 1) // 2
        self.conv = nn.Conv2d(n_channels, out_ch, k, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        """
        Forward pass for the ConvBNReLU block.
        """
        return self.act(self.bn(self.conv(x)))

class DepthwiseSeparable(nn.Module):
    """
    A depthwise separable convolution block, which is more efficient
    than a standard convolution. It consists of a depthwise convolution
    followed by a pointwise convolution.
    """
    def __init__(self, n_channels, out_ch, k=3, stride=1):
        super().__init__()
        pad = (k-1)//2
        self.dw = nn.Conv2d(n_channels, n_channels, kernel_size=k, stride=stride, padding=pad, groups=n_channels, bias=False)
        self.pw = nn.Conv2d(n_channels, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        """
        Forward pass for the DepthwiseSeparable block.
        """
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


# Polarized Self-Attention (PSA)
class PolarizedSelfAttention(nn.Module):
    """
    Implementation of Polarized Self-Attention (PSA), which combines
    channel and spatial attention mechanisms.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()
    
    def channel_attention(self, x):
        """
        Computes channel attention for the input tensor.
        """
        b, c, _, _ = x.size()
        y = self.channel_pool(x).view(b, c)
        y = self.channel_fc(y).view(b, c, 1, 1)
        return x * y
    
    def spatial_attention(self, x):
        """
        Computes spatial attention for the input tensor.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.spatial_conv(y)
        y = self.spatial_sigmoid(y)
        return x * y
    
    def forward(self, x):
        """
        Forward pass for the PolarizedSelfAttention block.
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# Mobile Attention Convolution (MAC)
class MAC(nn.Module):
    """
    Mobile Attention Convolution (MAC) block, which combines an
    inverted residual structure with a Polarized Self-Attention module.
    """
    def __init__(self, n_channels, out_ch, expansion=4):
        super().__init__()
        hidden_dim = n_channels * expansion
        
        self.expand = nn.Conv2d(n_channels, hidden_dim, 1, bias=False) if n_channels != hidden_dim else nn.Identity()
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.psa = PolarizedSelfAttention(hidden_dim)
        
        self.project = nn.Conv2d(hidden_dim, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        
        self.act = nn.GELU()
        self.use_residual = (n_channels == out_ch)
    
    def forward(self, x):
        """
        Forward pass for the MAC block.
        """
        identity = x
        
        x = self.expand(x)
        x = self.bn1(x)
        x = self.act(x)
        
        x = self.dw_conv(x)
        x = self.bn2(x)
        x = self.act(x)
        
        x = self.psa(x)
        
        x = self.project(x)
        x = self.bn3(x)
        
        if self.use_residual:
            x = x + identity
            
        return self.act(x)


# Multi-scale Large-kernel Dual Attention (MLDA)
class MLDA(nn.Module):
    """
    Multi-scale Large-kernel Dual Attention (MLDA) block, which
    captures features at multiple scales using different kernel sizes.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.local_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.local_bn = nn.BatchNorm2d(channels)
        
        self.channel_att = PolarizedSelfAttention(channels)
        
        self.branch0 = DepthwiseSeparable(channels, channels, k=5)
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        self.fuse = nn.Conv2d(channels * 4, channels, 1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(channels)
        self.fuse_act = nn.GELU()
        
        self.final_conv = nn.Conv2d(channels, channels, 1, bias=False)
        
    def forward(self, x):
        """
        Forward pass for the MLDA block.
        """
        local = self.local_conv(x)
        local = self.local_bn(local)
        
        x_att = self.channel_att(local)
        
        b0 = self.branch0(x_att)
        b1 = self.branch1(x_att)
        b2 = self.branch2(x_att)
        b3 = self.branch3(x_att)
        
        multi_scale = torch.cat([b0, b1, b2, b3], dim=1)
        fused = self.fuse(multi_scale)
        fused = self.fuse_bn(fused)
        fused = self.fuse_act(fused)
        
        out = self.final_conv(fused)
        return out


# Mobile Multi-scale Attention Convolution (MMAC)
class MMAC(nn.Module):
    """
    Mobile Multi-scale Attention Convolution (MMAC) block, which
    combines MAC and MLDA blocks.
    """
    def __init__(self, n_channels, out_ch):
        super().__init__()
        self.mac1 = MAC(n_channels, out_ch)
        self.mac2 = MAC(out_ch, out_ch)
        self.mlda = MLDA(out_ch)
        
    def forward(self, x):
        """
        Forward pass for the MMAC block.
        """
        x = self.mac1(x)
        x = self.mac2(x)
        x = self.mlda(x)
        return x

# CPCF fusion module - FIXED VERSION
class ResizeOp(nn.Module):
    """Learnable resize operation with projection to target channels"""
    def __init__(self, n_channels, out_ch, mode='up'):
        """
        Initializes the ResizeOp module.
        """
        super().__init__()
        self.mode = mode
        self.conv = nn.Conv2d(n_channels, out_ch, 1, bias=False)
        
    def forward(self, x):
        """
        Forward pass for the ResizeOp module.
        """
        x = self.conv(x)
        if self.mode == 'up':
            return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            return F.avg_pool2d(x, kernel_size=2, stride=2)

class CPCFModule(nn.Module):
    """
    Fixed CPCF module with proper channel projection. This module
    fuses features from different stages of the network.
    """
    def __init__(self, target_channels, stage, f1_ch, f2_ch, f3_ch, f4_ch):
        super().__init__()
        self.stage = stage
        self.target_channels = target_channels
        
        # Projection layers for each input feature to target_channels
        self.proj1 = nn.Conv2d(f1_ch, target_channels, 1, bias=False)
        self.proj2 = nn.Conv2d(f2_ch, target_channels, 1, bias=False)
        self.proj3 = nn.Conv2d(f3_ch, target_channels, 1, bias=False)
        self.proj4 = nn.Conv2d(f4_ch, target_channels, 1, bias=False)
        
        # Resize operations
        self.rup = ResizeOp(target_channels, target_channels, mode='up')
        self.rdown = ResizeOp(target_channels, target_channels, mode='down')
        
        # Attention block
        self.att = MLDA(target_channels)
        
    def forward(self, f1, f2, f3, f4):
        """
        Forward pass for the CPCFModule.
        """
        # Project all features to target channel dimension
        f1_p = self.proj1(f1)
        f2_p = self.proj2(f2)
        f3_p = self.proj3(f3)
        f4_p = self.proj4(f4)
        
        # Implement the exact fusion strategy from paper Eq.(3)
        if self.stage == 1:
            # S1 = Rup(Rup(Rup(f4) + f3) + f2) + f1
            s = self.rup(self.rup(self.rup(f4_p) + f3_p) + f2_p) + f1_p
            ref_feature = f1_p
        elif self.stage == 2:
            # S2 = Rup(Rup(f4) + f3) + Rdown(f1) + f2
            s = self.rup(self.rup(f4_p) + f3_p) + self.rdown(f1_p) + f2_p
            ref_feature = f2_p
        elif self.stage == 3:
            # S3 = Rdown(Rdown(f1) + f2) + Rup(f4) + f3
            s = self.rdown(self.rdown(f1_p) + f2_p) + self.rup(f4_p) + f3_p
            ref_feature = f3_p
        elif self.stage == 4:
            # S4 = Rdown(Rdown(Rdown(f1) + f2) + f3) + f4
            s = self.rdown(self.rdown(self.rdown(f1_p) + f2_p) + f3_p) + f4_p
            ref_feature = f4_p
        
        # Apply attention and skip connection
        m = self.att(s) + ref_feature
        return m

class ClsHead(nn.Module):
    """
    A classification head that takes features from multiple levels of
    the network and produces a final classification output.
    """
    def __init__(
        self,
        f3_channels,
        f4_channels,
        f5_channels,
        hidden_dim=128,
        n_classes=4
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        total_channels = (
            f3_channels +
            f4_channels +
            f5_channels
        )

        self.fc = nn.Sequential(
            nn.Linear(
                total_channels,
                hidden_dim
            ),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(
                hidden_dim,
                n_classes
            )
        )


    def forward(self, f3, f4, f5):
        """
        Forward pass for the classification head.
        """
        f3 = self.pool(f3)
        f4 = self.pool(f4)
        f5 = self.pool(f5)

        f3 = torch.flatten(f3, start_dim=1)
        f4 = torch.flatten(f4, start_dim=1)
        f5 = torch.flatten(f5, start_dim=1)
        
        x = torch.cat([f3, f4, f5], dim=1)
        cls = self.fc(x)
        return cls