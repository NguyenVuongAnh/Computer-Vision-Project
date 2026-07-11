import torch
import torch.nn as nn
from .components import *

# Complete CMAC-Net
class CMACNet(nn.Module):
    def __init__(self, n_channels=3, base_ch=32, seg_classes=5, cls_classes=4):
        super().__init__()
        

        # Encoder (MDAC Backbone)
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, base_ch, 3, stride=2, padding=1),
            nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.GELU()
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(MAC(base_ch, base_ch))
        self.pool1 = nn.MaxPool2d(2)
        
        # Stage 2
        self.stage2 = nn.Sequential(
            MAC(base_ch, base_ch * 2),
            MAC(base_ch * 2, base_ch * 2)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # Stage 3
        self.stage3 = nn.Sequential(
            MMAC(base_ch * 2, base_ch * 4),
            MMAC(base_ch * 4, base_ch * 4),
            MMAC(base_ch * 4, base_ch * 4)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # Stage 4
        self.stage4 = nn.Sequential(
            MMAC(base_ch * 4, base_ch * 8),
            MMAC(base_ch * 8, base_ch * 8),
            MMAC(base_ch * 8, base_ch * 8),
            MMAC(base_ch * 8, base_ch * 8),
            MMAC(base_ch * 8, base_ch * 8),
            MMAC(base_ch * 8, base_ch * 8)
        )
        self.pool4 = nn.MaxPool2d(2)
        
        # Stage 5 (bottleneck)
        self.stage5 = nn.Sequential(
            MMAC(base_ch * 8, base_ch * 16),
            MMAC(base_ch * 16, base_ch * 16),
            MMAC(base_ch * 16, base_ch * 16)
        )

        
        # CPCF Modules with proper channel handling
        # Define the actual channel dimensions for each stage
        f1_ch = base_ch      # Stage 1 output channels
        f2_ch = base_ch * 2  # Stage 2 output channels  
        f3_ch = base_ch * 4  # Stage 3 output channels
        f4_ch = base_ch * 8  # Stage 4 output channels
        
        self.cpcf1 = CPCFModule(f1_ch, stage=1, f1_ch=f1_ch, f2_ch=f2_ch, f3_ch=f3_ch, f4_ch=f4_ch)
        self.cpcf2 = CPCFModule(f2_ch, stage=2, f1_ch=f1_ch, f2_ch=f2_ch, f3_ch=f3_ch, f4_ch=f4_ch)
        self.cpcf3 = CPCFModule(f3_ch, stage=3, f1_ch=f1_ch, f2_ch=f2_ch, f3_ch=f3_ch, f4_ch=f4_ch)
        self.cpcf4 = CPCFModule(f4_ch, stage=4, f1_ch=f1_ch, f2_ch=f2_ch, f3_ch=f3_ch, f4_ch=f4_ch)
        

        # Decoder
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(base_ch * 16, base_ch * 8, k=3) 
        )
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(base_ch * 8, base_ch * 4, k=3) 
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(base_ch * 4, base_ch * 2, k=3)  
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(base_ch * 2, base_ch, k=3)  
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(base_ch, base_ch, k=3)
        )
        
        # Final output
        self.seg_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch, seg_classes, kernel_size=1),
        )
        self.cls_head = ClsHead(
            f3_channels=128,
            f4_channels=256,
            f5_channels=512,
            hidden_dim=128,
            n_classes=4
        )
    def forward(self, x):
        # Initial stem
        x = self.stem(x)
        
        # Encoder forward pass
        f1 = self.stage1(x)               # base_ch
        f2 = self.stage2(self.pool1(f1))  # base_ch * 2
        f3 = self.stage3(self.pool2(f2))  # base_ch * 4
        f4 = self.stage4(self.pool3(f3))  # base_ch * 8
        f5 = self.stage5(self.pool4(f4))  # base_ch * 16
        
        # CPCF fusion
        m1 = self.cpcf1(f1, f2, f3, f4)
        m2 = self.cpcf2(f1, f2, f3, f4)
        m3 = self.cpcf3(f1, f2, f3, f4)
        m4 = self.cpcf4(f1, f2, f3, f4)
        
        # Decoder with skip connections
        d4 = self.up5(f5) + m4
        d3 = self.up4(d4) + m3
        d2 = self.up3(d3) + m2
        d1 = self.up2(d2) + m1
        d0 = self.up1(d1)
        # Final output
        seg = self.seg_head(d0)
        with torch.no_grad():
            cls = self.cls_head(f3, f4, f5)
        return seg, cls