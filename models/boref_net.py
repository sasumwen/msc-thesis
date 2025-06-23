
"""
boref_net.py

A 3D U-Net variant (BoRefAttnNet) with boundary attention blocks
in each decoder stage. Outputs:
 - seg_logits: (B, n_classes, D,H,W)
 - boundary_logits: (B, 1, D,H,W) from the last up block

Usage:
  from boref_attn_net import BoRefAttnNet
  model = BoRefAttnNet(n_channels=1, n_classes=6, bilinear=True, gn_groups=8, use_checkpoint=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


class DoubleConv3D(nn.Module):
    """
    (Conv3D -> GroupNorm -> ReLU) x2
    """
    def __init__(self, in_ch, out_ch, gn_groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(gn_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(gn_groups, out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class Down3D(nn.Module):
    """
    Downscale with MaxPool3d(2) => DoubleConv3D
    """
    def __init__(self, in_ch, out_ch, gn_groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv3D(in_ch, out_ch, gn_groups=gn_groups)
        )
    def forward(self, x):
        return self.net(x)

class BoundaryAttentionBlock3D(nn.Module):
    """
    Learns a boundary attention map (B,1,D,H,W) for features (B,C,D,H,W),
    then gates the features by that map.
    Returns (refined_features, boundary_logits).
    """
    def __init__(self, in_channels, gn_groups=8):
        super().__init__()
        mid_ch = max(in_channels // 4, 8)
        self.conv1 = nn.Conv3d(in_channels, mid_ch, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(gn_groups, mid_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(mid_ch, 1, kernel_size=1)  # single-channel boundary logits

    def forward(self, x):
        """
        x: shape (B, in_channels, D,H,W)
        returns:
          x_refined: shape (B, in_channels, D,H,W)
          boundary_logits: shape (B,1,D,H,W)
        """
        bndry = self.conv1(x)
        bndry = self.gn1(bndry)
        bndry = self.relu(bndry)
        bndry = self.conv2(bndry)   # boundary logits

        # [0,1] attention map
        attn_map = torch.sigmoid(bndry)

        # Gate the features (basic  multiplication)
        x_refined = x * attn_map

        return x_refined, bndry

class Up3DWithBAttn(nn.Module):
    """
    Upsampling + skip concat + boundary attention + DoubleConv
    Returns (refined_features, boundary_logits).
    """
    def __init__(self, in_channels, out_channels, bilinear=True, gn_groups=8):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            # TransposeConv to halve the channels
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2
            )
        else:
            # Upsample + 1x1 conv
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                nn.Conv3d(in_channels, in_channels // 2, kernel_size=1)
            )

        # Boundary attentio on d concatenated features
        self.battn = BoundaryAttentionBlock3D((in_channels // 2) + out_channels, gn_groups=gn_groups)
        # the usual double conv
        self.conv = DoubleConv3D((in_channels // 2) + out_channels, out_channels, gn_groups=gn_groups)

    def forward(self, x_deep, x_skip):
        # Upsample
        x_deep = self.up(x_deep)

        # Match shapes (pad if smaller)
        diffZ = x_skip.size(2) - x_deep.size(2)
        diffY = x_skip.size(3) - x_deep.size(3)
        diffX = x_skip.size(4) - x_deep.size(4)
        if diffZ > 0 or diffY > 0 or diffX > 0:
            x_deep = F.pad(
                x_deep,
                [diffX//2, diffX - diffX//2,
                 diffY//2, diffY - diffY//2,
                 diffZ//2, diffZ - diffZ//2]
            )
        # (If diff<0 => do cropping, omitted here for brevity.)

        # Concat skip
        x_cat = torch.cat([x_skip, x_deep], dim=1)

        # Boundary attention to refine features
        x_refined, boundary_logits = self.battn(x_cat)

        # DoubleConv
        out = self.conv(x_refined)

        return out, boundary_logits


# FULL BOUNDARY-ATTENTION U-NET

class BoRefAttnNet(nn.Module):
    """
    3D U-Net with boundary attention blocks in each decoder stage.
    Produces:
     - seg_logits: (B, n_classes, D,H,W)
     - boundary_logits: (B,1,D,H,W) from the last scale.
    """
    def __init__(self, n_channels=1, n_classes=96, bilinear=True, gn_groups=8, use_checkpoint=False):
        super().__init__()
        self.n_channels   = n_channels
        self.n_classes    = n_classes
        self.bilinear     = bilinear
        self.gn_groups    = gn_groups
        self.use_checkpoint = use_checkpoint

        # Encoder
        self.inc   = DoubleConv3D(n_channels, 32, gn_groups=gn_groups)
        self.down1 = Down3D(32, 64,  gn_groups=gn_groups)
        self.down2 = Down3D(64, 128, gn_groups=gn_groups)
        self.down3 = Down3D(128,256, gn_groups=gn_groups)
        factor     = 2 if bilinear else 1
        self.down4 = Down3D(256, 512//factor, gn_groups=gn_groups)

        # Decoder
        self.up1 = Up3DWithBAttn(512//factor, 256, bilinear=bilinear, gn_groups=gn_groups)
        self.up2 = Up3DWithBAttn(256, 128, bilinear=bilinear, gn_groups=gn_groups)
        self.up3 = Up3DWithBAttn(128, 64,  bilinear=bilinear, gn_groups=gn_groups)
        self.up4 = Up3DWithBAttn(64,  32,  bilinear=bilinear, gn_groups=gn_groups)

        # Output
        self.out_seg = nn.Conv3d(32, n_classes, kernel_size=1)

    def _cp_run(self, layer, *args):
        """
        I use this helper to run a layer with gradient checkpointing if use_checkpoint=True.
        """
        if self.use_checkpoint:
            return cp.checkpoint(layer, *args)
        else:
            return layer(*args)

    def forward(self, x):
        #  Encoder 
        x1 = self._cp_run(self.inc, x)      # out: 32
        x2 = self._cp_run(self.down1, x1)   # out: 64
        x3 = self._cp_run(self.down2, x2)   # out: 128
        x4 = self._cp_run(self.down3, x3)   # out: 256
        x5 = self._cp_run(self.down4, x4)   # out: 512 or 256 if bilinear

        # Decoder (with boundary attention at each scale)
        x, bndry1 = self._cp_run(self.up1, x5, x4)
        x, bndry2 = self._cp_run(self.up2, x,  x3)
        x, bndry3 = self._cp_run(self.up3, x,  x2)
        x, bndry4 = self._cp_run(self.up4, x,  x1)

        # final seg
        seg_logits = self.out_seg(x)        # (B, n_classes, D,H,W)
        # I  define the "final boundary map" as the last scale's boundary logits
        boundary_logits = bndry4            # (B,1,D,H,W)

        return seg_logits, boundary_logits

