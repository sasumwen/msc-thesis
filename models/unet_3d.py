import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class DoubleConv3D(nn.Module):
    """
    (Conv3D -> GroupNorm -> ReLU) repeated twice
    Using GroupNorm is often more stable than BatchNorm with small 3D batch sizes.
    """
    def __init__(self, in_channels, out_channels, gn_groups=8):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=gn_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=gn_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """
    Downscaling with maxpool(2) -> DoubleConv3D.
    """
    def __init__(self, in_channels, out_channels, gn_groups=8):
        super(Down3D, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels, gn_groups=gn_groups)
        )

    def forward(self, x):
        return self.down(x)


class Up3D(nn.Module):
    """
    Upscaling then DoubleConv3D.
    If bilinear=True, uses interpolation + 1x1 conv; else uses transposed conv.
    """
    def __init__(self, in_channels, out_channels, bilinear=True, gn_groups=8):
        super(Up3D, self).__init__()
        self.bilinear = bilinear

        if bilinear:
            # Interpolate + 1x1 conv to halve channels
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                nn.Conv3d(in_channels, in_channels // 2, kernel_size=1)
            )
        else:
            # ConvTranspose3d
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)
        total_in = out_channels + (in_channels // 2)
        self.conv = DoubleConv3D(total_in, out_channels, gn_groups=gn_groups)

    def forward(self, x1, x2):
        # x1 is the deeper feature to upsample
        # x2 is the skip connection
        x1 = self.up(x1)

        # If necessary, pad/crop to match shapes
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)

        # Pad if x1 is smaller
        if diffZ > 0 or diffY > 0 or diffX > 0:
            x1 = F.pad(
                x1,
                [diffX // 2, diffX - diffX // 2,
                 diffY // 2, diffY - diffY // 2,
                 diffZ // 2, diffZ - diffZ // 2]
            )

        # Crop if x1 is larger
        if diffZ < 0:
            start = -diffZ // 2
            end   = x1.size(2) + diffZ // 2
            x1 = x1[:, :, start:end, :, :]
        if diffY < 0:
            start = -diffY // 2
            end   = x1.size(3) + diffY // 2
            x1 = x1[:, :, :, start:end, :]
        if diffX < 0:
            start = -diffX // 2
            end   = x1.size(4) + diffX // 2
            x1 = x1[:, :, :, :, start:end]

        # Concat skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    """ 1x1 convolution to get n_classes output channels """
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    A 3D U-Net with GroupNorm replacing BatchNorm.
    Channel progression: [64,128,256,512,1024].
    """
    def __init__(self, n_channels=1, n_classes=6, bilinear=True, gn_groups=8):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear
        self.gn_groups  = gn_groups

        self.inc   = DoubleConv3D(n_channels, 64, gn_groups)
        self.down1 = Down3D(64, 128, gn_groups)
        self.down2 = Down3D(128, 256, gn_groups)
        self.down3 = Down3D(256, 512, gn_groups)
        factor     = 2 if bilinear else 1
        self.down4 = Down3D(512, 1024 // factor, gn_groups)

        self.up1   = Up3D(1024 // factor, 512,  bilinear, gn_groups)
        self.up2   = Up3D(512, 256,  bilinear, gn_groups)
        self.up3   = Up3D(256, 128,  bilinear, gn_groups)
        self.up4   = Up3D(128, 64,   bilinear, gn_groups)
        self.outc  = OutConv3D(64, n_classes)

    def forward(self, x):
        x1 = checkpoint(self.inc,x)         # (B,64, ..)
        x2 = checkpoint(self.down1,x1)      # (B,128, ..)
        x3 = checkpoint(self.down2,x2)      # (B,256, ..)
        x4 = checkpoint(self.down3,x3)      # (B,512, ..)
        x5 = checkpoint(self.down4,x4)      # (B,1024, ..) or (B,512,..) if bilinear

        x  = checkpoint(self.up1,x5, x4)
        x  = checkpoint(self.up2,x,  x3)
        x  = checkpoint(self.up3,x,  x2)
        x  = checkpoint(self.up4,x,  x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    # Simple test
    model = UNet3D(n_channels=1, n_classes=6, bilinear=True, gn_groups=8)
    print(model)

    dummy = torch.randn(1,1,128,128,128)  # a 128^3 patch
    out   = model(dummy)
    print("Output shape:", out.shape)  # should be (1, 96, 128,128,128)
