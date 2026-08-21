#修改了上采样

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=6, base_ch=32):
        super(UNet2D, self).__init__()

        self.inc = DoubleConv(in_channels + 1, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch, base_ch * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch * 2, base_ch * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch * 4, base_ch * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch * 8, base_ch * 16))

        # Transposed Convs for upsampling
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_ch * 16, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_ch * 2, base_ch)

        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def add_position_channel(self, x):
        B, C, H, W = x.shape
        pos = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W)
        pos = pos.repeat(B, 1, H, 1)
        return torch.cat([x, pos], dim=1)

    def pad_to_match(self, x, ref):
        diffY = ref.size(2) - x.size(2)
        diffX = ref.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x

    def forward(self, x):
        x = self.add_position_channel(x)

        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        d4 = self.up4(e5)
        d4 = self.pad_to_match(d4, e4)
        d4 = self.conv4(torch.cat([e4, d4], dim=1))

        d3 = self.up3(d4)
        d3 = self.pad_to_match(d3, e3)
        d3 = self.conv3(torch.cat([e3, d3], dim=1))

        d2 = self.up2(d3)
        d2 = self.pad_to_match(d2, e2)
        d2 = self.conv2(torch.cat([e2, d2], dim=1))

        d1 = self.up1(d2)
        d1 = self.pad_to_match(d1, e1)
        d1 = self.conv1(torch.cat([e1, d1], dim=1))

        out = self.outc(d1)
        return out
