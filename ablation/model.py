"""用于组件消融的标准二维U-Net。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """两次卷积、批归一化和ReLU。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UNet(nn.Module):
    """支持任意奇数尺寸输入的标准U-Net。"""

    def __init__(
        self, in_channels: int = 1, num_classes: int = 6, base_channels: int = 32
    ) -> None:
        super().__init__()
        channels = base_channels
        self.input_block = DoubleConv(in_channels, channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(channels, channels * 2))
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(channels * 2, channels * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(channels * 4, channels * 8)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(channels * 8, channels * 16)
        )

        self.up4 = nn.ConvTranspose2d(channels * 16, channels * 8, 2, stride=2)
        self.conv4 = DoubleConv(channels * 16, channels * 8)
        self.up3 = nn.ConvTranspose2d(channels * 8, channels * 4, 2, stride=2)
        self.conv3 = DoubleConv(channels * 8, channels * 4)
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, 2, stride=2)
        self.conv2 = DoubleConv(channels * 4, channels * 2)
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.conv1 = DoubleConv(channels * 2, channels)
        self.output = nn.Conv2d(channels, num_classes, 1)

    @staticmethod
    def _match_size(inputs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """对称补齐上采样特征，使其尺寸与跳跃连接一致。"""

        delta_height = reference.size(2) - inputs.size(2)
        delta_width = reference.size(3) - inputs.size(3)
        return F.pad(
            inputs,
            [
                delta_width // 2,
                delta_width - delta_width // 2,
                delta_height // 2,
                delta_height - delta_height // 2,
            ],
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder1 = self.input_block(inputs)
        encoder2 = self.down1(encoder1)
        encoder3 = self.down2(encoder2)
        encoder4 = self.down3(encoder3)
        encoder5 = self.down4(encoder4)

        decoder4 = self._match_size(self.up4(encoder5), encoder4)
        decoder4 = self.conv4(torch.cat([encoder4, decoder4], dim=1))
        decoder3 = self._match_size(self.up3(decoder4), encoder3)
        decoder3 = self.conv3(torch.cat([encoder3, decoder3], dim=1))
        decoder2 = self._match_size(self.up2(decoder3), encoder2)
        decoder2 = self.conv2(torch.cat([encoder2, decoder2], dim=1))
        decoder1 = self._match_size(self.up1(decoder2), encoder1)
        decoder1 = self.conv1(torch.cat([encoder1, decoder1], dim=1))
        return self.output(decoder1)

