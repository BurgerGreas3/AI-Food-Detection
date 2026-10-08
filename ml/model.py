"""Small U-Net, randomly initialised. No pretrained weights anywhere in this project."""
import numpy as np
import torch
import torch.nn as nn


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 RGB -> float 3xHxW in [-1, 1]. Used by training AND inference (must match)."""
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    return (x - 0.5) / 0.5


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """4 down / 4 up. Input H and W must be multiples of 16. base=16 -> ~1.9M params."""

    def __init__(self, n_classes: int, base: int = 16):
        super().__init__()
        c = [base * 2**i for i in range(5)]
        self.enc = nn.ModuleList([_block(3, c[0])] + [_block(c[i], c[i + 1]) for i in range(4)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(4)])
        self.dec = nn.ModuleList([_block(c[i] * 2, c[i]) for i in range(4)])
        self.head = nn.Conv2d(c[0], n_classes, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc[0](x)
        skips = [x]
        for e in self.enc[1:]:
            x = e(self.pool(x))
            skips.append(x)
        for i in reversed(range(4)):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], dim=1))
        return self.head(x)
